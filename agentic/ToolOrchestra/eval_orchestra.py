#!/usr/bin/env python3
"""
ToolOrchestra Standalone Eval
=============================
Standalone evaluation script supporting three benchmarks:

  1. tau2   — τ2-Bench agent evaluation (func_call tasks)
  2. hle    — HLE (Humanity's Last Exam) QA evaluation
  3. frames — FRAMES QA evaluation

Usage:

    # τ2-Bench
    python eval_orchestra.py --benchmark tau2 \
        --model-url http://127.0.0.1:30000/v1 \
        --model-name Qwen/Qwen3-8B

    # HLE (JSONL or Parquet / HF shard directory via --eval-data)
    python eval_orchestra.py --benchmark hle \
        --model-url http://127.0.0.1:30000/v1 \
        --model-name Qwen/Qwen3-8B \
        --max-turns 15 \
        --eval-data /path/to/hle.parquet

    # FRAMES
    python eval_orchestra.py --benchmark frames \
        --model-url http://127.0.0.1:30000/v1 \
        --model-name Qwen/Qwen3-8B \
        --max-turns 15

Prerequisites:
    1. Orchestrator SGLang service is running
    2. Expert SGLang services are running (30001, 30002, 30003)
    3. Retrieval service is running (port 8000, required by the search tool for HLE/FRAMES)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from qa_eval_load import load_eval_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_orchestra")

for _lib in ("httpx", "httpcore", "openai", "urllib3", "asyncio", "filelock"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
#  Progress helpers
# --------------------------------------------------------------------------- #

def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"

EXPERT_ENGINE_MAP: dict[str, dict] = {
    # port 30001: Qwen3-32B-FP8 — search-1, reasoner-1, answer-1, expert-1
    "Qwen/Qwen3-32B": {"url": "http://127.0.0.1:30001/v1", "context_length": 163840},
    # port 30002: Qwen2.5-Coder-32B — reasoner-2
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"url": "http://127.0.0.1:30002/v1", "context_length": 163840},
    # port 30003: Qwen2.5-Math-7B — answer-math-2
    "Qwen/Qwen2.5-Math-7B-Instruct": {"url": "http://127.0.0.1:30003/v1", "context_length": 163840},
    # port 30004: Qwen3-14B — expert-3, reasoner-3
    "Qwen/Qwen3-14B": {"url": "http://127.0.0.1:30004/v1", "context_length": 163840},
    # port 30005: DeepSeek-R1-Distill-Qwen-32B — answer-2
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"url": "http://127.0.0.1:30005/v1", "context_length": 163840},
    # port 30006: Qwen3-30B-A3B — expert-2, answer-3
    "Qwen/Qwen3-30B-A3B": {"url": "http://127.0.0.1:30006/v1", "context_length": 163840},
    # port 30007: Qwen2.5-Math-72B — answer-math-1
    "Qwen/Qwen2.5-Math-72B-Instruct": {"url": "http://127.0.0.1:30007/v1", "context_length": 163840},
}

RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"


def _derive_generate_url(model_url: str) -> str:
    """Derive SGLang /generate URL from OpenAI-compatible endpoint URL.

    Example: http://127.0.0.1:30000/v1 -> http://127.0.0.1:30000/generate
    """
    base = model_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/") + "/generate"


def _load_tokenizer(model_name: str):
    """Load tokenizer for chat template rendering (same as training)."""
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer from %s ...", model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Tokenizer loaded: vocab_size=%d", tok.vocab_size)
    return tok


def _truncate_middle_turns(tokenizer, messages: list[dict], max_tokens: int,
                           tools: list[dict] | None = None) -> list[dict]:
    """Truncate middle turns of a conversation to fit within max_tokens.

    Keeps the first message (system) and last message (latest user turn)
    intact, removes middle turns from oldest to newest.
    Port of orchestra_solver.OrchestraSolver._truncate_middle_turns.
    """
    if len(messages) <= 2:
        return messages

    tools_token_len = (
        len(tokenizer(str(tools), add_special_tokens=False)["input_ids"])
        if tools else 0
    )
    max_message_tokens = max(1024, max_tokens - tools_token_len)

    def _msg_tokens(msg):
        content = msg.get("content", "") or ""
        return len(tokenizer(content, add_special_tokens=False)["input_ids"])

    total = sum(_msg_tokens(m) for m in messages)
    if total <= max_message_tokens:
        return messages

    msgs = list(messages)
    head = msgs[:1]
    tail = msgs[-1:]
    middle = msgs[1:-1]

    head_tokens = _msg_tokens(head[0])
    tail_tokens = _msg_tokens(tail[0])
    remaining = max_message_tokens - head_tokens - tail_tokens

    kept_middle: list[dict] = []
    for m in reversed(middle):
        m_tok = _msg_tokens(m)
        if remaining >= m_tok:
            kept_middle.insert(0, m)
            remaining -= m_tok
        else:
            break

    return head + kept_middle + tail


def _truncate_expert_messages(messages: list[dict], max_tokens: int,
                              context_length: int) -> list[dict]:
    """Truncate messages for expert models.

    Port of ExpertCallerTool._truncate_messages — uses char-based estimation
    (expert models are called via OpenAI API so we don't have their tokenizer).
    """
    budget = context_length - max_tokens - 256
    if budget <= 0:
        return messages[-1:]

    def _est_tokens(msg):
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return max(1, len(str(content)) // 3)

    total = sum(_est_tokens(m) for m in messages)
    if total <= budget:
        return messages

    if len(messages) <= 2:
        return messages

    head = messages[:1]
    tail = messages[-1:]
    middle = list(messages[1:-1])

    head_tokens = sum(_est_tokens(m) for m in head)
    tail_tokens = sum(_est_tokens(m) for m in tail)
    remaining = budget - head_tokens - tail_tokens

    kept: list[dict] = []
    for m in reversed(middle):
        t = _est_tokens(m)
        if remaining >= t:
            kept.insert(0, m)
            remaining -= t
        else:
            break

    return head + kept + tail

QA_TOOLS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation", "tools.json")
if not os.path.isfile(QA_TOOLS_JSON):
    QA_TOOLS_JSON = "/data/slime-agentic/agentic/ToolOrchestra/evaluation/tools.json"

QA_ALL_TOOLS = {
    "enhance_reasoning": {"model": ["reasoner-1", "reasoner-2", "reasoner-3"]},
    "answer":            {"model": ["answer-1", "answer-2", "answer-3", "answer-math-1", "answer-math-2"]},
    "search":            {"model": ["search-1"]},
}

QA_MODEL_MAPPING = {
    "search-1":      "Qwen/Qwen3-32B",
    "reasoner-1":    "Qwen/Qwen3-32B",
    "reasoner-2":    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "reasoner-3":    "Qwen/Qwen3-14B",
    "answer-1":      "Qwen/Qwen3-32B",
    "answer-2":      "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "answer-3":      "Qwen/Qwen3-30B-A3B",
    "answer-math-1": "Qwen/Qwen2.5-Math-72B-Instruct",
    "answer-math-2": "Qwen/Qwen2.5-Math-7B-Instruct",
}

# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #

@dataclass
class EvalResult:
    eid: str
    domain: str
    category: str
    correctness: float = 0.0
    total_cost: float = 0.0
    total_latency: float = 0.0
    tool_counts: dict[str, int] = field(default_factory=dict)
    num_turns: int = 0
    reward_info: dict = field(default_factory=dict)
    response_text: str = ""
    error: str = ""
    trajectory: list[dict] = field(default_factory=list)

# --------------------------------------------------------------------------- #
#  Progress tracker
# --------------------------------------------------------------------------- #

class _ProgressTracker:
    """Live progress display for evaluation runs."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.score_sum = 0.0
        self._start = time.monotonic()

    def update(self, result: "EvalResult | None" = None, skipped: bool = False):
        self.completed += 1
        if skipped:
            self.skipped += 1
        elif result is not None:
            self.score_sum += result.correctness
            if result.correctness > 0:
                self.passed += 1
            else:
                self.failed += 1
        self._display(result, skipped)

    def _display(self, result: "EvalResult | None" = None, skipped: bool = False):
        elapsed = time.monotonic() - self._start
        evaluated = self.passed + self.failed
        avg = self.score_sum / evaluated if evaluated > 0 else 0.0

        if 0 < self.completed < self.total:
            eta = (self.total - self.completed) * elapsed / self.completed
            eta_s = _fmt_time(eta)
        else:
            eta_s = "-"

        pct = self.completed / max(self.total, 1)
        w = 25
        filled = int(w * pct)
        bar = "█" * filled + "░" * (w - filled)

        last = ""
        if skipped:
            last = " | (skipped)"
        elif result is not None:
            mark = "✓" if result.correctness > 0 else "✗"
            last = f" | {result.eid} {mark}"

        print(
            f"[Eval] [{bar}] {self.completed}/{self.total} "
            f"| avg_score={avg:.3f} pass={self.passed} fail={self.failed} "
            f"| {_fmt_time(elapsed)} elapsed, eta {eta_s}"
            f"{last}",
            flush=True,
        )

    def finish(self):
        elapsed = time.monotonic() - self._start
        evaluated = self.passed + self.failed
        avg = self.score_sum / evaluated if evaluated > 0 else 0.0
        print(
            f"\n[Eval] Done! {evaluated} evaluated in {_fmt_time(elapsed)} "
            f"| avg_score={avg:.3f} pass={self.passed} fail={self.failed} "
            f"skipped={self.skipped}",
            flush=True,
        )


# --------------------------------------------------------------------------- #
#  Tau2Adapter (lite copy — avoids import cycles)
# --------------------------------------------------------------------------- #

_TAU2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tau2")
_TAU2_CLI = os.path.join(_TAU2_DIR, "cli.py")

_TAU2_PYTHON = "/usr/bin/python3"


class _Tau2Lite:
    """Minimal tau2 adapter for eval (mirrors tau2_adapter.py without slime deps)."""

    POLL_S = 0.5
    MAX_WAIT_S = 600

    def __init__(self, domain: str, task_path: str, output_file: str,
                 transfer_dir: str, user_llm: str, max_steps: int):
        self.domain = domain
        self.task_path = task_path
        self.output_file = output_file
        self.transfer_dir = transfer_dir
        self.user_llm = user_llm
        self.max_steps = max_steps
        self._step = 0
        self._proc = None

    async def start(self):
        os.makedirs(self.transfer_dir, exist_ok=True)
        env = os.environ.copy()
        parent = os.path.dirname(_TAU2_DIR)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{parent}:{existing}" if existing else parent

        cmd = [
            _TAU2_PYTHON, _TAU2_CLI,
            "--domain", self.domain,
            "--task_path", self.task_path,
            "--output_file", self.output_file,
            "--cur_transfer_dir", self.transfer_dir,
            "--agent-llm", "train",
            "--user-llm", self.user_llm,
            "--max-steps", str(self.max_steps),
            "--max-concurrency", "1",
            "--num-tasks", "1",
            "--use_model_tool",
        ]
        logger.debug("[Tau2Lite] CMD: %s", " ".join(cmd))
        logger.debug("[Tau2Lite] PYTHONPATH: %s", env["PYTHONPATH"][:200])
        import subprocess
        self._proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _read_stderr(self) -> str:
        if self._proc and self._proc.stderr:
            try:
                data = self._proc.stderr.read()
                return data.decode("utf-8", errors="replace")[:3000] if data else ""
            except Exception:
                return ""
        return ""

    async def wait_for_input(self) -> Optional[dict]:
        path = os.path.join(self.transfer_dir, f"input_{self._step}.json")
        waited = 0.0
        while not os.path.isfile(path):
            if self.is_done():
                return None
            if self._proc and self._proc.poll() is not None:
                stderr = self._read_stderr()
                logger.error("[Tau2Lite] subprocess exited with code %s. stderr:\n%s",
                             self._proc.returncode, stderr)
                return None
            if waited >= self.MAX_WAIT_S:
                logger.error("[Tau2Lite] timeout waiting for input_%d.json (%ds)", self._step, self.MAX_WAIT_S)
                return None
            await asyncio.sleep(self.POLL_S)
            waited += self.POLL_S

        for _ in range(5):
            try:
                with open(path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                await asyncio.sleep(0.3)
        with open(path) as f:
            return json.load(f)

    async def write_output(self, content: Optional[str], tool_calls: Optional[list]):
        path = os.path.join(self.transfer_dir, f"output_{self._step}.json")
        with open(path, "w") as f:
            json.dump({"content": content, "tool_calls": tool_calls or []}, f, ensure_ascii=False, indent=2)
        self._step += 1

    def is_done(self) -> bool:
        return os.path.isfile(os.path.join(self.transfer_dir, "done"))

    @staticmethod
    def _extract_from_sim(sim_data: dict) -> dict | None:
        """Extract reward_info from a tau2 simulation result.

        Mirrors Tau2Adapter._extract_from_sim to handle edge cases
        (e.g. tau_run_error, non-dict reward_info).
        """
        if "reward_info" in sim_data and sim_data["reward_info"]:
            ri = sim_data["reward_info"]
            if isinstance(ri, dict):
                return ri
            return {"reward": ri, "info": {}}
        if "tau_run_error" in sim_data:
            return {"reward": 0.0, "info": {"error": sim_data["tau_run_error"]}}
        return None

    def get_reward_info(self) -> dict:
        search_dirs = []
        base = self.output_file
        if base.endswith(".json"):
            search_dirs.append(base[:-5])
        search_dirs.append(os.path.dirname(base))
        if os.path.isfile(base):
            search_dirs.insert(0, base)

        for cand in search_dirs:
            if os.path.isfile(cand):
                try:
                    with open(cand) as f:
                        data = json.load(f)
                    ri = self._extract_from_sim(data)
                    if ri is not None:
                        return ri
                except Exception:
                    pass
                continue
            if not os.path.isdir(cand):
                continue
            for fname in sorted(os.listdir(cand)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(cand, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    ri = self._extract_from_sim(data)
                    if ri is not None:
                        return ri
                except Exception:
                    continue
        return {"reward": 0.0, "info": {"error": "no reward_info found"}}

    async def cleanup(self):
        if self._proc and self._proc.poll() is None:
            import signal
            try:
                self._proc.send_signal(signal.SIGTERM)
                await asyncio.sleep(2)
                if self._proc.poll() is None:
                    self._proc.kill()
            except OSError:
                pass
        if os.path.isdir(self.transfer_dir):
            try:
                shutil.rmtree(self.transfer_dir)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
#  Expert caller (simplified)
# --------------------------------------------------------------------------- #

async def call_expert(
    messages: list[dict],
    expert: str,
    model_mapping: dict,
    tools: Optional[list] = None,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    max_retries: int = 2,
) -> dict:
    """Call an expert model via OpenAI-compatible API. Returns {content, tool_calls}.

    Includes message truncation and retry logic to match training's
    ExpertCallerTool.execute_with_messages behaviour.
    """
    model_name = model_mapping.get(expert)
    if not model_name:
        return {"content": f"[Error] Unknown expert: {expert}", "tool_calls": None}

    engine = EXPERT_ENGINE_MAP.get(model_name)
    if not engine:
        return {"content": f"[Error] No engine for model: {model_name}", "tool_calls": None}

    base_url = engine["url"] if isinstance(engine, dict) else engine
    context_length = engine.get("context_length", 163840) if isinstance(engine, dict) else 163840
    _http = httpx.AsyncClient(trust_env=False)
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=180.0,
                         http_client=_http)

    messages = _truncate_expert_messages(messages, max_tokens, context_length)

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools

            completion = await client.chat.completions.create(**kwargs)
            msg = completion.choices[0].message

            tc_list = None
            if msg.tool_calls:
                tc_list = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = tc.function.arguments
                    tc_list.append({"name": tc.function.name, "arguments": args})

            raw = msg.content or ""
            return {"content": raw, "tool_calls": tc_list, "raw_output": raw}
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.error("[ExpertCaller] expert=%s model=%s failed after %d retries: %s",
                             expert, model_name, max_retries, e)
                return {"content": f"[Error] {e}", "tool_calls": None, "raw_output": f"[Error] {e}"}
            import random as _random
            backoff = min(2 ** attempt, 60) + _random.random()
            logger.warning("[ExpertCaller] expert=%s retry %d/%d in %.1fs: %s",
                           expert, attempt + 1, max_retries, backoff, e)
            await asyncio.sleep(backoff)

    return {"content": "[Error] All retries exhausted.", "tool_calls": None, "raw_output": ""}


# --------------------------------------------------------------------------- #
#  Orchestrator caller
# --------------------------------------------------------------------------- #

async def call_orchestrator(
    generate_url: str,
    tokenizer,
    messages: list[dict],
    tools: Optional[list] = None,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    context_length: int = 131072,
) -> dict:
    """Call orchestrator via SGLang /generate (matches training path exactly).

    Training uses tokenizer.apply_chat_template + SGLang /generate + XML parsing.
    This function replicates that pipeline to avoid train/eval mismatch.
    max_tokens=16384 matches --rollout-max-response-len in train_orchestra.sh.
    """
    try:
        messages = _truncate_middle_turns(
            tokenizer, messages, context_length - max_tokens - 32, tools,
        )

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        available = context_length - len(prompt_token_ids) - 32
        effective_max = min(max_tokens, max(1, available))

        payload = {
            "text": prompt_text,
            "sampling_params": {
                "max_new_tokens": effective_max,
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
            resp = await client.post(generate_url, json=payload)
            resp.raise_for_status()
            raw = resp.json()

        response_text = raw["text"]

        # Parse tool calls from raw text with XML parser (same as training's
        # tool_call_parser.parse_all_tool_calls)
        all_tcs = _parse_xml_tool_calls(response_text)

        content: str | None = response_text
        tc_list: list[dict] | None = None
        if all_tcs:
            tc_list = all_tcs
            content = None

        return {"content": content, "tool_calls": tc_list, "raw_output": response_text}
    except Exception as e:
        logger.error("[Orchestrator] %s", e)
        return {"content": f"[Error] {e}", "tool_calls": None, "raw_output": f"[Error] {e}"}


def _parse_xml_tool_calls(text: str) -> list[dict]:
    """Parse <tool_call>{...}</tool_call> blocks from model output."""
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    results = []
    for m in pattern.finditer(text):
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and "name" in data:
                results.append(data)
        except json.JSONDecodeError:
            continue
    return results


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> from text, matching original ToolOrchestra's
    ``response_content.split('</think>')[-1].strip()``."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _extract_message(text: str) -> str:
    """Extract plain message from model output (strip <think> / <message> tags)."""
    cleaned = _strip_thinking(text)
    m = re.search(r"<message>(.*?)</message>", cleaned, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return cleaned.strip() or "I will help you with that."


# --------------------------------------------------------------------------- #
#  QA eval logic: HLE / FRAMES (search → reasoning → answer loop)
# --------------------------------------------------------------------------- #

async def _retrieve(query: str, eid: str = "", topk: int = 50) -> list[str]:
    """Call FAISS retrieval service."""
    import httpx
    payload = {"queries": [query[:400]], "topk": topk, "return_scores": True, "eid": eid}
    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            resp = await client.post(RETRIEVAL_URL, json=payload)
            data = resp.json()
        docs = []
        for r in data[0]:
            content = r.get("document", {}).get("content") or r.get("document", {}).get("contents", "")
            if content:
                docs.append(content)
        return docs
    except Exception as e:
        logger.warning("[retrieve] %s", e)
        return []


async def _expert_answer(
    query: str,
    role_name: str,
    answer_gt: str,
    judge_client: Optional[AsyncOpenAI] = None,
) -> tuple[str, bool]:
    """Call an expert to produce an answer, then check correctness."""
    model_name = QA_MODEL_MAPPING.get(role_name)
    if not model_name:
        return "", False

    engine = EXPERT_ENGINE_MAP.get(model_name)
    if not engine:
        return "", False
    base_url = engine["url"] if isinstance(engine, dict) else engine
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=180.0,
                         http_client=httpx.AsyncClient(trust_env=False))

    system_msg = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]
    try:
        completion = await client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=8192, temperature=0.2,
        )
        response_str = completion.choices[0].message.content or ""
    except Exception as e:
        logger.error("[expert_answer] %s", e)
        return "", False

    pred = ""
    if "\\boxed{" in response_str:
        parts = response_str.split("\\boxed{")[-1].split("}")[:-1]
        pred = "}".join(parts).strip()
    elif "<answer>" in response_str:
        pred = response_str.split("<answer>")[-1].split("</answer>")[0].strip()

    if not pred or len(pred.split()) > 500:
        return pred, False
    if pred.strip().lower() == answer_gt.strip().lower():
        return pred, True
    if answer_gt.strip().lower() in pred.strip().lower():
        return pred, True
    return pred, False


async def _expert_search_query(
    problem: str,
    context: str,
    role_name: str,
) -> str:
    """Ask expert to write a search query."""
    model_name = QA_MODEL_MAPPING.get(role_name)
    if not model_name:
        return problem[:200]

    engine = EXPERT_ENGINE_MAP.get(model_name)
    if not engine:
        return problem[:200]
    base_url = engine["url"] if isinstance(engine, dict) else engine
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=180.0,
                         http_client=httpx.AsyncClient(trust_env=False))

    prompt = f"{context}\n\nQuestion: {problem}\n" \
             "Instead of directly answering, write a concise search query. " \
             "Wrap it within <query> and </query>."
    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.2,
        )
        text = completion.choices[0].message.content or ""
        m = re.search(r"<query>(.*?)</query>", text, re.DOTALL)
        if m and len(m.group(1).strip()) >= 5:
            return m.group(1).strip()
    except Exception:
        pass
    return problem[:200]


async def eval_qa(
    example: dict,
    generate_url: str,
    tokenizer,
    qa_tools: list[dict],
    max_turns: int = 15,
    temperature: float = 0.7,
    context_length: int = 131072,
) -> EvalResult:
    """Evaluate a QA example (HLE / FRAMES style)."""
    eid = example.get("id", "unknown")
    question = example["question"]
    answer_gt = example.get("answer", "")

    doc_list: list[str] = []
    code_list: list[dict] = []
    used_tools: list[str] = []
    tool_counts: dict[str, int] = defaultdict(int)
    trajectory: list[dict] = []
    total_latency = 0.0

    final_correct = False
    final_pred = ""

    for step in range(max_turns):
        doc_str = ""
        for i, doc in enumerate(doc_list):
            doc_str += f"Doc {i+1}: {doc[:2000]}\n\n"

        code_str = ""
        for cp in code_list:
            code_str += f"```python\n{cp['code']}\n```\n```output\n{cp['output']}\n```\n\n"

        context = ""
        if doc_str:
            context = "Documents:\n" + doc_str
        if code_str:
            context += "\nCode results:\n" + code_str

        tools_to_send = list(qa_tools)
        if len(used_tools) > 1 and used_tools[-1] == used_tools[-2]:
            tools_to_send = [t for t in tools_to_send
                             if t["function"]["name"] != used_tools[-1]]

        messages = [
            {"role": "system", "content": "You are good at using tools."},
            {"role": "user", "content": f"Problem: {question}\n\n{context}\n\nChoose an appropriate tool."},
        ]

        t0 = time.monotonic()
        orch_resp = await call_orchestrator(
            generate_url, tokenizer, messages, tools_to_send,
            temperature=temperature,
            context_length=context_length,
        )
        orch_elapsed = (time.monotonic() - t0) * 1000
        total_latency += orch_elapsed

        orch_tcs = orch_resp.get("tool_calls")
        if not orch_tcs:
            trajectory.append({
                "step": step,
                "orchestrator": {"raw_output": orch_resp.get("raw_output", ""), "content": orch_resp.get("content"), "tool_calls": None, "latency_ms": round(orch_elapsed, 1)},
                "action": "stop (no tool call)",
            })
            break

        tc = orch_tcs[0]
        tool_name = tc.get("name", "")
        tool_args = tc.get("arguments", {})

        if tool_name not in QA_ALL_TOOLS:
            trajectory.append({
                "step": step,
                "orchestrator": {"raw_output": orch_resp.get("raw_output", ""), "content": orch_resp.get("content"), "tool_calls": orch_tcs, "latency_ms": round(orch_elapsed, 1)},
                "action": f"stop (unknown tool: {tool_name})",
            })
            break

        role_name = tool_args.get("model", "")
        valid_roles = QA_ALL_TOOLS[tool_name]["model"]
        if role_name not in valid_roles:
            role_name = valid_roles[0]

        used_tools.append(tool_name)
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        step_record: dict = {
            "step": step,
            "orchestrator": {
                "raw_output": orch_resp.get("raw_output", ""),
                "content": orch_resp.get("content"),
                "tool_calls": orch_tcs,
                "latency_ms": round(orch_elapsed, 1),
            },
            "tool": tool_name,
            "role": role_name,
            "model": QA_MODEL_MAPPING.get(role_name),
        }

        if tool_name == "search":
            t0 = time.monotonic()
            query = await _expert_search_query(question, context, role_name)
            docs = await _retrieve(query, eid=eid)
            search_elapsed = (time.monotonic() - t0) * 1000
            total_latency += search_elapsed
            new_docs = []
            for d in reversed(docs):
                if d not in doc_list:
                    doc_list.append(d)
                    new_docs.append(d)
            step_record["search_query"] = query
            step_record["docs_retrieved"] = len(docs)
            step_record["docs_new"] = len(new_docs)
            step_record["latency_ms"] = round(search_elapsed, 1)

        elif tool_name == "enhance_reasoning":
            step_record["action"] = "enhance_reasoning (skipped)"

        elif tool_name == "answer":
            prompt = context.strip() + "\n\nProblem:\n" + question
            t0 = time.monotonic()
            pred, correct = await _expert_answer(prompt, role_name, answer_gt)
            answer_elapsed = (time.monotonic() - t0) * 1000
            total_latency += answer_elapsed
            final_pred = pred
            final_correct = correct
            step_record["prediction"] = pred
            step_record["correct"] = correct
            step_record["answer_gt"] = answer_gt
            step_record["latency_ms"] = round(answer_elapsed, 1)
            trajectory.append(step_record)
            break

        trajectory.append(step_record)

    return EvalResult(
        eid=eid,
        domain="qa",
        category="qa",
        correctness=1.0 if final_correct else 0.0,
        total_latency=total_latency,
        tool_counts=dict(tool_counts),
        num_turns=len(used_tools),
        response_text=final_pred,
        trajectory=trajectory,
    )


# --------------------------------------------------------------------------- #
#  Core eval logic: func_call tasks (tau2-driven)
# --------------------------------------------------------------------------- #

async def eval_func_call(
    example: dict,
    generate_url: str,
    tokenizer,
    max_turns: int = 12,
    user_llm: str = "qwen-turbo-latest",
    temperature: float = 0.7,
    context_length: int = 131072,
) -> EvalResult:
    """Evaluate a single func_call example using tau2 environment."""
    meta = example.get("metadata", {})
    eid = meta.get("eid", "unknown")
    domain = eid.split("____")[0] if "____" in eid else "unknown"
    model_mapping = meta.get("model_mapping", {})
    tool_pricing = meta.get("tool_pricing", {})

    transfer_dir = tempfile.mkdtemp(prefix="tau2_eval_")
    output_file = os.path.join(transfer_dir, "output.json")

    task_path = os.path.join(transfer_dir, "task.json")
    task_data = meta.get("example")
    if task_data:
        if not isinstance(task_data, list):
            task_data = [task_data]
        with open(task_path, "w") as f:
            json.dump(task_data, f, indent=2)

    if model_mapping:
        mm_path = os.path.join(transfer_dir, "model_mapping.json")
        with open(mm_path, "w") as f:
            json.dump({"model_mapping": model_mapping, "tool_pricing": tool_pricing}, f)

    adapter = _Tau2Lite(
        domain=domain,
        task_path=task_path,
        output_file=output_file,
        transfer_dir=transfer_dir,
        user_llm=user_llm,
        max_steps=max_turns * 4,
    )

    turns_info: list[dict] = []
    trajectory: list[dict] = []
    total_cost = 0.0
    total_latency = 0.0
    tool_counts: dict[str, int] = defaultdict(int)

    try:
        await adapter.start()

        for turn_idx in range(max_turns * 4):
            if adapter.is_done():
                break

            inp = await adapter.wait_for_input()
            if inp is None:
                break

            messages = inp.get("messages", [])
            tools = inp.get("tools", [])

            t0 = time.monotonic()
            orch_resp = await call_orchestrator(
                generate_url, tokenizer, messages, tools,
                temperature=temperature,
                context_length=context_length,
            )
            orch_elapsed = (time.monotonic() - t0) * 1000

            orch_tcs = orch_resp.get("tool_calls")
            orch_content = orch_resp.get("content")

            # --- call_expert detection: check FIRST tool call only ---
            # Training (orchestra_solver.py L274-278) does:
            #   tc = all_tcs[0] if all_tcs else None
            #   if tc and not tc.error and tc.name == "call_expert": ...
            # Must match this: only the first TC decides routing.
            expert_name = None
            if orch_tcs:
                first_tc = orch_tcs[0]
                if first_tc.get("name") == "call_expert":
                    expert_name = first_tc.get("arguments", {}).get("expert", "expert-1")

            output_content = None
            output_tool_calls = None
            expert_elapsed = 0.0
            expert_resp = None

            if expert_name:
                tool_counts[expert_name] += 1
                t0 = time.monotonic()
                expert_resp = await call_expert(
                    messages=inp.get("original_messages", messages),
                    expert=expert_name,
                    model_mapping=model_mapping,
                    tools=inp.get("original_tools", tools),
                    temperature=temperature,
                )
                expert_elapsed = (time.monotonic() - t0) * 1000
                output_content = expert_resp.get("content") or ""
                output_tool_calls = expert_resp.get("tool_calls")
            elif orch_tcs:
                # Forward ALL parsed tool calls to tau2 (matching training)
                output_tool_calls = orch_tcs
            else:
                output_content = _extract_message(
                    orch_resp.get("raw_output") or orch_content or ""
                )

            total_latency += orch_elapsed + expert_elapsed

            turn_record = {
                "turn": turn_idx,
                "expert": expert_name,
                "orch_ms": round(orch_elapsed, 1),
                "expert_ms": round(expert_elapsed, 1),
            }
            turns_info.append(turn_record)

            # Record full trajectory step
            # turn 0: save full tool definitions (contains call_expert with model_mapping desc)
            # later turns: save only names to keep file size manageable
            tools_entry = tools if turn_idx == 0 else [t["function"]["name"] for t in tools if "function" in t]
            trajectory.append({
                "turn": turn_idx,
                "messages": messages,
                "tools": tools_entry,
                "orchestrator": {
                    "raw_output": orch_resp.get("raw_output", ""),
                    "content": orch_content,
                    "tool_calls": orch_tcs,
                    "latency_ms": round(orch_elapsed, 1),
                },
                "expert_called": expert_name,
                "expert_model": model_mapping.get(expert_name) if expert_name else None,
                "expert_response": {
                    "raw_output": expert_resp.get("raw_output", "") if expert_resp else None,
                    "content": expert_resp.get("content") if expert_resp else None,
                    "tool_calls": expert_resp.get("tool_calls") if expert_resp else None,
                    "latency_ms": round(expert_elapsed, 1),
                } if expert_name else None,
                "output_to_tau2": {
                    "content": output_content,
                    "tool_calls": output_tool_calls,
                },
            })

            await adapter.write_output(output_content, output_tool_calls)

        reward_info = adapter.get_reward_info()

    except Exception as e:
        logger.error("[eval_func_call] eid=%s error: %s", eid, e)
        reward_info = {"reward": 0.0, "error": str(e)}
    finally:
        await adapter.cleanup()

    correctness = float(reward_info.get("reward", 0.0))

    return EvalResult(
        eid=eid,
        domain=domain,
        category="func_call",
        correctness=correctness,
        total_cost=total_cost,
        total_latency=total_latency,
        tool_counts=dict(tool_counts),
        num_turns=len(turns_info),
        reward_info=reward_info,
        trajectory=trajectory,
    )


# --------------------------------------------------------------------------- #
#  Batch runner
# --------------------------------------------------------------------------- #

async def eval_all(
    examples: list[dict],
    generate_url: str,
    tokenizer,
    output_dir: str,
    concurrency: int = 4,
    max_turns: int = 12,
    user_llm: str = "qwen-turbo-latest",
    temperature: float = 0.7,
    benchmark: str = "tau2",
    qa_tools: list[dict] | None = None,
    context_length: int = 131072,
) -> list[EvalResult]:
    """Run evaluation on all examples with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    results: list[Optional[EvalResult]] = [None] * len(examples)
    progress = _ProgressTracker(len(examples))

    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    async def run_one(idx: int, ex: dict):
        async with sem:
            if benchmark in ("hle", "frames"):
                eid = ex.get("id", f"unknown_{idx}")
                category = "qa"
            else:
                meta = ex.get("metadata", {})
                eid = meta.get("eid", f"unknown_{idx}")
                category = meta.get("category", "qa")

            if benchmark == "tau2" and category != "func_call":
                progress.update(skipped=True)
                return

            logger.debug("Starting eid=%s category=%s", eid, category)

            if benchmark in ("hle", "frames"):
                result = await eval_qa(
                    ex, generate_url, tokenizer,
                    qa_tools=qa_tools or [],
                    max_turns=max_turns,
                    temperature=temperature,
                    context_length=context_length,
                )
            else:
                result = await eval_func_call(
                    ex, generate_url, tokenizer,
                    max_turns=max_turns,
                    user_llm=user_llm,
                    temperature=temperature,
                    context_length=context_length,
                )

            results[idx] = result

            # ── per-example result (metrics) ──
            eid_safe = eid.replace("/", "_").replace(os.sep, "_")
            # Recreate dirs if missing (e.g. another eval with --clean-previous
            # deleted this run's output_dir while we were still in flight).
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(traj_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{eid_safe}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "eid": result.eid,
                    "domain": result.domain,
                    "category": result.category,
                    "correctness": result.correctness,
                    "total_cost": result.total_cost,
                    "total_latency_ms": result.total_latency,
                    "tool_counts": result.tool_counts,
                    "num_turns": result.num_turns,
                    "reward_info": result.reward_info,
                    "error": result.error,
                }, f, ensure_ascii=False, indent=2)

            # ── per-example trajectory (full trace for inspection) ──
            if result.trajectory:
                traj_path = os.path.join(traj_dir, f"{eid_safe}_traj.json")
                with open(traj_path, "w") as f:
                    json.dump({
                        "eid": result.eid,
                        "domain": result.domain,
                        "category": result.category,
                        "correctness": result.correctness,
                        "num_turns": result.num_turns,
                        "reward_info": result.reward_info,
                        "trajectory": result.trajectory,
                    }, f, ensure_ascii=False, indent=2)

            progress.update(result)

    tasks = [asyncio.create_task(run_one(i, ex)) for i, ex in enumerate(examples)]
    await asyncio.gather(*tasks)

    progress.finish()
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------- #
#  Metrics aggregation & reporting
# --------------------------------------------------------------------------- #

def aggregate_and_report(results: list[EvalResult], output_dir: str):
    """Compute and print per-domain + overall metrics, save summary."""
    if not results:
        logger.warning("No results to aggregate.")
        return

    by_domain: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_domain[r.domain].append(r)

    header = f"{'Domain':<16} {'Count':>6} {'Correct':>8} {'Acc%':>7} {'AvgTurns':>9} {'AvgLatMs':>10}"
    sep = "-" * len(header)

    lines = [sep, header, sep]

    domain_stats = {}
    for domain in sorted(by_domain.keys()):
        dr = by_domain[domain]
        n = len(dr)
        n_correct = sum(1 for r in dr if r.correctness > 0)
        acc = n_correct / n if n > 0 else 0.0
        avg_turns = sum(r.num_turns for r in dr) / n if n > 0 else 0.0
        avg_latency = sum(r.total_latency for r in dr) / n if n > 0 else 0.0

        lines.append(f"{domain:<16} {n:>6} {n_correct:>8} {acc * 100:>6.1f}% {avg_turns:>9.1f} {avg_latency:>10.0f}")
        domain_stats[domain] = {
            "count": n,
            "correct": n_correct,
            "accuracy": round(acc, 4),
            "avg_turns": round(avg_turns, 2),
            "avg_latency_ms": round(avg_latency, 1),
        }

    n_total = len(results)
    n_total_correct = sum(1 for r in results if r.correctness > 0)
    overall_acc = n_total_correct / n_total if n_total > 0 else 0.0
    overall_turns = sum(r.num_turns for r in results) / n_total if n_total > 0 else 0.0
    overall_lat = sum(r.total_latency for r in results) / n_total if n_total > 0 else 0.0

    lines.append(sep)
    lines.append(f"{'OVERALL':<16} {n_total:>6} {n_total_correct:>8} {overall_acc * 100:>6.1f}% {overall_turns:>9.1f} {overall_lat:>10.0f}")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report + "\n")

    tool_usage: dict[str, int] = defaultdict(int)
    for r in results:
        for k, v in r.tool_counts.items():
            tool_usage[k] += v

    if tool_usage:
        print("Expert usage distribution:")
        for k in sorted(tool_usage, key=tool_usage.get, reverse=True):
            print(f"  {k}: {tool_usage[k]}")
        print()

    error_count = sum(1 for r in results if r.error)
    if error_count:
        print(f"Errors: {error_count}/{n_total}")
        for r in results:
            if r.error:
                print(f"  {r.eid}: {r.error[:200]}")
        print()

    summary = {
        "overall": {
            "count": n_total,
            "correct": n_total_correct,
            "accuracy": round(overall_acc, 4),
            "avg_turns": round(overall_turns, 2),
            "avg_latency_ms": round(overall_lat, 1),
            "errors": error_count,
        },
        "per_domain": domain_stats,
        "tool_usage": dict(tool_usage),
    }

    summary_path = os.path.join(output_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Summary saved to %s", summary_path)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="ToolOrchestra standalone eval")

    parser.add_argument("--benchmark", type=str, default="tau2",
                        choices=["tau2", "hle", "frames"],
                        help="Benchmark to evaluate: tau2 | hle | frames")
    parser.add_argument("--model-url", type=str, default="http://127.0.0.1:30000/v1",
                        help="Orchestrator SGLang endpoint (e.g. http://host:30000/v1). "
                             "The /generate URL is derived automatically.")
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name or local path for the orchestrator "
                             "(used to load the tokenizer for chat template)")
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Path to eval data: JSONL, or for hle/frames also .parquet / "
                             "directory of *.parquet shards (auto from benchmark if omitted)")
    parser.add_argument("--output-dir", type=str, default="/data/eval_results",
                        help="Base directory for results. A timestamped sub-dir is created "
                             "automatically: <base>/<benchmark>_<timestamp>/")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent evaluations")
    parser.add_argument("--max-turns", type=int, default=12,
                        help="Max orchestrator turns per example")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for orchestrator")
    parser.add_argument("--context-length", type=int, default=131072,
                        help="Orchestrator model context length (must match SGLang server)")
    parser.add_argument("--user-llm", type=str, default="qwen-turbo-latest",
                        help="LLM for tau2 user simulator")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of examples (for quick testing)")
    parser.add_argument("--domains", type=str, nargs="*", default=None,
                        help="Filter to specific domains (e.g., weather travel)")
    parser.add_argument("--expert-urls", type=str, nargs="*", default=None,
                        help="Override expert URLs: model_name=url pairs, e.g., "
                             "'Qwen/Qwen3-32B=http://host:30001/v1'")
    parser.add_argument("--qa-tools-json", type=str, default=None,
                        help="Path to tools.json for QA benchmarks (hle/frames)")
    parser.add_argument("--clean-previous", action="store_true", default=False,
                        help="Delete previous eval result directories for this "
                             "benchmark before starting a new run")
    parser.add_argument("--tau2-python", type=str, default=None,
                        help="Python interpreter for tau2 subprocess (tau2 needs "
                             "toml/litellm etc. which may not be in sglang env). "
                             "Defaults to /usr/bin/python3.")

    args = parser.parse_args()

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _DEFAULT_EVAL_DATA = {
        "tau2": os.path.join(_SCRIPT_DIR, "data", "eval_tau2.jsonl"),
        "hle": os.path.join(_SCRIPT_DIR, "evaluation", "hle.jsonl"),
        "frames": os.path.join(_SCRIPT_DIR, "evaluation", "frames.jsonl"),
    }
    if args.eval_data is None:
        args.eval_data = _DEFAULT_EVAL_DATA[args.benchmark]

    if args.expert_urls:
        for pair in args.expert_urls:
            model, url = pair.split("=", 1)
            EXPERT_ENGINE_MAP[model] = {"url": url, "context_length": 163840}

    # ── Override tau2 Python interpreter if specified ──
    if args.tau2_python:
        global _TAU2_PYTHON
        _TAU2_PYTHON = args.tau2_python
    logger.info("tau2 subprocess Python: %s", _TAU2_PYTHON)

    # ── Load tokenizer (required for chat template, same as training) ──
    tokenizer = _load_tokenizer(args.model_name)

    # ── Derive SGLang /generate URL ──
    generate_url = _derive_generate_url(args.model_url)
    logger.info("SGLang /generate URL: %s", generate_url)

    # ── Optionally clean previous eval results for this benchmark ──
    if args.clean_previous and os.path.isdir(args.output_dir):
        import glob as _glob
        logger.warning(
            "--clean-previous: removing all %s_* under %s. Do not run two "
            "evals on the same output-dir at once, or the other run's folder "
            "may be deleted mid-flight.",
            args.benchmark,
            args.output_dir,
        )
        pattern = os.path.join(args.output_dir, f"{args.benchmark}_*")
        for old_dir in sorted(_glob.glob(pattern)):
            if os.path.isdir(old_dir):
                logger.info("Removing previous result dir: %s", old_dir)
                shutil.rmtree(old_dir, ignore_errors=True)

    # ── Build output directory with benchmark + timestamp ──
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.benchmark}_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    qa_tools = None
    if args.benchmark in ("hle", "frames"):
        tools_path = args.qa_tools_json or QA_TOOLS_JSON
        if os.path.isfile(tools_path):
            with open(tools_path) as f:
                qa_tools = json.load(f)
            logger.info("Loaded %d QA tools from %s", len(qa_tools), tools_path)
        else:
            logger.error("QA tools.json not found at %s", tools_path)
            return

    logger.info("Benchmark: %s", args.benchmark)
    logger.info("Loading eval data from %s", args.eval_data)
    examples = load_eval_examples(args.eval_data, args.benchmark)

    if args.domains:
        domain_set = set(args.domains)
        filtered = []
        for ex in examples:
            eid = ex.get("metadata", {}).get("eid", ex.get("id", ""))
            domain = eid.split("____")[0] if "____" in eid else ""
            if domain in domain_set:
                filtered.append(ex)
        logger.info("Filtered %d -> %d examples for domains %s",
                     len(examples), len(filtered), args.domains)
        examples = filtered

    if args.max_examples and len(examples) > args.max_examples:
        examples = examples[:args.max_examples]
        logger.info("Limited to %d examples", args.max_examples)

    logger.info("Total examples: %d", len(examples))
    logger.info("Orchestrator: %s @ %s (generate: %s)",
                args.model_name, args.model_url, generate_url)
    logger.info("Context length: %d", args.context_length)
    logger.info("Output dir: %s", output_dir)

    results = asyncio.run(eval_all(
        examples=examples,
        generate_url=generate_url,
        tokenizer=tokenizer,
        output_dir=output_dir,
        concurrency=args.concurrency,
        max_turns=args.max_turns,
        user_llm=args.user_llm,
        temperature=args.temperature,
        benchmark=args.benchmark,
        qa_tools=qa_tools,
        context_length=args.context_length,
    ))

    aggregate_and_report(results, output_dir)


if __name__ == "__main__":
    main()
