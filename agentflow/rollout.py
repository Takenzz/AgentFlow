import json
import os
import re
import traceback
import asyncio
from pathlib import Path
from typing import Any

import aiohttp

from slime.rollout.sglang_rollout import GenerateState
from slime.rollout.rm_hub.math_dapo_utils import last_boxed_only_string, remove_boxed, normalize_final_answer
from slime.utils.types import Sample
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.processing_utils import encode_image_for_rollout_engine
from core.llm_engine import APIEngine, SGLangEngine
from core.solver import Solver

TOOLS_DIR = Path(__file__).parent / "tools"
_SAVE_TRAJECTORY = os.environ.get("SAVE_TRAJECTORY", "0").lower() in ("1", "true", "yes")
TRAJECTORY_DIR = Path(__file__).parent / "trajectories" if _SAVE_TRAJECTORY else None
DEFAULT_API_BASE = os.environ.get("AGENTFLOW_API_BASE", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("AGENTFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
DEFAULT_API_MODEL = os.environ.get("AGENTFLOW_API_MODEL", "gpt-4o-mini")
API_TIMEOUT = float(os.environ.get("AGENTFLOW_API_TIMEOUT", "180"))
API_MAX_RETRIES = int(os.environ.get("AGENTFLOW_API_MAX_RETRIES", "3"))


def _parse_bool_env(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    value = value.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean env value: {value!r}")

# ── Eval score tracking ────────────────────────────────────────────────────────
EVAL_SCORES_FILE = Path(__file__).parent / "eval_scores.json"
_eval_initialized = False


def _ensure_eval_file() -> None:
    global _eval_initialized
    if not _eval_initialized:
        if EVAL_SCORES_FILE.exists():
            EVAL_SCORES_FILE.unlink()
        EVAL_SCORES_FILE.write_text("[]")
        _eval_initialized = True


def _extract_original_question(prompt) -> str:
    """Try to recover the original question text from the prompt."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if msg.get("role") == "user":
                return msg["content"]
        return str(prompt)
    return str(prompt)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _api_engine(
    *,
    tokenizer,
    sampling_params: dict[str, Any],
    max_new_tokens: int,
    role: str,
    default_base: str | None = None,
    default_key: str | None = None,
    default_model: str | None = None,
) -> APIEngine:
    prefix = f"AGENTFLOW_{role.upper()}"
    base_url = _env(f"{prefix}_API_BASE", default_base or DEFAULT_API_BASE)
    api_key = _env(f"{prefix}_API_KEY", default_key or DEFAULT_API_KEY)
    model = _env(f"{prefix}_MODEL", default_model or DEFAULT_API_MODEL)
    enable_thinking = _parse_bool_env(
        _env(f"{prefix}_ENABLE_THINKING", _env("AGENTFLOW_API_ENABLE_THINKING", "false"))
    )
    thinking_budget_raw = _env(f"{prefix}_THINKING_BUDGET", _env("AGENTFLOW_API_THINKING_BUDGET"))
    thinking_budget = int(thinking_budget_raw) if thinking_budget_raw not in (None, "") else None
    if not base_url:
        raise ValueError(f"{prefix}_API_BASE or AGENTFLOW_API_BASE must be set.")
    if not model:
        raise ValueError(f"{prefix}_MODEL or AGENTFLOW_API_MODEL must be set.")

    return APIEngine(
        url=base_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
        api_key=api_key,
        model_name=model,
        timeout=API_TIMEOUT,
        max_retries=API_MAX_RETRIES,
        api_enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )


def eval_log(rollout_id, args, data, extra_metrics) -> bool:
    """Custom eval log function — saves per-step eval details to eval_scores.json.
    Returns False so the framework still runs its own default logging.
    """
    _ensure_eval_file()

    step = compute_rollout_step(args, rollout_id)

    all_entries = []
    for dataset_name, dataset_data in data.items():
        rewards = dataset_data.get("rewards", [])
        samples = dataset_data.get("samples", [])

        details = []
        for i, sample in enumerate(samples):
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            question = metadata.get("original_question", "")
            if not question:
                question = _extract_original_question(sample.prompt)

            label = str(sample.label) if sample.label is not None else ""

            # Prefer final_output for pred display
            final_out = metadata.get("final_output", "") or sample.response or ""
            boxed = last_boxed_only_string(final_out)
            pred = normalize_final_answer(remove_boxed(boxed)) if boxed else _extract_final_answer(final_out)

            score = rewards[i] if i < len(rewards) else None

            details.append({
                "question": question,
                "pred": pred,
                "label": label,
                "score": score,
                "final_output": final_out,
            })

        mean_score = sum(rewards) / len(rewards) if rewards else 0.0
        all_entries.append({
            "dataset": dataset_name,
            "step": step,
            "rollout_id": rollout_id,
            "mean_score": mean_score,
            "num_samples": len(samples),
            "details": details,
        })

    try:
        existing = json.loads(EVAL_SCORES_FILE.read_text())
    except Exception:
        existing = []
    existing.extend(all_entries)
    EVAL_SCORES_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    return False


# ── Generate ──────────────────────────────────────────────────────────────────


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    """
    Async AgentFlow generation function compliant with the Slime framework.
    Pipeline: original question -> Planner.plan() -> sample
    """
    assert not getattr(args, "partial_rollout", False), "Partial rollout is not supported for AgentFlow generation at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    engine = SGLangEngine(url=url, tokenizer=state.tokenizer, sampling_params=sampling_params, max_new_tokens=4096, enable_thinking=False)

    question = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[-1]["content"]

    # Save original question in metadata so reward_func can use it
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["original_question"] = question

    # During RL only planner turns contribute to loss. Executor, verifier, and
    # tools use API services; final_output is deterministic extraction.
    generate_engine = _api_engine(
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=4096,
        role="executor",
    )
    coder_engine = _api_engine(
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=4096,
        role="coder",
        default_base=generate_engine.url,
        default_key=generate_engine.api_key,
        default_model=generate_engine.model_name,
    )
    try:
        engine_map = {
            "default":       engine,
            "planner":       engine,
            "executor":      generate_engine,
            "verifier":      generate_engine,
            "base_generator": generate_engine,
            "python_coder":  coder_engine,
        }
        solver = Solver(engine_map=engine_map, tools_dir=str(TOOLS_DIR), trajectory_dir=str(TRAJECTORY_DIR) if TRAJECTORY_DIR else None)
        label = str(sample.label) if sample.label is not None else None
        out = await solver.solve(question, label=label)
        if out is None:
            sample.status = Sample.Status.ABORTED
            sample.rollout_log_probs = []
            return sample

        sample.prompt          = out.prompt_text
        sample.response        = out.response
        sample.tokens          = out.prompt_token_ids + out.token_ids
        sample.response_length = len(out.token_ids)
        sample.loss_mask       = out.loss_mask if out.loss_mask is not None else [1] * len(out.token_ids)
        sample.rollout_log_probs = out.log_probs
        sample.status = Sample.Status.TRUNCATED if out.finish_reason == "length" else Sample.Status.COMPLETED
        sample.metadata["final_output"] = out.final_output or ""

        if out.turns:
            sample.train_metadata = {"turns": out.turns}

    except Exception:
        traceback.print_exc()
        sample.response = ""
        sample.rollout_log_probs = []
        sample.status = Sample.Status.FAILED

    return sample

# ── Reward function ───────────────────────────────────────────────────────────

def _extract_final_answer(response: str) -> str:
    """Extract the final answer from solver's multi-turn response.

    Tries in order:
    1. \\boxed{...} from the response
    2. Last numeric value from the final output section
    3. Last line of the response
    """
    boxed = last_boxed_only_string(response)
    if boxed:
        return normalize_final_answer(remove_boxed(boxed))

    # Try to find the "final output" section and extract answer from there
    # The solver appends a final_output at the end
    lines = response.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        match = re.search(r'(?:answer|result|is|=)\s*[:\s]*\$?\\?boxed\{([^}]+)\}', line, re.IGNORECASE)
        if match:
            return normalize_final_answer(match.group(1))
        match_num = re.search(r'(?:answer|result|is|=)\s*[:\s]*\$?\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)', line, re.IGNORECASE)
        if match_num:
            return match_num.group(1).strip()

    # Fallback: return last non-empty line (truncated)
    for line in reversed(lines):
        line = line.strip()
        if line:
            return line[:200]
    return ""


def _teacher_payload(input_ids: list[int]) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    return payload


def _trim_teacher_log_probs(teacher_out: dict[str, Any], response_length: int) -> list[float]:
    """Extract teacher token logprobs aligned to the generated response."""
    meta = teacher_out.get("meta_info", {})
    input_logprobs = meta.get("input_token_logprobs")
    if not input_logprobs:
        raise ValueError(f"Teacher response missing input_token_logprobs: {teacher_out}")

    # SGLang leaves the first input token without a next-token logprob.
    token_logprobs = [item[0] for item in input_logprobs[1:]]
    if response_length <= 0:
        return []
    if len(token_logprobs) < response_length:
        raise ValueError(
            f"Teacher returned {len(token_logprobs)} token logprobs, "
            f"but response_length={response_length}."
        )
    return [float(x) for x in token_logprobs[-response_length:]]


async def _request_teacher_log_probs(
    session: aiohttp.ClientSession,
    args: Any,
    *,
    input_ids: list[int],
    response_length: int,
    multimodal_inputs: dict[str, Any] | None = None,
) -> list[float]:
    payload = _teacher_payload(input_ids)

    if multimodal_inputs and multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image)
            for image in multimodal_inputs["images"]
        ]

    async with session.post(args.rm_url, json=payload) as resp:
        resp.raise_for_status()
        teacher_out = await resp.json()
    return _trim_teacher_log_probs(teacher_out, response_length)


async def reward_func(args: Any, sample: Sample, **kwargs) -> dict[str, Any]:
    """Compute AgentFlow reward.

    If --rm-url is set, fetch teacher logprobs for OPD. AgentFlow later expands
    one trajectory into independent Planner turns in custom_convert.py, so the
    teacher query is also done per turn.

    If --rm-url is not set, fall back to a lightweight answer-match reward.
    """
    if not getattr(args, "rm_url", None):
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        final_output = metadata.get("final_output", "") or sample.response or ""
        boxed = last_boxed_only_string(final_output)
        pred = normalize_final_answer(remove_boxed(boxed)) if boxed else _extract_final_answer(final_output)
        label = normalize_final_answer(str(sample.label)) if sample.label is not None else ""
        score = 1.0 if pred and label and pred == label else 0.0
        return {
            "score": score,
            "acc": score == 1.0,
            "pred": pred,
            "gt": label,
        }

    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("AGENTFLOW_OPD_TEACHER_TIMEOUT", "180")))
    connector = aiohttp.TCPConnector(limit=int(os.environ.get("AGENTFLOW_OPD_TEACHER_CONN_LIMIT", "64")))

    turns = []
    if isinstance(sample.train_metadata, dict):
        turns = list(sample.train_metadata.get("turns") or [])

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if turns:
            teacher_log_probs = await asyncio.gather(
                *[
                    _request_teacher_log_probs(
                        session,
                        args,
                        input_ids=list(turn["tokens"]),
                        response_length=int(turn["response_length"]),
                    )
                    for turn in turns
                ]
            )
            return {
                "score": 0.0,
                "acc": False,
                "opd": {
                    "mode": "turns",
                    "teacher_log_probs": teacher_log_probs,
                },
            }

        teacher_log_probs = await _request_teacher_log_probs(
            session,
            args,
            input_ids=list(sample.tokens),
            response_length=int(sample.response_length),
            multimodal_inputs=sample.multimodal_inputs,
        )
        return {
            "score": 0.0,
            "acc": False,
            "opd": {
                "mode": "sample",
                "teacher_log_probs": teacher_log_probs,
            },
        }
