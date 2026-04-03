"""
ToolOrchestra Reward — per-sample 特征提取器
=============================================
reward_func 只负责提取每个 rollout 的原始特征：
    correctness (0/1), total_cost, total_latency, tool_counts

真正的 preference-weighted reward + GRPO 标准化在 custom_convert.py 中完成，
因为它需要看到同一道题的所有 rollout 做相对比较（min-max 归一化）。

score 字段返回 correctness（0/1），作为 custom_convert 的基础输入。

QA 二次判分（可选）：
    当 category 为 qa（非 func_call）且规则匹配判为错误时，调用百炼（DashScope）API
    让模型判断学答是否与标准答案等价；若判定正确则将 correctness 置为 1。
    环境变量（与 LLM_CALL.py 共享）：
      QA_REWARD_JUDGE_ENABLED=1        （默认开启，设为 0 关闭）
      DASHSCOPE_API_KEY                （API Key，LLM_CALL.py 中已有默认值）
      DASHSCOPE_BASE_URL               （默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
      QA_REWARD_JUDGE_MODEL=qwen-turbo-latest  （判对错用 turbo 够了，省 token）
      QA_REWARD_JUDGE_TIMEOUT=60
      QA_REWARD_JUDGE_MAX_CHARS=12000  （单段截断长度）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

ROLLOUT_LOG_DIR = os.environ.get("ROLLOUT_LOG_DIR", "/data/rollout_logs")
_rollout_counter: dict[str, int] = {"train": 0, "eval": 0}

_QA_JUDGE_SYSTEM = """You are an answer grader. Decide if the student answer is correct with respect to the ground truth.
Accept equivalent paraphrases, equivalent math expressions, and minor formatting differences.
Reject if the student answer is factually wrong, misses required parts of the ground truth, or contradicts it.
Reply with one JSON object only, no markdown or other text: {"correct": true} or {"correct": false}"""


def _normalize_text(text: str) -> str:
    """Normalize whitespace, punctuation, casing for robust comparison."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _check_correctness_qa(final_output: str, label: str) -> float:
    """
    QA correctness: normalized exact match.
    Handles numeric answers, short answers, and longer text.
    """
    if not label or not final_output:
        return 0.0

    norm_label = _normalize_text(label)
    norm_output = _normalize_text(final_output)

    if norm_label == norm_output:
        return 1.0

    if norm_label in norm_output:
        label_tokens = norm_label.split()
        output_tokens = norm_output.split()
        if len(label_tokens) <= 3 and len(output_tokens) > 10 * len(label_tokens):
            return 0.0
        return 1.0

    return 0.0


def _qa_judge_enabled() -> bool:
    return os.environ.get("QA_REWARD_JUDGE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _truncate_for_judge(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2] + "\n...[truncated]...\n" + text[-max_chars // 2 :]


def _parse_llm_judge_correctness(raw: str) -> bool | None:
    """Parse model output for a boolean correct verdict. None = unparseable."""
    text = (raw or "").strip()
    if not text:
        return None
    # Strip optional ```json fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "correct" in obj:
            return bool(obj["correct"])
    except (json.JSONDecodeError, TypeError):
        pass
    # Inline JSON substring
    m = re.search(r"\{[^{}]*\"correct\"\s*:\s*(true|false)[^{}]*\}", text, re.I | re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "correct" in obj:
                return bool(obj["correct"])
        except (json.JSONDecodeError, TypeError):
            pass
    m2 = re.search(r'"correct"\s*:\s*(true|false)', text, re.I)
    if m2:
        return m2.group(1).lower() == "true"
    return None


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 reasoning traces) before parsing."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def _async_qa_llm_judge(
    final_output: str,
    label: str,
    question: str = "",
) -> tuple[bool, str]:
    """
    Call DashScope (百炼) API to judge QA correctness.
    Returns (is_correct, raw_response_or_error).
    On failure, returns (False, error message).
    """
    # 复用 LLM_CALL.py 的百炼配置
    api_key  = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DASHSCOPE_API_KEY is not set. "
            "Please export DASHSCOPE_API_KEY=<your-api-key> before running."
        )
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model    = os.environ.get("QA_REWARD_JUDGE_MODEL", "qwen-turbo-latest")
    timeout  = float(os.environ.get("QA_REWARD_JUDGE_TIMEOUT", "60"))
    max_chars = int(os.environ.get("QA_REWARD_JUDGE_MAX_CHARS", "12000"))

    # Build user message — include question for better context-dependent judgement
    parts = []
    if question.strip():
        parts.append(f"Question:\n{_truncate_for_judge(question, max_chars)}")
    parts.append(f"Ground truth answer:\n{_truncate_for_judge(label, max_chars)}")
    parts.append(f"Student answer:\n{_truncate_for_judge(final_output, max_chars)}")
    parts.append('Is the student answer correct? Output JSON only: {"correct": true} or {"correct": false}')
    user_content = "\n\n".join(parts)

    from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _QA_JUDGE_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except (APITimeoutError, APIConnectionError, BadRequestError, OSError) as e:
        logger.warning("[reward] QA LLM judge failed: %s", e)
        return False, f"[judge_error] {e}"
    except Exception as e:
        logger.warning("[reward] QA LLM judge unexpected error: %s", e)
        return False, f"[judge_error] {e}"

    # Strip any residual thinking traces before parsing
    parsed_raw = _strip_thinking(raw)
    verdict = _parse_llm_judge_correctness(parsed_raw)
    if verdict is None:
        logger.warning("[reward] QA LLM judge unparseable response: %s", raw[:300])
        return False, raw
    return verdict, raw


def _check_correctness_func_call(
    final_output: str,
    label: str,
    turns: list[dict],
) -> float:
    """
    func_call correctness: JSON exact match on function calls (aligned with
    the original ToolOrchestra toolcall.py validator).
    Falls back to checking tau2 reward_info if available in turns.
    """
    tau2_reward = _extract_tau2_reward(turns)
    if tau2_reward is not None:
        return 1.0 if tau2_reward > 0 else 0.0

    if not label:
        return 0.0

    try:
        answer_calls = json.loads(label)
        if isinstance(answer_calls, dict):
            answer_calls = [answer_calls]
    except (json.JSONDecodeError, TypeError):
        return _check_correctness_qa(final_output, label)

    result_calls = _extract_tool_calls_from_output(final_output)
    if not result_calls and not answer_calls:
        return 1.0
    if not result_calls or not answer_calls:
        return 0.0

    try:
        counter_result = Counter(
            (item["name"], json.dumps(item.get("arguments", {}), sort_keys=True))
            for item in result_calls
        )
        counter_answer = Counter(
            (item["name"], json.dumps(item.get("arguments", {}), sort_keys=True))
            for item in answer_calls
        )
    except (TypeError, KeyError):
        return 0.0

    return 1.0 if counter_result == counter_answer else 0.0


def _extract_tool_calls_from_output(text: str) -> list[dict]:
    """Extract tool_call JSON blocks from model output."""
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    calls = []
    for m in pattern.finditer(text):
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and "name" in data:
                calls.append(data)
        except json.JSONDecodeError:
            continue
    return calls


def _extract_tau2_reward(turns: list[dict]) -> float | None:
    """If tau2 reward_info was stored in turns, extract it."""
    for t in turns:
        ri = t.get("tau2_reward_info")
        if ri is not None:
            return float(ri.get("reward", 0)) if isinstance(ri, dict) else float(ri)
    return None


def _infer_orch_pricing(tool_pricing: dict) -> dict:
    """
    When trained_model_type is not in tool_pricing, use the cheapest
    model's pricing as a proxy for the orchestrator (smaller model).
    """
    if not tool_pricing:
        return {}
    cheapest = None
    cheapest_cost = float("inf")
    for model_name, pricing in tool_pricing.items():
        cost = pricing.get("input_tokens_per_million", 0) + pricing.get("output_tokens_per_million", 0)
        if cost < cheapest_cost:
            cheapest_cost = cost
            cheapest = pricing
    return cheapest or {}


def extract_features(
    final_output: str,
    label: str,
    turns: list[dict],
    tool_pricing: dict,
    model_mapping: dict,
    category: str = "qa",
    trained_model_type: str = "",
) -> dict:
    """
    Extract per-rollout raw features for downstream reward computation.

    Returns dict with:
        correctness:  0.0 or 1.0
        total_cost:   sum of orchestrator + expert token costs across turns
        total_latency: sum of latency_ms across turns
        tool_counts:  {role_name: call_count}  (how many times each expert was called)
    """
    if category == "func_call":
        correctness = _check_correctness_func_call(final_output, label, turns)
    else:
        correctness = _check_correctness_qa(final_output, label)

    total_cost = 0.0
    total_latency = 0.0
    tool_counts: dict[str, int] = {}

    orch_pricing = tool_pricing.get(trained_model_type, {}) if trained_model_type else {}
    if not orch_pricing:
        orch_pricing = _infer_orch_pricing(tool_pricing)
    orch_price_in = orch_pricing.get("input_tokens_per_million", 0)
    orch_price_out = orch_pricing.get("output_tokens_per_million", 0)

    for t in turns:
        role_name    = t.get("role_name")
        latency      = t.get("latency_ms", 0.0)
        total_latency += latency

        if role_name:
            tool_counts[role_name] = tool_counts.get(role_name, 0) + 1

        orch_in = t.get("orch_input_tokens", 0)
        orch_out = t.get("orch_output_tokens", 0)
        total_cost += orch_in * orch_price_in + orch_out * orch_price_out

        if role_name:
            expert_in = t.get("input_tokens", 0)
            expert_out = t.get("output_tokens", 0)
            model_name = model_mapping.get(role_name, "")
            pricing = tool_pricing.get(model_name, {})
            price_in = pricing.get("input_tokens_per_million", 0)
            price_out = pricing.get("output_tokens_per_million", 0)
            total_cost += expert_in * price_in + expert_out * price_out

    return {
        "correctness":   correctness,
        "total_cost":    total_cost,
        "total_latency": total_latency,
        "tool_counts":   tool_counts,
    }


def _save_rollout(sample: Any, reward_dict: dict) -> None:
    """Save a single rollout to disk under eval/ or train/."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    is_eval = metadata.get("_is_eval", False)
    phase = "eval" if is_eval else "train"
    out_dir = os.path.join(ROLLOUT_LOG_DIR, phase)
    os.makedirs(out_dir, exist_ok=True)

    _rollout_counter[phase] += 1
    seq = _rollout_counter[phase]

    eid = metadata.get("eid", "unknown")
    category = metadata.get("category", "unknown")
    ts = time.strftime("%Y%m%d_%H%M%S")

    fname = f"{ts}_{seq:05d}_{category}_{eid}.json"

    record = {
        "index": getattr(sample, "index", None),
        "eid": eid,
        "category": category,
        "status": sample.status.value if hasattr(sample, "status") and sample.status else None,
        "prompt": sample.prompt if isinstance(sample.prompt, str) else "[chat]",
        "response": getattr(sample, "response", ""),
        "response_length": getattr(sample, "response_length", 0),
        "reward": reward_dict,
        "label": str(sample.label) if sample.label is not None else "",
        "final_output": metadata.get("final_output", ""),
    }

    try:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


async def reward_func(args: Any, sample: Any, **kwargs) -> dict:
    """
    slime framework entry point.

    Returns correctness as "score" (0/1). The full preference-weighted
    reward is computed later in custom_convert where cross-rollout
    comparison is possible.

    Also stores raw features in the result dict for custom_convert to read.
    """
    metadata     = sample.metadata if isinstance(sample.metadata, dict) else {}
    final_output = metadata.get("final_output", "") or getattr(sample, "response", "") or ""
    label        = str(sample.label) if sample.label is not None else ""
    turns        = (getattr(sample, "train_metadata", None) or {}).get("turns", [])

    # Extract original question for the LLM judge (best-effort)
    prompt = sample.prompt
    if isinstance(prompt, list):
        question = next(
            (m["content"] for m in reversed(prompt) if m.get("role") == "user"),
            "",
        )
    else:
        question = str(prompt) if prompt else ""

    category = metadata.get("category", "qa")
    features = extract_features(
        final_output       = final_output,
        label              = label,
        turns              = turns,
        tool_pricing       = metadata.get("tool_pricing", {}),
        model_mapping      = metadata.get("model_mapping", {}),
        category           = category,
        trained_model_type = metadata.get("trained_model_type", ""),
    )

    qa_judge_used = False
    qa_judge_verdict: bool | None = None
    qa_judge_raw = ""

    # QA only: if rule-based says wrong, ask LLM whether answers are equivalent
    if (
        category != "func_call"
        and features["correctness"] < 0.5
        and _qa_judge_enabled()
        and final_output.strip()
        and label.strip()
    ):
        qa_judge_used = True
        ok, qa_judge_raw = await _async_qa_llm_judge(final_output, label, question=question)
        qa_judge_verdict = ok
        if ok:
            features["correctness"] = 1.0
            features["qa_judge_override"] = True

    if hasattr(sample, "train_metadata") and isinstance(sample.train_metadata, dict):
        sample.train_metadata["reward_features"] = features
    elif hasattr(sample, "train_metadata"):
        sample.train_metadata = {"reward_features": features}

    result = {
        "score":          features["correctness"],
        "correctness":    features["correctness"],
        "total_cost":     features["total_cost"],
        "total_latency":  features["total_latency"],
        "tool_counts":    features["tool_counts"],
        "pred":           final_output,
        "gt":             label,
        "qa_judge_used":  qa_judge_used,
        "qa_judge_verdict": qa_judge_verdict,
        "qa_judge_raw":   qa_judge_raw[:2000] if qa_judge_raw else "",
    }

    _save_rollout(sample, result)

    return result
