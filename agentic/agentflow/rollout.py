import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.rollout.rm_hub.math_dapo_utils import last_boxed_only_string, remove_boxed, normalize_final_answer
from slime.utils.types import Sample
from slime.utils.metric_utils import compute_rollout_step
from core.llm_engine import APIEngine, SGLangEngine
from core.solver import Solver
from core.rewarder import Rewarder

TOOLS_DIR = Path(__file__).parent / "tools"
_SAVE_TRAJECTORY = os.environ.get("SAVE_TRAJECTORY", "0").lower() in ("1", "true", "yes")
TRAJECTORY_DIR = Path(__file__).parent / "trajectories" if _SAVE_TRAJECTORY else None
DEFAULT_API_BASE = os.environ.get("AGENTFLOW_API_BASE", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("AGENTFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
DEFAULT_API_MODEL = os.environ.get("AGENTFLOW_API_MODEL", "gpt-4o-mini")
API_TIMEOUT = float(os.environ.get("AGENTFLOW_API_TIMEOUT", "180"))
API_MAX_RETRIES = int(os.environ.get("AGENTFLOW_API_MAX_RETRIES", "3"))

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

    # During RL only planner turns contribute to loss. All support roles use
    # API services so four local GPUs can stay focused on small-planner training.
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
            "final_output":  generate_engine,  # Always use the base model to generate the answer; excluded from training
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


async def reward_func(args: Any, sample: Sample, **kwargs) -> dict:
    """
    Uses Rewarder.compute_reward (LLM judge) to compare the model answer against the ground truth.
    Uses the original question stored in metadata rather than the prompt overwritten by the solver.
    The Rewarder uses the fixed generate_engine (port 30000) instead of the training planner engine,
    ensuring the reward signal remains stable throughout RL training.
    """
    state = GenerateState(args)
    engine = _api_engine(
        tokenizer=state.tokenizer,
        sampling_params={},
        max_new_tokens=2048,
        role="rewarder",
        default_model=_env("AGENTFLOW_EXECUTOR_MODEL", DEFAULT_API_MODEL),
        default_base=_env("AGENTFLOW_EXECUTOR_API_BASE", DEFAULT_API_BASE),
        default_key=_env("AGENTFLOW_EXECUTOR_API_KEY", DEFAULT_API_KEY),
    )
    rewarder = Rewarder(llm_engine=engine)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    question = metadata.get("original_question", "")
    if not question:
        question = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[-1]["content"]

    label = str(sample.label) if sample.label is not None else ""

    final_output = metadata.get("final_output", "") or sample.response or ""

    boxed = last_boxed_only_string(final_output)
    pred = normalize_final_answer(remove_boxed(boxed)) if boxed else _extract_final_answer(final_output)

    if pred and label and pred == label:
        score = 1.0
    else:
        score = await rewarder.compute_reward(
            question=question,
            model_response=final_output,
            groundtruth=label,
        )
    return {
        "score": score,
        "acc": score == 1.0,
        "pred": pred,
        "gt": label,
    }
