#!/usr/bin/env python3
"""
AgentFlow standalone evaluation script
Forward inference only; no training logic. Directly reuses the Solver / Rewarder pipeline.

Typical usage (server already running):
    python eval_agentflow.py \\
        --tokenizer /data/model/qwen25_7b/ \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output eval_results.json \\
        --concurrency 16

Typical usage (auto-launch SGLang server):
    python eval_agentflow.py \\
        --model /data/AgentFlow_pro-Qwen25-7B-RL/ \\
        --start-servers --tp 4 \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output eval_results.json
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import httpx

# ── Ensure agentflow modules are importable regardless of working directory ────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from core.llm_engine import SGLangEngine     # noqa: E402
from core.solver import Solver               # noqa: E402
from core.rewarder import Rewarder           # noqa: E402
from slime.rollout.rm_hub.math_dapo_utils import (   # noqa: E402
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
import slime.utils.http_utils as _http_utils  # noqa: E402


def _init_http_client(concurrency: int = 256) -> None:
    """Initialize the global AsyncClient for slime http_utils.
    The training pipeline handles this via init_http_client(args); the eval script
    bypasses the training framework and must call this once before any SGLang requests.
    """
    if _http_utils._http_client is None:
        _http_utils._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=concurrency),
            timeout=httpx.Timeout(None),
        )

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_agentflow")

TOOLS_DIR = _SCRIPT_DIR / "tools"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(path: str, input_key: str = "prompt", label_key: str = "label") -> list[dict]:
    """Load a {question, label} list from a JSONL file. Supports chat messages format for input_key."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get(input_key, "")
            label = obj.get(label_key, "")
            # chat messages format -> take the last user message
            if isinstance(question, list):
                for msg in reversed(question):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question = msg["content"]
                        break
                else:
                    question = str(question)
            samples.append({"question": str(question), "label": str(label)})
    return samples


# ── Answer extraction utilities ───────────────────────────────────────────────

def _extract_pred(text: str) -> str:
    """Extract the predicted answer from the response (prefer \\boxed{...}; otherwise return last line truncated)."""
    boxed = last_boxed_only_string(text)
    if boxed:
        return normalize_final_answer(remove_boxed(boxed))
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1][:200] if lines else ""


# ── SGLang server management ──────────────────────────────────────────────────

def _wait_for_server(url: str, timeout: int = 300, interval: int = 5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(interval)
    return False


class SGLangServer:
    """Launch and manage a SGLang HTTP service in a subprocess."""

    def __init__(
        self,
        model_path: str,
        port: int,
        tp: int,
        mem_fraction: float,
        ctx_len: int,
        extra_args: list[str] | None = None,
    ):
        self.port = port
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--tp", str(tp),
            "--mem-fraction-static", str(mem_fraction),
            "--context-length", str(ctx_len),
            "--trust-remote-code",
        ] + (extra_args or [])
        logger.info("Starting SGLang server (port=%d): %s", port, " ".join(cmd))
        self._proc = subprocess.Popen(cmd)

    def wait_ready(self, timeout: int = 300) -> bool:
        url = f"http://127.0.0.1:{self.port}/health"
        logger.info("Waiting for port=%d to be ready (up to %ds)…", self.port, timeout)
        ok = _wait_for_server(url, timeout=timeout)
        if ok:
            logger.info("port=%d is ready.", self.port)
        else:
            logger.error("port=%d did not become ready within %ds.", self.port, timeout)
        return ok

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ── Single-sample evaluation ───────────────────────────────────────────────────

async def _eval_one(
    solver: Solver,
    rewarder: Rewarder,
    question: str,
    label: str,
    sampling_params: dict,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
) -> dict:
    async with semaphore:
        try:
            out = await solver.solve(question, label=label)
        except Exception as exc:
            logger.warning("[%d/%d] Solver exception: %s", idx + 1, total, exc)
            return {
                "idx": idx,
                "question": question,
                "label": label,
                "pred": "",
                "final_output": "",
                "score": 0.0,
                "error": str(exc),
            }

        if out is None:
            return {
                "idx": idx,
                "question": question,
                "label": label,
                "pred": "",
                "final_output": "",
                "score": 0.0,
                "error": "solver returned None",
            }

        final_output = out.final_output or ""
        pred = _extract_pred(final_output)

        if pred and label and pred == label:
            score = 1.0
        else:
            try:
                score = await rewarder.compute_reward(
                    question=question,
                    model_response=final_output,
                    groundtruth=label,
                )
            except Exception as exc:
                logger.warning("[%d/%d] Rewarder exception: %s", idx + 1, total, exc)
                score = 0.0

        logger.info(
            "[%d/%d] score=%.1f | pred=%.40s | label=%.40s",
            idx + 1, total, score, pred, label,
        )
        return {
            "idx": idx,
            "question": question,
            "label": label,
            "pred": pred,
            "final_output": final_output,
            "score": score,
        }


# ── Batch evaluation main coroutine ────────────────────────────────────────────

async def run_eval(
    samples: list[dict],
    planner_url: str,
    executor_url: str,
    coder_url: str,
    tokenizer,
    sampling_params: dict,
    concurrency: int,
    max_steps: int,
    trajectory_dir: str | None,
) -> list[dict]:
    """Run Solver + Rewarder concurrently over all samples and return a list of results.

    The three engines are consistent with rollout.py:
      planner_engine  -> "planner" / "default"
                         corresponds to rollout.py's engine (sglang_router)
      executor_engine -> "executor" / "verifier" / "base_generator" / rewarder
                         corresponds to rollout.py's generate_engine (hardcoded port 30000)
      coder_engine    -> "python_coder"
                         corresponds to rollout.py's coder_engine (hardcoded port 30001)
    """
    # Ensure the global HTTP client for slime http_utils is initialized
    # (the training path does this via the framework; eval must trigger it manually)
    _init_http_client(concurrency=concurrency * 4)

    max_new_tokens = sampling_params.get("max_new_tokens", 2048)

    # Planner: main model, responsible for plan / next_step / final_output; thinking disabled
    planner_engine = SGLangEngine(
        url=planner_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )
    # Executor / base_generator: general-purpose generation engine
    executor_engine = SGLangEngine(
        url=executor_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
    )
    # Coder / python_coder: code generation engine
    coder_engine = SGLangEngine(
        url=coder_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
    )
    # Rewarder consistent with reward_func: uses generate_engine (corresponding to executor_url)
    rewarder_engine = SGLangEngine(
        url=executor_url,
        tokenizer=tokenizer,
        sampling_params={},
        max_new_tokens=2048,
    )

    engine_map = {
        "default":        planner_engine,
        "planner":        planner_engine,
        "executor":       executor_engine,
        "verifier":       executor_engine,
        "base_generator": executor_engine,
        "python_coder":   coder_engine,
        "final_output":   executor_engine,  # Consistent with training: always use base model to generate the answer
    }

    solver = Solver(
        engine_map=engine_map,
        tools_dir=str(TOOLS_DIR),
        trajectory_dir=trajectory_dir,
        max_steps=max_steps,
    )
    rewarder = Rewarder(llm_engine=rewarder_engine)
    semaphore = asyncio.Semaphore(concurrency)
    total = len(samples)

    tasks = [
        _eval_one(
            solver, rewarder,
            s["question"], s["label"],
            sampling_params, semaphore,
            i, total,
        )
        for i, s in enumerate(samples)
    ]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["idx"])  # Preserve original order


# ── CLI argument parsing ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AgentFlow standalone evaluation (forward inference only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / Tokenizer
    model_grp = p.add_argument_group("Model configuration")
    model_grp.add_argument("--model", default=None,
                           help="HF model path (required when --start-servers is used)")
    model_grp.add_argument("--tokenizer", default=None,
                           help="HF tokenizer path (defaults to --model)")

    # Server connection (used when the service is already running)
    # Corresponds to the three engines in rollout.py:
    #   planner_url   -> engine          (sglang_router / main model)
    #   executor_url  -> generate_engine (port 30000, executor / verifier / base_generator / final_output)
    #   coder_url     -> coder_engine    (port 30001, python_coder)
    srv_grp = p.add_argument_group("Server connection")
    srv_grp.add_argument("--planner-url",  default="http://127.0.0.1:30000/generate",
                         help="SGLang generate URL for the Planner / default engine")
    srv_grp.add_argument("--executor-url", default="http://127.0.0.1:30001/generate",
                         help="SGLang generate URL for the Executor / base_generator engine")
    srv_grp.add_argument("--coder-url",    default="http://127.0.0.1:30002/generate",
                         help="SGLang generate URL for the Verifier / python_coder engine")

    # Auto-launch servers
    auto_grp = p.add_argument_group("Auto-launch SGLang servers")
    auto_grp.add_argument("--start-servers", action="store_true",
                          help="Auto-launch three SGLang servers (requires --model)")
    auto_grp.add_argument("--planner-port",  type=int, default=30000,
                          help="Planner server port")
    auto_grp.add_argument("--executor-port", type=int, default=30001,
                          help="Executor server port")
    auto_grp.add_argument("--coder-port",    type=int, default=30002,
                          help="Coder server port")
    auto_grp.add_argument("--tp",         type=int, default=4,
                          help="Tensor Parallel size per server")
    auto_grp.add_argument("--mem-fraction", type=float, default=0.7,
                          help="SGLang mem-fraction-static")
    auto_grp.add_argument("--ctx-len",    type=int, default=65536,
                          help="SGLang context length")

    # Evaluation data: --eval-data NAME PATH [NAME2 PATH2 ...]
    data_grp = p.add_argument_group("Evaluation data")
    data_grp.add_argument("--eval-data", nargs="+", metavar="NAME_OR_PATH",
                          help="Dataset list, format: name path [name path ...]")
    data_grp.add_argument("--input-key", default="prompt", help="Question field name")
    data_grp.add_argument("--label-key", default="label",  help="Answer field name")
    data_grp.add_argument("--num-samples", type=int, default=None,
                          help="Max samples to take per dataset (for debugging)")

    # Sampling parameters
    samp_grp = p.add_argument_group("Sampling parameters")
    samp_grp.add_argument("--temperature",    type=float, default=0.7)
    samp_grp.add_argument("--top-p",          type=float, default=0.95)
    samp_grp.add_argument("--max-new-tokens", type=int,   default=4096)

    # Inference control
    infer_grp = p.add_argument_group("Inference control")
    infer_grp.add_argument("--concurrency", type=int, default=16,
                           help="Maximum number of concurrent evaluation coroutines")
    infer_grp.add_argument("--max-steps",   type=int, default=5,
                           help="Maximum number of tool-call steps for the Solver")

    # Output
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument("--output",         default="eval_results.json",
                         help="Path to the JSON file for saving full evaluation results")
    out_grp.add_argument("--trajectory-dir", default=None,
                         help="If set, save Solver trajectories to this directory")
    out_grp.add_argument("--verbose", action="store_true",
                         help="Enable DEBUG logging")

    return p.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_path = args.tokenizer or args.model
    if not tokenizer_path:
        logger.error("Please provide --tokenizer or --model.")
        sys.exit(1)
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # ── Dataset ─────────────────────────────────────────────────────────────────
    if not args.eval_data or len(args.eval_data) % 2 != 0:
        logger.error("--eval-data requires an even number of arguments as name-path pairs, e.g.:\n"
                     "  --eval-data aime /data/aime-2024/aime-2024.jsonl")
        sys.exit(1)

    datasets: dict[str, list[dict]] = {}
    it = iter(args.eval_data)
    for name, path in zip(it, it):
        logger.info("Loading dataset '%s': %s", name, path)
        samples = load_dataset(path, args.input_key, args.label_key)
        if args.num_samples:
            samples = samples[:args.num_samples]
        datasets[name] = samples
        logger.info("  → %d samples", len(samples))

    # ── Servers ─────────────────────────────────────────────────────────────────
    # Three engine URL mappings (consistent with rollout.py):
    #   planner_url  → sglang_router / primary model (planner / default)
    #   executor_url → generate_engine               (executor / base_generator)
    #   coder_url    → coder_engine                  (verifier / python_coder)
    servers: list[SGLangServer] = []
    planner_url  = args.planner_url
    executor_url = args.executor_url
    coder_url    = args.coder_url

    if args.start_servers:
        if not args.model:
            logger.error("--start-servers requires --model to be specified.")
            sys.exit(1)

        planner_srv = SGLangServer(
            args.model, args.planner_port, args.tp,
            args.mem_fraction, args.ctx_len,
        )
        executor_srv = SGLangServer(
            args.model, args.executor_port, args.tp,
            args.mem_fraction, args.ctx_len,
        )
        coder_srv = SGLangServer(
            args.model, args.coder_port, args.tp,
            args.mem_fraction, args.ctx_len,
        )
        servers.extend([planner_srv, executor_srv, coder_srv])

        planner_url  = f"http://127.0.0.1:{args.planner_port}/generate"
        executor_url = f"http://127.0.0.1:{args.executor_port}/generate"
        coder_url    = f"http://127.0.0.1:{args.coder_port}/generate"

        for srv in servers:
            if not srv.wait_ready(timeout=300):
                logger.error("SGLang server failed to start; aborting evaluation.")
                for s in servers:
                    s.stop()
                sys.exit(1)

    sampling_params = {
        "temperature":    args.temperature,
        "top_p":          args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    # ── Evaluation loop ─────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    try:
        for dataset_name, samples in datasets.items():
            logger.info(
                "Starting evaluation of '%s' (%d samples, concurrency=%d, max_steps=%d)…",
                dataset_name, len(samples), args.concurrency, args.max_steps,
            )
            t0 = time.time()

            results = asyncio.run(
                run_eval(
                    samples=samples,
                    planner_url=planner_url,
                    executor_url=executor_url,
                    coder_url=coder_url,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    concurrency=args.concurrency,
                    max_steps=args.max_steps,
                    trajectory_dir=args.trajectory_dir,
                )
            )

            elapsed = time.time() - t0
            scores = [r["score"] for r in results]
            accuracy = sum(scores) / len(scores) if scores else 0.0

            logger.info(
                "Dataset '%s': accuracy=%.3f (%d/%d) elapsed %.1fs",
                dataset_name, accuracy, int(sum(scores)), len(scores), elapsed,
            )

            all_results[dataset_name] = {
                "accuracy":       accuracy,
                "num_correct":    int(sum(scores)),
                "num_total":      len(scores),
                "elapsed_seconds": round(elapsed, 2),
                "details":        results,
            }

    finally:
        for srv in servers:
            srv.stop()

    # ── Save results ────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("Results saved to: %s", output_path)

    # ── Print summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    for name, res in all_results.items():
        print(
            f"  {name:20s}  {res['accuracy']:.1%}"
            f"  ({res['num_correct']}/{res['num_total']})"
            f"  {res['elapsed_seconds']:.1f}s"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
