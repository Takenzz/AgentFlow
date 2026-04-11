#!/usr/bin/env python3
"""
Baseline evaluation script
Single-turn QA without the AgentFlow framework: question → model answer → correctness check → aggregate score.

Usage (server already running):
    python eval_baseline.py \\
        --tokenizer /data/model/qwen25_7b/ \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output baseline_results.json

Usage (auto-launch SGLang server):
    python eval_baseline.py \\
        --model /data/AgentFlow_pro-Qwen25-7B-RL/ \\
        --start-server --tp 4 \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output baseline_results.json
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx

# ── Ensure the agentflow directory is importable ───────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from core.llm_engine import SGLangEngine                     # noqa: E402
from core.rewarder import Rewarder                           # noqa: E402
from slime.rollout.rm_hub.math_dapo_utils import (          # noqa: E402
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
import slime.utils.http_utils as _http_utils                # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_baseline")


# ── HTTP client initialization ─────────────────────────────────────────────────

def _init_http_client(concurrency: int = 256) -> None:
    """Initialize the global AsyncClient for slime http_utils (the training framework does this automatically; eval must trigger it manually)."""
    if _http_utils._http_client is None:
        _http_utils._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=concurrency),
            timeout=httpx.Timeout(None),
        )


# ── SGLang server management ───────────────────────────────────────────────────

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
    def __init__(self, model_path: str, port: int, tp: int,
                 mem_fraction: float, ctx_len: int):
        self.port = port
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--tp", str(tp),
            "--mem-fraction-static", str(mem_fraction),
            "--context-length", str(ctx_len),
            "--trust-remote-code",
        ]
        logger.info("Starting SGLang server (port=%d)", port)
        self._proc = subprocess.Popen(cmd)

    def wait_ready(self, timeout: int = 300) -> bool:
        url = f"http://127.0.0.1:{self.port}/health"
        logger.info("Waiting for port=%d to be ready (up to %ds)…", self.port, timeout)
        ok = _wait_for_server(url, timeout=timeout)
        if ok:
            logger.info("port=%d is ready.", self.port)
        else:
            logger.error("port=%d timed out during startup.", self.port)
        return ok

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset(path: str, input_key: str = "prompt",
                 label_key: str = "label") -> list[dict]:
    """Load a list of {question, label} dicts from a JSONL file; supports chat messages format."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get(input_key, "")
            label = obj.get(label_key, "")
            if isinstance(question, list):
                for msg in reversed(question):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question = msg["content"]
                        break
                else:
                    question = str(question)
            samples.append({"question": str(question), "label": str(label)})
    return samples


# ── Answer extraction ──────────────────────────────────────────────────────────

def _extract_pred(text: str) -> str:
    """Extract \\boxed{...} if present; otherwise return the last line (truncated)."""
    boxed = last_boxed_only_string(text)
    if boxed:
        return normalize_final_answer(remove_boxed(boxed))
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1][:200] if lines else ""


# ── Single-sample evaluation ───────────────────────────────────────────────────

async def _eval_one(
    engine: SGLangEngine,
    rewarder: Rewarder,
    tokenizer,
    question: str,
    label: str,
    sampling_params: dict,
    system_prompt: str | None,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
) -> dict:
    async with semaphore:
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        try:
            out = await engine.generate(messages, sampling_params=sampling_params)
            response = out.response
        except Exception as exc:
            logger.warning("[%d/%d] generate error: %s", idx + 1, total, exc)
            return {
                "idx": idx, "question": question, "label": label,
                "pred": "", "response": "", "score": 0.0, "error": str(exc),
            }

        pred = _extract_pred(response)

        # Fast path: exact string match between pred and label → score 1.0
        if pred and label and pred == label:
            score = 1.0
        else:
            try:
                score = await rewarder.compute_reward(
                    question=question,
                    model_response=response,
                    groundtruth=label,
                )
            except Exception as exc:
                logger.warning("[%d/%d] rewarder error: %s", idx + 1, total, exc)
                score = 0.0

        logger.info(
            "[%d/%d] score=%.1f | pred=%.40s | label=%.40s",
            idx + 1, total, score, pred, label,
        )
        return {
            "idx":      idx,
            "question": question,
            "label":    label,
            "pred":     pred,
            "response": response,
            "score":    score,
        }


# ── Batch evaluation coroutine ─────────────────────────────────────────────────

async def run_eval(
    samples: list[dict],
    model_url: str,
    rewarder_url: str,
    tokenizer,
    sampling_params: dict,
    system_prompt: str | None,
    concurrency: int,
) -> list[dict]:
    _init_http_client(concurrency=concurrency * 4)

    engine = SGLangEngine(
        url=model_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=sampling_params.get("max_new_tokens", 32768),
        enable_thinking=False,
    )
    rewarder_engine = SGLangEngine(
        url=rewarder_url,
        tokenizer=tokenizer,
        sampling_params={},
        max_new_tokens=512,
    )
    rewarder = Rewarder(llm_engine=rewarder_engine)
    semaphore = asyncio.Semaphore(concurrency)
    total = len(samples)

    tasks = [
        _eval_one(
            engine, rewarder, tokenizer,
            s["question"], s["label"],
            sampling_params, system_prompt,
            semaphore, i, total,
        )
        for i, s in enumerate(samples)
    ]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["idx"])


# ── CLI argument parsing ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline evaluation (single-turn QA, without AgentFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / tokenizer
    g = p.add_argument_group("Model configuration")
    g.add_argument("--model",     default=None, help="HF model path (required when using --start-server)")
    g.add_argument("--tokenizer", default=None, help="HF tokenizer path (defaults to --model)")

    # Server connection
    g = p.add_argument_group("Server connection (when the server is already running)")
    g.add_argument("--model-url",    default="http://127.0.0.1:30000/generate",
                   help="SGLang URL for model inference")
    g.add_argument("--rewarder-url", default="http://127.0.0.1:30000/generate",
                   help="SGLang URL for the rewarder (defaults to sharing the same server as the model)")

    # Auto-launch server
    g = p.add_argument_group("Auto-launch SGLang server")
    g.add_argument("--start-server", action="store_true",
                   help="Auto-launch an SGLang server (requires --model)")
    g.add_argument("--port",         type=int,   default=30000)
    g.add_argument("--tp",           type=int,   default=4)
    g.add_argument("--mem-fraction", type=float, default=0.7)
    g.add_argument("--ctx-len",      type=int,   default=65536)

    # Evaluation data
    g = p.add_argument_group("Evaluation data")
    g.add_argument("--eval-data",   nargs="+", metavar="NAME_OR_PATH",
                   help="Dataset list in the format: name path [name path ...]")
    g.add_argument("--input-key",   default="prompt")
    g.add_argument("--label-key",   default="label")
    g.add_argument("--num-samples", type=int, default=None,
                   help="Maximum number of samples per dataset (for debugging)")

    # Sampling parameters
    g = p.add_argument_group("Sampling parameters")
    g.add_argument("--temperature",    type=float, default=0.0)
    g.add_argument("--top-p",          type=float, default=0.95)
    g.add_argument("--max-new-tokens", type=int,   default=4096)
    g.add_argument("--system-prompt",  type=str,   default=None,
                   help="Optional system prompt; omit to send no system message")

    # Inference control
    g = p.add_argument_group("Inference control")
    g.add_argument("--concurrency", type=int, default=32,
                   help="Maximum number of concurrent evaluation coroutines")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--output",  default="baseline_results.json")
    g.add_argument("--verbose", action="store_true")

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
            samples = samples[: args.num_samples]
        datasets[name] = samples
        logger.info("  → %d samples", len(samples))

    # ── Server ─────────────────────────────────────────────────────────────────
    server = None
    model_url    = args.model_url
    rewarder_url = args.rewarder_url

    if args.start_server:
        if not args.model:
            logger.error("--start-server requires --model to be specified.")
            sys.exit(1)
        server = SGLangServer(
            args.model, args.port, args.tp,
            args.mem_fraction, args.ctx_len,
        )
        if not server.wait_ready(timeout=300):
            logger.error("SGLang server failed to start; aborting evaluation.")
            server.stop()
            sys.exit(1)
        model_url    = f"http://127.0.0.1:{args.port}/generate"
        rewarder_url = model_url

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
                "Starting evaluation of '%s' (%d samples, concurrency=%d)…",
                dataset_name, len(samples), args.concurrency,
            )
            t0 = time.time()

            results = asyncio.run(
                run_eval(
                    samples=samples,
                    model_url=model_url,
                    rewarder_url=rewarder_url,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    system_prompt=args.system_prompt,
                    concurrency=args.concurrency,
                )
            )

            elapsed = time.time() - t0
            scores  = [r["score"] for r in results]
            accuracy = sum(scores) / len(scores) if scores else 0.0

            logger.info(
                "Dataset '%s': accuracy=%.3f (%d/%d) elapsed %.1fs",
                dataset_name, accuracy, int(sum(scores)), len(scores), elapsed,
            )

            all_results[dataset_name] = {
                "accuracy":        accuracy,
                "num_correct":     int(sum(scores)),
                "num_total":       len(scores),
                "elapsed_seconds": round(elapsed, 2),
                "details":         results,
            }

    finally:
        if server:
            server.stop()

    # ── Save results ────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("Results saved to: %s", output_path)

    # ── Print summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Baseline Evaluation Results Summary")
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
