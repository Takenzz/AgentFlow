#!/usr/bin/env python3
"""
AgentFlow 独立评测脚本。

角色（在 AgentFlow 整体流程中的位置）：
    与 `rollout.py` 对照：rollout.py 是训练闭环的一环（slime trainer 每步调用
    `generate` → 计算 reward → 触发 GRPO 更新）；本脚本则是**纯前向评测**——直接
    复用 `Solver` / `Rewarder` 那一套 pipeline，不经过 slime，不跑 GRPO 更新、
    不做参数回传，只关心在某个 checkpoint 下模型的准确率。

入口函数：
    - `main()`：CLI 入口。解析参数 → 加载 tokenizer / 数据 → （可选）拉起 SGLang
      服务 → 调用 `run_eval`。
    - `run_eval(...)`：构造 planner / executor / coder 三个固定 engine、初始化
      `Solver` + `Rewarder`，用 `asyncio.Semaphore` 控制并发，对每条样本调用
      `_eval_one`。
    - `_eval_one(...)`：单条样本的评测——`Solver.solve` 跑一条完整多轮轨迹，取
      `final_output` 抽出 pred，先走字符串精确匹配快路径，否则落到 Rewarder 做
      LLM judge 打分。

输出给下游的是什么：
    - 标准输出：按数据集打印准确率、正确数 / 总数、耗时的汇总表。
    - `--output`（默认 `eval_results.json`）：完整结果 JSON，包括每条样本的
      question / label / pred / final_output / score / 错误信息等，便于查 case。
    - `--trajectory-dir`（可选）：把 Solver 的完整轨迹 dump 出来，用于调试。

Checkpoint 加载（Megatron → HF）：
    本脚本接收的 `--model` 必须是 **HuggingFace 格式**的模型路径（SGLang 需要 HF
    格式）。训练过程中 slime 保存的是 **Megatron ckpt**，需要先用
    `convert_agentflow_to_hf.sh` 把 Megatron ckpt 转成 HF 格式，再把转换后的目录传
    给 `--model`（或 `--tokenizer`）。三路 engine（planner / executor / coder）
    会各自启动一个 SGLang 服务加载同一份 HF ckpt。

典型用法（server 已启动）：
    python eval_agentflow.py \\
        --tokenizer /data/model/qwen25_7b/ \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output eval_results.json \\
        --concurrency 16

典型用法（自动拉起 SGLang）：
    python eval_agentflow.py \\
        --model /data/AgentFlow_Qwen25-1.5B-RL-HF/ \\
        --start-servers --tp 1 --use-api-for-non-planner \\
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

from core.llm_engine import SGLangEngine, APIEngine  # noqa: E402
from core.solver import Solver               # noqa: E402
from core.rewarder import Rewarder           # noqa: E402
from slime.rollout.rm_hub.math_dapo_utils import (   # noqa: E402
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
import slime.utils.http_utils as _http_utils  # noqa: E402


def _init_http_client(concurrency: int = 256) -> None:
    """初始化 slime.http_utils 的全局 AsyncClient。

    训练流程中由 `init_http_client(args)` 自动完成；本评测脚本绕过了训练框架，
    因此必须在发起任何 SGLang 请求前手动调用一次。
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
    """从 JSONL 文件加载 `{question, label}` 列表。支持 input_key 为 chat messages 格式。"""
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
    """从 response 中抽取预测答案（优先 `\\boxed{...}`，否则取最后非空行并截断到 200 字）。"""
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
    """在子进程中拉起并管理一个 SGLang HTTP 推理服务。"""

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
    planner_backend: str,
    planner_url: str,
    executor_url: str,
    coder_url: str,
    tokenizer,
    sampling_params: dict,
    concurrency: int,
    max_steps: int,
    trajectory_dir: str | None,
    planner_api_key: str | None = None,
    planner_model_name: str | None = None,
    use_api_for_non_planner: bool = False,
    executor_api_key: str | None = None,
    executor_model_name: str | None = None,
    coder_api_key: str | None = None,
    coder_model_name: str | None = None,
    rewarder_api_key: str | None = None,
    rewarder_model_name: str | None = None,
    api_timeout: float = 180.0,
    api_max_retries: int = 3,
) -> list[dict]:
    """并发地用 Solver + Rewarder 评测全部样本，返回结果列表。

    三路 engine 与 rollout.py 保持一致：
      planner_engine  -> "planner" / "default"
                         对应 rollout.py 里的 engine（sglang_router）
      executor_engine -> "executor" / "verifier" / "base_generator" / rewarder
                         对应 rollout.py 里的 generate_engine（固定 30000 端口）
      coder_engine    -> "python_coder"
                         对应 rollout.py 里的 coder_engine（固定 30001 端口）

    当 ``use_api_for_non_planner=True`` 时，除 Planner 外的 engine
    （executor / verifier / base_generator / python_coder / final_output / rewarder）
    全部切换为 OpenAI 兼容的远程 API 调用（APIEngine），Planner 保持 SGLangEngine
    以便评测被训练的模型本身。
    """
    # 确保 slime http_utils 的全局 AsyncClient 已经初始化
    # （训练链路由框架代为初始化；eval 必须手动触发）
    _init_http_client(concurrency=concurrency * 4)

    max_new_tokens = sampling_params.get("max_new_tokens", 2048)

    # Planner：主模型（被评测对象）。本地小模型走 SGLang；大模型对照可走 API。
    if planner_backend == "api":
        planner_engine = APIEngine(
            url=planner_url,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
            api_key=planner_api_key,
            model_name=planner_model_name,
            timeout=api_timeout,
            max_retries=api_max_retries,
        )
    else:
        planner_engine = SGLangEngine(
            url=planner_url,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
        )

    if use_api_for_non_planner:
        # Executor / base_generator / verifier / final_output 全部走 API
        executor_engine = APIEngine(
            url=executor_url,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            api_key=executor_api_key,
            model_name=executor_model_name,
            timeout=api_timeout,
            max_retries=api_max_retries,
        )
        # Coder / python_coder 走 API（可选用独立 model / base_url）
        coder_engine = APIEngine(
            url=coder_url,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            api_key=coder_api_key,
            model_name=coder_model_name,
            timeout=api_timeout,
            max_retries=api_max_retries,
        )
        # Rewarder：LLM-as-judge，使用单独的 API 配置（默认复用 executor 的）
        rewarder_engine = APIEngine(
            url=executor_url,
            tokenizer=tokenizer,
            sampling_params={},
            max_new_tokens=2048,
            api_key=rewarder_api_key or executor_api_key,
            model_name=rewarder_model_name or executor_model_name,
            timeout=api_timeout,
            max_retries=api_max_retries,
        )
    else:
        # Executor / base_generator：通用生成引擎
        executor_engine = SGLangEngine(
            url=executor_url,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
        )
        # Coder / python_coder：代码生成引擎
        coder_engine = SGLangEngine(
            url=coder_url,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
        )
        # Rewarder 与 reward_func 保持一致：走 generate_engine（即 executor_url）
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
        "final_output":   executor_engine,  # 与训练保持一致：最终答案始终由 base 模型生成
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
    return sorted(results, key=lambda r: r["idx"])  # 保留原始顺序


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

    # Planner connection (local SGLang or OpenAI-compatible API)
    # Corresponds to the three engines in rollout.py:
    #   planner_url   -> engine          (sglang_router / main model)
    #   executor_url  -> generate_engine (executor / verifier / base_generator / final_output)
    #   coder_url     -> coder_engine    (python_coder)
    srv_grp = p.add_argument_group("Planner connection")
    srv_grp.add_argument("--planner-backend", choices=["sglang", "api"], default="sglang",
                         help="Run Planner from local SGLang or OpenAI-compatible API")
    srv_grp.add_argument("--planner-url",  default="http://127.0.0.1:30000/generate",
                         help="SGLang generate URL for Planner, or API base_url when --planner-backend=api")
    srv_grp.add_argument("--planner-api-base", default=None,
                         help="OpenAI-compatible base_url for Planner API; overrides --planner-url in API mode")
    srv_grp.add_argument("--planner-api-key", default=None,
                         help="API key for Planner API (env OPENAI_API_KEY fallback if not provided)")
    srv_grp.add_argument("--planner-model", default=None,
                         help="Model name for Planner API, required when --planner-backend=api")

    # Server connection for legacy all-local mode.
    srv_grp = p.add_argument_group("Local support-role server connection")
    srv_grp.add_argument("--executor-url", default="http://127.0.0.1:30001/generate",
                         help="SGLang generate URL for the Executor / base_generator engine")
    srv_grp.add_argument("--coder-url",    default="http://127.0.0.1:30002/generate",
                         help="SGLang generate URL for the Verifier / python_coder engine")

    # Auto-launch servers
    auto_grp = p.add_argument_group("Auto-launch SGLang servers")
    auto_grp.add_argument("--start-servers", action="store_true",
                          help="Auto-launch the local Planner SGLang server (requires --model). "
                               "Support-role servers are only launched when --use-api-for-non-planner is off.")
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

    # OpenAI-compatible API for non-planner engines
    api_grp = p.add_argument_group("API for non-planner engines")
    api_grp.add_argument("--use-api-for-non-planner", action="store_true",
                         help="Route executor / verifier / base_generator / "
                              "python_coder / final_output / rewarder to an "
                              "OpenAI-compatible API instead of SGLang.")
    api_grp.add_argument("--executor-api-base", default=None,
                         help="OpenAI-compatible base_url for executor/verifier/"
                              "base_generator/final_output (e.g. https://api.openai.com/v1)")
    api_grp.add_argument("--executor-api-key",  default=None,
                         help="API key for executor API (env OPENAI_API_KEY "
                              "fallback if not provided)")
    api_grp.add_argument("--executor-model",    default=None,
                         help="Model name served by the executor API, e.g. gpt-4o-mini")
    api_grp.add_argument("--coder-api-base",    default=None,
                         help="OpenAI-compatible base_url for python_coder "
                              "(defaults to --executor-api-base)")
    api_grp.add_argument("--coder-api-key",     default=None,
                         help="API key for coder API (defaults to --executor-api-key)")
    api_grp.add_argument("--coder-model",       default=None,
                         help="Model name for coder API, e.g. gpt-4o-mini / deepseek-coder")
    api_grp.add_argument("--rewarder-api-base", default=None,
                         help="OpenAI-compatible base_url for reward judge "
                              "(defaults to --executor-api-base)")
    api_grp.add_argument("--rewarder-api-key",  default=None,
                         help="API key for rewarder API (defaults to --executor-api-key)")
    api_grp.add_argument("--rewarder-model",    default=None,
                         help="Model name for reward judge (defaults to --executor-model)")
    api_grp.add_argument("--api-timeout", type=float, default=180.0,
                         help="Timeout in seconds for OpenAI-compatible API calls")
    api_grp.add_argument("--api-max-retries", type=int, default=3,
                         help="Maximum retry count for OpenAI-compatible API calls")

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
    #
    # 当 --use-api-for-non-planner 时，executor / coder 使用远程 API，
    # 此处只需要启动 planner 一个 SGLang 服务即可。
    servers: list[SGLangServer] = []
    planner_url  = args.planner_api_base or args.planner_url
    executor_url = args.executor_url
    coder_url    = args.coder_url

    if args.planner_backend == "api":
        if not (args.planner_api_base or args.planner_url):
            logger.error("--planner-backend=api requires --planner-api-base (or --planner-url as API base_url).")
            sys.exit(1)
        if planner_url.endswith("/generate"):
            logger.error("--planner-backend=api expects an OpenAI-compatible base_url, not an SGLang /generate URL.")
            sys.exit(1)
        if not args.planner_model:
            logger.error("--planner-backend=api requires --planner-model.")
            sys.exit(1)
        if not args.planner_api_key:
            args.planner_api_key = os.environ.get("OPENAI_API_KEY")
        logger.info("Planner will use OpenAI-compatible API: %s (model=%s)", planner_url, args.planner_model)

    # 非 planner 走 API 时，把 executor_url / coder_url 覆盖为 API base_url
    if args.use_api_for_non_planner:
        if not args.executor_api_base:
            logger.error("--use-api-for-non-planner requires --executor-api-base.")
            sys.exit(1)
        if not args.executor_model:
            logger.error("--use-api-for-non-planner requires --executor-model.")
            sys.exit(1)
        executor_url = args.executor_api_base
        coder_url    = args.coder_api_base or args.executor_api_base
        logger.info("Non-planner engines will use OpenAI-compatible API:")
        logger.info("  executor/verifier/base_gen/final_output -> %s (model=%s)",
                    executor_url, args.executor_model)
        logger.info("  python_coder                            -> %s (model=%s)",
                    coder_url, args.coder_model or args.executor_model)

    if args.start_servers:
        if args.planner_backend == "api":
            logger.info("--start-servers ignored for Planner API mode.")
        else:
            if not args.model:
                logger.error("--start-servers requires --model to be specified.")
                sys.exit(1)

            planner_srv = SGLangServer(
                args.model, args.planner_port, args.tp,
                args.mem_fraction, args.ctx_len,
            )
            servers.append(planner_srv)

            # 仅在不使用 API 时才启动 executor / coder 的 SGLang 服务
            if not args.use_api_for_non_planner:
                executor_srv = SGLangServer(
                    args.model, args.executor_port, args.tp,
                    args.mem_fraction, args.ctx_len,
                )
                coder_srv = SGLangServer(
                    args.model, args.coder_port, args.tp,
                    args.mem_fraction, args.ctx_len,
                )
                servers.extend([executor_srv, coder_srv])

            planner_url  = f"http://127.0.0.1:{args.planner_port}/generate"
            if not args.use_api_for_non_planner:
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
                    planner_backend=args.planner_backend,
                    planner_url=planner_url,
                    executor_url=executor_url,
                    coder_url=coder_url,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    concurrency=args.concurrency,
                    max_steps=args.max_steps,
                    trajectory_dir=args.trajectory_dir,
                    planner_api_key=args.planner_api_key,
                    planner_model_name=args.planner_model,
                    use_api_for_non_planner=args.use_api_for_non_planner,
                    executor_api_key=args.executor_api_key,
                    executor_model_name=args.executor_model,
                    coder_api_key=args.coder_api_key or args.executor_api_key,
                    coder_model_name=args.coder_model or args.executor_model,
                    rewarder_api_key=args.rewarder_api_key or args.executor_api_key,
                    rewarder_model_name=args.rewarder_model or args.executor_model,
                    api_timeout=args.api_timeout,
                    api_max_retries=args.api_max_retries,
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
