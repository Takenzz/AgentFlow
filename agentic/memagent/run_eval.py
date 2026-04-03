"""
run_eval.py — MemAgent 评测总入口（slime-agentic 版）
====================================================
对标 /data/MemAgent/taskutils/memory_eval/run.py，
管理模型配置、启动评测任务。

不负责启动 SGLang 服务——请在调用本脚本前手动启动，
或使用 MemAgent 的 serve/llm070.py：
    python /data/MemAgent/serve/llm070.py --model <ckpt> --tp <n>

用法（评测单个模型）：
    python run_eval.py \\
        --ckpt /data/MemAgent_Qwen25-7B-RL/global_step_500 \\
        --name my_7b \\
        --tasks hqa \\
        --data-root /data/hotpotqa

用法（评测所有预设模型）：
    python run_eval.py --run-all

环境变量：
    SERVE_PORT / SERVE_HOST / DATAROOT / RULER_DATAROOT
    MEM_CHUNK_TOKENS / MEM_MAX_MEMORY / MEM_MAX_FINAL / MEM_MAX_CHUNKS
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── 常量（与 ruler_hqa.py / ruler_general.py 对齐）──────────────────────────
RULER_HQA_LENGTHS      = [50, 100, 200, 400, 800, 1600, 3200, 6400]
RULER_HQA_LENGTHS_OVER1M = [12800, 25600]

RULER_SPLITS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "fwe", "qa_1",
]
RULER_LENGTHS = [8192, 16384, 32768, 65536, 131072, 262144, 524288]

SCRIPT_DIR = Path(__file__).parent


@dataclass
class ModelConfig:
    """单个模型的评测配置，对应 MemAgent run.py 的 Config 类。"""
    name: str
    ckpt: str                        # HuggingFace 路径或本地目录
    api: str  = "recurrent"          # recurrent | openai
    n_proc: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    max_input_len: int = 120000      # openai 模式专用
    max_output_len: int = 10000      # openai 模式专用
    # 是否额外跑 >1M 的超长测试（仅对 MemoryAgent 模型有意义）
    include_over1m: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        """服务器中注册的模型名称（SGLang 用目录名作为 model id）。"""
        p = Path(self.ckpt)
        return p.name if p.is_dir() else self.ckpt

    def run_hqa(
        self,
        data_root: str,
        save_dir: str,
        lengths: list[int] | None = None,
        force: bool = False,
        n_proc: int | None = None,
    ) -> None:
        if lengths is None:
            lengths = RULER_HQA_LENGTHS
            if self.include_over1m:
                lengths = lengths + RULER_HQA_LENGTHS_OVER1M

        script = str(SCRIPT_DIR / "eval_ruler_hqa.py")
        for length in lengths:
            cmd = [
                sys.executable, script,
                "--model",     self.model_name,
                "--tokenizer", self.ckpt,
                "--length",    str(length),
                "--data-root", data_root,
                "--save-dir",  str(Path(save_dir) / f"ruler_hqa_{length}"),
                "--save-file", self.name,
                "--api",       self.api,
                "--n-proc",    str(n_proc or self.n_proc),
                "--temperature", str(self.temperature),
                "--top-p",     str(self.top_p),
            ]
            if self.api == "openai":
                cmd += [
                    "--max-input-len",  str(self.max_input_len),
                    "--max-output-len", str(self.max_output_len),
                ]
            if force:
                cmd.append("--force")
            _run(cmd, self.extra_env)

    def run_general(
        self,
        data_root: str,
        save_dir: str,
        splits: list[str] | None = None,
        lengths: list[int] | None = None,
        force: bool = False,
        n_proc: int | None = None,
    ) -> None:
        splits  = splits  or RULER_SPLITS
        lengths = lengths or RULER_LENGTHS

        script = str(SCRIPT_DIR / "eval_ruler_general.py")
        for split in splits:
            for length in lengths:
                # ruler_general.py 原版也跳过了 qa_1 在 262144 以上的长度
                if split == "qa_1" and length > 262144:
                    continue
                cmd = [
                    sys.executable, script,
                    "--split",     split,
                    "--length",    str(length),
                    "--model",     self.model_name,
                    "--tokenizer", self.ckpt,
                    "--data-root", data_root,
                    "--save-dir",  str(Path(save_dir) / f"ruler_{split}_{length}"),
                    "--save-file", self.name,
                    "--api",       self.api,
                    "--n-proc",    str(n_proc or self.n_proc),
                    "--temperature", str(self.temperature),
                    "--top-p",     str(self.top_p),
                ]
                if force:
                    cmd.append("--force")
                _run(cmd, self.extra_env)


def _run(cmd: list[str], extra_env: dict[str, str]) -> None:
    env = {**os.environ, **extra_env}
    print(f"\n[run_eval] {' '.join(cmd)}")
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        print(f"[warn] command exited with code {p.returncode}")


# ── 预设模型配置（对标 MemAgent run.py 的 CONFIGS）────────────────────────────
# 在这里添加你训练完的 checkpoint 或对照基线。
# extra_env 中可以覆盖 MEM_CHUNK_TOKENS 等超参。

RECURRENT_ENV = {
    "MEM_CHUNK_TOKENS": "5000",
    "MEM_MAX_MEMORY":   "1024",
    "MEM_MAX_FINAL":    "256",
    "MEM_MAX_CHUNKS":   "512",
}

PRESET_CONFIGS: list[ModelConfig] = [
    # ── 训练结果（替换 ckpt 路径） ──────────────────────────────────────────
    ModelConfig(
        name="MemAgent-7B-slime",
        ckpt="/data/MemAgent_Qwen25-7B-RL/global_step_latest",
        api="recurrent",
        n_proc=256,
        include_over1m=True,
        extra_env=RECURRENT_ENV,
    ),
    # ── 对照基线：官方 MemoryAgent ──────────────────────────────────────────
    ModelConfig(
        name="MemoryAgent-7B-official",
        ckpt="BytedTsinghua-SIA/RL-MemoryAgent-7B",
        api="recurrent",
        n_proc=256,
        include_over1m=True,
        extra_env=RECURRENT_ENV,
    ),
    # ── 对照基线：直接长上下文生成 ──────────────────────────────────────────
    ModelConfig(
        name="Qwen25-7B-1M-openai",
        ckpt="Qwen/Qwen2.5-7B-Instruct-1M",
        api="openai",
        n_proc=256,
        max_input_len=990000,
        max_output_len=10000,
    ),
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MemAgent evaluation runner — slime-agentic version"
    )
    # 单模型快速评测
    parser.add_argument("--ckpt",      default=None, help="Checkpoint path or HF model id")
    parser.add_argument("--name",      default=None, help="Run name for output files")
    parser.add_argument("--api",       default="recurrent", choices=["recurrent", "openai"])
    parser.add_argument("--n-proc",    type=int, default=256)
    parser.add_argument("--over1m",    action="store_true",
                        help="Also run 12800/25600-doc tests")

    # 任务选择
    parser.add_argument(
        "--tasks", nargs="+", default=["hqa"],
        choices=["hqa", "general", "all"],
        help="Which eval suites to run",
    )
    parser.add_argument(
        "--hqa-lengths", nargs="+", type=int, default=None,
        help="Override HQA lengths, e.g. --hqa-lengths 50 200 800",
    )
    parser.add_argument(
        "--general-splits", nargs="+", default=None,
        choices=RULER_SPLITS,
    )
    parser.add_argument(
        "--general-lengths", nargs="+", type=int, default=None,
    )

    # 路径
    parser.add_argument("--data-root",         default=os.getenv("DATAROOT", "/data/hotpotqa"),
                        help="Directory with eval_{length}.json (HQA)")
    parser.add_argument("--ruler-data-root",   default=os.getenv("RULER_DATAROOT", "/data/ruler"),
                        help="Directory with eval_{split}_{length}.json (RULER general)")
    parser.add_argument("--save-dir",          default="results",
                        help="Root directory for all output files")

    parser.add_argument("--force",      action="store_true")
    parser.add_argument("--run-all",    action="store_true",
                        help="Run all PRESET_CONFIGS (ignores --ckpt)")
    args = parser.parse_args()

    run_tasks = set()
    for t in args.tasks:
        if t == "all":
            run_tasks |= {"hqa", "general"}
        else:
            run_tasks.add(t)

    if args.run_all:
        configs = PRESET_CONFIGS
    else:
        if not args.ckpt:
            parser.error("Either --ckpt or --run-all is required.")
        configs = [ModelConfig(
            name=args.name or Path(args.ckpt).name,
            ckpt=args.ckpt,
            api=args.api,
            n_proc=args.n_proc,
            include_over1m=args.over1m,
            extra_env=RECURRENT_ENV if args.api == "recurrent" else {},
        )]

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {cfg.name}  ({cfg.ckpt})")
        print(f"{'='*60}")

        if "hqa" in run_tasks:
            lengths = args.hqa_lengths
            if lengths is None and cfg.include_over1m:
                lengths = RULER_HQA_LENGTHS + RULER_HQA_LENGTHS_OVER1M
            cfg.run_hqa(
                data_root=args.data_root,
                save_dir=args.save_dir,
                lengths=lengths,
                force=args.force,
                n_proc=args.n_proc,
            )

        if "general" in run_tasks:
            cfg.run_general(
                data_root=args.ruler_data_root,
                save_dir=args.save_dir,
                splits=args.general_splits,
                lengths=args.general_lengths,
                force=args.force,
                n_proc=args.n_proc,
            )

    print("\n[run_eval] All done.")


if __name__ == "__main__":
    main()
