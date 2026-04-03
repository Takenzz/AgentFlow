"""
将 MemAgent 的 hotpotqa parquet 数据转换为 slime 兼容的 JSONL 格式。

MemAgent parquet 字段：
    prompt        : [{"role": "user", "content": question}]
    context       : "Document 1:\n...\n\nDocument 2:\n..."
    reward_model  : {"style": "rule", "ground_truth": ["answer1", ...]}
    extra_info    : {"index": 0, "question": ..., "num_docs": 200}
    data_source   : "hotpotqa"
    ability       : "memory"

输出 JSONL 字段（slime 约定）：
    prompt    : question string            (--input-key prompt)
    label     : first answer string        (--label-key label)
    metadata  : {
        "context"      : 长文档文本,
        "ground_truth" : [所有可接受答案],  # reward_func 多答案匹配用
        "num_docs"     : int,
        "data_source"  : str,
    }

用法：
    python prepare_data.py \\
        --input  /path/to/hotpotqa_train.parquet \\
        --output /path/to/hotpotqa_train.jsonl

    # 也可以直接从 HuggingFace 拉取
    python prepare_data.py \\
        --hf-dataset BytedTsinghua-SIA/hotpotqa \\
        --hf-split  train \\
        --output /path/to/hotpotqa_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert_row(row: dict) -> dict | None:
    """将 MemAgent 的一行转换为 slime JSONL 格式。

    兼容两种格式：
      训练集格式（parquet）: prompt(list) / reward_model / extra_info / context
      评估集格式（eval_*.json）: input / answers / num_docs / context
    """
    context = row.get("context", "")
    if not context:
        return None

    # ── 评估集格式：input + answers ──────────────────────────────────────────
    if "input" in row:
        question = row["input"]
        answers = row.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        label = answers[0] if answers else ""
        if not question or not label:
            return None
        return {
            "prompt":   question,
            "label":    label,
            "metadata": {
                "context":      context,
                "ground_truth": answers,
                "num_docs":     row.get("num_docs", 0),
                "data_source":  "hotpotqa",
            },
        }

    # ── 训练集格式：prompt(list) + reward_model ──────────────────────────────
    prompt_field = row.get("prompt", [])
    if isinstance(prompt_field, list) and prompt_field:
        question = prompt_field[0].get("content", "") if isinstance(prompt_field[0], dict) else str(prompt_field[0])
    elif isinstance(prompt_field, str):
        question = prompt_field
    else:
        question = row.get("extra_info", {}).get("question", "")

    if not question:
        return None

    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except Exception:
            reward_model = {}
    ground_truth = reward_model.get("ground_truth", [])
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    label = ground_truth[0] if ground_truth else ""
    if not label:
        return None

    extra_info = row.get("extra_info", {}) or {}

    return {
        "prompt":   question,
        "label":    label,
        "metadata": {
            "context":      context,
            "ground_truth": ground_truth,
            "num_docs":     extra_info.get("num_docs", 0),
            "data_source":  row.get("data_source", "hotpotqa"),
        },
    }


def convert_parquet(input_path: str, output_path: str) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(input_path)
    rows = df.to_dict(orient="records")
    return _write_jsonl(rows, output_path)


def convert_hf(dataset_name: str, split: str, output_path: str) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets is required. pip install datasets", file=sys.stderr)
        sys.exit(1)

    # 优先使用环境变量 HF_ENDPOINT，其次自动尝试镜像
    import os
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 用 HfFileSystem 精确列出目标 split 的文件，避免多 split 格式不一致的推断错误
    from huggingface_hub import HfFileSystem, list_repo_files
    fs = HfFileSystem()
    exts = (".parquet", ".json", ".jsonl", ".csv")

    # list_repo_files 比 glob 更可靠，能列出所有文件
    all_files = list(list_repo_files(dataset_name, repo_type="dataset"))
    split_files = [
        f"hf://datasets/{dataset_name}/{p}"
        for p in all_files
        if split in Path(p).name and Path(p).suffix in exts
    ]
    if split_files:
        fmt = "parquet" if split_files[0].endswith(".parquet") else "json"
        ds = load_dataset(fmt, data_files={split: split_files}, split=split)
    else:
        # 最后回退：可能 split 名藏在目录里（如 data/split/xxx.parquet）
        split_files = [
            f"hf://datasets/{dataset_name}/{p}"
            for p in all_files
            if (f"/{split}/" in p or f"/{split}-" in p) and Path(p).suffix in exts
        ]
        if not split_files:
            raise FileNotFoundError(
                f"Cannot find files for split '{split}' in dataset '{dataset_name}'. "
                f"Available files: {all_files[:20]}"
            )
        fmt = "parquet" if split_files[0].endswith(".parquet") else "json"
        ds = load_dataset(fmt, data_files={split: split_files}, split=split)
    rows = [dict(r) for r in ds]
    return _write_jsonl(rows, output_path)


def _write_jsonl(rows: list[dict], output_path: str) -> int:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            out = convert_row(row)
            if out is None:
                skipped += 1
                continue
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written}  Skipped: {skipped}  → {output_path}")
    return written


def convert_hf_file(dataset_name: str, filename: str, output_path: str) -> int:
    """直接下载 HF repo 中的指定文件并转换，用于 eval_*.json 等非标准 split 文件。"""
    import os
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        repo_id=dataset_name,
        filename=filename,
        repo_type="dataset",
    )
    suffix = Path(local_path).suffix.lower()
    if suffix == ".parquet":
        return convert_parquet(local_path, output_path)
    else:
        with open(local_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            rows = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
        else:
            rows = data
        return _write_jsonl(rows, output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert MemAgent parquet to slime JSONL")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",      help="Local parquet file path")
    group.add_argument("--hf-dataset", help="HuggingFace dataset name, e.g. BytedTsinghua-SIA/hotpotqa")
    parser.add_argument("--hf-split",  default="train", help="HF split (default: train)")
    parser.add_argument("--hf-file",   default=None,
                        help="直接指定 HF repo 中的文件名，如 eval_1600.json（用于非标准 split）")
    parser.add_argument("--output",    required=True, help="Output JSONL file path")
    args = parser.parse_args()

    if args.input:
        convert_parquet(args.input, args.output)
    elif args.hf_file:
        convert_hf_file(args.hf_dataset, args.hf_file, args.output)
    else:
        convert_hf(args.hf_dataset, args.hf_split, args.output)


if __name__ == "__main__":
    main()
