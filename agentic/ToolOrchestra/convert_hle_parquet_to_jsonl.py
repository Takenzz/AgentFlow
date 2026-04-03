#!/usr/bin/env python3
"""Convert HLE (or similar) Parquet → JSONL for eval_orchestra.py."""

from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from qa_eval_load import load_qa_examples_parquet


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parquet file or HF-style shard directory → JSONL (id, question, answer)"
    )
    ap.add_argument("input", help="Single .parquet or directory of *.parquet")
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path, e.g. evaluation/hle.jsonl",
    )
    args = ap.parse_args()

    examples = load_qa_examples_parquet(args.input)
    out_abs = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples → {out_abs}")


if __name__ == "__main__":
    main()
