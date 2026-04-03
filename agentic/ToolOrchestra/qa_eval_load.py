"""
Load QA eval examples (HLE / FRAMES) from JSONL or Parquet.

Parquet column names are auto-mapped to the schema expected by eval_orchestra.eval_qa:
  id, question, answer
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)


def _cell_to_jsonable(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "item") and callable(getattr(v, "item")):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def _row_to_plain_dict(raw: dict) -> dict:
    return {str(k): _cell_to_jsonable(v) for k, v in raw.items()}


def _normalize_qa_example(row: dict, row_index: int) -> dict:
    """Map a dataset row to eval_qa format: id, question, answer."""
    out: dict = {}
    id_keys = ("id", "uuid", "problem_id", "example_id", "question_id", "eid")
    for k in id_keys:
        if k in row and row[k] is not None and str(row[k]).strip():
            out["id"] = str(row[k]).strip()
            break
    if "id" not in out:
        out["id"] = f"row_{row_index}"

    q_keys = ("question", "problem", "prompt", "query", "text", "instruction")
    for k in q_keys:
        if k not in row or row[k] is None:
            continue
        v = row[k]
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False).strip()
        else:
            s = str(v).strip()
        if s:
            out["question"] = s
            break
    if "question" not in out:
        raise ValueError(f"no question-like column; keys={list(row.keys())}")

    a_keys = (
        "answer",
        "ground_truth",
        "gold",
        "correct_answer",
        "author_answer",
        "final_answer",
        "solution",
        "target",
    )
    out["answer"] = ""
    for k in a_keys:
        if k not in row or row[k] is None:
            continue
        v = row[k]
        if isinstance(v, (dict, list)):
            out["answer"] = json.dumps(v, ensure_ascii=False)
        else:
            out["answer"] = str(v).strip()
        break

    if "metadata" in row and isinstance(row["metadata"], dict):
        out["metadata"] = row["metadata"]
    return out


def _parquet_file_paths(source: str) -> list[str]:
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        paths = sorted(
            glob.glob(os.path.join(source, "*.parquet"))
            + glob.glob(os.path.join(source, "*.pq"))
        )
        if not paths:
            raise FileNotFoundError(f"No *.parquet or *.pq under {source}")
        return paths
    raise FileNotFoundError(source)


def load_qa_examples_parquet(source: str) -> list[dict]:
    """Read one Parquet file or a directory of Parquet shards (e.g. HF dataset export)."""
    try:
        import pandas as pd
    except ImportError:
        pd = None

    paths = _parquet_file_paths(source)
    all_rows: list[dict] = []
    for p in paths:
        if pd is not None:
            df = pd.read_parquet(p)
            records = df.to_dict("records")
        else:
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise RuntimeError(
                    "Reading Parquet requires pandas or pyarrow. "
                    "Install: pip install pandas pyarrow"
                ) from e
            table = pq.read_table(p)
            records = table.to_pylist()
        for raw in records:
            all_rows.append(_row_to_plain_dict(raw))

    examples: list[dict] = []
    for i, row in enumerate(all_rows):
        try:
            examples.append(_normalize_qa_example(row, i))
        except ValueError as e:
            logger.warning("skip row %d: %s", i, e)
    return examples


def load_eval_examples_jsonl(path: str) -> list[dict]:
    examples: list[dict] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        first_nonempty = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first_nonempty and (
                "git-lfs.github.com" in line or line.startswith("version https://git-lfs")
            ):
                raise ValueError(
                    f"{path} is a Git LFS pointer, not JSONL. "
                    "Pull Git LFS or point --eval-data to a real .jsonl / .parquet file."
                )
            first_nonempty = False
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                if "git-lfs" in line.lower():
                    raise ValueError(
                        f"{path}: invalid JSON (Git LFS pointer?). "
                        "Use real data or Parquet via --eval-data."
                    ) from e
                raise
    return examples


def load_eval_examples(path: str, benchmark: str) -> list[dict]:
    """
    Load examples for eval_orchestra.main().

    - tau2: JSONL file only.
    - hle / frames: JSONL file, or .parquet/.pq file, or directory of Parquet shards.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    parquets_in_dir: list[str] = []
    if os.path.isdir(path):
        parquets_in_dir = sorted(
            glob.glob(os.path.join(path, "*.parquet"))
            + glob.glob(os.path.join(path, "*.pq"))
        )

    is_parquet_file = os.path.isfile(path) and path.lower().endswith((".parquet", ".pq"))
    use_parquet = is_parquet_file or bool(parquets_in_dir)

    if use_parquet:
        if benchmark == "tau2":
            raise ValueError("tau2 benchmark expects JSONL --eval-data, not Parquet.")
        if parquets_in_dir:
            logger.info("Loading %d Parquet shard(s) from %s", len(parquets_in_dir), path)
        else:
            logger.info("Loading Parquet: %s", path)
        return load_qa_examples_parquet(path)

    if os.path.isdir(path):
        raise ValueError(
            f"benchmark={benchmark}: eval-data must be a JSONL file (or Parquet dir for hle/frames): {path}"
        )

    return load_eval_examples_jsonl(path)
