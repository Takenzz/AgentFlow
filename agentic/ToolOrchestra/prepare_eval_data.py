#!/usr/bin/env python3
"""
Prepare ToolOrchestra evaluation data
======================================
Generate evaluation JSONL from raw ToolOrchestra eval data (frames.jsonl + tau2 original_tasks.json),
with model_mapping consistent with the training data (data_slime_full.jsonl).

Output:
  - <SCRIPT_DIR>/data/eval_frames.jsonl   — QA evaluation (frames benchmark)
  - <SCRIPT_DIR>/data/eval_tau2.jsonl     — func_call evaluation (tau2 benchmark)
  - <SCRIPT_DIR>/data/eval_combined.jsonl — combined (tau2 + QA)

Usage:
    python prepare_eval_data.py
    python prepare_eval_data.py --max-per-domain 10 --max-qa 100
"""

import argparse
import json
import os
from pathlib import Path

# ── model_mapping consistent with the training data ──────────────────────── #
FUNC_CALL_MODEL_MAPPING = {
    "expert-1": "Qwen/Qwen3-32B",
    "expert-2": "Qwen/Qwen3-30B-A3B",
    "expert-3": "Qwen/Qwen3-14B",
}

QA_MODEL_MAPPING = {
    "search-1":      "Qwen/Qwen3-32B",
    "reasoner-1":    "Qwen/Qwen3-32B",
    "reasoner-2":    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "reasoner-3":    "Qwen/Qwen3-14B",
    "answer-1":      "Qwen/Qwen3-32B",
    "answer-2":      "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "answer-3":      "Qwen/Qwen3-30B-A3B",
    "answer-math-1": "Qwen/Qwen2.5-Math-72B-Instruct",
    "answer-math-2": "Qwen/Qwen2.5-Math-7B-Instruct",
}

TOOL_PRICING = {
    "Qwen/Qwen3-32B":                           {"input_tokens_per_million": 6e-7,  "output_tokens_per_million": 1.2e-6},
    "Qwen/Qwen2.5-Coder-32B-Instruct":          {"input_tokens_per_million": 8e-7,  "output_tokens_per_million": 8e-7},
    "Qwen/Qwen2.5-Math-7B-Instruct":            {"input_tokens_per_million": 2e-7,  "output_tokens_per_million": 2e-7},
    "Qwen/Qwen3-14B":                           {"input_tokens_per_million": 3e-7,  "output_tokens_per_million": 3e-7},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"input_tokens_per_million": 8e-7,  "output_tokens_per_million": 8e-7},
    "Qwen/Qwen3-30B-A3B":                       {"input_tokens_per_million": 5e-7,  "output_tokens_per_million": 5e-7},
    "Qwen/Qwen2.5-Math-72B-Instruct":           {"input_tokens_per_million": 1.2e-6, "output_tokens_per_million": 1.2e-6},
}

# Directory containing this script (used as the relative base)
_SCRIPT_DIR = Path(__file__).parent.resolve()

# tau2 data directory (local copy in slime-agentic)
TAU2_DATA_DIR = _SCRIPT_DIR / "tau2_data"

# Training data (used to extract task examples for each domain)
TRAIN_DATA_JSONL = _SCRIPT_DIR / "data" / "data.jsonl"

# QA data (frames)
FRAMES_JSONL = _SCRIPT_DIR / "evaluation" / "frames.jsonl"

# Output directory
OUTPUT_DIR = _SCRIPT_DIR / "data"


def _call_expert_tool_def() -> dict:
    """tool definition for func_call orchestrator (call_expert)."""
    return {
        "type": "function",
        "function": {
            "name": "call_expert",
            "description": "Delegate the current task turn to an expert model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expert": {
                        "type": "string",
                        "enum": list(FUNC_CALL_MODEL_MAPPING.keys()),
                        "description": (
                            "The expert to call. Choices: ['expert-1', 'expert-2', 'expert-3']. "
                            "expert-1: strong function-calling ability; excellent across most domains (Qwen3-32B). "
                            "expert-2: strong MoE reasoning model with broad capability (Qwen3-30B-A3B). "
                            "expert-3: lightweight model for simpler tasks (Qwen3-14B)."
                        ),
                    }
                },
                "required": ["expert"],
            },
        },
    }


def _extract_problem(task: dict) -> str:
    """Extract user-facing problem description from a tau2 task."""
    us = task.get("user_scenario", {})
    if isinstance(us, dict):
        instr = us.get("instructions", {})
        if isinstance(instr, dict):
            return instr.get("task_instructions", "")
        if isinstance(instr, str):
            return instr
    if isinstance(us, str):
        return us
    desc = task.get("description", {})
    if isinstance(desc, dict):
        return desc.get("purpose", "")
    return ""


def build_tau2_eval(max_per_domain: int) -> list[dict]:
    """Build func_call eval examples.

    Strategy:
    1. Domains with dedicated task files (airline, mock, retail, telecom):
       use original_tasks.json / tasks.json
    2. Other domains (bank, medicine, movie, etc.):
       extract unique tasks from data.jsonl (training data)
       — take the LAST max_per_domain examples per domain to avoid overlap
         with what may have been used early in training
    """
    examples = []

    # ── Step 1: domains with dedicated task files ────────────────────────── #
    domains_from_files: set[str] = set()
    if TAU2_DATA_DIR.exists():
        for domain_dir in sorted(TAU2_DATA_DIR.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            if domain in ("user_simulator",):
                continue

            # Prefer original_tasks.json (curated), fall back to tasks.json
            task_file = domain_dir / "original_tasks.json"
            if not task_file.exists():
                task_file = domain_dir / "tasks.json"
            if not task_file.exists():
                continue

            with open(task_file) as f:
                tasks = json.load(f)
            if not isinstance(tasks, list) or not tasks:
                continue

            tasks = tasks[:max_per_domain]
            domains_from_files.add(domain)

            for task in tasks:
                task_id = str(task.get("id", "0"))
                eid = f"{domain}____{task_id}"
                examples.append({
                    "problem": _extract_problem(task),
                    "answer": "",
                    "tools": [_call_expert_tool_def()],
                    "metadata": {
                        "eid": eid,
                        "category": "func_call",
                        "model_mapping": FUNC_CALL_MODEL_MAPPING.copy(),
                        "tool_pricing": TOOL_PRICING.copy(),
                        "pref_vec": {"accuracy": 1.0, "cost": -0.3, "latency": -0.1},
                        "example": task,
                    },
                })
            print(f"  {domain}: {len(tasks)} tasks (from task file)")

    # ── Step 2: remaining domains — extract from data.jsonl ──────────────── #
    if TRAIN_DATA_JSONL.exists():
        domain_tasks: dict[str, list] = {}
        with open(TRAIN_DATA_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                eid = row.get("eid") or row.get("id", "")
                domain = eid.split("____")[0] if "____" in eid else ""
                cat = row.get("category", "")
                if not domain or domain in domains_from_files or cat != "func_call":
                    continue
                task_data = row.get("example")
                if not task_data:
                    continue
                if domain not in domain_tasks:
                    domain_tasks[domain] = []
                domain_tasks[domain].append((eid, row.get("problem", ""), task_data))

        for domain, task_list in sorted(domain_tasks.items()):
            # Take the last max_per_domain to reduce training overlap
            selected = task_list[-max_per_domain:]
            for eid, problem, task_data in selected:
                examples.append({
                    "problem": problem,
                    "answer": "",
                    "tools": [_call_expert_tool_def()],
                    "metadata": {
                        "eid": eid,
                        "category": "func_call",
                        "model_mapping": FUNC_CALL_MODEL_MAPPING.copy(),
                        "tool_pricing": TOOL_PRICING.copy(),
                        "pref_vec": {"accuracy": 1.0, "cost": -0.3, "latency": -0.1},
                        "example": task_data,
                    },
                })
            print(f"  {domain}: {len(selected)} tasks (from data.jsonl)")

    return examples


def build_frames_eval(max_qa: int) -> list[dict]:
    """Build QA eval examples from frames.jsonl."""
    if not FRAMES_JSONL.exists():
        print(f"[WARN] frames.jsonl not found: {FRAMES_JSONL}")
        return []

    examples = []
    with open(FRAMES_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # frames.jsonl format: {id, question, answer}
            # Keep as-is for frames benchmark (eval_orchestra reads it directly)
            examples.append(row)
            if max_qa and len(examples) >= max_qa:
                break

    print(f"  frames QA: {len(examples)} examples")
    return examples


def build_qa_slime_eval(max_qa: int) -> list[dict]:
    """Convert frames examples to slime metadata format (for combined eval)."""
    raw = build_frames_eval(max_qa)
    examples = []
    for row in raw:
        eid = f"stem____{row['id']}"
        example = {
            "problem": row["question"],
            "answer": row.get("answer", ""),
            "tools": [],   # QA tools are passed globally via --qa-tools-json
            "metadata": {
                "eid": eid,
                "category": "qa",
                "model_mapping": QA_MODEL_MAPPING.copy(),
                "tool_pricing": TOOL_PRICING.copy(),
                "pref_vec": {"accuracy": 1.0, "cost": -0.3, "latency": -0.1},
                "example": row,
            },
        }
        examples.append(example)
    return examples


def write_jsonl(path: Path, examples: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[WRITTEN] {path} ({len(examples)} examples)")


def main():
    parser = argparse.ArgumentParser(description="Prepare ToolOrchestra eval data")
    parser.add_argument("--max-per-domain", type=int, default=20,
                        help="Max tau2 tasks per domain (default: 20)")
    parser.add_argument("--max-qa", type=int, default=200,
                        help="Max QA (frames) examples (default: 200, 0=all)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    max_qa = args.max_qa or 0

    print("=== Building tau2 func_call eval ===")
    tau2_examples = build_tau2_eval(args.max_per_domain)

    print("\n=== Building frames QA eval ===")
    frames_examples = build_frames_eval(max_qa)         # raw format for --benchmark frames
    qa_slime = build_qa_slime_eval(max_qa)              # slime format for combined

    print("\n=== Writing output files ===")
    # tau2 eval JSONL (replaces the wrong eval_100.jsonl)
    write_jsonl(out / "eval_tau2.jsonl", tau2_examples)
    # frames eval stays as-is (already at evaluation/frames.jsonl)
    # combined eval for convenience
    combined = tau2_examples + qa_slime
    write_jsonl(out / "eval_combined.jsonl", combined)

    print(f"\nDone.")
    print(f"  tau2 eval:     {out}/eval_tau2.jsonl  ({len(tau2_examples)} examples)")
    print(f"  frames eval:   {FRAMES_JSONL}  ({len(frames_examples)} examples, unchanged)")
    print(f"  combined eval: {out}/eval_combined.jsonl  ({len(combined)} examples)")
    print()
    print("Usage:")
    print("  # tau2 benchmark:")
    print(f"  EVAL_DATA={out}/eval_tau2.jsonl BENCHMARK=tau2 bash eval_orchestra.sh")
    print("  # frames benchmark (uses original frames.jsonl directly):")
    print(f"  BENCHMARK=frames bash eval_orchestra.sh")


if __name__ == "__main__":
    main()
