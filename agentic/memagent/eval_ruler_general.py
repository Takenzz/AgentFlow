"""
eval_ruler_general.py — RULER 合成任务评测脚本（slime-agentic 版）
=================================================================
与 /data/MemAgent/taskutils/memory_eval/ruler_general.py 核心思想、
数据格式、评测指标完全对齐，推理侧使用 rollout.py 的提示词
（\\boxed{} 格式，与训练一致）。

评测数据：MemAgent 格式的 eval_{split}_{length}.json
    字段：input（问题）、outputs（期望答案列表）、context（文档文本）

任务列表及指标：
    - niah_single_{1-3}, niah_multikey_{1-3},
      niah_multivalue, niah_multiquery,
      vt, fwe              → sub_EM（所有期望串均在预测中出现的比例）
    - qa_1, qa_2           → F1 / EM / sub_EM（标准 QA 指标）

用法：
    python eval_ruler_general.py \\
        --split  niah_single_1 \\
        --length 131072 \\
        --model  Qwen2.5-7B-Instruct \\
        --tokenizer /path/to/tokenizer \\
        --data-root /path/to/ruler_data \\
        --save-dir results/ruler_general \\
        --save-file my_model

环境变量（与 run_memagent_7b.sh 保持一致）：
    SERVE_HOST / SERVE_PORT / MEM_CHUNK_TOKENS / MEM_MAX_MEMORY
    MEM_MAX_FINAL / MEM_MAX_CHUNKS / MEM_MAX_CTX_TOKENS / DATAROOT
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp
from tqdm import tqdm
from transformers import AutoTokenizer

# ── 环境变量 ─────────────────────────────────────────────────────────────────
SERVE_HOST      = os.getenv("SERVE_HOST",        "127.0.0.1")
SERVE_PORT      = os.getenv("SERVE_PORT",        "8000")
CHUNK_TOKENS    = int(os.getenv("MEM_CHUNK_TOKENS", "5000"))
MAX_MEMORY_TOKS = int(os.getenv("MEM_MAX_MEMORY",   "1024"))
MAX_FINAL_TOKS  = int(os.getenv("MEM_MAX_FINAL",    "256"))
MAX_CHUNKS      = int(os.getenv("MEM_MAX_CHUNKS",   "512"))
MAX_CTX_TOKENS  = int(os.getenv("MEM_MAX_CTX_TOKENS", str(10 ** 12)))
BASE_URL        = f"http://{SERVE_HOST}:{SERVE_PORT}/v1"
API_KEY         = os.getenv("SERVE_API_KEY", "EMPTY")
DATAROOT        = os.getenv("DATAROOT", "/data/ruler")

# QA 任务需要拼装上下文的外部数据文件（路径可通过环境变量覆盖）
SQUAD_PATH    = os.getenv("SQUAD_PATH",    "")   # qa_1
HOTPOTQA_PATH = os.getenv("HOTPOTQA_PATH", "")   # qa_2

# 所有支持的 RULER 任务及其指标类型
RULER_TASKS = {
    "niah_single_1":  "sub_em",
    "niah_single_2":  "sub_em",
    "niah_single_3":  "sub_em",
    "niah_multikey_1":"sub_em",
    "niah_multikey_2":"sub_em",
    "niah_multikey_3":"sub_em",
    "niah_multivalue":"sub_em",
    "niah_multiquery":"sub_em",
    "vt":             "sub_em",
    "fwe":            "sub_em",
    "qa_1":           "qa",
    "qa_2":           "qa",
}

# ── 提示词模板（与 rollout.py 完全一致）──────────────────────────────────────
_MEMORY_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

_FINAL_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

_NO_MEMORY = "No previous memory"
_STOP_TOKEN_STRINGS = ["<|im_end|>", "<|endoftext|>"]


# ── 答案提取（与 rollout.py 一致）─────────────────────────────────────────────

def _strip_stop_tokens(text: str) -> str:
    for tok in _STOP_TOKEN_STRINGS:
        text = text.replace(tok, "")
    return text.strip()


def _last_boxed_only_string(s: str) -> str | None:
    if "\\boxed " in s:
        return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i, right_brace_idx, opens = idx, None, 0
    while i < len(s):
        if s[i] == "{":
            opens += 1
        if s[i] == "}":
            opens -= 1
            if opens == 0:
                right_brace_idx = i
                break
        i += 1
    return s[idx: right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    left = "\\boxed{"
    assert s.startswith(left) and s.endswith("}")
    return s[len(left):-1]


def _extract_boxed(text: str) -> str:
    s = _last_boxed_only_string(text)
    if s is None:
        return ""
    try:
        return _remove_boxed(s).strip()
    except Exception:
        return ""


# ── 评测指标 ──────────────────────────────────────────────────────────────────

def _string_match_all(pred: str, refs: list[str]) -> float:
    """RULER 非 QA 任务指标：所有期望串都在 pred 中出现的比例。
    与 ruler_general.py string_match_all 完全一致。"""
    if not refs:
        return 0.0
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in refs) / len(refs)


def _normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):
        return " ".join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    p = _normalize_answer(prediction).split()
    g = _normalize_answer(ground_truth).split()
    if not p or not g:
        return 0.0, 0.0, 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    prec   = num_same / len(p)
    recall = num_same / len(g)
    f1     = 2 * prec * recall / (prec + recall)
    return f1, prec, recall


def _exact_match(prediction: str, ground_truth: str) -> float:
    p = _normalize_answer(prediction)
    g = _normalize_answer(ground_truth)
    if p in ("yes", "no", "noanswer") and p != g:
        return 0.0
    if g in ("yes", "no", "noanswer") and p != g:
        return 0.0
    return float(p == g)


def _sub_exact_match(prediction: str, ground_truth: str) -> float:
    p = _normalize_answer(prediction)
    g = _normalize_answer(ground_truth)
    return float((g in p) or (p in g))


def _score_item(pred: str, outputs: list[str], metric_type: str) -> dict:
    """计算单条样本的分数，返回 dict 供写入 JSONL。"""
    if metric_type == "qa":
        gold = outputs[0] if outputs else ""
        f1, _, _ = _f1_score(pred, gold)
        return {
            "judge_f1":     f1,
            "judge_em":     _exact_match(pred, gold),
            "judge_sub_em": _sub_exact_match(pred, gold),
        }
    else:
        return {
            "judge_sub_em": _string_match_all(pred, outputs),
        }


def _agg_metrics(records: list[dict], metric_type: str) -> dict:
    n = len(records)
    if not n:
        return {}
    keys = ("judge_f1", "judge_em", "judge_sub_em") if metric_type == "qa" else ("judge_sub_em",)
    result = {}
    for k in keys:
        vals = [r[k] for r in records if k in r]
        if vals:
            result[k.replace("judge_", "")] = round(sum(vals) / len(vals), 4)
    result["total"] = n
    return result


# ── HTTP 工具 ────────────────────────────────────────────────────────────────

async def _chat_once(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    payload = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    async with session.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


# ── 推理逻辑 ─────────────────────────────────────────────────────────────────

async def _recurrent_infer(
    item: dict,
    model: str,
    tokenizer,
    temperature: float,
    top_p: float,
    sem: asyncio.Semaphore,
) -> str:
    question = item["input"].strip()
    context  = item["context"].strip()

    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ctx_ids) > MAX_CTX_TOKENS:
        half = MAX_CTX_TOKENS // 2
        ctx_ids = ctx_ids[:half] + ctx_ids[-half:]

    chunks = [
        ctx_ids[i: i + CHUNK_TOKENS]
        for i in range(0, len(ctx_ids), CHUNK_TOKENS)
    ][:MAX_CHUNKS]

    async with sem:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=86400)
        ) as session:
            memory = _NO_MEMORY
            for chunk_ids in chunks:
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                msg = _MEMORY_TEMPLATE.format(
                    prompt=question, memory=memory, chunk=chunk_text
                )
                raw = await _chat_once(
                    session, model,
                    [{"role": "user", "content": msg}],
                    temperature, top_p, MAX_MEMORY_TOKS,
                )
                memory = _strip_stop_tokens(raw) or memory

            final_msg = _FINAL_TEMPLATE.format(prompt=question, memory=memory)
            response = await _chat_once(
                session, model,
                [{"role": "user", "content": final_msg}],
                temperature, top_p, MAX_FINAL_TOKS,
            )
    return response.strip()


async def _openai_infer(
    item: dict,
    model: str,
    tokenizer,
    temperature: float,
    top_p: float,
    max_input_len: int,
    max_output_len: int,
    sem: asyncio.Semaphore,
) -> str:
    question = item["input"].strip()
    context  = item["context"].strip()
    prompt   = (
        f"{context}\n\n"
        f"Question: {question}\n"
        "Please put the answer in \\boxed{}.\n\n"
        "Answer:"
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input_len:
        prompt = tokenizer.decode(ids[:max_input_len], skip_special_tokens=True)

    async with sem:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=86400)
        ) as session:
            response = await _chat_once(
                session, model,
                [{"role": "user", "content": prompt}],
                temperature, top_p, max_output_len,
            )
    return response.strip()


# ── 主评测流程 ────────────────────────────────────────────────────────────────

async def run_eval(
    data: list[dict],
    args: argparse.Namespace,
    tokenizer,
    metric_type: str,
) -> None:
    out_path = Path(args.save_dir) / f"{args.save_file}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cached_ids: set[int] = set()
    if out_path.exists() and not args.force:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    cached_ids.add(json.loads(line)["_id"])
                except Exception:
                    pass

    todo = [item for item in data if item["_id"] not in cached_ids]
    print(
        f"Data: {len(data)}  Cached: {len(cached_ids)}  "
        f"Todo: {len(todo)}  Concurrency: {args.n_proc}"
    )
    if not todo:
        print("All done (cached).")
        _print_existing_stats(out_path, metric_type)
        return

    sem = asyncio.Semaphore(args.n_proc)

    async def process_one(item: dict) -> dict | None:
        try:
            if args.api == "recurrent":
                response = await _recurrent_infer(
                    item, args.model, tokenizer,
                    args.temperature, args.top_p, sem,
                )
            else:
                response = await _openai_infer(
                    item, args.model, tokenizer,
                    args.temperature, args.top_p,
                    args.max_input_len, args.max_output_len, sem,
                )
        except Exception:
            import traceback
            traceback.print_exc()
            return None

        outputs = item.get("outputs", [])
        if isinstance(outputs, str):
            outputs = [outputs]
        pred = _extract_boxed(response[-300:].lower()) or ""

        result = {
            "_id":     item["_id"],
            "pred":    pred,
            "answer":  outputs,
            "response": response,
        }
        result.update(_score_item(pred, outputs, metric_type))
        for k, v in item.items():
            if k not in ("context", "response") and k not in result:
                result[k] = v
        return result

    tasks = [process_one(item) for item in todo]

    records: list[dict] = []
    n_err = 0
    first_shown = False
    fout = open(out_path, "a", encoding="utf-8")
    pbar = tqdm(total=len(tasks), desc=f"ruler_general[{args.split}/{args.length}]")

    for coro in asyncio.as_completed(tasks):
        result = await coro
        pbar.update(1)
        if result is None:
            n_err += 1
            continue
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        records.append(result)
        if not first_shown:
            first_shown = True
            _print_sample(result, metric_type)

    pbar.close()
    fout.close()

    all_records = records[:]
    if cached_ids:
        with open(out_path, encoding="utf-8") as f:
            all_records = [json.loads(l) for l in f if l.strip()]

    stats = _agg_metrics(all_records, metric_type)
    print(f"\n=== ruler_general [{args.split}/{args.length}]  "
          f"total={stats.get('total', 0)}  errors={n_err} ===")
    for k in ("f1", "em", "sub_em"):
        if k in stats:
            print(f"  {k}: {round(stats[k] * 100, 2)}")


def _print_sample(result: dict, metric_type: str) -> None:
    sep = "=" * 40
    print(f"\n{sep} Sample {result['_id']} {sep}")
    print(f"[response]  {result['response'][:500]}")
    print(f"[pred]      {result['pred']}")
    print(f"[answer]    {result['answer']}")
    key = "judge_sub_em" if metric_type != "qa" else "judge_f1"
    print(f"[{key}]  {result.get(key, 'n/a')}")
    print(sep)


def _print_existing_stats(out_path: Path, metric_type: str) -> None:
    records = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        return
    stats = _agg_metrics(records, metric_type)
    print(f"Existing results ({stats.get('total', 0)} samples):")
    for k in ("f1", "em", "sub_em"):
        if k in stats:
            print(f"  {k}: {round(stats[k] * 100, 2)}")


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def _assemble_qa_context(items: list[dict], split: str, data_root: str) -> list[dict]:
    """
    qa_1 / qa_2 任务的 context 字段是文档索引列表，需要从外部文件拼装。
    与 ruler_general.py 的 set_context 逻辑一致。
    """
    doc_prompt = "Document {i}:\n{document}"

    if split == "qa_1":
        path = SQUAD_PATH or str(Path(data_root) / "squad.json")
        with open(path, encoding="utf-8") as f:
            squad = json.load(f)
        all_docs = sorted(list({
            p["context"]
            for d in squad["data"]
            for p in d["paragraphs"]
        }))
    elif split == "qa_2":
        path = HOTPOTQA_PATH or str(Path(data_root) / "hotpotqa_dev.json")
        with open(path, encoding="utf-8") as f:
            hotpot = json.load(f)
        all_docs = sorted(list({
            f"{t}\n{''.join(p)}"
            for d in hotpot
            for t, p in d["context"]
        }))
    else:
        raise ValueError(f"Unknown QA split: {split}")

    for item in items:
        indices = item["context"]
        docs = [all_docs[i] for i in indices]
        item["context"] = "\n\n".join(
            doc_prompt.format(i=i + 1, document=d) for i, d in enumerate(docs)
        )
    return items


def load_data(data_root: str, split: str, length: int) -> tuple[list[dict], str]:
    """返回 (data, metric_type)。"""
    data_path = Path(data_root) / f"eval_{split}_{length}.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find: {data_path}\n"
            f"Set --data-root or DATAROOT to the directory with "
            f"eval_{{split}}_{{length}}.json files."
        )

    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = list(raw.values())

    data = []
    for idx, item in enumerate(raw):
        item = dict(item)
        item.setdefault("_id", idx)
        # 确保 outputs 是 list
        if "outputs" not in item and "answers" in item:
            item["outputs"] = item.pop("answers")
        if isinstance(item.get("outputs"), str):
            item["outputs"] = [item["outputs"]]
        data.append(item)

    metric_type = RULER_TASKS.get(split, "sub_em")

    # QA 任务需要拼装上下文
    if metric_type == "qa" and data and isinstance(data[0].get("context"), list):
        data = _assemble_qa_context(data, split, data_root)

    return data, metric_type


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RULER general eval — slime-agentic version"
    )
    parser.add_argument(
        "--split", type=str, default="niah_single_1",
        choices=list(RULER_TASKS.keys()),
        help="RULER task split",
    )
    parser.add_argument(
        "--length", type=int, default=131072,
        choices=[8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
        help="Context length in tokens",
    )
    parser.add_argument(
        "--data-root", default=DATAROOT,
        help="Directory containing eval_{split}_{length}.json files",
    )
    parser.add_argument("--save-dir",  "-s", default="results/ruler_general",
                        help="Output directory for JSONL results")
    parser.add_argument("--save-file", "-f", default="model",
                        help="Output filename stem")
    parser.add_argument("--model",     "-m", required=True)
    parser.add_argument("--tokenizer", "-t", required=True)
    parser.add_argument(
        "--api", default="recurrent", choices=["recurrent", "openai"],
        help="recurrent: chunk-by-chunk (default); openai: single-turn long context",
    )
    parser.add_argument("--n-proc",    "-n", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p",     type=float, default=0.95)
    parser.add_argument("--max-input-len",  type=int, default=120000)
    parser.add_argument("--max-output-len", type=int, default=10000)
    parser.add_argument("--sampling",  type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"[config] SERVE={SERVE_HOST}:{SERVE_PORT}  api={args.api}  "
          f"split={args.split}  length={args.length}")
    print(f"[config] CHUNK_TOKENS={CHUNK_TOKENS}  MAX_MEMORY={MAX_MEMORY_TOKS}  "
          f"MAX_FINAL={MAX_FINAL_TOKS}  MAX_CHUNKS={MAX_CHUNKS}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    data, metric_type = load_data(args.data_root, args.split, args.length)
    print(f"[data] {len(data)} samples  metric_type={metric_type}")

    if args.sampling > 1:
        base = data[:]
        data = []
        for s in range(args.sampling):
            for item in base:
                new_item = copy.deepcopy(item)
                new_item["_id"] = item["_id"] * args.sampling + s
                data.append(new_item)

    asyncio.run(run_eval(data, args, tokenizer, metric_type))


if __name__ == "__main__":
    main()
