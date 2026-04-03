"""
ToolOrchestra rollout.py
------------------------
符合 slime 框架约定的 generate / reward_func 入口。

    generate()   — 从 sample 解包数据，创建 OrchestraSolver，调用 solve()
    reward_func  — 从 reward.py re-export，供 slime 通过 --custom-rm-path 加载
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from agentflow.core.llm_engine import SGLangEngine
from orchestra_solver import OrchestraSolver
from tools.expert_caller.tool import EXPERT_ENGINE_MAP
from reward import reward_func  # noqa: F401  slime 通过 rollout.reward_func 加载

RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"
# 单条 sample rollout 超时（秒），防止长尾 straggler 拖慢整个 batch
ROLLOUT_SAMPLE_TIMEOUT = 360


async def generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """
    slime 框架调用入口（每条 sample 调用一次）。

    sample.prompt   — 问题文本（--input-key problem）
    sample.label    — ground truth（--label-key answer）
    sample.metadata — 其余字段，由 data loader 从 JSONL 的 "metadata" 字段填充，包含：
                          category / tools / model_mapping / eid / pref_vec / tool_pricing
    """
    state = GenerateState(args)
    url   = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    orch_engine = SGLangEngine(
        url=url,
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        enable_thinking=True,
    )
    engine_map = {"default": orch_engine}

    if isinstance(sample.prompt, list):
        question = next(
            (m["content"] for m in reversed(sample.prompt) if m.get("role") == "user"),
            str(sample.prompt),
        )
    else:
        question = str(sample.prompt)

    label = str(sample.label) if sample.label is not None else None

    if not isinstance(sample.metadata, dict):
        sample.metadata = {}

    sample.metadata["_is_eval"] = evaluation

    try:
        solver = OrchestraSolver(
            engine_map=engine_map,
            sample_meta=sample.metadata,
            retrieval_url=RETRIEVAL_URL,
            expert_engine_map=EXPERT_ENGINE_MAP,
            max_turns=12,
        )
        out = await asyncio.wait_for(
            solver.solve(question, label=label),
            timeout=ROLLOUT_SAMPLE_TIMEOUT,
        )

        if out is None:
            sample.status = Sample.Status.ABORTED
            sample.rollout_log_probs = []
            return sample

        sample.prompt            = out.prompt_text
        sample.response          = out.response
        sample.tokens            = out.prompt_token_ids + out.token_ids
        sample.response_length   = len(out.token_ids)
        sample.loss_mask         = out.loss_mask if out.loss_mask is not None else [1] * len(out.token_ids)
        sample.rollout_log_probs = out.log_probs
        sample.status            = (
            Sample.Status.TRUNCATED if out.finish_reason == "length"
            else Sample.Status.COMPLETED
        )
        sample.metadata["final_output"] = out.final_output or ""

        if out.turns:
            sample.train_metadata = {"turns": out.turns}

    except TimeoutError:
        print(f"[rollout] sample timed out after {ROLLOUT_SAMPLE_TIMEOUT}s, marking FAILED")
        sample.response          = ""
        sample.rollout_log_probs = []
        sample.status            = Sample.Status.FAILED
    except Exception:
        traceback.print_exc()
        sample.response          = ""
        sample.rollout_log_probs = []
        sample.status            = Sample.Status.FAILED

    return sample
