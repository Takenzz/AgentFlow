"""
OrchestraSolver
---------------
Main loop for the ToolOrchestra Orchestrator, conforming to the slime-agentic Solver interface:

    __init__(engine_map, sample_meta, ...)   — receives engines + sample metadata
    solve(question, label=None)              — fixed interface, consistent with agentflow.Solver
    returns GenerationOutput                 — contains turns / loss_mask for custom_convert to unroll

Per-turn training sequence:
    Orchestrator output tokens → loss_mask = 1
    tool result (injected into prompt) tokens → loss_mask = 0
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Optional

from slime.utils.http_utils import post as slime_post

from agentflow.core.llm_engine import GenerationOutput
from prompt_builder import PromptBuilder
from tau2_adapter import Tau2Adapter
from tool_call_parser import parse_tool_call, parse_all_tool_calls
from tools.search_retrieval.tool import SearchRetrievalTool
from tools.expert_caller.tool import ExpertCallerTool

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 12


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks for concise display, keep tool_call and message."""
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return stripped.strip() if stripped.strip() else text[:300]


def _extract_message_for_tau2(text: str) -> str:
    """Extract clean message content for tau2 from raw model output.

    Strips <think> blocks and extracts text from <message> tags if present.
    tau2 expects plain text, not XML-wrapped content.
    """
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    m = re.search(r"<message>(.*?)</message>", cleaned, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    cleaned = cleaned.strip()
    return cleaned if cleaned else "Wait a minute, I will take it very soon"


class OrchestraSolver:
    """
    Args:
        engine_map:        {"default": SGLangEngine, ...}
                           Uses "orchestrator" or "default" as the Orchestrator engine.
                           SGLangEngine carries the tokenizer and the SGLang /generate URL.
        sample_meta:       Per-sample metadata with fields:
                               category       "qa" | "func_call"
                               tools          Tool definition list (OpenAI function calling format)
                               model_mapping  {role: model_name}
                               eid            Sample ID (used for FAISS document filtering)
                               pref_vec       Preference weights (used for reward)
                               answer         Ground truth (used for reward)
        retrieval_url:     Address of the FAISS retrieval service
        expert_engine_map: {model_name: base_url}, used by ExpertCallerTool
        max_turns:         Maximum number of tool-calling turns
    """

    def __init__(
        self,
        engine_map: dict[str, Any],
        sample_meta: dict,
        retrieval_url: str = "http://127.0.0.1:8000/retrieve",
        expert_engine_map: dict | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ):
        self._engine    = engine_map.get("orchestrator") or engine_map["default"]
        self._meta      = sample_meta
        self._max_turns = max_turns
        self._builder   = PromptBuilder()
        self._context_parts: list[str] = []

        eid = sample_meta.get("eid", "") or sample_meta.get("id", "")
        self._search = SearchRetrievalTool(retrieval_url=retrieval_url, eid=eid)
        self._expert = ExpertCallerTool(
            model_mapping=sample_meta.get("model_mapping", {}),
            expert_engine_map=expert_engine_map or {},
            max_tokens=8192,
        )
        self._expert_consecutive_failures = 0
        self._expert_circuit_breaker_limit = 2

    # ── Fixed interface ───────────────────────────────────────────────────────

    async def solve(self, question: str, label: str | None = None) -> GenerationOutput:
        meta     = self._meta
        category = meta.get("category", "qa")

        if category == "func_call":
            return await self._solve_func_call(question, label)

        return await self._solve_qa(question, label)

    # ── QA path: orchestrator-driven multi-turn ──────────────────────────────

    async def _solve_qa(self, question: str, label: str | None = None) -> GenerationOutput:
        meta  = self._meta
        tools = meta.get("tools", [])
        messages, tools = self._builder.build_qa(problem=question, tools_field=tools)

        turns: list[dict] = []
        first_out: Optional[GenerationOutput] = None
        finish_reason = "stop"

        for turn_idx in range(self._max_turns):
            gen_out = await self._generate_with_tools(messages, tools)
            finish_reason = gen_out.finish_reason

            if first_out is None:
                first_out = gen_out

            tc = parse_tool_call(gen_out.response)
            messages.append({"role": "assistant", "content": gen_out.response})

            orch_prompt_tokens = len(gen_out.prompt_token_ids)
            orch_response_tokens = len(gen_out.token_ids)

            turns.append({
                "tokens":            list(gen_out.prompt_token_ids) + list(gen_out.token_ids),
                "response_length":   len(gen_out.token_ids),
                "loss_mask":         [1] * len(gen_out.token_ids),
                "rollout_log_probs": list(gen_out.log_probs),
                "tool_name":         tc.name if tc and not tc.error else None,
                "role_name":         (tc.arguments.get("model") or tc.arguments.get("expert")) if tc and not tc.error else None,
                "input_tokens":      0,
                "output_tokens":     0,
                "latency_ms":        0.0,
                "orch_input_tokens": orch_prompt_tokens,
                "orch_output_tokens": orch_response_tokens,
                "_response":         gen_out.response,
            })

            if tc is None or tc.error:
                break

            logger.info("[OrchestraSolver] turn=%d tool=%r", turn_idx, tc.name)

            tool_result, done, final_answer, tool_stats = await self._execute_tool(tc, question)
            turns[-1]["input_tokens"]  = tool_stats["input_tokens"]
            turns[-1]["output_tokens"] = tool_stats["output_tokens"]
            turns[-1]["latency_ms"]    = tool_stats["latency_ms"]
            turns[-1]["_tool_result"]  = final_answer if done else (tool_result or "")

            if done:
                return self._build_output(turns, first_out, finish_reason, final_answer or "")

            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            messages = self._builder.append_tool_result(
                messages, tool_call_id, tc.name, tool_result or ""
            )

        return self._build_output(turns, first_out, finish_reason, final_answer="")

    # ── func_call path: tau2-driven simulation ───────────────────────────────

    async def _solve_func_call(self, question: str, label: str | None = None) -> GenerationOutput:
        """
        func_call tasks are driven by the tau2 environment subprocess.
        tau2 writes input_N.json (messages + tools) for us to respond to.
        We generate with the orchestrator model, optionally route call_expert
        to an expert LLM, then write output_N.json back.
        """
        meta = self._meta
        domain = meta.get("domain", "")
        if not domain:
            eid = meta.get("eid", "") or meta.get("id", "")
            if "____" in eid:
                domain = eid.split("____")[0]
            else:
                domain = "mock"
        task_path = meta.get("task_path", "")
        transfer_dir = meta.get("cur_transfer_dir", "")
        user_llm = meta.get("user_llm", "qwen-turbo-latest")

        if not transfer_dir:
            import tempfile
            transfer_dir = tempfile.mkdtemp(prefix="tau2_orch_")
        else:
            import os as _os
            _os.makedirs(transfer_dir, exist_ok=True)

        import os as _os
        output_file = meta.get("output_file", "") or _os.path.join(transfer_dir, "output.json")

        if not task_path and meta.get("example"):
            task_path = _os.path.join(transfer_dir, "task.json")
            example_data = meta["example"]
            if not isinstance(example_data, list):
                example_data = [example_data]
            with open(task_path, "w") as f:
                json.dump(example_data, f, indent=2)

        # Write model_mapping + tool_pricing so tau2 subprocess can dynamically build call_expert tool
        mm = meta.get("model_mapping", {})
        if mm:
            with open(_os.path.join(transfer_dir, "model_mapping.json"), "w") as f:
                json.dump({
                    "model_mapping": mm,
                    "tool_pricing": meta.get("tool_pricing", {}),
                }, f)

        adapter = Tau2Adapter(
            domain=domain,
            task_path=task_path,
            output_file=output_file,
            transfer_dir=transfer_dir,
            user_llm=user_llm,
            agent_llm="train",
            max_steps=self._max_turns * 4,
            use_model_tool=True,
        )

        turns: list[dict] = []
        first_out: Optional[GenerationOutput] = None
        finish_reason = "stop"
        prev_msg_count = 0

        try:
            await adapter.start()

            while not adapter.is_done():
                try:
                    inp = await adapter.wait_for_input()
                except (TimeoutError, RuntimeError) as e:
                    logger.error("[OrchestraSolver] tau2 error: %s", e)
                    break

                if inp is None:
                    break

                messages = inp.get("messages", [])
                tools = inp.get("tools", [])

                if turns and prev_msg_count > 0:
                    new_msgs = messages[prev_msg_count:]
                    result_parts = []
                    for m in new_msgs:
                        role = m.get("role", "")
                        content = m.get("content", "")
                        if role in ("tool", "function"):
                            name = m.get("name", "tool")
                            result_parts.append(f"[{name}] {content}")
                        elif role == "user":
                            result_parts.append(f"[User] {_strip_thinking(content)}")
                    if result_parts:
                        turns[-1]["_tool_result"] = "\n".join(result_parts)

                prev_msg_count = len(messages)

                gen_out = await self._generate_with_tools(messages, tools)
                finish_reason = gen_out.finish_reason

                if first_out is None:
                    first_out = gen_out

                response_text = gen_out.response
                all_tcs = parse_all_tool_calls(response_text)
                tc = all_tcs[0] if all_tcs else None

                expert_name = None
                if tc and not tc.error and tc.name == "call_expert":
                    expert_name = tc.arguments.get("expert", "expert-1")

                output_tool_calls = None
                t0 = time.monotonic()
                if expert_name:
                    resp = await self._expert.execute_with_messages(
                        messages=inp.get("original_messages", messages),
                        expert=expert_name,
                        tools=inp.get("original_tools", tools),
                        max_tokens=8192,
                    )
                    elapsed = (time.monotonic() - t0) * 1000
                    output_content = resp.get("content", "")
                    output_tool_calls = resp.get("tool_calls")
                else:
                    elapsed = 0.0
                    if all_tcs:
                        output_tool_calls = [
                            {"name": t.name, "arguments": t.arguments}
                            for t in all_tcs if not t.error
                        ]
                        output_content = None
                    else:
                        output_content = _extract_message_for_tau2(response_text)
                        output_tool_calls = None

                sent_summary = ""
                if expert_name:
                    sent_summary = f"→ call_expert({expert_name})"
                    if output_tool_calls:
                        tc_names = [tc_item["name"] for tc_item in output_tool_calls]
                        sent_summary += f" → expert tool_calls: {tc_names}"
                    if output_content:
                        sent_summary += f"\n  expert reply: {output_content}"
                elif output_tool_calls:
                    tc_names = [tc_item["name"] for tc_item in output_tool_calls]
                    sent_summary = f"→ direct tool_calls: {tc_names}"
                elif output_content:
                    sent_summary = f"→ message: {output_content}"

                orch_prompt_tokens = len(gen_out.prompt_token_ids)
                orch_response_tokens = len(gen_out.token_ids)

                turns.append({
                    "tokens":            list(gen_out.prompt_token_ids) + list(gen_out.token_ids),
                    "response_length":   len(gen_out.token_ids),
                    "loss_mask":         [1] * len(gen_out.token_ids),
                    "rollout_log_probs": list(gen_out.log_probs),
                    "tool_name":         "call_expert" if expert_name else (tc.name if tc and not tc.error else None),
                    "role_name":         expert_name,
                    "input_tokens":      orch_prompt_tokens,
                    "output_tokens":     orch_response_tokens,
                    "latency_ms":        round(elapsed, 1),
                    "orch_input_tokens": orch_prompt_tokens,
                    "orch_output_tokens": orch_response_tokens,
                    "_response":         response_text,
                    "_sent_action":      sent_summary,
                })

                await adapter.write_output(output_content, output_tool_calls)

            reward_info = adapter.get_reward_info()
            if turns:
                turns[-1]["tau2_reward_info"] = reward_info

        finally:
            await adapter.cleanup()

        return self._build_output(turns, first_out, finish_reason, final_answer="")

    # ── Internal: generation (with tools injection) ───────────────────────────

    async def _generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> GenerationOutput:
        """
        Build prompt_text using tokenizer.apply_chat_template(tools=...),
        then call SGLang /generate and return a GenerationOutput (containing token_ids / log_probs).
        """
        tokenizer = self._engine.tokenizer

        sampling_params = dict(self._engine.sampling_params)
        context_limit = sampling_params.pop("context_length", 131072)
        max_new = sampling_params.get("max_new_tokens", 8192)

        tools_token_len = len(tokenizer(str(tools), add_special_tokens=False)["input_ids"]) if tools else 0
        max_prompt_tokens = context_limit - max_new - 32
        max_message_tokens = max(1024, max_prompt_tokens - tools_token_len)
        messages = self._truncate_middle_turns(tokenizer, messages, max_message_tokens)

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        available = context_limit - len(prompt_token_ids) - 32
        if available < max_new:
            sampling_params["max_new_tokens"] = max(1, available)

        payload = {
            "text": prompt_text,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        raw = await slime_post(self._engine.url, payload)

        meta         = raw["meta_info"]
        finish_reason = meta["finish_reason"]["type"]
        response      = raw["text"]
        token_ids     = [item[1] for item in meta["output_token_logprobs"]]
        log_probs     = [item[0] for item in meta["output_token_logprobs"]]

        return GenerationOutput(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            response=response,
            token_ids=token_ids,
            log_probs=log_probs,
            finish_reason=finish_reason,
        )

    # ── Internal: tool execution ──────────────────────────────────────────────

    @staticmethod
    def _truncate_middle_turns(tokenizer, messages: list[dict], max_tokens: int) -> list[dict]:
        """
        Truncate middle turns of a conversation to fit within max_tokens.
        Keeps the first message (system) and last message (latest user turn)
        intact, removes middle turns from oldest to newest.
        Mirrors original ToolOrchestra's cut_middle_turns.
        """
        if len(messages) <= 2:
            return messages

        def _msg_tokens(msg):
            content = msg.get("content", "") or ""
            return len(tokenizer(content, add_special_tokens=False)["input_ids"])

        total = sum(_msg_tokens(m) for m in messages)
        if total <= max_tokens:
            return messages

        msgs = list(messages)
        head = msgs[:1]
        tail = msgs[-1:]
        middle = msgs[1:-1]

        head_tokens = _msg_tokens(head[0])
        tail_tokens = _msg_tokens(tail[0])
        remaining = max_tokens - head_tokens - tail_tokens

        kept_middle = []
        for m in reversed(middle):
            m_tok = _msg_tokens(m)
            if remaining >= m_tok:
                kept_middle.insert(0, m)
                remaining -= m_tok
            else:
                break

        return head + kept_middle + tail

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._engine.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _make_stats(self, query_text: str, result_text: str | None, elapsed_ms: float) -> dict:
        return {
            "input_tokens":  self._count_tokens(query_text),
            "output_tokens": self._count_tokens(result_text) if result_text else 0,
            "latency_ms":    round(elapsed_ms, 1),
        }

    async def _execute_tool(
        self,
        tc,
        question: str,
    ) -> tuple[Optional[str], bool, Optional[str], dict]:
        """
        Routes to the correct tool-specific executor.
        Returns: (tool_result, done, final_answer, tool_stats)
        """
        name = tc.name

        needs_expert = name in ("answer", "search", "enhance_reasoning", "call_expert")
        if needs_expert and self._expert_consecutive_failures >= self._expert_circuit_breaker_limit:
            logger.warning(
                "[OrchestraSolver] Expert circuit breaker open (%d consecutive failures), "
                "skipping tool=%r",
                self._expert_consecutive_failures, name,
            )
            result = (
                "[Expert unavailable] The expert service is not responding after multiple attempts. "
                "Please answer based on available information."
            )
            return result, False, None, {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0}

        if name == "answer":
            res = await self._execute_answer(tc, question)
        elif name == "search":
            res = await self._execute_search(tc, question)
        elif name == "enhance_reasoning":
            res = await self._execute_reasoning(tc, question)
        elif name == "call_expert":
            res = await self._execute_call_expert(tc, question)
        else:
            result = f"[OrchestraSolver] Unknown tool: {name!r}"
            return result, False, None, {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0}

        tool_result = res[0]
        if tool_result and "[ExpertCallerTool Error]" in str(tool_result):
            self._expert_consecutive_failures += 1
        else:
            self._expert_consecutive_failures = 0

        return res

    # ── answer: ask the expert to produce a structured answer ─────────────────

    async def _execute_answer(
        self, tc, question: str,
    ) -> tuple[Optional[str], bool, Optional[str], dict]:
        role_name = tc.arguments.get("model") or tc.arguments.get("expert") or ""
        context = "\n\n".join(self._context_parts)
        query = f"{question}\n\n{context}".strip() if context else question
        query += (
            "\n\nBased on all the information above, provide your final answer. "
            "Wrap the final answer within <answer> and </answer>."
        )

        t0 = time.monotonic()
        result = await self._expert.execute(
            query=query,
            expert=role_name,
            system_message="You are a helpful expert. Provide a clear, accurate answer.",
            max_tokens=8192,
        )
        elapsed = (time.monotonic() - t0) * 1000

        answer_text = self._extract_answer_tag(result) if result else ""
        return None, True, answer_text or result, self._make_stats(query, result, elapsed)

    @staticmethod
    def _extract_answer_tag(text: str) -> str:
        """Extract final answer from expert response.

        Tries multiple formats in order:
        1. <answer>...</answer> tags
        2. \\boxed{...} (LaTeX, possibly nested braces)
        3. **Final Answer** line patterns
        """
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        if r"\boxed{" in cleaned:
            parts = cleaned.split(r"\boxed{")
            last_boxed = parts[-1]
            depth, end = 1, 0
            for i, ch in enumerate(last_boxed):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            extracted = last_boxed[:end].strip()
            if extracted:
                return extracted

        for pattern in [
            r"\*\*Final Answer\*\*[:\s]*(.+?)(?:\n|$)",
            r"(?:final answer|the answer) (?:is|=)[:\s]*(.+?)(?:\.|,|\n|$)",
        ]:
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                ans = m.group(1).strip().strip("*").strip()
                if ans:
                    return ans

        return ""

    # ── search: two-phase — expert writes query first, then retrieval is called ─

    async def _execute_search(
        self, tc, question: str,
    ) -> tuple[Optional[str], bool, Optional[str], dict]:
        role_name = tc.arguments.get("model") or tc.arguments.get("expert") or ""
        context = "\n\n".join(self._context_parts)
        query_gen_prompt = f"{context}\n\n" if context else ""
        query_gen_prompt += (
            f"Question: {question}\n"
            "Instead of directly answering the question, please write a search query "
            "that would help find the relevant information. "
            "Wrap the query within <query> and </query>."
        )

        t0 = time.monotonic()
        query_response = await self._expert.execute(
            query=query_gen_prompt,
            expert=role_name,
            system_message="You are a search query writer. Generate concise, targeted search queries.",
            max_tokens=8192,
        )
        elapsed_query = (time.monotonic() - t0) * 1000

        search_query = self._extract_query_tag(query_response, fallback=question[:200])

        faiss_docs = await self._search.execute(search_query)
        if faiss_docs and not faiss_docs.startswith("[SearchRetrievalTool Error]"):
            self._context_parts.append(f"[Search by {role_name}]\n{faiss_docs}")
            result_text = faiss_docs
        else:
            self._context_parts.append(f"[Search by {role_name}] No results found.")
            result_text = "No relevant documents found."

        total_query_text = query_gen_prompt
        return result_text, False, None, self._make_stats(total_query_text, result_text, elapsed_query)

    @staticmethod
    def _extract_query_tag(text: str, fallback: str = "") -> str:
        if not text:
            return fallback
        m = re.search(r"<query>(.*?)</query>", text, re.DOTALL)
        if m:
            q = m.group(1).strip()
            if len(q) >= 3:
                return q
        return fallback

    # ── enhance_reasoning: deep reasoning with a dedicated system prompt ────────

    async def _execute_reasoning(
        self, tc, question: str,
    ) -> tuple[Optional[str], bool, Optional[str], dict]:
        role_name = tc.arguments.get("model") or tc.arguments.get("expert") or ""
        context = "\n\n".join(self._context_parts)
        query = f"{question}\n\n{context}".strip() if context else question
        query += "\n\nPlease think through this step by step, showing your reasoning process in detail."

        t0 = time.monotonic()
        result = await self._expert.execute(
            query=query,
            expert=role_name,
            system_message=(
                "You are an expert reasoning assistant. Break down complex problems "
                "step by step. Show your thought process clearly."
            ),
            max_tokens=8192,
        )
        elapsed = (time.monotonic() - t0) * 1000
        self._context_parts.append(f"[Reasoning by {role_name}]\n{result}")
        return result, False, None, self._make_stats(query, result, elapsed)

    # ── call_expert: func_call task — pass full conversation + tools for native tool calling ──

    async def _execute_call_expert(
        self, tc, question: str,
    ) -> tuple[Optional[str], bool, Optional[str], dict]:
        expert = tc.arguments.get("expert", "expert-1")
        category = self._meta.get("category", "qa")

        if category == "func_call":
            return await self._execute_call_expert_func_call(tc, expert)

        query = tc.arguments.get("query", question)
        t0 = time.monotonic()
        result = await self._expert.execute(query=query, expert=expert)
        elapsed = (time.monotonic() - t0) * 1000
        return result, False, None, self._make_stats(query, result, elapsed)

    async def _execute_call_expert_func_call(
        self, tc, expert: str,
    ) -> tuple[Optional[str], bool, Optional[str], dict]:
        """
        func_call: pass full conversation history + tools schema to the expert
        for native function calling (aligned with original ToolOrchestra).
        """
        input_messages = self._meta.get("input_messages") or self._meta.get("messages", [])
        input_tools = self._meta.get("input_tools") or self._meta.get("tools", [])

        if not input_messages:
            query = tc.arguments.get("query", "")
            t0 = time.monotonic()
            result = await self._expert.execute(query=query, expert=expert)
            elapsed = (time.monotonic() - t0) * 1000
            return result, False, None, self._make_stats(query, result, elapsed)

        query_text = json.dumps(input_messages[-1] if input_messages else {})

        t0 = time.monotonic()
        resp = await self._expert.execute_with_messages(
            messages=input_messages,
            expert=expert,
            tools=input_tools if input_tools else None,
            max_tokens=8192,
        )
        elapsed = (time.monotonic() - t0) * 1000

        if resp.get("tool_calls"):
            result_text = json.dumps(resp["tool_calls"], ensure_ascii=False)
        else:
            result_text = resp.get("content", "")

        return result_text, False, None, self._make_stats(query_text, result_text, elapsed)

    # ── Internal: assemble GenerationOutput ──────────────────────────────────

    @staticmethod
    def _build_output(
        turns: list[dict],
        first_out: Optional[GenerationOutput],
        finish_reason: str,
        final_answer: str,
    ) -> GenerationOutput:
        """
        Same turns concatenation logic as agentflow.Solver:
          turn[0]: only response tokens are concatenated (prompt stored separately in prompt_token_ids)
          turn[1+]: full tokens concatenated (prompt portion loss_mask=0, response portion loss_mask=1)
        """
        if not turns or first_out is None:
            return GenerationOutput(
                prompt_text="", prompt_token_ids=[], response="",
                token_ids=[], log_probs=[], finish_reason=finish_reason,
                loss_mask=[], final_output=final_answer, turns=turns,
            )

        first = turns[0]
        first_prompt_len = len(first["tokens"]) - first["response_length"]
        prompt_token_ids = first["tokens"][:first_prompt_len]

        cat_token_ids = list(first["tokens"][first_prompt_len:])
        cat_loss_mask = list(first["loss_mask"])
        cat_log_probs = list(first["rollout_log_probs"])

        for t in turns[1:]:
            t_prompt_len = len(t["tokens"]) - t["response_length"]
            cat_token_ids += t["tokens"]
            cat_loss_mask += [0] * t_prompt_len + t["loss_mask"]
            cat_log_probs += [0.0] * t_prompt_len + t["rollout_log_probs"]

        parts = []
        for i, t_dict in enumerate(turns):
            orch_out = t_dict.get("_response", "")
            tool_res = t_dict.get("_tool_result", "")
            sent_action = t_dict.get("_sent_action", "")

            orch_display = _strip_thinking(orch_out)
            parts.append(f"=== Turn {i} [Orchestrator] ===\n{orch_display}")

            if sent_action:
                parts.append(f"  {sent_action}")

            if tool_res:
                tool_name = t_dict.get("tool_name", "tool")
                role_name = t_dict.get("role_name", "")
                label = f"{tool_name}({role_name})" if role_name else tool_name
                parts.append(f"--- Tool/Env Result [{label}] ---\n{tool_res}")
        if final_answer:
            parts.append(f"=== Final Answer ===\n{final_answer}")

        reward_info = None
        for t_dict in reversed(turns):
            ri = t_dict.get("tau2_reward_info")
            if ri is not None:
                reward_info = ri
                break
        if reward_info is not None:
            parts.append(f"=== Reward ===\n{json.dumps(reward_info, ensure_ascii=False)}")

        response_text = "\n\n".join(parts)

        return GenerationOutput(
            prompt_text=first_out.prompt_text,
            prompt_token_ids=prompt_token_ids,
            response=response_text or first_out.response,
            token_ids=cat_token_ids,
            log_probs=cat_log_probs,
            finish_reason=finish_reason,
            loss_mask=cat_loss_mask,
            final_output=final_answer,
            turns=turns,
        )

    # ── Internal: func_call initial conversation ──────────────────────────────

    @staticmethod
    def _build_func_call_conv(meta: dict) -> list[dict]:
        try:
            known_info = meta["example"]["user_scenario"]["instructions"]["known_info"]
        except (KeyError, TypeError):
            known_info = meta.get("problem", "")
        return [
            {"role": "system", "content": "You are an orchestrator. You must ALWAYS delegate the user request by calling the call_expert tool. Never answer directly without calling call_expert first."},
            {"role": "user",   "content": known_info},
        ]
