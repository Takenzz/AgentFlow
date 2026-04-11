"""
Expert Caller Tool
------------------
Routes requests to specialized expert models via OpenAI-compatible APIs.

Supports three calling modes:
1. Simple query (str) -> user message
2. Structured messages (list[dict]) -> full conversation history
3. Native tool calling -> pass tools schema for func_call tasks

model_mapping example:
    {
        "expert-1":      "Qwen/Qwen3-32B",
        "expert-2":      "Qwen/Qwen3-30B-A3B",
        "expert-3":      "Qwen/Qwen3-14B",
        "answer-1":      "Qwen/Qwen3-32B",
        "answer-2":      "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "answer-3":      "Qwen/Qwen3-30B-A3B",
        "answer-math-1": "Qwen/Qwen2.5-Math-72B-Instruct",
        "answer-math-2": "Qwen/Qwen2.5-Math-7B-Instruct",
        "reasoner-1":    "Qwen/Qwen3-32B",
        "reasoner-2":    "Qwen/Qwen2.5-Coder-32B-Instruct",
        "reasoner-3":    "Qwen/Qwen3-14B",
        "search-1":      "Qwen/Qwen3-32B",
    }

expert_engine_map example (model_name -> OpenAI-compatible base_url):
    {
        "Qwen/Qwen3-32B":                           "http://127.0.0.1:30001/v1",
        "Qwen/Qwen2.5-Coder-32B-Instruct":          "http://127.0.0.1:30002/v1",
        "Qwen/Qwen2.5-Math-7B-Instruct":            "http://127.0.0.1:30003/v1",
        "Qwen/Qwen2.5-Math-72B-Instruct":           "http://127.0.0.1:30004/v1",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "http://127.0.0.1:30005/v1",
        "Qwen/Qwen3-30B-A3B":                       "http://127.0.0.1:30006/v1",
        "Qwen/Qwen3-14B":                           "http://127.0.0.1:30007/v1",
    }
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

# expert_engine_map: model_name -> {"url": str, "context_length": int}
# context_length must match --context-length used when launching the expert SGLang server.
# GPU assignment (see launch.sh, B300 × 8 GPUs):
#   GPU 0  → Qwen3-32B-FP8                port 30001  (shared with retrieval service and Math-7B)
#   GPU 0  → Qwen2.5-Math-7B              port 30003  (shared with retrieval service and FP8)
#   GPU 1  → DeepSeek-R1-Distill-Qwen-32B port 30005  (mem ~0.45)
#   GPU 1  → Qwen3-30B-A3B                port 30006  (mem ~0.45, shared with distill)
#   GPU 2  → Qwen2.5-Coder-32B            port 30002  (mem ~0.45)
#   GPU 2  → Qwen3-14B                    port 30007  (mem ~0.45, shared with coder)
#   GPU 3  → Qwen2.5-Math-72B             port 30004
EXPERT_ENGINE_MAP: dict[str, dict] = {
    "Qwen/Qwen3-32B":                           {"url": "http://127.0.0.1:30001/v1", "context_length": 163840},
    "Qwen/Qwen2.5-Coder-32B-Instruct":          {"url": "http://127.0.0.1:30002/v1", "context_length": 163840},
    "Qwen/Qwen2.5-Math-7B-Instruct":            {"url": "http://127.0.0.1:30003/v1", "context_length": 163840},
    "Qwen/Qwen2.5-Math-72B-Instruct":           {"url": "http://127.0.0.1:30004/v1", "context_length": 163840},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"url": "http://127.0.0.1:30005/v1", "context_length": 163840},
    "Qwen/Qwen3-30B-A3B":                       {"url": "http://127.0.0.1:30006/v1", "context_length": 163840},
    "Qwen/Qwen3-14B":                           {"url": "http://127.0.0.1:30007/v1", "context_length": 163840},
}

TOOL_NAME = "Expert_Caller_Tool"

TOOL_DESCRIPTION = """
Call a specialized expert model to help solve a problem.
Available experts are defined by the current task's model_mapping.
Use this tool when you need specialized reasoning (math, coding, general).
"""


class ExpertCallerTool:
    """
    Routes to expert models. Instantiated once per sample with that
    sample's model_mapping.
    """

    def __init__(
        self,
        model_mapping: dict[str, str],
        expert_engine_map: dict[str, Any],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        max_retries: int = 1,
        context_length: int = 163840,
    ):
        self.model_mapping = model_mapping
        self.expert_engine_map = expert_engine_map
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.context_length = context_length

    @staticmethod
    def _estimate_tokens(msg: dict) -> int:
        """Rough token estimate: count chars in content/tool results, divide by 3."""
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        return max(1, len(str(content)) // 3)

    def _truncate_messages(
        self, messages: list[dict], effective_max: int, context_length: int
    ) -> list[dict]:
        """
        Truncate messages to fit within context_length - effective_max - 256 tokens.
        Strategy: always keep first (system) message and last message;
        drop middle messages from oldest to newest until budget is met.
        """
        budget = context_length - effective_max - 256
        if budget <= 0:
            return messages[-1:]

        total = sum(self._estimate_tokens(m) for m in messages)
        if total <= budget:
            return messages

        if len(messages) <= 2:
            # Can't remove anything meaningful; return as-is and let the API error
            logger.warning(
                "[ExpertCallerTool] messages still too long after truncation attempt "
                "(%d est. tokens > budget %d)", total, budget,
            )
            return messages

        head = messages[:1]
        tail = messages[-1:]
        middle = list(messages[1:-1])

        head_tokens = sum(self._estimate_tokens(m) for m in head)
        tail_tokens = sum(self._estimate_tokens(m) for m in tail)
        remaining = budget - head_tokens - tail_tokens

        kept: list[dict] = []
        for m in reversed(middle):
            t = self._estimate_tokens(m)
            if remaining >= t:
                kept.insert(0, m)
                remaining -= t
            else:
                break

        dropped = len(middle) - len(kept)
        if dropped:
            logger.warning(
                "[ExpertCallerTool] truncated %d middle messages to fit context "
                "(budget=%d est. tokens)", dropped, budget,
            )
        return head + kept + tail

    def _get_model_and_url(self, expert: str) -> tuple[str, str, int]:
        model_name = self.model_mapping.get(expert)
        if model_name is None:
            raise ValueError(
                f"Expert role '{expert}' not found in model_mapping. "
                f"Available roles: {list(self.model_mapping.keys())}"
            )
        entry = self.expert_engine_map.get(model_name)
        if entry is None:
            raise ValueError(
                f"Model '{model_name}' not found in expert_engine_map. "
                f"Available models: {list(self.expert_engine_map.keys())}"
            )
        # Support both {"url": ..., "context_length": ...} and plain str formats
        if isinstance(entry, dict):
            base_url = entry["url"]
            context_length = entry.get("context_length", self.context_length)
        else:
            base_url = entry
            context_length = self.context_length
        return model_name, base_url, context_length

    async def execute(
        self,
        query: str,
        expert: str = "expert-1",
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Simple mode: send a text query to the expert.

        Args:
            query:          The question/context to send
            expert:         Expert role key in model_mapping
            system_message: Optional system prompt to prepend
            max_tokens:     Override default max_tokens for this call
        """
        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})

        return await self._call(
            expert=expert,
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
        )

    async def execute_with_messages(
        self,
        messages: list[dict],
        expert: str = "expert-1",
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """
        Structured mode: send full conversation history, optionally with
        tools schema for native tool calling (func_call tasks).

        Args:
            messages:   Full conversation history (system/user/assistant/tool)
            expert:     Expert role key in model_mapping
            tools:      OpenAI-format tools schema for native function calling
            max_tokens: Override default max_tokens

        Returns:
            dict with keys:
                content:    text response (may be None if tool_calls present)
                tool_calls: list of tool call dicts (may be None)
                raw:        the full ChatCompletionMessage object
        """
        try:
            model_name, base_url, context_length = self._get_model_and_url(expert)
        except ValueError as e:
            return {"content": f"[ExpertCallerTool Error] {e}", "tool_calls": None, "raw": None}

        from openai import (
            AsyncOpenAI,
            APIConnectionError,
            APITimeoutError,
            BadRequestError,
            InternalServerError,
            RateLimitError,
        )

        client = AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=180.0)
        effective_max = max_tokens or self.max_tokens
        messages = self._truncate_messages(messages, effective_max, context_length)

        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": effective_max,
                    "temperature": self.temperature,
                }
                if tools:
                    kwargs["tools"] = tools

                completion = await client.chat.completions.create(**kwargs)
                msg = completion.choices[0].message

                tool_calls = None
                if msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        try:
                            args = _json.loads(tc.function.arguments)
                        except (TypeError, _json.JSONDecodeError):
                            args = tc.function.arguments
                        tool_calls.append({
                            "name": tc.function.name,
                            "arguments": args,
                        })

                return {
                    "content": msg.content or "",
                    "tool_calls": tool_calls,
                    "raw": msg,
                }

            except BadRequestError as e:
                logger.error(
                    "[ExpertCallerTool] expert=%s model=%s bad request (not retrying): %s",
                    expert, model_name, e,
                )
                return {
                    "content": f"[ExpertCallerTool Error] {type(e).__name__}: {e}",
                    "tool_calls": None,
                    "raw": None,
                }
            except (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError) as e:
                if attempt + 1 >= self.max_retries:
                    logger.error(
                        "[ExpertCallerTool] expert=%s model=%s failed after %d retries: %s",
                        expert, model_name, self.max_retries, e,
                    )
                    return {
                        "content": f"[ExpertCallerTool Error] {type(e).__name__}: {e}",
                        "tool_calls": None,
                        "raw": None,
                    }
                backoff = min(2 ** attempt, 60) + random.random()
                logger.warning(
                    "[ExpertCallerTool] expert=%s retry %d/%d in %.1fs: %s",
                    expert, attempt + 1, self.max_retries, backoff, e,
                )
                await asyncio.sleep(backoff)

        return {"content": "[ExpertCallerTool Error] All retries exhausted.", "tool_calls": None, "raw": None}

    async def _call(
        self,
        expert: str,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        """Low-level: send messages and return text content."""
        try:
            model_name, base_url, context_length = self._get_model_and_url(expert)
        except ValueError as e:
            return f"[ExpertCallerTool Error] {e}"

        from openai import (
            AsyncOpenAI,
            APIConnectionError,
            APITimeoutError,
            BadRequestError,
            InternalServerError,
            RateLimitError,
        )

        client = AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=180.0)

        for attempt in range(self.max_retries):
            try:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
                response = completion.choices[0].message.content or ""
                logger.debug(
                    "[ExpertCallerTool] expert=%s model=%s response_len=%d",
                    expert, model_name, len(response),
                )
                return response

            except BadRequestError as e:
                logger.error(
                    "[ExpertCallerTool] expert=%s model=%s bad request (not retrying): %s",
                    expert, model_name, e,
                )
                return f"[ExpertCallerTool Error] {type(e).__name__}: {e}"
            except (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError) as e:
                if attempt + 1 >= self.max_retries:
                    logger.error(
                        "[ExpertCallerTool] expert=%s model=%s failed after %d retries: %s",
                        expert, model_name, self.max_retries, e,
                    )
                    return f"[ExpertCallerTool Error] {type(e).__name__}: {e}"
                backoff = min(2 ** attempt, 60) + random.random()
                logger.warning(
                    "[ExpertCallerTool] expert=%s retry %d/%d in %.1fs: %s",
                    expert, attempt + 1, self.max_retries, backoff, e,
                )
                await asyncio.sleep(backoff)

        return "[ExpertCallerTool Error] All retries exhausted."
