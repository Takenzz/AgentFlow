from dataclasses import dataclass, field
import asyncio
import logging
import random
from typing import Any

from slime.utils.http_utils import post
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    prompt_text: str
    prompt_token_ids: list[int]
    response: str
    token_ids: list[int]
    log_probs: list[float]
    finish_reason: str  # "stop" | "length" | "abort"
    # Filled by the solver in multi-turn mode: same length as token_ids;
    # 1 = model-generated tokens that participate in loss, 0 = injected prompt tokens that do not
    loss_mask: list[int] | None = None
    # The planner's final consolidated answer, used for reward scoring
    final_output: str | None = None
    # Multi-turn split: each turn is an independent training sequence;
    # custom_convert uses this field to unroll turns
    turns: list[dict] | None = None


class SGLangEngine:
    """
    Lightweight wrapper around the SGLang HTTP /generate endpoint.
    All modules (Planner, Solver, etc.) share the same instance to issue LLM calls uniformly.
    """

    def __init__(self, url: str, tokenizer: Any, sampling_params: dict, max_new_tokens: int | None = None, enable_thinking: bool = False):
        self.url = url
        self.tokenizer = tokenizer
        self.sampling_params = dict(sampling_params)
        self.enable_thinking = enable_thinking
        if max_new_tokens is not None:
            self.sampling_params["max_new_tokens"] = max_new_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict | None = None,
    ) -> GenerationOutput:
        """
        Accept standard chat messages and return a GenerationOutput.
        If sampling_params is provided, it overrides the defaults set at initialization.
        """
        params = sampling_params if sampling_params is not None else self.sampling_params

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        payload = {
            "text": prompt_text,
            "sampling_params": params,
            "return_logprob": True,
        }
        out = await post(self.url, payload)

        meta = out["meta_info"]
        finish_reason = meta["finish_reason"]["type"]
        response = out["text"]
        token_ids = [item[1] for item in meta["output_token_logprobs"]]
        log_probs  = [item[0] for item in meta["output_token_logprobs"]]

        return GenerationOutput(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            response=response,
            token_ids=token_ids,
            log_probs=log_probs,
            finish_reason=finish_reason,
        )


class APIEngine(SGLangEngine):
    """使用 OpenAI 兼容 Chat Completions API 模拟 SGLang 推理引擎。

    用于评测或非训练链路：调用第三方 / 自建的 OpenAI 兼容 API，拿到 response 文本即可。
    由于 API 不返回逐 token 的 logprob，也与本地 tokenizer 的 token 边界不保证一致，
    因此其输出 **不参与训练 loss**：
        - prompt_text / prompt_token_ids: 仍然用本地 tokenizer 渲染，保持与 SGLangEngine 一致，
          方便上层 Solver / 日志统一处理；
        - response: 从 API 返回的 message.content 取得；
        - token_ids / log_probs: 置为空列表（不参与 loss）；
        - finish_reason: 透传 API 返回的 finish_reason（若无则为空字符串）。
    """

    def __init__(
        self,
        url: str,
        tokenizer: Any,
        sampling_params: dict,
        max_new_tokens: int | None = None,
        enable_thinking: bool = False,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 20.0,
        require_non_empty: bool = True,
    ):
        super().__init__(url, tokenizer, sampling_params, max_new_tokens, enable_thinking)
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.require_non_empty = require_non_empty
        # 复用项目其它模块（eval_orchestra / expert_caller 等）的调用风格：
        # 用 AsyncOpenAI 作为底层异步 HTTP 客户端，支持 OpenAI 兼容的第三方 / 自建服务。
        self.client = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=url,
            timeout=timeout,
            max_retries=0,
        )

    def _build_chat_kwargs(self, messages: list[dict[str, str]], params: dict[str, Any]) -> dict[str, Any]:
        if not self.model_name:
            raise ValueError("APIEngine requires model_name for OpenAI-compatible chat completions.")

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if "temperature" in params:
            kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            kwargs["top_p"] = params["top_p"]
        if "max_new_tokens" in params:
            kwargs["max_tokens"] = params["max_new_tokens"]
        elif "max_tokens" in params:
            kwargs["max_tokens"] = params["max_tokens"]
        if "stop" in params:
            kwargs["stop"] = params["stop"]
        return kwargs

    async def _create_completion_with_retry(self, kwargs: dict[str, Any]):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = await self.client.chat.completions.create(**kwargs)
                if not completion.choices:
                    raise RuntimeError("API response contains no choices.")
                choice = completion.choices[0]
                content = (choice.message.content or "") if choice.message is not None else ""
                if self.require_non_empty and not content.strip():
                    raise RuntimeError("API response content is empty.")
                return completion
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                wait = min(self.retry_max_wait, self.retry_min_wait * (2 ** attempt))
                wait = wait * (0.75 + random.random() * 0.5)
                logger.warning(
                    "API completion failed (model=%s, attempt=%d/%d): %s; retrying in %.1fs",
                    self.model_name,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(
            f"API completion failed after {self.max_retries + 1} attempts "
            f"(model={self.model_name}, base_url={self.url}): {last_exc}"
        ) from last_exc

    async def generate(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict | None = None,
    ) -> GenerationOutput:
        """调用 OpenAI 兼容 /chat/completions 接口，返回与 SGLangEngine 同构的 GenerationOutput。

        只有 prompt_text / prompt_token_ids / response 三个字段有效；
        token_ids / log_probs 置空，finish_reason 透传 API 返回值。
        """
        params = dict(sampling_params if sampling_params is not None else self.sampling_params)

        # 1) 仍然在本地用 tokenizer 渲染 prompt_text / prompt_token_ids，保持和 SGLangEngine 一致。
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # 2) 把 SGLang 风格的 sampling_params 映射到 OpenAI chat completions 字段。
        kwargs = self._build_chat_kwargs(messages, params)

        # 3) 用 AsyncOpenAI 异步发起请求；失败时做 bounded retry，不阻塞事件循环。
        completion = await self._create_completion_with_retry(kwargs)

        choice = completion.choices[0]
        response = (choice.message.content or "") if choice.message is not None else ""
        finish_reason = choice.finish_reason or ""

        return GenerationOutput(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            response=response,
            token_ids=[],
            log_probs=[],
            finish_reason=finish_reason,
        )
