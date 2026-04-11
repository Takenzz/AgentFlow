from dataclasses import dataclass, field
from typing import Any

from slime.utils.http_utils import post


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


