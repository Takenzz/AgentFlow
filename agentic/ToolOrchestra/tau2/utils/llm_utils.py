# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------------

import json
import re
import os
import copy
import time
from typing import Any, Optional
import pickle
import uuid
from loguru import logger

try:
    import litellm
    from litellm import completion, completion_cost
    from litellm.caching.caching import Cache
    from litellm.main import ModelResponse, Usage
    _HAS_LITELLM = True
except ImportError:
    litellm = None
    completion = None
    completion_cost = None
    Cache = None
    ModelResponse = None
    Usage = None
    _HAS_LITELLM = False

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
import random
from tau2.environment.tool import Tool
try:
    from LLM_CALL import get_llm_response
except ImportError:
    try:
        import sys as _sys
        _repo = os.getenv("REPO_PATH")
        if _repo:
            _sys.path.append(_repo)
        from LLM_CALL import get_llm_response
    except (ImportError, TypeError):
        def get_llm_response(*args, **kwargs):
            raise RuntimeError(
                "get_llm_response is not available. "
                "Ensure LLM_CALL.py is on PYTHONPATH or set REPO_PATH."
            )

# litellm._turn_on_debug()

TOOL_PRICING = {
    "gpt-5": {
        "input_tokens_per_million": 1.25/1000000,
        "output_tokens_per_million": 10/1000000
    },
    "gpt-5-mini": {
        "input_tokens_per_million": 0.25/1000000,
        "output_tokens_per_million": 2/1000000
    },
    "Qwen/Qwen3-32B": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen2.5-Math-72B-Instruct": {
        "input_tokens_per_million": 0.9/1000000,
        "output_tokens_per_million": 0.9/1000000
    },
    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "input_tokens_per_million": 0.9/1000000,
        "output_tokens_per_million": 0.9/1000000
    },
    "Qwen/Qwen3-8B": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
    },
    "code_interpreter_per_second": 0.0000083,
    "tavily": {
        "search": 0.01,
        "extract": 0.002
    },
}

MODEL_TYPE = "Qwen/Qwen3-8B"

POLICY_STRINGS = [
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """You should escalate to a human agent if and only if the request cannot be handled within the scope of your actions. To escalate, use the tool call transfer_to_human_agents

You should try your best to resolve the issue before escalating the user to a human agent.
""",
    """You should try your best to resolve the issue for the user before transferring the user to a human agent.""",
    """Make sure you try all the possible ways to resolve the user's issue before transferring to a human agent.
""",
    """Make sure you try all the relevant resolution steps before transferring the user to a human agent.
""",
    """Transfer to human agent
- Transfer to a human agent only if:
  - the user explicitly asks for a human agent, or
  - the request cannot be handled within this policy and available tools (for example, authentication cannot be completed because the user cannot provide an email).
- To transfer: first call transfer_to_human_agents with a concise summary of the user’s issue, then send the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.""",
    """Transfer to human agents
- Transfer only if:
  - the user explicitly asks for a human agent, or
  - the request cannot be handled with the available policy and tools.
- To transfer: first call transfer_to_human_agents with a concise summary; then send the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.
""",
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions, or if the user explicitly asks for a human agent. To transfer, first make a tool call to transfer_to_human_agents with a brief summary, and then send the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.
""",
    """- If the user does not know their booking_id, the agent cannot locate it via tools and should transfer to a human agent.
""",
    """- If the user does not know their booking_id, the agent cannot locate it via tools and should transfer to a human agent.
""",
    """- If the show is completed, do not cancel; transfer to a human agent.
""",
    """## Compensation

- The tools do not support compensation or ex gratia certificates. If the user requests compensation, transfer to a human agent.
""",
    """- If a request cannot be fulfilled with available tools (e.g., locating a booking without booking_id, changing showtime, applying promotions), transfer to a human agent following the transfer procedure.""",
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """## Transfer to Human Agent

Transfer the user to a human agent if:
- The request cannot be handled within the scope of your actions or tools.
- The user explicitly asks for a human agent.

To transfer, first make a tool call to transfer_to_human_agents with a brief summary, and then send: 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'""",
    """Transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """## Human agent transfer

Transfer only if:
- The user explicitly asks for a human agent, or
- The request cannot be handled within the scope of these tools and policies.

To transfer:
- First, call the tool to transfer_to_human_agents with a concise summary.
- Then send: 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'""",
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. If a transfer is needed, inform the user with the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.
""",
    """Out-of-scope examples (transfer to a human agent if requested):
- Changes to student profile data (e.g., editing name, program, or address) beyond what tools support.
- Advisor assignments or financial aid adjustments beyond what tools support.
- Any requests requiring procedures or systems not represented by the available tools.""",
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """- If the requested change is not supported by available tools, transfer to a human agent.
""",
    """- If the policy is unclear or the user’s situation is not covered, transfer to a human agent.
""",
    """- If the user requests actions not supported (e.g., changing package on an existing booking, changing the number of travelers), or if package policies are unclear, transfer to a human agent:
  - First call transfer_to_human_agents with a concise summary.
  - Then send: 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'""",
  """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """- If a request is outside the scope of available tools or violates this policy, deny the request or transfer to a human agent if necessary (using the transfer procedure above).""",
    """- If a request cannot be handled with the available tools and policies, transfer to a human agent:
  - First call transfer_to_human_agents with a concise summary.
  - Then send: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.""",
]

EXPERT_POLICT = """You should transfer the user to an expert if you are not confident about how to reply or which tool to use. To transfer, first make a tool call to call_expert, and then choose an expert based on the task difficulty, i.e. strong expert on tricky task, and weaker expert on simpler task. Think carefully about whether you are confident to meet user expectation, or call an appropriate expert based on both expert cost and performance."""

if _HAS_LITELLM:
    if USE_LANGFUSE:
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]

    litellm.drop_params = True

    if LLM_CACHE_ENABLED:
        if DEFAULT_LLM_CACHE_TYPE == "redis":
            logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
            litellm.cache = Cache(
                type=DEFAULT_LLM_CACHE_TYPE,
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
                ttl=REDIS_CACHE_TTL,
            )
        elif DEFAULT_LLM_CACHE_TYPE == "local":
            logger.info("LiteLLM: Using local cache")
            litellm.cache = Cache(
                type="local",
                ttl=REDIS_CACHE_TTL,
            )
        else:
            raise ValueError(
                f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
            )
        litellm.enable_cache()
    else:
        litellm.disable_cache()

from datetime import datetime
import string
def generate_random_string(length):
    """Generates a random string of specified length using alphanumeric characters."""
    characters = string.ascii_letters + string.digits + '~!@#$%^&*()-=_+[]'
    return ''.join(random.choice(characters) for _ in range(length))


ALLOW_SONNET_THINKING = False

# if not ALLOW_SONNET_THINKING:
#     logger.warning("Sonnet thinking is disabled")


# Qualitative capability descriptions keyed by model short-name (after last "/").
# Used by _build_extra_tool to enrich the call_expert parameter description.
_MODEL_QUAL_DESC: dict[str, str] = {
    "Qwen3-32B":                        "strong general reasoning and function-calling; excellent across most domains",
    "Qwen3-32B-FP8":                    "strong general reasoning and function-calling; excellent across most domains",
    "Qwen3-30B-A3B":                    "strong function-calling ability; cost-efficient MoE model",
    "Qwen3-14B":                        "efficient function-calling; lightweight and fast",
    "Qwen2.5-Coder-32B-Instruct":       "code-focused reasoning; best for programming and algorithmic problems",
    "DeepSeek-R1-Distill-Qwen-32B":     "deep chain-of-thought reasoning; best for complex multi-step problems",
    "Qwen2.5-Math-72B-Instruct":        "heavy mathematics; for advanced math problems",
    "Qwen2.5-Math-7B-Instruct":         "light mathematics; for simple arithmetic and basic math",
}


def _build_extra_tool(model_mapping: dict, tool_pricing: dict = None) -> dict:
    """Dynamically build the call_expert tool based on the actual model_mapping and pricing."""
    experts = sorted(model_mapping.keys()) if model_mapping else ["expert-1"]
    choices_str = str(experts)
    tool_pricing = tool_pricing or {}

    desc_parts = []
    pricing_rows = []
    for e in experts:
        model_name = model_mapping.get(e, "unknown")
        short = model_name.rsplit("/", 1)[-1] if "/" in model_name else model_name
        qual = _MODEL_QUAL_DESC.get(short, "")
        if qual:
            desc_parts.append(f"{e} ({short}): {qual}")
        else:
            desc_parts.append(f"{e} ({short})")
        # Build pricing row if available
        price_info = tool_pricing.get(model_name, {})
        if price_info:
            inp = price_info.get("input_tokens_per_million", 0)
            out = price_info.get("output_tokens_per_million", 0)
            pricing_rows.append(f"{e} | ${inp*1e6:.2f} | ${out*1e6:.2f}")

    expert_desc = "; ".join(desc_parts)
    full_desc = f"The expert to call. Choices: {choices_str}. {expert_desc}."
    if pricing_rows:
        header = "Model | price per million input tokens | price per million output tokens"
        table = "\n".join([header] + pricing_rows)
        full_desc += f" The table below shows the pricing of each expert:\n{table}"
    full_desc += "\nThink carefully about whether you are confident to meet user expectation, or call an appropriate expert based on both expert cost and performance."

    return {
        "type": "function",
        "function": {
            "name": "call_expert",
            "description": "Call the expert such that the user request can be better solved compared to existing functions",
            "parameters": {
                "properties": {
                    "expert": {
                        "description": full_desc,
                        "title": "Expression",
                        "type": "string"
                    }
                },
                "required": ["expert"],
                "title": "parameters",
                "type": "object"
            }
        }
    }


def _load_model_mapping(transfer_dir: str) -> tuple[dict, dict]:
    """Load model_mapping.json from transfer_dir if available. Returns (model_mapping, tool_pricing)."""
    if not transfer_dir:
        return {}, {}
    mm_path = os.path.join(transfer_dir, "model_mapping.json")
    if os.path.isfile(mm_path):
        try:
            with open(mm_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "model_mapping" in data:
                return data["model_mapping"], data.get("tool_pricing", {})
            # Backward compat: plain {role: model} format
            return data, {}
        except (json.JSONDecodeError, OSError):
            pass
    return {}, {}


# Backward-compatible default (used when model_mapping.json is absent)
extra_tool = _build_extra_tool({"expert-1": "Qwen/Qwen3-32B"})
try:
    tokenizer = AutoTokenizer.from_pretrained("/data/models/qwen3_8b") if AutoTokenizer else None
except Exception:
    tokenizer = None
def cut_middle_turns(tokenizer,messages,max_length):
    exec_count = 0
    while exec_count<10:
        try:
            exec_count += 1
            messages_str = ''
            start_identifier = generate_random_string(15)
            end_identifier = generate_random_string(15)
            assert not start_identifier in str(messages) and not end_identifier in str(messages) and start_identifier!=end_identifier
            for mid,m in enumerate(messages):
                messages_str += f"{m}{start_identifier}{mid}{end_identifier}"
            token_ids = tokenizer(str(messages_str))['input_ids']
            if len(token_ids)<=max_length:
                return messages
            p1_tokens = tokenizer.batch_decode(token_ids[:max_length//2])
            p1 = ''.join(p1_tokens)
            p1_idx = int(p1.split(start_identifier)[-1].split(end_identifier)[0])
            p2_tokens = tokenizer.batch_decode(token_ids[-max_length//2:])
            p2 = ''.join(p2_tokens)
            p2_idx = int(p2.split(end_identifier)[0].split(start_identifier)[-1])
            return messages[:p1_idx+1]+messages[p2_idx:]
        except Exception as cut_error:
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('cut_error_message.json','w') as f:
                json.dump({
                    'messages': messages,
                    'max_length': max_length,
                    'current_time': str(formatted_time)
                },f,indent=2)
    raise ValueError(f'cut_middle_turns error')

def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.usage
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message],model,use_model_tool) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            if 'qwen' in model.lower() or 'train' in model.lower() or 'huggingface' in model.lower() or 'orchestrator' in model.lower():
                cur_content =  message.content
                if use_model_tool:
                    for s in POLICY_STRINGS:
                        cur_content = cur_content.replace(s,EXPERT_POLICT)
                litellm_messages.append({"role": "system", "content": cur_content+'  Wrap thinking process between <think> </think>, message between <message> </message> and the tool call between <tool_call> </tool_call> .'})
            else:
                litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages

def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    debug = False,
    role=None,
    cur_transfer_dir=None,
    use_model_tool=False,
    model_config_path=None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if role!='user' and role!='assistant' and role!='evaluator':
        raise ValueError(f'unknown role {role}')
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages,model=model,use_model_tool=use_model_tool)
    llm_messages = to_litellm_messages(messages,model='o3',use_model_tool=False)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    original_tools = copy.deepcopy(tools)
    if role=='assistant' and model=='train':
        step_idx = 0
        while os.path.isfile(os.path.join(cur_transfer_dir,f"input_{step_idx}.json")):
            step_idx += 1
        if use_model_tool:
            updated_tools = []
            for t in tools:
                if t['function']['name']!='transfer_to_human_agents':
                    updated_tools.append(t)
            mm, tp = _load_model_mapping(cur_transfer_dir)
            updated_tools += [_build_extra_tool(mm, tp) if mm else extra_tool]
        else:
            updated_tools = tools
        with open(os.path.join(cur_transfer_dir,f"input_{step_idx}.json"),'w') as f:
            json.dump({
                'messages': litellm_messages,
                'original_messages': llm_messages,
                'tools': updated_tools,
                'original_tools': tools
            }, f, indent=2)
        assert not os.path.isfile(os.path.join(cur_transfer_dir,f"wait_output_{step_idx}"))
        with open(os.path.join(cur_transfer_dir,f"wait_output_{step_idx}"),'w') as f:
            f.write(f"wait_output_{step_idx}")
        while not os.path.isfile(os.path.join(cur_transfer_dir,f"output_{step_idx}.json")):
            time.sleep(5)
        with open(os.path.join(cur_transfer_dir,f"output_{step_idx}.json")) as f:
            response = json.load(f)
        assert not os.path.isfile(os.path.join(cur_transfer_dir,f"after_wait_output_{step_idx}"))
        with open(os.path.join(cur_transfer_dir,f"after_wait_output_{step_idx}"),'w') as f:
            f.write(f"wait_output_{step_idx}:\n{response}")
    else:
        response = get_llm_response(model=model,messages=litellm_messages,tools=tools,return_raw_response=True,retry_count=2,max_length=40000)
        tool_calls = []
        if not isinstance(response,str) and response.choices[0].message.tool_calls:
            for one_tool_call in response.choices[0].message.tool_calls:
                tool_calls.append({
                    'name': one_tool_call.function.name,
                    'arguments': json.loads(one_tool_call.function.arguments)
                })
        input_tokens = 0
        output_tokens = 0
        if isinstance(response,str):
            response_content = "Wait a minute, I will take it very soon"
        else:
            response_content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        response = {
            'content': response_content,
            'tool_calls': tool_calls,
            'mode_to_call': f'{role}_'+model,
            'calling_messages': litellm_messages,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        }

    cost = 0
    usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
    }
    content = response['content']
    tool_calls = []
    if response['tool_calls']:
        assert isinstance(response['tool_calls'],list)
        for one_tool_call in response['tool_calls']:
            my_uuid = uuid.uuid4()
            uuid_string1 = str(my_uuid)
            tool_calls.append(ToolCall(
                    id=f"{uuid_string1}",
                    name=one_tool_call['name'],
                    arguments=one_tool_call['arguments'],
                ))
            if len(tool_calls)>5:
                break
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response,
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
