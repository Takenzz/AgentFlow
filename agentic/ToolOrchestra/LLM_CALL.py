# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openai
from openai import AzureOpenAI
from openai import OpenAI
import requests
import time
import os
import json
import subprocess
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

KEYS_DIR = 'keys'
if not os.path.isdir(KEYS_DIR):
    os.makedirs(KEYS_DIR,exist_ok=True)

def convert_openai_tools_to_claude(openai_tools: list) -> list:
    claude_tools = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            raise ValueError(f"Unsupported tool type: {tool.get('type')}")
        
        fn = tool["function"]
        claude_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
        })
    return claude_tools

def normalize_messages_for_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Detects and corrects common Chat Completions tool-message issues:
      1) In assistant messages, each entry in `tool_calls` must have:
         {
           "id": "...",
           "type": "function",
           "function": {"name": "<fn_name>", "arguments": "<json string>"}
         }
         - Moves top-level `name` / `arguments` into `function`.
         - Ensures `type == "function"`.
         - JSON-serializes non-string `arguments`.

      2) In tool messages:
         - Ensures `content` is a string; JSON-serializes if dict/list.
         - Ensures `tool_call_id` exists. If missing, tries to pair with the
           most recent unmatched assistant tool_call ID (by order).

      3) Removes illegal extra fields at `tool_calls` top level.

    Returns:
        (fixed_messages, issues)
        - fixed_messages: deep-copied, corrected messages list
        - issues: human-readable list of detected/corrected problems
    """
    fixed = deepcopy(messages)
    issues = []

    # Build a set of valid function names from `tools` (optional validation)
    valid_fn_names = set()
    if tools:
        for t in tools:
            try:
                if t.get("type") == "function":
                    fn = t.get("function", {})
                    name = fn.get("name")
                    if isinstance(name, str):
                        valid_fn_names.add(name)
            except Exception:
                pass

    # Track assistant tool_calls -> to match subsequent tool results
    pending_tool_call_ids = []

    # First pass: fix assistant tool_calls and record pending IDs
    for i, msg in enumerate(fixed):
        role = msg.get("role")
        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            for j, tc in enumerate(msg["tool_calls"]):
                # Ensure container objects exist
                if not isinstance(tc, dict):
                    issues.append(f"[assistant#{i}] tool_calls[{j}] is not an object; replaced with empty object.")
                    msg["tool_calls"][j] = tc = {}

                # Move name/arguments into function
                fn_obj = tc.get("function") or {}
                moved = False

                if "name" in tc:
                    fn_obj["name"] = tc.pop("name")
                    moved = True
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: moved top-level 'name' into 'function.name'.")

                if "arguments" in tc:
                    fn_obj["arguments"] = tc.pop("arguments")
                    moved = True
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: moved top-level 'arguments' into 'function.arguments'.")

                # Ensure function object present
                if "function" not in tc:
                    tc["function"] = fn_obj if fn_obj else {}
                elif moved:
                    tc["function"].update(fn_obj)

                # Ensure type is "function"
                if tc.get("type") != "function":
                    tc["type"] = "function"
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: set 'type' to 'function'.")

                # Ensure arguments is a string
                if "arguments" in tc["function"]:
                    args_val = tc["function"]["arguments"]
                    if not isinstance(args_val, str):
                        try:
                            tc["function"]["arguments"] = json.dumps(args_val, ensure_ascii=False)
                            issues.append(f"[assistant#{i}] tool_calls[{j}]: JSON-serialized non-string 'function.arguments'.")
                        except Exception:
                            tc["function"]["arguments"] = "{}"
                            issues.append(f"[assistant#{i}] tool_calls[{j}]: failed to serialize arguments; defaulted to '{{}}'.")

                else:
                    # Provide default empty JSON object
                    tc["function"]["arguments"] = "{}"
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: added default empty 'function.arguments'.")

                # Validate function name if possible
                fn_name = tc.get("function", {}).get("name")
                if isinstance(fn_name, str):
                    if valid_fn_names and fn_name not in valid_fn_names:
                        issues.append(f"[assistant#{i}] tool_calls[{j}]: unknown function '{fn_name}' (not in tools).")
                else:
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: missing 'function.name'.")

                # Track pending tool_call_id for pairing
                tc_id = tc.get("id")
                if isinstance(tc_id, str):
                    pending_tool_call_ids.append(tc_id)
                else:
                    # If missing id, synthesize a stable one
                    tc_id = f"call_{i}_{j}"
                    tc["id"] = tc_id
                    pending_tool_call_ids.append(tc_id)
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: synthesized missing 'id' -> '{tc_id}'.")

                # Remove illegal top-level keys except allowed
                allowed = {"id", "type", "function"}
                extraneous = [k for k in list(tc.keys()) if k not in allowed]
                for k in extraneous:
                    tc.pop(k, None)
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: removed unsupported top-level field '{k}'.")

    # Second pass: fix tool messages (pair to pending assistant calls)
    # We'll consume from the front of pending_tool_call_ids in order.
    for i, msg in enumerate(fixed):
        if msg.get("role") == "tool":
            # tool_call_id
            if not msg.get("tool_call_id"):
                if pending_tool_call_ids:
                    inferred = pending_tool_call_ids.pop(0)
                    msg["tool_call_id"] = inferred
                    issues.append(f"[tool#{i}]: added missing 'tool_call_id' -> '{inferred}'.")
                else:
                    issues.append(f"[tool#{i}]: missing 'tool_call_id' and none could be inferred.")

            # content must be string
            content = msg.get("content")
            if not isinstance(content, str):
                try:
                    msg["content"] = json.dumps(content, ensure_ascii=False)
                    issues.append(f"[tool#{i}]: JSON-serialized non-string 'content'.")
                except Exception:
                    msg["content"] = ""
                    issues.append(f"[tool#{i}]: failed to serialize content; set to empty string.")

            # Remove fields illegal for tool role (defensive)
            for bad in ("name", "type", "function"):
                if bad in msg:
                    msg.pop(bad, None)
                    issues.append(f"[tool#{i}]: removed illegal field '{bad}'.")

        # If someone mistakenly returned a tool result as role='assistant' with tool_call_id,
        # quietly convert it to role='tool' (optional but handy).
        if msg.get("role") == "assistant" and "tool_call_id" in msg:
            msg["role"] = "tool"
            issues.append(f"[assistant#{i}]: message had 'tool_call_id'; converted role to 'tool'.")

    return fixed, issues

def convert_openai_messages_to_claude(openai_messages):
    claude_messages = []
    for m in openai_messages:
        if "tool_calls" in m:
            m['content'] += '\n\n'+str(m["tool_calls"])
            m.pop("tool_calls")
            claude_messages.append(m)
        elif m['role']=='tool':
            claude_messages.append({
                "role": 'user',
                "content": "Tool call result: "+m['content']
            })
        else:
            claude_messages.append(m)
    return claude_messages

def get_openai_token(p_token_url, p_client_id, p_client_secret, p_scope, **kwargs):
    try:
        with open(os.path.join(KEYS_DIR,f'openai_key.json')) as f:
            key = json.load(f)
        if time.time()<key['expire_at']:
            return key["access_token"]
    except:
        pass
    
    response = requests.post(
        p_token_url,
        data={"grant_type": "client_credentials", "client_id": p_client_id,
                "client_secret": p_client_secret, "scope": p_scope}
    )
    response.raise_for_status()
    token = response.json()

    with open(os.path.join(KEYS_DIR,f'openai_key.json'),'w') as f:
        json.dump({
            "access_token": token["access_token"],
            'expire_at': time.time()+900
        },f,indent=2)
    os.chmod(str(os.path.join(KEYS_DIR,f'openai_key.json')), 0o777)

    return token["access_token"]

def get_claude_token():
    try:
        with open(os.path.join(KEYS_DIR,'claude_key.json')) as f:
            key = json.load(f)
        if time.time()<key['expire_at']:
            return key["access_token"]
    except:
        pass

    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    command = f"""curl -s --location 'https://5kbfxgaqc3xgz8nhid1x1r8cfestoypn-trofuum-oc.ssa.nvidia.com/token' --header 'Content-Type: application/x-www-form-urlencoded' --header "Authorization: Basic $(echo -n {client_id}:{client_secret} | base64 -w0)" --data-urlencode 'grant_type=client_credentials' --data-urlencode 'scope=awsanthropic-readwrite azureopenai-readwrite' | jq -r '.access_token'"""
    result = subprocess.check_output(command, shell=True, text=True).strip()

    with open(os.path.join(KEYS_DIR,'claude_key.json'),'w') as f:
        json.dump({
            "access_token": result,
            'expire_at': time.time()+900
        },f,indent=2)
    os.chmod(str(os.path.join(KEYS_DIR,'claude_key.json')), 0o777)


    return result


def get_azure_openai_client():
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    token_url = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
    scope = "azureopenai-readwrite"
    token = get_openai_token(token_url, client_id, client_secret, scope)

    openai.api_type = "azure"
    openai.api_base = "https://prod.api.nvidia.com/llm/v1/azure/"
    openai.api_version = "2025-04-01-preview"
    openai.api_key = token
    client = AzureOpenAI(
        api_key=token,
        api_version="2025-04-01-preview",
        azure_endpoint="https://prod.api.nvidia.com/llm/v1/azure/"
    )
    print("AzureOpenAI instance created successfully.")
    return client


def get_openai_client():
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    token_url = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
    scope = "azureopenai-readwrite"

    print("Getting OAuth token...")
    print(f"client_id: {client_id}, client_secret: {client_secret}, token_url: {token_url}, scope: {scope}")
    token = get_openai_token(token_url, client_id, client_secret, scope)

    client = OpenAI(
        api_key=token,
        base_url="https://prod.api.nvidia.com/llm/v1/azure/v1",
        default_query={"api-version": "preview"}, 
    )
    print("OpenAI instance created successfully.")
    return client

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-turbo-latest")

_dashscope_client = None

def _get_dashscope_client():
    global _dashscope_client
    if _dashscope_client is None:
        if not DASHSCOPE_API_KEY:
            raise EnvironmentError(
                "DASHSCOPE_API_KEY is not set. "
                "Please export DASHSCOPE_API_KEY=<your-api-key> before running."
            )
        _dashscope_client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
    return _dashscope_client


def get_llm_response(model, messages, temperature=1.0, return_raw_response=False, tools=None, show_messages=False, model_type=None, max_length=1024, model_config=None, model_config_idx=0, model_config_path=None, payload=None, openai_client_type='azure_openai', **kwargs):
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]

    if max_length == 1024:
        max_length = 8000

    dashscope_model = DASHSCOPE_MODEL
    client = _get_dashscope_client()

    retry_count = kwargs.get("retry_count", 3)
    for attempt in range(retry_count + 1):
        try:
            create_kwargs = dict(
                model=dashscope_model,
                messages=messages,
                temperature=temperature,
                max_tokens=min(max_length, 8000),
            )
            if tools:
                create_kwargs["tools"] = tools

            chat_completion = client.chat.completions.create(**create_kwargs)

            if return_raw_response:
                return chat_completion
            else:
                return chat_completion.choices[0].message.content
        except Exception as error:
            print(f'[LLM_CALL] DashScope error (attempt {attempt+1}/{retry_count+1}): {error}')
            if attempt < retry_count:
                time.sleep(min(5 * (attempt + 1), 30))
            else:
                raise


