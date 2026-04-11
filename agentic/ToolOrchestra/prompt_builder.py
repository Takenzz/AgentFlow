"""
PromptBuilder
-------------
Builds the Orchestrator's input prompt from training data samples.

Supports two categories:
- qa:        question + context (retrieved documents / code snippets / existing answers),
             tool list: enhance_reasoning / search / answer
- func_call: driven by the tau2 environment, tool: call_expert

Output format:
    messages: list[dict]   standard chat messages
    tools:    list[dict]   tool definition list in OpenAI function calling format
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = "You are an orchestrator. You must ALWAYS call the appropriate tool to handle user requests. Never answer directly without using a tool."


def _to_tools_list(tools_field: Any) -> list[dict]:
    """
    The tools field in the data may be a single dict (func_call) or a list (qa);
    normalize to a list in either case.
    """
    if isinstance(tools_field, list):
        return tools_field
    if isinstance(tools_field, dict):
        return [tools_field]
    return []


def _build_context_str(
    documents: list[str],
    code_snippets: list[dict],
    attempts: list[dict],
    max_doc_chars: int = 4000,
    max_context_chars: int = 24000,
) -> str:
    """
    Concatenate retrieved documents, code snippets, and existing answers into a context string.
    Total length is truncated to max_context_chars.
    """
    doc_str = ""
    for i, doc in enumerate(documents, 1):
        doc_str += f"Doc {i}: {doc[:max_doc_chars]}\n\n"

    code_str = ""
    for piece in code_snippets:
        code_str += f"```python\n{piece.get('code', '')}\n```\n\n"
        code_str += f"```output\n{piece.get('output', '')}\n```\n\n"

    attempt_str = ""
    for i, attempt in enumerate(attempts, 1):
        attempt_str += f"Attempt {i} by {attempt.get('model', '?')}: {attempt.get('answer', '')}\n"

    combined = code_str + attempt_str
    if len(combined) > max_context_chars:
        combined = combined[:max_context_chars]

    remaining = max_context_chars - len(combined)
    if doc_str and remaining > 0:
        doc_str = doc_str[:remaining]
        context = "Documents:\n" + doc_str + combined
    else:
        context = combined

    return context.strip()


class PromptBuilder:
    """
    Utility class for building the Orchestrator's input prompt.
    Each call to build() returns (messages, tools).
    """

    def build_qa(
        self,
        problem: str,
        tools_field: Any,
        documents: list[str] | None = None,
        code_snippets: list[dict] | None = None,
        attempts: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Build a prompt for the qa category.

        Args:
            problem:       Question text
            tools_field:   The tools field from the data (list or dict)
            documents:     List of already-retrieved documents
            code_snippets: List of code execution results; each item contains code / output
            attempts:      List of existing answer attempts; each item contains model / answer

        Returns:
            (messages, tools)
        """
        documents = documents or []
        code_snippets = code_snippets or []
        attempts = attempts or []

        context_str = _build_context_str(documents, code_snippets, attempts)

        user_content = f"Problem: {problem}"
        if context_str:
            user_content += f"\n\n{context_str}"
        user_content += "\n\nChoose an appropriate tool."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        tools = _to_tools_list(tools_field)
        return messages, tools

    def build_func_call(
        self,
        conversation: list[dict],
        tools_field: Any,
    ) -> tuple[list[dict], list[dict]]:
        """
        Build a prompt for the func_call category.
        conversation is the multi-turn dialogue passed in by the tau2 environment
        (already contains system/user/assistant/tool turns).

        Args:
            conversation:  Full multi-turn conversation messages
            tools_field:   The tools field from the data

        Returns:
            (messages, tools)
        """
        tools = _to_tools_list(tools_field)
        return conversation, tools

    def append_tool_result(
        self,
        messages: list[dict],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict]:
        """
        Append a tool execution result to the conversation (as a tool role message),
        so the Orchestrator can continue reasoning in the next turn.

        Args:
            messages:     Current conversation
            tool_call_id: The tool call ID from the corresponding assistant message
            tool_name:    Name of the tool
            result:       String result returned by the tool

        Returns:
            New messages list with the tool message appended
        """
        return messages + [
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        ]
