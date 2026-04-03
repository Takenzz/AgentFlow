"""
PromptBuilder
-------------
根据训练数据样本构建 Orchestrator 的输入 prompt。

支持两种 category：
- qa:        问题 + 上下文（检索文档 / 代码片段 / 已有答案），工具列表为 enhance_reasoning / search / answer
- func_call: 由 tau2 环境驱动，工具为 call_expert

输出格式：
    messages: list[dict]   标准 chat messages
    tools:    list[dict]   OpenAI function call 格式的工具定义列表
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = "You are an orchestrator. You must ALWAYS call the appropriate tool to handle user requests. Never answer directly without using a tool."


def _to_tools_list(tools_field: Any) -> list[dict]:
    """
    数据里 tools 字段可能是单个 dict（func_call）或 list（qa），统一转成 list。
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
    将检索文档、代码片段、已有答案拼接成上下文字符串。
    整体长度截断到 max_context_chars。
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
    构建 Orchestrator 输入 prompt 的工具类。
    每次调用 build() 返回 (messages, tools)。
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
        构建 qa 类型的 prompt。

        Args:
            problem:       问题文本
            tools_field:   数据中的 tools 字段（list 或 dict）
            documents:     已检索到的文档列表
            code_snippets: 代码执行结果列表，每项含 code / output
            attempts:      已有的答案尝试列表，每项含 model / answer

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
        构建 func_call 类型的 prompt。
        conversation 是 tau2 环境传入的多轮对话（已含 system/user/assistant/tool 轮次）。

        Args:
            conversation:  完整的多轮对话 messages
            tools_field:   数据中的 tools 字段

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
        把工具执行结果追加到对话中（tool role），供下一轮 Orchestrator 继续推理。

        Args:
            messages:     当前对话
            tool_call_id: 对应 assistant 消息中的 tool call id
            tool_name:    工具名称
            result:       工具返回的字符串结果

        Returns:
            追加了 tool 消息的新 messages 列表
        """
        return messages + [
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        ]
