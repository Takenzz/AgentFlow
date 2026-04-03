"""
Search Retrieval Tool
---------------------
调用本地 FAISS 检索服务（retrieval_general_thought.py）检索相关文档。

检索服务接口：
    POST http://127.0.0.1:8000/retrieve
    {
        "queries": ["查询文本"],
        "topk": 5,
        "return_scores": true,
        "eid": "样本id"   # 用于过滤只返回该样本相关的文档
    }
"""

from __future__ import annotations

import logging
import asyncio
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)

TOOL_NAME = "Search_Retrieval_Tool"

TOOL_DESCRIPTION = """
Search for relevant documents using local dense retrieval.
Use this tool when you need to find supporting information or facts to answer a question.
"""

TOOL_DEMO_COMMANDS = {
    "command": 'execution = tool.execute(query="What is the boiling point of water?")',
    "description": "Search for documents relevant to the query.",
}


class SearchRetrievalTool:
    """
    对接本地 FAISS 检索服务的工具。
    每个样本实例化一次，携带该样本的 eid（用于文档过滤）。
    """

    def __init__(
        self,
        retrieval_url: str = "http://127.0.0.1:8000/retrieve",
        eid: Optional[str] = None,
        topk: int = 5,
        max_content_length: int = 500,
        timeout: float = 30.0,
    ):
        """
        Args:
            retrieval_url:      FAISS 检索服务地址
            eid:                样本 id，用于检索服务内部的文档过滤
            topk:               返回的最大文档数
            max_content_length: 每篇文档截断的最大字符数
            timeout:            HTTP 请求超时秒数
        """
        self.retrieval_url = retrieval_url
        self.eid = eid
        self.topk = topk
        self.max_content_length = max_content_length
        self.timeout = timeout

    def _format_results(self, results: list) -> str:
        """将检索结果列表格式化为可读字符串。"""
        if not results:
            return "No relevant documents found."

        lines = []
        for i, item in enumerate(results, 1):
            doc = item.get("document", {})
            content = doc.get("content", doc.get("contents", "")).strip()
            score = item.get("score", 0.0)
            if content:
                content = content[: self.max_content_length]
                lines.append(f"[Doc {i}] (score={score:.4f})\n{content}")

        return "\n\n".join(lines) if lines else "No relevant documents found."

    async def execute(self, query: str) -> str:
        """
        检索与 query 相关的文档，返回格式化的文档字符串。

        Args:
            query: 检索查询文本

        Returns:
            格式化的检索结果字符串；出错时返回错误描述（不抛异常）。
        """
        payload = {
            "queries": [query],
            "topk": self.topk,
            "return_scores": True,
            "eid": self.eid,
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(self.retrieval_url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(
                            "[SearchRetrievalTool] HTTP %d: %s", resp.status, text[:200]
                        )
                        return f"[SearchRetrievalTool Error] HTTP {resp.status}: {text[:200]}"
                    data = await resp.json()

        except asyncio.TimeoutError:
            return f"[SearchRetrievalTool Error] Request timed out after {self.timeout}s"
        except aiohttp.ClientConnectionError as e:
            return f"[SearchRetrievalTool Error] Connection failed: {e}"
        except Exception as e:
            return f"[SearchRetrievalTool Error] {type(e).__name__}: {e}"

        # data 是 [[{document, score}, ...]] 结构
        if not isinstance(data, list) or not data:
            return "No relevant documents found."

        results = data[0]
        formatted = self._format_results(results)

        logger.debug(
            "[SearchRetrievalTool] query=%r eid=%s docs_returned=%d",
            query[:50], self.eid, len(results),
        )
        return formatted
