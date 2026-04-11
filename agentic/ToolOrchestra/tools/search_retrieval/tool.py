"""
Search Retrieval Tool
---------------------
Calls the local FAISS retrieval service (retrieval_general_thought.py) to retrieve relevant documents.

Retrieval service API:
    POST http://127.0.0.1:8000/retrieve
    {
        "queries": ["query text"],
        "topk": 5,
        "return_scores": true,
        "eid": "sample_id"   # used to filter and return only documents relevant to this sample
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
    Tool that interfaces with the local FAISS retrieval service.
    Instantiated once per sample, carrying that sample's eid (used for document filtering).
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
            retrieval_url:      Address of the FAISS retrieval service
            eid:                Sample ID used for document filtering inside the retrieval service
            topk:               Maximum number of documents to return
            max_content_length: Maximum characters to include from each document
            timeout:            HTTP request timeout in seconds
        """
        self.retrieval_url = retrieval_url
        self.eid = eid
        self.topk = topk
        self.max_content_length = max_content_length
        self.timeout = timeout

    def _format_results(self, results: list) -> str:
        """Format a list of retrieval results into a human-readable string."""
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
        Retrieve documents relevant to the query and return a formatted string.

        Args:
            query: The retrieval query text

        Returns:
            Formatted retrieval results string; on error returns an error description without raising.
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

        # data has the structure [[{document, score}, ...]]
        if not isinstance(data, list) or not data:
            return "No relevant documents found."

        results = data[0]
        formatted = self._format_results(results)

        logger.debug(
            "[SearchRetrievalTool] query=%r eid=%s docs_returned=%d",
            query[:50], self.eid, len(results),
        )
        return formatted
