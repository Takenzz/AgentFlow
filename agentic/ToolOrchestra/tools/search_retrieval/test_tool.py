"""
Test SearchRetrievalTool
Usage:
    cd /home/ubuntu/slime-agentic/agentic/ToolOrchestra
    conda run -n orche python tools/search_retrieval/test_tool.py
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tools.search_retrieval.tool import SearchRetrievalTool

RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"

# Take a few real eids from the dataset
with open("/home/ubuntu/ToolOrchestra/data/general_thought_example_urls.json") as f:
    all_eids = list(json.load(f).keys())

VALID_EID = all_eids[0]


async def test_normal_retrieval():
    print(f"=== Test 1: Normal retrieval (eid={VALID_EID}) ===")
    tool = SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid=VALID_EID, topk=3)
    result = await tool.execute("how to prove root is unique using derivatives")
    print(result)
    print()


async def test_topk():
    print(f"=== Test 2: topk=1 ===")
    tool = SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid=VALID_EID, topk=1)
    result = await tool.execute("calculus derivative proof")
    print(result)
    print()


async def test_multiple_eids():
    print("=== Test 3: Retrieval results for different eids ===")
    for eid in all_eids[:3]:
        tool = SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid=eid, topk=2)
        result = await tool.execute("mathematics proof theorem")
        doc_count = result.count("[Doc")
        print(f"  eid={eid}: returned {doc_count} documents")
        if doc_count > 0:
            first_line = result.split("\n")[0]
            print(f"    {first_line}")
    print()


async def test_invalid_eid():
    print("=== Test 4: Invalid eid ===")
    tool = SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid="nonexistent_999", topk=3)
    result = await tool.execute("test query")
    print(result)
    print()


async def test_connection_error():
    print("=== Test 5: Connection failure ===")
    tool = SearchRetrievalTool(retrieval_url="http://127.0.0.1:9999/retrieve", eid=VALID_EID)
    result = await tool.execute("test query")
    print(result)
    print()


async def test_concurrent():
    print("=== Test 6: Concurrent requests ===")
    tools = [
        SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid=eid, topk=2)
        for eid in all_eids[:5]
    ]
    queries = [
        "derivative proof",
        "polynomial equation",
        "calculus theorem",
        "mathematical analysis",
        "root uniqueness",
    ]
    tasks = [tool.execute(q) for tool, q in zip(tools, queries)]
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        doc_count = result.count("[Doc")
        print(f"  Concurrent request {i+1}: returned {doc_count} documents")
    print()


async def main():
    await test_normal_retrieval()
    await test_topk()
    await test_multiple_eids()
    await test_invalid_eid()
    await test_connection_error()
    await test_concurrent()
    print("All tests completed.")


if __name__ == "__main__":
    asyncio.run(main())
