"""
测试 SearchRetrievalTool
运行方式：
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

# 从数据集里取几个真实的 eid
with open("/home/ubuntu/ToolOrchestra/data/general_thought_example_urls.json") as f:
    all_eids = list(json.load(f).keys())

VALID_EID = all_eids[0]


async def test_normal_retrieval():
    print(f"=== 测试1: 正常检索 (eid={VALID_EID}) ===")
    tool = SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid=VALID_EID, topk=3)
    result = await tool.execute("how to prove root is unique using derivatives")
    print(result)
    print()


async def test_topk():
    print(f"=== 测试2: topk=1 ===")
    tool = SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid=VALID_EID, topk=1)
    result = await tool.execute("calculus derivative proof")
    print(result)
    print()


async def test_multiple_eids():
    print("=== 测试3: 不同 eid 的检索结果 ===")
    for eid in all_eids[:3]:
        tool = SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid=eid, topk=2)
        result = await tool.execute("mathematics proof theorem")
        doc_count = result.count("[Doc")
        print(f"  eid={eid}: 返回 {doc_count} 篇文档")
        if doc_count > 0:
            first_line = result.split("\n")[0]
            print(f"    {first_line}")
    print()


async def test_invalid_eid():
    print("=== 测试4: 无效 eid ===")
    tool = SearchRetrievalTool(retrieval_url=RETRIEVAL_URL, eid="nonexistent_999", topk=3)
    result = await tool.execute("test query")
    print(result)
    print()


async def test_connection_error():
    print("=== 测试5: 连接失败 ===")
    tool = SearchRetrievalTool(retrieval_url="http://127.0.0.1:9999/retrieve", eid=VALID_EID)
    result = await tool.execute("test query")
    print(result)
    print()


async def test_concurrent():
    print("=== 测试6: 并发请求 ===")
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
        print(f"  并发请求 {i+1}: 返回 {doc_count} 篇文档")
    print()


async def main():
    await test_normal_retrieval()
    await test_topk()
    await test_multiple_eids()
    await test_invalid_eid()
    await test_connection_error()
    await test_concurrent()
    print("所有测试完成。")


if __name__ == "__main__":
    asyncio.run(main())
