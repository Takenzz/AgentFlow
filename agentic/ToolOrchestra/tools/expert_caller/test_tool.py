"""
快速测试 ExpertCallerTool。
运行前确保对应的 vllm 服务已启动：
  CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/qwen3_8b --port 30001 ...
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from tools.expert_caller.tool import ExpertCallerTool


# 模拟一条训练数据的 model_mapping
MODEL_MAPPING = {
    "expert-1": "Qwen/Qwen3-32B",
    "expert-2": "Qwen/Qwen3-32B",
    "expert-3": "Qwen/Qwen2.5-Coder-32B-Instruct",
}

# vllm 服务地址（改成实际端口）
EXPERT_ENGINE_MAP = {
    "Qwen/Qwen3-32B":                   "http://127.0.0.1:30001/v1",
    "Qwen/Qwen2.5-Coder-32B-Instruct":  "http://127.0.0.1:30002/v1",
    "Qwen/Qwen2.5-Math-7B-Instruct":    "http://127.0.0.1:30003/v1",
}


async def main():
    tool = ExpertCallerTool(
        model_mapping=MODEL_MAPPING,
        expert_engine_map=EXPERT_ENGINE_MAP,
    )

    # 测试路由解析
    model_name, base_url = tool._get_model_and_url("expert-1")
    print(f"expert-1 -> model={model_name}, url={base_url}")

    model_name, base_url = tool._get_model_and_url("expert-3")
    print(f"expert-3 -> model={model_name}, url={base_url}")

    # 测试错误处理
    result = await tool.execute(query="test", expert="nonexistent")
    print(f"nonexistent expert: {result}")

    # 实际调用（需要 vllm 服务在线）
    # result = await tool.execute(
    #     query="What is 2 + 2? Answer briefly.",
    #     expert="expert-1",
    # )
    # print(f"expert-1 response: {result}")


if __name__ == "__main__":
    asyncio.run(main())
