"""
Quick test for ExpertCallerTool.
Before running, ensure the corresponding vllm service is started:
  CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/qwen3_8b --port 30001 ...
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from tools.expert_caller.tool import ExpertCallerTool


# Simulate a model_mapping from one training sample
MODEL_MAPPING = {
    "expert-1": "Qwen/Qwen3-32B",
    "expert-2": "Qwen/Qwen3-32B",
    "expert-3": "Qwen/Qwen2.5-Coder-32B-Instruct",
}

# vllm service URLs (update to actual ports)
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

    # Test routing resolution
    model_name, base_url = tool._get_model_and_url("expert-1")
    print(f"expert-1 -> model={model_name}, url={base_url}")

    model_name, base_url = tool._get_model_and_url("expert-3")
    print(f"expert-3 -> model={model_name}, url={base_url}")

    # Test error handling
    result = await tool.execute(query="test", expert="nonexistent")
    print(f"nonexistent expert: {result}")

    # Live call (requires the vllm service to be running)
    # result = await tool.execute(
    #     query="What is 2 + 2? Answer briefly.",
    #     expert="expert-1",
    # )
    # print(f"expert-1 response: {result}")


if __name__ == "__main__":
    asyncio.run(main())
