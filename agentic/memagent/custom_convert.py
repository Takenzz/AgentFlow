"""
MemAgent custom_convert — 多轮轨迹展开
======================================
直接复用 agentflow 的实现：
  - 将每条 sample 的 turns 展开为独立训练序列
  - 奖励按轮次数均摊（reward / T_i），对应论文的 J_Flow-GRPO 目标
  - 按 global_batch_size 截断保证整除
"""

from agentflow.custom_convert import custom_convert  # noqa: F401
