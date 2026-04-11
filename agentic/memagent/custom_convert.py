"""
MemAgent custom_convert — multi-turn trajectory unrolling
=========================================================
Directly reuses the agentflow implementation:
  - Unrolls each sample's turns into independent training sequences
  - Rewards are evenly distributed across turns (reward / T_i), corresponding to
    the J_Flow-GRPO objective in the paper
  - Truncated to a multiple of global_batch_size to ensure divisibility
"""

from agentflow.custom_convert import custom_convert  # noqa: F401
