from __future__ import annotations

import re

from tools.base import BaseTool

TOOL_NAME = "Local_Math_Deduction_Tool"

TOOL_DESCRIPTION = """
A constrained local reasoning tool. It can derive one narrow math relation,
identify one relevant theorem, or check one small logical transformation.
It must not solve the full original problem, compute the final answer, or
produce a full step-by-step solution.
"""

TOOL_DEMO_COMMANDS = {
    "command": (
        'execution = tool.execute(query="""Given a triangle with circumradius R and '
        'inradius r, state the standard formula for OI^2 only.""")'
    ),
    "description": "Derive one local identity or theorem needed by the Planner.",
}


_BROAD_REQUEST_PATTERNS = [
    r"\bsolve\s+(?:the|this)\s+(?:problem|question)\b",
    r"\bcomplete\s+(?:solution|proof)\b",
    r"\bfull\s+(?:solution|proof)\b",
    r"\bfinal\s+answer\b",
    r"\bboxed\s*\{",
    r"\b(find|calculate|compute|determine)\b.{0,80}\b(product\s+of\b|ab\s*(?:\\cdot|\*|and)\s*ac\b)",
]


def _looks_like_full_solution_request(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return any(re.search(pattern, normalized) for pattern in _BROAD_REQUEST_PATTERNS)


class Base_Generator_Tool(BaseTool):
    def __init__(self, llm_engine):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            demo_commands=TOOL_DEMO_COMMANDS

        )
        self.llm_engine = llm_engine

    async def execute(self, query: str) -> str:
        if _looks_like_full_solution_request(query):
            return (
                "NEEDS_SMALLER_SUBGOAL: Local_Math_Deduction_Tool only accepts one "
                "narrow lemma, identity, or local transformation. Rewrite the request "
                "as a specific intermediate relation instead of asking for the final answer."
            )

        system_prompt = """
You are Local_Math_Deduction_Tool inside an agent workflow.

Your role:
- Answer exactly one narrow mathematical sub-question.
- State a local identity, lemma, relation, or short derivation.
- Keep the result usable by a Planner that will decide the next step.

Strict limits:
- Do not solve the full original problem.
- Do not compute or reveal the final answer to the original problem.
- Do not produce a boxed answer.
- Do not write a long tutorial or multi-stage proof.
- If the user request is too broad, return NEEDS_SMALLER_SUBGOAL with one sentence explaining the missing narrower target.

Return format:
LOCAL_RESULT: <one local result>
ASSUMPTIONS: <inputs used>
LIMITS: <what this result does not solve>
NEXT_NEEDED: <the next local quantity or check, if any>
"""
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": query},
        ]
        out = await self.llm_engine.generate(messages)
        return out.response
