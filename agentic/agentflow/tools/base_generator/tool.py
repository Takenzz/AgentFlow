from __future__ import annotations

import re

from tools.base import BaseTool

TOOL_NAME = "Local_Math_Deduction_Tool"

TOOL_DESCRIPTION = """
A constrained local reasoning tool. It can derive one narrow math relation,
identify one relevant theorem, or check one small logical transformation.
It must not solve the full original problem, compute the final answer, or
produce a full step-by-step solution. The planner must provide the local
question; the tool should not choose the overall strategy.
"""

TOOL_DEMO_COMMANDS = {
    "command": (
        'execution = tool.execute(query="""State one local relationship requested '
        'by the planner, using only the supplied variables and assumptions.""")'
    ),
    "description": "Derive one local identity, relation, or check needed by the Planner.",
}


_BROAD_REQUEST_PATTERNS = [
    r"\bsolve\s+(?:the|this)\s+(?:problem|question)\b",
    r"\banswer\s+(?:the|this)\s+(?:problem|question)\b",
    r"\bcomplete\s+(?:solution|proof)\b",
    r"\bfull\s+(?:solution|proof)\b",
    r"\b(?:entire|whole)\s+(?:solution|proof|derivation)\b",
    r"\bfinal\s+answer\b",
    r"\b(?:produce|give|return)\b.{0,60}\banswer\b",
    r"\bboxed\s*\{",
    r"\b(?:find|calculate|compute|determine)\b.{0,80}\b(?:the\s+)?(?:answer|final\s+value|requested\s+value|target\s+quantity)\b",
]

_LOCAL_REQUEST_MARKERS = [
    "identity",
    "lemma",
    "relation",
    "relationship",
    "theorem",
    "derive",
    "state",
    "verify",
    "check",
    "transform",
    "equivalence",
    "necessary condition",
    "sufficient condition",
    "local",
]


def _looks_like_full_solution_request(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    if any(re.search(pattern, normalized) for pattern in _BROAD_REQUEST_PATTERNS):
        return True

    has_local_marker = any(marker in normalized for marker in _LOCAL_REQUEST_MARKERS)
    broad_target = re.search(
        r"\b(?:find|calculate|compute|determine|count)\b.{0,120}\b(?:number|value|sum|probability|least|greatest|maximum|minimum)\b",
        normalized,
    )
    # Long target-shaped requests without a local-reasoning marker are usually
    # copied from the original problem and should be narrowed by the planner.
    if broad_target and not has_local_marker and len(normalized) > 260:
        return True

    return False


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
- If the request asks you to choose the overall strategy or produce the target answer, return NEEDS_SMALLER_SUBGOAL.

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
