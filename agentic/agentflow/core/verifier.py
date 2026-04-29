import logging
import re
from typing import Any

from .llm_engine import SGLangEngine
from .memory import Memory

logger = logging.getLogger(__name__)


class Verifier:
    def __init__(self, llm_engine: SGLangEngine, available_tools: list[str], tools_metadata: dict):
        self.llm_engine = llm_engine
        self.available_tools = available_tools
        self.tools_metadata = tools_metadata

    async def verificate_context(self, question: str, query_analysis: str, step_count: int, memory: Memory) -> Any:

        prompt_memory_verification = f"""
Task: Evaluate whether the accumulated tool results are sufficient, or whether more tool calls are needed.

Context:
- **Query:** {question}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.tools_metadata}
- **Planning Brief:** {query_analysis}
- **Memory (Tools Used & Results):** {memory.get_actions()}

Instructions:
1.  Use the original query only to understand the requested final target.
2.  Judge sufficiency using only facts, formulas, computations, and intermediate values already present in Memory.
3.  Do NOT solve the problem yourself, derive new formulas, perform arithmetic, or fill in missing reasoning.
4.  If Memory contains a tool error, an unknown tool, NEEDS_SMALLER_SUBGOAL, NEEDS_NUMERIC_SUBGOAL, or an incomplete local result, conclude CONTINUE.
5.  Conclude STOP only when Memory already contains enough reliable tool results for final_output to synthesize the answer without new reasoning or calculation.
6.  If more information is needed, name the smallest missing local sub-goal and the tool type that could provide it.

Final Determination:
-   If Memory is sufficient, explain why and conclude with "Conclusion: STOP".
-   If more information is needed, explain what is missing and conclude with "Conclusion: CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE".
"""
        messages = [{"role": "user", "content": prompt_memory_verification}]
        verifier_out = await self.llm_engine.generate(messages)
        logger.debug("verifier response: %s", verifier_out.response)
        analysis, conclusion = self.parse_conclusion(verifier_out.response)
        logger.debug("verifier conclusion: %s", conclusion)
        return analysis, conclusion, verifier_out

    def parse_conclusion(self, response: str) -> tuple[str, str]:
        analysis = response
        pattern = r'conclusion\**:?\s*\**\s*(STOP|CONTINUE)\b'
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            return analysis, matches[-1].group(1).upper()

        last_lines = response.strip().splitlines()[-3:]
        tail = "\n".join(last_lines).lower()
        if re.search(r'\bstop\b', tail):
            return analysis, 'STOP'
        elif re.search(r'\bcontinue\b', tail):
            return analysis, 'CONTINUE'
        else:
            logger.warning("No valid conclusion (STOP or CONTINUE) found in the response. Defaulting to CONTINUE.")
            return analysis, 'CONTINUE'
