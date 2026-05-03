import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import List

from .llm_engine import SGLangEngine, GenerationOutput
from .memory import Memory

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self, llm_engine: SGLangEngine, available_tools: List = None, toolbox_metadata: dict = None,
                 tools_dir: str = None):
        self.llm_engine = llm_engine

        if available_tools is None and toolbox_metadata is None and tools_dir is not None:
            available_tools, toolbox_metadata = self._discover_tools(tools_dir)

        self.available_tools = available_tools or []
        self.toolbox_metadata = toolbox_metadata or {}

    @staticmethod
    def _preload_tools_base(tools_path: Path):
        """Register tools/base.py under the fixed name 'tools.base' in sys.modules,
        so that `from tools.base import ...` in each tool.py hits the cache directly,
        independent of sys.path ordering or environment differences."""
        import types as _types

        base_file = tools_path / "base.py"
        if not base_file.exists() or "tools.base" in sys.modules:
            return

        # Register the virtual parent package 'tools' (if it doesn't already exist)
        if "tools" not in sys.modules:
            tools_pkg = _types.ModuleType("tools")
            tools_pkg.__path__ = [str(tools_path)]
            tools_pkg.__package__ = "tools"
            sys.modules["tools"] = tools_pkg

        spec = importlib.util.spec_from_file_location("tools.base", base_file)
        base_mod = importlib.util.module_from_spec(spec)
        sys.modules["tools.base"] = base_mod
        spec.loader.exec_module(base_mod)
        sys.modules["tools"].base = base_mod

    @staticmethod
    def _discover_tools(tools_dir: str):
        """Scan the tools directory and load TOOL_NAME and TOOL_DESCRIPTION from tool.py in each subdirectory."""
        available_tools = []
        toolbox_metadata = {}

        tools_path = Path(tools_dir)
        # Pre-register tools.base in sys.modules so that imports in tool.py reliably hit the cache
        Planner._preload_tools_base(tools_path)

        for tool_file in sorted(tools_path.rglob("tool.py")):
            module_name = f"_tool_{tool_file.parent.name}"
            spec = importlib.util.spec_from_file_location(module_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.warning("[Planner] Skipping %s, failed to load: %s", tool_file, e)
                continue

            tool_name = getattr(module, "TOOL_NAME", None)
            tool_description = getattr(module, "TOOL_DESCRIPTION", None)

            if tool_name:
                available_tools.append(tool_name)
                toolbox_metadata[tool_name] = {
                    "description": tool_description or "",
                }

        return available_tools, toolbox_metadata

    def _build_prompt(self, question: str) -> str:
        return f"""Task: Build a concise planning brief for an agent workflow.

Inputs:
- Query: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

Instructions:
1. Identify the final target and the information explicitly given.
2. Describe the decomposition policy in generic terms: what kind of local result should be collected first, what kind of local computation/check may follow, and what evidence is needed before stopping.
3. Mention which available tool type is suitable for each kind of local operation.
4. Do NOT solve the problem, derive domain-specific formulas, perform arithmetic, choose a final answer, or include answer-seeking content.
5. Keep this as a routing/planning note, not a solution.

Keep the entire response concise and under 250 words.

"""


    async def plan(self, question: str) -> GenerationOutput:
        planning_prompt = self._build_prompt(question)
        messages = [{"role": "user", "content": planning_prompt}]
        return await self.llm_engine.generate(messages)

    @staticmethod
    def _is_final_aggregation_action(action: dict) -> bool:
        """Whether the Planner marked this step as the narrow final aggregation.

        Tools only perform local work. The global decision that a local
        calculation is the final aggregation must come from the Planner's
        sub-goal and executor command, not from tool-emitted text.
        """
        tool_name = str(action.get("tool_name", "")).strip()
        if tool_name != "Python_Code_Generator_Tool":
            return False

        text = " ".join(
            [
                str(action.get("sub_goal", "")),
                str(action.get("command", "")),
            ]
        ).lower()
        final_markers = [
            "final aggregation",
            "final value",
            "requested final value",
            "final target",
            "final requested value",
        ]
        local_markers = [
            "using only",
            "supplied",
            "recorded",
            "previous",
            "intermediate",
            "relevant data",
        ]
        broad_markers = [
            "original problem",
            "original query",
            "entire problem",
            "whole problem",
            "full solution",
            "solve from scratch",
        ]
        return (
            any(marker in text for marker in final_markers)
            and any(marker in text for marker in local_markers)
            and not any(marker in text for marker in broad_markers)
        )

    @staticmethod
    def _extract_computed_result(result: str) -> str:
        match = re.search(r"COMPUTED_RESULT:\s*(.*)", result, re.DOTALL)
        if not match:
            return ""
        body = match.group(1).strip()
        body = re.split(r"\n(?:TOOL_STATUS|LOCAL_RESULT|ASSUMPTIONS|LIMITS|NEXT_NEEDED):", body, maxsplit=1)[0].strip()
        return body

    @staticmethod
    def has_successful_planner_marked_final_aggregation(memory: Memory) -> bool:
        for action in memory.get_actions().values():
            result = str(action.get("result", ""))
            if (
                "TOOL_STATUS: OK" in result
                and Planner._is_final_aggregation_action(action)
                and Planner._last_nonempty_line(Planner._extract_computed_result(result))
            ):
                return True
        return False

    @staticmethod
    def _last_nonempty_line(text: str) -> str:
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _extract_final_aggregation_candidates(memory: Memory) -> tuple[list[tuple[str, int]], bool]:
        """Return final aggregation results and later blocking status.

        The final answer stage is deterministic and only trusts a successful
        computation result when the step is explicitly marked by the Planner as
        a narrow final aggregation over recorded intermediate values, and the
        Verifier has issued STOP at or after that step.
        """
        candidates: list[tuple[str, int]] = []
        blocking_after_last_candidate = False

        actions = list(memory.get_actions().items())
        stop_indices = [
            idx
            for idx, (_, action) in enumerate(actions)
            if str(action.get("verifier_conclusion", "")).strip().upper() == "STOP"
        ]
        last_stop_index = max(stop_indices) if stop_indices else -1
        last_candidate_index = -1
        for idx, (_, action) in enumerate(actions):
            result = str(action.get("result", ""))
            if "TOOL_STATUS: OK" not in result:
                continue
            if not Planner._is_final_aggregation_action(action):
                continue
            if last_stop_index < idx:
                continue

            computed = Planner._extract_computed_result(result)
            candidate = Planner._last_nonempty_line(computed)
            candidate = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", candidate).strip()
            if candidate and candidate.upper() not in {"NONE", "UNKNOWN", "INSUFFICIENT_TOOL_RESULTS"}:
                candidates.append((candidate, idx))
                last_candidate_index = idx

        if last_candidate_index >= 0:
            for _, action in actions[last_candidate_index + 1:]:
                result = str(action.get("result", ""))
                if (
                    "NEEDS_SMALLER_SUBGOAL" in result
                    or "NEEDS_NUMERIC_SUBGOAL" in result
                    or "TOOL_STATUS: ERROR" in result
                    or "Execution error:" in result
                    or "Code generation error:" in result
                    or "Command parse error" in result
                    or "Tool load error" in result
                ):
                    blocking_after_last_candidate = True
                    break

        return candidates, blocking_after_last_candidate

    @staticmethod
    def _deterministic_final_response(memory: Memory) -> str:
        candidates, blocking_after_last_candidate = Planner._extract_final_aggregation_candidates(memory)
        unique_candidates = []
        for candidate, _ in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        if len(unique_candidates) == 1 and not blocking_after_last_candidate:
            candidate = unique_candidates[0]
            return (
                "The memory contains a unique Planner-marked final aggregation result. "
                "No additional derivation is performed.\n\n"
                f"\\boxed{{{candidate}}}"
            )

        reason = "no successful Planner-marked final aggregation"
        if len(unique_candidates) > 1:
            reason = "multiple conflicting final aggregation values"
        elif blocking_after_last_candidate:
            reason = "a blocking tool result appears after the last final aggregation"

        return (
            f"The tool memory is insufficient for deterministic final extraction: {reason}.\n\n"
            "\\boxed{INSUFFICIENT_TOOL_RESULTS}"
        )

    def _build_next_step_content(
        self,
        query: str,
        query_analysis: str,
        memory: Memory,
        step_count: int,
        max_step_count: int,
    ) -> str:
        """Build the user message content for the next_step turn (shared by generate_next_step and get_next_step_prompt_text)."""
        return f"""
Task: Determine the next atomic tool call for an agent workflow.
Context:
- **Query:** {query}
- **Planning Brief:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}
- **Step Index:** {step_count}
- **Max Steps:** {max_step_count}

Instructions:
1. Select exactly one available tool for the next atomic step.
2. Create one narrow Sub-Goal that the selected tool can complete in one call without choosing the overall solution strategy.
3. Provide all necessary **context**: explicit data already known, variables already introduced, prior local results to use, and the exact output expected from this call.
4. For a reasoning tool, request one local identity, relationship, transformation, or consistency check.
5. For a computation tool, request one explicit calculation, simplification, enumeration, or symbolic check with enough inputs and constraints that the tool does not need to infer the method.
6. When all necessary intermediate results are already in Previous Steps, request one narrow final aggregation from the computation tool. The Sub-Goal must say it is a final aggregation over recorded intermediate values and ask the tool to print only the requested final value.
7. If Previous Steps contains `verifier_conclusion: CONTINUE_FINAL_AGGREGATION_REQUIRED`, the next step must be that narrow Python final aggregation over recorded values.
8. If a previous tool result is broad, refused, contradictory, or unusable, create a smaller sub-goal instead of repeating the same request.

Response Format:
Justification: <one concise sentence about why this tool is next>
Context: <compact facts and previous results needed by the tool>
Sub-Goal: <one atomic objective for the tool>
Tool Name: <exactly one name from Available Tools>

Rules:
- Select only ONE tool.
- The Tool Name must exactly match one of the Available Tools.
- Do not ask any tool for a complete solution or broad strategy; the only allowed final-target request is a narrow final aggregation from recorded intermediate results.
- Do not copy the full original query into the Sub-Goal; include only the data needed for this local operation.
- Do not ask the tool to decide whether its result is globally final; only ask it to compute the local final aggregation you specify.
- Output only the four response-format lines above. No numbering, Markdown bullets, or extra sections.

"""



    async def generate_next_step(
        self,
        query: str,
        memory: Memory,
        query_analysis: str,
        step_count: int,
        max_step_count: int,
    ) -> GenerationOutput:
        content = self._build_next_step_content(query, query_analysis, memory, step_count, max_step_count)
        messages = [{"role": "user", "content": content}]
        return await self.llm_engine.generate(messages)

    async def generate_final_output(
        self, query_analysis: str, question: str, memory: Memory, llm_engine=None
    ) -> GenerationOutput:
        prompt_text = (
            "DETERMINISTIC_FINAL_OUTPUT_EXTRACTOR\n"
            "Rule: output a final answer only when Actions Taken contain exactly one "
            "successful Planner-marked final aggregation result, a Verifier STOP at or after it, "
            "and no later blocking tool result.\n"
            f"Query: {question}\n"
            f"Planning Brief: {query_analysis}\n"
            f"Actions Taken: {memory.get_actions()}\n"
        )
        response = self._deterministic_final_response(memory)
        return GenerationOutput(
            prompt_text=prompt_text,
            prompt_token_ids=[],
            response=response,
            token_ids=[],
            log_probs=[],
            finish_reason="stop",
        )
