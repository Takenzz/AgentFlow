import importlib.util
import logging
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
6. If a previous tool result is broad, refused, contradictory, or unusable, create a smaller sub-goal instead of repeating the same request.

Response Format:
Justification: <one concise sentence about why this tool is next>
Context: <compact facts and previous results needed by the tool>
Sub-Goal: <one atomic objective for the tool>
Tool Name: <exactly one name from Available Tools>

Rules:
- Select only ONE tool.
- The Tool Name must exactly match one of the Available Tools.
- Do not ask any tool for the final answer, a complete solution, or a broad strategy.
- Do not copy the full original query into the Sub-Goal; include only the data needed for this local operation.
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
    ) -> str:
        prompt_generate_final_output = f"""
Task: Generate the final answer from the accumulated tool results only.

Context:
- **Query:** {question}
- **Planning Brief:** {query_analysis}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Use the original query only to understand what answer format is requested.
2. Use only final candidates, facts, formulas, computations, and intermediate values that explicitly appear in Actions Taken.
3. Do NOT introduce new derivations, unstated theorems, missing arithmetic, consistency repair, or independent problem solving at this final stage.
4. Do NOT correct a tool result using your own reasoning. If the Actions Taken are contradictory, incomplete, or require any new calculation, end with \\boxed{{INSUFFICIENT_TOOL_RESULTS}}.
5. If Actions Taken contain one reliable final candidate already supported by the recorded local results, produce a concise synthesis and end with that value enclosed in \\boxed{{}}.
"""
        messages = [{"role": "user", "content": prompt_generate_final_output}]
        engine = llm_engine if llm_engine is not None else self.llm_engine
        return await engine.generate(messages)
