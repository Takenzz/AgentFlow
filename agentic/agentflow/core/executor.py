import importlib.util
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

from .llm_engine import SGLangEngine

# Tool name mapping: Static fallback mapping (long external names to internal)
LOCAL_MATH_TOOL_NAME = "Local_Math_Deduction_Tool"
PYTHON_CODE_TOOL_NAME = "Python_Code_Generator_Tool"

TOOL_NAME_MAPPING_LONG = {
    LOCAL_MATH_TOOL_NAME: {
        "class_name": "Base_Generator_Tool",
        "dir_name": "base_generator"
    },
    PYTHON_CODE_TOOL_NAME: {
        "class_name": "Python_Coder_Tool",
        "dir_name": "python_coder"
    },
}

TOOL_NAME_ALIASES = {
    "Base_Generator_Tool": LOCAL_MATH_TOOL_NAME,
    "base_generator": LOCAL_MATH_TOOL_NAME,
    "Generalist_Solution_Generator_Tool": LOCAL_MATH_TOOL_NAME,
    "Python_Coder_Tool": PYTHON_CODE_TOOL_NAME,
    "python_coder": PYTHON_CODE_TOOL_NAME,
}

class Executor:
    def __init__(
        self,
        llm_engine: SGLangEngine,
        available_tools: list[str],
        tools_engine_map: dict[str, Any] | None = None,
    ):
        self.llm_engine = llm_engine
        # key can be TOOL_NAME (external name), class_name, or dir_name; falls back to llm_engine if not found
        self.tools_engine_map: dict[str, Any] = tools_engine_map or {}
        self.available_tools = available_tools

    def _extract_command(self, response: Any) -> str:
        def normalize_code(code: str) -> str:
            normalized = code.strip()
            normalized = re.sub(r"<\|im_end\|>\s*$", "", normalized).strip()
            normalized = re.sub(r"^```python\s*", "", normalized, flags=re.IGNORECASE)
            normalized = re.sub(r"^```", "", normalized).strip()
            normalized = re.sub(r"```$", "", normalized).strip()
            normalized = re.sub(r"^\([^)]*pid=\d+[^)]*\)\s*", "", normalized)
            return normalized.strip()

        if hasattr(response, "response"):
            response = response.response

        command = "No command found."
        if isinstance(response, str):
            text = response.strip()
            command_match = re.search(
                r"Generated Command:.*?```python\s*\n(.*?)```",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if command_match:
                command = command_match.group(1).strip()
            else:
                loose_matches = re.findall(
                    r"```python\s*\n(.*?)```",
                    text,
                    re.DOTALL | re.IGNORECASE,
                )
                if loose_matches:
                    command = max(loose_matches, key=lambda x: len(x.strip())).strip()
                else:
                    direct_match = re.search(
                        r"(execution\s*=\s*tool\.execute\(.*\))",
                        text,
                        re.DOTALL,
                    )
                    if direct_match:
                        command = direct_match.group(1).strip()
        else:
            command = "Invalid response type."

        return normalize_code(command)

    async def generate_tool_command(self, query: str, context: str, sub_goal: str, tool_name: str, tool_metadata: dict, step_count: int) -> str:
        mapping = self._resolve_tool_mapping(tool_name)
        if mapping["dir_name"] == "base_generator":
            effective_tool_name = LOCAL_MATH_TOOL_NAME
        elif mapping["dir_name"] == "python_coder":
            effective_tool_name = PYTHON_CODE_TOOL_NAME
        else:
            effective_tool_name = tool_name.strip().strip("`").strip()

        selected_metadata = {}
        if isinstance(tool_metadata, dict):
            selected_metadata = (
                tool_metadata.get(effective_tool_name)
                or tool_metadata.get(tool_name)
                or {}
            )

        prompt_generate_tool_command = f"""
Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {query}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {effective_tool_name}
- **Selected Tool Metadata:** {selected_metadata}
- **Relevant Data:** {context}

Instructions:
1.  Analyze the tool's required parameters from its metadata.
2.  Construct valid Python code that addresses the sub-goal using the provided context and data.
3.  The command must include at least one call to `tool.execute()`.
4.  Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5.  Please give the exact numbers and parameters should be used in the `tool.execute()` call.
6.  **IMPORTANT: `tool.execute()` only accepts a single keyword argument: `query`. Do NOT pass any other keyword arguments (e.g. no `context`, `data`, `input`, etc.).**
7.  **IMPORTANT: The `query` value MUST be a plain-language description of what to compute. Do NOT put Python code inside the query string.**
8.  **IMPORTANT: Always wrap the `query` value in triple double-quotes (`\"\"\"...\"\"\"`). This prevents syntax errors from special characters or apostrophes.**
9.  **IMPORTANT: The `query` value must be a narrow tool request derived from the Sub-Goal and Relevant Data. Do NOT copy the full original Query unless the Sub-Goal explicitly needs every part of it.**
10. **For `Local_Math_Deduction_Tool`, ask for exactly one local identity, theorem, relationship, or short derivation. Do NOT ask it to solve the full problem or produce the final answer.**
11. **For `Python_Code_Generator_Tool`, ask for exactly one explicit calculation, simplification, enumeration, or symbolic check. Include the inputs, constraints, and requested printed output. Do NOT ask it to choose the strategy, plan a proof, or solve the full problem.**
12. If the Sub-Goal is broad, preserve the narrowest executable part and omit any request for a final answer.

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Format Example for a reasoning-tool local request:
Generated Command:
```python
execution = tool.execute(query=\"\"\"State the one local relationship requested in the Sub-Goal, using only the supplied variables and assumptions.\"\"\")
```

Format Example for a computation-tool local request:
Generated Command:
```python
execution = tool.execute(query=\"\"\"Evaluate the explicit local calculation supplied in Relevant Data and print only the requested result.\"\"\")
```
"""
        messages = [{"role": "user", "content": prompt_generate_tool_command}]
        tool_cmd_out = await self.llm_engine.generate(messages)
        command = self._extract_command(tool_cmd_out.response)
        return command, tool_cmd_out

    def _parse_command_kwargs(self, command: str) -> dict:
        """Extract kwargs by executing the command against a fake tool object.

        Falls back to regex extraction when the command is not valid Python
        (e.g. the LLM embedded raw Python code with mismatched quotes inside
        the query string).
        """
        captured: dict = {}

        class _FakeTool:
            def execute(self_, **kwargs):  # noqa: N805
                captured.update(kwargs)

        # Level 1: exec the command directly (works for well-formed commands)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                exec(command, {"tool": _FakeTool()})  # noqa: S102
            if captured:
                return captured
        except Exception:
            pass

        # Level 2: regex extraction — try triple-quote delimiters first, then single/double
        for delim in ['"""', "'''", '"', "'"]:
            esc = re.escape(delim)
            m = re.search(
                rf'tool\.execute\(\s*query\s*=\s*{esc}(.*?){esc}\s*\)\s*$',
                command,
                re.DOTALL,
            )
            if m:
                return {"query": m.group(1)}

        # Level 3: greedy grab — take everything between query= and the last closing paren
        m = re.search(r'tool\.execute\(\s*query\s*=\s*([\s\S]+)\)\s*$', command)
        if m:
            val = m.group(1).strip()
            for delim in ['"""', "'''", '"', "'"]:
                if val.startswith(delim):
                    val = val[len(delim):]
                    if val.endswith(delim):
                        val = val[: -len(delim)]
                    break
            return {"query": val}

        raise ValueError(f"Cannot parse command: {command!r}")

    def _resolve_tool_mapping(self, tool_name: str) -> dict:
        """Resolve a tool name (TOOL_NAME / class_name / dir_name) to its mapping dict."""
        cleaned = tool_name.strip().strip("`").strip()

        # Default behavior is strict: the Planner must emit the exact public
        # tool name listed in Available Tools. Legacy aliases can be enabled
        # only when explicitly evaluating old checkpoints.
        if cleaned in TOOL_NAME_MAPPING_LONG:
            return TOOL_NAME_MAPPING_LONG[cleaned]

        allow_aliases = os.getenv("AGENTFLOW_ALLOW_TOOL_ALIASES", "false").lower() in {
            "1", "true", "yes", "y", "on",
        }
        if allow_aliases:
            canonical_name = TOOL_NAME_ALIASES.get(cleaned)
            if canonical_name and canonical_name in TOOL_NAME_MAPPING_LONG:
                return TOOL_NAME_MAPPING_LONG[canonical_name]

        valid_tools = sorted(TOOL_NAME_MAPPING_LONG.keys())
        raise ValueError(
            f"Unknown tool_name: {tool_name!r}. Valid tool names are: {valid_tools}. "
            "Set AGENTFLOW_ALLOW_TOOL_ALIASES=true only for legacy checkpoint evaluation."
        )

    def _load_tool(self, tool_name: str, tools_dir: str):
        """Dynamically load and instantiate a tool class given its external tool_name."""
        mapping = self._resolve_tool_mapping(tool_name)

        class_name = mapping["class_name"]
        dir_name = mapping["dir_name"]
        tool_file = Path(tools_dir) / dir_name / "tool.py"
        if not tool_file.exists():
            raise FileNotFoundError(f"Tool file not found: {tool_file}")

        module_key = f"_tool_{dir_name}"
        if module_key not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_key, tool_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module
            spec.loader.exec_module(module)
        else:
            module = sys.modules[module_key]

        cls = getattr(module, class_name)
        engine = self.tools_engine_map.get(dir_name, self.llm_engine)
        return cls(llm_engine=engine)

    async def execute_command(self, tool_name: str, command: str, tools_dir: str) -> Any:
        """Load the tool, parse the command kwargs, and call tool.execute(**kwargs).

        Never raises — returns an error string on failure so the solver can
        continue to the next step instead of crashing the entire rollout.
        """
        try:
            tool = self._load_tool(tool_name, tools_dir)
        except Exception as exc:
            return f"Tool load error ({tool_name}): {exc}"

        try:
            kwargs = self._parse_command_kwargs(command)
        except Exception as exc:
            return f"Command parse error: {exc}"

        try:
            execution = await tool.execute(**kwargs)
            return execution
        except Exception as exc:
            return f"Tool execution error ({tool_name}): {exc}"
