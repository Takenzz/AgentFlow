from typing import Any


class Memory:

    def __init__(self):
        self.actions: dict[str, dict[str, Any]] = {}

    MAX_RESULT_CHARS = 4096

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        result_str = str(result) if result is not None else ""
        if len(result_str) > self.MAX_RESULT_CHARS:
            half = self.MAX_RESULT_CHARS // 2
            result_str = result_str[:half] + f"\n... [truncated {len(result_str) - self.MAX_RESULT_CHARS} chars] ...\n" + result_str[-half:]

        self.actions[f"Action Step {step_count}"] = {
            'tool_name': tool_name,
            'sub_goal': sub_goal,
            'command': command,
            'result': result_str,
        }

    def set_action_conclusion(self, step_count: int, conclusion: str) -> None:
        key = f"Action Step {step_count}"
        if key in self.actions:
            self.actions[key]["verifier_conclusion"] = str(conclusion)

    def get_actions(self) -> dict[str, dict[str, Any]]:
        return self.actions
