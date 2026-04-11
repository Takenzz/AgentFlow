import json
import logging
import uuid
from pathlib import Path
from typing import Any

from .llm_engine import GenerationOutput
from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .formatters import extract_context_subgoal_and_tool
from .memory import Memory

logger = logging.getLogger(__name__)

class Solver:
    """
    engine_map example:
        {
            "default":  sglang_engine,   # required; used as fallback for unspecified modules
            "executor": qwen_engine,
            "verifier": qwen_engine,
        }
    "planner" and "final_output" automatically fall back to "default" when not specified.
    """

    MAX_TOTAL_TOKENS = 131072

    def __init__(
        self,
        engine_map: dict[str, Any],
        tools_dir: str = None,
        max_steps: int = 5,
        trajectory_dir: str = None,
    ):
        def _get(module: str):
            return engine_map.get(module) or engine_map["default"]

        self._engine = _get("default")   # Keep default engine for tokenizer access
        self._tools_dir = tools_dir
        self.max_steps = max_steps
        self._trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        if self._trajectory_dir:
            self._trajectory_dir.mkdir(parents=True, exist_ok=True)

        self._final_output_engine = _get("final_output")
        self.planner  = Planner(llm_engine=_get("planner"),  tools_dir=tools_dir)
        _module_keys = {"default", "planner", "executor", "verifier"}
        tools_engine_map = {k: v for k, v in engine_map.items() if k not in _module_keys}
        self.executor = Executor(
            llm_engine=_get("executor"),
            available_tools=self.planner.available_tools,
            tools_engine_map=tools_engine_map,
        )
        self.verifier = Verifier(llm_engine=_get("verifier"), available_tools=self.planner.available_tools, tools_metadata=self.planner.toolbox_metadata)

    def _save_trajectory(self, trajectory: dict) -> None:
        if self._trajectory_dir is None:
            return
        file_path = self._trajectory_dir / f"{uuid.uuid4().hex}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)
        logger.debug("trajectory saved to %s", file_path)

    async def solve(self, question: str, label: str | None = None) -> GenerationOutput:
        memory = Memory()
        trajectory = {
            "question": question,
            "label": label,
            "analysis": "",
            "steps": [],
            "final_output": "",
        }

        turns = []
        total_token_count = 0

        # ── Plan (independent training sequence #0) ──
        analysis = await self.planner.plan(question)
        turns.append({
            "tokens": list(analysis.prompt_token_ids) + list(analysis.token_ids),
            "response_length": len(analysis.token_ids),
            "loss_mask": [1] * len(analysis.token_ids),
            "rollout_log_probs": list(analysis.log_probs),
        })
        total_token_count += len(analysis.prompt_token_ids) + len(analysis.token_ids)
        response_text = analysis.response
        finish_reason = analysis.finish_reason
        trajectory["analysis"] = analysis.response

        for step_count in range(self.max_steps):
            try:
                # ── next_step (independent training sequences #1, #2, ...) ──
                next_step = await self.planner.generate_next_step(
                    question, memory, analysis.response,
                    step_count=step_count, max_step_count=self.max_steps,
                )
            except Exception as e:
                logger.warning("[step %d] planner.generate_next_step failed: %s", step_count, e)
                break

            turns.append({
                "tokens": list(next_step.prompt_token_ids) + list(next_step.token_ids),
                "response_length": len(next_step.token_ids),
                "loss_mask": [1] * len(next_step.token_ids),
                "rollout_log_probs": list(next_step.log_probs),
            })
            total_token_count += len(next_step.prompt_token_ids) + len(next_step.token_ids)
            response_text += (
                f"\n\n===== [next_step prompt] =====\n{next_step.prompt_text}"
                f"\n===== [next_step response] =====\n{next_step.response}"
            )
            finish_reason = next_step.finish_reason

            context, sub_goal, tool_name = extract_context_subgoal_and_tool(next_step.response)

            # ── Execute tool (not included in training sequences); result written to memory ──
            try:
                tool_command, _ = await self.executor.generate_tool_command(
                    question, context, sub_goal, tool_name,
                    self.planner.toolbox_metadata, step_count=step_count,
                )
            except Exception as e:
                logger.warning("[step %d] executor.generate_tool_command failed: %s", step_count, e)
                tool_command = f"Error generating command: {e}"

            execution_result = await self.executor.execute_command(tool_name, tool_command, self._tools_dir)
            memory.add_action(step_count, tool_name, sub_goal, tool_command, execution_result)
            logger.debug("[step %d] execution_result: %s", step_count, execution_result)

            response_text += (
                f"\n===== [executor] tool={tool_name} =====\n{tool_command}"
                f"\n===== [execution_result] =====\n{execution_result}"
            )

            # ── Verifier (not included in training sequences); used only to decide whether to continue ──
            try:
                _, conclusion, _ = await self.verifier.verificate_context(
                    question, analysis.response, step_count=step_count, memory=memory,
                )
            except Exception as e:
                logger.warning("[step %d] verifier failed: %s", step_count, e)
                conclusion = "STOP"

            logger.debug("[step %d] conclusion: %s", step_count, conclusion)
            response_text += f"\n===== [verifier] conclusion={conclusion} =====\n"

            trajectory["steps"].append({
                "step_count": step_count,
                "next_step": next_step.response,
                "tool_name": tool_name,
                "sub_goal": sub_goal,
                "tool_command": tool_command,
                "execution_result": execution_result,
                "conclusion": conclusion,
            })

            if conclusion == "STOP":
                break

            if total_token_count >= self.MAX_TOTAL_TOKENS:
                logger.info("token budget exhausted (%d >= %d), stopping early at step %d",
                            total_token_count, self.MAX_TOTAL_TOKENS, step_count)
                break

        # ── final_output (generated by the fixed final_output engine; not added to training turns) ──
        # Reward is still computed from final_output.response, but the base model's output does not contribute to loss.
        final_output_text = ""
        try:
            final_output = await self.planner.generate_final_output(
                analysis.response, question, memory, llm_engine=self._final_output_engine
            )
            final_output_text = final_output.response
            response_text += (
                f"\n\n===== [final_output prompt] =====\n{final_output.prompt_text}"
                f"\n===== [final_output response] =====\n{final_output.response}"
            )
            trajectory["final_output"] = final_output.response
        except Exception as e:
            logger.warning("planner.generate_final_output failed: %s", e)
            trajectory["final_output"] = ""
        trajectory["full_response"] = response_text

        self._save_trajectory(trajectory)

        # Build concatenated sequence for framework compatibility (reward, eval, status, etc. still need a complete sample)
        # The actual training data is unrolled by custom_convert via turns
        first = turns[0]
        first_prompt_len = len(first["tokens"]) - first["response_length"]
        prompt_token_ids = first["tokens"][:first_prompt_len]

        cat_token_ids = list(first["tokens"][first_prompt_len:])
        cat_loss_mask = list(first["loss_mask"])
        cat_log_probs = list(first["rollout_log_probs"])

        for t in turns[1:]:
            t_prompt_len = len(t["tokens"]) - t["response_length"]
            cat_token_ids += t["tokens"]
            cat_loss_mask += [0] * t_prompt_len + t["loss_mask"]
            cat_log_probs += [0.0] * t_prompt_len + t["rollout_log_probs"]

        return GenerationOutput(
            prompt_text=analysis.prompt_text,
            prompt_token_ids=prompt_token_ids,
            response=response_text,
            token_ids=cat_token_ids,
            log_probs=cat_log_probs,
            finish_reason=finish_reason,
            loss_mask=cat_loss_mask,
            final_output=final_output_text,
            turns=turns,
        )
