"""
Tau2 Subprocess Adapter
=======================
Manages a tau2 simulation subprocess and communicates with it via the
file-based protocol (input_N.json / output_N.json).

Usage in orchestra_solver for func_call tasks:
    adapter = Tau2Adapter(domain, task_path, output_file, ...)
    await adapter.start()
    while not adapter.is_done():
        inp = await adapter.wait_for_input()
        # ... generate model response ...
        await adapter.write_output(content, tool_calls)
    reward_info = adapter.get_reward_info()
    await adapter.cleanup()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TAU2_DIR = Path(__file__).resolve().parent / "tau2"
_TAU2_CLI = _TAU2_DIR / "cli.py"

POLL_INTERVAL_S = 0.5
MAX_WAIT_S = 600


class Tau2Adapter:
    """
    Manages one tau2 simulation subprocess for a single func_call task.

    The tau2 subprocess drives the simulation loop. It writes input_N.json
    when it needs the agent to respond, and reads output_N.json for the
    agent's response. When done, it writes a "done" file.
    """

    def __init__(
        self,
        domain: str,
        task_path: str,
        output_file: str,
        transfer_dir: str | None = None,
        user_llm: str = "qwen-turbo-latest",
        agent_llm: str = "train",
        max_steps: int = 30,
        use_model_tool: bool = True,
        task_id: str | None = None,
        num_tasks: int = 1,
    ):
        self.domain = domain
        self.task_path = task_path
        self.output_file = output_file
        self.user_llm = user_llm
        self.agent_llm = agent_llm
        self.max_steps = max_steps
        self.use_model_tool = use_model_tool
        self.task_id = task_id
        self.num_tasks = num_tasks

        if transfer_dir:
            self.transfer_dir = transfer_dir
            self._owns_transfer_dir = False
        else:
            self.transfer_dir = tempfile.mkdtemp(prefix="tau2_transfer_")
            self._owns_transfer_dir = True

        self._step_idx = 0
        self._process: Optional[subprocess.Popen] = None

    async def start(self) -> None:
        os.makedirs(self.transfer_dir, exist_ok=True)

        tau2_parent = str(_TAU2_DIR.parent)
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{tau2_parent}:{existing}" if existing else tau2_parent

        cmd = [
            "python", str(_TAU2_CLI),
            "--domain", self.domain,
            "--task_path", self.task_path,
            "--output_file", self.output_file,
            "--cur_transfer_dir", self.transfer_dir,
            "--agent-llm", self.agent_llm,
            "--user-llm", self.user_llm,
            "--max-steps", str(self.max_steps),
            "--max-concurrency", "1",
            "--num-tasks", str(self.num_tasks),
        ]
        if self.task_id:
            cmd.extend(["--task-ids", self.task_id])
        if self.use_model_tool:
            cmd.append("--use_model_tool")

        logger.info("[Tau2Adapter] Starting: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    async def wait_for_input(self) -> dict:
        """
        Wait for tau2 to write input_{step_idx}.json, return its contents.

        Returns dict with keys: messages, tools, original_messages, original_tools
        """
        input_path = os.path.join(self.transfer_dir, f"input_{self._step_idx}.json")
        waited = 0.0

        while not os.path.isfile(input_path):
            if self.is_done():
                return None

            if self._process and self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(
                    f"tau2 subprocess exited with code {self._process.returncode}: {stderr[:2000]}"
                )

            if waited >= MAX_WAIT_S:
                raise TimeoutError(
                    f"Waited {MAX_WAIT_S}s for {input_path}, tau2 may be stuck"
                )

            await asyncio.sleep(POLL_INTERVAL_S)
            waited += POLL_INTERVAL_S

        for _read_attempt in range(5):
            try:
                with open(input_path, "r") as f:
                    data = json.load(f)
                break
            except json.JSONDecodeError:
                await asyncio.sleep(0.3)
        else:
            with open(input_path, "r") as f:
                data = json.load(f)

        logger.info("[Tau2Adapter] Read input_%d.json (%d messages, %d tools)",
                     self._step_idx,
                     len(data.get("messages", [])),
                     len(data.get("tools", [])))
        return data

    async def write_output(
        self,
        content: str | None,
        tool_calls: list[dict] | None,
    ) -> None:
        """Write output_{step_idx}.json for tau2 to consume."""
        output_path = os.path.join(self.transfer_dir, f"output_{self._step_idx}.json")

        response = {
            "content": content,
            "tool_calls": tool_calls or [],
        }

        with open(output_path, "w") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        logger.info("[Tau2Adapter] Wrote output_%d.json", self._step_idx)
        self._step_idx += 1

    def is_done(self) -> bool:
        return os.path.isfile(os.path.join(self.transfer_dir, "done"))

    @property
    def step_count(self) -> int:
        return self._step_idx

    def get_reward_info(self) -> dict:
        """
        Read reward_info from tau2 output files after simulation is done.

        tau2's run.py saves per-task results as individual JSON files in a
        directory derived from output_file (strip the .json suffix). Each
        file contains a full simulation with a 'reward_info' field.
        """
        search_dirs = []
        base = self.output_file
        if base.endswith(".json"):
            search_dirs.append(base[:-5])
        search_dirs.append(os.path.dirname(base))
        if os.path.isfile(base):
            search_dirs.insert(0, "__file__:" + base)

        for candidate in search_dirs:
            if candidate.startswith("__file__:"):
                fpath = candidate[len("__file__:"):]
                try:
                    with open(fpath, "r") as f:
                        sim_data = json.load(f)
                    ri = self._extract_from_sim(sim_data)
                    if ri is not None:
                        return ri
                except (json.JSONDecodeError, OSError):
                    pass
                continue

            if not os.path.isdir(candidate):
                continue
            for fname in sorted(os.listdir(candidate)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(candidate, fname)
                try:
                    with open(fpath, "r") as f:
                        sim_data = json.load(f)
                    ri = self._extract_from_sim(sim_data)
                    if ri is not None:
                        return ri
                except (json.JSONDecodeError, OSError):
                    continue

        logger.warning("[Tau2Adapter] No reward_info found near %s", self.output_file)
        return {"reward": 0.0, "info": {"error": "no reward_info found"}}

    @staticmethod
    def _extract_from_sim(sim_data: dict) -> dict | None:
        if "reward_info" in sim_data and sim_data["reward_info"]:
            ri = sim_data["reward_info"]
            if isinstance(ri, dict):
                return ri
            return {"reward": ri, "info": {}}
        if "tau_run_error" in sim_data:
            return {"reward": 0.0, "info": {"error": sim_data["tau_run_error"]}}
        return None

    async def cleanup(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._process.send_signal(signal.SIGTERM)
                await asyncio.sleep(2)
                if self._process.poll() is None:
                    self._process.kill()
            except OSError:
                pass

        if self._owns_transfer_dir and os.path.isdir(self.transfer_dir):
            try:
                shutil.rmtree(self.transfer_dir)
            except OSError as e:
                logger.warning("[Tau2Adapter] Failed to cleanup %s: %s", self.transfer_dir, e)
