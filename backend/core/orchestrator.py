"""
orchestrator.py

Plans and distributes tasks across agents.
Uses DeepSeek for task analysis and planning,
then executes steps in parallel or sequentially
based on dependencies.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from backend.core.agent_registry import AgentRegistry
from backend.core.model_registry import ModelRegistry
from backend.core.ollama_client import OllamaClient


class OrchestratorError(Exception):
    """Raised when orchestration fails."""

    pass


class Orchestrator:
    """
    Plans and distributes tasks across agents.

    Flow:
    1. Receive user task
    2. Send to DeepSeek -> get execution plan (JSON)
    3. Resolve step dependencies
    4. Run independent steps in parallel (Semaphore controlled)
    5. Run dependent steps sequentially
    6. Collect results -> send to DeepSeek for final report
    """

    def __init__(
        self,
        agents_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None,
        max_concurrent: int = 2,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.ollama = ollama_client or OllamaClient()
        self.registry = AgentRegistry(agents_dir=agents_dir)
        self.semaphore = asyncio.Semaphore(max_concurrent)

        config_dir = config_dir or (
            Path(__file__).parent.parent.parent / "config"
        )
        self.model_registry = ModelRegistry(
            config_path=config_dir / "models.json"
        )

    async def run(self, user_input: str) -> dict:
        """
        Main entry point. Receives user input and returns final report.

        Args:
            user_input: Raw user task description

        Returns:
            Dict with 'success', 'plan', 'results', and 'report' fields.
        """
        plan = await self._plan(user_input)

        if not plan or not plan.get("steps"):
            return {
                "success": False,
                "error": "Could not generate execution plan",
                "plan": None,
                "results": [],
                "report": None,
            }

        results = await self._execute_plan(plan)
        report = await self._report(user_input, plan, results)

        return {
            "success": True,
            "plan": plan,
            "results": results,
            "report": report,
        }

    async def _plan(self, user_input: str) -> dict:
        """
        Send user input to DeepSeek and get an execution plan.

        Returns:
            Plan dict with 'steps' list.
        """
        orchestration_model = self.model_registry.get_model_by_role(
            "orchestration"
        )
        model_id = orchestration_model["id"]

        available_agents = self.registry.list_all()
        agents_description = json.dumps(
            [
                {
                    "id": a["id"],
                    "capabilities": a.get("capabilities", []),
                    "input_types": a.get("input_types", []),
                }
                for a in available_agents
            ],
            indent=2,
        )

        system_prompt = (
            "You are a task orchestrator. "
            "Analyze the user's request and create an execution plan. "
            "Return ONLY a valid JSON object with this exact structure:\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "agent": "agent_id",\n'
            '      "task_type": "task_type_string",\n'
            '      "input": "what this step should do",\n'
            '      "depends_on": []\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- depends_on contains step ids that must complete before this step\n"
            "- Steps with empty depends_on can run in parallel\n"
            "- Use only agents from the available agents list"
        )

        user_message = (
            f"Available agents:\n{agents_description}\n\n"
            f"User request: {user_input}\n\n"
            "Create an execution plan."
        )

        try:
            response = await self.ollama.chat(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                keep_alive="0",
            )

            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                return {"steps": []}

            plan = json.loads(response[json_start:json_end])
            return plan

        except Exception:
            return {"steps": []}

    async def _execute_plan(self, plan: dict) -> list[dict]:
        """
        Execute all steps respecting dependencies.

        Independent steps run in parallel (Semaphore controlled).
        Dependent steps wait for their dependencies.
        """
        steps = plan.get("steps", [])
        completed: dict[int, dict] = {}
        remaining = list(steps)

        while remaining:
            ready = [
                s
                for s in remaining
                if all(dep in completed for dep in s.get("depends_on", []))
            ]

            if not ready:
                break

            tasks = [
                self._execute_step(step, completed) for step in ready
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(ready, results):
                if isinstance(result, Exception):
                    completed[step["id"]] = {
                        "step_id": step["id"],
                        "success": False,
                        "error": str(result),
                        "output": None,
                    }
                else:
                    completed[step["id"]] = result
                remaining.remove(step)

        return list(completed.values())

    async def _execute_step(self, step: dict, context: dict) -> dict:
        """Execute a single plan step using the appropriate agent."""
        async with self.semaphore:
            agent_id = step.get("agent")
            agent_class = self.registry.get_class(agent_id)

            if agent_class is None:
                return {
                    "step_id": step["id"],
                    "success": False,
                    "error": f"Agent '{agent_id}' not found in registry",
                    "output": None,
                }

            agents_dir = self.registry.agents_dir
            memory_dir = agents_dir / agent_id / "memory"

            agent = agent_class(
                agent_id=agent_id,
                memory_dir=memory_dir,
                ollama_client=self.ollama,
            )

            task = {
                "type": step.get("task_type", "unknown"),
                "input": step.get("input", ""),
                "context": context,
            }

            result = await agent.run(task)
            result["step_id"] = step["id"]
            return result

    async def _report(
        self, user_input: str, plan: dict, results: list[dict]
    ) -> str:
        """Synthesize all results into a final report via DeepSeek."""
        orchestration_model = self.model_registry.get_model_by_role(
            "orchestration"
        )
        model_id = orchestration_model["id"]

        results_summary = json.dumps(results, indent=2, ensure_ascii=False)

        system_content = (
            "You are a helpful assistant. "
            "Synthesize the agent results into a clear, "
            "concise report for the user."
        )
        user_content = (
            f"Original request: {user_input}\n\n"
            f"Agent results:\n{results_summary}\n\n"
            "Write a final report."
        )

        try:
            report = await self.ollama.chat(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                keep_alive="0",
            )
            return report

        except Exception as e:
            return f"Report generation failed: {e}"
