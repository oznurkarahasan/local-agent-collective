"""
orchestrator.py

Plans and distributes tasks across agents.
Uses Gemma 4 (CEO) for task analysis, planning, and final reporting.
Executes steps in parallel or sequentially based on dependencies.

Model Strategy:
- Gemma 4 E4B  → orchestration / planning / final report  (keep_alive="0")
- nomic-embed  → embedding, stays warm                     (keep_alive="-1")
- Other agents → their own preferred_model_roles           (keep_alive="0")
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

from backend.core.agent_registry import AgentRegistry
from backend.core.model_registry import ModelRegistry
from backend.core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Models that should stay loaded in VRAM permanently (too small/too frequent to reload)
ALWAYS_WARM_ROLES = {"embedding"}


class OrchestratorError(Exception):
    """Raised when orchestration fails."""

    pass

def _extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from model output.

    Handles:
    - DeepSeek R1 <think>...</think> blocks
    - ```json ... ``` markdown fences
    - Trailing commentary after the closing brace
    - Nested braces (finds the outermost balanced pair)

    Args:
        text: Raw model response string.

    Returns:
        Parsed dict, or None if extraction fails.
    """
    if not text:
        return None

    # 1. Strip <think>...</think> blocks (DeepSeek R1, QwQ style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Strip ```json ... ``` or ``` ... ``` markdown fences
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text)

    # 3. Find the outermost balanced { } pair
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed after extraction: %s", exc)
        return None


class Orchestrator:
    """
    Plans and distributes tasks across agents.

    Flow:
    1. Receive user task
    2. Send to Gemma 4 (CEO) → get execution plan (JSON)
    3. Resolve step dependencies
    4. Run independent steps in parallel (Semaphore controlled)
    5. Run dependent steps sequentially
    6. Collect results → send to Gemma 4 for final report

    Keep-Alive Strategy:
    - Orchestration model (Gemma 4): keep_alive="0"  → unload after each use
    - Embedding model (nomic-embed): keep_alive="-1" → stay loaded permanently
    - All other models:              keep_alive="0"  → unload after each use
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

        config_dir = config_dir or (Path(__file__).parent.parent.parent / "config")
        self.model_registry = ModelRegistry(config_path=config_dir / "models.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    async def initialize(self):
        """
        Explicitly warm up models before first run.
        Ensures embedding model is ready in VRAM to avoid cold-start delays.
        """
        logger.info("Initializing Orchestrator: warming up system models...")
        await self._warm_up_models()

    async def run(self, user_input: str) -> dict:
        """
        Main entry point. Receives user input and returns final report.

        Args:
            user_input: Raw user task description

        Returns:
            Dict with 'success', 'plan', 'results', and 'report' fields.
        """
        plan = await self._plan(user_input)

        if plan is None:
            return {
                "success": False,
                "error": "Could not generate execution plan",
                "plan": None,
                "results": [],
                "report": None,
            }

        steps = plan.get("steps", [])
        
        if not steps:
            # Direct conversation or unrecognized task - skip agent execution
            results = []
            report = await self._report(user_input, plan, results)
        else:
            results = await self._execute_plan(plan)
            report = await self._report(user_input, plan, results)

        return {
            "success": True,
            "plan": plan,
            "results": results,
            "report": report,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _warm_up_models(self) -> None:
        """
        Pre-load always-warm models into VRAM.

        Called once at startup. Failures are logged but do not crash the
        orchestrator — cold-start is still better than no start.
        """
        for role in ALWAYS_WARM_ROLES:
            try:
                model = self.model_registry.get_model_by_role(role)
                if model:
                    # Send an empty embed to load the model; keep_alive="-1" keeps it hot
                    await self.ollama.embed(
                        model=model["id"],
                        text="warmup",
                        keep_alive="-1",
                    )
                    logger.info("Warmed up model '%s' (role: %s)", model["id"], role)
            except Exception as exc:
                logger.warning("Could not warm up role '%s': %s", role, exc)

    async def _plan(self, user_input: str) -> dict:
        """
        Send user input to Gemma 4 (CEO) and get an execution plan.

        Returns:
            Plan dict with 'steps' list, or empty dict on failure.
        """
        # CEO / orchestration model = Gemma 4 E4B
        orchestration_model = self.model_registry.get_model_by_role("orchestration")
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
            "You are a task orchestrator (CEO). "
            "Analyze the user's request and create an execution plan. "
            "Return ONLY a valid JSON object — no markdown, no explanation, no <think> tags.\n"
            "Required structure:\n"
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
            "}\n\n"
            "Rules:\n"
            "- depends_on lists step ids that must complete before this step runs\n"
            "- Steps with empty depends_on can run in parallel\n"
            "- Use only agents from the available agents list\n"
            "- Output raw JSON only — no backticks, no prose\n"
            "- If the user is just having a casual conversation, asking a general question, or no agents apply, return an empty steps list (`\"steps\": []`)"
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
                keep_alive="0",  # Gemma 4 unloads after planning is done
            )

            plan = _extract_json(response)
            if plan is None:
                logger.warning("_plan: could not extract JSON from response:\n%s", response[:500])
                return {"steps": []}

            return plan

        except Exception as exc:
            logger.error("_plan failed: %s", exc)
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
                # Dependency cycle or unknown dep — bail out gracefully
                logger.error(
                    "_execute_plan: deadlock detected. Remaining steps: %s",
                    [s["id"] for s in remaining],
                )
                break

            tasks = [self._execute_step(step, completed) for step in ready]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(ready, batch_results):
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

    async def _report(self, user_input: str, plan: dict, results: list[dict]) -> str:
        """
        Synthesize all results into a final report via Gemma 4 (CEO).
        """
        orchestration_model = self.model_registry.get_model_by_role("orchestration")
        model_id = orchestration_model["id"]

        results_summary = json.dumps(results, indent=2, ensure_ascii=False)

        system_content = (
            "You are a helpful assistant (CEO) of Indis.ai local agent collective. "
            "If agent results are provided, synthesize them into a clear report. "
            "If no agent results are provided, answer the user normally like a chatbot. "
            "Write in plain text — no JSON, no markdown headers."
        )

        if not results:
            user_content = f"User: {user_input}\n\nAnswer:"
        else:
            user_content = (
                f"Original request: {user_input}\n\n"
                f"Agent results:\n{results_summary}\n\n"
                "Write a final report synthesizing these results."
            )

        try:
            report = await self.ollama.chat(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                keep_alive="0",  # Unload after report is written
            )
            return report

        except Exception as exc:
            logger.error("_report failed: %s", exc)
            return f"Report generation failed: {exc}"