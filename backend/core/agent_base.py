"""
agent_base.py

Abstract base class for all agents in the system.
Every agent must inherit from AgentBase and implement
the required abstract methods.
"""

import json
from abc import ABC, abstractmethod

from pathlib import Path
from typing import Optional

from backend.core.memory_manager import MemoryManager
from backend.core.ollama_client import OllamaClient
from backend.core.model_registry import ModelRegistry


class AgentBase(ABC):
    """
    Abstract base class that all agents must inherit from.

    Provides:
    - ollama_client for model communication
    - memory_manager for adaptive memory
    - pre-task error scanning
    - post-task skill rate update
    - training sample generation
    """

    def __init__(
        self,
        agent_id: str,
        memory_dir: Path,
        ollama_client: Optional[OllamaClient] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent e.g. 'rag_agent'
            memory_dir: Path to agent's memory directory
            ollama_client: OllamaClient instance. Creates a new one if not provided.
        """
        self.agent_id = agent_id
        self.memory = MemoryManager(memory_dir=memory_dir)
        self.ollama = ollama_client or OllamaClient()
        self.model_registry = model_registry

    # --- Abstract methods (must be implemented by each agent) ---

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """
        Return list of capabilities this agent supports.

        Example: ['document_qa', 'pdf_analysis', 'multilingual']
        """

    @abstractmethod
    def get_required_model_role(self) -> str:
        """
        Return the model role this agent requires.

        Example: 'rag', 'code_analysis', 'orchestration'
        """

    @abstractmethod
    async def _execute(self, task: dict) -> dict:
        """
        Core task execution logic. Implemented by each agent.

        Args:
            task: Task dict with at minimum a 'type' and 'input' field

        Returns:
            Result dict with at minimum a 'success' and 'output' field
        """

    # --- Main entry point ---

    async def run(self, task: dict) -> dict:
        """
        Run a task with full memory integration.

        Flow:
        1. Scan errors for known issues related to this task
        2. Execute the task
        3. Update skill rates based on outcome
        4. Save training sample
        5. Log any errors that occurred

        Args:
            task: Task dict

        Returns:
            Result dict
        """
        task_context = task.get("type", "") + " " + str(task.get("input", ""))

        # Step 1: Pre-task — scan for known errors
        known_errors = self.memory.scan_errors(task_context)
        if known_errors:
            task["known_issues"] = [
                {"error": e["error_type"], "solution": e["solution"]}
                for e in known_errors
            ]

        # Step 2: Execute
        result = {"success": False, "output": None, "error": None}
        try:
            result = await self._execute(task)
            result.setdefault("success", True)

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["output"] = None

            # Log the error
            self.memory.log_error(
                error_type=type(e).__name__,
                context=task_context,
                solution=None,
            )

        # Step 3: Post-task — update skill rates
        skill_name = task.get("type", "unknown_task")
        skills = self.memory.get_skills()
        skill = next((s for s in skills if s["name"] == skill_name), None)

        if skill is None:
            skill = self.memory.add_skill(
                name=skill_name,
                description=f"Auto-registered skill for task type: {skill_name}",
            )

        self.memory.update_skill_rate(skill["id"], success=result["success"])

        # Step 4: Save training sample
        if result["success"] and result.get("output"):
            self.generate_training_sample(
                task=task,
                result=result,
            )

        return result

    # --- Training sample generation ---

    def generate_training_sample(self, task: dict, result: dict) -> dict:
        """
        Save a completed task as a training candidate for future fine-tuning.

        Args:
            task: The task that was executed
            result: The result produced

        Returns:
            Created training sample dict.
        """
        return self.memory.save_training_sample(
            instruction=task.get("type", "unknown"),
            input_text=json.dumps(task.get("input", ""), ensure_ascii=False),
            output=json.dumps(result.get("output", ""), ensure_ascii=False),
            agent_id=self.agent_id,
        )

    # --- Utility ---

    def get_agent_info(self) -> dict:
        """Return basic info about this agent."""
        return {
            "id": self.agent_id,
            "capabilities": self.get_capabilities(),
            "required_model_role": self.get_required_model_role(),
            "skills": self.memory.get_skills(),
            "error_count": len(self.memory.get_errors()),
        }

    def get_model_id_for_role(self, role: str, default: Optional[str] = None) -> str:
        """
        Resolve a model ID for a given role via the model registry.

        Args:
            role: The role to look up e.g. 'rag'
            default: Default model ID if role is not found

        Returns:
            Model ID string.
        """
        if self.model_registry:
            model_info = self.model_registry.get_model_by_role(role)
            if model_info:
                return model_info["id"]

        if default:
            return default

        raise ValueError(f"No model found for role '{role}' and no default provided.")
