"""
qa_agent/agent.py

QA Agent — handles logical reasoning and validation questions without relying on a vector DB.
Uses deepseek-r1:1.5b for reasoning.
"""

import json
from pathlib import Path
from typing import Optional

from backend.core.agent_base import AgentBase
from backend.core.ollama_client import OllamaClient
from backend.core.model_registry import ModelRegistry


class QaAgent(AgentBase):
    """
    QA Agent — logical analysis and reasoning.
    """

    def __init__(
        self,
        agent_id: str = "qa_agent",
        memory_dir: Optional[Path] = None,
        ollama_client: Optional[OllamaClient] = None,
        config_path: Optional[Path] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        if memory_dir is None:
            memory_dir = Path(__file__).parent / "memory"

        super().__init__(
            agent_id=agent_id,
            memory_dir=memory_dir,
            ollama_client=ollama_client,
            model_registry=model_registry,
        )

        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        self.config = self._load_config(config_path)

        self.chat_model = self.get_model_id_for_role(
            "reasoning", default="deepseek-r1:1.5b"
        )

    def _load_config(self, config_path: Path) -> dict:
        """Load agent configuration."""
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_capabilities(self) -> list[str]:
        return self.config.get(
            "capabilities",
            ["qa", "reasoning", "quality_control", "validation"],
        )

    def get_required_model_role(self) -> str:
        return "reasoning"

    async def _execute(self, task: dict) -> dict:
        """
        Execute QA tasks. Supports any query or reasoning request.
        """
        task_input = task.get("input", "")

        if not str(task_input).strip():
            return {
                "success": False,
                "output": None,
                "error": "Input cannot be empty for QA tasks",
            }

        answer = await self.generate_answer(task_input)

        return {
            "success": True,
            "output": {
                "answer": answer,
                "task_type": task.get("type", "qa"),
            },
        }

    async def generate_answer(self, user_input: str) -> str:
        """Generate response via Ollama."""
        system_prompt = self.config.get(
            "system_prompt",
            "You are a helpful reasoning logic assistant.",
        )

        return await self.ollama.chat(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_input)},
            ],
            keep_alive="0",
        )
