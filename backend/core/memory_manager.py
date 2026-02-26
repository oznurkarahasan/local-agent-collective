"""
memory_manager.py

Adaptive memory system for agents.
Each agent maintains two memory files:
- skills.json: tracks skill success rates
- errors.json: logs errors and their solutions
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class MemoryManager:
    """Manages adaptive memory (skills + errors) for a single agent."""

    def __init__(self, memory_dir: Path):
        """
        Initialize memory manager for an agent.

        Args:
            memory_dir: Path to agent's memory directory
                        e.g. agents/rag_agent/memory/
        """
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.skills_path = memory_dir / "skills.json"
        self.errors_path = memory_dir / "errors.json"
        self.training_path = memory_dir / "training_candidates.json"

        self._init_files()

    def _init_files(self) -> None:
        """Create memory files if they don't exist."""
        if not self.skills_path.exists():
            self._write_json(self.skills_path, {"skills": []})

        if not self.errors_path.exists():
            self._write_json(self.errors_path, {"errors": []})

        if not self.training_path.exists():
            self._write_json(self.training_path, {"candidates": []})

    # --- Skills ---

    def get_skills(self) -> list[dict]:
        """Return all skills."""
        return self._read_json(self.skills_path).get("skills", [])

    def add_skill(self, name: str, description: str) -> dict:
        """
        Register a new skill.

        Args:
            name: Skill name e.g. 'PDF reading'
            description: What this skill does

        Returns:
            Created skill dict.
        """
        data = self._read_json(self.skills_path)
        skill = {
            "id": f"sk_{uuid.uuid4().hex[:8]}",
            "name": name,
            "description": description,
            "success_rate": 1.0,
            "total_runs": 0,
            "last_used": self._now(),
        }
        data["skills"].append(skill)
        self._write_json(self.skills_path, data)
        return skill

    def update_skill_rate(self, skill_id: str, success: bool) -> Optional[dict]:
        """
        Update success rate for a skill after a run.

        Args:
            skill_id: Skill ID to update
            success: Whether the skill run succeeded

        Returns:
            Updated skill dict, or None if not found.
        """
        data = self._read_json(self.skills_path)
        for skill in data["skills"]:
            if skill["id"] == skill_id:
                total = skill.get("total_runs", 0) + 1
                current_rate = skill.get("success_rate", 1.0)
                new_rate = (
                    (current_rate * (total - 1)) + (1.0 if success else 0.0)
                ) / total
                skill["success_rate"] = round(new_rate, 4)
                skill["total_runs"] = total
                skill["last_used"] = self._now()
                self._write_json(self.skills_path, data)
                return skill
        return None

    # --- Errors ---

    def get_errors(self) -> list[dict]:
        """Return all logged errors."""
        return self._read_json(self.errors_path).get("errors", [])

    def log_error(
        self,
        error_type: str,
        context: str,
        solution: Optional[str] = None,
    ) -> dict:
        """
        Log a new error.

        Args:
            error_type: Exception type e.g. 'FileNotFoundError'
            context: What was happening when the error occurred
            solution: Known fix if available

        Returns:
            Created error dict.
        """
        data = self._read_json(self.errors_path)
        error = {
            "id": f"err_{uuid.uuid4().hex[:8]}",
            "error_type": error_type,
            "context": context,
            "solution": solution,
            "timestamp": self._now(),
            "resolved": solution is not None,
        }
        data["errors"].append(error)
        self._write_json(self.errors_path, data)
        return error

    def resolve_error(self, error_id: str, solution: str) -> Optional[dict]:
        """
        Mark an error as resolved with a solution.

        Args:
            error_id: Error ID to resolve
            solution: The fix that worked

        Returns:
            Updated error dict, or None if not found.
        """
        data = self._read_json(self.errors_path)
        for error in data["errors"]:
            if error["id"] == error_id:
                error["resolved"] = True
                error["solution"] = solution
                self._write_json(self.errors_path, data)
                return error
        return None

    def scan_errors(self, context: str) -> list[dict]:
        """
        Find previously resolved errors similar to the given context.
        Used before a task to apply known fixes proactively.

        Args:
            context: Description of the current task/situation

        Returns:
            List of resolved errors with matching keywords.
        """
        errors = self.get_errors()
        resolved = [e for e in errors if e.get("resolved") and e.get("solution")]

        context_words = set(context.lower().split())
        matches = []

        for error in resolved:
            error_words = set(error.get("context", "").lower().split())
            if context_words & error_words:
                matches.append(error)

        return matches

    # --- Training Candidates ---

    def save_training_sample(
        self,
        instruction: str,
        input_text: str,
        output: str,
        agent_id: str,
    ) -> dict:
        """
        Save a training candidate for future fine-tuning.

        Args:
            instruction: The task/goal
            input_text: Context provided to the agent
            output: Agent's response
            agent_id: Which agent produced this sample

        Returns:
            Created training sample dict.
        """
        data = self._read_json(self.training_path)
        sample = {
            "id": f"train_{uuid.uuid4().hex[:8]}",
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "agent_id": agent_id,
            "quality_score": None,
            "approved": False,
            "timestamp": self._now(),
        }
        data["candidates"].append(sample)
        self._write_json(self.training_path, data)
        return sample

    # --- Helpers ---

    def _read_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
