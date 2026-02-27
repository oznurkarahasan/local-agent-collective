"""
agent_registry.py

Auto-discovers and manages all agents in the agents/ directory.
No hardcoded agent names — registry scans config.json files automatically.
Adding a new agent requires only a new folder with config.json and agent.py.
"""

import importlib.util
import json
from pathlib import Path
from typing import Optional

from backend.core.agent_base import AgentBase


class AgentRegistry:
    """Auto-discovers and manages all enabled agents."""

    def __init__(self, agents_dir: Optional[Path] = None):
        """
        Initialize registry and discover all agents.

        Args:
            agents_dir: Path to agents directory.
                        Defaults to agents/ relative to project root.
        """
        if agents_dir is None:
            agents_dir = Path(__file__).parent.parent.parent / "agents"

        self.agents_dir = agents_dir
        self._configs: dict[str, dict] = {}
        self._classes: dict[str, type] = {}

        self._discover()

    def _discover(self) -> None:
        """Scan agents directory and load all enabled agents."""
        if not self.agents_dir.exists():
            return

        for folder in sorted(self.agents_dir.iterdir()):
            if not folder.is_dir():
                continue

            config_path = folder / "config.json"
            agent_path = folder / "agent.py"

            if not config_path.exists():
                continue

            config = self._load_config(config_path)
            if not config.get("enabled", True):
                continue

            agent_id = config.get("id")
            if not agent_id:
                continue

            self._configs[agent_id] = config

            if agent_path.exists():
                agent_class = self._import_agent(agent_path, agent_id)
                if agent_class:
                    self._classes[agent_id] = agent_class

    def _load_config(self, config_path: Path) -> dict:
        """Load and parse agent config.json."""
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _import_agent(self, agent_path: Path, agent_id: str) -> Optional[type]:
        """
        Dynamically import agent.py and return the agent class.

        Looks for a class that inherits from AgentBase.
        """
        try:
            spec = importlib.util.spec_from_file_location(
                f"agents.{agent_id}.agent", agent_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, AgentBase)
                    and attr is not AgentBase
                ):
                    return attr

        except Exception:
            pass

        return None

    def find_by_capability(self, capability: str) -> list[dict]:
        """
        Find all enabled agents that support the given capability.

        Args:
            capability: Capability string e.g. 'document_qa', 'code_analysis'

        Returns:
            List of matching agent config dicts.
        """
        return [
            config
            for config in self._configs.values()
            if capability in config.get("capabilities", [])
        ]

    def find_by_input_type(self, file_type: str) -> list[dict]:
        """
        Find all enabled agents that can handle the given file type.

        Args:
            file_type: File extension e.g. 'pdf', 'py', 'txt'

        Returns:
            List of matching agent config dicts.
        """
        file_type = file_type.lstrip(".")
        return [
            config
            for config in self._configs.values()
            if file_type in config.get("input_types", [])
        ]

    def list_all(self) -> list[dict]:
        """Return all discovered and enabled agent configs."""
        return list(self._configs.values())

    def get_config(self, agent_id: str) -> Optional[dict]:
        """Return config for a specific agent by ID."""
        return self._configs.get(agent_id)

    def get_class(self, agent_id: str) -> Optional[type]:
        """Return the agent class for a specific agent ID."""
        return self._classes.get(agent_id)

    def is_registered(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        return agent_id in self._configs
