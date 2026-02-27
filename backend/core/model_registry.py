"""
model_registry.py

Manages model configurations loaded from config/models.json.
Provides model lookup by role or ID, and cross-references with
available models reported by the Ollama service.
"""

import json
from pathlib import Path
from typing import Optional
from backend.core.ollama_client import OllamaClient


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found in registry."""

    pass


class ModelRegistry:
    """Registry for AI model configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the registry by loading models.json.

        Args:
            config_path: Path to models.json. Defaults to config/models.json
                         relative to project root.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "models.json"

        self.config_path = config_path
        self._models: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load model definitions from config file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._models = data.get("models", [])

    def get_model_by_role(self, role: str) -> dict:
        """
        Find the first model that supports the given role.

        Args:
            role: Role string e.g. 'rag', 'embedding', 'orchestration'

        Returns:
            Model config dict.

        Raises:
            ModelNotFoundError: If no model supports the given role.
        """
        for model in self._models:
            if role in model.get("roles", []):
                return model

        raise ModelNotFoundError(
            f"No model found for role '{role}'. "
            f"Available roles: {self._get_all_roles()}"
        )

    def get_model_by_id(self, model_id: str) -> dict:
        """
        Find a model by its exact ID.

        Args:
            model_id: Model identifier e.g. 'qwen3:4b'

        Returns:
            Model config dict.

        Raises:
            ModelNotFoundError: If model ID is not in registry.
        """
        for model in self._models:
            if model.get("id") == model_id:
                return model

        raise ModelNotFoundError(
            f"Model '{model_id}' not found in registry. "
            f"Registered models: {[m['id'] for m in self._models]}"
        )

    def list_registered_models(self) -> list[dict]:
        """
        Return all models defined in config/models.json.

        Returns:
            List of model config dicts.
        """
        return list(self._models)

    async def list_available_models(self) -> list[dict]:
        """
        Cross-reference registered models with models installed in Ollama.

        Returns:
            List of registered models that are also available in Ollama,
            each enriched with an 'available' boolean field.
        """
        client = OllamaClient()
        try:
            ollama_models = await client.list_models()
            ollama_ids = {m.get("name", "") for m in ollama_models}
        except Exception:
            ollama_ids = set()

        result = []
        for model in self._models:
            model_copy = dict(model)
            model_copy["available"] = model["id"] in ollama_ids
            result.append(model_copy)

        return result

    def _get_all_roles(self) -> list[str]:
        """Return all unique roles across all registered models."""
        roles = set()
        for model in self._models:
            roles.update(model.get("roles", []))
        return sorted(roles)
