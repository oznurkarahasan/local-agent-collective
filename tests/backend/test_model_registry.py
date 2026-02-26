"""
Tests for model_registry.py
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from backend.core.model_registry import ModelRegistry, ModelNotFoundError


@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary models.json for testing."""
    config = {
        "models": [
            {
                "id": "deepseek-r1:7b",
                "roles": ["orchestration", "planning", "reasoning"],
                "context_window": 131072,
                "languages": ["en", "tr"],
                "size_gb": 4.7,
                "keep_alive": "0",
            },
            {
                "id": "qwen3:4b",
                "roles": ["rag", "reasoning", "multilingual"],
                "context_window": 32768,
                "languages": ["en", "tr", "zh"],
                "size_gb": 2.5,
                "keep_alive": "0",
            },
            {
                "id": "nomic-embed-text:v1.5",
                "roles": ["embedding"],
                "context_window": 8192,
                "languages": ["multilingual"],
                "size_gb": 0.274,
                "keep_alive": "0",
            },
        ]
    }
    config_file = tmp_path / "models.json"
    config_file.write_text(json.dumps(config))
    return config_file


@pytest.fixture
def registry(sample_config):
    """Create a ModelRegistry instance with test config."""
    return ModelRegistry(config_path=sample_config)


# --- initialization ---

def test_registry_loads_models(registry):
    models = registry.list_registered_models()
    assert len(models) == 3


def test_registry_raises_if_config_missing():
    with pytest.raises(FileNotFoundError):
        ModelRegistry(config_path=Path("/nonexistent/models.json"))


# --- get_model_by_role ---

def test_get_model_by_role_success(registry):
    model = registry.get_model_by_role("embedding")
    assert model["id"] == "nomic-embed-text:v1.5"


def test_get_model_by_role_orchestration(registry):
    model = registry.get_model_by_role("orchestration")
    assert model["id"] == "deepseek-r1:7b"


def test_get_model_by_role_rag(registry):
    model = registry.get_model_by_role("rag")
    assert model["id"] == "qwen3:4b"


def test_get_model_by_role_not_found(registry):
    with pytest.raises(ModelNotFoundError):
        registry.get_model_by_role("nonexistent_role")


# --- get_model_by_id ---

def test_get_model_by_id_success(registry):
    model = registry.get_model_by_id("qwen3:4b")
    assert model["id"] == "qwen3:4b"
    assert "rag" in model["roles"]


def test_get_model_by_id_not_found(registry):
    with pytest.raises(ModelNotFoundError):
        registry.get_model_by_id("nonexistent:model")


# --- list_registered_models ---

def test_list_registered_models(registry):
    models = registry.list_registered_models()
    ids = [m["id"] for m in models]
    assert "deepseek-r1:7b" in ids
    assert "qwen3:4b" in ids
    assert "nomic-embed-text:v1.5" in ids


# --- list_available_models ---

@pytest.mark.asyncio
async def test_list_available_models_all_available(registry):
    mock_ollama_models = [
        {"name": "deepseek-r1:7b"},
        {"name": "qwen3:4b"},
        {"name": "nomic-embed-text:v1.5"},
    ]

    with patch(
        "backend.core.model_registry.OllamaClient.list_models",
        new=AsyncMock(return_value=mock_ollama_models),
    ):
        models = await registry.list_available_models()
        assert all(m["available"] for m in models)


@pytest.mark.asyncio
async def test_list_available_models_none_available(registry):
    with patch(
        "backend.core.model_registry.OllamaClient.list_models",
        new=AsyncMock(return_value=[]),
    ):
        models = await registry.list_available_models()
        assert all(not m["available"] for m in models)


@pytest.mark.asyncio
async def test_list_available_models_ollama_unreachable(registry):
    with patch(
        "backend.core.model_registry.OllamaClient.list_models",
        new=AsyncMock(side_effect=Exception("connection refused")),
    ):
        models = await registry.list_available_models()
        assert all(not m["available"] for m in models)
