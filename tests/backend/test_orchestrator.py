"""
Tests for orchestrator.py
Matches Gemma 4 (CEO) and DeepSeek R1 (QA) orchestration logic.
"""

import json
import pytest
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.orchestrator import Orchestrator, _extract_json


@pytest.fixture
def mock_agents_dir(tmp_path):
    """Create a mock agents directory with a compliant test agent."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    dev_dir = agents_dir / "dev_agent"
    dev_dir.mkdir()
    (dev_dir / "memory").mkdir()

    config = {
        "id": "dev_agent",
        "name": "Dev Agent",
        "capabilities": ["code_generation"],
        "input_types": ["py", "js"],
        "enabled": True,
    }
    (dev_dir / "config.json").write_text(json.dumps(config))

    # Updated: Orchestrator now calls agent.run(task)
    agent_code = '''
from backend.core.agent_base import AgentBase
class DevAgent(AgentBase):
    async def run(self, task):
        return {"success": True, "output": f"Result for: {task.get('input')}"}
'''
    (dev_dir / "agent.py").write_text(agent_code)
    return agents_dir


@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a mock config with the updated model names."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    models = {
        "models": [
            {
                "id": "gemma4:e4b",
                "roles": ["orchestration", "planning", "ceo"],
                "size_gb": 9.6,
                "keep_alive": "0"
            },
            {
                "id": "nomic-embed-text:v1.5",
                "roles": ["embedding"],
                "size_gb": 0.27,
                "keep_alive": "-1"
            },
            {
                "id": "qwen2.5-coder:3b",
                "roles": ["dev"],
                "size_gb": 1.9,
                "keep_alive": "0"
            }
        ]
    }
    (config_dir / "models.json").write_text(json.dumps(models))
    return config_dir


@pytest.fixture
def orchestrator(mock_agents_dir, mock_config_dir):
    """Create an Orchestrator with mock directories and mock Ollama."""
    mock_ollama = MagicMock()
    mock_ollama.chat = AsyncMock()
    mock_ollama.embed = AsyncMock()
    
    return Orchestrator(
        agents_dir=mock_agents_dir,
        config_dir=mock_config_dir,
        max_concurrent=2,
        ollama_client=mock_ollama,
    )


# --- initialization ---

@pytest.mark.asyncio
async def test_orchestrator_initializes_and_warms_up(orchestrator):
    assert orchestrator is not None
    await orchestrator.initialize()
    assert orchestrator.ollama.embed.called


# --- _extract_json (New Logic) ---

def test_extract_json_handles_thinking_and_markdown():
    noisy_input = "<think>Logic...</think>"