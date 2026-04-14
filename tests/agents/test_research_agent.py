"""
Tests for research_agent/agent.py
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agents.research_agent.agent import ResearchAgent
from backend.core.ollama_client import OllamaClient


@pytest.fixture
def mock_ollama():
    client = MagicMock(spec=OllamaClient)
    client.chat = AsyncMock(return_value="This is research output.")
    return client


@pytest.fixture
def agent(tmp_path, mock_ollama):
    research = ResearchAgent(
        memory_dir=tmp_path / "memory",
        ollama_client=mock_ollama,
    )
    return research


def test_agent_id(agent):
    assert agent.agent_id == "research_agent"


def test_get_capabilities(agent):
    caps = agent.get_capabilities()
    assert "research" in caps
    assert "analysis" in caps


def test_get_required_model_role(agent):
    assert agent.get_required_model_role() == "research"


@pytest.mark.asyncio
async def test_execute_empty_input(agent):
    result = await agent._execute({"type": "research", "input": ""})
    assert result["success"] is False
    assert "empty" in result["error"]


@pytest.mark.asyncio
async def test_execute_success(agent):
    result = await agent._execute({
        "type": "research",
        "input": "Explain quantum computing",
    })
    assert result["success"] is True
    assert result["output"]["answer"] == "This is research output."


@pytest.mark.asyncio
async def test_generate_answer(agent):
    answer = await agent.generate_answer("Research topic")
    assert answer == "This is research output."
    agent.ollama.chat.assert_called_once()
    assert agent.ollama.chat.call_args[1]["model"] == "phi3.5:latest"
