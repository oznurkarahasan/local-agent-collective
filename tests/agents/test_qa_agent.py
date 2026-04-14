"""
Tests for qa_agent/agent.py
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agents.qa_agent.agent import QaAgent
from backend.core.ollama_client import OllamaClient


@pytest.fixture
def mock_ollama():
    client = MagicMock(spec=OllamaClient)
    client.chat = AsyncMock(return_value="This is a reasoned answer.")
    return client


@pytest.fixture
def agent(tmp_path, mock_ollama):
    qa = QaAgent(
        memory_dir=tmp_path / "memory",
        ollama_client=mock_ollama,
    )
    return qa


def test_agent_id(agent):
    assert agent.agent_id == "qa_agent"


def test_get_capabilities(agent):
    caps = agent.get_capabilities()
    assert "qa" in caps
    assert "reasoning" in caps


def test_get_required_model_role(agent):
    assert agent.get_required_model_role() == "reasoning"


@pytest.mark.asyncio
async def test_execute_empty_input(agent):
    result = await agent._execute({"type": "qa", "input": "   "})
    assert result["success"] is False
    assert "empty" in result["error"]


@pytest.mark.asyncio
async def test_execute_success(agent):
    result = await agent._execute({
        "type": "qa",
        "input": "Why is the sky blue?",
    })
    assert result["success"] is True
    assert result["output"]["answer"] == "This is a reasoned answer."


@pytest.mark.asyncio
async def test_generate_answer(agent):
    answer = await agent.generate_answer("Test question")
    assert answer == "This is a reasoned answer."
    agent.ollama.chat.assert_called_once()
    assert agent.ollama.chat.call_args[1]["model"] == "deepseek-r1:1.5b"
