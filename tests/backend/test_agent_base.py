"""
Tests for agent_base.py
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.agent_base import AgentBase
from backend.core.ollama_client import OllamaClient


class ConcreteAgent(AgentBase):
    """Concrete implementation of AgentBase for testing."""

    def get_capabilities(self) -> list[str]:
        return ["test_capability", "another_capability"]

    def get_required_model_role(self) -> str:
        return "rag"

    async def _execute(self, task: dict) -> dict:
        if task.get("input") == "fail":
            raise ValueError("Intentional test failure")
        return {"success": True, "output": f"processed: {task.get('input')}"}


@pytest.fixture
def agent(tmp_path):
    """Create a ConcreteAgent with temporary memory directory."""
    mock_ollama = MagicMock(spec=OllamaClient)
    return ConcreteAgent(
        agent_id="test_agent",
        memory_dir=tmp_path / "memory",
        ollama_client=mock_ollama,
    )


# --- initialization ---

def test_agent_id(agent):
    assert agent.agent_id == "test_agent"


def test_agent_capabilities(agent):
    caps = agent.get_capabilities()
    assert "test_capability" in caps
    assert "another_capability" in caps


def test_agent_required_model_role(agent):
    assert agent.get_required_model_role() == "rag"


def test_agent_memory_initialized(agent):
    assert agent.memory is not None


def test_agent_ollama_initialized(agent):
    assert agent.ollama is not None


# --- run: success path ---

@pytest.mark.asyncio
async def test_run_success(agent):
    task = {"type": "test_task", "input": "hello"}
    result = await agent.run(task)
    assert result["success"] is True
    assert result["output"] == "processed: hello"


@pytest.mark.asyncio
async def test_run_registers_skill(agent):
    task = {"type": "test_task", "input": "hello"}
    await agent.run(task)
    skills = agent.memory.get_skills()
    assert any(s["name"] == "test_task" for s in skills)


@pytest.mark.asyncio
async def test_run_updates_skill_rate_on_success(agent):
    task = {"type": "test_task", "input": "hello"}
    await agent.run(task)
    skills = agent.memory.get_skills()
    skill = next(s for s in skills if s["name"] == "test_task")
    assert skill["success_rate"] == 1.0
    assert skill["total_runs"] == 1


@pytest.mark.asyncio
async def test_run_saves_training_sample_on_success(agent):
    task = {"type": "test_task", "input": "hello"}
    await agent.run(task)
    data = agent.memory._read_json(agent.memory.training_path)
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["agent_id"] == "test_agent"


# --- run: failure path ---

@pytest.mark.asyncio
async def test_run_failure_returns_error(agent):
    task = {"type": "test_task", "input": "fail"}
    result = await agent.run(task)
    assert result["success"] is False
    assert result["error"] == "Intentional test failure"
    assert result["output"] is None


@pytest.mark.asyncio
async def test_run_logs_error_on_failure(agent):
    task = {"type": "test_task", "input": "fail"}
    await agent.run(task)
    errors = agent.memory.get_errors()
    assert len(errors) == 1
    assert errors[0]["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_run_updates_skill_rate_on_failure(agent):
    task = {"type": "test_task", "input": "fail"}
    await agent.run(task)
    skills = agent.memory.get_skills()
    skill = next(s for s in skills if s["name"] == "test_task")
    assert skill["success_rate"] == 0.0


@pytest.mark.asyncio
async def test_run_no_training_sample_on_failure(agent):
    task = {"type": "test_task", "input": "fail"}
    await agent.run(task)
    data = agent.memory._read_json(agent.memory.training_path)
    assert len(data["candidates"]) == 0


# --- pre-task error scanning ---

@pytest.mark.asyncio
async def test_run_injects_known_issues(agent):
    agent.memory.log_error(
        error_type="FileNotFoundError",
        context="test_task hello world",
        solution="Use absolute path",
    )
    task = {"type": "test_task", "input": "hello world"}
    result = await agent.run(task)
    assert result["success"] is True


# --- get_agent_info ---

def test_get_agent_info(agent):
    info = agent.get_agent_info()
    assert info["id"] == "test_agent"
    assert "test_capability" in info["capabilities"]
    assert info["required_model_role"] == "rag"
    assert isinstance(info["skills"], list)
    assert isinstance(info["error_count"], int)


# --- generate_training_sample ---

@pytest.mark.asyncio
async def test_generate_training_sample(agent):
    task = {"type": "summarize", "input": "some text"}
    result = {"success": True, "output": "summary"}
    sample = agent.generate_training_sample(task=task, result=result)
    assert sample["instruction"] == "summarize"
    assert sample["agent_id"] == "test_agent"
    assert sample["approved"] is False
