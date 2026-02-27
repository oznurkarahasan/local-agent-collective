"""
Tests for orchestrator.py
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.orchestrator import Orchestrator, OrchestratorError


@pytest.fixture
def mock_agents_dir(tmp_path):
    """Create a mock agents directory with one test agent."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    rag_dir = agents_dir / "rag_agent"
    rag_dir.mkdir()
    (rag_dir / "memory").mkdir()

    config = {
        "id": "rag_agent",
        "name": "RAG Agent",
        "capabilities": ["document_qa"],
        "input_types": ["pdf", "txt"],
        "enabled": True,
    }
    (rag_dir / "config.json").write_text(json.dumps(config))

    agent_code = '''
from backend.core.agent_base import AgentBase

class RagAgent(AgentBase):
    def get_capabilities(self):
        return ["document_qa"]

    def get_required_model_role(self):
        return "rag"

    async def _execute(self, task):
        return {"success": True, "output": f"RAG result for: {task.get('input')}"}
'''
    (rag_dir / "agent.py").write_text(agent_code)
    return agents_dir


@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a mock config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    models = {
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
                "roles": ["rag", "reasoning"],
                "context_window": 32768,
                "languages": ["en", "tr"],
                "size_gb": 2.5,
                "keep_alive": "0",
            },
        ]
    }
    (config_dir / "models.json").write_text(json.dumps(models))
    return config_dir


@pytest.fixture
def orchestrator(mock_agents_dir, mock_config_dir):
    """Create an Orchestrator with mock directories."""
    mock_ollama = MagicMock()
    mock_ollama.chat = AsyncMock()
    return Orchestrator(
        agents_dir=mock_agents_dir,
        config_dir=mock_config_dir,
        max_concurrent=2,
        ollama_client=mock_ollama,
    )


# --- initialization ---

def test_orchestrator_initializes(orchestrator):
    assert orchestrator is not None
    assert orchestrator.semaphore is not None


def test_orchestrator_has_registry(orchestrator):
    assert orchestrator.registry is not None


def test_orchestrator_has_model_registry(orchestrator):
    assert orchestrator.model_registry is not None


# --- _plan ---

@pytest.mark.asyncio
async def test_plan_returns_valid_structure(orchestrator):
    plan_json = json.dumps({
        "steps": [
            {
                "id": 1,
                "agent": "rag_agent",
                "task_type": "document_qa",
                "input": "summarize the document",
                "depends_on": [],
            }
        ]
    })
    orchestrator.ollama.chat = AsyncMock(return_value=plan_json)
    plan = await orchestrator._plan("summarize my document")
    assert "steps" in plan
    assert len(plan["steps"]) == 1


@pytest.mark.asyncio
async def test_plan_handles_invalid_json(orchestrator):
    orchestrator.ollama.chat = AsyncMock(return_value="not valid json")
    plan = await orchestrator._plan("do something")
    assert plan == {"steps": []}


@pytest.mark.asyncio
async def test_plan_handles_ollama_error(orchestrator):
    orchestrator.ollama.chat = AsyncMock(side_effect=Exception("connection error"))
    plan = await orchestrator._plan("do something")
    assert plan == {"steps": []}


# --- _execute_plan ---

@pytest.mark.asyncio
async def test_execute_plan_single_step(orchestrator):
    plan = {
        "steps": [
            {
                "id": 1,
                "agent": "rag_agent",
                "task_type": "document_qa",
                "input": "test input",
                "depends_on": [],
            }
        ]
    }
    results = await orchestrator._execute_plan(plan)
    assert len(results) == 1
    assert results[0]["success"] is True


@pytest.mark.asyncio
async def test_execute_plan_unknown_agent(orchestrator):
    plan = {
        "steps": [
            {
                "id": 1,
                "agent": "nonexistent_agent",
                "task_type": "something",
                "input": "test",
                "depends_on": [],
            }
        ]
    }
    results = await orchestrator._execute_plan(plan)
    assert len(results) == 1
    assert results[0]["success"] is False
    assert "not found" in results[0]["error"]


@pytest.mark.asyncio
async def test_execute_plan_respects_dependencies(orchestrator):
    plan = {
        "steps": [
            {
                "id": 1,
                "agent": "rag_agent",
                "task_type": "document_qa",
                "input": "first step",
                "depends_on": [],
            },
            {
                "id": 2,
                "agent": "rag_agent",
                "task_type": "document_qa",
                "input": "second step",
                "depends_on": [1],
            },
        ]
    }
    results = await orchestrator._execute_plan(plan)
    assert len(results) == 2
    assert all(r["success"] for r in results)


@pytest.mark.asyncio
async def test_execute_plan_parallel_steps(orchestrator):
    plan = {
        "steps": [
            {
                "id": 1,
                "agent": "rag_agent",
                "task_type": "document_qa",
                "input": "parallel step 1",
                "depends_on": [],
            },
            {
                "id": 2,
                "agent": "rag_agent",
                "task_type": "document_qa",
                "input": "parallel step 2",
                "depends_on": [],
            },
        ]
    }
    results = await orchestrator._execute_plan(plan)
    assert len(results) == 2
    assert all(r["success"] for r in results)


# --- _report ---

@pytest.mark.asyncio
async def test_report_returns_string(orchestrator):
    orchestrator.ollama.chat = AsyncMock(return_value="Final report content")
    report = await orchestrator._report(
        user_input="test request",
        plan={"steps": []},
        results=[{"success": True, "output": "done"}],
    )
    assert isinstance(report, str)
    assert report == "Final report content"


@pytest.mark.asyncio
async def test_report_handles_error(orchestrator):
    orchestrator.ollama.chat = AsyncMock(side_effect=Exception("model error"))
    report = await orchestrator._report(
        user_input="test",
        plan={},
        results=[],
    )
    assert "failed" in report.lower()


# --- run ---

@pytest.mark.asyncio
async def test_run_empty_plan(orchestrator):
    orchestrator.ollama.chat = AsyncMock(return_value="no json here")
    result = await orchestrator.run("do something")
    assert result["success"] is False
    assert result["plan"] is None


@pytest.mark.asyncio
async def test_run_full_flow(orchestrator):
    plan_json = json.dumps({
        "steps": [
            {
                "id": 1,
                "agent": "rag_agent",
                "task_type": "document_qa",
                "input": "analyze document",
                "depends_on": [],
            }
        ]
    })

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return plan_json
        return "Final report: task completed successfully"

    orchestrator.ollama.chat = mock_chat
    result = await orchestrator.run("analyze my document")
    assert result["success"] is True
    assert result["report"] is not None
    assert len(result["results"]) == 1
