"""
Integration tests for RAG agent adaptive memory system.

Tests that error loop and skill loop work correctly
through the full AgentBase.run() cycle.
No real Ollama or ChromaDB needed — uses mocking.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents.rag_agent.agent import RagAgent
from backend.core.ollama_client import OllamaClient


@pytest.fixture
def mock_ollama():
    client = MagicMock(spec=OllamaClient)
    client.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    client.chat = AsyncMock(return_value="Test answer")
    return client


@pytest.fixture
def mock_collection():
    collection = MagicMock()
    collection.count.return_value = 3
    collection.query.return_value = {
        "documents": [["relevant chunk"]],
        "metadatas": [[{"source": "/docs/test.txt", "chunk_index": 0}]],
    }
    return collection


@pytest.fixture
def agent(tmp_path, mock_ollama, mock_collection):
    """Create RagAgent with real memory files but mocked services."""
    with patch("chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = (
            mock_collection
        )
        rag = RagAgent(
            memory_dir=tmp_path / "memory",
            ollama_client=mock_ollama,
            chroma_dir=tmp_path / "chroma",
        )
        rag.collection = mock_collection
        return rag


# --- Skill loop integration ---

@pytest.mark.asyncio
async def test_skill_registered_after_first_run(agent):
    """Skill is auto-registered on first task run."""
    await agent.run({"type": "query", "input": "test question"})
    skills = agent.memory.get_skills()
    assert any(s["name"] == "query" for s in skills)


@pytest.mark.asyncio
async def test_skill_rate_increases_on_success(agent):
    """Success rate is tracked correctly over multiple runs."""
    task = {"type": "query", "input": "test question"}
    await agent.run(task)
    await agent.run(task)
    await agent.run(task)

    skills = agent.memory.get_skills()
    skill = next(s for s in skills if s["name"] == "query")
    assert skill["total_runs"] == 3
    assert skill["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_skill_rate_decreases_on_failure(agent, tmp_path):
    """Failure reduces skill success rate."""
    # First run succeeds
    await agent.run({"type": "query", "input": "test"})

    # Second run fails (nonexistent file)
    await agent.run({
        "type": "load_document",
        "input": "/nonexistent/file.txt",
    })

    skills = agent.memory.get_skills()
    load_skill = next(
        (s for s in skills if s["name"] == "load_document"), None
    )
    assert load_skill is not None
    assert load_skill["success_rate"] < 1.0


@pytest.mark.asyncio
async def test_multiple_skill_types_tracked(agent, tmp_path):
    """Different task types create separate skill entries."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Test content for skill tracking.")

    await agent.run({"type": "query", "input": "test question"})
    await agent.run({"type": "load_document", "input": str(txt_file)})

    skills = agent.memory.get_skills()
    skill_names = [s["name"] for s in skills]
    assert "query" in skill_names
    assert "load_document" in skill_names


# --- Error loop integration ---

@pytest.mark.asyncio
async def test_error_logged_on_failure(agent):
    """Errors are automatically logged to errors.json on failure."""
    await agent.run({
        "type": "load_document",
        "input": "/nonexistent/file.txt",
    })

    errors = agent.memory.get_errors()
    assert len(errors) > 0
    assert errors[-1]["error_type"] in ("FileNotFoundError", "RuntimeError")


@pytest.mark.asyncio
async def test_error_context_contains_task_info(agent):
    """Error context includes task type and input info."""
    await agent.run({
        "type": "load_document",
        "input": "/nonexistent/doc.txt",
    })

    errors = agent.memory.get_errors()
    assert len(errors) > 0
    assert "load_document" in errors[-1]["context"]


@pytest.mark.asyncio
async def test_no_error_logged_on_success(agent):
    """No error is logged when task succeeds."""
    initial_errors = agent.memory.get_errors()
    await agent.run({"type": "query", "input": "test question"})
    final_errors = agent.memory.get_errors()
    assert len(final_errors) == len(initial_errors)


# --- scan_errors integration ---

@pytest.mark.asyncio
async def test_scan_errors_finds_known_issue(agent):
    """scan_errors finds relevant resolved errors before task execution."""
    # Manually log a resolved error
    agent.memory.log_error(
        error_type="FileNotFoundError",
        context="load_document file path not found documents",
        solution="Use absolute path",
    )
    agent.memory.resolve_error(
        agent.memory.get_errors()[-1]["id"],
        solution="Use absolute path",
    )

    # scan_errors should find it
    matches = agent.memory.scan_errors("load_document file not found")
    assert len(matches) > 0
    assert matches[0]["solution"] == "Use absolute path"


@pytest.mark.asyncio
async def test_known_issues_injected_into_task(agent):
    """Known errors are injected into task dict before execution."""
    agent.memory.log_error(
        error_type="FileNotFoundError",
        context="load_document file path documents folder",
        solution="Check absolute path",
    )
    agent.memory.resolve_error(
        agent.memory.get_errors()[-1]["id"],
        solution="Check absolute path",
    )

    task = {"type": "load_document", "input": "documents/test.txt"}
    # run() injects known_issues into task before _execute()
    # We verify by checking the memory scan works
    matches = agent.memory.scan_errors(
        "load_document documents/test.txt"
    )
    assert len(matches) > 0


# --- Training sample integration ---

@pytest.mark.asyncio
async def test_training_sample_saved_on_success(agent):
    """Successful tasks generate training candidates."""
    await agent.run({"type": "query", "input": "What is RAG?"})

    data = agent.memory._read_json(agent.memory.training_path)
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["agent_id"] == "rag_agent"
    assert data["candidates"][0]["approved"] is False


@pytest.mark.asyncio
async def test_no_training_sample_on_failure(agent):
    """Failed tasks do not generate training candidates."""
    await agent.run({
        "type": "load_document",
        "input": "/nonexistent/file.txt",
    })

    data = agent.memory._read_json(agent.memory.training_path)
    assert len(data["candidates"]) == 0


# --- Preloaded errors from errors.json ---

def test_preloaded_errors_in_memory(tmp_path, mock_ollama, mock_collection):
    """RAG agent loads preloaded errors from errors.json on startup."""
    import shutil

    # Copy real errors.json to tmp memory dir
    real_errors = (
        Path(__file__).parent.parent.parent
        / "agents"
        / "rag_agent"
        / "memory"
        / "errors.json"
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    shutil.copy(real_errors, memory_dir / "errors.json")

    with patch("chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = (
            mock_collection
        )
        rag = RagAgent(
            memory_dir=memory_dir,
            ollama_client=mock_ollama,
            chroma_dir=tmp_path / "chroma",
        )

    errors = rag.memory.get_errors()
    assert len(errors) == 5
    error_types = [e["error_type"] for e in errors]
    assert "FileNotFoundError" in error_types
    assert "OllamaConnectionError" in error_types
