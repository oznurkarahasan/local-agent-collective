"""
Integration tests for coder agent adaptive memory system.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents.coder_agent.agent import CoderAgent
from backend.core.ollama_client import OllamaClient


@pytest.fixture
def mock_ollama():
    client = MagicMock(spec=OllamaClient)
    client.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    client.chat = AsyncMock(return_value="This function adds two numbers.")
    return client


@pytest.fixture
def mock_collection():
    collection = MagicMock()
    collection.count.return_value = 3
    collection.query.return_value = {
        "documents": [["def add(a, b):\n    return a + b"]],
        "metadatas": [
            [{"source": "/code/math.py", "chunk_index": 0, "language": "python"}]
        ],
    }
    return collection


@pytest.fixture
def agent(tmp_path, mock_ollama, mock_collection):
    with patch("chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = (
            mock_collection
        )
        coder = CoderAgent(
            memory_dir=tmp_path / "memory",
            ollama_client=mock_ollama,
            chroma_dir=tmp_path / "chroma",
        )
        coder.collection = mock_collection
        return coder


# --- Skill loop ---

@pytest.mark.asyncio
async def test_skill_registered_after_first_run(agent):
    await agent.run({"type": "query", "input": "test question"})
    skills = agent.memory.get_skills()
    assert any(s["name"] == "query" for s in skills)


@pytest.mark.asyncio
async def test_skill_rate_increases_on_success(agent):
    task = {"type": "query", "input": "test question"}
    await agent.run(task)
    await agent.run(task)
    await agent.run(task)
    skills = agent.memory.get_skills()
    skill = next(s for s in skills if s["name"] == "query")
    assert skill["total_runs"] == 3
    assert skill["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_skill_rate_decreases_on_failure(agent):
    await agent.run({"type": "query", "input": "test"})
    await agent.run({"type": "load_code", "input": "/nonexistent/file.py"})
    skills = agent.memory.get_skills()
    load_skill = next((s for s in skills if s["name"] == "load_code"), None)
    assert load_skill is not None
    assert load_skill["success_rate"] < 1.0


@pytest.mark.asyncio
async def test_multiple_skill_types_tracked(agent, tmp_path):
    py_file = tmp_path / "test.py"
    py_file.write_text("def hello():\n    return 'world'")
    await agent.run({"type": "query", "input": "test question"})
    await agent.run({"type": "load_code", "input": str(py_file)})
    skills = agent.memory.get_skills()
    skill_names = [s["name"] for s in skills]
    assert "query" in skill_names
    assert "load_code" in skill_names


# --- Error loop ---

@pytest.mark.asyncio
async def test_error_logged_on_failure(agent):
    await agent.run({"type": "load_code", "input": "/nonexistent/file.py"})
    errors = agent.memory.get_errors()
    assert len(errors) > 0
    assert errors[-1]["error_type"] == "FileNotFoundError"


@pytest.mark.asyncio
async def test_no_error_logged_on_success(agent):
    initial_errors = agent.memory.get_errors()
    await agent.run({"type": "query", "input": "test question"})
    final_errors = agent.memory.get_errors()
    assert len(final_errors) == len(initial_errors)


# --- Training samples ---

@pytest.mark.asyncio
async def test_training_sample_saved_on_success(agent):
    await agent.run({"type": "query", "input": "What does add do?"})
    data = agent.memory._read_json(agent.memory.training_path)
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["agent_id"] == "coder_agent"


@pytest.mark.asyncio
async def test_no_training_sample_on_failure(agent):
    await agent.run({"type": "load_code", "input": "/nonexistent/file.py"})
    data = agent.memory._read_json(agent.memory.training_path)
    assert len(data["candidates"]) == 0


def test_preloaded_errors_in_memory(tmp_path, mock_ollama, mock_collection):
    import json
    from pathlib import Path

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    error_file = memory_dir / "errors.json"
    
    mock_data = {
        "errors": [
            {"error_type": "FileNotFoundError", "task_type": "load_code", "context": "missing file"},
            {"error_type": "UnicodeDecodeError", "task_type": "load_code", "context": "bad encoding"},
            {"error_type": "SyntaxError", "task_type": "coding", "context": "bad python"},
            {"error_type": "TypeError", "task_type": "coding", "context": "none type"},
            {"error_type": "ValueError", "task_type": "coding", "context": "invalid value"}
        ]
    }
    
    with open(error_file, "w") as f:
        json.dump(mock_data, f)

    with patch("chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = (
            mock_collection
        )
        coder = CoderAgent(
            memory_dir=memory_dir,
            ollama_client=mock_ollama,
            chroma_dir=tmp_path / "chroma",
        )
    errors = coder.memory.get_errors()
    assert len(errors) == 5
    error_types = [e["error_type"] for e in errors]
    assert "FileNotFoundError" in error_types
    assert "UnicodeDecodeError" in error_types