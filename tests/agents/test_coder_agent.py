"""
Tests for coder_agent/agent.py

Uses mocking — no real Ollama or ChromaDB needed in CI.
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
    client.chat = AsyncMock(return_value="This function calculates the sum.")
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
    """Create a CoderAgent with mocked dependencies."""
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


# --- initialization ---

def test_agent_id(agent):
    assert agent.agent_id == "coder_agent"


def test_get_capabilities(agent):
    caps = agent.get_capabilities()
    assert "code_analysis" in caps
    assert "code_explanation" in caps


def test_get_required_model_role(agent):
    assert agent.get_required_model_role() == "code_analysis"


# --- load_code ---

def test_load_code_python(agent, tmp_path):
    py_file = tmp_path / "test.py"
    py_file.write_text("def hello():\n    return 'world'")
    code = agent.load_code(py_file)
    assert "def hello" in code


def test_load_code_javascript(agent, tmp_path):
    js_file = tmp_path / "test.js"
    js_file.write_text("function hello() { return 'world'; }")
    code = agent.load_code(js_file)
    assert "function hello" in code


def test_load_code_unsupported_type(agent, tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        agent.load_code(bad_file)


def test_load_code_not_found(agent, tmp_path):
    with pytest.raises(FileNotFoundError):
        agent.load_code(tmp_path / "nonexistent.py")


# --- chunk_code ---

def test_chunk_code_python(agent):
    code = "def foo():\n    pass\n\ndef bar():\n    pass\n" * 20
    chunks = agent.chunk_code(code, ".py")
    assert len(chunks) > 1


def test_chunk_code_short(agent):
    code = "def foo():\n    return 42"
    chunks = agent.chunk_code(code, ".py")
    assert len(chunks) == 1


def test_chunk_code_javascript(agent):
    code = "function foo() {}\nfunction bar() {}\n" * 20
    chunks = agent.chunk_code(code, ".js")
    assert len(chunks) > 0


def test_chunk_code_go(agent):
    code = "func foo() {}\nfunc bar() {}\n" * 20
    chunks = agent.chunk_code(code, ".go")
    assert len(chunks) > 0


# --- embed_and_store ---

@pytest.mark.asyncio
async def test_embed_and_store(agent):
    chunks = ["def foo():\n    pass", "def bar():\n    return 42"]
    count = await agent.embed_and_store(
        chunks, source="/code/test.py", language="python"
    )
    assert count == 2
    agent.collection.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_embed_and_store_metadata(agent):
    chunks = ["def foo():\n    pass"]
    await agent.embed_and_store(
        chunks, source="/code/test.py", language="python"
    )
    call_kwargs = agent.collection.upsert.call_args[1]
    assert call_kwargs["metadatas"][0]["language"] == "python"
    assert call_kwargs["metadatas"][0]["source"] == "/code/test.py"


# --- query ---

@pytest.mark.asyncio
async def test_query_returns_chunks(agent):
    results = await agent.query("What does the add function do?")
    assert len(results) == 1
    assert "def add" in results[0]["text"]
    assert results[0]["language"] == "python"


@pytest.mark.asyncio
async def test_query_empty_collection(agent, mock_collection):
    mock_collection.count.return_value = 0
    mock_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
    }
    results = await agent.query("any question")
    assert results == []


# --- generate_answer ---

@pytest.mark.asyncio
async def test_generate_answer(agent):
    context = [
        {
            "text": "def add(a, b):\n    return a + b",
            "source": "/code/math.py",
            "language": "python",
        }
    ]
    answer = await agent.generate_answer("What does add do?", context)
    assert answer == "This function calculates the sum."
    agent.ollama.chat.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_uses_coder_model(agent):
    context = [{"text": "code", "source": "file.py", "language": "python"}]
    await agent.generate_answer("explain", context)
    call_args = agent.ollama.chat.call_args
    assert call_args[1]["model"] == "qwen2.5-coder:3b"


# --- _execute ---

@pytest.mark.asyncio
async def test_execute_unknown_task_type_falls_back_to_query(agent):
    result = await agent._execute({"type": "unknown", "input": "test"})
    assert result["success"] is True
    assert "answer" in result["output"]


@pytest.mark.asyncio
async def test_execute_query_empty_question(agent):
    result = await agent._execute({"type": "query", "input": ""})
    assert result["success"] is False
    assert "empty" in result["error"]


@pytest.mark.asyncio
async def test_execute_query_success(agent):
    result = await agent._execute({
        "type": "query",
        "input": "What does add function do?",
    })
    assert result["success"] is True
    assert "answer" in result["output"]


@pytest.mark.asyncio
async def test_execute_load_code_not_found(agent):
    with pytest.raises(FileNotFoundError):
        await agent._execute({
            "type": "load_code",
            "input": "/nonexistent/file.py",
        })


@pytest.mark.asyncio
async def test_execute_load_code_success(agent, tmp_path):
    py_file = tmp_path / "test.py"
    py_file.write_text("def hello():\n    return 'world'")
    result = await agent._execute({
        "type": "load_code",
        "input": str(py_file),
    })
    assert result["success"] is True
    assert result["output"]["language"] == "python"
    assert result["output"]["chunks_stored"] >= 1


# --- run (AgentBase integration) ---

@pytest.mark.asyncio
async def test_run_query_task(agent):
    result = await agent.run({
        "type": "query",
        "input": "What does this code do?",
    })
    assert result["success"] is True


@pytest.mark.asyncio
async def test_run_registers_skill(agent):
    await agent.run({"type": "query", "input": "test question"})
    skills = agent.memory.get_skills()
    assert any(s["name"] == "query" for s in skills)
