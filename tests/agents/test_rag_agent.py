"""
Tests for rag_agent/agent.py

Uses mocking — no real Ollama or ChromaDB connection needed in CI.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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
    collection.count.return_value = 5
    collection.query.return_value = {
        "documents": [["chunk 1 text", "chunk 2 text"]],
        "metadatas": [
            [
                {"source": "/docs/test.pdf", "chunk_index": 0},
                {"source": "/docs/test.pdf", "chunk_index": 1},
            ]
        ],
    }
    return collection


@pytest.fixture
def agent(tmp_path, mock_ollama, mock_collection):
    """Create a RagAgent with mocked dependencies."""
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


# --- initialization ---

def test_agent_id(agent):
    assert agent.agent_id == "rag_agent"


def test_get_capabilities(agent):
    caps = agent.get_capabilities()
    assert "document_qa" in caps


def test_get_required_model_role(agent):
    assert agent.get_required_model_role() == "rag"


# --- load_document ---

def test_load_document_txt(agent, tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world. This is a test document.")

    docs = agent.load_document(txt_file)
    assert len(docs) > 0
    assert "Hello world" in docs[0].page_content


def test_load_document_unsupported_type(agent, tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content")

    with pytest.raises(ValueError, match="Unsupported file type"):
        agent.load_document(bad_file)


def test_load_document_not_found(agent, tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        agent.load_document(tmp_path / "nonexistent.txt")


# --- chunk_document ---

def test_chunk_document_splits_long_text(agent, tmp_path):
    from langchain_core.documents import Document

    long_text = "word " * 500
    docs = [Document(page_content=long_text, metadata={})]
    chunks = agent.chunk_document(docs)
    assert len(chunks) > 1


def test_chunk_document_short_text(agent, tmp_path):
    from langchain_core.documents import Document

    docs = [Document(page_content="Short text.", metadata={})]
    chunks = agent.chunk_document(docs)
    assert len(chunks) == 1


# --- embed_and_store ---

@pytest.mark.asyncio
async def test_embed_and_store(agent, tmp_path):
    from langchain_core.documents import Document

    chunks = [
        Document(page_content="chunk one", metadata={}),
        Document(page_content="chunk two", metadata={}),
    ]
    count = await agent.embed_and_store(chunks, source="/docs/test.txt")
    assert count == 2
    agent.collection.upsert.assert_called_once()


# --- query ---

@pytest.mark.asyncio
async def test_query_returns_chunks(agent):
    results = await agent.query("What is this document about?")
    assert len(results) == 2
    assert results[0]["text"] == "chunk 1 text"
    assert results[0]["source"] == "/docs/test.pdf"


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
        {"text": "The sky is blue.", "source": "/docs/nature.txt"},
    ]
    answer = await agent.generate_answer("What color is the sky?", context)
    assert answer == "Test answer"
    agent.ollama.chat.assert_called_once()


# --- _execute ---

@pytest.mark.asyncio
async def test_execute_unknown_task_type(agent):
    result = await agent._execute({"type": "unknown", "input": "test"})
    assert result["success"] is False
    assert "Unknown task type" in result["error"]


@pytest.mark.asyncio
async def test_execute_query_empty_question(agent):
    result = await agent._execute({"type": "query", "input": ""})
    assert result["success"] is False
    assert "empty" in result["error"]


@pytest.mark.asyncio
async def test_execute_query_success(agent):
    result = await agent._execute({
        "type": "query",
        "input": "What is this about?",
    })
    assert result["success"] is True
    assert "answer" in result["output"]


@pytest.mark.asyncio
async def test_execute_load_document_not_found(agent):
    with pytest.raises(FileNotFoundError):
        await agent._execute({
            "type": "load_document",
            "input": "/nonexistent/file.txt",
        })


@pytest.mark.asyncio
async def test_execute_load_document_success(agent, tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Test document content for RAG agent.")

    result = await agent._execute({
        "type": "load_document",
        "input": str(txt_file),
    })
    assert result["success"] is True
    assert result["output"]["chunks_stored"] >= 1


# --- run (full AgentBase integration) ---

@pytest.mark.asyncio
async def test_run_query_task(agent):
    result = await agent.run({
        "type": "query",
        "input": "What is this document about?",
    })
    assert result["success"] is True


@pytest.mark.asyncio
async def test_run_registers_skill(agent):
    await agent.run({"type": "query", "input": "test question"})
    skills = agent.memory.get_skills()
    assert any(s["name"] == "query" for s in skills)
