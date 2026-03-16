"""
API endpoint tests for the FastAPI web UI backend.
Uses TestClient — no real Ollama or ChromaDB needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import io


@pytest.fixture
def mock_rag_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "success": True,
        "output": {
            "answer": "Test answer",
            "sources": ["/tmp/test.txt"],
            "chunks_used": 2,
        },
    })
    agent.get_agent_info.return_value = {
        "id": "rag_agent",
        "capabilities": ["document_qa"],
        "required_model_role": "rag",
        "skills": [],
        "error_count": 0,
    }
    agent.collection = MagicMock()
    agent.collection.count.return_value = 1
    agent.memory = MagicMock()
    agent.memory.get_skills.return_value = []
    agent.memory.get_errors.return_value = []
    return agent


@pytest.fixture
def mock_coder_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "success": True,
        "output": {
            "answer": "This function adds two numbers.",
            "sources": ["/tmp/test.py"],
            "language": "python",
            "chunks_used": 3,
        },
    })
    agent.get_agent_info.return_value = {
        "id": "coder_agent",
        "capabilities": ["code_analysis"],
        "required_model_role": "code_analysis",
        "skills": [],
        "error_count": 0,
    }
    agent.collection = MagicMock()
    agent.collection.count.return_value = 1
    agent.memory = MagicMock()
    agent.memory.get_skills.return_value = []
    agent.memory.get_errors.return_value = []
    return agent


@pytest.fixture
def mock_ollama():
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.list_models = AsyncMock(return_value=[
        {"name": "nomic-embed-text:v1.5"},
        {"name": "qwen3:4b"},
        {"name": "qwen2.5-coder:3b"},
    ])
    return client


@pytest.fixture
def client(mock_ollama, mock_rag_agent, mock_coder_agent):
    with patch("frontend.api.dependencies.get_ollama", return_value=mock_ollama):
        with patch("frontend.api.dependencies.get_rag_agent", return_value=mock_rag_agent):
            with patch("frontend.api.dependencies.get_coder_agent", return_value=mock_coder_agent):
                with patch("frontend.api.routers.status.get_ollama", return_value=mock_ollama):
                    with patch("frontend.api.routers.status.get_rag_agent", return_value=mock_rag_agent):
                        with patch("frontend.api.routers.status.get_coder_agent", return_value=mock_coder_agent):
                            with patch("frontend.api.routers.query.get_rag_agent", return_value=mock_rag_agent):
                                with patch("frontend.api.routers.query.get_coder_agent", return_value=mock_coder_agent):
                                    with patch("frontend.api.routers.documents.get_rag_agent", return_value=mock_rag_agent):
                                        with patch("frontend.api.routers.documents.get_coder_agent", return_value=mock_coder_agent):
                                            from frontend.api.main import app
                                            return TestClient(app)


# --- health ---

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# --- status ---

def test_get_status(client):
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert "ollama" in data
    assert "agents" in data
    assert "rag_agent" in data["agents"]
    assert "coder_agent" in data["agents"]


def test_get_memory_stats(client):
    res = client.get("/status/memory")
    assert res.status_code == 200
    data = res.json()
    assert "rag_agent" in data
    assert "coder_agent" in data
    assert "skills" in data["rag_agent"]
    assert "errors" in data["rag_agent"]


# --- documents ---

def test_list_documents_empty(client):
    res = client.get("/documents")
    assert res.status_code == 200
    assert "files" in res.json()


def test_load_document_txt(client, mock_rag_agent):
    mock_rag_agent.run = AsyncMock(return_value={
        "success": True,
        "output": {"chunks_stored": 3, "language": ""},
    })
    file_content = b"This is a test document."
    res = client.post(
        "/documents/load",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["filename"] == "test.txt"
    assert data["agent"] == "rag_agent"


def test_load_code_python(client, mock_coder_agent):
    mock_coder_agent.run = AsyncMock(return_value={
        "success": True,
        "output": {"chunks_stored": 5, "language": "python"},
    })
    file_content = b"def hello():\n    return 'world'"
    res = client.post(
        "/documents/load",
        files={"file": ("test.py", io.BytesIO(file_content), "text/plain")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["agent"] == "coder_agent"
    assert data["language"] == "python"


def test_load_unsupported_type(client):
    res = client.post(
        "/documents/load",
        files={"file": ("test.xyz", io.BytesIO(b"content"), "text/plain")},
    )
    assert res.status_code == 400


def test_load_document_agent_failure(mock_ollama, mock_coder_agent):
    failing_rag = MagicMock()
    failing_rag.run = AsyncMock(return_value={
        "success": False,
        "error": "Something went wrong",
    })
    failing_rag.get_agent_info.return_value = {"id": "rag_agent", "skills": [], "error_count": 0}
    failing_rag.memory = MagicMock()

    with patch("frontend.api.routers.documents.get_rag_agent", return_value=failing_rag):
        with patch("frontend.api.routers.documents.get_coder_agent", return_value=mock_coder_agent):
            from frontend.api.main import app
            from fastapi.testclient import TestClient
            c = TestClient(app)
            res = c.post(
                "/documents/load",
                files={"file": ("fail.txt", io.BytesIO(b"content"), "text/plain")},
            )
    assert res.status_code == 500


# --- query ---

def test_query_success(client):
    res = client.post(
        "/query",
        json={"question": "What is this about?"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "question" in data
    assert "results" in data


def test_query_empty_question(client):
    res = client.post(
        "/query",
        json={"question": "   "},
    )
    assert res.status_code == 400


def test_query_target_documents(client):
    res = client.post(
        "/query",
        json={"question": "What is this?", "target": "documents"},
    )
    assert res.status_code == 200


def test_query_target_code(client):
    res = client.post(
        "/query",
        json={"question": "What does this function do?", "target": "code"},
    )
    assert res.status_code == 200


def test_query_target_both(client):
    res = client.post(
        "/query",
        json={"question": "Explain everything", "target": "both"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "documents" in data["results"] or "code" in data["results"]


# --- root ---

def test_root_returns_response(client):
    res = client.get("/")
    assert res.status_code == 200
