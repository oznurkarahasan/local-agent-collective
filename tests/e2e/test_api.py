"""
API endpoint tests for the FastAPI web UI backend.
Uses TestClient — no real Ollama or ChromaDB needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from fastapi.testclient import TestClient
import io

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
def mock_orchestrator():
    orchestrator = AsyncMock()
    orchestrator.initialize = AsyncMock()

    orchestrator.run = AsyncMock(return_value={
        "success": True,
        "plan": {"steps": []},
        "results": [{"step_id": 1, "success": True, "output": {"answer": "Mock answer"}}],
        "report": "Mock report"
    })

    orchestrator._execute_plan = AsyncMock(return_value=[
        {"step_id": 1, "success": True, "output": {"chunks_stored": 5, "language": "python"}}
    ])

    class MockRagAgentClass:
        def __init__(self, **kwargs):
            self.memory = MagicMock()
            self.memory.get_skills.return_value = []
            self.memory.get_errors.return_value = []
        def get_agent_info(self):
            return {"id": "rag_agent", "capabilities": [], "skills": [], "error_count": 0}

    class MockCoderAgentClass:
        def __init__(self, **kwargs):
            self.memory = MagicMock()
            self.memory.get_skills.return_value = []
            self.memory.get_errors.return_value = []
        def get_agent_info(self):
            return {"id": "coder_agent", "capabilities": [], "skills": [], "error_count": 0}

    registry = MagicMock()
    def get_class_side_effect(agent_id):
        if agent_id == "rag_agent":
            return MockRagAgentClass
        if agent_id == "coder_agent":
            return MockCoderAgentClass
        return None

    registry.get_class.side_effect = get_class_side_effect
    registry.agents_dir = Path("/tmp")
    orchestrator.registry = registry

    return orchestrator

from frontend.api.dependencies import get_ollama, get_orchestrator

@pytest.fixture
def client(mock_ollama, mock_orchestrator):
    from frontend.api.main import app
    app.dependency_overrides[get_ollama] = lambda: mock_ollama
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    
    with TestClient(app) as test_client:
        yield test_client
        
    app.dependency_overrides.clear()

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

def test_load_document_txt(client, mock_orchestrator):
    mock_orchestrator._execute_plan = AsyncMock(return_value=[{
        "success": True,
        "output": {"chunks_stored": 3, "language": ""},
    }])
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

def test_load_code_python(client, mock_orchestrator):
    mock_orchestrator._execute_plan = AsyncMock(return_value=[{
        "success": True,
        "output": {"chunks_stored": 5, "language": "python"},
    }])
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

def test_load_document_agent_failure(client, mock_orchestrator):
    mock_orchestrator._execute_plan = AsyncMock(return_value=[{
        "success": False,
        "error": "Something went wrong",
    }])
    
    res = client.post(
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
    assert "orchestrator" in data["results"]

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
    assert "results" in data
    assert "orchestrator" in data["results"]

# --- root ---

def test_root_returns_response(client):
    res = client.get("/")
    assert res.status_code == 200
