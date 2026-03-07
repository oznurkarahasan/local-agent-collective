"""
Tests for CLI interface.
Uses mocking — no real Ollama or ChromaDB needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from interface.cli import CLI


@pytest.fixture
def mock_ollama():
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.list_models = AsyncMock(
        return_value=[
            {"name": "nomic-embed-text:v1.5"},
            {"name": "qwen3:4b"},
        ]
    )
    client.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    client.chat = AsyncMock(return_value="Test answer")
    return client


@pytest.fixture
def mock_coder_agent():
    agent = MagicMock()
    agent.run = AsyncMock(
        return_value={
            "success": True,
            "output": {
                "answer": "Test code answer",
                "sources": ["/code/test.py"],
                "chunks_used": 2,
            },
        }
    )
    agent.get_agent_info.return_value = {
        "id": "coder_agent",
        "capabilities": ["code_analysis"],
        "required_model_role": "code_analysis",
        "skills": [],
        "error_count": 0,
    }
    agent.memory = MagicMock()
    agent.memory.get_skills.return_value = []
    agent.memory.get_errors.return_value = []
    return agent


@pytest.fixture
def mock_rag_agent():
    agent = MagicMock()
    agent.run = AsyncMock(
        return_value={
            "success": True,
            "output": {
                "answer": "Test answer",
                "sources": ["/docs/test.txt"],
                "chunks_used": 2,
            },
        }
    )
    agent.get_agent_info.return_value = {
        "id": "rag_agent",
        "capabilities": ["document_qa"],
        "required_model_role": "rag",
        "skills": [],
        "error_count": 0,
    }
    agent.memory = MagicMock()
    agent.memory.get_skills.return_value = [
        {
            "name": "query",
            "success_rate": 0.95,
            "total_runs": 20,
        }
    ]
    agent.memory.get_errors.return_value = []
    return agent


@pytest.fixture
def cli(mock_ollama, mock_rag_agent, mock_coder_agent):
    """Create CLI with mocked dependencies."""
    with patch("interface.cli.OllamaClient", return_value=mock_ollama):
        with patch("interface.cli.RagAgent", return_value=mock_rag_agent):
            with patch("interface.cli.CoderAgent", return_value=mock_coder_agent):
                c = CLI()
                c.ollama = mock_ollama
                c.rag_agent = mock_rag_agent
                c.coder_agent = mock_coder_agent
                return c


# --- system check ---


@pytest.mark.asyncio
async def test_check_system_ollama_running(cli, capsys):
    await cli._check_system()
    captured = capsys.readouterr()
    assert "✓ Ollama is running" in captured.out


@pytest.mark.asyncio
async def test_check_system_ollama_not_running(cli, capsys):
    cli.ollama.ping = AsyncMock(return_value=False)
    with pytest.raises(SystemExit):
        await cli._check_system()
    captured = capsys.readouterr()
    assert "✗ Ollama is not running" in captured.out


# --- load command ---


@pytest.mark.asyncio
async def test_cmd_load_success(cli, tmp_path, capsys):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Test content")

    cli.rag_agent.run = AsyncMock(
        return_value={
            "success": True,
            "output": {"file": str(txt_file), "chunks_stored": 3},
        }
    )

    await cli._cmd_load(str(txt_file))
    captured = capsys.readouterr()
    assert "✓" in captured.out
    assert "3 chunks" in captured.out


@pytest.mark.asyncio
async def test_cmd_load_file_not_found(cli, capsys):
    await cli._cmd_load("/nonexistent/file.txt")
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower()


@pytest.mark.asyncio
async def test_cmd_load_no_args(cli, capsys):
    await cli._cmd_load("")
    captured = capsys.readouterr()
    assert "Usage" in captured.out


@pytest.mark.asyncio
async def test_cmd_load_adds_to_loaded_docs(cli, tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("content")

    cli.rag_agent.run = AsyncMock(
        return_value={
            "success": True,
            "output": {"file": str(txt_file), "chunks_stored": 1},
        }
    )

    await cli._cmd_load(str(txt_file))
    assert len(cli.loaded_docs) == 1


# --- ask command ---


@pytest.mark.asyncio
async def test_cmd_ask_no_docs_loaded(cli, capsys):
    cli.loaded_docs = []
    await cli._cmd_ask("What is this?")
    captured = capsys.readouterr()
    assert "No files loaded" in captured.out


@pytest.mark.asyncio
async def test_cmd_ask_success(cli, capsys):
    cli.loaded_docs = ["/docs/test.txt"]
    await cli._cmd_ask("What is this document about?")
    captured = capsys.readouterr()
    assert "Test answer" in captured.out


@pytest.mark.asyncio
async def test_cmd_ask_no_question(cli, capsys):
    await cli._cmd_ask("")
    captured = capsys.readouterr()
    assert "Usage" in captured.out


# --- list docs command ---


def test_cmd_list_docs_empty(cli, capsys):
    cli.loaded_docs = []
    cli._cmd_list_docs()
    captured = capsys.readouterr()
    assert "No files loaded" in captured.out


def test_cmd_list_docs_with_files(cli, capsys):
    cli.loaded_docs = ["/docs/test.txt", "/docs/report.pdf"]
    cli._cmd_list_docs()
    captured = capsys.readouterr()
    assert "test.txt" in captured.out
    assert "report.pdf" in captured.out


# --- status command ---


@pytest.mark.asyncio
async def test_cmd_status(cli, capsys):
    await cli._cmd_status()
    captured = capsys.readouterr()
    assert "Ollama" in captured.out
    assert "RAG Agent" in captured.out


# --- memory stats command ---


def test_cmd_memory_stats(cli, capsys):
    cli._cmd_memory_stats()
    captured = capsys.readouterr()
    assert "Skills" in captured.out
    assert "query" in captured.out


# --- command routing ---


@pytest.mark.asyncio
async def test_handle_exit_command(cli, capsys):
    with pytest.raises(SystemExit):
        await cli._handle_command("exit")


@pytest.mark.asyncio
async def test_handle_help_command(cli, capsys):
    await cli._handle_command("help")
    captured = capsys.readouterr()
    assert "load" in captured.out
    assert "ask" in captured.out


@pytest.mark.asyncio
async def test_handle_unknown_command(cli, capsys):
    await cli._handle_command("unknowncmd")
    captured = capsys.readouterr()
    assert "Unknown command" in captured.out


@pytest.mark.asyncio
async def test_handle_list_docs(cli, capsys):
    await cli._handle_command("list docs")
    captured = capsys.readouterr()
    assert "No files loaded" in captured.out


@pytest.mark.asyncio
async def test_handle_memory_stats(cli, capsys):
    await cli._handle_command("memory stats")
    captured = capsys.readouterr()
    assert "Skills" in captured.out
