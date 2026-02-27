"""
Tests for memory_manager.py
"""

import pytest
from pathlib import Path
from backend.core.memory_manager import MemoryManager


@pytest.fixture
def memory(tmp_path):
    """Create a MemoryManager with a temporary directory."""
    return MemoryManager(memory_dir=tmp_path / "test_agent" / "memory")


# --- initialization ---

def test_memory_files_created(tmp_path):
    memory_dir = tmp_path / "agent" / "memory"
    MemoryManager(memory_dir=memory_dir)
    assert (memory_dir / "skills.json").exists()
    assert (memory_dir / "errors.json").exists()
    assert (memory_dir / "training_candidates.json").exists()


def test_initial_skills_empty(memory):
    assert memory.get_skills() == []


def test_initial_errors_empty(memory):
    assert memory.get_errors() == []


# --- skills ---

def test_add_skill(memory):
    skill = memory.add_skill("PDF reading", "Reads and chunks PDF files")
    assert skill["name"] == "PDF reading"
    assert skill["success_rate"] == 1.0
    assert skill["total_runs"] == 0
    assert skill["id"].startswith("sk_")


def test_get_skills_after_add(memory):
    memory.add_skill("PDF reading", "Reads PDF")
    memory.add_skill("Semantic search", "Searches ChromaDB")
    skills = memory.get_skills()
    assert len(skills) == 2


def test_update_skill_rate_success(memory):
    skill = memory.add_skill("PDF reading", "Reads PDF")
    updated = memory.update_skill_rate(skill["id"], success=True)
    assert updated["total_runs"] == 1
    assert updated["success_rate"] == 1.0


def test_update_skill_rate_failure(memory):
    skill = memory.add_skill("PDF reading", "Reads PDF")
    updated = memory.update_skill_rate(skill["id"], success=False)
    assert updated["total_runs"] == 1
    assert updated["success_rate"] == 0.0


def test_update_skill_rate_mixed(memory):
    skill = memory.add_skill("PDF reading", "Reads PDF")
    memory.update_skill_rate(skill["id"], success=True)
    memory.update_skill_rate(skill["id"], success=True)
    updated = memory.update_skill_rate(skill["id"], success=False)
    assert updated["total_runs"] == 3
    assert round(updated["success_rate"], 2) == 0.67


def test_update_skill_rate_not_found(memory):
    result = memory.update_skill_rate("nonexistent_id", success=True)
    assert result is None


# --- errors ---

def test_log_error(memory):
    error = memory.log_error(
        error_type="FileNotFoundError",
        context="PDF path was wrong",
        solution="Use absolute path",
    )
    assert error["error_type"] == "FileNotFoundError"
    assert error["resolved"] is True
    assert error["id"].startswith("err_")


def test_log_error_without_solution(memory):
    error = memory.log_error(
        error_type="OllamaConnectionError",
        context="Ollama not running",
    )
    assert error["resolved"] is False
    assert error["solution"] is None


def test_resolve_error(memory):
    error = memory.log_error(
        error_type="OllamaConnectionError",
        context="Ollama not running",
    )
    resolved = memory.resolve_error(error["id"], solution="Start Ollama first")
    assert resolved["resolved"] is True
    assert resolved["solution"] == "Start Ollama first"


def test_resolve_error_not_found(memory):
    result = memory.resolve_error("nonexistent_id", solution="fix")
    assert result is None


def test_get_errors_after_log(memory):
    memory.log_error("TypeError", "wrong type", "cast to string")
    memory.log_error("ValueError", "bad value")
    errors = memory.get_errors()
    assert len(errors) == 2


# --- scan_errors ---

def test_scan_errors_finds_match(memory):
    memory.log_error(
        error_type="FileNotFoundError",
        context="PDF file not found in documents folder",
        solution="Check the file path",
    )
    matches = memory.scan_errors("PDF documents folder missing")
    assert len(matches) == 1


def test_scan_errors_no_match(memory):
    memory.log_error(
        error_type="FileNotFoundError",
        context="PDF file not found",
        solution="Check path",
    )
    matches = memory.scan_errors("database connection timeout")
    assert len(matches) == 0


def test_scan_errors_unresolved_not_returned(memory):
    memory.log_error(
        error_type="FileNotFoundError",
        context="PDF file not found",
    )
    matches = memory.scan_errors("PDF file")
    assert len(matches) == 0


# --- training candidates ---

def test_save_training_sample(memory):
    sample = memory.save_training_sample(
        instruction="Summarize the document",
        input_text="Document content here",
        output="Summary of the document",
        agent_id="rag_agent",
    )
    assert sample["approved"] is False
    assert sample["quality_score"] is None
    assert sample["id"].startswith("train_")
    assert sample["agent_id"] == "rag_agent"
