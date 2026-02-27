"""
Tests for agent_registry.py
"""

import json
import pytest
from pathlib import Path
from backend.core.agent_registry import AgentRegistry
from backend.core.agent_base import AgentBase


def create_agent_folder(
    base_dir: Path,
    agent_id: str,
    capabilities: list,
    input_types: list,
    enabled: bool = True,
    create_agent_py: bool = True,
) -> Path:
    """Helper to create a mock agent folder with config and agent.py."""
    folder = base_dir / agent_id
    folder.mkdir(parents=True)

    config = {
        "id": agent_id,
        "name": f"{agent_id} Agent",
        "capabilities": capabilities,
        "input_types": input_types,
        "enabled": enabled,
    }
    (folder / "config.json").write_text(json.dumps(config))

    if create_agent_py:
        agent_code = f'''
from backend.core.agent_base import AgentBase
from pathlib import Path

class {agent_id.title().replace("_", "")}Agent(AgentBase):
    def get_capabilities(self):
        return {capabilities}

    def get_required_model_role(self):
        return "rag"

    async def _execute(self, task):
        return {{"success": True, "output": "done"}}
'''
        (folder / "agent.py").write_text(agent_code)

    return folder


@pytest.fixture
def agents_dir(tmp_path):
    """Create a temporary agents directory with mock agents."""
    base = tmp_path / "agents"
    base.mkdir()

    create_agent_folder(
        base, "rag_agent",
        capabilities=["document_qa", "pdf_analysis"],
        input_types=["pdf", "txt", "docx"],
    )
    create_agent_folder(
        base, "coder_agent",
        capabilities=["code_analysis", "code_generation"],
        input_types=["py", "js", "ts"],
    )
    create_agent_folder(
        base, "disabled_agent",
        capabilities=["something"],
        input_types=["txt"],
        enabled=False,
    )

    return base


@pytest.fixture
def registry(agents_dir):
    """Create an AgentRegistry with test agents directory."""
    return AgentRegistry(agents_dir=agents_dir)


# --- discovery ---

def test_discovers_enabled_agents(registry):
    agents = registry.list_all()
    ids = [a["id"] for a in agents]
    assert "rag_agent" in ids
    assert "coder_agent" in ids


def test_disabled_agent_not_discovered(registry):
    agents = registry.list_all()
    ids = [a["id"] for a in agents]
    assert "disabled_agent" not in ids


def test_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty_agents"
    empty_dir.mkdir()
    registry = AgentRegistry(agents_dir=empty_dir)
    assert registry.list_all() == []


def test_nonexistent_directory(tmp_path):
    registry = AgentRegistry(agents_dir=tmp_path / "nonexistent")
    assert registry.list_all() == []


# --- find_by_capability ---

def test_find_by_capability_success(registry):
    results = registry.find_by_capability("document_qa")
    assert len(results) == 1
    assert results[0]["id"] == "rag_agent"


def test_find_by_capability_multiple(registry):
    results = registry.find_by_capability("code_analysis")
    assert len(results) == 1
    assert results[0]["id"] == "coder_agent"


def test_find_by_capability_not_found(registry):
    results = registry.find_by_capability("nonexistent_capability")
    assert results == []


# --- find_by_input_type ---

def test_find_by_input_type_pdf(registry):
    results = registry.find_by_input_type("pdf")
    assert len(results) == 1
    assert results[0]["id"] == "rag_agent"


def test_find_by_input_type_with_dot(registry):
    results = registry.find_by_input_type(".py")
    assert len(results) == 1
    assert results[0]["id"] == "coder_agent"


def test_find_by_input_type_not_found(registry):
    results = registry.find_by_input_type("xlsx")
    assert results == []


# --- get_config ---

def test_get_config_success(registry):
    config = registry.get_config("rag_agent")
    assert config is not None
    assert config["id"] == "rag_agent"


def test_get_config_not_found(registry):
    config = registry.get_config("nonexistent_agent")
    assert config is None


# --- get_class ---

def test_get_class_success(registry):
    cls = registry.get_class("rag_agent")
    assert cls is not None
    assert issubclass(cls, AgentBase)


def test_get_class_not_found(registry):
    cls = registry.get_class("nonexistent_agent")
    assert cls is None


# --- is_registered ---

def test_is_registered_true(registry):
    assert registry.is_registered("rag_agent") is True


def test_is_registered_false(registry):
    assert registry.is_registered("nonexistent") is False


# --- no agent.py ---

def test_agent_without_agent_py(tmp_path):
    base = tmp_path / "agents"
    base.mkdir()
    create_agent_folder(
        base, "config_only_agent",
        capabilities=["something"],
        input_types=["txt"],
        create_agent_py=False,
    )
    registry = AgentRegistry(agents_dir=base)
    assert registry.is_registered("config_only_agent") is True
    assert registry.get_class("config_only_agent") is None
