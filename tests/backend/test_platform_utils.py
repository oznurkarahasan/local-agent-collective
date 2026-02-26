"""
Tests for platform_utils.py
"""

import platform
from pathlib import Path
from backend.core.platform_utils import PlatformUtils


def test_get_base_dir_returns_path():
    base = PlatformUtils.get_base_dir()
    assert isinstance(base, Path)
    assert base.exists()


def test_get_base_dir_platform_specific():
    base = PlatformUtils.get_base_dir()
    os_name = platform.system()

    if os_name == "Linux":
        assert ".local/share/local-agent-collective" in str(base)
    elif os_name == "Darwin":
        assert "Application Support/local-agent-collective" in str(base)
    elif os_name == "Windows":
        assert "local-agent-collective" in str(base)


def test_get_ollama_url_default():
    import os
    os.environ.pop("OLLAMA_BASE_URL", None)
    url = PlatformUtils.get_ollama_url()
    assert url == "http://localhost:11434"


def test_get_ollama_url_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    url = PlatformUtils.get_ollama_url()
    assert url == "http://ollama:11434"


def test_is_ollama_running_returns_bool():
    result = PlatformUtils.is_ollama_running()
    assert isinstance(result, bool)


def test_get_ollama_install_instructions_not_empty():
    instructions = PlatformUtils.get_ollama_install_instructions()
    assert isinstance(instructions, str)
    assert len(instructions) > 0


def test_get_vector_store_dir():
    path = PlatformUtils.get_vector_store_dir()
    assert isinstance(path, Path)
    assert "vector_store" in str(path)


def test_get_documents_dir():
    path = PlatformUtils.get_documents_dir()
    assert isinstance(path, Path)
    assert "documents" in str(path)
