"""
Shared instances — singleton pattern.
The Orchestrator is initialized once at startup and reused across requests.
"""

from pathlib import Path
from backend.core.ollama_client import OllamaClient
from backend.core.orchestrator import Orchestrator

_ollama: OllamaClient | None = None
_orchestrator: Orchestrator | None = None


def get_ollama() -> OllamaClient:
    global _ollama
    if _ollama is None:
        _ollama = OllamaClient()
    return _ollama


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(ollama_client=get_ollama())
    return _orchestrator


CODE_EXTENSIONS = {".py", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".java"}
DOC_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
