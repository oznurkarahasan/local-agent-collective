"""
Shared agent instances — singleton pattern.
Agents are initialized once at startup and reused across requests.
"""

from pathlib import Path
from backend.core.ollama_client import OllamaClient
from backend.core.platform_utils import PlatformUtils
from agents.rag_agent.agent import RagAgent
from agents.coder_agent.agent import CoderAgent


_ollama: OllamaClient | None = None
_rag_agent: RagAgent | None = None
_coder_agent: CoderAgent | None = None


def get_ollama() -> OllamaClient:
    global _ollama
    if _ollama is None:
        _ollama = OllamaClient()
    return _ollama


def get_rag_agent() -> RagAgent:
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RagAgent(
            memory_dir=Path("agents/rag_agent/memory"),
            ollama_client=get_ollama(),
            chroma_dir=PlatformUtils.get_vector_store_dir(),
        )
    return _rag_agent


def get_coder_agent() -> CoderAgent:
    global _coder_agent
    if _coder_agent is None:
        _coder_agent = CoderAgent(
            memory_dir=Path("agents/coder_agent/memory"),
            ollama_client=get_ollama(),
            chroma_dir=PlatformUtils.get_vector_store_dir(),
        )
    return _coder_agent


CODE_EXTENSIONS = {".py", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".java"}
DOC_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
