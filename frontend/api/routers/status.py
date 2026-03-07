"""
Status router — system health and agent memory stats.
"""

from fastapi import APIRouter
from frontend.api.dependencies import get_ollama, get_rag_agent, get_coder_agent
from backend.core.platform_utils import PlatformUtils

router = APIRouter(prefix="/status", tags=["status"])


@router.get("")
async def get_status():
    """System status — Ollama, models, agents."""
    ollama = get_ollama()
    ollama_ok = await ollama.ping()

    models = []
    if ollama_ok:
        models = await ollama.list_models()

    rag_info = get_rag_agent().get_agent_info()
    coder_info = get_coder_agent().get_agent_info()

    return {
        "ollama": ollama_ok,
        "platform": PlatformUtils.OS,
        "models": [m.get("name") for m in models],
        "agents": {
            "rag_agent": {
                "ready": True,
                "skills": len(rag_info["skills"]),
            },
            "coder_agent": {
                "ready": True,
                "skills": len(coder_info["skills"]),
            },
        },
    }


@router.get("/memory")
async def get_memory_stats():
    """Agent memory statistics."""
    rag = get_rag_agent()
    coder = get_coder_agent()

    return {
        "rag_agent": {
            "skills": rag.memory.get_skills(),
            "errors": rag.memory.get_errors(),
        },
        "coder_agent": {
            "skills": coder.memory.get_skills(),
            "errors": coder.memory.get_errors(),
        },
    }
