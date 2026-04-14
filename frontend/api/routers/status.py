"""
Status router — system health and agent memory stats.
"""

from fastapi import APIRouter, Depends
from frontend.api.dependencies import get_ollama, get_orchestrator
from backend.core.platform_utils import PlatformUtils

router = APIRouter(prefix="/status", tags=["status"])


@router.get("")
async def get_status(orchestrator=Depends(get_orchestrator), ollama=Depends(get_ollama)):
    """System status — Ollama, models, agents."""
    ollama_ok = await ollama.ping()

    models = []
    if ollama_ok:
        models = await ollama.list_models()

    rag_info = {"skills": []}
    coder_info = {"skills": []}

    rag_class = orchestrator.registry.get_class("rag_agent")
    if rag_class:
        rag = rag_class(
            agent_id="rag_agent",
            memory_dir=orchestrator.registry.agents_dir / "rag_agent" / "memory",
            ollama_client=ollama,
        )
        rag_info = rag.get_agent_info()

    coder_class = orchestrator.registry.get_class("coder_agent")
    if coder_class:
        coder = coder_class(
            agent_id="coder_agent",
            memory_dir=orchestrator.registry.agents_dir / "coder_agent" / "memory",
            ollama_client=ollama,
        )
        coder_info = coder.get_agent_info()

    return {
        "ollama": ollama_ok,
        "platform": PlatformUtils.OS,
        "models": [m.get("name") for m in models],
        "agents": {
            "rag_agent": {
                "ready": bool(rag_class),
                "skills": len(rag_info.get("skills", [])),
            },
            "coder_agent": {
                "ready": bool(coder_class),
                "skills": len(coder_info.get("skills", [])),
            },
        },
    }


@router.get("/memory")
async def get_memory_stats(orchestrator=Depends(get_orchestrator), ollama=Depends(get_ollama)):
    """Agent memory statistics."""
    rag_stats = {"skills": [], "errors": []}
    coder_stats = {"skills": [], "errors": []}

    rag_class = orchestrator.registry.get_class("rag_agent")
    if rag_class:
        rag = rag_class(
            agent_id="rag_agent",
            memory_dir=orchestrator.registry.agents_dir / "rag_agent" / "memory",
            ollama_client=ollama,
        )
        rag_stats = {"skills": rag.memory.get_skills(), "errors": rag.memory.get_errors()}

    coder_class = orchestrator.registry.get_class("coder_agent")
    if coder_class:
        coder = coder_class(
            agent_id="coder_agent",
            memory_dir=orchestrator.registry.agents_dir / "coder_agent" / "memory",
            ollama_client=ollama,
        )
        coder_stats = {"skills": coder.memory.get_skills(), "errors": coder.memory.get_errors()}

    return {
        "rag_agent": rag_stats,
        "coder_agent": coder_stats,
    }
