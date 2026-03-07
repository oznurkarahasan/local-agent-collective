"""
FastAPI application — Indis.ai Web UI backend.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from frontend.api.routers import documents, query, status
from frontend.api.dependencies import get_rag_agent, get_coder_agent, get_ollama


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup."""
    get_ollama()
    get_rag_agent()
    get_coder_agent()
    yield


app = FastAPI(
    title="Indis.ai",
    description="Local AI Agent Collective — Independent Intelligence",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(status.router)
app.include_router(documents.router)
app.include_router(query.router)

# Static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the frontend."""
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Indis.ai API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}
