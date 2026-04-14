"""
FastAPI application — Indis.ai Web UI backend.
Orchestrates multiple local agents via the central Orchestrator.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Routers
from frontend.api.routers import documents, query, status
from frontend.api.dependencies import get_orchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IndisAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager:
    Initializes the Orchestrator and warms up VRAM models on startup.
    """
    logger.info("Starting Indis.ai Local Agent Collective...")

    try:
        # This is where the magic happens:
        # Pre-loads nomic-embed and prepares the system for the RTX 3070 Ti
        orchestrator = get_orchestrator()
        await orchestrator.initialize()
        logger.info("System initialized: VRAM models are warm and ready.")
    except Exception as e:
        logger.error(f"Critical initialization failure: {e}")

    yield

    logger.info("Shutting down Indis.ai...")


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

# Include Routers
app.include_router(status.router)
app.include_router(documents.router)
app.include_router(query.router)  # This is likely where user queries hit the Orchestrator

# Static files & Frontend serving
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the Web UI."""
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Indis.ai API is running", "docs": "/docs"}


@app.get("/health")
async def health():
    """System health check."""
    return {"status": "ok", "engine": "Ollama", "gpu_vram": "8GB Optimized"}