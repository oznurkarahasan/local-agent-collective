"""
Documents router — file upload and listing.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from frontend.api.dependencies import (
    get_rag_agent,
    get_coder_agent,
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
)

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory loaded files list
_loaded_files: list[dict] = []


@router.post("/load")
async def load_document(file: UploadFile = File(...)):
    """Upload and load a document or code file."""
    suffix = Path(file.filename).suffix.lower()

    if suffix not in CODE_EXTENSIONS and suffix not in DOC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{suffix}'. "
                f"Documents: {sorted(DOC_EXTENSIONS)}, "
                f"Code: {sorted(CODE_EXTENSIONS)}"
            ),
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        if suffix in CODE_EXTENSIONS:
            agent = get_coder_agent()
            task_type = "load_code"
            agent_name = "coder_agent"
        else:
            agent = get_rag_agent()
            task_type = "load_document"
            agent_name = "rag_agent"

        result = await agent.run({"type": task_type, "input": str(tmp_path)})

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to load file"),
            )

        chunks = result["output"].get("chunks_stored", 0)
        language = result["output"].get("language", "")

        entry = {
            "filename": file.filename,
            "agent": agent_name,
            "chunks": chunks,
            "language": language,
            "suffix": suffix,
        }
        _loaded_files.append(entry)

        return {
            "success": True,
            "filename": file.filename,
            "agent": agent_name,
            "chunks_stored": chunks,
            "language": language,
        }

    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("")
async def list_documents():
    """List all loaded files."""
    return {"files": _loaded_files}
