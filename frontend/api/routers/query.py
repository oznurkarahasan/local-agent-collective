"""
Query router — ask questions about loaded documents and code.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from frontend.api.dependencies import get_rag_agent, get_coder_agent

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    target: str = "auto"  # "auto" | "documents" | "code" | "both"


@router.post("")
async def query(request: QueryRequest):
    """Ask a question about loaded files."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    results = {}

    rag = get_rag_agent()
    coder = get_coder_agent()

    run_rag = request.target in ("documents", "both") or (
        request.target == "auto" and rag.collection.count() > 0
    )
    run_coder = request.target in ("code", "both") or (
        request.target == "auto" and coder.collection.count() > 0
    )

    if run_rag:
        rag_result = await rag.run({"type": "query", "input": request.question})
        if rag_result["success"] and isinstance(rag_result["output"], dict):
            results["documents"] = rag_result["output"]

    if run_coder:
        coder_result = await coder.run({"type": "query", "input": request.question})
        if coder_result["success"] and isinstance(coder_result["output"], dict):
            results["code"] = coder_result["output"]

    if not results:
        return {
            "question": request.question,
            "results": {},
            "message": "No relevant content found.",
        }

    return {
        "question": request.question,
        "results": results,
    }
