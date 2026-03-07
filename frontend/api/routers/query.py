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

    if request.target in ("auto", "documents", "both"):
        rag = get_rag_agent()
        rag_result = await rag.run({"type": "query", "input": request.question})
        if rag_result["success"] and isinstance(rag_result["output"], dict):
            results["documents"] = rag_result["output"]

    if request.target in ("auto", "code", "both"):
        coder = get_coder_agent()
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
