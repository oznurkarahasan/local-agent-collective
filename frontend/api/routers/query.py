"""
Query router — ask questions about loaded documents and code.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from frontend.api.dependencies import get_orchestrator

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    target: str = "auto"  # "auto" | "documents" | "code" | "both"


@router.post("")
async def query(request: QueryRequest, orchestrator=Depends(get_orchestrator)):
    """Ask a question about loaded files."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    prompt = request.question
    if request.target != "auto":
        prompt = f"[Focus on {request.target}] {prompt}"

    result = await orchestrator.run(user_input=prompt)

    if not result.get("success"):
        return {
            "question": request.question,
            "results": {},
            "message": result.get("error", "Orchestrator failed to process the request."),
        }

    return {
        "question": request.question,
        "results": {
            "orchestrator": {
                "report": result.get("report"),
                "plan": result.get("plan"),
            }
        },
    }
