from fastapi import APIRouter, Depends, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from app.services.rag.pipeline import rag_pipeline
from app.core.security import get_current_user
from app.db.mongodb import get_database
from typing import Optional
import uuid

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: Optional[dict] = Depends(get_current_user),
    db=Depends(get_database)
):
    """
    Main chat endpoint — processes a student query through the full RAG pipeline.
    Includes input/output guardrails, retrieval, and LLM generation.
    """
    user_id = current_user["_id"] if current_user else None
    response, audit_log = await rag_pipeline.process_query(request, user_id)

    # Persist audit log
    await db.query_logs.insert_one(audit_log)

    return response


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 20,
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """Retrieve chat history for a session."""
    logs = await db.query_logs.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(limit).to_list(length=limit)

    return {
        "session_id": session_id,
        "messages": [
            {
                "query": log["query"],
                "response": log["response"],
                "confidence_score": log.get("confidence_score"),
                "guardrail_status": log.get("guardrail_status"),
                "sources": log.get("sources", []),
                "timestamp": log.get("timestamp")
            }
            for log in reversed(logs)
        ]
    }


@router.get("/sessions")
async def get_user_sessions(
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """Get all sessions for the current user."""
    pipeline = [
        {"$match": {"user_id": current_user["_id"]}},
        {"$group": {
            "_id": "$session_id",
            "last_query": {"$last": "$query"},
            "last_timestamp": {"$max": "$timestamp"},
            "message_count": {"$sum": 1}
        }},
        {"$sort": {"last_timestamp": -1}},
        {"$limit": 50}
    ]
    sessions = await db.query_logs.aggregate(pipeline).to_list(length=50)
    return {"sessions": sessions}
