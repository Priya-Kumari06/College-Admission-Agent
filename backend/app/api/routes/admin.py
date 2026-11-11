from fastapi import APIRouter, Depends, Query
from app.core.security import require_admin, require_officer_or_admin
from app.db.mongodb import get_database
from datetime import datetime, timedelta
from typing import Optional

router = APIRouter(prefix="/admin", tags=["Admin & Analytics"])


@router.get("/analytics/overview")
async def get_overview(
    current_user=Depends(require_officer_or_admin),
    db=Depends(get_database)
):
    """Dashboard overview: total queries, guardrail stats, top categories."""
    total_queries = await db.query_logs.count_documents({})
    total_documents = await db.documents.count_documents({})
    total_users = await db.users.count_documents({})
    handoffs = await db.query_logs.count_documents({"human_handoff": True})

    # Guardrail status breakdown
    pipeline_status = [
        {"$group": {"_id": "$guardrail_status", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    guardrail_stats = await db.query_logs.aggregate(pipeline_status).to_list(20)

    # Query category breakdown
    pipeline_category = [
        {"$group": {"_id": "$query_category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    category_stats = await db.query_logs.aggregate(pipeline_category).to_list(20)

    # Average confidence score
    pipeline_conf = [
        {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence_score"}}}
    ]
    conf_result = await db.query_logs.aggregate(pipeline_conf).to_list(1)
    avg_confidence = conf_result[0]["avg_confidence"] if conf_result else 0.0

    # Average response time
    pipeline_time = [
        {"$group": {"_id": None, "avg_time": {"$avg": "$response_time_ms"}}}
    ]
    time_result = await db.query_logs.aggregate(pipeline_time).to_list(1)
    avg_response_time = time_result[0]["avg_time"] if time_result else 0.0

    return {
        "overview": {
            "total_queries": total_queries,
            "total_documents": total_documents,
            "total_users": total_users,
            "human_handoffs": handoffs,
            "avg_confidence_score": round(avg_confidence, 3),
            "avg_response_time_ms": round(avg_response_time, 1)
        },
        "guardrail_breakdown": {item["_id"]: item["count"] for item in guardrail_stats},
        "category_breakdown": {item["_id"]: item["count"] for item in category_stats}
    }


@router.get("/analytics/trends")
async def get_query_trends(
    days: int = Query(7, ge=1, le=90),
    current_user=Depends(require_officer_or_admin),
    db=Depends(get_database)
):
    """Daily query trend for the last N days."""
    since = datetime.utcnow() - timedelta(days=days)
    pipeline = [
        {"$match": {"timestamp": {"$gte": since.isoformat()}}},
        {"$group": {
            "_id": {"$substr": ["$timestamp", 0, 10]},
            "count": {"$sum": 1},
            "avg_confidence": {"$avg": "$confidence_score"}
        }},
        {"$sort": {"_id": 1}}
    ]
    trends = await db.query_logs.aggregate(pipeline).to_list(90)
    return {"trends": trends}


@router.get("/logs/queries")
async def get_query_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    guardrail_status: Optional[str] = None,
    current_user=Depends(require_officer_or_admin),
    db=Depends(get_database)
):
    """View paginated query logs (Admin/Officer only)."""
    query_filter = {}
    if guardrail_status:
        query_filter["guardrail_status"] = guardrail_status

    total = await db.query_logs.count_documents(query_filter)
    logs = await db.query_logs.find(query_filter) \
        .sort("timestamp", -1) \
        .skip((page - 1) * page_size) \
        .limit(page_size) \
        .to_list(length=page_size)

    return {
        "logs": [
            {
                "session_id": log.get("session_id"),
                "query": log.get("query"),
                "response_preview": log.get("response", "")[:200],
                "query_category": log.get("query_category"),
                "guardrail_status": log.get("guardrail_status"),
                "confidence_score": log.get("confidence_score"),
                "human_handoff": log.get("human_handoff"),
                "response_time_ms": log.get("response_time_ms"),
                "timestamp": log.get("timestamp"),
                "guardrail_violations": log.get("guardrail_violations", [])
            }
            for log in logs
        ],
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.get("/logs/violations")
async def get_guardrail_violations(
    current_user=Depends(require_admin),
    db=Depends(get_database)
):
    """Get all guardrail violation logs (Admin only)."""
    logs = await db.query_logs.find(
        {"guardrail_status": {"$ne": "passed"}}
    ).sort("timestamp", -1).limit(100).to_list(100)

    return {
        "violations": [
            {
                "query": log.get("query"),
                "guardrail_status": log.get("guardrail_status"),
                "violations": log.get("guardrail_violations", []),
                "timestamp": log.get("timestamp")
            }
            for log in logs
        ],
        "total": len(logs)
    }


@router.get("/users")
async def list_users(
    current_user=Depends(require_admin),
    db=Depends(get_database)
):
    """List all users (Admin only)."""
    users = await db.users.find({}, {"password_hash": 0}).to_list(200)
    return {
        "users": [
            {
                "id": u["_id"],
                "name": u["name"],
                "email": u["email"],
                "role": u["role"],
                "is_active": u.get("is_active", True),
                "created_at": u.get("created_at", "").isoformat() if hasattr(u.get("created_at", ""), "isoformat") else ""
            }
            for u in users
        ]
    }
