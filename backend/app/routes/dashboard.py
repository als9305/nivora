from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional
from app.database.repositories import AuthLogRepository

router = APIRouter(prefix="/api/dashboard", tags=["Security Dashboard"])

@router.get("/stats")
async def get_dashboard_stats():
    stats = await AuthLogRepository.get_dashboard_stats()
    return {
        "success": True,
        "data": stats
    }

@router.get("/history")
async def get_auth_history(
    limit: int = Query(50, ge=1, le=500),
    decision: Optional[str] = Query("ALL", description="Filter by ALL, ALLOW, STEP-UP, BLOCK, DEEPFAKE")
):
    logs = await AuthLogRepository.get_all_logs(limit=limit, decision_filter=decision)
    return {
        "success": True,
        "total": len(logs),
        "filter": decision,
        "logs": logs
    }

@router.get("/incidents")
async def get_incidents():
    # Incidents are blocked attempts or deepfake attacks
    all_logs = await AuthLogRepository.get_all_logs(limit=100)
    incidents = [
        l for l in all_logs
        if l.get("decision") == "BLOCK" or l.get("deepfake_probability", 0) > 0.5
    ]
    return {
        "success": True,
        "total": len(incidents),
        "incidents": incidents
    }

@router.get("/incident/{log_id}")
async def get_incident_by_id(log_id: str):
    log_entry = await AuthLogRepository.get_log_by_id(log_id)
    if not log_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"success": False, "message": f"Incident with ID {log_id} not found."}
        )
    return {
        "success": True,
        "incident": log_entry
    }
