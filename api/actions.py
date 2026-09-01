from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from pydantic import BaseModel

router = APIRouter()

# Import services
from services.action_orchestrator import ActionOrchestrator
from services.audit_trail import AuditTrail
from models.growth_recommender import GrowthRecommender

# Initialize services
action_orchestrator = ActionOrchestrator()
audit_trail = AuditTrail()
growth_recommender = GrowthRecommender()

class RecommendationRequest(BaseModel):
    merchant_id: str
    merchant_data: Dict[str, Any]
    churn_prediction: Dict[str, Any]

class ActionExecutionRequest(BaseModel):
    merchant_id: str
    action_type: str
    parameters: Dict[str, Any]

@router.post("/recommendations/generate")
async def generate_recommendations(request: RecommendationRequest) -> Dict[str, Any]:
    """Generate growth recommendations for a merchant."""
    try:
        recommendations = growth_recommender.generate_recommendations(
            request.merchant_data,
            request.churn_prediction
        )
        
        # Log recommendation generation
        audit_trail.log_event(
            merchant_id=request.merchant_id,
            event_type="recommendations_generated",
            details={
                "recommendation_count": len(recommendations),
                "merchant_status": request.merchant_data.get("status")
            }
        )
        
        # Get summary
        summary = growth_recommender.get_recommendation_summary(recommendations)
        
        return {
            "merchant_id": request.merchant_id,
            "recommendations": [rec.dict() for rec in recommendations],
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/actions/execute")
async def execute_action(
    request: ActionExecutionRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Execute a growth action for a merchant."""
    try:
        # Create a simple recommendation from the request
        from models.schemas import GrowthRecommendation
        
        recommendation = GrowthRecommendation(
            recommendation_id=str(uuid.uuid4()),
            merchant_id=request.merchant_id,
            action_type=request.action_type,
            priority=1,
            expected_impact=0,
            reasoning="Manual action execution",
            parameters=request.parameters,
            created_at=datetime.now()
        )
        
        # Execute the action
        action = await action_orchestrator.execute_recommendation(
            recommendation,
            request.merchant_data if hasattr(request, 'merchant_data') else {}
        )
        
        return {
            "action_id": action.action_id,
            "status": action.status,
            "result": action.result,
            "executed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/actions/batch-execute")
async def batch_execute_actions(
    requests: List[ActionExecutionRequest],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Execute multiple actions in batch."""
    try:
        actions = []
        
        for request in requests:
            from models.schemas import GrowthRecommendation
            
            recommendation = GrowthRecommendation(
                recommendation_id=str(uuid.uuid4()),
                merchant_id=request.merchant_id,
                action_type=request.action_type,
                priority=1,
                expected_impact=0,
                reasoning="Batch action execution",
                parameters=request.parameters,
                created_at=datetime.now()
            )
            
            action = await action_orchestrator.execute_recommendation(
                recommendation,
                request.merchant_data if hasattr(request, 'merchant_data') else {}
            )
            actions.append(action)
        
        return {
            "batch_id": str(uuid.uuid4()),
            "total_actions": len(actions),
            "actions": [
                {
                    "action_id": a.action_id,
                    "status": a.status,
                    "result": a.result
                }
                for a in actions
            ],
            "executed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/actions/{action_id}")
async def get_action_status(action_id: str) -> Dict[str, Any]:
    """Get status of a specific action."""
    action = action_orchestrator.get_action_status(action_id)
    
    if not action:
        raise HTTPException(status_code=404, detail=f"Action {action_id} not found")
    
    return {
        "action_id": action.action_id,
        "merchant_id": action.merchant_id,
        "action_type": action.action_type,
        "status": action.status,
        "result": action.result,
        "error_message": action.error_message,
        "started_at": action.started_at.isoformat() if action.started_at else None,
        "completed_at": action.completed_at.isoformat() if action.completed_at else None
    }

@router.get("/actions/merchant/{merchant_id}")
async def get_merchant_actions(merchant_id: str) -> Dict[str, Any]:
    """Get all actions for a merchant."""
    actions = action_orchestrator.get_merchant_actions(merchant_id)
    
    return {
        "merchant_id": merchant_id,
        "total_actions": len(actions),
        "actions": [
            {
                "action_id": a.action_id,
                "action_type": a.action_type,
                "status": a.status,
                "created_at": a.started_at.isoformat() if a.started_at else None
            }
            for a in actions
        ]
    }

@router.post("/actions/{action_id}/cancel")
async def cancel_action(action_id: str) -> Dict[str, Any]:
    """Cancel a pending action."""
    success = action_orchestrator.cancel_action(action_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Action cannot be cancelled")
    
    return {
        "action_id": action_id,
        "cancelled": True,
        "cancelled_at": datetime.now().isoformat()
    }

@router.get("/actions/stats")
async def get_action_stats() -> Dict[str, Any]:
    """Get action execution statistics."""
    return action_orchestrator.get_execution_stats()

@router.get("/audit/{merchant_id}")
async def get_merchant_audit_trail(
    merchant_id: str,
    limit: int = 100,
    event_type: Optional[str] = None
) -> Dict[str, Any]:
    """Get audit trail for a merchant."""
    logs = audit_trail.get_merchant_logs(merchant_id, limit, event_type)
    summary = audit_trail.get_merchant_summary(merchant_id)
    
    return {
        "merchant_id": merchant_id,
        "summary": summary,
        "logs": [
            {
                "log_id": log.log_id,
                "event_type": log.event_type,
                "details": log.details,
                "timestamp": log.timestamp.isoformat(),
                "severity": log.severity
            }
            for log in logs
        ]
    }

@router.get("/audit/system/summary")
async def get_system_audit_summary() -> Dict[str, Any]:
    """Get system-wide audit summary."""
    return audit_trail.get_system_summary()

@router.get("/audit/recent")
async def get_recent_audit_logs(limit: int = 50) -> Dict[str, Any]:
    """Get recent audit logs across all merchants."""
    logs = audit_trail.get_recent_logs(limit)
    
    return {
        "total_logs": len(logs),
        "logs": [
            {
                "log_id": log.log_id,
                "merchant_id": log.merchant_id,
                "event_type": log.event_type,
                "details": log.details,
                "timestamp": log.timestamp.isoformat(),
                "severity": log.severity
            }
            for log in logs
        ]
    }

@router.get("/audit/errors")
async def get_error_audit_logs(limit: int = 50) -> Dict[str, Any]:
    """Get recent error audit logs."""
    logs = audit_trail.get_error_logs(limit)
    
    return {
        "total_error_logs": len(logs),
        "logs": [
            {
                "log_id": log.log_id,
                "merchant_id": log.merchant_id,
                "event_type": log.event_type,
                "details": log.details,
                "timestamp": log.timestamp.isoformat(),
                "severity": log.severity
            }
            for log in logs
        ]
    }