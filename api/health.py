from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

router = APIRouter()

# In-memory storage for demo (would use database in production)
merchant_data_store: Dict[str, Dict[str, Any]] = {}
churn_predictions: Dict[str, Dict[str, Any]] = {}
health_scores: Dict[str, Dict[str, Any]] = {}

@router.get("/merchants/{merchant_id}")
async def get_merchant_health(merchant_id: str) -> Dict[str, Any]:
    """Get health assessment for a specific merchant."""
    if merchant_id not in merchant_data_store:
        raise HTTPException(status_code=404, detail=f"Merchant {merchant_id} not found")
    
    merchant_data = merchant_data_store[merchant_id]
    prediction = churn_predictions.get(merchant_id, {})
    health_score = health_scores.get(merchant_id, {})
    
    return {
        "merchant_id": merchant_id,
        "business_name": merchant_data.get("business_name"),
        "status": merchant_data.get("status"),
        "health_score": health_score,
        "churn_prediction": prediction,
        "last_updated": datetime.now().isoformat()
    }

@router.get("/merchants")
async def list_merchants(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """List merchants with optional filtering."""
    merchants = list(merchant_data_store.values())
    
    if status:
        merchants = [m for m in merchants if m.get("status") == status]
    
    total = len(merchants)
    merchants = merchants[offset:offset + limit]
    
    return {
        "total": total,
        "merchants": merchants,
        "limit": limit,
        "offset": offset
    }

@router.get("/merchants/{merchant_id}/risk-factors")
async def get_risk_factors(merchant_id: str) -> Dict[str, Any]:
    """Get detailed risk factors for a merchant."""
    if merchant_id not in merchant_data_store:
        raise HTTPException(status_code=404, detail=f"Merchant {merchant_id} not found")
    
    merchant_data = merchant_data_store[merchant_id]
    prediction = churn_predictions.get(merchant_id, {})
    
    risk_factors = []
    
    # Analyze merchant metrics for risk factors
    if merchant_data.get("failure_rate", 0) > 0.1:
        risk_factors.append({
            "factor": "High payment failure rate",
            "value": merchant_data["failure_rate"],
            "threshold": 0.1,
            "severity": "high"
        })
    
    if merchant_data.get("refund_rate", 0) > 0.05:
        risk_factors.append({
            "factor": "Elevated refund rate",
            "value": merchant_data["refund_rate"],
            "threshold": 0.05,
            "severity": "medium"
        })
    
    if merchant_data.get("days_since_last_transaction", 0) > 30:
        risk_factors.append({
            "factor": "Extended inactivity period",
            "value": merchant_data["days_since_last_transaction"],
            "threshold": 30,
            "severity": "high"
        })
    
    if merchant_data.get("chargeback_count", 0) > 5:
        risk_factors.append({
            "factor": "Excessive chargebacks",
            "value": merchant_data["chargeback_count"],
            "threshold": 5,
            "severity": "critical"
        })
    
    return {
        "merchant_id": merchant_id,
        "risk_factors": risk_factors,
        "total_risk_factors": len(risk_factors),
        "overall_risk_level": prediction.get("risk_level", "unknown")
    }

@router.post("/merchants/{merchant_id}/update-metrics")
async def update_merchant_metrics(merchant_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Update merchant metrics (for testing)."""
    if merchant_id in merchant_data_store:
        merchant_data_store[merchant_id].update(metrics)
    else:
        merchant_data_store[merchant_id] = metrics
    
    return {
        "merchant_id": merchant_id,
        "updated": True,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/merchants/{merchant_id}/transactions")
async def get_merchant_transactions(
    merchant_id: str,
    days: int = Query(30, ge=1, le=90),
    limit: int = Query(100, ge=1, le=500)
) -> Dict[str, Any]:
    """Get recent transactions for a merchant."""
    if merchant_id not in merchant_data_store:
        raise HTTPException(status_code=404, detail=f"Merchant {merchant_id} not found")
    
    # In real implementation, this would query transaction database
    # For demo, return simulated transactions
    transactions = []
    for i in range(min(limit, 10)):
        transactions.append({
            "transaction_id": f"T{merchant_id}{i+1:03d}",
            "amount": 1000 + i * 100,
            "status": "captured" if i % 5 != 0 else "failed",
            "created_at": datetime.now().isoformat(),
            "payment_method": ["upi", "card", "netbanking"][i % 3]
        })
    
    return {
        "merchant_id": merchant_id,
        "transactions": transactions,
        "count": len(transactions),
        "period_days": days
    }

@router.get("/health/summary")
async def get_health_summary() -> Dict[str, Any]:
    """Get overall health summary across all merchants."""
    total_merchants = len(merchant_data_store)
    
    status_counts = {}
    risk_level_counts = {}
    
    for merchant in merchant_data_store.values():
        status = merchant.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        
        merchant_id = merchant.get("merchant_id")
        if merchant_id in churn_predictions:
            risk_level = churn_predictions[merchant_id].get("risk_level", "unknown")
            risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1
    
    return {
        "total_merchants": total_merchants,
        "status_distribution": status_counts,
        "risk_level_distribution": risk_level_counts,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health/trends")
async def get_health_trends(days: int = Query(7, ge=1, le=30)) -> Dict[str, Any]:
    """Get health trends over time."""
    # In real implementation, this would query historical data
    # For demo, return simulated trends
    trends = []
    for i in range(days):
        trends.append({
            "date": datetime.now().isoformat(),
            "total_merchants": len(merchant_data_store),
            "at_risk_count": sum(1 for m in merchant_data_store.values() if m.get("status") == "at_risk"),
            "churned_count": sum(1 for m in merchant_data_store.values() if m.get("status") == "churned"),
            "average_health_score": 75.5  # Simulated
        })
    
    return {
        "period_days": days,
        "trends": trends,
        "summary": {
            "average_health_score": 75.5,
            "trend_direction": "stable"
        }
    }