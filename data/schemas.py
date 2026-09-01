from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MerchantStatus(str, Enum):
    ACTIVE = "active"
    AT_RISK = "at_risk"
    CHURNED = "churned"
    RECOVERED = "recovered"

class Transaction(BaseModel):
    transaction_id: str
    merchant_id: str
    amount: float
    currency: str = "INR"
    status: str  # "captured", "failed", "refunded", "pending"
    payment_method: str
    created_at: datetime
    failure_reason: Optional[str] = None

class MerchantMetrics(BaseModel):
    merchant_id: str
    business_name: str
    category: str
    registration_date: datetime
    last_transaction_date: Optional[datetime] = None
    
    # Revenue metrics (last 30 days)
    total_revenue: float = 0.0
    transaction_count: int = 0
    average_order_value: float = 0.0
    refund_rate: float = 0.0
    failure_rate: float = 0.0
    
    # Engagement metrics
    days_since_last_transaction: int = 0
    weekly_transaction_trend: List[float] = []  # Last 8 weeks
    revenue_growth_rate: float = 0.0
    
    # Risk indicators
    chargeback_count: int = 0
    dispute_rate: float = 0.0
    failed_payment_attempts: int = 0

class ChurnPrediction(BaseModel):
    merchant_id: str
    churn_probability: float
    risk_factors: List[str]
    confidence_score: float
    prediction_timestamp: datetime
    model_version: str = "1.0"

class GrowthRecommendation(BaseModel):
    recommendation_id: str
    merchant_id: str
    action_type: str  # "discount", "campaign", "retry", "outreach"
    priority: int  # 1-5, 1 being highest
    expected_impact: float  # Expected revenue recovery in INR
    reasoning: str
    parameters: Dict[str, Any]
    created_at: datetime

class ActionExecution(BaseModel):
    action_id: str
    merchant_id: str
    recommendation_id: str
    action_type: str
    status: str  # "pending", "executing", "completed", "failed"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0

class AuditLog(BaseModel):
    log_id: str
    merchant_id: str
    action_id: Optional[str] = None
    event_type: str  # "prediction", "recommendation", "action", "error"
    details: Dict[str, Any]
    timestamp: datetime
    severity: str = "info"  # "info", "warning", "error"

class MerchantHealthScore(BaseModel):
    merchant_id: str
    health_score: float  # 0-100
    risk_level: str  # "low", "medium", "high", "critical"
    contributing_factors: List[str]
    last_updated: datetime
    trend: str  # "improving", "stable", "declining"