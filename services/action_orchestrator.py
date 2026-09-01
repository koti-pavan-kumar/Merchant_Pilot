import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging
from data.schemas import ActionExecution, GrowthRecommendation, AuditLog
from services.razorpay_client import RazorpayClient
from services.audit_trail import AuditTrail

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionOrchestrator:
    def __init__(self):
        self.razorpay_client = RazorpayClient()
        self.audit_trail = AuditTrail()
        self.active_actions: Dict[str, ActionExecution] = {}
        
    async def execute_recommendation(
        self, 
        recommendation: GrowthRecommendation,
        merchant_data: Dict[str, Any]
    ) -> ActionExecution:
        """Execute a growth recommendation."""
        action_id = str(uuid.uuid4())
        
        # Create action execution record
        action = ActionExecution(
            action_id=action_id,
            merchant_id=recommendation.merchant_id,
            recommendation_id=recommendation.recommendation_id,
            action_type=recommendation.action_type,
            status="pending",
            started_at=datetime.now()
        )
        
        self.active_actions[action_id] = action
        
        # Log action start
        self.audit_trail.log_event(
            merchant_id=recommendation.merchant_id,
            event_type="action_start",
            details={
                "action_id": action_id,
                "action_type": recommendation.action_type,
                "parameters": recommendation.parameters
            }
        )
        
        try:
            # Update status to executing
            action.status = "executing"
            
            # Execute based on action type
            if recommendation.action_type == "discount":
                result = await self._execute_discount_campaign(recommendation, merchant_data)
            elif recommendation.action_type == "retry":
                result = await self._execute_payment_retry(recommendation, merchant_data)
            elif recommendation.action_type == "outreach":
                result = await self._execute_customer_outreach(recommendation, merchant_data)
            elif recommendation.action_type == "campaign":
                result = await self._execute_product_campaign(recommendation, merchant_data)
            else:
                result = {"success": False, "error": f"Unknown action type: {recommendation.action_type}"}
            
            # Update action with result
            action.result = result
            action.status = "completed" if result.get("success") else "failed"
            action.completed_at = datetime.now()
            
            # Log completion
            self.audit_trail.log_event(
                merchant_id=recommendation.merchant_id,
                event_type="action_complete",
                details={
                    "action_id": action_id,
                    "status": action.status,
                    "result": result
                },
                severity="info" if result.get("success") else "warning"
            )
            
        except Exception as e:
            action.status = "failed"
            action.error_message = str(e)
            action.completed_at = datetime.now()
            
            # Log error
            self.audit_trail.log_event(
                merchant_id=recommendation.merchant_id,
                event_type="action_error",
                details={
                    "action_id": action_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                severity="error"
            )
        
        return action
    
    async def _execute_discount_campaign(
        self, 
        recommendation: GrowthRecommendation,
        merchant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a discount campaign."""
        params = recommendation.parameters
        
        # Create payment link with discount
        amount = int(merchant_data.get("average_order_value", 1000) * 100)  # Convert to paise
        
        result = self.razorpay_client.execute_with_retry(
            self.razorpay_client.create_payment_link,
            amount=amount,
            description=f"Special offer: {params.get('discount_percentage', 10)}% discount"
        )
        
        if result["success"]:
            return {
                "success": True,
                "payment_link_id": result["payment_link_id"],
                "short_url": result["short_url"],
                "discount_percentage": params.get("discount_percentage"),
                "duration_days": params.get("duration_days"),
                "target_segments": params.get("target_segments")
            }
        else:
            return result
    
    async def _execute_payment_retry(
        self, 
        recommendation: GrowthRecommendation,
        merchant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute payment retry sequence."""
        params = recommendation.parameters
        
        # Simulate retry logic for failed payments
        retry_intervals = params.get("retry_intervals", [1, 3, 7])
        max_retries = params.get("max_retries", 3)
        
        # In real implementation, this would trigger actual retry logic
        # For demo, we'll simulate the retry sequence
        
        retry_results = []
        for i, interval in enumerate(retry_intervals[:max_retries]):
            # Simulate retry attempt
            retry_result = {
                "attempt": i + 1,
                "interval_days": interval,
                "status": "scheduled",
                "scheduled_at": datetime.now().isoformat()
            }
            retry_results.append(retry_result)
        
        return {
            "success": True,
            "retry_strategy": params.get("strategy", "moderate"),
            "retry_schedule": retry_results,
            "fallback_methods": params.get("fallback_methods", []),
            "expected_recovery": recommendation.expected_impact
        }
    
    async def _execute_customer_outreach(
        self, 
        recommendation: GrowthRecommendation,
        merchant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute customer outreach campaign."""
        params = recommendation.parameters
        
        # Create customer record if needed
        customer_result = self.razorpay_client.execute_with_retry(
            self.razorpay_client.create_customer,
            name=merchant_data.get("business_name", "Customer"),
            notes={"campaign": "win_back", "source": "merchant_pilot"}
        )
        
        if customer_result["success"]:
            return {
                "success": True,
                "customer_id": customer_result["customer_id"],
                "channels": params.get("channels", []),
                "message_templates": params.get("message_templates", []),
                "timing": params.get("timing", "immediate"),
                "outreach_scheduled": True
            }
        else:
            return customer_result
    
    async def _execute_product_campaign(
        self, 
        recommendation: GrowthRecommendation,
        merchant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute product recommendation campaign."""
        params = recommendation.parameters
        
        # Create order for campaign
        order_result = self.razorpay_client.execute_with_retry(
            self.razorpay_client.create_order,
            amount=1000,  # Minimum order amount
            receipt=f"campaign_{merchant_data.get('merchant_id', 'unknown')}"
        )
        
        if order_result["success"]:
            return {
                "success": True,
                "order_id": order_result["order_id"],
                "recommendation_count": params.get("recommendation_count", 5),
                "personalization_level": params.get("personalization_level", "basic"),
                "cross_sell_enabled": params.get("cross_sell_enabled", False),
                "campaign_status": "active"
            }
        else:
            return order_result
    
    async def execute_batch(
        self, 
        recommendations: List[GrowthRecommendation],
        merchant_data_list: List[Dict[str, Any]]
    ) -> List[ActionExecution]:
        """Execute multiple recommendations in batch."""
        actions = []
        
        for recommendation, merchant_data in zip(recommendations, merchant_data_list):
            action = await self.execute_recommendation(recommendation, merchant_data)
            actions.append(action)
        
        return actions
    
    def get_action_status(self, action_id: str) -> Optional[ActionExecution]:
        """Get status of an action."""
        return self.active_actions.get(action_id)
    
    def get_merchant_actions(self, merchant_id: str) -> List[ActionExecution]:
        """Get all actions for a merchant."""
        return [
            action for action in self.active_actions.values()
            if action.merchant_id == merchant_id
        ]
    
    def cancel_action(self, action_id: str) -> bool:
        """Cancel a pending action."""
        action = self.active_actions.get(action_id)
        if action and action.status == "pending":
            action.status = "cancelled"
            action.completed_at = datetime.now()
            
            self.audit_trail.log_event(
                merchant_id=action.merchant_id,
                event_type="action_cancelled",
                details={"action_id": action_id}
            )
            return True
        return False
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about action execution."""
        total_actions = len(self.active_actions)
        completed = sum(1 for a in self.active_actions.values() if a.status == "completed")
        failed = sum(1 for a in self.active_actions.values() if a.status == "failed")
        pending = sum(1 for a in self.active_actions.values() if a.status in ["pending", "executing"])
        
        return {
            "total_actions": total_actions,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "success_rate": completed / total_actions if total_actions > 0 else 0,
            "active_merchants": len(set(a.merchant_id for a in self.active_actions.values()))
        }