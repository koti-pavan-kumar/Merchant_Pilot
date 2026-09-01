import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from data.schemas import GrowthRecommendation, MerchantMetrics

class GrowthRecommender:
    def __init__(self, llm_provider: str = "openai"):
        self.llm_provider = llm_provider
        self.recommendation_templates = self._load_recommendation_templates()
        
    def _load_recommendation_templates(self) -> Dict[str, Any]:
        """Load recommendation templates for different scenarios."""
        return {
            "discount_campaign": {
                "action_type": "discount",
                "description": "Create targeted discount campaign",
                "parameters": {
                    "discount_percentage": [5, 10, 15, 20],
                    "duration_days": [7, 14, 30],
                    "target_segments": ["all", "inactive", "high_value"]
                }
            },
            "payment_retry": {
                "action_type": "retry",
                "description": "Optimize payment retry sequence",
                "parameters": {
                    "retry_intervals": [1, 3, 7],  # days
                    "max_retries": [3, 5, 7],
                    "fallback_methods": ["upi", "card", "netbanking"]
                }
            },
            "customer_outreach": {
                "action_type": "outreach",
                "description": "Personalized customer communication",
                "parameters": {
                    "channels": ["email", "sms", "push"],
                    "message_templates": ["win_back", "feedback", "offer"],
                    "timing": ["immediate", "next_day", "weekly"]
                }
            },
            "product_recommendation": {
                "action_type": "campaign",
                "description": "AI-driven product recommendations",
                "parameters": {
                    "recommendation_count": [3, 5, 10],
                    "personalization_level": ["basic", "advanced"],
                    "cross_sell_enabled": [True, False]
                }
            }
        }
    
    def generate_recommendations(
        self, 
        merchant_data: Dict[str, Any], 
        churn_prediction: Dict[str, Any]
    ) -> List[GrowthRecommendation]:
        """Generate personalized growth recommendations for a merchant."""
        recommendations = []
        
        # Analyze merchant's specific issues
        risk_factors = churn_prediction.get("risk_factors", [])
        churn_probability = churn_prediction.get("churn_probability", 0)
        
        # Generate recommendations based on risk factors
        for risk_factor in risk_factors:
            risk_lower = risk_factor.lower()
            
            if "revenue" in risk_lower or "transaction" in risk_lower:
                recommendations.extend(self._generate_revenue_recommendations(merchant_data, churn_probability))
            
            if "failure" in risk_lower or "payment" in risk_lower:
                recommendations.extend(self._generate_payment_recommendations(merchant_data, churn_probability))
            
            if "refund" in risk_lower or "chargeback" in risk_lower:
                recommendations.extend(self._generate_risk_mitigation_recommendations(merchant_data, churn_probability))
        
        # Add general recommendations based on churn probability
        if churn_probability > 0.6:
            recommendations.extend(self._generate_urgent_recommendations(merchant_data, churn_probability))
        
        # Deduplicate and prioritize
        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations = self._prioritize_recommendations(recommendations)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _generate_revenue_recommendations(
        self, 
        merchant_data: Dict[str, Any], 
        churn_probability: float
    ) -> List[GrowthRecommendation]:
        """Generate recommendations for revenue-related issues."""
        recommendations = []
        
        # Discount campaign for low revenue
        if merchant_data.get("total_revenue", 0) < 100000:
            discount_pct = min(20, max(5, int(churn_probability * 25)))
            recommendations.append(GrowthRecommendation(
                recommendation_id=str(uuid.uuid4()),
                merchant_id=merchant_data["merchant_id"],
                action_type="discount",
                priority=1 if churn_probability > 0.7 else 2,
                expected_impact=merchant_data.get("total_revenue", 0) * 0.15,
                reasoning=f"Revenue is below threshold. A {discount_pct}% discount campaign can boost transactions.",
                parameters={
                    "discount_percentage": discount_pct,
                    "duration_days": 14,
                    "target_segments": ["inactive", "high_value"],
                    "budget_limit": merchant_data.get("total_revenue", 0) * 0.1
                },
                created_at=datetime.now()
            ))
        
        # Cross-sell recommendation
        if merchant_data.get("average_order_value", 0) < 1000:
            recommendations.append(GrowthRecommendation(
                recommendation_id=str(uuid.uuid4()),
                merchant_id=merchant_data["merchant_id"],
                action_type="campaign",
                priority=2,
                expected_impact=merchant_data.get("total_revenue", 0) * 0.1,
                reasoning="Low average order value suggests cross-sell opportunity.",
                parameters={
                    "recommendation_count": 5,
                    "personalization_level": "advanced",
                    "cross_sell_enabled": True
                },
                created_at=datetime.now()
            ))
        
        return recommendations
    
    def _generate_payment_recommendations(
        self, 
        merchant_data: Dict[str, Any], 
        churn_probability: float
    ) -> List[GrowthRecommendation]:
        """Generate recommendations for payment-related issues."""
        recommendations = []
        
        # Payment retry optimization
        failure_rate = merchant_data.get("failure_rate", 0)
        if failure_rate > 0.1:
            retry_strategy = "aggressive" if churn_probability > 0.7 else "moderate"
            recommendations.append(GrowthRecommendation(
                recommendation_id=str(uuid.uuid4()),
                merchant_id=merchant_data["merchant_id"],
                action_type="retry",
                priority=1,
                expected_impact=merchant_data.get("total_revenue", 0) * 0.2,
                reasoning=f"High failure rate ({failure_rate:.1%}) indicates payment retry optimization needed.",
                parameters={
                    "retry_intervals": [1, 3, 7] if retry_strategy == "moderate" else [0.5, 1, 2],
                    "max_retries": 5,
                    "fallback_methods": ["upi", "card", "netbanking"],
                    "strategy": retry_strategy
                },
                created_at=datetime.now()
            ))
        
        # Failed payment recovery
        failed_attempts = merchant_data.get("failed_payment_attempts", 0)
        if failed_attempts > 10:
            recommendations.append(GrowthRecommendation(
                recommendation_id=str(uuid.uuid4()),
                merchant_id=merchant_data["merchant_id"],
                action_type="outreach",
                priority=2,
                expected_impact=failed_attempts * 500,  # Assume ₹500 per recovered payment
                reasoning=f" {failed_attempts} failed payment attempts need customer outreach.",
                parameters={
                    "channels": ["email", "sms"],
                    "message_templates": ["payment_reminder", "alternative_method"],
                    "timing": "immediate"
                },
                created_at=datetime.now()
            ))
        
        return recommendations
    
    def _generate_risk_mitigation_recommendations(
        self, 
        merchant_data: Dict[str, Any], 
        churn_probability: float
    ) -> List[GrowthRecommendation]:
        """Generate recommendations for risk mitigation."""
        recommendations = []
        
        # Refund rate optimization
        refund_rate = merchant_data.get("refund_rate", 0)
        if refund_rate > 0.05:
            recommendations.append(GrowthRecommendation(
                recommendation_id=str(uuid.uuid4()),
                merchant_id=merchant_data["merchant_id"],
                action_type="campaign",
                priority=2,
                expected_impact=merchant_data.get("total_revenue", 0) * 0.05,
                reasoning=f"High refund rate ({refund_rate:.1%}) suggests product/service quality issues.",
                parameters={
                    "campaign_type": "quality_feedback",
                    "survey_enabled": True,
                    "incentive_amount": 100,
                    "follow_up_actions": ["discount", "support_ticket"]
                },
                created_at=datetime.now()
            ))
        
        return recommendations
    
    def _generate_urgent_recommendations(
        self, 
        merchant_data: Dict[str, Any], 
        churn_probability: float
    ) -> List[GrowthRecommendation]:
        """Generate urgent recommendations for high churn risk."""
        recommendations = []
        
        # Win-back campaign
        recommendations.append(GrowthRecommendation(
            recommendation_id=str(uuid.uuid4()),
            merchant_id=merchant_data["merchant_id"],
            action_type="outreach",
            priority=1,
            expected_impact=merchant_data.get("total_revenue", 0) * 0.3,
            reasoning=f"Critical churn risk ({churn_probability:.1%}). Urgent win-back campaign needed.",
            parameters={
                "campaign_type": "win_back",
                "channels": ["email", "sms", "push"],
                "personalization": "high",
                "offer_type": "exclusive_discount",
                "discount_percentage": 25,
                "urgency_level": "high"
            },
            created_at=datetime.now()
        ))
        
        return recommendations
    
    def _deduplicate_recommendations(self, recommendations: List[GrowthRecommendation]) -> List[GrowthRecommendation]:
        """Remove duplicate recommendations."""
        seen = set()
        unique_recommendations = []
        
        for rec in recommendations:
            # Create a key based on action type and parameters
            key = f"{rec.action_type}_{json.dumps(rec.parameters, sort_keys=True)}"
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _prioritize_recommendations(self, recommendations: List[GrowthRecommendation]) -> List[GrowthRecommendation]:
        """Sort recommendations by priority and expected impact."""
        return sorted(
            recommendations,
            key=lambda x: (x.priority, -x.expected_impact)
        )
    
    def explain_recommendation(self, recommendation: GrowthRecommendation) -> str:
        """Generate human-readable explanation for a recommendation."""
        explanation_templates = {
            "discount": "Offer a {discount_percentage}% discount for {duration_days} days to {target_segments} customers. Expected to increase transactions by {expected_impact:.0f} INR.",
            "retry": "Implement {strategy} payment retry with intervals of {retry_intervals} days. Expected to recover {expected_impact:.0f} INR from failed payments.",
            "outreach": "Send {message_templates} messages via {channels}. Expected to re-engage {expected_impact:.0f} worth of dormant customers.",
            "campaign": "Launch {campaign_type} campaign with {personalization_level} personalization. Expected to boost revenue by {expected_impact:.0f} INR."
        }
        
        template = explanation_templates.get(recommendation.action_type, "Take action to improve merchant health.")
        
        try:
            return template.format(**recommendation.parameters, expected_impact=recommendation.expected_impact)
        except KeyError:
            return f"Action: {recommendation.action_type}. Expected impact: ₹{recommendation.expected_impact:.0f}"
    
    def get_recommendation_summary(self, recommendations: List[GrowthRecommendation]) -> Dict[str, Any]:
        """Generate summary of all recommendations."""
        if not recommendations:
            return {"total_recommendations": 0, "total_expected_impact": 0}
        
        total_impact = sum(rec.expected_impact for rec in recommendations)
        action_types = [rec.action_type for rec in recommendations]
        
        return {
            "total_recommendations": len(recommendations),
            "total_expected_impact": total_impact,
            "action_type_distribution": {
                action_type: action_types.count(action_type) 
                for action_type in set(action_types)
            },
            "priority_distribution": {
                "high": sum(1 for rec in recommendations if rec.priority == 1),
                "medium": sum(1 for rec in recommendations if rec.priority == 2),
                "low": sum(1 for rec in recommendations if rec.priority >= 3)
            }
        }