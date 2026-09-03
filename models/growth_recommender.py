"""
Growth Recommender — Now powered by LLM analysis.

This module used to be pure if/else rules.
Now it delegates to LLMMerchantAnalyst for real AI reasoning,
and wraps the results in the GrowthRecommendation schema.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schemas import GrowthRecommendation
from services.llm_analyst import LLMMerchantAnalyst


class GrowthRecommender:
    """AI-powered growth recommender using LLM analysis."""

    def __init__(self, llm_provider: str = "openai"):
        self.llm_provider = llm_provider
        self.analyst = LLMMerchantAnalyst()

    def generate_recommendations(
        self,
        merchant_data: Dict[str, Any],
        churn_prediction: Dict[str, Any],
        transaction_summary: Optional[Dict[str, Any]] = None,
    ) -> List[GrowthRecommendation]:
        """
        Generate personalized growth recommendations using LLM analysis.

        This is the main entry point. It:
        1. Sends merchant data to the LLM analyst
        2. Gets back intelligent recommendations with reasoning
        3. Wraps them in GrowthRecommendation objects
        """
        # Get LLM analysis
        analysis = self.analyst.analyze_merchant(
            merchant_data=merchant_data,
            churn_prediction=churn_prediction,
            transaction_summary=transaction_summary,
        )

        # Convert LLM recommendations to GrowthRecommendation objects
        recommendations = []
        for i, rec in enumerate(analysis.get("recommendations", [])):
            recommendation = GrowthRecommendation(
                recommendation_id=str(uuid.uuid4()),
                merchant_id=merchant_data.get("merchant_id", "unknown"),
                action_type=rec.get("action_type", "campaign"),
                priority=i + 1,
                expected_impact=rec.get("expected_impact", 0),
                reasoning=self._build_reasoning(rec, analysis),
                parameters=rec.get("parameters", {}),
                created_at=datetime.now(),
            )
            recommendations.append(recommendation)

        return recommendations

    def _build_reasoning(self, rec: Dict, analysis: Dict) -> str:
        """Build a detailed reasoning string combining LLM analysis."""
        parts = []

        # Add the specific action reasoning
        if rec.get("reasoning"):
            parts.append(rec["reasoning"])

        # Add confidence and urgency context
        confidence = rec.get("confidence", 0)
        urgency = rec.get("urgency", "this_week")

        if confidence >= 0.8:
            parts.append("High confidence recommendation based on clear data patterns.")
        elif confidence >= 0.6:
            parts.append("Moderate confidence — data supports this action but monitoring recommended.")
        else:
            parts.append("Lower confidence — recommended as exploratory action with measurement.")

        if urgency == "immediate":
            parts.append("⚠️ URGENT: Action needed within 24 hours.")
        elif urgency == "this_week":
            parts.append("Schedule for this week for optimal impact.")

        return " ".join(parts)

    def explain_recommendation(self, recommendation: GrowthRecommendation) -> str:
        """Generate human-readable explanation for a recommendation."""
        # The reasoning field already contains the LLM explanation
        return recommendation.reasoning

    def get_recommendation_summary(
        self, recommendations: List[GrowthRecommendation]
    ) -> Dict[str, Any]:
        """Generate summary of all recommendations."""
        if not recommendations:
            return {
                "total_recommendations": 0,
                "total_expected_impact": 0,
                "model_used": self.analyst.client and "llm" or "rule-based",
            }

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
                "low": sum(1 for rec in recommendations if rec.priority >= 3),
            },
            "model_used": self.analyst.client and "llm" or "rule-based",
        }
