"""
LLM-Powered Merchant Analyst

This is the REAL AI behind MerchantPilot.
Instead of hardcoded if/else rules, it uses an LLM to:
1. Analyze a merchant's transaction patterns
2. Identify root causes of churn risk
3. Generate personalized recovery strategies
4. Explain its reasoning in plain English

Falls back to rule-based logic when no LLM API key is configured.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMMerchantAnalyst:
    """AI-powered merchant analysis using LLM reasoning."""

    def __init__(self):
        self.client = None
        self.model = "gpt-4o-mini"  # Cost-effective for this use case

        # Initialize OpenAI if key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("[LIVE] OpenAI LLM initialized")
            except Exception as e:
                logger.warning(f"[WARN] OpenAI init failed: {e}")
        else:
            logger.info("[FALLBACK] No OpenAI key — using rule-based analysis")

    def analyze_merchant(
        self,
        merchant_data: Dict[str, Any],
        churn_prediction: Dict[str, Any],
        transaction_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a merchant using LLM and return intelligent recommendations.

        Returns:
            {
                "analysis": "LLM's analysis of the merchant",
                "risk_assessment": "Detailed risk breakdown",
                "recommendations": [...],
                "explanation": "Why these actions will help",
                "confidence": 0.0-1.0,
                "model_used": "gpt-4o-mini" or "rule-based"
            }
        """
        # Build context for the LLM
        context = self._build_context(merchant_data, churn_prediction, transaction_summary)

        if self.client:
            return self._analyze_with_llm(context, merchant_data)
        else:
            return self._analyze_with_rules(context, merchant_data)

    def _build_context(
        self,
        merchant_data: Dict[str, Any],
        churn_prediction: Dict[str, Any],
        transaction_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a detailed context string for the LLM."""

        context = f"""MERCHANT ANALYSIS REQUEST
========================

Merchant Profile:
- ID: {merchant_data.get('merchant_id', 'Unknown')}
- Business: {merchant_data.get('business_name', 'Unknown')}
- Category: {merchant_data.get('category', 'Unknown')}
- Status: {merchant_data.get('status', 'Unknown')}

Financial Metrics:
- Total Revenue: ₹{merchant_data.get('total_revenue', 0):,.0f}
- Average Order Value: ₹{merchant_data.get('average_order_value', 0):,.0f}
- Transaction Count: {merchant_data.get('transaction_count', 0)}

Risk Indicators:
- Failure Rate: {merchant_data.get('failure_rate', 0):.1%}
- Refund Rate: {merchant_data.get('refund_rate', 0):.1%}
- Chargeback Count: {merchant_data.get('chargeback_count', 0)}
- Dispute Rate: {merchant_data.get('dispute_rate', 0):.1%}
- Failed Payment Attempts: {merchant_data.get('failed_payment_attempts', 0)}

Activity:
- Days Since Last Transaction: {merchant_data.get('days_since_last_transaction', 0)}
- Revenue Growth Rate: {merchant_data.get('revenue_growth_rate', 0):.1%}

Churn Prediction:
- Churn Probability: {churn_prediction.get('churn_probability', 0):.1%}
- Risk Level: {churn_prediction.get('risk_level', 'unknown')}
- Risk Factors: {', '.join(churn_prediction.get('risk_factors', ['None identified']))}
"""
        if transaction_summary:
            context += f"""
Transaction Summary:
- Last 30 Days Transactions: {transaction_summary.get('count_30d', 'N/A')}
- Last 30 Days Revenue: ₹{transaction_summary.get('revenue_30d', 0):,.0f}
- Success Rate: {transaction_summary.get('success_rate_30d', 0):.1%}
- Average Daily Revenue: ₹{transaction_summary.get('avg_daily_revenue', 0):,.0f}
"""
        return context

    def _analyze_with_llm(
        self, context: str, merchant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use OpenAI GPT to analyze the merchant."""

        system_prompt = """You are MerchantPilot AI, an expert financial analyst for Razorpay merchants.

Your job is to analyze merchant health data and recommend specific, actionable recovery strategies.

For each merchant, you must:
1. Identify the TOP 3 most critical issues (ranked by revenue impact)
2. Recommend ONE specific action for each issue
3. Explain WHY this action will work (with data reasoning)
4. Estimate the expected revenue recovery in INR
5. Assign a confidence score (0.0-1.0) based on data quality

IMPORTANT RULES:
- Be specific: "Send 10% discount via email to inactive users" NOT "Consider a discount"
- Be honest: If data is insufficient, say so with low confidence
- Think like a business owner: What would YOU do to save this merchant?
- Every recommendation must be measurable and bounded

Return your response as valid JSON with this exact structure:
{
    "analysis": "2-3 sentence executive summary of merchant health",
    "risk_assessment": "Detailed breakdown of what's wrong and why",
    "recommendations": [
        {
            "action_type": "discount|retry|outreach|campaign",
            "title": "Short action title",
            "description": "Detailed action to take",
            "parameters": {"specific": "action parameters"},
            "expected_impact": 5000,
            "reasoning": "Why this specific action will work",
            "confidence": 0.8,
            "urgency": "immediate|this_week|this_month"
        }
    ],
    "explanation": "Overall strategy explanation for the merchant",
    "confidence": 0.75
}
"""

        user_prompt = f"""Analyze this Razorpay merchant and recommend recovery actions:

{context}

Provide your analysis as JSON. Be specific, data-driven, and actionable."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Low temperature for consistent analysis
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            # Ensure required fields exist
            result.setdefault("analysis", "Analysis completed")
            result.setdefault("risk_assessment", "Risk assessed")
            result.setdefault("recommendations", [])
            result.setdefault("explanation", "Strategy formulated")
            result.setdefault("confidence", 0.7)
            result["model_used"] = self.model

            logger.info(
                f"[LIVE] LLM analysis complete — "
                f"{len(result['recommendations'])} recommendations, "
                f"confidence: {result['confidence']}"
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] LLM returned invalid JSON: {e}")
            return self._analyze_with_rules(context, merchant_data)

        except Exception as e:
            logger.error(f"[ERROR] LLM analysis failed: {e}")
            return self._analyze_with_rules(context, merchant_data)

    def _analyze_with_rules(
        self, context: str, merchant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback: rule-based analysis when LLM is unavailable."""

        churn_prob = merchant_data.get("failure_rate", 0)
        revenue = merchant_data.get("total_revenue", 0)
        failure_rate = merchant_data.get("failure_rate", 0)
        refund_rate = merchant_data.get("refund_rate", 0)
        days_inactive = merchant_data.get("days_since_last_transaction", 0)
        failed_attempts = merchant_data.get("failed_payment_attempts", 0)

        recommendations = []

        # Rule 1: High failure rate → payment retry optimization
        if failure_rate > 0.1:
            impact = revenue * 0.2
            recommendations.append({
                "action_type": "retry",
                "title": "Optimize Payment Retry Sequence",
                "description": f"Implement aggressive retry strategy for {failed_attempts} failed payments. Use UPI → Card → Netbanking fallback with 1h/4h/24h intervals.",
                "parameters": {
                    "retry_intervals_hours": [1, 4, 24],
                    "max_retries": 5,
                    "fallback_methods": ["upi", "card", "netbanking"],
                    "strategy": "aggressive"
                },
                "expected_impact": int(impact),
                "reasoning": f"Failure rate is {failure_rate:.1%} — optimizing retry timing can recover ~20% of failed payments.",
                "confidence": 0.75,
                "urgency": "immediate"
            })

        # Rule 2: High refund rate → quality feedback campaign
        if refund_rate > 0.05:
            impact = revenue * 0.1
            recommendations.append({
                "action_type": "campaign",
                "title": "Quality Feedback & Retention Campaign",
                "description": "Launch feedback survey with ₹100 incentive. Identify root causes of refunds and address them proactively.",
                "parameters": {
                    "campaign_type": "quality_feedback",
                    "survey_enabled": True,
                    "incentive_amount": 100,
                    "channels": ["email", "sms"]
                },
                "expected_impact": int(impact),
                "reasoning": f"Refund rate is {refund_rate:.1%} — understanding WHY customers refund prevents future losses.",
                "confidence": 0.65,
                "urgency": "this_week"
            })

        # Rule 3: Inactive merchant → win-back campaign
        if days_inactive > 30:
            impact = revenue * 0.3
            recommendations.append({
                "action_type": "outreach",
                "title": "Merchant Win-Back Campaign",
                "description": f"Merchant inactive for {days_inactive} days. Send personalized win-back email with exclusive 15% discount offer.",
                "parameters": {
                    "channels": ["email", "sms", "push"],
                    "offer_type": "exclusive_discount",
                    "discount_percentage": 15,
                    "message_template": "win_back",
                    "urgency_level": "high"
                },
                "expected_impact": int(impact),
                "reasoning": f"Inactive for {days_inactive} days — urgent outreach needed to prevent permanent churn.",
                "confidence": 0.7,
                "urgency": "immediate"
            })

        # Rule 4: Low revenue → cross-sell campaign
        if revenue < 100000 and not recommendations:
            impact = revenue * 0.15
            recommendations.append({
                "action_type": "discount",
                "title": "Revenue Boost Discount Campaign",
                "description": "Create 10% discount campaign targeting inactive and high-value customers for 14 days.",
                "parameters": {
                    "discount_percentage": 10,
                    "duration_days": 14,
                    "target_segments": ["inactive", "high_value"],
                    "budget_limit": int(revenue * 0.1)
                },
                "expected_impact": int(impact),
                "reasoning": f"Revenue is ₹{revenue:,.0f} — discount campaign can stimulate transaction volume.",
                "confidence": 0.6,
                "urgency": "this_week"
            })

        # Rule 5: Failed payment outreach
        if failed_attempts > 10:
            impact = failed_attempts * 500
            recommendations.append({
                "action_type": "outreach",
                "title": "Failed Payment Customer Outreach",
                "description": f"Contact {failed_attempts} customers with failed payments. Offer alternative payment methods and assistance.",
                "parameters": {
                    "channels": ["email", "sms"],
                    "message_templates": ["payment_reminder", "alternative_method"],
                    "timing": "immediate",
                    "customer_count": failed_attempts
                },
                "expected_impact": int(impact),
                "reasoning": f"{failed_attempts} failed attempts represent ₹{impact:,.0f} in at-risk revenue.",
                "confidence": 0.7,
                "urgency": "immediate"
            })

        # If no specific rules triggered, add a general health check
        if not recommendations:
            recommendations.append({
                "action_type": "campaign",
                "title": "Merchant Health Check & Optimization",
                "description": "Schedule a health review. Analyze transaction patterns and optimize payment flow.",
                "parameters": {
                    "campaign_type": "health_check",
                    "analysis_depth": "comprehensive"
                },
                "expected_impact": int(revenue * 0.05),
                "reasoning": "General health check to identify hidden optimization opportunities.",
                "confidence": 0.5,
                "urgency": "this_month"
            })

        # Sort by expected impact
        recommendations.sort(key=lambda x: x["expected_impact"], reverse=True)

        total_impact = sum(r["expected_impact"] for r in recommendations)

        return {
            "analysis": f"Merchant has a {'high' if churn_prob > 0.6 else 'moderate'} churn risk "
                       f"with Rs.{revenue:,.0f} revenue. "
                       f"Key issues: {', '.join(merchant_data.get('risk_factors', ['general health']))}.",
            "risk_assessment": f"Failure rate: {failure_rate:.1%}, "
                             f"Refund rate: {refund_rate:.1%}, "
                             f"Days inactive: {days_inactive}, "
                             f"Failed attempts: {failed_attempts}",
            "recommendations": recommendations[:3],
            "explanation": f"Generated {len(recommendations)} recommendations targeting "
                         f"Rs.{total_impact:,.0f} in expected revenue recovery.",
            "confidence": 0.6,
            "model_used": "rule-based (no LLM key configured)",
        }

    def analyze_batch(
        self,
        merchants: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Analyze multiple merchants efficiently."""
        results = []
        for merchant, prediction in zip(merchants, predictions):
            result = self.analyze_merchant(merchant, prediction)
            results.append(result)
        return results
