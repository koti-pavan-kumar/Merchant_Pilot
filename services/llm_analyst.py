"""
LLM-Powered Merchant Analyst

This is the REAL AI behind MerchantPilot.
Instead of hardcoded if/else rules, it uses an LLM to:
1. Analyze a merchant's transaction patterns
2. Identify root causes of churn risk
3. Generate personalized recovery strategies
4. Explain its reasoning in plain English

Supports: Gemini (primary) → OpenAI (fallback) → Rule-based (ultimate fallback)
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Ensure .env is loaded
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Try to import Gemini (new SDK first, then legacy)
try:
    from google import genai as genai_new
    GEMINI_AVAILABLE = True
    GEMINI_NEW_SDK = True
except ImportError:
    GEMINI_NEW_SDK = False
    try:
        import google.generativeai as genai_new
        GEMINI_AVAILABLE = True
        GEMINI_NEW_SDK = False
    except ImportError:
        GEMINI_AVAILABLE = False

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


SYSTEM_PROMPT = """You are MerchantPilot AI, an expert financial analyst for Razorpay merchants.

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
- Use INR (Indian Rupees) for all monetary values
- Use plain text for rupee symbol (write Rs. not the symbol)

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


class LLMMerchantAnalyst:
    """AI-powered merchant analysis using LLM reasoning."""

    def __init__(self):
        self.client = None
        self.gemini_model = None
        self.provider = "rule-based"
        self.model_name = "unknown"

        # Priority 1: Try Gemini (free tier available)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if GEMINI_AVAILABLE and gemini_key:
            try:
                if GEMINI_NEW_SDK:
                    self.gemini_client = genai_new.Client(api_key=gemini_key)
                    self.gemini_model = "gemini-3.6-flash"
                else:
                    genai_new.configure(api_key=gemini_key)
                    self.gemini_client = None
                    self.gemini_model = genai_new.GenerativeModel("gemini-3.6-flash")
                self.provider = "gemini"
                self.model_name = "gemini-3.6-flash"
                logger.info("[LIVE] Gemini LLM initialized")
                return
            except Exception as e:
                logger.warning(f"[WARN] Gemini init failed: {e}")

        # Priority 2: Try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and openai_key:
            try:
                self.client = OpenAI(api_key=openai_key)
                self.provider = "openai"
                self.model_name = "gpt-4o-mini"
                logger.info("[LIVE] OpenAI LLM initialized")
                return
            except Exception as e:
                logger.warning(f"[WARN] OpenAI init failed: {e}")

        # Priority 3: Rule-based fallback
        logger.info("[FALLBACK] No LLM key — using rule-based analysis")

    def analyze_merchant(
        self,
        merchant_data: Dict[str, Any],
        churn_prediction: Dict[str, Any],
        transaction_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a merchant using LLM and return intelligent recommendations."""
        context = self._build_context(merchant_data, churn_prediction, transaction_summary)

        if self.provider == "gemini":
            return self._analyze_with_gemini(context, merchant_data)
        elif self.provider == "openai":
            return self._analyze_with_openai(context, merchant_data)
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
- Total Revenue: Rs.{merchant_data.get('total_revenue', 0):,.0f}
- Average Order Value: Rs.{merchant_data.get('average_order_value', 0):,.0f}
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
- Last 30 Days Revenue: Rs.{transaction_summary.get('revenue_30d', 0):,.0f}
- Success Rate: {transaction_summary.get('success_rate_30d', 0):.1%}
- Average Daily Revenue: Rs.{transaction_summary.get('avg_daily_revenue', 0):,.0f}
"""
        return context

    def _analyze_with_gemini(
        self, context: str, merchant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use Google Gemini to analyze the merchant."""
        user_prompt = f"{SYSTEM_PROMPT}\n\nAnalyze this Razorpay merchant and recommend recovery actions:\n\n{context}\n\nProvide your analysis as JSON. Be specific, data-driven, and actionable."

        try:
            if hasattr(self, 'gemini_client') and self.gemini_client and GEMINI_NEW_SDK:
                # New SDK
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=user_prompt,
                )
                result_text = response.text
            else:
                # Legacy SDK
                response = self.gemini_model.generate_content(user_prompt)
                result_text = response.text

            # Clean up markdown code fences if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text.strip())

            result.setdefault("analysis", "Analysis completed")
            result.setdefault("risk_assessment", "Risk assessed")
            result.setdefault("recommendations", [])
            result.setdefault("explanation", "Strategy formulated")
            result.setdefault("confidence", 0.7)
            result["model_used"] = self.model_name
            result["provider"] = "gemini"

            logger.info(
                f"[LIVE] Gemini analysis complete — "
                f"{len(result['recommendations'])} recommendations"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] Gemini returned invalid JSON: {e}")
            return self._analyze_with_rules(context, merchant_data)

        except Exception as e:
            logger.error(f"[ERROR] Gemini analysis failed: {e}")
            return self._analyze_with_rules(context, merchant_data)

    def _analyze_with_openai(
        self, context: str, merchant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use OpenAI GPT to analyze the merchant."""
        user_prompt = f"Analyze this Razorpay merchant and recommend recovery actions:\n\n{context}\n\nProvide your analysis as JSON. Be specific, data-driven, and actionable."

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            result.setdefault("analysis", "Analysis completed")
            result.setdefault("risk_assessment", "Risk assessed")
            result.setdefault("recommendations", [])
            result.setdefault("explanation", "Strategy formulated")
            result.setdefault("confidence", 0.7)
            result["model_used"] = self.model_name
            result["provider"] = "openai"

            logger.info(
                f"[LIVE] OpenAI analysis complete — "
                f"{len(result['recommendations'])} recommendations"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] OpenAI returned invalid JSON: {e}")
            return self._analyze_with_rules(context, merchant_data)

        except Exception as e:
            logger.error(f"[ERROR] OpenAI analysis failed: {e}")
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

        if failure_rate > 0.1:
            impact = revenue * 0.2
            recommendations.append({
                "action_type": "retry",
                "title": "Optimize Payment Retry Sequence",
                "description": f"Implement aggressive retry strategy for {failed_attempts} failed payments.",
                "parameters": {"retry_intervals_hours": [1, 4, 24], "max_retries": 5, "fallback_methods": ["upi", "card", "netbanking"]},
                "expected_impact": int(impact),
                "reasoning": f"Failure rate is {failure_rate:.1%} — optimizing retry timing can recover ~20% of failed payments.",
                "confidence": 0.75, "urgency": "immediate"
            })

        if refund_rate > 0.05:
            impact = revenue * 0.1
            recommendations.append({
                "action_type": "campaign",
                "title": "Quality Feedback & Retention Campaign",
                "description": "Launch feedback survey with Rs.100 incentive.",
                "parameters": {"campaign_type": "quality_feedback", "survey_enabled": True, "incentive_amount": 100},
                "expected_impact": int(impact),
                "reasoning": f"Refund rate is {refund_rate:.1%} — understanding WHY prevents future losses.",
                "confidence": 0.65, "urgency": "this_week"
            })

        if days_inactive > 30:
            impact = revenue * 0.3
            recommendations.append({
                "action_type": "outreach",
                "title": "Merchant Win-Back Campaign",
                "description": f"Merchant inactive for {days_inactive} days. Send win-back email with 15% discount.",
                "parameters": {"channels": ["email", "sms"], "discount_percentage": 15, "urgency_level": "high"},
                "expected_impact": int(impact),
                "reasoning": f"Inactive for {days_inactive} days — urgent outreach needed.",
                "confidence": 0.7, "urgency": "immediate"
            })

        if revenue < 100000 and not recommendations:
            impact = revenue * 0.15
            recommendations.append({
                "action_type": "discount",
                "title": "Revenue Boost Discount Campaign",
                "description": "Create 10% discount campaign for 14 days.",
                "parameters": {"discount_percentage": 10, "duration_days": 14, "target_segments": ["inactive", "high_value"]},
                "expected_impact": int(impact),
                "reasoning": f"Revenue is Rs.{revenue:,.0f} — discount campaign can boost transactions.",
                "confidence": 0.6, "urgency": "this_week"
            })

        if failed_attempts > 10:
            impact = failed_attempts * 500
            recommendations.append({
                "action_type": "outreach",
                "title": "Failed Payment Customer Outreach",
                "description": f"Contact {failed_attempts} customers with failed payments.",
                "parameters": {"channels": ["email", "sms"], "customer_count": failed_attempts},
                "expected_impact": int(impact),
                "reasoning": f"{failed_attempts} failed attempts represent Rs.{impact:,.0f} in at-risk revenue.",
                "confidence": 0.7, "urgency": "immediate"
            })

        if not recommendations:
            recommendations.append({
                "action_type": "campaign",
                "title": "Merchant Health Check & Optimization",
                "description": "Schedule a health review.",
                "parameters": {"campaign_type": "health_check"},
                "expected_impact": int(revenue * 0.05),
                "reasoning": "General health check to identify optimization opportunities.",
                "confidence": 0.5, "urgency": "this_month"
            })

        recommendations.sort(key=lambda x: x["expected_impact"], reverse=True)
        total_impact = sum(r["expected_impact"] for r in recommendations)

        return {
            "analysis": f"Merchant has {'high' if churn_prob > 0.6 else 'moderate'} churn risk with Rs.{revenue:,.0f} revenue. Key issues: {', '.join(merchant_data.get('risk_factors', ['general health']))}.",
            "risk_assessment": f"Failure rate: {failure_rate:.1%}, Refund rate: {refund_rate:.1%}, Days inactive: {days_inactive}, Failed attempts: {failed_attempts}",
            "recommendations": recommendations[:3],
            "explanation": f"Generated {len(recommendations)} recommendations targeting Rs.{total_impact:,.0f} in revenue recovery.",
            "confidence": 0.6,
            "model_used": "rule-based (no LLM key configured)",
            "provider": "rule-based",
        }

    def analyze_batch(
        self,
        merchants: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Analyze multiple merchants efficiently."""
        return [self.analyze_merchant(m, p) for m, p in zip(merchants, predictions)]
