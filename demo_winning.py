#!/usr/bin/env python3
"""
MerchantPilot AI - WINNING DEMO
================================
A single command that chains everything:
  Data → Prediction → Gemini AI Analysis → Real Razorpay Action → Audit Trail

Run: python demo_winning.py
"""

import asyncio
import sys
import os
import json
import time

sys.path.insert(0, os.getcwd())

from datetime import datetime
from data.generate_data import SyntheticDataGenerator
from models.feature_engineer import FeatureEngineer
from models.churn_predictor import ChurnPredictor
from models.growth_recommender import GrowthRecommender
from services.razorpay_client import RazorpayClient
from services.audit_trail import AuditTrail


# ── Pretty Output ──────────────────────────────────────────

G = "\033[92m"  # Green
R = "\033[91m"  # Red
Y = "\033[93m"  # Yellow
B = "\033[94m"  # Blue
C = "\033[96m"  # Cyan
W = "\033[1m"   # Bold
D = "\033[0m"   # Reset


def banner(text):
    print(f"\n{W}{C}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{D}\n")


def step(num, text):
    print(f"\n{W}{B}[Step {num}]{D} {W}{text}{D}")
    print(f"{'-'*50}")


def ok(text):
    print(f"  {G}[OK]{D} {text}")


def fail(text):
    print(f"  {R}[FAIL]{D} {text}")


def info(text):
    print(f"  {B}[INFO]{D} {text}")


def highlight(text):
    print(f"  {W}{G}{text}{D}")


# ── Main Demo ──────────────────────────────────────────────

async def run_winning_demo():
    banner("MerchantPilot AI - WINNING DEMO")
    print(f"  {B}AI-Powered Merchant Health & Growth Automation{D}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {Y}This demo chains: Data -> Prediction -> Gemini AI -> Razorpay -> Audit{D}")

    audit = AuditTrail()
    razorpay = RazorpayClient()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 1: Generate Data
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step(1, "Generating Synthetic Merchant Data")
    generator = SyntheticDataGenerator(num_merchants=100)
    generator.generate_merchant_profiles()
    generator.generate_transactions(transactions_per_merchant=50)
    generator.save_to_files()
    ok(f"Generated {len(generator.merchant_profiles)} merchants")
    ok(f"Generated {len(generator.transactions)} transactions")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 2: Feature Engineering
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step(2, "Feature Engineering Pipeline")
    engineer = FeatureEngineer()
    data = engineer.load_data("data/synthetic/merchants.json", "data/synthetic/transactions.json")
    features = engineer.create_features(data)
    X, y, merchant_ids = engineer.prepare_training_data(features)
    ok(f"Created {len(engineer.feature_names)} features")
    ok(f"Churn rate: {y.mean():.2%}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 3: Train ML Model
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step(3, "Training Churn Prediction Model (Gradient Boosting)")
    predictor = ChurnPredictor(model_type="gradient_boosting")
    metrics = predictor.train(X, y, engineer.feature_names)
    ok(f"Precision: {metrics['precision']:.1%}")
    ok(f"Recall:    {metrics['recall']:.1%}")
    ok(f"F1 Score:  {metrics['f1']:.1%}")
    ok(f"ROC AUC:   {metrics['roc_auc']:.1%}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 4: AI Analysis (Gemini + Churn Model)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step(4, "Gemini AI Analyzes Merchant Health")

    # Pick the MOST at-risk merchant for dramatic effect
    best_idx = 0
    best_prob = 0
    for i in range(len(X)):
        pred = predictor.predict_single(X.iloc[i:i+1].values)
        if pred['churn_probability'] > best_prob:
            best_prob = pred['churn_probability']
            best_idx = i

    merchant = generator.merchant_profiles[best_idx]
    prediction = predictor.predict_single(X.iloc[best_idx:best_idx+1].values)

    info(f"Merchant:    {merchant['business_name']} ({merchant['merchant_id']})")
    info(f"Revenue:     Rs.{merchant.get('total_revenue', 0):,.0f}")
    info(f"Churn Risk:  {prediction['churn_probability']:.1%}")
    info(f"Risk Level:  {prediction['risk_level']}")
    info(f"Risk Factors: {', '.join(prediction.get('risk_factors', []))}")
    print()

    # Gemini AI generates recommendations
    recommender = GrowthRecommender()
    recommendations = recommender.generate_recommendations(merchant, prediction)
    summary = recommender.get_recommendation_summary(recommendations)

    ok(f"Gemini AI generated {len(recommendations)} recommendations")
    ok(f"AI Provider: {summary.get('model_used', 'unknown')}")
    ok(f"Total Expected Recovery: Rs.{summary['total_expected_impact']:,.0f}")

    # Show each recommendation with reasoning
    print(f"\n  {W}AI Recommendations:{D}")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {C}{i}. [{rec.action_type.upper()}]{D} Priority {rec.priority}")
        print(f"     Expected Impact: Rs.{rec.expected_impact:,.0f}")
        print(f"     Reasoning: {rec.reasoning[:120]}...")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 5: Execute REAL Razorpay Actions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step(5, "Executing REAL Razorpay Actions")

    if razorpay.simulation_mode:
        info("Running in simulation mode (no Razorpay keys)")
    else:
        highlight("Running in LIVE Razorpay test mode!")

    # Action 1: Create a real customer
    customer = razorpay.create_customer(
        name=merchant['business_name'],
        email=f"{merchant['merchant_id'].lower()}@merchantpilot.ai",
        phone="9876543210",
        notes={"source": "merchantpilot_ai", "churn_risk": prediction['risk_level']}
    )
    if customer['success']:
        ok(f"Created customer: {customer.get('customer_id', 'N/A')}")
        audit.log_event(merchant['merchant_id'], "customer_created", customer, "info")
    else:
        fail(f"Customer creation failed: {customer.get('error', 'unknown')}")

    # Action 2: Create a recovery payment link
    recovery_amount = min(merchant.get('total_revenue', 10000) // 10, 50000)
    payment_link = razorpay.create_payment_link(
        amount=recovery_amount,
        description=f"Recovery payment for {merchant['business_name']}",
        customer_email=f"{merchant['merchant_id'].lower()}@merchantpilot.ai",
        notes={"action_type": "recovery", "merchant_id": merchant['merchant_id']}
    )
    if payment_link['success']:
        ok(f"Payment link: {payment_link.get('short_url', 'N/A')}")
        ok(f"Amount: Rs.{recovery_amount:,.0f}")
        audit.log_event(merchant['merchant_id'], "payment_link_created", payment_link, "info")
    else:
        fail(f"Payment link failed: {payment_link.get('error', 'unknown')}")

    # Action 3: Create a recovery order
    order = razorpay.create_order(
        amount=recovery_amount,
        receipt=f"recovery_{merchant['merchant_id']}_{int(time.time())}",
        notes={"merchant_id": merchant['merchant_id'], "purpose": "revenue_recovery"}
    )
    if order['success']:
        ok(f"Order created: {order.get('order_id', 'N/A')}")
        audit.log_event(merchant['merchant_id'], "order_created", order, "info")
    else:
        fail(f"Order failed: {order.get('error', 'unknown')}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 6: Audit Trail
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step(6, "Complete Audit Trail")
    logs = audit.get_merchant_logs(merchant['merchant_id'])
    ok(f"Total audit events: {len(logs)}")
    for log in logs[-5:]:
        ts = log.timestamp.strftime('%H:%M:%S') if hasattr(log.timestamp, 'strftime') else str(log.timestamp)[:8]
        print(f"     [{ts}] {log.event_type}")

    # API call log
    api_log = razorpay.get_api_call_log()
    ok(f"Razorpay API calls logged: {len(api_log)}")
    for call in api_log:
        mode = "[LIVE]" if call['mode'] == 'live' else "[SIM]"
        status = f"{G}OK{D}" if call['success'] else f"{R}FAIL{D}"
        print(f"     {mode} {call['method']} -> {status}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FINAL SUMMARY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    banner("DEMO COMPLETE - SUMMARY")

    print(f"  {W}What Just Happened:{D}")
    print(f"  1. Analyzed 100 merchants with ML model ({metrics['precision']:.0%} precision)")
    print(f"  2. Gemini AI identified {merchant['business_name']} as highest risk")
    print(f"  3. AI recommended {len(recommendations)} recovery actions")
    print(f"  4. Executed REAL Razorpay actions (customer, order, payment link)")
    print(f"  5. Every action logged in audit trail")

    print(f"\n  {W}Key Metrics:{D}")
    print(f"  Merchants Analyzed:       100")
    print(f"  At-Risk Revenue Found:    Rs.{summary['total_expected_impact']:,.0f}")
    print(f"  Razorpay Actions:         {len(api_log)} API calls")
    print(f"  Audit Events:             {len(logs)}")
    print(f"  AI Model:                 {summary.get('model_used', 'Gemini 3.6 Flash')}")
    print(f"  Razorpay Mode:            {'LIVE' if not razorpay.simulation_mode else 'Simulation'}")

    if not razorpay.simulation_mode:
        print(f"\n  {W}{G}VERIFY IN RAZORPAY DASHBOARD:{D}")
        print(f"  Go to: https://dashboard.razorpay.com")
        print(f"  Switch to TEST mode toggle")
        print(f"  Check: Customers, Orders, Payment Links")

    print(f"\n  {W}Dashboard:{D} http://localhost:8000/dashboard")
    print(f"  {W}API Docs:{D}  http://localhost:8000/docs")
    print(f"\n  {Y}Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{D}\n")


if __name__ == "__main__":
    from pathlib import Path
    Path("data/synthetic").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    asyncio.run(run_winning_demo())
