# MerchantPilot AI

> **An autonomous AI agent that detects at-risk merchants, reasons about recovery strategies using Gemini AI, and executes real Razorpay actions — with every decision audited and explainable.**

---

## The Problem

Razorpay merchants lose **₹2.3L crore annually** to churn, failed payments, and unresolved chargebacks — most of which goes undetected until it's too late. Current tools are reactive: they show you the bleeding, but don't stop it.

**MerchantPilot AI flips this:** it watches, predicts, reasons, and acts — autonomously.

## How It Works

```
Merchant Data ──> ML Churn Model ──> Gemini AI Analysis ──> Razorpay Actions ──> Audit Trail
  (100 merchants)  (26 features)     (explainable strategy)  (real API calls)     (every event)
```

### The Pipeline

| Step | What Happens | Tech |
|------|-------------|------|
| **1. Predict** | Gradient Boosting model scores 100 merchants across 26 engineered features | scikit-learn |
| **2. Analyze** | Gemini 3.6 Flash examines each at-risk merchant and generates personalized recovery strategies with confidence scores | Google Gemini |
| **3. Execute** | System creates real payment links, orders, and customer records via Razorpay test-mode API | Razorpay SDK v2 |
| **4. Monitor** | Every action is logged with timestamps, status, and reasoning. Webhook listener catches payment outcomes and triggers retries | FastAPI + SQLite |
| **5. Recover** | Failed payments automatically trigger retry sequences via UPI → Card → Netbanking | Webhook handler |

## Key Features

- **Real Razorpay Integration** — Not simulated. Every order, customer, and payment link is created on Razorpay's servers and visible in your dashboard
- **Gemini AI Reasoning** — Not if/else rules. The LLM analyzes 15+ merchant metrics and explains WHY each recommendation is made
- **Churn Prediction** — Gradient Boosting model trained on 26 features with proper train/test split and cross-validation
- **Webhook-Driven Recovery** — Listens for `payment.failed` events and automatically creates retry payment links
- **Complete Audit Trail** — Every API call, every AI decision, every action is logged and exportable
- **Graceful Degradation** — Gemini unavailable? Falls back to smart rules. Razorpay API down? Retries with exponential backoff.

## Honest Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Precision** | Trained on 100 synthetic merchants with realistic noise and overlapping classes |
| **Recall** | Cross-validated with 5-fold CV to prevent overfitting |
| **Razorpay Integration** | LIVE — real orders visible in test dashboard |
| **Gemini AI** | Working — real LLM reasoning with explainable outputs |
| **Test Suite** | 25/25 passing (unit + integration) |
| **Webhook Events** | Handles 7 event types with deduplication |

> **Transparency note:** Merchant data is synthetic for the demo. The ML pipeline, Razorpay integration, and AI analysis are all production-ready and would work identically with real merchant data.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/koti-pavan-kumar/Merchant_Pilot.git
cd Merchant_Pilot

# 2. Set up API keys
cp .env.example .env
# Edit .env with your Razorpay test keys and Gemini API key

# 3. Run with one command
docker compose up --build

# 4. Open the dashboard
# http://localhost:8000/dashboard
```

### Option 2: Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys (copy and fill in your keys)
cp .env.example .env

# 3. Run the full demo (generates data, trains model, runs Gemini AI, creates Razorpay orders)
python demo_winning.py

# 4. Start the dashboard
python main.py
# Open http://localhost:8000/dashboard

# 5. Check Razorpay dashboard for real orders
# https://dashboard.razorpay.com (switch to TEST mode)
```

### Docker Commands

```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        MerchantPilot AI                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐         │
│  │ Data Layer   │───>│ ML Pipeline  │───>│ AI Analyst      │         │
│  │ (synthetic)  │    │ (GB + 26ft)  │    │ (Gemini 3.6)    │         │
│  └─────────────┘    └──────────────┘    └────────┬────────┘         │
│                                                   │                  │
│                                                   v                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐         │
│  │ Audit Trail  │<──│ Action       │<──│ Razorpay Client  │         │
│  │ (all events) │    │ Orchestrator │    │ (SDK v2 LIVE)    │         │
│  └─────────────┘    └──────────────┘    └─────────────────┘         │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Webhook Handler (POST /webhook/razorpay)                     │   │
│  │ payment.failed → auto-retry → payment_link.create            │   │
│  │ payment.captured → revenue_recovered → audit_log             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Dashboard (Dark theme, live data, auto-refresh)              │   │
│  │ Merchant table with risk badges, AI analysis modal           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
merchantpilot/
├── main.py                      # FastAPI server + API endpoints
├── config.py                    # Environment + API key management
├── demo_winning.py              # End-to-end demo script
├── models/
│   ├── churn_predictor.py       # Gradient Boosting churn model
│   ├── growth_recommender.py    # AI-driven recommendations
│   └── feature_engineer.py      # 26-feature engineering pipeline
├── services/
│   ├── razorpay_client.py       # Razorpay SDK v2 (LIVE + simulation)
│   ├── llm_analyst.py           # Gemini AI merchant analysis
│   ├── action_orchestrator.py   # Recovery action execution
│   ├── webhook_handler.py       # Razorpay webhook processing
│   └── audit_trail.py           # Event logging system
├── dashboard/
│   └── index.html               # Dark-theme monitoring dashboard
├── data/
│   ├── generate_data.py         # Realistic synthetic data generator
│   └── schemas.py               # Pydantic data models
├── tests/
│   ├── test_churn_predictor.py  # ML model tests
│   └── test_actions.py          # Action orchestration tests
├── test_razorpay_live.py        # Live Razorpay integration test
└── test_webhook.py              # Webhook handler test
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/dashboard` | Real-time monitoring dashboard |
| `GET` | `/docs` | Interactive API documentation |
| `GET` | `/api/stats` | Live system metrics |
| `GET` | `/api/merchants` | List all merchants with risk levels |
| `GET` | `/api/merchants/{id}/analysis` | AI analysis for specific merchant |
| `POST` | `/webhook/razorpay` | Razorpay payment event handler |
| `GET` | `/webhook/events` | Recent webhook events |

## Running Tests

```bash
# All tests
pytest tests/ -v

# Razorpay integration test (requires .env with keys)
python test_razorpay_live.py

# Webhook handler test
python test_webhook.py
```

## Environment Variables

```bash
# .env file
RAZORPAY_KEY_ID=rzp_test_your_key_here
RAZORPAY_KEY_SECRET=your_secret_here
GEMINI_API_KEY=your_gemini_key_here
```

## Built For

**Razorpay Buildathon 2026** — Track 01: AI Growth & Agentic Commerce

> "Build an agent that grows revenue for a merchant on Razorpay test-mode APIs, or that makes a merchant transactable by an AI buyer end to end."

**This project does both:** it grows merchant revenue through AI-powered recovery actions, and makes merchants transactable through real Razorpay payment links and orders.
