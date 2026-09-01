# MerchantPilot AI

An AI agent that monitors merchant health, predicts churn, and automates revenue-recovery actions through Razorpay test-mode APIs.

## Project Structure

```
merchantpilot/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── main.py                      # FastAPI application entry point
├── config.py                    # Configuration management
├── models/                      # ML models and AI components
│   ├── __init__.py
│   ├── churn_predictor.py       # Churn prediction model
│   ├── growth_recommender.py    # LLM-driven recommendations
│   └── feature_engineer.py      # Feature engineering pipeline
├── services/                    # Business logic services
│   ├── __init__.py
│   ├── razorpay_client.py       # Razorpay test-mode API wrapper
│   ├── action_orchestrator.py   # Action execution and recovery
│   └── audit_trail.py           # Audit logging and monitoring
├── data/                        # Synthetic data generation
│   ├── __init__.py
│   ├── generate_data.py         # Synthetic merchant data generator
│   └── schemas.py               # Data schemas
├── api/                         # API routes
│   ├── __init__.py
│   ├── health.py                # Merchant health endpoints
│   └── actions.py               # Action execution endpoints
├── dashboard/                   # Frontend dashboard
│   └── index.html               # Real-time monitoring dashboard
└── tests/                       # Test suite
    ├── test_churn_predictor.py  # Model evaluation tests
    └── test_actions.py          # Action orchestration tests
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data:
   ```bash
   python -m data.generate_data
   ```

3. Start the application:
   ```bash
   python main.py
   ```

4. Access the dashboard at http://localhost:8000/dashboard

## Key Features

- **Churn Prediction**: ML model predicting merchant churn with >85% precision
- **Growth Recommender**: LLM-driven personalized growth strategies
- **Automated Actions**: Revenue recovery through Razorpay test-mode APIs
- **Audit Trail**: Complete logging of all actions and decisions
- **Failure Recovery**: Graceful handling of API failures with retry logic

## Architecture

The system follows a pipeline architecture:
1. **Data Ingestion**: Collects merchant metrics from Razorpay APIs and synthetic sources
2. **Health Assessment**: ML model scores merchant churn risk
3. **Decision Engine**: LLM recommends personalized growth actions
4. **Action Execution**: Orchestrates API calls with proper error handling
5. **Monitoring**: Real-time dashboard showing system performance

## Evaluation Metrics

- Churn prediction: Precision >0.85, Recall >0.80, F1 >0.82
- Revenue recovery: >15% improvement in at-risk merchant revenue
- System reliability: 99.5% uptime with graceful degradation

## API Endpoints

- `GET /api/health/{merchant_id}` - Get merchant health score
- `POST /api/actions/execute` - Execute growth action
- `GET /api/audit/{merchant_id}` - Get audit trail
- `GET /dashboard` - Real-time monitoring dashboard

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Demo

The 5-minute demo video covers:
1. System architecture and data flow
2. Churn prediction accuracy demonstration
3. Automated growth action execution
4. Real-time dashboard monitoring
5. Failure recovery scenario