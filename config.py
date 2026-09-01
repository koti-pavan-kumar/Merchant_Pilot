import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

class Settings(BaseModel):
    # Application settings
    APP_NAME: str = "MerchantPilot AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Razorpay test-mode settings
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "rzp_test_1234567890")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "test_secret_key")
    
    # LLM settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///merchantpilot.db")
    
    # ML model settings
    MODEL_PATH: str = "models/saved_models"
    CHURN_THRESHOLD: float = 0.7  # Probability threshold for churn prediction
    
    # Action settings
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: int = 5
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Audit settings
    AUDIT_LOG_PATH: str = "logs/audit.log"
    
    # Dashboard settings
    DASHBOARD_UPDATE_INTERVAL: int = 30  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    return Settings()

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models/saved_models", exist_ok=True)