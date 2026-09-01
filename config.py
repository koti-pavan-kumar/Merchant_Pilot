import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] Loaded .env from: {env_path}")
except ImportError:
    pass


class Settings(BaseModel):
    # Application settings
    APP_NAME: str = "MerchantPilot AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Razorpay test-mode settings
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "")

    # LLM settings (optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")

    # ML model settings
    MODEL_PATH: str = "models/saved_models"
    CHURN_THRESHOLD: float = 0.7

    # Action settings
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: int = 2
    RATE_LIMIT_PER_MINUTE: int = 60

    # Audit settings
    AUDIT_LOG_PATH: str = "logs/audit.log"

    # Dashboard settings
    DASHBOARD_UPDATE_INTERVAL: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def has_razorpay_keys(self) -> bool:
        """Check if real Razorpay API keys are configured."""
        return (
            bool(self.RAZORPAY_KEY_ID)
            and bool(self.RAZORPAY_KEY_SECRET)
            and self.RAZORPAY_KEY_ID != "rzp_test_1234567890"
            and "YOUR_KEY_HERE" not in self.RAZORPAY_KEY_ID
        )


def get_settings() -> Settings:
    return Settings()


# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models/saved_models", exist_ok=True)
