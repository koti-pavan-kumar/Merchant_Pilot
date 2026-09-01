try:
    import razorpay
except ImportError:
    razorpay = None

from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RazorpayClient:
    def __init__(self):
        self.settings = get_settings()
        self.simulation_mode = True  # Default to simulation mode for demo
        
        # Try to initialize real client if razorpay is available
        if razorpay is not None and self.settings.RAZORPAY_KEY_ID != "rzp_test_1234567890":
            try:
                self.client = razorpay.Client(
                    auth=(self.settings.RAZORPAY_KEY_ID, self.settings.RAZORPAY_KEY_SECRET)
                )
                self.simulation_mode = False
                logger.info("Razorpay client initialized in LIVE mode")
            except Exception as e:
                logger.warning(f"Failed to initialize Razorpay client: {e}. Using simulation mode.")
                self.client = None
        else:
            self.client = None
            logger.info("Razorpay client initialized in SIMULATION mode (demo)")
        
        self.retry_attempts = self.settings.MAX_RETRY_ATTEMPTS
        self.retry_delay = self.settings.RETRY_DELAY_SECONDS
    
    def _generate_mock_id(self, prefix: str = "pay") -> str:
        """Generate a mock ID for simulation."""
        return f"{prefix}_{random.randint(100000000000, 999999999999)}"
    
    def create_payment_link(
        self, 
        amount: int, 
        currency: str = "INR",
        description: str = "Payment for services",
        customer_email: Optional[str] = None,
        customer_phone: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a payment link for the merchant."""
        # Simulation mode - return mock response
        if self.simulation_mode:
            payment_link_id = self._generate_mock_id("plink")
            logger.info(f"[SIMULATION] Payment link created: {payment_link_id}")
            return {
                "success": True,
                "payment_link_id": payment_link_id,
                "short_url": f"https://rzp.io/i/{payment_link_id[-8:]}",
                "amount": amount,
                "currency": currency,
                "created_at": datetime.now().isoformat(),
                "simulation_mode": True
            }
        
        # Live mode - call real API
        try:
            payload = {
                "amount": amount,
                "currency": currency,
                "description": description,
                "callback_url": "https://example.com/callback",
                "callback_method": "get"
            }
            
            if customer_email:
                payload["customer"] = {"email": customer_email}
            if customer_phone:
                payload["customer"] = payload.get("customer", {})
                payload["customer"]["contact"] = customer_phone
            
            response = self.client.payment_link.create(payload)
            
            logger.info(f"Payment link created: {response['id']}")
            return {
                "success": True,
                "payment_link_id": response["id"],
                "short_url": response["short_url"],
                "amount": amount,
                "currency": currency,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create payment link: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def create_order(
        self, 
        amount: int, 
        currency: str = "INR",
        receipt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an order for the merchant."""
        # Simulation mode
        if self.simulation_mode:
            order_id = self._generate_mock_id("order")
            logger.info(f"[SIMULATION] Order created: {order_id}")
            return {
                "success": True,
                "order_id": order_id,
                "amount": amount,
                "currency": currency,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "simulation_mode": True
            }
        
        # Live mode
        try:
            payload = {
                "amount": amount,
                "currency": currency,
                "receipt": receipt or f"order_{int(time.time())}"
            }
            
            response = self.client.order.create(payload)
            
            logger.info(f"Order created: {response['id']}")
            return {
                "success": True,
                "order_id": response["id"],
                "amount": amount,
                "currency": currency,
                "status": response["status"],
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create order: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def process_refund(
        self, 
        payment_id: str, 
        amount: Optional[int] = None,
        reason: str = "requested_by_customer"
    ) -> Dict[str, Any]:
        """Process a refund for a payment."""
        # Simulation mode
        if self.simulation_mode:
            refund_id = self._generate_mock_id("refund")
            logger.info(f"[SIMULATION] Refund processed: {refund_id}")
            return {
                "success": True,
                "refund_id": refund_id,
                "payment_id": payment_id,
                "amount": amount or 1000,
                "status": "processed",
                "processed_at": datetime.now().isoformat(),
                "simulation_mode": True
            }
        
        # Live mode
        try:
            payload = {
                "payment_id": payment_id,
                "reason": reason
            }
            
            if amount:
                payload["amount"] = amount
            
            response = self.client.payment.refund(payment_id, payload)
            
            logger.info(f"Refund processed: {response['id']}")
            return {
                "success": True,
                "refund_id": response["id"],
                "payment_id": payment_id,
                "amount": response.get("amount", amount),
                "status": response["status"],
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process refund: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def fetch_payment(self, payment_id: str) -> Dict[str, Any]:
        """Fetch payment details."""
        if self.simulation_mode:
            return {
                "success": True,
                "payment": {
                    "id": payment_id,
                    "amount": 1000,
                    "status": "captured",
                    "created_at": datetime.now().isoformat()
                },
                "simulation_mode": True
            }
        
        try:
            response = self.client.payment.fetch(payment_id)
            
            return {
                "success": True,
                "payment": response
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch payment: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def fetch_payments(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict[str, Any]:
        """Fetch all payments within a date range."""
        if self.simulation_mode:
            return {
                "success": True,
                "payments": [
                    {"id": self._generate_mock_id("pay"), "amount": random.randint(100, 5000), "status": "captured"}
                    for _ in range(10)
                ],
                "count": 10,
                "simulation_mode": True
            }
        
        try:
            payload = {}
            if from_date:
                payload["from"] = int(datetime.fromisoformat(from_date).timestamp())
            if to_date:
                payload["to"] = int(datetime.fromisoformat(to_date).timestamp())
            
            response = self.client.payment.all(payload)
            
            return {
                "success": True,
                "payments": response.get("items", []),
                "count": response.get("count", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch payments: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def create_customer(
        self, 
        name: str, 
        email: Optional[str] = None,
        phone: Optional[str] = None,
        notes: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Create a customer record."""
        if self.simulation_mode:
            customer_id = self._generate_mock_id("cust")
            logger.info(f"[SIMULATION] Customer created: {customer_id}")
            return {
                "success": True,
                "customer_id": customer_id,
                "name": name,
                "created_at": datetime.now().isoformat(),
                "simulation_mode": True
            }
        
        try:
            payload = {"name": name}
            
            if email:
                payload["email"] = email
            if phone:
                payload["contact"] = phone
            if notes:
                payload["notes"] = notes
            
            response = self.client.customer.create(payload)
            
            logger.info(f"Customer created: {response['id']}")
            return {
                "success": True,
                "customer_id": response["id"],
                "name": name,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create customer: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def create_subscription(
        self, 
        plan_id: str, 
        customer_id: str,
        start_at: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a subscription for a customer."""
        if self.simulation_mode:
            subscription_id = self._generate_mock_id("sub")
            logger.info(f"[SIMULATION] Subscription created: {subscription_id}")
            return {
                "success": True,
                "subscription_id": subscription_id,
                "plan_id": plan_id,
                "customer_id": customer_id,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "simulation_mode": True
            }
        
        try:
            payload = {
                "plan_id": plan_id,
                "customer_id": customer_id,
                "total_count": 12
            }
            
            if start_at:
                payload["start_at"] = start_at
            
            response = self.client.subscription.create(payload)
            
            logger.info(f"Subscription created: {response['id']}")
            return {
                "success": True,
                "subscription_id": response["id"],
                "plan_id": plan_id,
                "customer_id": customer_id,
                "status": response["status"],
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def execute_with_retry(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Execute a function with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                result = func(*args, **kwargs)
                if result.get("success"):
                    return result
                else:
                    last_error = result.get("error", "Unknown error")
                    logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} exception: {last_error}")
            
            # Exponential backoff with jitter (skip delay in simulation mode for speed)
            if not self.simulation_mode and attempt < self.retry_attempts - 1:
                delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
        
        return {
            "success": False,
            "error": f"All {self.retry_attempts} attempts failed. Last error: {last_error}",
            "attempts": self.retry_attempts
        }
    
    def simulate_failure(self, failure_type: str = "random") -> Dict[str, Any]:
        """Simulate API failures for testing error handling."""
        failures = {
            "network": "Network connection timeout",
            "rate_limit": "Rate limit exceeded",
            "auth": "Authentication failed",
            "validation": "Invalid request parameters",
            "server": "Internal server error"
        }
        
        if failure_type == "random":
            failure_type = random.choice(list(failures.keys()))
        
        return {
            "success": False,
            "error": failures.get(failure_type, "Unknown error"),
            "error_type": failure_type,
            "simulated": True
        }
    
    def get_mode(self) -> str:
        """Get current client mode."""
        return "simulation" if self.simulation_mode else "live"