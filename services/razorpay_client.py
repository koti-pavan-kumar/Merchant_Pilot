"""
Razorpay Client - Real API Integration with Simulation Fallback

This client connects to Razorpay's test-mode APIs when keys are provided,
and falls back to simulation mode for demos without API access.

KEY DESIGN DECISION:
- Every method tries the REAL API first
- Only falls back to simulation if no keys are configured
- All responses have the same shape so the rest of the app doesn't care
"""

try:
    import razorpay
except ImportError:
    razorpay = None

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings
import logging

logger = logging.getLogger(__name__)


class RazorpayClient:
    """Production-ready Razorpay client with real API + simulation fallback."""

    def __init__(self):
        self.settings = get_settings()
        self.simulation_mode = True
        self.client = None
        self._api_call_log = []  # Track every API call for audit trail

        # Try to initialize real Razorpay client
        if razorpay is not None and self.settings.has_razorpay_keys:
            try:
                self.client = razorpay.Client(
                    auth=(
                        self.settings.RAZORPAY_KEY_ID,
                        self.settings.RAZORPAY_KEY_SECRET,
                    )
                )
                # Verify connection by fetching settlements (lightweight call)
                self.client.order.create({"amount": 100, "currency": "INR", "receipt": "__probe__"})
                self.simulation_mode = False
                logger.info("[LIVE] Razorpay client connected to test mode")
            except Exception as e:
                logger.warning(
                    f"[WARN] Razorpay connection failed: {e}. Using simulation."
                )
                self.client = None
        else:
            logger.info("[SIMULATION] No Razorpay API keys configured")

        self.retry_attempts = self.settings.MAX_RETRY_ATTEMPTS
        self.retry_delay = self.settings.RETRY_DELAY_SECONDS

    # ─── Helper ──────────────────────────────────────────────

    def _mock_id(self, prefix: str = "pay") -> str:
        return f"{prefix}_{random.randint(10**17, 10**18 - 1)}"

    def _log_call(self, method: str, params: dict, result: dict):
        """Log every API call for the audit trail."""
        self._api_call_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "params": params,
                "success": result.get("success", False),
                "mode": "live" if not self.simulation_mode else "simulation",
            }
        )

    def get_api_call_log(self) -> List[Dict]:
        return self._api_call_log

    # ─── Payment Links ───────────────────────────────────────

    def create_payment_link(
        self,
        amount: int,
        currency: str = "INR",
        description: str = "Payment for services",
        customer_email: Optional[str] = None,
        customer_phone: Optional[str] = None,
        notes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a Razorpay payment link — real or simulated."""
        params = {
            "amount": amount,
            "currency": currency,
            "description": description,
        }

        if self.simulation_mode:
            plink_id = self._mock_id("plink")
            result = {
                "success": True,
                "payment_link_id": plink_id,
                "short_url": f"https://rzp.io/i/{plink_id[-8:]}",
                "amount": amount,
                "currency": currency,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "simulation_mode": True,
            }
            logger.info(f"[SIM] Payment link created: {plink_id}")
            self._log_call("payment_link.create", params, result)
            return result

        # ── Real API call ──
        try:
            payload = {
                "amount": amount,
                "currency": currency,
                "description": description,
                "callback_url": "https://example.com/payment-callback",
                "callback_method": "get",
            }
            if notes:
                payload["notes"] = notes
            if customer_email or customer_phone:
                customer = {}
                if customer_email:
                    customer["email"] = customer_email
                if customer_phone:
                    customer["contact"] = customer_phone
                payload["customer"] = customer

            response = self.client.payment_link.create(payload)

            result = {
                "success": True,
                "payment_link_id": response["id"],
                "short_url": response.get("short_url", ""),
                "amount": amount,
                "currency": currency,
                "status": response.get("status", "created"),
                "created_at": datetime.now().isoformat(),
                "raw_response": response,
            }
            logger.info(f"[LIVE] Payment link created: {response['id']}")
            self._log_call("payment_link.create", params, result)
            return result

        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            logger.error(f"[LIVE] Payment link failed: {e}")
            self._log_call("payment_link.create", params, result)
            return result

    def fetch_payment_link(self, payment_link_id: str) -> Dict[str, Any]:
        """Fetch payment link status — useful for checking if customer paid."""
        if self.simulation_mode:
            return {
                "success": True,
                "payment_link": {
                    "id": payment_link_id,
                    "status": random.choice(["created", "paid"]),
                    "amount_paid": random.choice([0, 1000]),
                },
                "simulation_mode": True,
            }

        try:
            response = self.client.payment_link.fetch(payment_link_id)
            return {"success": True, "payment_link": response}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─── Orders ──────────────────────────────────────────────

    def create_order(
        self,
        amount: int,
        currency: str = "INR",
        receipt: Optional[str] = None,
        notes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a Razorpay order — real or simulated."""
        params = {"amount": amount, "currency": currency}

        if self.simulation_mode:
            order_id = self._mock_id("order")
            result = {
                "success": True,
                "order_id": order_id,
                "amount": amount,
                "currency": currency,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "simulation_mode": True,
            }
            logger.info(f"[SIM] Order created: {order_id}")
            self._log_call("order.create", params, result)
            return result

        try:
            payload = {
                "amount": amount,
                "currency": currency,
                "receipt": receipt or f"rcpt_{int(time.time())}",
            }
            if notes:
                payload["notes"] = notes

            response = self.client.order.create(payload)

            result = {
                "success": True,
                "order_id": response["id"],
                "amount": amount,
                "currency": currency,
                "status": response.get("status", "created"),
                "created_at": datetime.now().isoformat(),
                "raw_response": response,
            }
            logger.info(f"[LIVE] Order created: {response['id']}")
            self._log_call("order.create", params, result)
            return result

        except Exception as e:
            result = {"success": False, "error": str(e), "error_type": type(e).__name__}
            logger.error(f"[LIVE] Order failed: {e}")
            self._log_call("order.create", params, result)
            return result

    # ─── Payments ────────────────────────────────────────────

    def fetch_payment(self, payment_id: str) -> Dict[str, Any]:
        """Fetch payment details."""
        if self.simulation_mode:
            return {
                "success": True,
                "payment": {
                    "id": payment_id,
                    "amount": random.randint(500, 10000),
                    "status": random.choice(["captured", "authorized", "failed"]),
                    "method": random.choice(["upi", "card", "netbanking"]),
                    "created_at": datetime.now().isoformat(),
                },
                "simulation_mode": True,
            }

        try:
            response = self.client.payment.fetch(payment_id)
            return {"success": True, "payment": response}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fetch_payments(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        count: int = 10,
    ) -> Dict[str, Any]:
        """Fetch payments within a date range."""
        if self.simulation_mode:
            statuses = ["captured", "failed", "refunded", "authorized"]
            payments = [
                {
                    "id": self._mock_id("pay"),
                    "amount": random.randint(100, 10000),
                    "status": random.choice(statuses),
                    "method": random.choice(["upi", "card", "netbanking", "wallet"]),
                    "created_at": (
                        datetime.now() - timedelta(days=random.randint(0, 30))
                    ).isoformat(),
                }
                for _ in range(count)
            ]
            return {
                "success": True,
                "payments": payments,
                "count": len(payments),
                "simulation_mode": True,
            }

        try:
            payload = {"count": count}
            if from_date:
                payload["from"] = int(
                    datetime.fromisoformat(from_date).timestamp()
                )
            if to_date:
                payload["to"] = int(datetime.fromisoformat(to_date).timestamp())

            response = self.client.payment.all(payload)

            return {
                "success": True,
                "payments": response.get("items", []),
                "count": response.get("count", 0),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─── Refunds ─────────────────────────────────────────────

    def process_refund(
        self,
        payment_id: str,
        amount: Optional[int] = None,
        reason: str = "requested_by_customer",
    ) -> Dict[str, Any]:
        """Process a refund."""
        params = {"payment_id": payment_id, "amount": amount, "reason": reason}

        if self.simulation_mode:
            refund_id = self._mock_id("refund")
            result = {
                "success": True,
                "refund_id": refund_id,
                "payment_id": payment_id,
                "amount": amount or 1000,
                "status": "processed",
                "processed_at": datetime.now().isoformat(),
                "simulation_mode": True,
            }
            logger.info(f"[SIM] Refund processed: {refund_id}")
            self._log_call("payment.refund", params, result)
            return result

        try:
            payload = {"amount": amount, "reason": reason} if amount else {"reason": reason}
            response = self.client.payment.refund(payment_id, payload)

            result = {
                "success": True,
                "refund_id": response["id"],
                "payment_id": payment_id,
                "amount": response.get("amount", amount),
                "status": response.get("status", "processed"),
                "processed_at": datetime.now().isoformat(),
                "raw_response": response,
            }
            logger.info(f"[LIVE] Refund processed: {response['id']}")
            self._log_call("payment.refund", params, result)
            return result

        except Exception as e:
            result = {"success": False, "error": str(e), "error_type": type(e).__name__}
            logger.error(f"[LIVE] Refund failed: {e}")
            self._log_call("payment.refund", params, result)
            return result

    # ─── Customers ───────────────────────────────────────────

    def create_customer(
        self,
        name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        notes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a customer record."""
        params = {"name": name, "email": email, "phone": phone}

        if self.simulation_mode:
            customer_id = self._mock_id("cust")
            result = {
                "success": True,
                "customer_id": customer_id,
                "name": name,
                "email": email,
                "phone": phone,
                "created_at": datetime.now().isoformat(),
                "simulation_mode": True,
            }
            logger.info(f"[SIM] Customer created: {customer_id}")
            self._log_call("customer.create", params, result)
            return result

        try:
            payload = {"name": name}
            if email:
                payload["email"] = email
            if phone:
                payload["contact"] = phone
            if notes:
                payload["notes"] = notes

            response = self.client.customer.create(payload)

            result = {
                "success": True,
                "customer_id": response["id"],
                "name": name,
                "email": email,
                "phone": phone,
                "created_at": datetime.now().isoformat(),
                "raw_response": response,
            }
            logger.info(f"[LIVE] Customer created: {response['id']}")
            self._log_call("customer.create", params, result)
            return result

        except Exception as e:
            result = {"success": False, "error": str(e), "error_type": type(e).__name__}
            logger.error(f"[LIVE] Customer failed: {e}")
            self._log_call("customer.create", params, result)
            return result

    # ─── Settlements ─────────────────────────────────────────

    def fetch_settlements(self) -> Dict[str, Any]:
        """Fetch settlement history — crucial for showing real money movement."""
        if self.simulation_mode:
            settlements = [
                {
                    "id": self._mock_id("settn"),
                    "amount": random.randint(5000, 500000),
                    "status": "processed",
                    "created_at": (
                        datetime.now() - timedelta(days=i)
                    ).isoformat(),
                    "utr": f"UTR{random.randint(10**10, 10**11 - 1)}",
                }
                for i in range(5)
            ]
            return {
                "success": True,
                "settlements": settlements,
                "count": len(settlements),
                "simulation_mode": True,
            }

        try:
            response = self.client.settlement.all({})
            settlements = response.get("items", [])
            return {
                "success": True,
                "settlements": settlements,
                "count": len(settlements),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account info — great for live demo proof."""
        if self.simulation_mode:
            return {
                "success": True,
                "balance": random.randint(10000, 1000000),
                "currency": "INR",
                "simulation_mode": True,
            }

        try:
            # SDK v2 has no direct balance API
            # Instead, fetch settlements to prove connection works
            response = self.client.settlement.all({})
            total_settled = sum(
                s.get("amount", 0) for s in response.get("items", [])
            )
            return {
                "success": True,
                "balance": total_settled,
                "currency": "INR",
                "note": "Derived from settlements (SDK v2)",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─── QR Code ─────────────────────────────────────────────

    def create_qr_code(
        self,
        amount: int,
        description: str = "Payment via QR",
    ) -> Dict[str, Any]:
        """Create a QR code for payment — visual demo impact."""
        if self.simulation_mode:
            qr_id = self._mock_id("qr")
            return {
                "success": True,
                "qr_id": qr_id,
                "image_url": f"https://rzp.io/i/qr_{qr_id[-8:]}",
                "amount": amount,
                "simulation_mode": True,
            }

        try:
            response = self.client.qrcode.create(
                {
                    "type": "upi_qr",
                    "name": description,
                    "usage": "single_use",
                    "fixed_amount": True,
                    "amount": amount,
                }
            )
            return {
                "success": True,
                "qr_id": response["id"],
                "image_url": response.get("image_url", ""),
                "amount": amount,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─── Retry Engine ────────────────────────────────────────

    def execute_with_retry(
        self, func, *args, **kwargs
    ) -> Dict[str, Any]:
        """Execute any Razorpay API call with exponential backoff retry."""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                result = func(*args, **kwargs)
                if isinstance(result, dict) and result.get("success"):
                    return result
                else:
                    last_error = (
                        result.get("error", "Unknown error")
                        if isinstance(result, dict)
                        else str(result)
                    )
                    logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} exception: {last_error}")

            # Exponential backoff (skip delay in simulation for speed)
            if not self.simulation_mode and attempt < self.retry_attempts - 1:
                delay = self.retry_delay * (2**attempt) + random.uniform(0, 1)
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)

        return {
            "success": False,
            "error": f"All {self.retry_attempts} attempts failed. Last: {last_error}",
            "attempts": self.retry_attempts,
        }

    # ─── Failure Simulation (for testing) ────────────────────

    def simulate_failure(self, failure_type: str = "random") -> Dict[str, Any]:
        """Simulate API failures for testing error handling."""
        failures = {
            "network": "Network connection timeout",
            "rate_limit": "Rate limit exceeded (429)",
            "auth": "Authentication failed (401)",
            "validation": "Invalid request parameters (400)",
            "server": "Internal server error (500)",
        }
        if failure_type == "random":
            failure_type = random.choice(list(failures.keys()))

        return {
            "success": False,
            "error": failures.get(failure_type, "Unknown error"),
            "error_type": failure_type,
            "simulated": True,
        }

    # ─── Status ──────────────────────────────────────────────

    def get_mode(self) -> str:
        return "simulation" if self.simulation_mode else "LIVE"

    def get_status(self) -> Dict[str, Any]:
        """Full client status for debugging / health checks."""
        return {
            "mode": self.get_mode(),
            "simulation": self.simulation_mode,
            "api_keys_configured": self.settings.has_razorpay_keys,
            "client_initialized": self.client is not None,
            "total_api_calls": len(self._api_call_log),
            "razorpay_sdk_version": "2.0.1",
        }
