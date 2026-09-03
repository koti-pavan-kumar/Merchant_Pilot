"""
Razorpay Webhook Handler

Listens for payment events from Razorpay and triggers automated recovery actions.

Events handled:
- payment.authorized → Log success, update merchant health
- payment.captured → Confirm payment, log revenue recovery
- payment.failed → Trigger retry, send notification
- payment.refunded → Log refund, update merchant metrics
- payment.dispute.created → Alert, start dispute response flow
- order.paid → Update order status
- order.expired → Trigger re-engagement

This is the BRAIN of the recovery system — it reacts to real payment events
and takes automated actions to recover revenue.
"""

import json
import hashlib
import hmac
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from services.razorpay_client import RazorpayClient
from services.audit_trail import AuditTrail

logger = logging.getLogger(__name__)
settings = get_settings()


class WebhookHandler:
    """Processes Razorpay webhook events and triggers recovery actions."""

    def __init__(self):
        self.razorpay = RazorpayClient()
        self.audit = AuditTrail()
        self.processed_events = set()  # Deduplication

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature from Razorpay."""
        webhook_secret = settings.RAZORPAY_KEY_SECRET
        if not webhook_secret:
            logger.warning("[WARN] No webhook secret configured, skipping verification")
            return True

        expected = hmac.new(
            webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def handle_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point — routes webhook events to the right handler.

        Returns:
            {"success": bool, "action_taken": str, "details": dict}
        """
        event_id = payload.get("id", "unknown")

        # Deduplication
        if event_id in self.processed_events:
            logger.info(f"[SKIP] Duplicate event: {event_id}")
            return {"success": True, "action_taken": "skipped_duplicate", "details": {}}

        self.processed_events.add(event_id)

        logger.info(f"[WEBHOOK] Received: {event_type} (id: {event_id})")

        # Route to handler
        handlers = {
            "payment.authorized": self._handle_payment_authorized,
            "payment.captured": self._handle_payment_captured,
            "payment.failed": self._handle_payment_failed,
            "payment.refunded": self._handle_payment_refunded,
            "payment.dispute.created": self._handle_dispute_created,
            "order.paid": self._handle_order_paid,
            "order.expired": self._handle_order_expired,
        }

        handler = handlers.get(event_type)
        if handler:
            result = handler(payload)
        else:
            result = {"success": True, "action_taken": "unhandled_event", "details": {}}
            logger.info(f"[SKIP] Unhandled event type: {event_type}")

        # Log to audit trail
        self.audit.log_event(
            merchant_id=payload.get("notes", {}).get("merchant_id", "unknown"),
            event_type=f"webhook_{event_type.replace('.', '_')}",
            details={
                "event_id": event_id,
                "event_type": event_type,
                "action_taken": result.get("action_taken", "none"),
                "payload_keys": list(payload.keys()),
            },
            severity="info" if result["success"] else "warning",
        )

        return result

    # ── Payment Events ──────────────────────────────────────

    def _handle_payment_authorized(self, payload: Dict) -> Dict[str, Any]:
        """Payment authorized but not captured yet."""
        payment_id = payload.get("id", "unknown")
        amount = payload.get("amount", 0) / 100  # Razorpay amounts are in paise
        merchant_id = payload.get("notes", {}).get("merchant_id", "unknown")

        logger.info(f"[AUTH] Payment authorized: {payment_id} (Rs.{amount})")

        return {
            "success": True,
            "action_taken": "logged_authorization",
            "details": {
                "payment_id": payment_id,
                "amount": amount,
                "merchant_id": merchant_id,
            }
        }

    def _handle_payment_captured(self, payload: Dict) -> Dict[str, Any]:
        """Payment successfully captured — revenue recovered!"""
        payment_id = payload.get("id", "unknown")
        amount = payload.get("amount", 0) / 100
        merchant_id = payload.get("notes", {}).get("merchant_id", "unknown")
        method = payload.get("method", "unknown")

        logger.info(f"[CAPTURED] Payment captured: {payment_id} (Rs.{amount}) via {method}")

        # This is a WIN — log it prominently
        self.audit.log_event(
            merchant_id=merchant_id,
            event_type="revenue_recovered",
            details={
                "payment_id": payment_id,
                "amount": amount,
                "method": method,
                "status": "captured",
            },
            severity="info",
        )

        return {
            "success": True,
            "action_taken": "revenue_recovered",
            "details": {
                "payment_id": payment_id,
                "amount": amount,
                "merchant_id": merchant_id,
            }
        }

    def _handle_payment_failed(self, payload: Dict) -> Dict[str, Any]:
        """
        Payment failed — THIS IS THE CRITICAL ONE.
        Triggers automatic recovery: retry + notification.
        """
        payment_id = payload.get("id", "unknown")
        amount = payload.get("amount", 0) / 100
        merchant_id = payload.get("notes", {}).get("merchant_id", "unknown")
        error_code = payload.get("error_code", "unknown")
        error_description = payload.get("error_description", "Payment failed")

        logger.warning(f"[FAILED] Payment failed: {payment_id} (Rs.{amount}) — {error_code}")

        # ── Recovery Action 1: Create retry payment link ──
        retry_link = self.razorpay.create_payment_link(
            amount=int(amount * 100),  # Convert back to paise
            description=f"Retry payment for failed transaction {payment_id}",
            notes={
                "merchant_id": merchant_id,
                "original_payment_id": payment_id,
                "action_type": "auto_retry",
            }
        )

        # ── Recovery Action 2: Log the failure for analysis ──
        self.audit.log_event(
            merchant_id=merchant_id,
            event_type="payment_failure_detected",
            details={
                "payment_id": payment_id,
                "amount": amount,
                "error_code": error_code,
                "error_description": error_description,
                "retry_link_created": retry_link.get("success", False),
                "retry_link": retry_link.get("short_url", "N/A"),
            },
            severity="warning",
        )

        return {
            "success": True,
            "action_taken": "recovery_triggered",
            "details": {
                "payment_id": payment_id,
                "amount": amount,
                "error_code": error_code,
                "retry_link": retry_link.get("short_url", "N/A"),
                "retry_created": retry_link.get("success", False),
            }
        }

    def _handle_payment_refunded(self, payload: Dict) -> Dict[str, Any]:
        """Payment refunded — log for analysis."""
        payment_id = payload.get("id", "unknown")
        amount = payload.get("amount", 0) / 100
        merchant_id = payload.get("notes", {}).get("merchant_id", "unknown")

        logger.info(f"[REFUND] Payment refunded: {payment_id} (Rs.{amount})")

        self.audit.log_event(
            merchant_id=merchant_id,
            event_type="payment_refunded",
            details={
                "payment_id": payment_id,
                "amount": amount,
            },
            severity="info",
        )

        return {
            "success": True,
            "action_taken": "refund_logged",
            "details": {"payment_id": payment_id, "amount": amount}
        }

    def _handle_dispute_created(self, payload: Dict) -> Dict[str, Any]:
        """Dispute/chargeback created — alert and start evidence collection."""
        dispute_id = payload.get("id", "unknown")
        payment_id = payload.get("payment_id", "unknown")
        amount = payload.get("amount", 0) / 100
        merchant_id = payload.get("notes", {}).get("merchant_id", "unknown")

        logger.warning(f"[DISPUTE] Dispute created: {dispute_id} for payment {payment_id}")

        self.audit.log_event(
            merchant_id=merchant_id,
            event_type="dispute_created",
            details={
                "dispute_id": dispute_id,
                "payment_id": payment_id,
                "amount": amount,
                "action_required": "evidence_submission",
            },
            severity="warning",
        )

        return {
            "success": True,
            "action_taken": "dispute_alerted",
            "details": {"dispute_id": dispute_id, "payment_id": payment_id}
        }

    # ── Order Events ────────────────────────────────────────

    def _handle_order_paid(self, payload: Dict) -> Dict[str, Any]:
        """Order fully paid — confirm and log."""
        order_id = payload.get("id", "unknown")
        amount = payload.get("amount", 0) / 100
        merchant_id = payload.get("notes", {}).get("merchant_id", "unknown")

        logger.info(f"[ORDER] Order paid: {order_id} (Rs.{amount})")

        self.audit.log_event(
            merchant_id=merchant_id,
            event_type="order_paid",
            details={"order_id": order_id, "amount": amount},
            severity="info",
        )

        return {
            "success": True,
            "action_taken": "order_confirmed",
            "details": {"order_id": order_id, "amount": amount}
        }

    def _handle_order_expired(self, payload: Dict) -> Dict[str, Any]:
        """Order expired — trigger re-engagement."""
        order_id = payload.get("id", "unknown")
        merchant_id = payload.get("notes", {}).get("merchant_id", "unknown")

        logger.warning(f"[EXPIRED] Order expired: {order_id}")

        # Create new payment link for the expired order
        new_link = self.razorpay.create_payment_link(
            amount=payload.get("amount", 10000),
            description=f"Re-engagement for expired order {order_id}",
            notes={
                "merchant_id": merchant_id,
                "original_order_id": order_id,
                "action_type": "re_engagement",
            }
        )

        self.audit.log_event(
            merchant_id=merchant_id,
            event_type="order_expired_reengagement",
            details={
                "order_id": order_id,
                "new_link_created": new_link.get("success", False),
                "new_link": new_link.get("short_url", "N/A"),
            },
            severity="info",
        )

        return {
            "success": True,
            "action_taken": "reengagement_triggered",
            "details": {
                "order_id": order_id,
                "new_link": new_link.get("short_url", "N/A"),
            }
        }

    def get_event_stats(self) -> Dict[str, Any]:
        """Get statistics about processed events."""
        return {
            "total_processed": len(self.processed_events),
            "event_ids": list(self.processed_events)[-10:],  # Last 10
        }
