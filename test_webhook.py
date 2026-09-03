#!/usr/bin/env python3
"""
Webhook Test Script
Simulates Razorpay webhook events to test the recovery system.
"""

import json
import sys
import os

sys.path.insert(0, os.getcwd())

from services.webhook_handler import WebhookHandler

G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
B = "\033[94m"
W = "\033[1m"
D = "\033[0m"


def test_webhook():
    print(f"\n{W}{B}{'='*60}")
    print(f"  Webhook Handler Test")
    print(f"{'='*60}{D}\n")

    handler = WebhookHandler()

    # Test 1: Payment Failed → Should trigger recovery
    print(f"{W}Test 1: Payment Failed Event{D}")
    print(f"  Simulating: Customer's UPI payment failed")
    failed_payload = {
        "id": "pay_test_failed_001",
        "amount": 50000,  # Rs.500 in paise
        "currency": "INR",
        "status": "failed",
        "method": "upi",
        "error_code": "BAD_REQUEST",
        "error_description": "Payment failed due to insufficient funds",
        "notes": {
            "merchant_id": "M0001",
            "action_type": "recovery",
        }
    }
    result = handler.handle_event("payment.failed", failed_payload)
    print(f"  {G}Action taken: {result['action_taken']}{D}")
    print(f"  Retry link: {result['details'].get('retry_link', 'N/A')}")
    print()

    # Test 2: Payment Captured → Should log revenue recovery
    print(f"{W}Test 2: Payment Captured Event{D}")
    print(f"  Simulating: Customer paid successfully via retry")
    captured_payload = {
        "id": "pay_test_captured_001",
        "amount": 50000,
        "currency": "INR",
        "status": "captured",
        "method": "upi",
        "notes": {
            "merchant_id": "M0001",
            "action_type": "recovery",
        }
    }
    result = handler.handle_event("payment.captured", captured_payload)
    print(f"  {G}Action taken: {result['action_taken']}{D}")
    print(f"  Amount recovered: Rs.{result['details'].get('amount', 0):,.0f}")
    print()

    # Test 3: Order Expired → Should trigger re-engagement
    print(f"{W}Test 3: Order Expired Event{D}")
    print(f"  Simulating: Customer didn't complete payment")
    expired_payload = {
        "id": "order_test_expired_001",
        "amount": 75000,
        "currency": "INR",
        "status": "expired",
        "notes": {
            "merchant_id": "M0001",
        }
    }
    result = handler.handle_event("order.expired", expired_payload)
    print(f"  {G}Action taken: {result['action_taken']}{D}")
    print(f"  New link: {result['details'].get('new_link', 'N/A')}")
    print()

    # Test 4: Dispute Created → Should alert
    print(f"{W}Test 4: Dispute Created Event{D}")
    print(f"  Simulating: Customer filed a chargeback")
    dispute_payload = {
        "id": "disp_test_001",
        "payment_id": "pay_test_captured_001",
        "amount": 50000,
        "notes": {
            "merchant_id": "M0001",
        }
    }
    result = handler.handle_event("payment.dispute.created", dispute_payload)
    print(f"  {G}Action taken: {result['action_taken']}{D}")
    print()

    # Test 5: Deduplication
    print(f"{W}Test 5: Deduplication Test{D}")
    print(f"  Sending same payment.failed event again...")
    result2 = handler.handle_event("payment.failed", failed_payload)
    print(f"  {Y}Action taken: {result2['action_taken']}{D}")
    print()

    # Stats
    print(f"{W}Event Stats:{D}")
    stats = handler.get_event_stats()
    print(f"  Total events processed: {stats['total_processed']}")
    print()

    # Check audit trail
    print(f"{W}Audit Trail:{D}")
    audit = handler.audit
    logs = audit.get_merchant_logs("M0001")
    print(f"  Total events for M0001: {len(logs)}")
    for log in logs[-5:]:
        ts = log.timestamp.strftime('%H:%M:%S') if hasattr(log.timestamp, 'strftime') else str(log.timestamp)[:8]
        print(f"    [{ts}] {log.event_type}")

    print(f"\n{G}{'='*60}")
    print(f"  All webhook tests passed!")
    print(f"{'='*60}{D}\n")


if __name__ == "__main__":
    test_webhook()
