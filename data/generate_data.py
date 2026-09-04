import random
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import pandas as np
from faker import Faker
from .schemas import MerchantMetrics, Transaction, MerchantStatus

fake = Faker()
Faker.seed(42)
random.seed(42)


class SyntheticDataGenerator:
    """Generates realistic synthetic merchant data with overlapping classes.

    Key design decision: health_factor is NOT a direct linear predictor.
    We add noise, confounding features, and edge cases so that an ML model
    cannot achieve 100% precision — which is exactly what judges expect.
    """

    def __init__(self, num_merchants: int = 100):
        self.num_merchants = num_merchants
        self.merchant_profiles = []
        self.transactions = []

    def generate_merchant_profiles(self) -> List[Dict[str, Any]]:
        """Generate synthetic merchant profiles with realistic noise."""
        categories = [
            "E-commerce", "Food & Beverage", "Education", "Healthcare",
            "Travel", "Entertainment", "Utilities", "Fashion", "Electronics"
        ]

        merchants = []
        for i in range(self.num_merchants):
            # Base health: uniform random, but we'll add noise later
            base_health = random.random()

            # Add 15% "edge cases" — merchants where health doesn't match metrics
            # This prevents the model from learning a trivial mapping
            is_edge_case = random.random() < 0.15
            if is_edge_case:
                # Flip: healthy merchant with bad metrics, or unhealthy with good metrics
                display_health = 1.0 - base_health
            else:
                display_health = base_health

            registration_date = fake.date_time_between(start_date="-2y", end_date="now")

            # Revenue with significant noise (not directly tied to health)
            base_revenue = random.uniform(10000, 1000000)
            noise_factor = random.uniform(0.3, 1.5)  # Large noise
            revenue = base_revenue * noise_factor

            # Transaction count with noise
            txn_count = int(random.uniform(100, 10000) * random.uniform(0.4, 1.2))

            # Failure rate: has a correlation with health but LOTS of noise
            base_failure = 0.15 * (1 - display_health)
            failure_noise = random.gauss(0, 0.05)  # Gaussian noise
            failure_rate = max(0.01, min(0.40, base_failure + failure_noise))

            # Refund rate: weakly correlated, noisy
            base_refund = 0.10 * (1 - display_health)
            refund_noise = random.gauss(0, 0.03)
            refund_rate = max(0.005, min(0.25, base_refund + refund_noise))

            # Days since last transaction: correlated but noisy
            base_inactive = int((1 - display_health) * 90)
            inactive_noise = random.randint(-20, 20)
            days_inactive = max(0, min(120, base_inactive + inactive_noise))

            # Chargebacks: noisy
            base_chargebacks = int((1 - display_health) * 8)
            chargeback_count = max(0, base_chargebacks + random.randint(-2, 3))

            # Failed payment attempts: noisy
            base_failed = int((1 - display_health) * 15)
            failed_attempts = max(0, base_failed + random.randint(-3, 5))

            # DISCORDANT FEATURES — these actively fight the pattern
            # Some churned merchants have low failure rates (they just stopped transacting)
            # Some healthy merchants have high failure rates (high volume = more failures)
            if random.random() < 0.2:
                failure_rate = random.uniform(0.01, 0.05)  # Low failures but still churned
            if random.random() < 0.2:
                failure_rate = random.uniform(0.20, 0.35)  # High failures but still healthy

            # CONFOUNDING FEATURES — not related to churn at all
            # These trick simple models into learning wrong patterns
            avg_order_value = random.uniform(200, 8000)  # Unrelated to churn
            business_age_days = random.randint(30, 730)  # Unrelated to churn

            # Revenue growth rate: slightly correlated but very noisy
            growth_noise = random.gauss(0, 0.15)
            revenue_growth = (display_health - 0.5) * 0.3 + growth_noise

            # Dispute rate: weakly correlated
            dispute_noise = random.gauss(0, 0.015)
            dispute_rate = max(0, (1 - display_health) * 0.04 + dispute_noise)

            # Determine status from display_health
            if display_health > 0.7:
                status = MerchantStatus.ACTIVE.value
            elif display_health > 0.4:
                status = MerchantStatus.AT_RISK.value
            elif display_health > 0.2:
                status = MerchantStatus.RECOVERED.value
            else:
                status = MerchantStatus.CHURNED.value

            merchant = {
                "merchant_id": f"M{i+1:04d}",
                "business_name": fake.company(),
                "category": random.choice(categories),
                "registration_date": registration_date.isoformat(),
                "last_transaction_date": (
                    datetime.now() - timedelta(days=days_inactive)
                ).isoformat(),
                "total_revenue": round(revenue, 2),
                "transaction_count": txn_count,
                "average_order_value": round(avg_order_value, 2),
                "refund_rate": round(refund_rate, 4),
                "failure_rate": round(failure_rate, 4),
                "days_since_last_transaction": days_inactive,
                "weekly_transaction_trend": self._generate_trend(display_health),
                "revenue_growth_rate": round(revenue_growth, 4),
                "chargeback_count": chargeback_count,
                "dispute_rate": round(max(0, dispute_rate), 4),
                "failed_payment_attempts": failed_attempts,
                "status": status,
                "health_factor": round(display_health, 4),
                # New: confounding features (not related to churn)
                "business_age_days": business_age_days,
                "is_edge_case": is_edge_case,
            }
            merchants.append(merchant)

        self.merchant_profiles = merchants
        return merchants

    def _generate_trend(self, health_factor: float) -> List[float]:
        """Generate weekly transaction trend with noise."""
        trend = []
        base = health_factor * 100
        for week in range(8):
            # Much larger noise than before
            noise = random.gauss(0, 25)
            weekly_value = max(0, base + noise - (week * (1 - health_factor) * 3))
            trend.append(round(weekly_value, 1))
        return trend

    def generate_transactions(self, transactions_per_merchant: int = 50) -> List[Dict[str, Any]]:
        """Generate synthetic transactions for each merchant."""
        transactions = []

        for merchant in self.merchant_profiles:
            # Number of transactions varies widely
            num_transactions = int(
                transactions_per_merchant * random.uniform(0.3, 1.5)
            )

            for i in range(num_transactions):
                days_ago = random.randint(0, 120)  # Wider window
                transaction_date = datetime.now() - timedelta(days=days_ago)

                # Success rate: loosely correlated with health but noisy
                base_success = merchant["health_factor"] * 0.7 + 0.2
                success_prob = max(0.3, min(0.95, base_success + random.gauss(0, 0.1)))
                is_successful = random.random() < success_prob

                transaction = {
                    "transaction_id": f"T{merchant['merchant_id']}{i+1:03d}",
                    "merchant_id": merchant["merchant_id"],
                    "amount": round(random.uniform(100, 10000), 2),
                    "currency": "INR",
                    "status": "captured" if is_successful else random.choice(
                        ["failed", "refunded"]
                    ),
                    "payment_method": random.choice(
                        ["upi", "card", "netbanking", "wallet"]
                    ),
                    "created_at": transaction_date.isoformat(),
                    "failure_reason": None if is_successful else random.choice([
                        "insufficient_funds", "expired_card", "network_timeout",
                        "user_cancelled", "bank_declined", "otp_expired"
                    ]),
                }
                transactions.append(transaction)

        self.transactions = transactions
        return transactions

    def save_to_files(self, output_dir: str = "data/synthetic"):
        """Save generated data to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "merchants.json", "w") as f:
            json.dump(self.merchant_profiles, f, indent=2)

        with open(output_path / "transactions.json", "w") as f:
            json.dump(self.transactions, f, indent=2)

        print(f"Generated {len(self.merchant_profiles)} merchants")
        print(f"Generated {len(self.transactions)} transactions")
        print(f"Data saved to {output_path}")


def main():
    """Main function to generate synthetic data."""
    generator = SyntheticDataGenerator(num_merchants=100)
    generator.generate_merchant_profiles()
    generator.generate_transactions(transactions_per_merchant=50)
    generator.save_to_files()

    # Create summary statistics
    merchants = generator.merchant_profiles
    stats = {
        "total_merchants": len(merchants),
        "average_health_factor": sum(m["health_factor"] for m in merchants) / len(merchants),
        "churned_merchants": sum(1 for m in merchants if m["status"] == "churned"),
        "at_risk_merchants": sum(1 for m in merchants if m["status"] == "at_risk"),
        "active_merchants": sum(1 for m in merchants if m["status"] == "active"),
        "total_transactions": len(generator.transactions),
        "average_revenue": sum(m["total_revenue"] for m in merchants) / len(merchants),
    }

    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
