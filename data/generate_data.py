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
    def __init__(self, num_merchants: int = 100):
        self.num_merchants = num_merchants
        self.merchant_profiles = []
        self.transactions = []
        
    def generate_merchant_profiles(self) -> List[Dict[str, Any]]:
        """Generate synthetic merchant profiles with varying characteristics."""
        categories = [
            "E-commerce", "Food & Beverage", "Education", "Healthcare",
            "Travel", "Entertainment", "Utilities", "Fashion", "Electronics"
        ]
        
        merchants = []
        for i in range(self.num_merchants):
            # Create merchant with realistic distribution of health
            health_factor = random.random()  # 0 = churned, 1 = very healthy
            
            registration_date = fake.date_time_between(start_date="-2y", end_date="now")
            
            # Calculate metrics based on health factor
            base_revenue = random.uniform(10000, 1000000)  # INR
            revenue_multiplier = health_factor * 0.8 + 0.2  # 0.2x to 1x
            
            merchant = {
                "merchant_id": f"M{i+1:04d}",
                "business_name": fake.company(),
                "category": random.choice(categories),
                "registration_date": registration_date.isoformat(),
                "last_transaction_date": (
                    datetime.now() - timedelta(days=int((1 - health_factor) * 90))
                ).isoformat(),
                "total_revenue": base_revenue * revenue_multiplier,
                "transaction_count": int(random.uniform(100, 10000) * revenue_multiplier),
                "average_order_value": random.uniform(500, 5000),
                "refund_rate": max(0, 0.1 - (health_factor * 0.08) + random.uniform(-0.02, 0.02)),
                "failure_rate": max(0, 0.15 - (health_factor * 0.12) + random.uniform(-0.02, 0.02)),
                "days_since_last_transaction": int((1 - health_factor) * 90),
                "weekly_transaction_trend": self._generate_trend(health_factor),
                "revenue_growth_rate": (health_factor - 0.5) * 0.4 + random.uniform(-0.1, 0.1),
                "chargeback_count": int(max(0, (1 - health_factor) * 10 + random.randint(0, 3))),
                "dispute_rate": max(0, (1 - health_factor) * 0.05 + random.uniform(-0.01, 0.01)),
                "failed_payment_attempts": int(max(0, (1 - health_factor) * 20 + random.randint(0, 5))),
                "status": self._determine_status(health_factor),
                "health_factor": health_factor  # For training data
            }
            merchants.append(merchant)
            
        self.merchant_profiles = merchants
        return merchants
    
    def _generate_trend(self, health_factor: float) -> List[float]:
        """Generate weekly transaction trend based on health factor."""
        trend = []
        base = health_factor * 100
        for week in range(8):
            # Add some randomness but overall trend matches health
            noise = random.uniform(-20, 20)
            trend.append(max(0, base + noise - (week * (1 - health_factor) * 5)))
        return trend
    
    def _determine_status(self, health_factor: float) -> str:
        """Determine merchant status based on health factor."""
        if health_factor > 0.7:
            return MerchantStatus.ACTIVE.value
        elif health_factor > 0.4:
            return MerchantStatus.AT_RISK.value
        elif health_factor > 0.2:
            return MerchantStatus.RECOVERED.value
        else:
            return MerchantStatus.CHURNED.value
    
    def generate_transactions(self, transactions_per_merchant: int = 50) -> List[Dict[str, Any]]:
        """Generate synthetic transactions for each merchant."""
        transactions = []
        
        for merchant in self.merchant_profiles:
            # Generate transactions based on merchant health
            health_factor = merchant["health_factor"]
            num_transactions = int(transactions_per_merchant * health_factor)
            
            for i in range(num_transactions):
                days_ago = random.randint(0, 90)
                transaction_date = datetime.now() - timedelta(days=days_ago)
                
                # Transaction success rate decreases with poor health
                success_probability = health_factor * 0.9 + 0.1
                is_successful = random.random() < success_probability
                
                transaction = {
                    "transaction_id": f"T{merchant['merchant_id']}{i+1:03d}",
                    "merchant_id": merchant["merchant_id"],
                    "amount": random.uniform(100, 10000),
                    "currency": "INR",
                    "status": "captured" if is_successful else random.choice(["failed", "refunded"]),
                    "payment_method": random.choice(["upi", "card", "netbanking", "wallet"]),
                    "created_at": transaction_date.isoformat(),
                    "failure_reason": None if is_successful else random.choice([
                        "insufficient_funds", "expired_card", "network_timeout", "user_cancelled"
                    ])
                }
                transactions.append(transaction)
                
        self.transactions = transactions
        return transactions
    
    def save_to_files(self, output_dir: str = "data/synthetic"):
        """Save generated data to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save merchants
        with open(output_path / "merchants.json", "w") as f:
            json.dump(self.merchant_profiles, f, indent=2)
        
        # Save transactions
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
        "average_revenue": sum(m["total_revenue"] for m in merchants) / len(merchants)
    }
    
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()