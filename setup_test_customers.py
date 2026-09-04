"""
Create 5 Razorpay test customers with realistic transaction patterns.
These are REAL Razorpay customers visible in the dashboard.
"""
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from config import get_settings
from services.razorpay_client import RazorpayClient

settings = get_settings()

# Realistic Indian business profiles
CUSTOMERS = [
    {
        "name": "Priya Sharma",
        "email": "priya.sharma@techsolutions.in",
        "phone": "9876543210",
        "business": "TechSolutions India Pvt Ltd",
        "category": "Electronics",
        "monthly_revenue": 85000,
        "failure_rate": 0.12,
        "days_inactive": 45,
        "description": "Electronics retail - high volume, moderate returns",
    },
    {
        "name": "Rajesh Patel",
        "email": "rajesh.patel@greenfresh.in",
        "phone": "9876543211",
        "business": "GreenFresh Organics",
        "category": "Food & Beverage",
        "monthly_revenue": 120000,
        "failure_rate": 0.08,
        "days_inactive": 0,
        "description": "Organic food delivery - steady growth, low churn",
    },
    {
        "name": "Anita Desai",
        "email": "anita.desai@stylehub.in",
        "phone": "9876543212",
        "business": "StyleHub Fashion",
        "category": "Fashion",
        "monthly_revenue": 65000,
        "failure_rate": 0.18,
        "days_inactive": 21,
        "description": "Fashion e-commerce - seasonal patterns, high refunds",
    },
    {
        "name": "Vikram Singh",
        "email": "vikram.singh@quickfix.in",
        "phone": "9876543213",
        "business": "QuickFix Services",
        "category": "Services",
        "monthly_revenue": 45000,
        "failure_rate": 0.25,
        "days_inactive": 67,
        "description": "Home services - payment failures, declining orders",
    },
    {
        "name": "Meera Nair",
        "email": "meera.nair@booknook.in",
        "phone": "9876543214",
        "business": "BookNook Education",
        "category": "Education",
        "monthly_revenue": 95000,
        "failure_rate": 0.06,
        "days_inactive": 0,
        "description": "EdTech platform - subscription model, low churn",
    },
]

def calculate_health_score(customer):
    """Calculate merchant health score based on metrics."""
    failure_penalty = customer["failure_rate"] * 100
    inactivity_penalty = min(customer["days_inactive"] / 90, 1) * 40
    score = max(0, min(100, int(100 - failure_penalty - inactivity_penalty)))
    return score

def risk_level(score):
    if score >= 70: return "low"
    if score >= 50: return "medium"
    if score >= 30: return "high"
    return "critical"

def main():
    print("=" * 60)
    print("  MerchantPilot AI - Test Customer Setup")
    print("=" * 60)
    
    razorpay = RazorpayClient()
    
    if razorpay.simulation_mode:
        print("\n[ERROR] Razorpay API keys not configured!")
        print("Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in .env")
        return
    
    print(f"\n  Mode: LIVE (Razorpay Test Mode)")
    print(f"  Creating {len(CUSTOMERS)} test customers...\n")
    
    created_customers = []
    
    for i, profile in enumerate(CUSTOMERS, 1):
        print(f"--- Customer {i}/{len(CUSTOMERS)}: {profile['business']} ---")
        
        # Create customer
        result = razorpay.create_customer(
            name=profile["name"],
            email=profile["email"],
            phone=profile["phone"],
            notes={
                "business": profile["business"],
                "category": profile["category"],
                "setup_source": "merchantpilot_test",
            }
        )
        
        if not result.get("success"):
            print(f"  [FAIL] Customer creation failed: {result.get('error', 'Unknown')}")
            continue
        
        customer_id = result["customer_id"]
        print(f"  [OK] Customer: {customer_id} ({profile['name']})")
        
        # Create a few orders for this customer (realistic pattern)
        orders_created = []
        amount_variants = [
            int(profile["monthly_revenue"] * 0.3),  # 30% of monthly
            int(profile["monthly_revenue"] * 0.15), # 15% of monthly
            int(profile["monthly_revenue"] * 0.08), # 8% of monthly
        ]
        
        for j, amount in enumerate(amount_variants):
            order_result = razorpay.create_order(
                amount=amount * 100,  # Convert to paise
                receipt=f"test_{customer_id}_{j+1}",
                notes={
                    "customer_id": customer_id,
                    "merchant_name": profile["business"],
                    "setup_source": "merchantpilot_test",
                }
            )
            
            if order_result.get("success"):
                orders_created.append(order_result["order_id"])
                print(f"  [OK] Order {j+1}: {order_result['order_id']} (Rs.{amount:,.0f})")
            else:
                print(f"  [WARN] Order {j+1} failed: {order_result.get('error', '')[:50]}")
            
            time.sleep(0.5)  # Rate limit
        
        # Create a payment link for recovery demo
        link_result = razorpay.create_payment_link(
            amount=amount_variants[0] * 100,
            description=f"Recovery: {profile['business']}",
            notes={
                "customer_id": customer_id,
                "action_type": "recovery",
                "setup_source": "merchantpilot_test",
            }
        )
        
        payment_link = ""
        if link_result.get("success"):
            payment_link = link_result.get("short_url", "")
            print(f"  [OK] Payment Link: {payment_link}")
        
        # Store customer data
        health = calculate_health_score(profile)
        customer_data = {
            "customer_id": customer_id,
            "razorpay_name": profile["name"],
            "razorpay_email": profile["email"],
            "razorpay_phone": profile["phone"],
            "business_name": profile["business"],
            "category": profile["category"],
            "description": profile["description"],
            "monthly_revenue": profile["monthly_revenue"],
            "failure_rate": profile["failure_rate"],
            "days_since_last_transaction": profile["days_inactive"],
            "health_score": health,
            "risk_level": risk_level(health),
            "total_revenue": profile["monthly_revenue"] * random.randint(6, 18),
            "orders_created": len(orders_created),
            "order_ids": orders_created,
            "payment_link": payment_link,
            "status": "churned" if health < 30 else "at_risk" if health < 50 else "active",
            "created_at": datetime.now().isoformat(),
            "is_real_razorpay_customer": True,
        }
        created_customers.append(customer_data)
        print()
    
    # Save to file
    output_path = Path("data/razorpay_customers.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(created_customers, f, indent=2)
    
    print("=" * 60)
    print(f"  CREATED {len(created_customers)}/{len(CUSTOMERS)} CUSTOMERS")
    print("=" * 60)
    
    for c in created_customers:
        score = c["health_score"]
        color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        print(f"  {color} {c['business_name']} | {c['customer_id']} | Score: {score} | {c['risk_level'].upper()}")
    
    print(f"\n  Saved to: {output_path}")
    print(f"  Check Razorpay Dashboard → Customers tab\n")

if __name__ == "__main__":
    main()
