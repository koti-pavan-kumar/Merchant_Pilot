import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import json
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.scaler = None
        
    def load_data(self, merchants_path: str, transactions_path: str) -> Dict[str, pd.DataFrame]:
        """Load synthetic data from JSON files."""
        with open(merchants_path, 'r') as f:
            merchants_data = json.load(f)
        
        with open(transactions_path, 'r') as f:
            transactions_data = json.load(f)
        
        merchants_df = pd.DataFrame(merchants_data)
        transactions_df = pd.DataFrame(transactions_data)
        
        return {"merchants": merchants_df, "transactions": transactions_df}
    
    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create feature matrix from merchant and transaction data."""
        merchants_df = data["merchants"] if isinstance(data["merchants"], pd.DataFrame) else pd.DataFrame(data["merchants"])
        transactions_df = data["transactions"] if isinstance(data["transactions"], pd.DataFrame) else pd.DataFrame(data["transactions"])
        
        # Aggregate transaction features per merchant
        transaction_features = self._aggregate_transaction_features(transactions_df)
        
        # Merge with merchant features
        features_df = merchants_df.merge(transaction_features, on="merchant_id", how="left")
        
        # Engineer additional features
        features_df = self._engineer_temporal_features(features_df)
        features_df = self._engineer_ratio_features(features_df)
        features_df = self._engineer_trend_features(features_df)
        
        # Select final features
        final_features = self._select_features(features_df)
        
        return final_features
    
    def _aggregate_transaction_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction data per merchant."""
        # Filter to last 90 days
        cutoff_date = datetime.now() - pd.Timedelta(days=90)
        transactions_df['created_at'] = pd.to_datetime(transactions_df['created_at'])
        recent_transactions = transactions_df[transactions_df['created_at'] >= cutoff_date]
        
        # Group by merchant and calculate aggregates
        aggregated = recent_transactions.groupby('merchant_id').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean', 'std', 'min', 'max'],
            'status': lambda x: (x == 'captured').sum() / len(x) if len(x) > 0 else 0,
            'failure_reason': lambda x: x.notna().sum()
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            'merchant_id', 'transaction_count_90d', 'total_amount_90d',
            'avg_amount_90d', 'std_amount_90d', 'min_amount_90d', 'max_amount_90d',
            'success_rate_90d', 'failure_count_90d'
        ]
        
        return aggregated
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        # Days since registration
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        df['days_since_registration'] = (datetime.now() - df['registration_date']).dt.days
        
        # Days since last transaction
        df['last_transaction_date'] = pd.to_datetime(df['last_transaction_date'])
        df['days_since_last_transaction'] = (datetime.now() - df['last_transaction_date']).dt.days
        
        # Transaction frequency (transactions per day since registration)
        df['transaction_frequency'] = df['transaction_count'] / df['days_since_registration'].clip(lower=1)
        
        return df
    
    def _engineer_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and percentage features."""
        # Revenue per transaction
        df['revenue_per_transaction'] = df['total_revenue'] / df['transaction_count'].clip(lower=1)
        
        # Refund to revenue ratio
        df['refund_to_revenue_ratio'] = df['refund_rate'] / (df['total_revenue'] / 1000).clip(lower=1)
        
        # Chargeback intensity
        df['chargeback_intensity'] = df['chargeback_count'] / df['transaction_count'].clip(lower=1)
        
        # Failure to success ratio
        df['failure_success_ratio'] = df['failure_rate'] / (1 - df['failure_rate']).clip(lower=0.01)
        
        return df
    
    def _engineer_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend-based features from weekly transaction data."""
        # Calculate trend slope from weekly_transaction_trend
        def calculate_trend(trend_list):
            if len(trend_list) < 2:
                return 0
            x = np.arange(len(trend_list))
            y = np.array(trend_list)
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        df['transaction_trend_slope'] = df['weekly_transaction_trend'].apply(calculate_trend)
        
        # Trend volatility (standard deviation of weekly trends)
        df['transaction_trend_volatility'] = df['weekly_transaction_trend'].apply(
            lambda x: np.std(x) if len(x) > 0 else 0
        )
        
        # Recent trend vs overall trend
        df['recent_trend_ratio'] = df['weekly_transaction_trend'].apply(
            lambda x: x[-1] / x[0] if len(x) > 1 and x[0] > 0 else 1
        )
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final features for the model."""
        feature_columns = [
            # Merchant profile features
            'total_revenue', 'transaction_count', 'average_order_value',
            'refund_rate', 'failure_rate',
            
            # Temporal features
            'days_since_registration', 'days_since_last_transaction',
            'transaction_frequency',
            
            # Ratio features
            'revenue_per_transaction', 'refund_to_revenue_ratio',
            'chargeback_intensity', 'failure_success_ratio',
            
            # Trend features
            'transaction_trend_slope', 'transaction_trend_volatility',
            'recent_trend_ratio',
            
            # Risk indicators
            'chargeback_count', 'dispute_rate', 'failed_payment_attempts',
            
            # Transaction aggregation features
            'transaction_count_90d', 'total_amount_90d', 'avg_amount_90d',
            'std_amount_90d', 'success_rate_90d', 'failure_count_90d',
            
            # Confounding features (not directly related to churn — makes ML harder)
            'business_age_days', 'revenue_growth_rate',
        ]
        
        # Select features that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        return df[available_features + ['merchant_id', 'health_factor']]
    
    def prepare_training_data(self, features_df: pd.DataFrame):
        """Prepare training data with features and target variable."""
        # Separate features and target
        X = features_df[self.feature_names].fillna(0)
        y = (features_df['health_factor'] < 0.4).astype(int)  # 1 = churned, 0 = not churned
        
        # Store merchant IDs for reference
        merchant_ids = features_df['merchant_id']
        
        return X, y, merchant_ids
    
    def transform_single(self, merchant_data: Dict[str, Any]) -> np.ndarray:
        """Transform a single merchant's data for prediction."""
        # Create a dataframe from the single merchant
        df = pd.DataFrame([merchant_data])
        
        # Apply the same feature engineering steps
        # Note: In production, you'd load and apply saved transformations
        # For demo purposes, we'll create a simplified feature vector
        
        features = []
        for feature_name in self.feature_names:
            if feature_name in df.columns:
                features.append(df[feature_name].iloc[0])
            else:
                features.append(0)  # Default value for missing features
        
        return np.array(features).reshape(1, -1)