import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.churn_predictor import ChurnPredictor
from models.feature_engineer import FeatureEngineer

class TestChurnPredictor:
    """Test suite for churn prediction model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.predictor = ChurnPredictor(model_type="gradient_boosting")
        self.feature_engineer = FeatureEngineer()
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        self.X = pd.DataFrame({
            'total_revenue': np.random.uniform(10000, 1000000, n_samples),
            'transaction_count': np.random.randint(100, 10000, n_samples),
            'average_order_value': np.random.uniform(500, 5000, n_samples),
            'refund_rate': np.random.uniform(0, 0.2, n_samples),
            'failure_rate': np.random.uniform(0, 0.15, n_samples),
            'days_since_registration': np.random.randint(30, 730, n_samples),
            'days_since_last_transaction': np.random.randint(0, 90, n_samples),
            'transaction_frequency': np.random.uniform(0.1, 5.0, n_samples),
            'revenue_per_transaction': np.random.uniform(100, 5000, n_samples),
            'refund_to_revenue_ratio': np.random.uniform(0, 0.1, n_samples),
            'chargeback_intensity': np.random.uniform(0, 0.05, n_samples),
            'failure_success_ratio': np.random.uniform(0, 0.2, n_samples),
            'transaction_trend_slope': np.random.uniform(-10, 10, n_samples),
            'transaction_trend_volatility': np.random.uniform(0, 20, n_samples),
            'recent_trend_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'chargeback_count': np.random.randint(0, 10, n_samples),
            'dispute_rate': np.random.uniform(0, 0.05, n_samples),
            'failed_payment_attempts': np.random.randint(0, 20, n_samples),
            'transaction_count_90d': np.random.randint(0, 500, n_samples),
            'total_amount_90d': np.random.uniform(0, 500000, n_samples),
            'avg_amount_90d': np.random.uniform(500, 5000, n_samples),
            'std_amount_90d': np.random.uniform(0, 2000, n_samples),
            'success_rate_90d': np.random.uniform(0.8, 1.0, n_samples),
            'failure_count_90d': np.random.randint(0, 50, n_samples)
        })
        
        # Generate target variable (churned = 1, not churned = 0)
        # Higher churn probability for merchants with poor metrics
        churn_probability = (
            (self.X['failure_rate'] > 0.1).astype(int) * 0.3 +
            (self.X['refund_rate'] > 0.1).astype(int) * 0.2 +
            (self.X['days_since_last_transaction'] > 30).astype(int) * 0.3 +
            (self.X['chargeback_count'] > 5).astype(int) * 0.2 +
            np.random.uniform(0, 0.2, n_samples)
        )
        self.y = (churn_probability > 0.5).astype(int)
        
        self.feature_names = list(self.X.columns)
    
    def test_model_creation(self):
        """Test model creation."""
        assert self.predictor.model is None
        
        self.predictor.create_model()
        assert self.predictor.model is not None
        assert self.predictor.pipeline is not None
    
    def test_model_training(self):
        """Test model training."""
        metrics = self.predictor.train(self.X, self.y, self.feature_names)
        
        assert metrics is not None
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_prediction(self):
        """Test model prediction."""
        self.predictor.train(self.X, self.y, self.feature_names)
        
        predictions, probabilities = self.predictor.predict(self.X[:10])
        
        assert len(predictions) == 10
        assert len(probabilities) == 10
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_single_prediction(self):
        """Test single merchant prediction."""
        self.predictor.train(self.X, self.y, self.feature_names)
        
        single_features = self.X.iloc[0].values.reshape(1, -1)
        result = self.predictor.predict_single(single_features)
        
        assert 'churn_prediction' in result
        assert 'churn_probability' in result
        assert 'risk_level' in result
        assert 'risk_factors' in result
        assert result['risk_level'] in ['low', 'medium', 'high', 'critical']
    
    def test_batch_evaluation(self):
        """Test batch evaluation."""
        self.predictor.train(self.X, self.y, self.feature_names)
        
        metrics = self.predictor.evaluate_on_batch(self.X[:100], self.y[:100])
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'batch_size' in metrics
        assert metrics['batch_size'] == 100
    
    def test_model_save_load(self):
        """Test model save and load."""
        self.predictor.train(self.X, self.y, self.feature_names)
        
        # Save model
        model_file, metrics_file = self.predictor.save_model("test_models")
        
        # Create new predictor and load model
        new_predictor = ChurnPredictor()
        new_predictor.load_model(model_file)
        
        # Test prediction with loaded model
        single_features = self.X.iloc[0].values.reshape(1, -1)
        result = new_predictor.predict_single(single_features)
        
        assert 'churn_prediction' in result
        assert 'churn_probability' in result
        
        # Cleanup
        os.remove(model_file)
        os.remove(metrics_file)
        os.rmdir("test_models")
    
    def test_model_info(self):
        """Test model info retrieval."""
        # Before training
        info = self.predictor.get_model_info()
        assert info['status'] == 'not_trained'
        
        # After training
        self.predictor.train(self.X, self.y, self.feature_names)
        info = self.predictor.get_model_info()
        
        assert info['status'] == 'trained'
        assert 'model_type' in info
        assert 'feature_count' in info
        assert 'training_metrics' in info

class TestFeatureEngineer:
    """Test suite for feature engineering."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engineer = FeatureEngineer()
        
        # Create minimal test data
        self.test_data = {
            "merchants": [
                {
                    "merchant_id": "M0001",
                    "business_name": "Test Shop",
                    "category": "E-commerce",
                    "registration_date": "2024-01-01T00:00:00",
                    "last_transaction_date": "2024-12-01T00:00:00",
                    "total_revenue": 100000,
                    "transaction_count": 500,
                    "average_order_value": 200,
                    "refund_rate": 0.05,
                    "failure_rate": 0.02,
                    "days_since_last_transaction": 30,
                    "weekly_transaction_trend": [100, 110, 120, 115, 125, 130, 135, 140],
                    "revenue_growth_rate": 0.1,
                    "chargeback_count": 2,
                    "dispute_rate": 0.01,
                    "failed_payment_attempts": 5,
                    "status": "active",
                    "health_factor": 0.8
                }
            ],
            "transactions": [
                {
                    "transaction_id": "T0001",
                    "merchant_id": "M0001",
                    "amount": 1000,
                    "currency": "INR",
                    "status": "captured",
                    "payment_method": "upi",
                    "created_at": "2024-12-01T10:00:00",
                    "failure_reason": None
                }
            ]
        }
    
    def test_feature_creation(self):
        """Test feature creation from raw data."""
        features_df = self.engineer.create_features(self.test_data)
        
        assert features_df is not None
        assert len(features_df) > 0
        assert 'merchant_id' in features_df.columns
        assert 'health_factor' in features_df.columns
    
    def test_feature_selection(self):
        """Test feature selection."""
        features_df = self.engineer.create_features(self.test_data)
        
        assert len(self.engineer.feature_names) > 0
        assert all(feature in features_df.columns for feature in self.engineer.feature_names)
    
    def test_training_data_preparation(self):
        """Test training data preparation."""
        features_df = self.engineer.create_features(self.test_data)
        X, y, merchant_ids = self.engineer.prepare_training_data(features_df)
        
        assert X.shape[0] == y.shape[0]
        assert len(merchant_ids) == X.shape[0]
        assert all(label in [0, 1] for label in y)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])