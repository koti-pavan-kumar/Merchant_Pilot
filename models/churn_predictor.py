import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.schemas import ChurnPrediction

class ChurnPredictor:
    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.pipeline = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        self.model_version = "1.0"
        
    def create_model(self):
        """Create the machine learning model."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model)
        ])
        
    def train(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]):
        """Train the churn prediction model."""
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train model
        self.create_model()
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "churn_rate": y.mean(),
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "model_version": self.model_version
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='f1')
        self.metrics["cv_f1_mean"] = cv_scores.mean()
        self.metrics["cv_f1_std"] = cv_scores.std()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.metrics["feature_importance"] = dict(zip(
                feature_names, self.model.feature_importances_
            ))
        
        print(f"Model trained successfully!")
        print(f"Test Metrics:")
        print(f"  Precision: {self.metrics['precision']:.3f}")
        print(f"  Recall: {self.metrics['recall']:.3f}")
        print(f"  F1 Score: {self.metrics['f1']:.3f}")
        print(f"  ROC AUC: {self.metrics['roc_auc']:.3f}")
        print(f"  CV F1: {self.metrics['cv_f1_mean']:.3f} ± {self.metrics['cv_f1_std']:.3f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict churn probability for merchants."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def predict_single(self, merchant_features) -> Dict[str, Any]:
        """Predict churn for a single merchant."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame with feature names to avoid sklearn warnings
        if not isinstance(merchant_features, pd.DataFrame):
            merchant_features = pd.DataFrame(
                merchant_features, columns=self.feature_names
            )
        
        prediction = self.pipeline.predict(merchant_features)[0]
        probability = self.pipeline.predict_proba(merchant_features)[0, 1]
        
        # Get risk factors based on feature importance
        risk_factors = self._identify_risk_factors(merchant_features)
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "risk_level": self._get_risk_level(probability),
            "risk_factors": risk_factors,
            "confidence": float(max(self.pipeline.predict_proba(merchant_features)[0])),
            "model_version": self.model_version
        }
    
    def _identify_risk_factors(self, features) -> List[str]:
        """Identify top risk factors based on feature importance."""
        if not hasattr(self.model, 'feature_importances_'):
            return ["insufficient_data"]
        
        importance = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        risk_factors = []
        for feature, imp in feature_importance[:5]:
            if imp > 0.05:
                risk_factors.append(feature.replace('_', ' ').title())
        
        return risk_factors if risk_factors else ["low_risk_profile"]
    
    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on churn probability."""
        if probability > 0.8:
            return "critical"
        elif probability > 0.6:
            return "high"
        elif probability > 0.4:
            return "medium"
        else:
            return "low"
    
    def evaluate_on_batch(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance on a batch of data."""
        predictions, probabilities = self.predict(X)
        
        metrics = {
            "precision": precision_score(y, predictions),
            "recall": recall_score(y, predictions),
            "f1": f1_score(y, predictions),
            "roc_auc": roc_auc_score(y, probabilities),
            "confusion_matrix": confusion_matrix(y, predictions).tolist(),
            "batch_size": len(X),
            "churn_rate": y.mean(),
            "predicted_churn_rate": predictions.mean()
        }
        
        return metrics
    
    def save_model(self, model_path: str = "models/saved_models"):
        """Save the trained model to disk."""
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = Path(model_path) / f"churn_model_{timestamp}.joblib"
        metrics_file = Path(model_path) / f"churn_metrics_{timestamp}.json"
        
        # Save model
        joblib.dump(self.pipeline, model_file)
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Model saved to {model_file}")
        print(f"Metrics saved to {metrics_file}")
        
        return model_file, metrics_file
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        self.pipeline = joblib.load(model_path)
        
        # Extract model type from pipeline
        self.model = self.pipeline.named_steps['model']
        self.model_type = type(self.model).__name__
        
        print(f"Model loaded from {model_path}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.pipeline is None:
            return {"status": "not_trained"}
        
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "training_metrics": self.metrics,
            "status": "trained"
        }