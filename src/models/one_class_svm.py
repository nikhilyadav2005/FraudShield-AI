"""
One-Class Support Vector Machine (SVM) Anomaly Detector.
Learns a non-linear decision boundary around normal data points in high-dimensional Hilbert space.
"""
from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.svm import OneClassSVM
from src.models.base import BaseAnomalyDetector

class OneClassSVMDetector(BaseAnomalyDetector):
    def __init__(self, contamination: float = 0.03, kernel: str = "rbf", gamma: str = "scale"):
        super().__init__(name="One-Class SVM", contamination=contamination)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = max(0.01, min(0.5, contamination))
        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OneClassSVMDetector":
        # If dataset is very large (> 10000), sample for SVM efficiency
        if len(X) > 10000:
            sample_idx = np.random.choice(len(X), size=10000, replace=False)
            self.model.fit(X[sample_idx])
        else:
            self.model.fit(X)
        
        raw_train_scores = -self.model.decision_function(X)
        self.min_score_ = float(np.percentile(raw_train_scores, 1))
        self.max_score_ = float(np.percentile(raw_train_scores, 99))
        self.is_fitted = True
        return self

    def predict_raw_scores(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)

    def explain_instance(self, x: np.ndarray, feature_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        x_flat = np.asarray(x).flatten()
        abs_vals = np.abs(x_flat)
        top_indices = np.argsort(abs_vals)[::-1][:top_k]

        explanations = []
        for idx in top_indices:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            val = float(x_flat[idx])
            explanations.append({
                "feature": feat_name,
                "value": round(val, 3),
                "contribution": round(abs(val), 3),
                "direction": "Outside high-density support boundary" if abs(val) > 1.5 else "Boundary proximity"
            })
        return explanations
