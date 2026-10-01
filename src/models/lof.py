"""
Local Outlier Factor (LOF) Anomaly Detector.
Measures local deviation of density with respect to its k-nearest neighbors.
Uses novelty=True to support streaming inference on unseen transactions.
"""
from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from src.models.base import BaseAnomalyDetector

class LOFDetector(BaseAnomalyDetector):
    def __init__(self, contamination: float = 0.03, n_neighbors: int = 20):
        super().__init__(name="Local Outlier Factor (LOF)", contamination=contamination)
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,  # Crucial for inference on new transactions
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LOFDetector":
        # Ensure n_neighbors <= sample count
        if len(X) <= self.n_neighbors:
            self.model.n_neighbors = max(2, len(X) - 1)
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
                "direction": "Low local neighbor density" if abs(val) > 1.5 else "Subtle neighbor outlier"
            })
        return explanations
