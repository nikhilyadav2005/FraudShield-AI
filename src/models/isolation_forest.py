"""
Isolation Forest Anomaly Detector.
Partitions feature space using random decision trees; anomalies require fewer splits to isolate.
"""
from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest
from src.models.base import BaseAnomalyDetector

class IsolationForestDetector(BaseAnomalyDetector):
    def __init__(self, contamination: float = 0.03, n_estimators: int = 150, random_state: int = 42):
        super().__init__(name="Isolation Forest", contamination=contamination)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        self.model.fit(X)
        # In sklearn Isolation Forest, lower score = more abnormal
        # We invert so higher = more abnormal
        raw_train_scores = -self.model.decision_function(X)
        self.min_score_ = float(np.percentile(raw_train_scores, 1))
        self.max_score_ = float(np.percentile(raw_train_scores, 99))
        self.is_fitted = True
        return self

    def predict_raw_scores(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)

    def explain_instance(self, x: np.ndarray, feature_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Explain anomaly based on single-tree leaf depth variance and magnitude.
        """
        x_2d = np.asarray(x).reshape(1, -1)
        # Individual sample feature deviations scaled by global variance
        scores = np.abs(x_2d[0])
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        explanations = []
        for idx in top_idx:
            feat = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            val = float(x_2d[0, idx])
            explanations.append({
                "feature": feat,
                "value": round(val, 3),
                "contribution": round(float(scores[idx]), 3),
                "direction": "Short path isolation / High deviation" if abs(val) > 1.5 else "Moderate anomaly"
            })
        return explanations
