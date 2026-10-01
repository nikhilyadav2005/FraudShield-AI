"""
Base anomaly detector interface.
Defines unified API for scoring, 3-tier categorization, and feature attribution.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from src.config import NORMAL_THRESHOLD, SUSPICIOUS_THRESHOLD, LABEL_NORMAL, LABEL_SUSPICIOUS, LABEL_FRAUD

class BaseAnomalyDetector(ABC):
    def __init__(self, name: str, contamination: float = 0.03):
        self.name = name
        self.contamination = contamination
        self.is_fitted = False
        self.min_score_: float = 0.0
        self.max_score_: float = 1.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseAnomalyDetector":
        """Fits anomaly detector on feature matrix X."""
        pass

    @abstractmethod
    def predict_raw_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns raw decision function scores where higher values indicate HIGHER anomaly.
        """
        pass

    def _calibrate_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Calibrates raw anomaly scores to a normalized 0.0 - 1.0 range.
        Uses robust percentile clipping (1st to 99th percentile) to avoid outlier skew.
        """
        if self.max_score_ <= self.min_score_:
            return np.full_like(raw_scores, 0.5, dtype=float)

        clipped = np.clip(raw_scores, self.min_score_, self.max_score_)
        calibrated = (clipped - self.min_score_) / (self.max_score_ - self.min_score_ + 1e-8)
        return np.round(calibrated, 4)

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes calibrated anomaly scores between 0.0 and 1.0."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted.")
        raw = self.predict_raw_scores(X)
        return self._calibrate_scores(raw)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Categorizes transactions into:
        - 'Normal' (score < 0.45)
        - 'Suspicious' (0.45 <= score < 0.70)
        - 'Potentially Fraudulent' (score >= 0.70)
        """
        scores = self.predict_anomaly_scores(X)
        labels = []
        for s in scores:
            if s >= SUSPICIOUS_THRESHOLD:
                labels.append(LABEL_FRAUD)
            elif s >= NORMAL_THRESHOLD:
                labels.append(LABEL_SUSPICIOUS)
            else:
                labels.append(LABEL_NORMAL)
        return np.array(labels)

    def predict_binary(self, X: np.ndarray, threshold: float = SUSPICIOUS_THRESHOLD) -> np.ndarray:
        """Returns 1 for anomaly (flagged) and 0 for normal based on threshold."""
        scores = self.predict_anomaly_scores(X)
        return (scores >= threshold).astype(int)

    def explain_instance(self, x: np.ndarray, feature_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Heuristic / Attribution explanation for an individual transaction.
        Identifies features with greatest deviation or contribution.
        """
        x_flat = np.asarray(x).flatten()
        abs_vals = np.abs(x_flat)
        top_indices = np.argsort(abs_vals)[::-1][:top_k]

        explanations = []
        for idx in top_indices:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            val = float(x_flat[idx])
            direction = "High deviation" if abs(val) > 1.5 else "Moderate anomaly contribution"
            explanations.append({
                "feature": feat_name,
                "value": round(val, 3),
                "contribution": round(abs(val), 3),
                "direction": direction
            })
        return explanations
