"""
Model Trainer and Evaluator.
Executes multi-model training, benchmarking, metric calculation (Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR),
and model artifact serialization.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

from src.config import MODEL_DIR, SUSPICIOUS_THRESHOLD
from src.models.base import BaseAnomalyDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.lof import LOFDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.autoencoder import AutoencoderDetector
from src.database.db_manager import DatabaseManager

class ModelTrainer:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.model_dir = Path(MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def initialize_models(self, contamination: float = 0.03) -> Dict[str, BaseAnomalyDetector]:
        return {
            "Isolation Forest": IsolationForestDetector(contamination=contamination),
            "Local Outlier Factor (LOF)": LOFDetector(contamination=contamination),
            "One-Class SVM": OneClassSVMDetector(contamination=contamination),
            "Neural Autoencoder": AutoencoderDetector(contamination=contamination)
        }

    def evaluate_model(
        self,
        model: BaseAnomalyDetector,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        threshold: float = SUSPICIOUS_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Evaluates model predictions. If y_true is provided, calculates standard classification metrics.
        """
        scores = model.predict_anomaly_scores(X)
        y_pred = (scores >= threshold).astype(int)

        metrics: Dict[str, Any] = {
            "model_name": model.name,
            "anomaly_count": int(np.sum(y_pred)),
            "anomaly_rate_pct": round(float(np.mean(y_pred) * 100), 2),
            "mean_anomaly_score": round(float(np.mean(scores)), 4),
            "max_anomaly_score": round(float(np.max(scores)), 4),
            "min_anomaly_score": round(float(np.min(scores)), 4)
        }

        if y_true is not None and len(np.unique(y_true)) > 1:
            try:
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # ROC-AUC & PR-AUC using continuous calibrated scores
                try:
                    roc_auc = roc_auc_score(y_true, scores)
                except Exception:
                    roc_auc = 0.5
                    
                try:
                    pr_auc = average_precision_score(y_true, scores)
                except Exception:
                    pr_auc = float(np.mean(y_true))

                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                    fpr, fnr = 0.0, 0.0

                metrics.update({
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1_score": round(float(f1), 4),
                    "roc_auc": round(float(roc_auc), 4),
                    "pr_auc": round(float(pr_auc), 4),
                    "fpr": round(float(fpr), 4),
                    "fnr": round(float(fnr), 4),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn)
                })
            except Exception as e:
                metrics["eval_error"] = str(e)

        return metrics

    def train_and_compare_all(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        contamination: float = 0.03,
        preprocessor: Any = None
    ) -> Tuple[Dict[str, BaseAnomalyDetector], pd.DataFrame, str]:
        """
        Trains all 4 anomaly detection models, compares their metrics, and selects the best model.
        Returns:
            trained_models: dict of name -> fitted model
            comparison_df: DataFrame of performance metrics
            best_model_name: string name of the top-performing model
        """
        models = self.initialize_models(contamination=contamination)
        trained_models = {}
        comparison_records = []

        eval_X = X_test if X_test is not None else X_train
        eval_y = y_test if y_test is not None else y_train

        for name, model in models.items():
            # Fit model
            model.fit(X_train)
            trained_models[name] = model

            # Evaluate
            metrics = self.evaluate_model(model, eval_X, eval_y)
            comparison_records.append(metrics)

        comparison_df = pd.DataFrame(comparison_records)

        # Select best model:
        # If ground truth labels exist, rank by PR-AUC (best for severe imbalance) or F1-Score.
        # Otherwise, rank by consistency of anomaly rate to expected contamination.
        if "pr_auc" in comparison_df.columns and comparison_df["pr_auc"].notnull().any():
            best_idx = comparison_df["pr_auc"].idxmax()
        elif "f1_score" in comparison_df.columns and comparison_df["f1_score"].notnull().any():
            best_idx = comparison_df["f1_score"].idxmax()
        else:
            # Pick model whose detected anomaly rate is closest to target contamination
            diffs = (comparison_df["anomaly_rate_pct"] - (contamination * 100)).abs()
            best_idx = diffs.idxmin()

        best_model_name = comparison_df.loc[best_idx, "model_name"]
        return trained_models, comparison_df, best_model_name

    def save_model_artifact(
        self,
        model: BaseAnomalyDetector,
        preprocessor: Any,
        version: str,
        metrics: Dict[str, Any],
        train_samples: int,
        set_active: bool = True
    ) -> str:
        """Serializes model and preprocessor pipeline and registers in DB."""
        filename = f"{model.name.lower().replace(' ', '_')}_{version}.joblib"
        save_path = self.model_dir / filename

        artifact = {
            "model": model,
            "preprocessor": preprocessor,
            "version": version,
            "metrics": metrics,
            "train_samples": train_samples
        }
        joblib.dump(artifact, save_path)

        # Register in SQLite
        feature_names = preprocessor.get_feature_names() if preprocessor else []
        self.db_manager.register_model({
            "model_name": model.name,
            "version": version,
            "algorithm": model.name,
            "contamination": model.contamination,
            "train_samples": train_samples,
            "features": feature_names,
            "metrics": metrics,
            "model_path": str(save_path),
            "is_active": set_active
        })

        return str(save_path)

    def load_model_artifact(self, model_path: str) -> Dict[str, Any]:
        """Loads serialized model and preprocessor artifact."""
        return joblib.load(model_path)
