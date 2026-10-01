"""
Continuous / Incremental Retraining Engine.
Executes champion-challenger retraining, rolling window adaptation,
drift-triggered batch updates, and model version promotion.
"""
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.database.db_manager import DatabaseManager
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry

class RetrainingEngine:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.registry = ModelRegistry(self.db_manager)
        self.trainer = ModelTrainer(self.db_manager)
        self.feature_engineer = FeatureEngineer()

    def generate_next_version(self, current_version: Optional[str]) -> str:
        """Increments version string (e.g., v1.0.0 -> v1.1.0)."""
        if not current_version or not current_version.startswith("v"):
            return "v1.0.0"
        try:
            parts = current_version.lstrip("v").split(".")
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"v{major}.{minor + 1}.0"
        except Exception:
            return f"v1.{int(datetime.utcnow().timestamp()) % 1000}.0"

    def execute_retraining(
        self,
        historical_df: pd.DataFrame,
        new_batch_df: pd.DataFrame,
        trigger_reason: str = "Drift Threshold Exceeded",
        window_size: int = 10000,
        model_name: Optional[str] = None,
        drift_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Incorporates new transaction data into existing dataset, retrains model,
        evaluates performance, and promotes new version.
        """
        active_model_meta = self.registry.get_active_model_details()
        prev_version = active_model_meta["version"] if active_model_meta else "v0.0.0"
        new_version = self.generate_next_version(prev_version)

        # 1. Merge datasets with window limit
        combined_df = pd.concat([historical_df, new_batch_df], ignore_index=True)
        if len(combined_df) > window_size:
            combined_df = combined_df.iloc[-window_size:].reset_index(drop=True)

        # 2. Feature Engineering
        fe_df = self.feature_engineer.transform(combined_df)

        # 3. Preprocessing
        preprocessor = DataPreprocessor()
        X_combined = preprocessor.fit_transform(fe_df)
        
        y_combined = None
        for lbl_col in ["Class", "is_fraud", "is_ground_truth_fraud"]:
            if lbl_col in combined_df.columns:
                y_combined = combined_df[lbl_col].values
                break

        target_model_name = model_name or (active_model_meta["algorithm"] if active_model_meta else "Isolation Forest")

        # 4. Train model candidates
        trained_models, comparison_df, best_name = self.trainer.train_and_compare_all(
            X_train=X_combined,
            y_train=y_combined,
            contamination=0.03,
            preprocessor=preprocessor
        )

        selected_model = trained_models.get(target_model_name, trained_models[best_name])
        selected_metrics = comparison_df[comparison_df["model_name"] == selected_model.name].iloc[0].to_dict()

        # 5. Save & Register new artifact
        artifact_path = self.trainer.save_model_artifact(
            model=selected_model,
            preprocessor=preprocessor,
            version=new_version,
            metrics=selected_metrics,
            train_samples=len(combined_df),
            set_active=True
        )

        # 6. Log Retraining Event
        self.db_manager.log_retraining_event(
            trigger_reason=trigger_reason,
            prev_ver=prev_version,
            new_ver=new_version,
            dataset_size=len(combined_df),
            drift_metrics=drift_metrics or {},
            status="SUCCESS",
            notes=f"Retrained {selected_model.name} on {len(combined_df)} samples. New PR-AUC/F1 evaluated."
        )

        return {
            "status": "SUCCESS",
            "previous_version": prev_version,
            "new_version": new_version,
            "active_model": selected_model.name,
            "train_samples": len(combined_df),
            "metrics": selected_metrics,
            "all_candidates": comparison_df.to_dict(orient="records"),
            "artifact_path": artifact_path
        }
