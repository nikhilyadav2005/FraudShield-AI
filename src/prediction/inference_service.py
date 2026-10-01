"""
Inference Service for real-time and batch credit card fraud scoring.
Applies active model, preprocessor, generates calibrated fraud scores,
3-tier classification, and human-interpretable reason codes.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

from src.database.db_manager import DatabaseManager
from src.models.registry import ModelRegistry
from src.data.feature_engineering import FeatureEngineer
from src.config import NORMAL_THRESHOLD, SUSPICIOUS_THRESHOLD, LABEL_NORMAL, LABEL_SUSPICIOUS, LABEL_FRAUD

class InferenceService:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.registry = ModelRegistry(self.db_manager)
        self.feature_engineer = FeatureEngineer()

    def score_transactions(
        self,
        df: pd.DataFrame,
        log_to_db: bool = True
    ) -> pd.DataFrame:
        """
        Scores a batch of transactions using the currently active model.
        Returns enriched DataFrame with anomaly_score, prediction_label, and risk_factors.
        """
        artifact = self.registry.load_active_artifact()
        if not artifact:
            raise RuntimeError("No active model found in registry. Please train a baseline model first.")

        model = artifact["model"]
        preprocessor = artifact["preprocessor"]
        version = artifact.get("version", "v1.0.0")

        # 1. Feature Engineering
        fe_df = self.feature_engineer.transform(df)

        # 2. Transform Features
        X = preprocessor.transform(fe_df)
        feature_names = preprocessor.get_feature_names()

        # 3. Predict Anomaly Scores
        scores = model.predict_anomaly_scores(X)

        # 4. Generate Predictions & Explanations
        result_df = df.copy()
        
        # Ensure robust Transaction_ID and transaction_id exist
        if "Transaction_ID" not in result_df.columns and "transaction_id" not in result_df.columns:
            found_id = None
            for c in result_df.columns:
                if c.lower() in ["transactionid", "tx_id", "id", "trans_num"]:
                    found_id = c
                    break
            if found_id:
                result_df["Transaction_ID"] = result_df[found_id].astype(str)
                result_df["transaction_id"] = result_df[found_id].astype(str)
            else:
                generated_ids = [f"TX-{100000 + idx}" for idx in range(len(result_df))]
                result_df["Transaction_ID"] = generated_ids
                result_df["transaction_id"] = generated_ids
        elif "Transaction_ID" in result_df.columns and "transaction_id" not in result_df.columns:
            result_df["transaction_id"] = result_df["Transaction_ID"].astype(str)
        elif "transaction_id" in result_df.columns and "Transaction_ID" not in result_df.columns:
            result_df["Transaction_ID"] = result_df["transaction_id"].astype(str)

        # Ensure robust Amount and amount exist
        if "Amount" not in result_df.columns and "amount" not in result_df.columns:
            for c in result_df.columns:
                if c.lower() == "amount":
                    result_df["Amount"] = result_df[c]
                    result_df["amount"] = result_df[c]
                    break
            if "Amount" not in result_df.columns:
                result_df["Amount"] = 0.0
                result_df["amount"] = 0.0
        elif "Amount" in result_df.columns and "amount" not in result_df.columns:
            result_df["amount"] = result_df["Amount"]
        elif "amount" in result_df.columns and "Amount" not in result_df.columns:
            result_df["Amount"] = result_df["amount"]

        result_df["anomaly_score"] = scores

        labels = []
        all_reasons = []

        for i, score in enumerate(scores):
            if score >= SUSPICIOUS_THRESHOLD:
                labels.append(LABEL_FRAUD)
            elif score >= NORMAL_THRESHOLD:
                labels.append(LABEL_SUSPICIOUS)
            else:
                labels.append(LABEL_NORMAL)

            # Generate reason codes for flagged transactions
            row_reasons = self._generate_domain_reasons(fe_df.iloc[i], score)
            
            # Model-level feature attributions
            if score >= NORMAL_THRESHOLD:
                model_attr = model.explain_instance(X[i], feature_names, top_k=3)
                for attr in model_attr:
                    feat_desc = f"{attr['feature']} ({attr['direction']})"
                    if feat_desc not in row_reasons:
                        row_reasons.append(feat_desc)

            all_reasons.append(row_reasons)

        result_df["prediction_label"] = labels
        result_df["risk_factors"] = all_reasons
        result_df["model_name"] = model.name
        result_df["model_version"] = version

        # 5. Log to DB
        if log_to_db:
            try:
                self.db_manager.log_transactions(result_df)
            except Exception as e:
                print(f"Warning: Failed to log transactions to DB: {e}")

        return result_df

    def score_single_transaction(self, txn_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Scores a single incoming transaction dictionary."""
        df = pd.DataFrame([txn_dict])
        scored_df = self.score_transactions(df, log_to_db=True)
        return scored_df.iloc[0].to_dict()

    def _generate_domain_reasons(self, row: pd.Series, score: float) -> List[str]:
        """Generates clear, banking-grade domain risk factors."""
        reasons = []
        amt = float(row.get("Amount", 0.0))
        hour = int(row.get("Hour", 12))
        vel = float(row.get("Transaction_Velocity_24h", 1.0))
        dist = float(row.get("Distance_From_Home_KM", 0.0))
        cat = str(row.get("Merchant_Category", "")).lower()

        if amt > 1000.0:
            reasons.append(f"High transaction amount (${amt:,.2f})")
        if hour in [0, 1, 2, 3, 4, 23]:
            reasons.append(f"Unusual transaction time ({hour:02d}:00 hours)")
        if vel > 8.0:
            reasons.append(f"Elevated velocity burst ({int(vel)} transactions in 24h)")
        if dist > 80.0:
            reasons.append(f"Significant distance from registered address ({dist:.1f} km)")
        if cat in ["crypto_exchange", "luxury_goods", "online_gaming"]:
            reasons.append(f"Elevated merchant category risk ({cat.replace('_', ' ').title()})")

        if not reasons and score >= NORMAL_THRESHOLD:
            reasons.append("Unusual multidimensional latent behavioral deviation")

        return reasons
