"""
Feature Engineering module for Credit Card Fraud Detection.
Derives domain-specific behavioral, temporal, velocity, and statistical features.
"""
from typing import List, Optional
import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        self.merchant_risk_map = {
            "luxury_goods": 0.85,
            "crypto_exchange": 0.90,
            "online_gaming": 0.75,
            "electronics": 0.65,
            "jewelry": 0.70,
            "travel_airline": 0.50,
            "supermarket": 0.15,
            "grocery": 0.10,
            "fuel_gas": 0.20,
            "restaurant": 0.25,
            "pharmacy": 0.15,
            "entertainment": 0.35,
            "other": 0.30
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features on training or batch data."""
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes engineered features without mutating input."""
        engineered = df.copy()

        # 1. Temporal & Cyclical Features
        if "Timestamp" in engineered.columns:
            ts = pd.to_datetime(engineered["Timestamp"], errors="coerce")
            if "Hour" not in engineered.columns:
                engineered["Hour"] = ts.dt.hour.fillna(12).astype(int)
            if "DayOfWeek" not in engineered.columns:
                engineered["DayOfWeek"] = ts.dt.dayofweek.fillna(2).astype(int)
        elif "Time" in engineered.columns:
            # If Time is elapsed seconds (Kaggle dataset style)
            if "Hour" not in engineered.columns:
                engineered["Hour"] = ((engineered["Time"] / 3600) % 24).astype(int)
            if "DayOfWeek" not in engineered.columns:
                engineered["DayOfWeek"] = (((engineered["Time"] / 3600) / 24) % 7).astype(int)
        
        if "Hour" not in engineered.columns:
            engineered["Hour"] = 12
        if "DayOfWeek" not in engineered.columns:
            engineered["DayOfWeek"] = 2

        # Night time flag (11 PM - 5 AM)
        engineered["Is_Night"] = engineered["Hour"].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)
        engineered["Is_Weekend"] = engineered["DayOfWeek"].apply(lambda d: 1 if d in [5, 6] else 0)

        # Cyclical Hour encoding
        engineered["Hour_Sin"] = np.sin(2 * np.pi * engineered["Hour"] / 24.0)
        engineered["Hour_Cos"] = np.cos(2 * np.pi * engineered["Hour"] / 24.0)

        # 2. Amount Dynamics
        if "Amount" in engineered.columns:
            engineered["Log_Amount"] = np.log1p(np.maximum(0, engineered["Amount"]))
            mean_amt = engineered["Amount"].mean()
            std_amt = engineered["Amount"].std()
            if std_amt > 0:
                engineered["Amount_ZScore_Global"] = (engineered["Amount"] - mean_amt) / std_amt
            else:
                engineered["Amount_ZScore_Global"] = 0.0

            q95 = engineered["Amount"].quantile(0.95)
            engineered["Is_High_Amount"] = (engineered["Amount"] > q95).astype(int)

        # 3. User & Velocity Features (if present or default simulated)
        if "Transaction_Velocity_24h" not in engineered.columns:
            if "User_ID" in engineered.columns:
                # Count frequency per user
                user_counts = engineered.groupby("User_ID")["Amount"].transform("count")
                engineered["Transaction_Velocity_24h"] = np.clip(user_counts, 1, 30)
            else:
                # Normal baseline distribution
                engineered["Transaction_Velocity_24h"] = 2

        if "Amount_ZScore_User" not in engineered.columns:
            engineered["Amount_ZScore_User"] = engineered.get("Amount_ZScore_Global", 0.0)

        if "Distance_From_Home_KM" not in engineered.columns:
            engineered["Distance_From_Home_KM"] = 8.5

        # 4. Merchant Category Risk Score
        if "Merchant_Category" in engineered.columns:
            cat_lower = engineered["Merchant_Category"].astype(str).str.lower().str.strip()
            engineered["Merchant_Risk_Score"] = cat_lower.map(self.merchant_risk_map).fillna(0.30)
        else:
            engineered["Merchant_Risk_Score"] = 0.30

        # 5. Device & Channel Flags
        if "Transaction_Type" in engineered.columns:
            engineered["Is_Online"] = engineered["Transaction_Type"].astype(str).str.lower().apply(
                lambda x: 1 if "online" in x or "web" in x or "e-commerce" in x else 0
            )
        else:
            engineered["Is_Online"] = 1

        return engineered
