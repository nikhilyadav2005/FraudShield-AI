"""
Data validation module for credit card transaction datasets.
Performs data health checks, schema validation, missing value detection,
and duplicate identification.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

class DataValidator:
    def __init__(self, required_columns: list = None):
        self.required_columns = required_columns or ["Amount"]

    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any], pd.DataFrame]:
        """
        Validates the input DataFrame.
        Returns:
            is_valid (bool): Whether required columns exist
            report (dict): Validation summary
            clean_df (pd.DataFrame): DataFrame with duplicate handling and sanitization
        """
        report: Dict[str, Any] = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "column_names": list(df.columns),
            "missing_values": {},
            "total_missing_cells": 0,
            "duplicate_rows": 0,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "warnings": [],
            "errors": []
        }

        # 1. Check required columns
        missing_required = [col for col in self.required_columns if col not in df.columns]
        if missing_required:
            report["errors"].append(f"Missing mandatory column(s): {', '.join(missing_required)}")
            is_valid = False
        else:
            is_valid = True

        # 2. Check missing values
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                pct = round((count / len(df)) * 100, 2)
                report["missing_values"][col] = {"count": int(count), "percentage": pct}
        report["total_missing_cells"] = int(null_counts.sum())

        if report["total_missing_cells"] > 0:
            report["warnings"].append(
                f"Dataset contains {report['total_missing_cells']} missing values across {len(report['missing_values'])} columns."
            )

        # 3. Check duplicate rows
        duplicates = int(df.duplicated().sum())
        report["duplicate_rows"] = duplicates
        if duplicates > 0:
            report["warnings"].append(f"Found {duplicates} duplicate transaction rows.")

        # 4. Check Amount anomalies
        if "Amount" in df.columns:
            negative_amounts = int((df["Amount"] < 0).sum())
            if negative_amounts > 0:
                report["warnings"].append(f"Found {negative_amounts} negative transaction amounts.")
            zero_amounts = int((df["Amount"] == 0).sum())
            if zero_amounts > 0:
                report["warnings"].append(f"Found {zero_amounts} zero-dollar transactions.")

        # 5. Check Ground Truth label presence
        has_label = False
        label_col = None
        for candidate in ["Class", "is_fraud", "is_fraudulent", "label", "is_ground_truth_fraud"]:
            if candidate in df.columns:
                has_label = True
                label_col = candidate
                break

        report["has_ground_truth"] = has_label
        report["label_column"] = label_col
        if has_label:
            val_counts = df[label_col].value_counts().to_dict()
            report["label_distribution"] = {str(k): int(v) for k, v in val_counts.items()}
            fraud_count = val_counts.get(1, 0)
            report["fraud_rate_pct"] = round((fraud_count / len(df)) * 100, 3)

        # Create cleaned baseline
        clean_df = df.copy()
        if duplicates > 0:
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)

        return is_valid, report, clean_df
