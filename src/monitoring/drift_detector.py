"""
Data Drift and Distribution Shift Detector.
Implements Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test,
Wasserstein distance, and automated retraining triggers.
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.config import PSI_NO_DRIFT, PSI_MODERATE_DRIFT, PSI_SIGNIFICANT_DRIFT, KS_P_VALUE_THRESHOLD

class DriftDetector:
    def __init__(self, psi_threshold: float = PSI_SIGNIFICANT_DRIFT, ks_p_threshold: float = KS_P_VALUE_THRESHOLD):
        self.psi_threshold = psi_threshold
        self.ks_p_threshold = ks_p_threshold

    @staticmethod
    def calculate_psi(reference: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) between reference and target distributions.
        """
        ref = reference[~np.isnan(reference)]
        tgt = target[~np.isnan(target)]
        if len(ref) < 10 or len(tgt) < 10:
            return 0.0

        # Binning based on reference percentiles
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref, quantiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        # Adjust outer edges
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(ref, bins=bins)
        tgt_counts, _ = np.histogram(tgt, bins=bins)

        # Proportions with small epsilon to avoid zero division / log(0)
        eps = 1e-4
        ref_pct = (ref_counts / len(ref)) + eps
        tgt_pct = (tgt_counts / len(tgt)) + eps

        # Normalize
        ref_pct = ref_pct / np.sum(ref_pct)
        tgt_pct = tgt_pct / np.sum(tgt_pct)

        psi_val = np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct))
        return float(np.round(psi_val, 4))

    def detect_feature_drift(
        self,
        ref_df: pd.DataFrame,
        target_df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes drift across numerical and categorical features between reference and target datasets.
        """
        if features is None:
            features = [c for c in ref_df.columns if c in target_df.columns and c not in ["Transaction_ID", "transaction_id", "timestamp", "Timestamp"]]

        feature_reports = {}
        max_psi = 0.0
        drifted_features_count = 0

        for col in features:
            if col not in ref_df.columns or col not in target_df.columns:
                continue

            ref_series = ref_df[col]
            tgt_series = target_df[col]

            if pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(tgt_series):
                ref_vals = ref_series.dropna().values
                tgt_vals = tgt_series.dropna().values

                if len(ref_vals) == 0 or len(tgt_vals) == 0:
                    continue

                psi = self.calculate_psi(ref_vals, tgt_vals)
                max_psi = max(max_psi, psi)

                # Kolmogorov-Smirnov 2-sample test
                ks_stat, ks_p = stats.ks_2samp(ref_vals, tgt_vals)

                # Wasserstein distance
                try:
                    w_dist = stats.wasserstein_distance(ref_vals, tgt_vals)
                except Exception:
                    w_dist = 0.0

                is_drifted = (psi >= self.psi_threshold) or (ks_p < self.ks_p_threshold and ks_stat > 0.15)
                if is_drifted:
                    drifted_features_count += 1

                feature_reports[col] = {
                    "type": "numeric",
                    "psi": psi,
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_p_value": round(float(ks_p), 4),
                    "wasserstein_distance": round(float(w_dist), 4),
                    "ref_mean": round(float(np.mean(ref_vals)), 2),
                    "tgt_mean": round(float(np.mean(tgt_vals)), 2),
                    "is_drifted": is_drifted,
                    "severity": "CRITICAL" if psi >= PSI_SIGNIFICANT_DRIFT else ("MODERATE" if psi >= PSI_MODERATE_DRIFT else "STABLE")
                }
            else:
                # Categorical Total Variation Distance
                ref_dist = ref_series.value_counts(normalize=True)
                tgt_dist = tgt_series.value_counts(normalize=True)
                all_cats = list(set(ref_dist.index).union(set(tgt_dist.index)))
                
                tvd = 0.5 * sum(abs(ref_dist.get(c, 0.0) - tgt_dist.get(c, 0.0)) for c in all_cats)
                is_drifted = tvd > 0.20
                if is_drifted:
                    drifted_features_count += 1

                feature_reports[col] = {
                    "type": "categorical",
                    "total_variation_distance": round(float(tvd), 4),
                    "is_drifted": is_drifted,
                    "severity": "CRITICAL" if tvd > 0.35 else ("MODERATE" if tvd > 0.20 else "STABLE")
                }

        # Overall System Drift Assessment
        total_feats = len(feature_reports)
        drift_ratio = (drifted_features_count / total_feats) if total_feats > 0 else 0.0

        if max_psi >= PSI_SIGNIFICANT_DRIFT or drift_ratio >= 0.30:
            status = "RETRAINING_TRIGGERED"
            recommendation = "Significant data/concept drift detected. Automatic retraining recommended immediately."
            drift_detected = True
        elif max_psi >= PSI_MODERATE_DRIFT or drift_ratio >= 0.15:
            status = "WARNING_MONITOR"
            recommendation = "Moderate distribution shift observed. Monitor model metrics closely."
            drift_detected = True
        else:
            status = "STABLE"
            recommendation = "Feature distributions are stable. No retraining required."
            drift_detected = False

        return {
            "status": status,
            "drift_detected": drift_detected,
            "max_psi": round(max_psi, 4),
            "drifted_features_count": drifted_features_count,
            "total_features_evaluated": total_feats,
            "drift_ratio_pct": round(drift_ratio * 100, 1),
            "recommendation": recommendation,
            "feature_details": feature_reports
        }
