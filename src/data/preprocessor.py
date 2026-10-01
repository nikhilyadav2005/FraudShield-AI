"""
Data Preprocessing pipeline for anomaly detection models.
Handles missing value imputation, robust scaling, categorical encoding,
and serialization of feature pipelines.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, numerical_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        self.fitted_feature_names: List[str] = []
        self.is_fitted = False

    def auto_detect_columns(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None):
        """Automatically categorizes DataFrame columns into numerical and categorical."""
        id_timestamp_patterns = {"id", "tx_id", "transaction_id", "timestamp", "time", "date", "class", "is_fraud", "is_fraudulent", "user_id"}
        user_excludes = set([c.lower() for c in (exclude_cols or [])])

        self.numerical_features = []
        self.categorical_features = []

        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in id_timestamp_patterns or col_lower in user_excludes:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                self.numerical_features.append(col)
            else:
                # Only include categorical if reasonable cardinality (< 50 unique values)
                if df[col].nunique() <= 50:
                    self.categorical_features.append(col)

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fits imputers, scaler, and encoder on dataset."""
        if not self.numerical_features and not self.categorical_features:
            self.auto_detect_columns(df)

        feature_names = []

        # Fit numerical
        if self.numerical_features:
            num_data = df[self.numerical_features].copy()
            imputed_num = self.num_imputer.fit_transform(num_data)
            self.scaler.fit(imputed_num)
            feature_names.extend(self.numerical_features)

        # Fit categorical
        if self.categorical_features:
            cat_data = df[self.categorical_features].astype(str)
            imputed_cat = self.cat_imputer.fit_transform(cat_data)
            self.encoder.fit(imputed_cat)
            cat_encoded_names = list(self.encoder.get_feature_names_out(self.categorical_features))
            feature_names.extend(cat_encoded_names)

        self.fitted_feature_names = feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms DataFrame into preprocessed numerical feature matrix."""
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before transforming.")

        matrices = []

        # Transform numerical
        if self.numerical_features:
            num_data = df.reindex(columns=self.numerical_features).copy()
            # If completely missing in new transaction, fill with zeros before imputation
            num_data = num_data.fillna(0.0)
            imputed_num = self.num_imputer.transform(num_data)
            scaled_num = self.scaler.transform(imputed_num)
            matrices.append(scaled_num)

        # Transform categorical
        if self.categorical_features:
            cat_data = df.reindex(columns=self.categorical_features).copy()
            cat_data = cat_data.fillna("Unknown").astype(str)
            imputed_cat = self.cat_imputer.transform(cat_data)
            encoded_cat = self.encoder.transform(imputed_cat)
            matrices.append(encoded_cat)

        if not matrices:
            raise ValueError("No features available for transformation.")

        return np.hstack(matrices)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        return self.fitted_feature_names
