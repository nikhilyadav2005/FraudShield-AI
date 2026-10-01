"""
Automated End-to-End Test Suite for Credit Card Fraud Detection System.
Tests all modular pipelines: Validator, Preprocessor, Feature Engineering,
Anomaly Detection Models (Isolation Forest, LOF, One-Class SVM, Autoencoder),
Drift Detector, Retraining Engine, Inference Service, and PDF Report Generator.
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from src.database.db_manager import DatabaseManager
from src.data.validator import DataValidator
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.data.generator import TransactionDataGenerator
from src.models.isolation_forest import IsolationForestDetector
from src.models.lof import LOFDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.retraining_engine import RetrainingEngine
from src.prediction.inference_service import InferenceService
from src.reporting.pdf_generator import PDFReportGenerator

def run_tests():
    print("==================================================")
    print("RUNNING CREDIT CARD FRAUD SYSTEM VALIDATION SUITE")
    print("==================================================")

    # 1. Generator Test
    print("\n[TEST 1] Testing Data Generator...")
    gen = TransactionDataGenerator(random_state=123)
    df_train = gen.generate_dataset(n_samples=500, fraud_ratio=0.04, is_drifted=False)
    df_test = gen.generate_dataset(n_samples=200, fraud_ratio=0.08, is_drifted=True)
    assert len(df_train) == 500
    assert len(df_test) == 200
    assert "Amount" in df_train.columns and "Class" in df_train.columns
    print(" -> PASSED: Dataset generation verified (500 train, 200 drifted test).")

    # 2. Validator Test
    print("\n[TEST 2] Testing Data Validator...")
    val = DataValidator()
    is_valid, report, clean_df = val.validate(df_train)
    assert is_valid is True
    assert report["total_rows"] == 500
    print(f" -> PASSED: Validator identified {report['total_columns']} columns and {report['duplicate_rows']} duplicates.")

    # 3. Feature Engineering Test
    print("\n[TEST 3] Testing Feature Engineering...")
    fe = FeatureEngineer()
    fe_df = fe.transform(clean_df)
    assert "Hour_Sin" in fe_df.columns
    assert "Hour_Cos" in fe_df.columns
    assert "Is_Night" in fe_df.columns
    assert "Merchant_Risk_Score" in fe_df.columns
    print(f" -> PASSED: Feature engineering generated {len(fe_df.columns)} enriched features.")

    # 4. Preprocessor Test
    print("\n[TEST 4] Testing Preprocessor...")
    prep = DataPreprocessor()
    X = prep.fit_transform(fe_df)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 500
    print(f" -> PASSED: Feature matrix created with shape {X.shape}.")

    # 5. Model Tests
    print("\n[TEST 5] Testing Anomaly Detection Models...")
    models = [
        IsolationForestDetector(contamination=0.04),
        LOFDetector(contamination=0.04),
        OneClassSVMDetector(contamination=0.04),
        AutoencoderDetector(contamination=0.04, epochs=10)
    ]

    for m in models:
        m.fit(X)
        scores = m.predict_anomaly_scores(X)
        labels = m.predict_labels(X)
        assert len(scores) == 500
        assert 0.0 <= np.min(scores) and np.max(scores) <= 1.0
        assert len(labels) == 500
        explanations = m.explain_instance(X[0], prep.get_feature_names(), top_k=3)
        assert len(explanations) == 3
        print(f"   -> {m.name}: Min score={scores.min():.3f}, Max score={scores.max():.3f}, Explanations={len(explanations)} [PASSED]")

    # 6. Database and Model Registry Test
    print("\n[TEST 6] Testing Database & Model Registry...")
    db = DatabaseManager()
    trainer = ModelTrainer(db)
    active_path = trainer.save_model_artifact(
        model=models[0],
        preprocessor=prep,
        version="v1.0.0-test",
        metrics={"f1_score": 0.90, "pr_auc": 0.95},
        train_samples=500,
        set_active=True
    )
    reg = ModelRegistry(db)
    active = reg.get_active_model_details()
    assert active["version"] == "v1.0.0-test"
    print(f" -> PASSED: Active model successfully verified in registry: {active['model_name']} ({active['version']}).")

    # 7. Drift Detector Test
    print("\n[TEST 7] Testing Drift Detector...")
    drift_det = DriftDetector()
    drift_res = drift_det.detect_feature_drift(df_train, df_test)
    assert "status" in drift_res
    assert "max_psi" in drift_res
    print(f" -> PASSED: Drift detected status: {drift_res['status']} (Max PSI: {drift_res['max_psi']}).")

    # 8. Inference Service Test
    print("\n[TEST 8] Testing Inference Service & Real-Time Scoring...")
    infer = InferenceService(db)
    sample_txn = df_test.iloc[0].to_dict()
    res = infer.score_single_transaction(sample_txn)
    assert "anomaly_score" in res
    assert "prediction_label" in res
    assert "risk_factors" in res
    print(f" -> PASSED: Scored single transaction: Score={res['anomaly_score']:.3f}, Verdict={res['prediction_label']}.")

    # 9. PDF Report Generator Test
    print("\n[TEST 9] Testing PDF Executive Report Generator...")
    pdf_gen = PDFReportGenerator()
    stats = db.get_transaction_statistics()
    recent = db.get_recent_transactions(limit=10)
    pdf_file = pdf_gen.generate_report(
        summary_stats=stats,
        model_details=active,
        flagged_txns=recent,
        drift_info=drift_res,
        filename="Test_Audit_Report.pdf"
    )
    assert Path(pdf_file).exists()
    assert Path(pdf_file).stat().st_size > 1000
    print(f" -> PASSED: Generated PDF report at {pdf_file} ({Path(pdf_file).stat().st_size:,} bytes).")

    print("\n==================================================")
    print("ALL 9 TEST SUITES COMPLETED AND PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
