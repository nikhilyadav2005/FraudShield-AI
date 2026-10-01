import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.config import DATA_DIR, RAW_DATA_DIR
from src.data.generator import TransactionDataGenerator
from src.data.validator import DataValidator
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.database.db_manager import DatabaseManager
from src.prediction.inference_service import InferenceService

def initialize():
    print("1. Generating baseline and drifted sample transaction datasets...")
    generator = TransactionDataGenerator(random_state=42)
    
    # Baseline dataset
    baseline_df = generator.generate_dataset(n_samples=4000, fraud_ratio=0.03, is_drifted=False)
    baseline_path = DATA_DIR / "sample_transactions.csv"
    baseline_df.to_csv(baseline_path, index=False)
    print(f"   Saved baseline dataset ({len(baseline_df)} rows) to {baseline_path}")

    # Drifted incoming batch
    drifted_df = generator.generate_dataset(n_samples=1000, fraud_ratio=0.06, is_drifted=True)
    drifted_path = DATA_DIR / "incoming_batch_drift.csv"
    drifted_df.to_csv(drifted_path, index=False)
    print(f"   Saved drifted incoming batch ({len(drifted_df)} rows) to {drifted_path}")

    print("2. Validating baseline data...")
    validator = DataValidator()
    is_valid, report, clean_df = validator.validate(baseline_df)
    print(f"   Validation status: {'Valid' if is_valid else 'Invalid'}, Rows: {report['total_rows']}")

    print("3. Engineering features...")
    fe = FeatureEngineer()
    fe_df = fe.transform(clean_df)

    print("4. Preprocessing features...")
    preprocessor = DataPreprocessor()
    X = preprocessor.fit_transform(fe_df)
    y = clean_df["Class"].values if "Class" in clean_df.columns else None

    print("5. Training and evaluating anomaly detection models...")
    db_mgr = DatabaseManager()
    trainer = ModelTrainer(db_mgr)
    trained_models, comparison_df, best_model_name = trainer.train_and_compare_all(
        X_train=X,
        y_train=y,
        contamination=0.03,
        preprocessor=preprocessor
    )
    print("   Model Benchmark Results:")
    print(comparison_df[["model_name", "anomaly_rate_pct", "precision", "recall", "f1_score", "pr_auc", "roc_auc"]])

    best_model = trained_models[best_model_name]
    best_metrics = comparison_df[comparison_df["model_name"] == best_model_name].iloc[0].to_dict()

    print(f"6. Registering best model: {best_model_name} (v1.0.0)...")
    trainer.save_model_artifact(
        model=best_model,
        preprocessor=preprocessor,
        version="v1.0.0",
        metrics=best_metrics,
        train_samples=len(clean_df),
        set_active=True
    )

    print("7. Priming database with initial scored transactions...")
    inference = InferenceService(db_mgr)
    inference.score_transactions(clean_df.head(600), log_to_db=True)
    print("   Database primed successfully!")

if __name__ == "__main__":
    initialize()
