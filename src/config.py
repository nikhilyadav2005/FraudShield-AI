"""
Configuration settings for the Credit Card Fraud Detection System.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models_storage"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_PATH = DATA_DIR / "fraud_system.db"

# Create directories if not exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Anomaly Detection Settings
DEFAULT_CONTAMINATION = 0.03  # 3% expected fraud rate
RANDOM_STATE = 42

# Score Thresholds for 3-tier classification (0.0 to 1.0)
NORMAL_THRESHOLD = 0.45
SUSPICIOUS_THRESHOLD = 0.70

# Categories
LABEL_NORMAL = "Normal"
LABEL_SUSPICIOUS = "Suspicious"
LABEL_FRAUD = "Potentially Fraudulent"

# Drift Monitoring Thresholds (Population Stability Index - PSI)
PSI_NO_DRIFT = 0.10
PSI_MODERATE_DRIFT = 0.25  # Warning threshold
PSI_SIGNIFICANT_DRIFT = 0.25  # Retraining trigger threshold
KS_P_VALUE_THRESHOLD = 0.05

# Supported Models
SUPPORTED_MODELS = [
    "Isolation Forest",
    "Local Outlier Factor (LOF)",
    "One-Class SVM",
    "Neural Autoencoder"
]

# Standard Feature Definitions
CORE_NUMERICAL_COLS = [
    "Amount",
    "Hour",
    "DayOfWeek",
    "Transaction_Velocity_24h",
    "Amount_ZScore_User",
    "Distance_From_Home_KM",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"
]

CORE_CATEGORICAL_COLS = [
    "Merchant_Category",
    "Transaction_Type",
    "Card_Type",
    "Device_Type"
]
