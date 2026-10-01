"""
Database manager for Credit Card Fraud Detection System.
Provides SQLite storage for transactions, predictions, model registry, and retraining events.
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from src.config import DATABASE_PATH

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DATABASE_PATH)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Scored Transactions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE,
                timestamp TEXT,
                amount REAL,
                merchant_category TEXT,
                transaction_type TEXT,
                card_type TEXT,
                device_type TEXT,
                anomaly_score REAL,
                prediction_label TEXT,
                model_name TEXT,
                model_version TEXT,
                risk_factors TEXT,
                is_ground_truth_fraud INTEGER DEFAULT NULL,
                created_at TEXT
            );
            """)

            # Model Registry Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                version TEXT UNIQUE,
                algorithm TEXT,
                contamination REAL,
                train_samples INTEGER,
                features TEXT,
                metrics TEXT,
                model_path TEXT,
                is_active INTEGER DEFAULT 0,
                created_at TEXT
            );
            """)

            # Retraining History & Trigger Logs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS retraining_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_reason TEXT,
                previous_version TEXT,
                new_version TEXT,
                dataset_size INTEGER,
                drift_metrics TEXT,
                status TEXT,
                notes TEXT,
                timestamp TEXT
            );
            """)

            # Data Drift Logs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_timestamp TEXT,
                model_version TEXT,
                dataset_name TEXT,
                sample_size INTEGER,
                drift_detected INTEGER,
                max_psi REAL,
                drift_summary TEXT
            );
            """)

            conn.commit()

    def log_transactions(self, df_records: pd.DataFrame):
        """Insert or update scored transactions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for _, row in df_records.iterrows():
                risk_factors = row.get("risk_factors", [])
                if isinstance(risk_factors, list):
                    risk_factors = json.dumps(risk_factors)
                elif not isinstance(risk_factors, str):
                    risk_factors = str(risk_factors)

                cursor.execute("""
                INSERT OR REPLACE INTO transactions (
                    transaction_id, timestamp, amount, merchant_category,
                    transaction_type, card_type, device_type, anomaly_score,
                    prediction_label, model_name, model_version, risk_factors,
                    is_ground_truth_fraud, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row.get("transaction_id", f"TX-{datetime.utcnow().timestamp()}")),
                    str(row.get("timestamp", datetime.utcnow().isoformat())),
                    float(row.get("amount", row.get("Amount", 0.0))),
                    str(row.get("merchant_category", row.get("Merchant_Category", "Unknown"))),
                    str(row.get("transaction_type", row.get("Transaction_Type", "Online"))),
                    str(row.get("card_type", row.get("Card_Type", "Credit"))),
                    str(row.get("device_type", row.get("Device_Type", "Mobile"))),
                    float(row.get("anomaly_score", 0.0)),
                    str(row.get("prediction_label", "Normal")),
                    str(row.get("model_name", "Default")),
                    str(row.get("model_version", "v1.0.0")),
                    risk_factors,
                    int(row["is_ground_truth_fraud"]) if "is_ground_truth_fraud" in row and pd.notnull(row["is_ground_truth_fraud"]) else (int(row["Class"]) if "Class" in row and pd.notnull(row["Class"]) else None),
                    datetime.utcnow().isoformat()
                ))
            conn.commit()

    def register_model(self, model_info: Dict[str, Any]):
        """Register a trained model version and set as active if specified."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if model_info.get("is_active", 0) == 1:
                cursor.execute("UPDATE model_registry SET is_active = 0")
            
            cursor.execute("""
            INSERT OR REPLACE INTO model_registry (
                model_name, version, algorithm, contamination,
                train_samples, features, metrics, model_path, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_info.get("model_name"),
                model_info.get("version"),
                model_info.get("algorithm"),
                float(model_info.get("contamination", 0.03)),
                int(model_info.get("train_samples", 0)),
                json.dumps(model_info.get("features", [])),
                json.dumps(model_info.get("metrics", {})),
                model_info.get("model_path"),
                1 if model_info.get("is_active", False) else 0,
                datetime.utcnow().isoformat()
            ))
            conn.commit()

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res["features"] = json.loads(res["features"]) if res["features"] else []
                res["metrics"] = json.loads(res["metrics"]) if res["metrics"] else {}
                return res
            # Fallback to most recent model
            cursor.execute("SELECT * FROM model_registry ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res["features"] = json.loads(res["features"]) if res["features"] else []
                res["metrics"] = json.loads(res["metrics"]) if res["metrics"] else {}
                return res
            return None

    def get_all_models(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry ORDER BY id DESC")
            rows = cursor.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                item["features"] = json.loads(item["features"]) if item["features"] else []
                item["metrics"] = json.loads(item["metrics"]) if item["metrics"] else {}
                result.append(item)
            return result

    def set_active_model(self, version: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE model_registry SET is_active = 0")
            cursor.execute("UPDATE model_registry SET is_active = 1 WHERE version = ?", (version,))
            conn.commit()

    def log_retraining_event(self, trigger_reason: str, prev_ver: str, new_ver: str, dataset_size: int, drift_metrics: Dict[str, Any], status: str = "SUCCESS", notes: str = ""):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO retraining_history (
                trigger_reason, previous_version, new_version, dataset_size,
                drift_metrics, status, notes, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trigger_reason, prev_ver, new_ver, dataset_size,
                json.dumps(drift_metrics), status, notes, datetime.utcnow().isoformat()
            ))
            conn.commit()

    def get_retraining_history(self) -> pd.DataFrame:
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM retraining_history ORDER BY id DESC", conn)
            return df

    def log_drift_check(self, model_version: str, dataset_name: str, sample_size: int, drift_detected: bool, max_psi: float, drift_summary: Dict[str, Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO drift_logs (
                check_timestamp, model_version, dataset_name, sample_size,
                drift_detected, max_psi, drift_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(), model_version, dataset_name,
                sample_size, 1 if drift_detected else 0, max_psi, json.dumps(drift_summary)
            ))
            conn.commit()

    def get_drift_logs(self, limit: int = 50) -> pd.DataFrame:
        with self.get_connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM drift_logs ORDER BY id DESC LIMIT {limit}", conn)
            return df

    def get_recent_transactions(self, limit: int = 1000) -> pd.DataFrame:
        with self.get_connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM transactions ORDER BY id DESC LIMIT {limit}", conn)
            return df

    def get_transaction_statistics(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(amount), SUM(CASE WHEN prediction_label != 'Normal' THEN 1 ELSE 0 END) FROM transactions")
            row = cursor.fetchone()
            total = row[0] or 0
            avg_amt = row[1] or 0.0
            flagged = row[2] or 0
            
            cursor.execute("SELECT SUM(CASE WHEN prediction_label = 'Potentially Fraudulent' THEN 1 ELSE 0 END) FROM transactions")
            fraud_count = cursor.fetchone()[0] or 0

            return {
                "total_transactions": total,
                "flagged_transactions": flagged,
                "fraud_transactions": fraud_count,
                "fraud_rate": (fraud_count / total * 100) if total > 0 else 0.0,
                "average_amount": avg_amt
            }
