"""
Realistic Credit Card Fraud Transaction Dataset Generator.
Synthesizes realistic credit card transactions incorporating Kaggle-like PCA components (V1-V8)
and real-world transaction features (Merchant, Channel, Device, Distance, Time, Velocity).
Supports generating baseline training datasets and drifted incoming batches for continuous learning.
"""
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

class TransactionDataGenerator:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

        self.merchant_categories = [
            "supermarket", "grocery", "restaurant", "fuel_gas", "pharmacy",
            "entertainment", "electronics", "travel_airline", "luxury_goods", "crypto_exchange"
        ]
        self.transaction_types = ["POS_Swipe", "POS_Chip", "Online_Ecom", "Contactless_NFC", "ATM_Withdrawal"]
        self.card_types = ["Visa", "MasterCard", "Amex", "Discover"]
        self.device_types = ["Mobile_App", "Desktop_Web", "POS_Terminal", "ATM"]
        self.countries = ["US", "US", "US", "US", "CA", "GB", "DE", "FR", "CN", "RU"]

    def generate_dataset(
        self,
        n_samples: int = 4000,
        fraud_ratio: float = 0.03,
        start_date: Optional[datetime] = None,
        is_drifted: bool = False
    ) -> pd.DataFrame:
        """
        Generates synthetic transaction dataset with realistic normal and fraud distributions.
        If is_drifted is True, alters distributions (concept/data drift) to simulate production drift.
        """
        rng = np.random.RandomState(self.random_state if not is_drifted else self.random_state + 100)
        start_dt = start_date or (datetime.utcnow() - timedelta(days=30))

        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud

        # --- NORMAL TRANSACTIONS ---
        if not is_drifted:
            # Baseline normal amounts: lognormal, mean ~ $65, std ~ $45
            normal_amounts = rng.lognormal(mean=3.8, sigma=0.8, size=n_normal)
            normal_amounts = np.clip(normal_amounts, 2.50, 1200.0)
            
            # Normal hours: peak during 8am - 10pm
            normal_hours = np.mod(np.round(rng.normal(loc=14.5, scale=4.0, size=n_normal)).astype(int), 24)
            normal_velocity = np.clip(rng.poisson(lam=2.5, size=n_normal), 1, 10)
            normal_distance = rng.exponential(scale=7.5, size=n_normal)  # mostly near home
        else:
            # Drifted normal: general inflation, slight shift in spending pattern
            normal_amounts = rng.lognormal(mean=4.2, sigma=0.9, size=n_normal)
            normal_amounts = np.clip(normal_amounts, 3.0, 1800.0)
            normal_hours = np.mod(np.round(rng.normal(loc=15.0, scale=4.5, size=n_normal)).astype(int), 24)
            normal_velocity = np.clip(rng.poisson(lam=3.8, size=n_normal), 1, 15)
            normal_distance = rng.exponential(scale=12.0, size=n_normal)

        # Baseline PCA features for normal transactions (standard normal)
        normal_v = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 8))

        # --- FRAUDULENT TRANSACTIONS ---
        if not is_drifted:
            # Typical fraud: higher amounts or micro-testing, odd hours (midnight), high velocity
            fraud_type_split = rng.rand(n_fraud)
            # 60% high value fraud, 40% micro-testing
            fraud_amounts = np.where(
                fraud_type_split > 0.4,
                rng.uniform(low=450.0, high=4800.0, size=n_fraud),
                rng.uniform(low=1.0, high=15.0, size=n_fraud)
            )
            # Odd hours: between 00:00 and 05:00
            fraud_hours = rng.choice([0, 1, 2, 3, 4, 22, 23], size=n_fraud)
            fraud_velocity = np.clip(rng.poisson(lam=9.0, size=n_fraud), 4, 30)
            fraud_distance = rng.exponential(scale=150.0, size=n_fraud) + 40.0
            
            # PCA features for fraud (shifted significantly, e.g. V2 positive, V4 high, V14 very negative)
            fraud_v = rng.normal(loc=0.0, scale=1.2, size=(n_fraud, 8))
            fraud_v[:, 0] -= 2.8  # V1 shift
            fraud_v[:, 1] += 3.1  # V2 shift
            fraud_v[:, 3] += 3.5  # V4 high
            fraud_v[:, 6] -= 4.2  # V7 negative
        else:
            # DRIFTED FRAUD: New fraud vectors (e.g. mid-day velocity fraud, crypto bots, synthetic IDs)
            fraud_amounts = rng.uniform(low=850.0, high=7500.0, size=n_fraud)
            fraud_hours = rng.choice(list(range(24)), size=n_fraud) # distributed throughout day
            fraud_velocity = np.clip(rng.poisson(lam=14.0, size=n_fraud), 8, 45)
            fraud_distance = rng.exponential(scale=350.0, size=n_fraud) + 120.0
            
            fraud_v = rng.normal(loc=0.0, scale=1.5, size=(n_fraud, 8))
            fraud_v[:, 2] -= 4.5  # V3 shift
            fraud_v[:, 4] += 5.0  # V5 shift
            fraud_v[:, 7] += 3.8  # V8 shift

        # Combine
        amounts = np.concatenate([normal_amounts, fraud_amounts])
        hours = np.concatenate([normal_hours, fraud_hours])
        velocities = np.concatenate([normal_velocity, fraud_velocity])
        distances = np.concatenate([normal_distance, fraud_distance])
        v_features = np.vstack([normal_v, fraud_v])
        labels = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

        # Shuffle indices
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        amounts = np.round(amounts[indices], 2)
        hours = hours[indices]
        velocities = velocities[indices]
        distances = np.round(distances[indices], 1)
        v_features = np.round(v_features[indices], 4)
        labels = labels[indices]

        # Generate timestamps
        timestamps = [
            (start_dt + timedelta(minutes=int(i * 12 + rng.randint(-5, 5)))).replace(hour=int(hours[i]))
            for i in range(n_samples)
        ]

        # Categorical distributions
        if not is_drifted:
            p_merch_normal = [0.30, 0.20, 0.20, 0.12, 0.08, 0.04, 0.03, 0.02, 0.005, 0.005]
            p_merch_fraud = [0.02, 0.02, 0.05, 0.04, 0.02, 0.05, 0.30, 0.20, 0.18, 0.12]
        else:
            # Drifted merchants: massive spike in crypto exchange and luxury goods
            p_merch_normal = [0.25, 0.18, 0.18, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03]
            p_merch_fraud = [0.01, 0.01, 0.02, 0.02, 0.01, 0.03, 0.20, 0.10, 0.35, 0.25]

        merchants = []
        txn_types = []
        for i in range(n_samples):
            is_f = labels[i] == 1
            p_m = p_merch_fraud if is_f else p_merch_normal
            merchants.append(rng.choice(self.merchant_categories, p=p_m))
            if is_f:
                txn_types.append(rng.choice(["Online_Ecom", "POS_Swipe", "ATM_Withdrawal"], p=[0.75, 0.15, 0.10]))
            else:
                txn_types.append(rng.choice(self.transaction_types, p=[0.35, 0.35, 0.20, 0.08, 0.02]))

        card_sample = rng.choice(self.card_types, size=n_samples, p=[0.50, 0.30, 0.15, 0.05])
        device_sample = rng.choice(self.device_types, size=n_samples, p=[0.45, 0.30, 0.20, 0.05])
        country_sample = rng.choice(self.countries, size=n_samples)

        # Assemble DataFrame
        df = pd.DataFrame({
            "Transaction_ID": [f"TX-{100000 + i}" for i in range(n_samples)],
            "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
            "Amount": amounts,
            "Hour": hours,
            "DayOfWeek": [t.weekday() for t in timestamps],
            "Merchant_Category": merchants,
            "Transaction_Type": txn_types,
            "Card_Type": card_sample,
            "Device_Type": device_sample,
            "Country": country_sample,
            "Distance_From_Home_KM": distances,
            "Transaction_Velocity_24h": velocities,
            "Amount_ZScore_User": np.round((amounts - np.median(amounts)) / (np.std(amounts) + 1e-6), 2),
            "V1": v_features[:, 0],
            "V2": v_features[:, 1],
            "V3": v_features[:, 2],
            "V4": v_features[:, 3],
            "V5": v_features[:, 4],
            "V6": v_features[:, 5],
            "V7": v_features[:, 6],
            "V8": v_features[:, 7],
            "Class": labels
        })

        return df
