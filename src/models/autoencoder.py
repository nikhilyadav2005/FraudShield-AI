"""
Neural Autoencoder Anomaly Detector.
Trains an encoder-decoder neural network to reconstruct normal transaction patterns.
High reconstruction error identifies anomalous/fraudulent transactions.
Provides PyTorch implementation with an automatic Scikit-learn MLP fallback.
"""
from typing import Optional, List, Dict, Any
import numpy as np
from src.models.base import BaseAnomalyDetector

# Check if PyTorch is installed
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.neural_network import MLPRegressor

if HAS_TORCH:
    class PyTorchAutoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int = 16):
            super().__init__()
            hidden_1 = max(latent_dim * 2, min(64, input_dim * 2))
            hidden_2 = max(latent_dim, hidden_1 // 2)

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_1),
                nn.BatchNorm1d(hidden_1),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_1, hidden_2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_2, latent_dim)
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_2, hidden_1),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_1, input_dim)
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed


class AutoencoderDetector(BaseAnomalyDetector):
    def __init__(
        self,
        contamination: float = 0.03,
        latent_dim: int = 16,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
        random_state: int = 42
    ):
        super().__init__(name="Neural Autoencoder", contamination=contamination)
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.pytorch_model = None
        self.sklearn_model = None
        self.input_dim = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoencoderDetector":
        self.input_dim = X.shape[1]
        
        # Check if PyTorch is available and compatible
        global HAS_TORCH
        if HAS_TORCH:
            try:
                self._fit_pytorch(X)
            except Exception:
                # If PyTorch fails runtime (e.g. backend issue), gracefully fallback to Sklearn MLP
                self._fit_sklearn(X)
        else:
            self._fit_sklearn(X)

        raw_scores = self.predict_raw_scores(X)
        self.min_score_ = float(np.percentile(raw_scores, 1))
        self.max_score_ = float(np.percentile(raw_scores, 99))
        self.is_fitted = True
        return self

    def _fit_pytorch(self, X: np.ndarray):
        torch.manual_seed(self.random_state)
        device = torch.device("cpu")  # CPU is robust and fast for tabular datasets

        self.pytorch_model = PyTorchAutoencoder(
            input_dim=self.input_dim,
            latent_dim=min(self.latent_dim, max(2, self.input_dim // 2))
        ).to(device)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.pytorch_model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.pytorch_model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                outputs = self.pytorch_model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

    def _fit_sklearn(self, X: np.ndarray):
        h1 = max(16, min(64, self.input_dim))
        h2 = max(8, self.latent_dim)
        self.sklearn_model = MLPRegressor(
            hidden_layer_sizes=(h1, h2, h1),
            activation="relu",
            solver="adam",
            max_iter=self.epochs * 2,
            random_state=self.random_state,
            early_stopping=True
        )
        self.sklearn_model.fit(X, X)

    def predict_raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes reconstruction error (MSE per sample)."""
        if self.pytorch_model is not None:
            self.pytorch_model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                recon = self.pytorch_model(tensor_x).cpu().numpy()
        elif self.sklearn_model is not None:
            recon = self.sklearn_model.predict(X)
        else:
            raise ValueError("Autoencoder has not been fitted.")

        # Mean Squared Reconstruction Error per row
        mse_errors = np.mean(np.square(X - recon), axis=1)
        return mse_errors

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Returns reconstructed feature matrix."""
        if self.pytorch_model is not None:
            self.pytorch_model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                return self.pytorch_model(tensor_x).cpu().numpy()
        elif self.sklearn_model is not None:
            return self.sklearn_model.predict(X)
        raise ValueError("Model not fitted.")

    def explain_instance(self, x: np.ndarray, feature_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Explain anomaly based on per-feature reconstruction error (x_j - x_hat_j)^2.
        Features that the neural network failed to reconstruct indicate the primary anomaly root causes.
        """
        x_2d = np.asarray(x).reshape(1, -1)
        recon_2d = self.reconstruct(x_2d)
        
        feature_errors = np.square(x_2d[0] - recon_2d[0])
        top_idx = np.argsort(feature_errors)[::-1][:top_k]

        explanations = []
        for idx in top_idx:
            feat = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            err = float(feature_errors[idx])
            val = float(x_2d[0, idx])
            recon_val = float(recon_2d[0, idx])
            explanations.append({
                "feature": feat,
                "value": round(val, 3),
                "reconstructed": round(recon_val, 3),
                "contribution": round(err, 3),
                "direction": f"High reconstruction error (actual {val:.2f} vs expected {recon_val:.2f})"
            })
        return explanations
