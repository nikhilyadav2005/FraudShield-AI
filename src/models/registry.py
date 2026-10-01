"""
Model Registry service.
Manages versioning, active model retrieval, switching, and artifact caching.
"""
from typing import Dict, Any, Optional, List
import joblib
from src.database.db_manager import DatabaseManager

class ModelRegistry:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self._cached_version = None
        self._cached_artifact = None

    def get_active_model_details(self) -> Optional[Dict[str, Any]]:
        return self.db_manager.get_active_model()

    def get_all_models(self) -> List[Dict[str, Any]]:
        return self.db_manager.get_all_models()

    def switch_active_model(self, version: str):
        self.db_manager.set_active_model(version)
        self._cached_version = None
        self._cached_artifact = None

    def load_active_artifact(self) -> Optional[Dict[str, Any]]:
        active_meta = self.get_active_model_details()
        if not active_meta:
            return None

        ver = active_meta["version"]
        if self._cached_version == ver and self._cached_artifact is not None:
            return self._cached_artifact

        path = active_meta["model_path"]
        try:
            artifact = joblib.load(path)
            self._cached_version = ver
            self._cached_artifact = artifact
            return artifact
        except Exception:
            return None
