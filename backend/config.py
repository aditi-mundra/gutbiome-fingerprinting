"""
backend/config.py
=================
Centralised application settings loaded from the .env file via
pydantic-settings. Every module imports `settings` from here —
never reads os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./gutbiome.db"

    # ── FastAPI server ────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # ── ML artifact directories (relative to project root) ───
    MODELS_DIR: str = "models"
    RESULTS_DIR: str = "data/results"
    PROCESSED_DIR: str = "data/processed"

    # Use modern Pydantic v2 settings configuration
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Derived properties ────────────────────────────────────

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse the comma-separated CORS_ORIGINS string into a list."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root (parent of backend/)."""
        return Path(__file__).resolve().parent.parent

    @property
    def models_path(self) -> Path:
        return self.project_root / self.MODELS_DIR

    @property
    def results_path(self) -> Path:
        return self.project_root / self.RESULTS_DIR

    @property
    def processed_path(self) -> Path:
        return self.project_root / self.PROCESSED_DIR

    # ── ML artifact file paths ────────────────────────────────

    @property
    def scaler_path(self) -> Path:
        return self.models_path / "scaler.joblib"

    @property
    def best_model_path(self) -> Path:
        return self.models_path / "best_model.joblib"

    @property
    def pca_path(self) -> Path:
        return self.models_path / "pca.joblib"

    @property
    def umap_path(self) -> Path:
        return self.models_path / "umap.joblib"

    @property
    def best_pipeline_path(self) -> Path:
        return self.models_path / "best_pipeline.txt"

    @property
    def feature_columns_path(self) -> Path:
        return self.processed_path / "feature_columns.txt"

    @property
    def umap_embeddings_path(self) -> Path:
        return self.processed_path / "umap_embeddings.npy"

    @property
    def fingerprints_csv(self) -> Path:
        return self.results_path / "fingerprints.csv"

    @property
    def cluster_summary_json(self) -> Path:
        return self.results_path / "cluster_summary.json"

    @property
    def similarity_network_json(self) -> Path:
        return self.results_path / "similarity_network.json"

    @property
    def embeddings_csv(self) -> Path:
        return self.results_path / "embeddings.csv"

    @property
    def feature_importance_csv(self) -> Path:
        return self.results_path / "feature_importance.csv"

    @property
    def metadata_csv(self) -> Path:
        return self.processed_path / "metadata.csv"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.
    """
    return Settings()


# Module-level singleton for direct import convenience
settings = get_settings()