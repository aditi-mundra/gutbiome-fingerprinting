"""
ml/config.py
============
Central configuration for the Microbiome Atlas ML pipeline.
All hyperparameters, file paths, column definitions, and thresholds
live here so every module imports from a single source of truth.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Project root (two levels up from this file)
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED: int = 42

# ─────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────
DATA_RAW_DIR        = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS_DIR    = PROJECT_ROOT / "data" / "results"
MODELS_DIR          = PROJECT_ROOT / "models"
DOCS_FIGURES_DIR    = PROJECT_ROOT / "docs" / "figures"

ABUNDANCE_CSV   = DATA_RAW_DIR / "abundance_stoolsubset.csv"
MARKERS_CSV     = DATA_RAW_DIR / "markers2clades_DB.csv"

# ─────────────────────────────────────────────
# Processed data artifacts
# ─────────────────────────────────────────────
FEATURES_NORMALIZED_NPY     = DATA_PROCESSED_DIR / "features_normalized.npy"
FEATURE_COLUMNS_TXT         = DATA_PROCESSED_DIR / "feature_columns.txt"
METADATA_CSV                = DATA_PROCESSED_DIR / "metadata.csv"
UMAP_EMBEDDINGS_NPY         = DATA_PROCESSED_DIR / "umap_embeddings.npy"
PCA_EMBEDDINGS_NPY          = DATA_PROCESSED_DIR / "pca_embeddings.npy"

# ─────────────────────────────────────────────
# Result artifacts
# ─────────────────────────────────────────────
EMBEDDINGS_CSV          = DATA_RESULTS_DIR / "embeddings.csv"
CLUSTER_ASSIGNMENTS_CSV = DATA_RESULTS_DIR / "cluster_assignments.csv"
FINGERPRINTS_CSV        = DATA_RESULTS_DIR / "fingerprints.csv"
EVALUATION_RESULTS_CSV  = DATA_RESULTS_DIR / "evaluation_results.csv"
CLUSTER_SUMMARY_JSON    = DATA_RESULTS_DIR / "cluster_summary.json"
SIMILARITY_NETWORK_JSON = DATA_RESULTS_DIR / "similarity_network.json"
FEATURE_IMPORTANCE_CSV  = DATA_RESULTS_DIR / "feature_importance.csv"

# ─────────────────────────────────────────────
# Model artifacts
# ─────────────────────────────────────────────
BEST_MODEL_JOBLIB   = MODELS_DIR / "best_model.joblib"
SCALER_JOBLIB       = MODELS_DIR / "scaler.joblib"
PCA_JOBLIB          = MODELS_DIR / "pca.joblib"
UMAP_JOBLIB         = MODELS_DIR / "umap.joblib"
KMEANS_JOBLIB       = MODELS_DIR / "kmeans.joblib"
GMM_JOBLIB          = MODELS_DIR / "gmm.joblib"
HDBSCAN_JOBLIB      = MODELS_DIR / "hdbscan.joblib"
BEST_PIPELINE_TXT   = MODELS_DIR / "best_pipeline.txt"

# ─────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────
FEATURE_PREFIX: str = "k__"

# Core metadata columns used by EDA, fingerprinting, and cluster characterisation
CORE_META_COLS: list[str] = [
    "sampleID",
    "subjectID",
    "disease",
    "age",
    "gender",
    "country",
    "bmi",
    "bodysite",
    "dataset_name",
]

# Sentinel strings that represent "not determined / missing" across all metadata fields
NULL_SENTINELS: list[str] = ["nd", "na", "-", " -", "n/a", "N/A", "", "none", "None"]

# Canonical gender mapping  →  None means unknown
GENDER_MAP: dict[str, str | None] = {
    "female": "female",
    "male":   "male",
    "-":      None,
    " -":     None,
    "nd":     None,
    "na":     None,
}

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
PREPROCESSING = {
    # Minimum non-zero prevalence for a feature to be kept (0–1)
    # Features present in fewer than this fraction of samples are dropped
    "min_prevalence": 0.01,
    # Scaler: "clr"  (centered log-ratio, best for compositional data)
    #         "standard"  (Z-score, simpler baseline)
    "scaler": "clr",
}

# ─────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────
PCA_PARAMS: dict = {
    "n_components": 50,
    "random_state": RANDOM_SEED,
    "whiten": False,
}

UMAP_PARAMS: dict = {
    "n_components": 2,
    "n_neighbors": 30,
    "min_dist": 0.1,
    "metric": "cosine",       # cosine works well for sparse compositional data
    "random_state": RANDOM_SEED,
    "low_memory": False,
}

# How many PCA components to feed into UMAP
PCA_FOR_UMAP_N_COMPONENTS: int = 50

# ─────────────────────────────────────────────
# Clustering — Pipeline A: PCA → KMeans
# ─────────────────────────────────────────────
KMEANS_PARAMS: dict = {
    "n_clusters": 8,
    "n_init": 20,
    "max_iter": 500,
    "random_state": RANDOM_SEED,
}

# ─────────────────────────────────────────────
# Clustering — Pipeline B: PCA → GMM
# ─────────────────────────────────────────────
GMM_PARAMS: dict = {
    "n_components": 8,
    "covariance_type": "full",
    "max_iter": 200,
    "n_init": 5,
    "random_state": RANDOM_SEED,
}

# ─────────────────────────────────────────────
# Clustering — Pipeline C: UMAP → HDBSCAN
# ─────────────────────────────────────────────
HDBSCAN_PARAMS: dict = {
    "min_cluster_size": 30,
    "min_samples": 10,
    "cluster_selection_epsilon": 0.0,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
EVALUATION = {
    # Silhouette is positively oriented (higher = better)
    # Davies-Bouldin is negatively oriented (lower = better)
    # Combined rank: rank_silhouette ASC + rank_db ASC → lowest total = winner
    "metrics": ["silhouette", "davies_bouldin"],
}

# Pipeline identifiers used as keys throughout the codebase
PIPELINE_IDS: list[str] = ["pca_kmeans", "pca_gmm", "umap_hdbscan"]

# ─────────────────────────────────────────────
# Fingerprinting
# ─────────────────────────────────────────────
FINGERPRINT = {
    # How many dominant microbes to report per sample
    "top_n_microbes": 10,
    # Minimum relative abundance to be considered "dominant" (0–1)
    "min_abundance_threshold": 0.001,
    # Shannon diversity thresholds
    "diversity_low_threshold":  1.5,
    "diversity_high_threshold": 3.0,
}

# ─────────────────────────────────────────────
# Novelty / uniqueness scoring
# ─────────────────────────────────────────────
NOVELTY = {
    # Percentile boundaries for novelty classification
    # Score = distance from cluster centroid, normalised 0-1
    "common_percentile":    33.0,   # below this → "Common"
    "moderate_percentile":  66.0,   # below this → "Moderately Unique"
    # above moderate_percentile → "Highly Unique"
}

# ─────────────────────────────────────────────
# Similarity engine
# ─────────────────────────────────────────────
SIMILARITY = {
    # Number of nearest neighbours to find per sample
    "n_neighbors": 10,
    # Cosine similarity threshold for drawing an edge in the network
    # Edge weight = cosine_similarity; edge is added if >= threshold
    "network_threshold": 0.90,
    # sklearn NearestNeighbors metric
    "metric": "cosine",
    # Maximum nodes to include in the exported network JSON
    # (subsample for Plotly performance)
    "max_network_nodes": 500,
}

# ─────────────────────────────────────────────
# Explainability (Random Forest on cluster labels)
# ─────────────────────────────────────────────
EXPLAINABILITY = {
    "n_estimators": 200,
    "max_depth": 8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    # How many top features to report per cluster in the JSON summary
    "top_features_per_cluster": 10,
    # How many top global features to save in feature_importance.csv
    "top_global_features": 50,
}

# ─────────────────────────────────────────────
# EDA figure settings
# ─────────────────────────────────────────────
EDA = {
    "figure_dpi": 150,
    "figure_format": "png",
    # How many top microbes to show in the abundance bar chart
    "top_microbes_n": 20,
    # How many most-variable features to show
    "top_variable_n": 20,
}

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("ATLAS_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def ensure_dirs() -> None:
    """Create all required output directories if they do not exist."""
    for directory in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_RESULTS_DIR,
        MODELS_DIR,
        DOCS_FIGURES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)