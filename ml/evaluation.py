"""
ml/evaluation.py
================
Evaluates all clustering pipelines using:
  - Silhouette Score      (higher is better,  range: -1 to 1)
  - Davies-Bouldin Index  (lower is better,   range: 0 to ∞)

Selection strategy:
  - Rank each pipeline by both metrics independently
  - Sum the ranks (lower = better overall)
  - The pipeline with the lowest combined rank wins
  - Ties broken by Silhouette Score (higher wins)

Noise points (HDBSCAN label == -1) are excluded before scoring
so they do not artificially inflate separation metrics.

Outputs:
  data/results/evaluation_results.csv
  models/best_pipeline.txt
  models/best_model.joblib  (copy of the winning model)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

from ml.config import (
    BEST_MODEL_JOBLIB,
    BEST_PIPELINE_TXT,
    EVALUATION_RESULTS_CSV,
    GMM_JOBLIB,
    HDBSCAN_JOBLIB,
    KMEANS_JOBLIB,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    PIPELINE_IDS,
    ensure_dirs,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Maps pipeline id → its persisted model file
_PIPELINE_MODEL_MAP: dict[str, Path] = {
    "pca_kmeans":   KMEANS_JOBLIB,
    "pca_gmm":      GMM_JOBLIB,
    "umap_hdbscan": HDBSCAN_JOBLIB,
}


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def _score_pipeline(
    X: np.ndarray,
    labels: np.ndarray,
    pipeline_id: str,
) -> dict[str, Any]:
    """
    Compute evaluation metrics for a single pipeline.

    Parameters
    ----------
    X           : embedding used for this pipeline (PCA or UMAP coords)
    labels      : (n_samples,) cluster labels
    pipeline_id : human-readable pipeline name

    Returns
    -------
    dict with keys: pipeline, n_clusters, n_noise, silhouette, davies_bouldin
    """
    # Exclude noise points (HDBSCAN label -1)
    mask          = labels != -1
    X_valid       = X[mask]
    labels_valid  = labels[mask]

    n_clusters = len(np.unique(labels_valid))
    n_noise    = int((labels == -1).sum())

    # Guard: if noise mask removed all samples (e.g. HDBSCAN all-noise result)
    if X_valid.shape[0] == 0:
        logger.warning(
            "%s: all %d samples are noise (label=-1) — metrics set to NaN.",
            pipeline_id, len(labels),
        )
        return {
            "pipeline":        pipeline_id,
            "n_clusters":      0,
            "n_noise":         int((labels == -1).sum()),
            "silhouette":      np.nan,
            "davies_bouldin":  np.nan,
        }

    # Need at least 2 distinct cluster labels to score
    if n_clusters < 2:
        logger.warning(
            "%s: only %d cluster(s) found — metrics set to NaN.",
            pipeline_id, n_clusters,
        )
        return {
            "pipeline":        pipeline_id,
            "n_clusters":      n_clusters,
            "n_noise":         n_noise,
            "silhouette":      np.nan,
            "davies_bouldin":  np.nan,
        }

    sil = silhouette_score(X_valid, labels_valid, metric="euclidean", random_state=42)
    db  = davies_bouldin_score(X_valid, labels_valid)

    logger.info(
        "%-15s  clusters=%2d  noise=%4d  silhouette=%.4f  davies_bouldin=%.4f",
        pipeline_id, n_clusters, n_noise, sil, db,
    )

    return {
        "pipeline":        pipeline_id,
        "n_clusters":      n_clusters,
        "n_noise":         n_noise,
        "silhouette":      round(float(sil), 6),
        "davies_bouldin":  round(float(db),  6),
    }


def _rank_and_select(results: list[dict[str, Any]]) -> tuple[str, pd.DataFrame]:
    """
    Rank pipelines and return the id of the best one.

    Ranking:
      - rank_sil : ascending rank of silhouette (higher sil → lower rank)
      - rank_db  : ascending rank of davies_bouldin (lower db → lower rank)
      - combined : rank_sil + rank_db  (lower = better)

    NaN pipelines are ranked last.
    """
    df = pd.DataFrame(results)

    # For silhouette, higher is better → rank descending so best gets rank 1
    df["rank_sil"] = df["silhouette"].rank(ascending=False, na_option="bottom")
    # For db, lower is better → rank ascending
    df["rank_db"]  = df["davies_bouldin"].rank(ascending=True,  na_option="bottom")
    df["combined_rank"] = df["rank_sil"] + df["rank_db"]

    # Break ties with silhouette (higher is better)
    best_row = df.sort_values(
        ["combined_rank", "silhouette"],
        ascending=[True, False],
    ).iloc[0]

    best_pipeline = str(best_row["pipeline"])
    logger.info("─" * 60)
    logger.info("Pipeline ranking:")
    for _, row in df.sort_values("combined_rank").iterrows():
        logger.info(
            "  %-15s  combined_rank=%.0f  sil=%.4f  db=%.4f",
            row["pipeline"], row["combined_rank"],
            row["silhouette"] if not pd.isna(row["silhouette"]) else float("nan"),
            row["davies_bouldin"] if not pd.isna(row["davies_bouldin"]) else float("nan"),
        )
    logger.info("→ Selected pipeline: %s", best_pipeline)
    logger.info("─" * 60)

    return best_pipeline, df


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_pipelines(
    labels_dict: dict[str, np.ndarray],
    X_pca: np.ndarray,
    X_umap: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    """
    Evaluate all clustering pipelines and auto-select the best.

    Parameters
    ----------
    labels_dict : {pipeline_id → label_array}
    X_pca       : PCA embedding used by kmeans/gmm pipelines
    X_umap      : UMAP embedding used by hdbscan pipeline

    Returns
    -------
    best_pipeline : pipeline id string
    results_df    : DataFrame with all scores and rankings
    """
    ensure_dirs()
    logger.info("═" * 60)
    logger.info("Evaluating clustering pipelines …")
    logger.info("═" * 60)

    # Map pipeline → evaluation space
    embedding_map: dict[str, np.ndarray] = {
        "pca_kmeans":   X_pca,
        "pca_gmm":      X_pca,
        "umap_hdbscan": X_umap,
    }

    results: list[dict[str, Any]] = []
    for pid in PIPELINE_IDS:
        if pid not in labels_dict:
            logger.warning("Pipeline '%s' missing from labels_dict — skipping.", pid)
            continue
        labels = labels_dict[pid]
        X      = embedding_map[pid]
        row    = _score_pipeline(X, labels, pid)
        results.append(row)

    if not results:
        raise ValueError("No valid pipeline results to evaluate.")

    best_pipeline, results_df = _rank_and_select(results)

    # Mark winner
    results_df["selected"] = results_df["pipeline"] == best_pipeline

    # Persist CSV
    results_df.to_csv(EVALUATION_RESULTS_CSV, index=False)
    logger.info("Evaluation results → %s", EVALUATION_RESULTS_CSV)

    # Persist best pipeline name
    BEST_PIPELINE_TXT.write_text(best_pipeline, encoding="utf-8")
    logger.info("Best pipeline name → %s", BEST_PIPELINE_TXT)

    # Copy best model to best_model.joblib
    source_model = _PIPELINE_MODEL_MAP.get(best_pipeline)
    if source_model and source_model.exists():
        shutil.copy2(source_model, BEST_MODEL_JOBLIB)
        logger.info("Best model copy   → %s", BEST_MODEL_JOBLIB)
    else:
        logger.warning(
            "Could not copy best model (source not found: %s).", source_model
        )

    return best_pipeline, results_df


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_best_pipeline() -> str:
    """Return the name of the best pipeline from disk."""
    if not BEST_PIPELINE_TXT.exists():
        raise FileNotFoundError(
            f"Best pipeline file not found: {BEST_PIPELINE_TXT}. "
            "Run evaluate_pipelines() first."
        )
    name = BEST_PIPELINE_TXT.read_text(encoding="utf-8").strip()
    logger.info("Best pipeline (from disk): %s", name)
    return name


def load_evaluation_results() -> pd.DataFrame:
    """Load evaluation_results.csv from disk."""
    if not EVALUATION_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {EVALUATION_RESULTS_CSV}. "
            "Run evaluate_pipelines() first."
        )
    return pd.read_csv(EVALUATION_RESULTS_CSV)