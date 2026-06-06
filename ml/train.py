"""
ml/train.py
===========
Lightweight orchestrator for the Microbiome Atlas ML pipeline.

Execution order
---------------
 1. Preprocessing      — load, clean, normalise, persist features + metadata
 2. EDA                — generate all exploratory figures to docs/figures/
 3. Dimensionality     — PCA (50-D) + UMAP (2-D), persist embeddings + models
    reduction
 4. Clustering         — Pipeline A (KMeans), B (GMM), C (HDBSCAN) in parallel
 5. Evaluation         — score all pipelines, auto-select best, persist results
 6. Fingerprinting     — per-sample dominant microbes, diversity, novelty
 7. Cluster summary    — per-cluster biological + metadata statistics
 8. Embeddings CSV     — unified atlas table for frontend
 9. Similarity         — KNN assignments table + similarity network JSON
10. Explainability     — Random Forest on cluster labels → feature importance
                         + enriched cluster narratives

Design rules
------------
- train.py contains ZERO business logic.  All logic lives in the modules.
- train.py contains NO direct data transformations.
- train.py does NOT connect to any database.
- Each step is timed and logged.
- A top-level exception handler prints a clear failure message with the
  failing step name so engineers can diagnose quickly.

Usage
-----
    # From the project root:
    python -m ml.train

    # Or directly:
    python ml/train.py
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator

# ── Config must be imported before any other ml module ──────────────────────
from ml.config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    ensure_dirs,
)

# ── Module imports ───────────────────────────────────────────────────────────
from ml.preprocessing       import run_preprocessing, load_raw_data, identify_columns
from ml.eda                  import run_eda
from ml.dimensionality_reduction import run_dimensionality_reduction
from ml.clustering           import run_all_pipelines
from ml.evaluation           import evaluate_pipelines
from ml.fingerprint          import compute_fingerprints, build_cluster_summary
from ml.similarity           import (
    build_cluster_assignments,
    build_similarity_network,
    save_embeddings_csv,
)
from ml.explainability       import run_explainability

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Timing context manager
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _step(name: str) -> Generator[None, None, None]:
    """Log start / end / elapsed time for each pipeline step."""
    logger.info("")
    logger.info("┌─ STEP: %s", name)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.error("└─ FAILED: %s  (%.1fs)", name, elapsed)
        raise
    elapsed = time.perf_counter() - t0
    logger.info("└─ DONE:   %s  (%.1fs)", name, elapsed)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the complete Microbiome Atlas ML pipeline end-to-end.

    All intermediate results are persisted by the called modules.
    train.py only passes return values between steps — it does not
    transform or manipulate data directly.
    """
    total_start = time.perf_counter()

    logger.info("═" * 65)
    logger.info("  MICROBIOME ATLAS — ML PIPELINE")
    logger.info("═" * 65)

    ensure_dirs()

    # ── 1. Preprocessing ────────────────────────────────────────────────────
    with _step("1 / 10  Preprocessing"):
        X_norm, metadata, feature_cols = run_preprocessing()

    # ── 2. EDA ──────────────────────────────────────────────────────────────
    # EDA needs the raw DataFrame (un-normalised) for biological plots.
    # We load it here via the same loader used by preprocessing so we don't
    # repeat I/O inside train.py.
    with _step("2 / 10  Exploratory Data Analysis"):
        raw_df = load_raw_data()
        _, meta_cols = identify_columns(raw_df)
        run_eda(raw_df, feature_cols, meta_cols)

    # ── 3. Dimensionality reduction ─────────────────────────────────────────
    with _step("3 / 10  Dimensionality Reduction  (PCA → UMAP)"):
        X_pca, X_umap, _pca_model, _umap_model = run_dimensionality_reduction(X_norm)

    # ── 4. Clustering ───────────────────────────────────────────────────────
    with _step("4 / 10  Clustering  (KMeans | GMM | HDBSCAN)"):
        labels_dict = run_all_pipelines(X_pca, X_umap)

    # ── 5. Evaluation + model selection ─────────────────────────────────────
    with _step("5 / 10  Evaluation  &  Pipeline Selection"):
        best_pipeline, eval_results = evaluate_pipelines(labels_dict, X_pca, X_umap)
        logger.info("Selected pipeline: %s", best_pipeline)

    # Resolve the winning label array and its embedding space
    best_labels  = labels_dict[best_pipeline]
    best_X_embed = X_pca if best_pipeline in ("pca_kmeans", "pca_gmm") else X_umap

    # ── 6. Fingerprinting ───────────────────────────────────────────────────
    with _step("6 / 10  Fingerprinting  (diversity | novelty | dominant microbes)"):
        # compute_fingerprints uses raw abundance for biological interpretation
        fingerprints_df = compute_fingerprints(
            df_raw       = raw_df,
            feature_cols = feature_cols,
            labels       = best_labels,
            X_embed      = best_X_embed,
            X_umap       = X_umap,
        )

    # ── 7. Cluster summary ──────────────────────────────────────────────────
    with _step("7 / 10  Cluster Summary  (biological + metadata statistics)"):
        cluster_summary = build_cluster_summary(
            fingerprints_df = fingerprints_df,
            metadata        = metadata,
            labels          = best_labels,
            eval_results    = eval_results,
        )
        logger.info(
            "Clusters discovered: %d  |  Noise points: %d",
            cluster_summary["n_clusters"],
            cluster_summary["n_noise"],
        )

    # ── 8. Embeddings CSV ───────────────────────────────────────────────────
    with _step("8 / 10  Embeddings CSV  (atlas table)"):
        sample_ids = metadata["sampleID"].astype(str).tolist()
        save_embeddings_csv(
            sample_ids = sample_ids,
            labels     = best_labels,
            X_pca      = X_pca,
            X_umap     = X_umap,
            metadata   = metadata,
        )

    # ── 9. Similarity engine ─────────────────────────────────────────────────
    with _step("9 / 10  Similarity  (KNN assignments + network)"):
        build_cluster_assignments(
            metadata        = metadata,
            labels          = best_labels,
            X_norm          = X_norm,
            X_umap          = X_umap,
            fingerprints_df = fingerprints_df,
        )
        build_similarity_network(
            X_norm     = X_norm,
            sample_ids = sample_ids,
            labels     = best_labels,
            X_umap     = X_umap,
            metadata   = metadata,
        )

    # ── 10. Explainability ──────────────────────────────────────────────────
    with _step("10 / 10  Explainability  (Random Forest → feature importance)"):
        _importance_df, _enriched_summary = run_explainability(
            X_norm          = X_norm,
            labels          = best_labels,
            feature_cols    = feature_cols,
            cluster_summary = cluster_summary,
        )

    # ── Summary ─────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start

    logger.info("")
    logger.info("═" * 65)
    logger.info("  PIPELINE COMPLETE  —  Total time: %.1f s", total_elapsed)
    logger.info("═" * 65)
    logger.info("")
    logger.info("Artifacts written:")
    logger.info("  models/")
    logger.info("    best_model.joblib")
    logger.info("    pca.joblib")
    logger.info("    umap.joblib")
    logger.info("    scaler.joblib")
    logger.info("    kmeans.joblib  |  gmm.joblib  |  hdbscan.joblib")
    logger.info("    best_pipeline.txt  →  %s", best_pipeline)
    logger.info("")
    logger.info("  data/results/")
    logger.info("    embeddings.csv")
    logger.info("    cluster_assignments.csv")
    logger.info("    fingerprints.csv")
    logger.info("    evaluation_results.csv")
    logger.info("    cluster_summary.json  (enriched with narratives)")
    logger.info("    similarity_network.json")
    logger.info("    feature_importance.csv")
    logger.info("")
    logger.info("  docs/figures/")
    logger.info("    dataset_overview.png  |  metadata_distributions.png")
    logger.info("    missing_values.png    |  top_microbes.png")
    logger.info("    most_variable_features.png")
    logger.info("    pca_variance.png      |  umap_preview.png")
    logger.info("")
    logger.info("Next step: start the FastAPI backend.")
    logger.info("  cd backend && uvicorn main:app --reload")
    logger.info("═" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)