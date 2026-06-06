"""
ml/clustering.py
================
Implements three clustering pipelines for the Microbiome Atlas:

  Pipeline A  — PCA (50 dims) → KMeans
  Pipeline B  — PCA (50 dims) → Gaussian Mixture Model (GMM)
  Pipeline C  — UMAP (2 dims) → HDBSCAN

Each pipeline function:
  - Accepts the appropriate embedding as input
  - Fits the model
  - Returns cluster label array
  - Persists the fitted model to disk (models/)

The selection of the best pipeline is handled by evaluation.py.
"""

from __future__ import annotations

import logging
from typing import Tuple

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from ml.config import (
    GMM_JOBLIB,
    GMM_PARAMS,
    HDBSCAN_JOBLIB,
    HDBSCAN_PARAMS,
    KMEANS_JOBLIB,
    KMEANS_PARAMS,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    ensure_dirs,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline A: PCA → KMeans
# ──────────────────────────────────────────────────────────────────────────────

def run_kmeans(
    X_pca: np.ndarray,
    save: bool = True,
) -> Tuple[np.ndarray, KMeans]:
    """
    Fit KMeans on PCA-reduced data.

    Parameters
    ----------
    X_pca : (n_samples, n_pca_components) PCA embedding
    save  : persist model to disk if True

    Returns
    -------
    labels : (n_samples,) integer cluster labels  [0 … k-1]
    model  : fitted KMeans object
    """
    ensure_dirs()

    logger.info(
        "Pipeline A — KMeans: n_clusters=%d on shape %s …",
        KMEANS_PARAMS["n_clusters"], X_pca.shape,
    )

    model  = KMeans(**KMEANS_PARAMS)
    labels = model.fit_predict(X_pca)

    cluster_sizes = {int(k): int((labels == k).sum()) for k in np.unique(labels)}
    logger.info("KMeans cluster sizes: %s", cluster_sizes)

    if save:
        joblib.dump(model, KMEANS_JOBLIB)
        logger.info("KMeans model → %s", KMEANS_JOBLIB)

    return labels, model


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline B: PCA → GMM
# ──────────────────────────────────────────────────────────────────────────────

def run_gmm(
    X_pca: np.ndarray,
    save: bool = True,
) -> Tuple[np.ndarray, GaussianMixture]:
    """
    Fit a Gaussian Mixture Model on PCA-reduced data.

    Soft assignments are computed internally; hard labels are returned.

    Parameters
    ----------
    X_pca : (n_samples, n_pca_components) PCA embedding
    save  : persist model to disk if True

    Returns
    -------
    labels : (n_samples,) integer cluster labels  [0 … k-1]
    model  : fitted GaussianMixture object
    """
    ensure_dirs()

    logger.info(
        "Pipeline B — GMM: n_components=%d on shape %s …",
        GMM_PARAMS["n_components"], X_pca.shape,
    )

    model  = GaussianMixture(**GMM_PARAMS)
    model.fit(X_pca)
    labels = model.predict(X_pca)

    cluster_sizes = {int(k): int((labels == k).sum()) for k in np.unique(labels)}
    logger.info("GMM cluster sizes: %s", cluster_sizes)
    logger.info(
        "GMM converged: %s  |  BIC: %.2f  |  AIC: %.2f",
        model.converged_, model.bic(X_pca), model.aic(X_pca),
    )

    if save:
        joblib.dump(model, GMM_JOBLIB)
        logger.info("GMM model → %s", GMM_JOBLIB)

    return labels, model


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline C: UMAP → HDBSCAN
# ──────────────────────────────────────────────────────────────────────────────

def run_hdbscan(
    X_umap: np.ndarray,
    save: bool = True,
) -> Tuple[np.ndarray, "hdbscan.HDBSCAN"]:
    """
    Fit HDBSCAN on 2-D UMAP embedding.

    HDBSCAN is density-based and does not require a preset number of clusters.
    Samples assigned to cluster label -1 are considered noise points.

    Parameters
    ----------
    X_umap : (n_samples, 2) UMAP embedding
    save   : persist model to disk if True

    Returns
    -------
    labels : (n_samples,) integer cluster labels  [-1 = noise, 0, 1, …]
    model  : fitted HDBSCAN object
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError as exc:
        raise ImportError(
            "hdbscan is required for Pipeline C. "
            "Install it with: pip install hdbscan"
        ) from exc

    ensure_dirs()

    logger.info(
        "Pipeline C — HDBSCAN: min_cluster_size=%d on shape %s …",
        HDBSCAN_PARAMS["min_cluster_size"], X_umap.shape,
    )

    model  = hdbscan_lib.HDBSCAN(**HDBSCAN_PARAMS)
    labels = model.fit_predict(X_umap)

    unique_labels = np.unique(labels)
    n_clusters    = int((unique_labels >= 0).sum())
    n_noise       = int((labels == -1).sum())
    cluster_sizes = {int(k): int((labels == k).sum()) for k in unique_labels if k >= 0}

    logger.info(
        "HDBSCAN: %d clusters found, %d noise points (%.1f%%)",
        n_clusters, n_noise, 100 * n_noise / len(labels),
    )
    logger.info("HDBSCAN cluster sizes: %s", cluster_sizes)

    if save:
        joblib.dump(model, HDBSCAN_JOBLIB)
        logger.info("HDBSCAN model → %s", HDBSCAN_JOBLIB)

    return labels, model


# ──────────────────────────────────────────────────────────────────────────────
# Cluster centroid computation (shared utility)
# ──────────────────────────────────────────────────────────────────────────────

def compute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute the centroid (mean position) of each cluster in embedding space.

    Parameters
    ----------
    X      : (n_samples, n_dims) embedding coordinates
    labels : (n_samples,) cluster labels

    Returns
    -------
    centroids : {cluster_id → centroid_array}
    Noise points (label == -1) are excluded.
    """
    centroids: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        if label == -1:
            continue
        mask           = labels == label
        centroids[int(label)] = X[mask].mean(axis=0)
    return centroids


# ──────────────────────────────────────────────────────────────────────────────
# Run all three pipelines in one call
# ──────────────────────────────────────────────────────────────────────────────

def run_all_pipelines(
    X_pca: np.ndarray,
    X_umap: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Fit all three clustering pipelines.

    Parameters
    ----------
    X_pca  : (n_samples, 50)  PCA embedding  — used by Pipelines A & B
    X_umap : (n_samples, 2)   UMAP embedding — used by Pipeline C

    Returns
    -------
    labels_dict : {
        "pca_kmeans":    np.ndarray,
        "pca_gmm":       np.ndarray,
        "umap_hdbscan":  np.ndarray,
    }
    """
    logger.info("Running all three clustering pipelines …")

    labels_kmeans,   _km  = run_kmeans(X_pca,  save=True)
    labels_gmm,      _gmm = run_gmm(X_pca,     save=True)
    labels_hdbscan,  _hdb = run_hdbscan(X_umap, save=True)

    return {
        "pca_kmeans":   labels_kmeans,
        "pca_gmm":      labels_gmm,
        "umap_hdbscan": labels_hdbscan,
    }