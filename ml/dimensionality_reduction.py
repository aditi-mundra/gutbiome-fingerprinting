"""
ml/dimensionality_reduction.py
================================
Reusable PCA and UMAP reduction functions for the Microbiome Atlas.

Both functions:
  - Accept a normalised feature matrix
  - Fit the model
  - Save the fitted model to disk (models/)
  - Save the embedding array to data/processed/
  - Return the embedding and the fitted model

Design note: PCA is run first (50 components) and its output is used
as the input for UMAP to improve stability and speed on high-dimensional
sparse data.
"""

from __future__ import annotations

import logging
from typing import Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA

from ml.config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    MODELS_DIR,
    PCA_EMBEDDINGS_NPY,
    PCA_FOR_UMAP_N_COMPONENTS,
    PCA_JOBLIB,
    PCA_PARAMS,
    RANDOM_SEED,
    UMAP_EMBEDDINGS_NPY,
    UMAP_JOBLIB,
    UMAP_PARAMS,
    ensure_dirs,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────────────────────────────────────

def run_pca(
    X: np.ndarray,
    n_components: int | None = None,
    save: bool = True,
) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA on the normalised feature matrix.

    Parameters
    ----------
    X            : (n_samples, n_features) normalised matrix
    n_components : number of PCA components; falls back to PCA_PARAMS default
    save         : if True, persist the model and embedding to disk

    Returns
    -------
    X_pca : (n_samples, n_components) PCA embedding
    pca   : fitted sklearn PCA object
    """
    ensure_dirs()

    params = {**PCA_PARAMS}
    if n_components is not None:
        params["n_components"] = n_components

    # Cap at min(n_samples, n_features) - 1
    max_components = min(X.shape[0], X.shape[1]) - 1
    if params["n_components"] > max_components:
        logger.warning(
            "Requested %d PCA components exceeds max (%d); capping.",
            params["n_components"], max_components,
        )
        params["n_components"] = max_components

    logger.info("Fitting PCA: n_components=%d on shape %s …",
                params["n_components"], X.shape)

    pca   = PCA(**params)
    X_pca = pca.fit_transform(X)

    ev_10 = pca.explained_variance_ratio_[:10].sum() * 100
    ev_all = pca.explained_variance_ratio_.sum() * 100
    logger.info(
        "PCA complete. First 10 components: %.1f%% variance | "
        "All %d components: %.1f%% variance",
        ev_10, params["n_components"], ev_all,
    )

    if save:
        np.save(PCA_EMBEDDINGS_NPY, X_pca)
        joblib.dump(pca, PCA_JOBLIB)
        logger.info("PCA embedding → %s", PCA_EMBEDDINGS_NPY)
        logger.info("PCA model     → %s", PCA_JOBLIB)

    return X_pca, pca


# ──────────────────────────────────────────────────────────────────────────────
# UMAP
# ──────────────────────────────────────────────────────────────────────────────

def run_umap(
    X_pca: np.ndarray,
    n_components: int = 2,
    save: bool = True,
) -> Tuple[np.ndarray, "umap.UMAP"]:
    """
    Fit UMAP on PCA-reduced data.

    UMAP is fitted on the PCA embedding (not raw features) for two reasons:
      1. PCA de-noises the high-dimensional sparse matrix before UMAP
      2. Greatly speeds up nearest-neighbour graph construction

    Parameters
    ----------
    X_pca        : (n_samples, n_pca_components) PCA embedding
    n_components : UMAP output dimensionality (default 2 for visualisation)
    save         : if True, persist model and embedding to disk

    Returns
    -------
    X_umap : (n_samples, n_components) UMAP embedding
    reducer: fitted UMAP object
    """
    try:
        import umap as umap_lib
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP reduction. "
            "Install it with: pip install umap-learn"
        ) from exc

    ensure_dirs()

    params = {**UMAP_PARAMS, "n_components": n_components}
    logger.info("Fitting UMAP: n_components=%d, n_neighbors=%d on shape %s …",
                params["n_components"], params["n_neighbors"], X_pca.shape)

    reducer = umap_lib.UMAP(**params)
    X_umap  = reducer.fit_transform(X_pca)

    logger.info("UMAP complete. Embedding shape: %s", X_umap.shape)

    if save:
        np.save(UMAP_EMBEDDINGS_NPY, X_umap)
        joblib.dump(reducer, UMAP_JOBLIB)
        logger.info("UMAP embedding → %s", UMAP_EMBEDDINGS_NPY)
        logger.info("UMAP model     → %s", UMAP_JOBLIB)

    return X_umap, reducer


# ──────────────────────────────────────────────────────────────────────────────
# Combined pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_dimensionality_reduction(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, PCA, "umap.UMAP"]:
    """
    Run PCA → UMAP in sequence.

    Parameters
    ----------
    X : (n_samples, n_features) normalised feature matrix

    Returns
    -------
    X_pca   : (n_samples, 50)  PCA embedding
    X_umap  : (n_samples, 2)   UMAP embedding
    pca     : fitted PCA
    reducer : fitted UMAP
    """
    ensure_dirs()

    # Step 1: PCA (full 50 components — used for clustering pipelines A & B)
    X_pca, pca = run_pca(X, n_components=PCA_FOR_UMAP_N_COMPONENTS, save=True)

    # Step 2: UMAP on PCA output (2D — used for pipeline C and atlas visualisation)
    X_umap, reducer = run_umap(X_pca, n_components=2, save=True)

    return X_pca, X_umap, pca, reducer


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_pca_embedding() -> Tuple[np.ndarray, PCA]:
    """Load persisted PCA embedding and model from disk."""
    if not PCA_EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            f"PCA embedding not found at {PCA_EMBEDDINGS_NPY}. "
            "Run run_dimensionality_reduction() first."
        )
    X_pca = np.load(PCA_EMBEDDINGS_NPY)
    pca   = joblib.load(PCA_JOBLIB)
    logger.info("Loaded PCA embedding: %s", X_pca.shape)
    return X_pca, pca


def load_umap_embedding() -> Tuple[np.ndarray, "umap.UMAP"]:
    """Load persisted UMAP embedding and model from disk."""
    if not UMAP_EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            f"UMAP embedding not found at {UMAP_EMBEDDINGS_NPY}. "
            "Run run_dimensionality_reduction() first."
        )
    X_umap  = np.load(UMAP_EMBEDDINGS_NPY)
    reducer = joblib.load(UMAP_JOBLIB)
    logger.info("Loaded UMAP embedding: %s", X_umap.shape)
    return X_umap, reducer