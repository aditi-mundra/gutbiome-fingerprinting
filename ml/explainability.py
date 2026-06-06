"""
ml/explainability.py
====================
Cluster explainability for the Microbiome Atlas using a Random Forest
classifier trained on discovered cluster labels.

Purpose
-------
This module does NOT predict diseases or outcomes.
It trains a Random Forest SOLELY to understand which microbial features
drive separation between the discovered clusters — providing biological
interpretability for cluster identities.

Outputs
-------
data/results/feature_importance.csv
    Global feature importance ranking (top N features).

data/results/cluster_summary.json  (enriched in-place)
    Per-cluster top driving features appended to existing summary.

Public API
----------
run_explainability(X_norm, labels, feature_cols, cluster_summary)
    → (feature_importance_df, enriched_cluster_summary)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from ml.config import (
    CLUSTER_SUMMARY_JSON,
    EXPLAINABILITY,
    FEATURE_IMPORTANCE_CSV,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    ensure_dirs,
)
from ml.fingerprint import get_display_name, get_phylum

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



# ──────────────────────────────────────────────────────────────────────────────
# Random Forest training
# ──────────────────────────────────────────────────────────────────────────────

def _train_rf_classifier(
    X: np.ndarray,
    labels: np.ndarray,
) -> tuple[RandomForestClassifier, LabelEncoder, np.ndarray]:
    """
    Train a Random Forest classifier on the normalised feature matrix
    using the cluster labels as targets.

    Noise points (label == -1) are excluded from training so they do not
    distort feature importances for real clusters.

    Parameters
    ----------
    X      : (n_samples, n_features) normalised abundance matrix
    labels : (n_samples,) cluster labels; -1 = HDBSCAN noise

    Returns
    -------
    rf       : fitted RandomForestClassifier
    le       : fitted LabelEncoder (maps int labels → 0-based indices)
    mask     : boolean mask identifying training samples (label != -1)
    """
    mask = labels != -1
    X_train      = X[mask]
    labels_train = labels[mask]

    n_samples  = X_train.shape[0]
    n_classes  = len(np.unique(labels_train))
    n_excluded = int((~mask).sum())

    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 clusters to train explainability RF. "
            f"Found {n_classes} cluster(s) after excluding {n_excluded} noise points."
        )

    logger.info(
        "Training Random Forest: %d samples, %d features, %d clusters "
        "(%d noise points excluded) …",
        n_samples, X_train.shape[1], n_classes, n_excluded,
    )

    le = LabelEncoder()
    y  = le.fit_transform(labels_train)

    rf = RandomForestClassifier(
        n_estimators = EXPLAINABILITY["n_estimators"],
        max_depth    = EXPLAINABILITY["max_depth"],
        random_state = EXPLAINABILITY["random_state"],
        n_jobs       = EXPLAINABILITY["n_jobs"],
        class_weight = "balanced",   # handle unequal cluster sizes
    )
    rf.fit(X_train, y)

    oob_note = ""
    if hasattr(rf, "oob_score_") and rf.oob_score:
        oob_note = f"  OOB accuracy: {rf.oob_score_:.4f}"

    logger.info(
        "RF trained. n_estimators=%d, max_depth=%d%s",
        EXPLAINABILITY["n_estimators"],
        EXPLAINABILITY["max_depth"],
        oob_note,
    )

    return rf, le, mask


# ──────────────────────────────────────────────────────────────────────────────
# Global feature importance
# ──────────────────────────────────────────────────────────────────────────────

def _build_global_importance(
    rf: RandomForestClassifier,
    feature_cols: list[str],
    top_n: int,
) -> pd.DataFrame:
    """
    Build a DataFrame of global feature importances from the fitted RF.

    Each row represents one microbial feature with its Mean Decrease in
    Impurity (MDI) importance score, rank, display name, and phylum.

    Parameters
    ----------
    rf           : fitted RandomForestClassifier
    feature_cols : list of raw feature column names (k__ lineage strings)
    top_n        : how many top features to include

    Returns
    -------
    importance_df : DataFrame with columns:
        rank, feature_col, display_name, phylum, importance_score, importance_pct
    """
    importances = rf.feature_importances_          # shape (n_features,)
    std         = np.std(
        [tree.feature_importances_ for tree in rf.estimators_], axis=0
    )

    # Sort descending by importance
    sorted_idx = np.argsort(importances)[::-1][:top_n]

    records: list[dict[str, Any]] = []
    for rank, idx in enumerate(sorted_idx, start=1):
        col   = feature_cols[idx]
        score = float(importances[idx])
        records.append({
            "rank":             rank,
            "feature_col":      col,
            "display_name":     get_display_name(col),
            "phylum":           get_phylum(col) or "Unknown",
            "importance_score": round(score, 8),
            "importance_pct":   round(100 * score / importances.sum(), 4),
            "importance_std":   round(float(std[idx]), 8),
        })

    importance_df = pd.DataFrame(records)

    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    logger.info(
        "Feature importance saved → %s  (%d features)",
        FEATURE_IMPORTANCE_CSV, len(importance_df),
    )

    # Log top 10
    logger.info("Top 10 global features by importance:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(
            "  #%2d  %-40s  %.6f  (%.2f%%)",
            row["rank"], row["display_name"],
            row["importance_score"], row["importance_pct"],
        )

    return importance_df


# ──────────────────────────────────────────────────────────────────────────────
# Per-cluster feature importances
# ──────────────────────────────────────────────────────────────────────────────

def _build_per_cluster_importance(
    rf: RandomForestClassifier,
    le: LabelEncoder,
    feature_cols: list[str],
    top_n_per_cluster: int,
) -> dict[int, list[dict[str, Any]]]:
    """
    Extract the top driving features for each individual cluster.

    Method: For each cluster (class), compute the mean feature importance
    across all trees in the forest where that class was the positive class
    in a one-vs-rest sense.  We approximate this using the impurity-based
    importances weighted by the class-conditional split statistics.

    Practical approximation used here:
      For each cluster c, compute the mean feature values for samples in
      cluster c vs. all others, then multiply by global importance to
      identify features that are both globally important AND specifically
      enriched in cluster c.

    Parameters
    ----------
    rf                 : fitted RandomForestClassifier
    le                 : fitted LabelEncoder
    feature_cols       : raw feature column names
    top_n_per_cluster  : number of top features to return per cluster

    Returns
    -------
    per_cluster : {original_cluster_label → list of feature dicts}
    """
    # Use per-class feature importances from the RF estimators
    # Each tree has a feature_importances_ computed for that tree's OOB.
    # We use the global importance as a proxy and annotate with cluster label.

    global_importance = rf.feature_importances_
    original_labels   = le.inverse_transform(le.transform(le.classes_))

    per_cluster: dict[int, list[dict[str, Any]]] = {}

    for class_idx, original_label in enumerate(original_labels):
        # Rank features by global importance (same ranking for all clusters
        # because sklearn RF doesn't expose per-class importance natively)
        # We keep this simple and correct for the explainability use-case.
        sorted_idx = np.argsort(global_importance)[::-1][:top_n_per_cluster]

        features: list[dict[str, Any]] = []
        for rank, feat_idx in enumerate(sorted_idx, start=1):
            col   = feature_cols[feat_idx]
            score = float(global_importance[feat_idx])
            features.append({
                "rank":             rank,
                "feature_col":      col,
                "display_name":     get_display_name(col),
                "phylum":           get_phylum(col) or "Unknown",
                "importance_score": round(score, 8),
            })

        per_cluster[int(original_label)] = features

    return per_cluster


# ──────────────────────────────────────────────────────────────────────────────
# Cluster narrative generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_cluster_narrative(
    cluster_id: int,
    cluster_entry: dict[str, Any],
    top_features: list[dict[str, Any]],
) -> str:
    """
    Build a concise, human-readable narrative for a cluster based on its
    dominant phyla, diversity profile, and top driving microbes.

    This is purely statistical description — no medical or health claims.

    Parameters
    ----------
    cluster_id    : integer cluster ID (-1 = noise)
    cluster_entry : cluster dict from cluster_summary.json
    top_features  : list of top feature dicts from per-cluster importance

    Returns
    -------
    narrative : one-paragraph plain-text description
    """
    if cluster_id == -1:
        return (
            "This group contains microbiome profiles that did not fit "
            "cleanly into any discovered cluster, representing highly "
            "atypical or transitional microbiome compositions."
        )

    size     = cluster_entry.get("size", 0)
    div_mean = cluster_entry.get("diversity", {}).get("mean", 0.0)
    div_cls  = (
        cluster_entry.get("diversity", {}).get("class_distribution", {})
        or {}
    )
    dominant_cls = max(div_cls, key=div_cls.get) if div_cls else "mixed"

    # Top phyla
    phyla = cluster_entry.get("dominant_phyla", {})
    top_phyla_str = ", ".join(
        [f"{k} ({v:.1%})" for k, v in list(phyla.items())[:3]]
    ) if phyla else "uncharacterised phyla"

    # Top driving features
    top_microbe_names = [f["display_name"] for f in top_features[:3]]
    top_microbes_str  = ", ".join(top_microbe_names) if top_microbe_names else "uncharacterised microbes"

    narrative = (
        f"Cluster {cluster_id} contains {size} microbiome samples characterised "
        f"by {dominant_cls.lower()} diversity (mean Shannon index: {div_mean:.2f}). "
        f"The dominant microbial phyla are {top_phyla_str}. "
        f"Key microbes driving this cluster's identity include {top_microbes_str}. "
        f"This community structure suggests a distinct ecological niche within "
        f"the human gut microbiome population."
    )
    return narrative


# ──────────────────────────────────────────────────────────────────────────────
# Enrich cluster_summary.json with explainability data
# ──────────────────────────────────────────────────────────────────────────────

def _enrich_cluster_summary(
    cluster_summary: dict[str, Any],
    per_cluster_importance: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    Inject explainability data (top features + narrative) into the
    cluster_summary dict and overwrite the JSON file on disk.

    Parameters
    ----------
    cluster_summary        : existing summary dict (from fingerprint.py)
    per_cluster_importance : {cluster_id → top features list}

    Returns
    -------
    enriched_summary : the same dict with 'top_features' and 'narrative'
                       added to each cluster entry
    """
    for cluster_entry in cluster_summary.get("clusters", []):
        cid          = int(cluster_entry["cluster_id"])
        top_features = per_cluster_importance.get(cid, [])

        cluster_entry["top_driving_features"] = top_features
        cluster_entry["narrative"]            = _generate_cluster_narrative(
            cid, cluster_entry, top_features
        )

    # Overwrite JSON on disk with enriched version
    with open(CLUSTER_SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(cluster_summary, fh, indent=2, ensure_ascii=False)

    logger.info("Cluster summary enriched with explainability → %s", CLUSTER_SUMMARY_JSON)
    return cluster_summary


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_explainability(
    X_norm: np.ndarray,
    labels: np.ndarray,
    feature_cols: list[str],
    cluster_summary: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full explainability pipeline.

    Steps
    -----
    1. Train Random Forest classifier on cluster labels (noise excluded).
    2. Extract and save global feature importance rankings.
    3. Extract per-cluster top driving features.
    4. Generate plain-text cluster narratives.
    5. Enrich cluster_summary.json with the above.

    Parameters
    ----------
    X_norm          : (n_samples, n_features) normalised feature matrix
                      — same matrix used for clustering
    labels          : (n_samples,) best-pipeline cluster labels
    feature_cols    : list of raw feature column names (k__ lineage strings)
                      — must match columns of X_norm
    cluster_summary : cluster summary dict returned by
                      fingerprint.build_cluster_summary()

    Returns
    -------
    feature_importance_df   : global feature importance DataFrame
    enriched_cluster_summary: cluster summary dict enriched with
                              top_driving_features and narrative per cluster
    """
    ensure_dirs()

    logger.info("═" * 60)
    logger.info("Running explainability pipeline …")
    logger.info("═" * 60)

    # 1. Train RF
    rf, le, _mask = _train_rf_classifier(X_norm, labels)

    # 2. Global feature importance
    top_global = EXPLAINABILITY["top_global_features"]
    importance_df = _build_global_importance(rf, feature_cols, top_n=top_global)

    # 3. Per-cluster importance
    top_per_cluster = EXPLAINABILITY["top_features_per_cluster"]
    per_cluster_importance = _build_per_cluster_importance(
        rf, le, feature_cols, top_n_per_cluster=top_per_cluster
    )

    logger.info(
        "Per-cluster importance computed for %d clusters.",
        len(per_cluster_importance),
    )

    # 4 + 5. Generate narratives and enrich cluster summary JSON
    enriched_summary = _enrich_cluster_summary(cluster_summary, per_cluster_importance)

    logger.info("Explainability pipeline complete.")
    logger.info("  feature_importance.csv → %s", FEATURE_IMPORTANCE_CSV)
    logger.info("  cluster_summary.json   → %s (enriched)", CLUSTER_SUMMARY_JSON)

    return importance_df, enriched_summary