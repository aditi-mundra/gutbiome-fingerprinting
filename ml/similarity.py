"""
ml/similarity.py
================
Builds the microbiome similarity engine for the Microbiome Atlas:

  1. NearestNeighborIndex  — sklearn KNN on normalised feature space
  2. Cosine similarity ranking per sample
  3. Per-sample neighbor table → cluster_assignments.csv
  4. Similarity network graph → similarity_network.json  (NetworkX + JSON)

The network JSON is structured for direct consumption by Plotly:
  {
    "nodes": [{"id": sample_id, "cluster": int, "umap_x": f, "umap_y": f, ...}],
    "edges": [{"source": sid, "target": sid, "weight": cosine_score}]
  }
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from ml.config import (
    CLUSTER_ASSIGNMENTS_CSV,
    EMBEDDINGS_CSV,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    SIMILARITY,
    SIMILARITY_NETWORK_JSON,
    ensure_dirs,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# Nearest-Neighbor Index
# ──────────────────────────────────────────────────────────────────────────────

class MicrobiomeNearestNeighbors:
    """
    Thin wrapper around sklearn NearestNeighbors with cosine metric.

    The index is built on the normalised feature matrix (not embeddings)
    so that similarity reflects biological profile overlap, not projection
    artefacts.
    """

    def __init__(
        self,
        n_neighbors: int = SIMILARITY["n_neighbors"],
        metric: str      = SIMILARITY["metric"],
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric      = metric
        self._nn         = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 because the query sample itself is returned
            metric=metric,
            algorithm="brute",            # brute is best for cosine on dense matrices
            n_jobs=-1,
        )
        self._sample_ids: list[str] = []
        self._fitted: bool = False

    def fit(self, X: np.ndarray, sample_ids: list[str]) -> "MicrobiomeNearestNeighbors":
        """
        Fit the index on the feature matrix.

        Parameters
        ----------
        X          : (n_samples, n_features) normalised abundance matrix
        sample_ids : list of sampleID strings (same order as X rows)
        """
        if len(sample_ids) != X.shape[0]:
            raise ValueError(
                f"sample_ids length ({len(sample_ids)}) must match X rows ({X.shape[0]})."
            )
        logger.info("Fitting KNN index: n=%d, metric=%s on shape %s …",
                    self.n_neighbors, self.metric, X.shape)

        # sklearn requires n_neighbors <= n_samples_fit; cap defensively
        effective_k = min(self.n_neighbors + 1, X.shape[0])
        if effective_k < self.n_neighbors + 1:
            logger.warning(
                "n_samples=%d is smaller than n_neighbors+1=%d; "
                "capping effective neighbors to %d.",
                X.shape[0], self.n_neighbors + 1, effective_k - 1,
            )
        self._nn.set_params(n_neighbors=effective_k)
        self._nn.fit(X)
        self._sample_ids = list(sample_ids)
        self._fitted     = True
        logger.info("KNN index fitted.")
        return self

    def query(self, X_query: np.ndarray) -> list[list[dict[str, Any]]]:
        """
        Find nearest neighbors for each row in X_query.

        Returns
        -------
        results : list of lists, one inner list per query sample.
                  Each inner list contains dicts:
                  {"rank": int, "sample_id": str, "cosine_score": float}
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before query().")

        distances, indices = self._nn.kneighbors(X_query)
        results: list[list[dict[str, Any]]] = []

        for row_dist, row_idx in zip(distances, indices):
            neighbors: list[dict[str, Any]] = []
            rank = 1
            for dist, idx in zip(row_dist, row_idx):
                # When querying the training set itself, first neighbour is itself
                # Skip it by detecting zero distance (or near-zero for cosine)
                if dist < 1e-9:
                    continue
                # cosine distance = 1 - cosine_similarity
                cos_score = float(max(0.0, 1.0 - dist))
                neighbors.append({
                    "rank":         rank,
                    "sample_id":    self._sample_ids[idx],
                    "cosine_score": round(cos_score, 6),
                })
                rank += 1
                if rank > self.n_neighbors:
                    break
            results.append(neighbors)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Build cluster assignments + neighbor table
# ──────────────────────────────────────────────────────────────────────────────

def build_cluster_assignments(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    X_norm: np.ndarray,
    X_umap: np.ndarray,
    fingerprints_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the full cluster_assignments.csv table.

    One row per sample with:
      sample_id, cluster_id, umap_x, umap_y, diversity_score,
      diversity_class, novelty_score, novelty_class,
      top_1_neighbor, top_1_cosine, … (for top 3 neighbours)
      + core metadata columns

    Returns
    -------
    assignments_df : saved to CLUSTER_ASSIGNMENTS_CSV
    """
    ensure_dirs()

    sample_ids = metadata["sampleID"].astype(str).tolist()

    # Build & query KNN
    knn = MicrobiomeNearestNeighbors(n_neighbors=3)
    knn.fit(X_norm, sample_ids)
    neighbor_results = knn.query(X_norm)

    # Flatten top-3 neighbors into columns
    neighbor_rows: list[dict[str, Any]] = []
    for neighbors in neighbor_results:
        row: dict[str, Any] = {}
        for n in neighbors[:3]:
            r = n["rank"]
            row[f"neighbor_{r}_id"]     = n["sample_id"]
            row[f"neighbor_{r}_cosine"] = n["cosine_score"]
        neighbor_rows.append(row)

    neighbor_df = pd.DataFrame(neighbor_rows)

    # Core assignment columns
    fp_slim = fingerprints_df[[
        "sample_id", "cluster_id",
        "umap_x", "umap_y",
        "diversity_score", "diversity_class",
        "novelty_score",  "novelty_class",
    ]].copy()

    # Merge metadata
    meta_slim = metadata[
        [c for c in ["sampleID", "disease", "age", "gender", "country", "bmi"]
         if c in metadata.columns]
    ].rename(columns={"sampleID": "sample_id"})

    assignments_df = (
        fp_slim
        .merge(meta_slim, on="sample_id", how="left")
        .reset_index(drop=True)
    )
    assignments_df = pd.concat(
        [assignments_df, neighbor_df.reset_index(drop=True)], axis=1
    )

    assignments_df.to_csv(CLUSTER_ASSIGNMENTS_CSV, index=False)
    logger.info("Cluster assignments → %s  (%d rows)", CLUSTER_ASSIGNMENTS_CSV, len(assignments_df))

    return assignments_df


# ──────────────────────────────────────────────────────────────────────────────
# Similarity Network
# ──────────────────────────────────────────────────────────────────────────────

def build_similarity_network(
    X_norm: np.ndarray,
    sample_ids: list[str],
    labels: np.ndarray,
    X_umap: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a similarity network graph and export it as JSON for Plotly.

    Nodes  : microbiome samples (subsampled to MAX_NETWORK_NODES)
    Edges  : cosine similarity >= SIMILARITY["network_threshold"]

    Parameters
    ----------
    X_norm     : (n_samples, n_features) normalised feature matrix
    sample_ids : list of sampleID strings
    labels     : (n_samples,) cluster labels
    X_umap     : (n_samples, 2) UMAP coordinates for node positions
    metadata   : cleaned metadata DataFrame

    Returns
    -------
    network_dict : {nodes: [...], edges: [...]} — also saved to JSON
    """
    ensure_dirs()
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required for similarity network. "
            "Install with: pip install networkx"
        ) from exc

    max_nodes  = SIMILARITY["max_network_nodes"]
    threshold  = SIMILARITY["network_threshold"]
    n_samples  = len(sample_ids)

    # Subsample if needed (for performance)
    if n_samples > max_nodes:
        rng      = random.Random(42)
        indices  = rng.sample(range(n_samples), max_nodes)
        indices  = sorted(indices)
        logger.info(
            "Subsampling %d / %d samples for network (threshold=%.2f) …",
            max_nodes, n_samples, threshold,
        )
    else:
        indices = list(range(n_samples))

    sub_ids    = [sample_ids[i] for i in indices]
    sub_X      = X_norm[indices]
    sub_labels = labels[indices]
    sub_umap   = X_umap[indices]

    # Metadata lookup
    meta_lookup = {}
    if "sampleID" in metadata.columns:
        meta_unique = metadata.drop_duplicates(subset=["sampleID"], keep="first")
        meta_lookup = meta_unique.set_index("sampleID").to_dict(orient="index")

    # Pairwise cosine similarity
    logger.info("Computing pairwise cosine similarities (%d × %d) …", len(sub_ids), len(sub_ids))
    sim_matrix = cosine_similarity(sub_X)   # shape (n_sub, n_sub)

    # Build NetworkX graph
    G = nx.Graph()

    for i, sid in enumerate(sub_ids):
        meta = meta_lookup.get(sid, {})
        G.add_node(sid, **{
            "cluster":   int(sub_labels[i]),
            "umap_x":    round(float(sub_umap[i, 0]), 4),
            "umap_y":    round(float(sub_umap[i, 1]), 4),
            "disease":   str(meta.get("disease", "unknown")),
            "country":   str(meta.get("country", "unknown")),
            "gender":    str(meta.get("gender", "unknown")),
        })

    n_edges = 0
    for i in range(len(sub_ids)):
        for j in range(i + 1, len(sub_ids)):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                G.add_edge(sub_ids[i], sub_ids[j], weight=round(sim, 4))
                n_edges += 1

    logger.info(
        "Network: %d nodes, %d edges (threshold=%.2f)", G.number_of_nodes(), n_edges, threshold
    )

    # Serialise to Plotly-friendly JSON
    nodes: list[dict[str, Any]] = [
        {"id": node, **data}
        for node, data in G.nodes(data=True)
    ]
    edges: list[dict[str, Any]] = [
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in G.edges(data=True)
    ]

    network_dict: dict[str, Any] = {
        "n_nodes":    G.number_of_nodes(),
        "n_edges":    G.number_of_edges(),
        "threshold":  threshold,
        "nodes":      nodes,
        "edges":      edges,
    }

    with open(SIMILARITY_NETWORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(network_dict, fh, indent=2, ensure_ascii=False)

    logger.info("Similarity network → %s", SIMILARITY_NETWORK_JSON)
    return network_dict


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings CSV (for backend + frontend atlas)
# ──────────────────────────────────────────────────────────────────────────────

def save_embeddings_csv(
    sample_ids: list[str],
    labels: np.ndarray,
    X_pca: np.ndarray,
    X_umap: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save a unified embeddings CSV used by the Atlas page.

    Columns: sample_id, cluster_id, umap_x, umap_y, pca_x, pca_y,
             disease, country, gender, age
    """
    meta_lookup = {}
    if "sampleID" in metadata.columns:
        meta_cols = [c for c in ["disease", "country", "gender", "age"] if c in metadata.columns]
        meta_subset = metadata[["sampleID"] + meta_cols].drop_duplicates(subset=["sampleID"], keep="first")
        meta_lookup = meta_subset.set_index("sampleID")[meta_cols].to_dict(orient="index")

    records: list[dict[str, Any]] = []
    for i, sid in enumerate(sample_ids):
        meta = meta_lookup.get(sid, {})
        records.append({
            "sample_id": sid,
            "cluster_id": int(labels[i]),
            "umap_x": round(float(X_umap[i, 0]), 6),
            "umap_y": round(float(X_umap[i, 1]), 6),
            "pca_x":  round(float(X_pca[i, 0]),  6),
            "pca_y":  round(float(X_pca[i, 1]),  6),
            "disease": str(meta.get("disease", "")),
            "country": str(meta.get("country", "")),
            "gender":  str(meta.get("gender",  "")),
            "age":     meta.get("age", None),
        })

    emb_df = pd.DataFrame(records)
    emb_df.to_csv(EMBEDDINGS_CSV, index=False)
    logger.info("Embeddings CSV → %s  (%d rows)", EMBEDDINGS_CSV, len(emb_df))
    return emb_df