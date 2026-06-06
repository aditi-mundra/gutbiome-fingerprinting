"""
ml/fingerprint.py
=================
Generates per-sample microbiome fingerprints and per-cluster summaries.

Per-sample fingerprint:
  - cluster_id           : assigned cluster
  - dominant_microbes    : top-N species by relative abundance
  - phylum_profile       : phylum-level abundance breakdown
  - diversity_score      : Shannon entropy of abundance vector
  - diversity_class      : "Low" | "Medium" | "High"
  - novelty_score        : normalised distance from cluster centroid (0–1)
  - novelty_class        : "Common" | "Moderately Unique" | "Highly Unique"
  - umap_x, umap_y       : 2-D position in the atlas

Per-cluster summary (cluster_summary.json):
  - size, dominant phyla, diversity stats, metadata distributions

Taxonomy parsing uses the pipe-delimited lineage format:
    k__Bacteria|p__Firmicutes|c__...|s__Faecalibacterium_prausnitzii
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from ml.config import (
    CLUSTER_SUMMARY_JSON,
    FINGERPRINT,
    FINGERPRINTS_CSV,
    FEATURE_PREFIX,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    NOVELTY,
    ensure_dirs,
)
from ml.clustering import compute_centroids

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Taxonomy helpers
# ──────────────────────────────────────────────────────────────────────────────

_LEVEL_PREFIXES = {
    "k": "Kingdom",
    "p": "Phylum",
    "c": "Class",
    "o": "Order",
    "f": "Family",
    "g": "Genus",
    "s": "Species",
    "t": "Strain",
}


def parse_lineage(col_name: str) -> dict[str, str]:
    """
    Parse a pipe-delimited taxonomic lineage column name.

    Example
    -------
    'k__Bacteria|p__Firmicutes|g__Faecalibacterium|s__Faecalibacterium_prausnitzii'
    →  {'kingdom': 'Bacteria', 'phylum': 'Firmicutes',
        'genus': 'Faecalibacterium', 'species': 'Faecalibacterium prausnitzii'}
    """
    parts  = col_name.split("|")
    result = {}
    for part in parts:
        if "__" not in part:
            continue
        level_code, _, name = part.partition("__")
        level_code = level_code.strip().lower()
        name       = name.strip().replace("_", " ")
        if name and name.lower() not in ("", "unclassified", "unknown"):
            long_name = _LEVEL_PREFIXES.get(level_code, level_code).lower()
            result[long_name] = name
    return result


def get_display_name(col_name: str) -> str:
    """
    Return a human-readable species or genus name from a lineage column name.
    Falls back to the most specific non-empty taxonomic level.
    """
    tax = parse_lineage(col_name)
    for level in ("species", "genus", "family", "order", "class", "phylum", "kingdom"):
        if level in tax:
            return tax[level]
    return col_name


def get_phylum(col_name: str) -> str | None:
    """Return the phylum name from a lineage column, or None if absent."""
    tax = parse_lineage(col_name)
    return tax.get("phylum")


# ──────────────────────────────────────────────────────────────────────────────
# Shannon Diversity
# ──────────────────────────────────────────────────────────────────────────────

def shannon_diversity(abundances: np.ndarray) -> float:
    """
    Shannon Entropy: H = -∑ p_i * log(p_i)

    Parameters
    ----------
    abundances : 1-D array of non-negative abundance values

    Returns
    -------
    H : Shannon entropy (0 = single species, increases with diversity)
    """
    a    = abundances.astype(np.float64)
    a    = a[a > 0]
    if len(a) == 0:
        return 0.0
    p    = a / a.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def classify_diversity(score: float) -> str:
    """Classify a Shannon diversity score into a human-readable tier."""
    low  = FINGERPRINT["diversity_low_threshold"]
    high = FINGERPRINT["diversity_high_threshold"]
    if score < low:
        return "Low"
    elif score < high:
        return "Medium"
    return "High"


# ──────────────────────────────────────────────────────────────────────────────
# Novelty scoring
# ──────────────────────────────────────────────────────────────────────────────

def compute_novelty_scores(
    X: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute a normalised novelty score (0–1) for each sample.

    Method:
      1. Compute Euclidean distance from each sample to its cluster centroid.
      2. Within each cluster, normalise distances to [0, 1].
      3. Noise points (label -1) receive score = 1.0 (maximally novel).

    Returns
    -------
    novelty_scores : (n_samples,) float array in [0, 1]
    """
    centroids      = compute_centroids(X, labels)
    novelty_scores = np.zeros(len(labels), dtype=np.float64)

    for cluster_id, centroid in centroids.items():
        mask       = labels == cluster_id
        distances  = np.linalg.norm(X[mask] - centroid, axis=1)
        d_min, d_max = distances.min(), distances.max()
        if d_max > d_min:
            novelty_scores[mask] = (distances - d_min) / (d_max - d_min)
        else:
            novelty_scores[mask] = 0.0  # all samples at same distance (single point)

    # Noise points → max novelty
    novelty_scores[labels == -1] = 1.0

    return novelty_scores


def classify_novelty(score: float) -> str:
    """Classify a normalised novelty score into a human-readable tier."""
    common_pct   = NOVELTY["common_percentile"] / 100
    moderate_pct = NOVELTY["moderate_percentile"] / 100
    if score <= common_pct:
        return "Common"
    elif score <= moderate_pct:
        return "Moderately Unique"
    return "Highly Unique"


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample fingerprinting
# ──────────────────────────────────────────────────────────────────────────────

def compute_fingerprints(
    df_raw: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    X_embed: np.ndarray,
    X_umap: np.ndarray,
) -> pd.DataFrame:
    """
    Generate one fingerprint row per sample.

    Parameters
    ----------
    df_raw       : raw abundance DataFrame (un-normalised, for absolute abundances)
    feature_cols : microbial feature column names
    labels       : (n_samples,) cluster labels from the best pipeline
    X_embed      : embedding used by the best pipeline (PCA or UMAP coords)
                   — used for novelty distance calculation
    X_umap       : 2-D UMAP embedding — always used for atlas position

    Returns
    -------
    fingerprints_df : DataFrame with one row per sample
    """
    ensure_dirs()
    n_top    = FINGERPRINT["top_n_microbes"]
    min_abun = FINGERPRINT["min_abundance_threshold"]

    # Pull raw abundances (un-normalised) for biological interpretation
    abun_matrix = df_raw[feature_cols].fillna(0).values.astype(np.float64)

    logger.info("Computing novelty scores …")
    novelty_scores = compute_novelty_scores(X_embed, labels)

    records: list[dict[str, Any]] = []

    for i, sample_id in enumerate(df_raw["sampleID"].values):
        abun_vec = abun_matrix[i]

        # ── Dominant microbes ──────────────────────────────────────────────
        top_idx = np.argsort(abun_vec)[::-1][:n_top]
        dominant: list[dict[str, Any]] = []
        for idx in top_idx:
            val = float(abun_vec[idx])
            if val < min_abun:
                break
            col  = feature_cols[idx]
            tax  = parse_lineage(col)
            dominant.append({
                "name":      get_display_name(col),
                "abundance": round(val, 6),
                "phylum":    tax.get("phylum", "Unknown"),
                "genus":     tax.get("genus", "Unknown"),
                "species":   tax.get("species", "Unknown"),
                "full_lineage": col,
            })

        # ── Phylum profile ─────────────────────────────────────────────────
        phylum_abun: dict[str, float] = {}
        for j, col in enumerate(feature_cols):
            ph = get_phylum(col)
            if ph:
                phylum_abun[ph] = phylum_abun.get(ph, 0.0) + float(abun_vec[j])
        total_abun = sum(phylum_abun.values())
        if total_abun > 0:
            phylum_profile = {k: round(v / total_abun, 6) for k, v in phylum_abun.items()}
        else:
            phylum_profile = {}

        # ── Diversity ──────────────────────────────────────────────────────
        div_score = shannon_diversity(abun_vec)
        div_class = classify_diversity(div_score)

        # ── Novelty ────────────────────────────────────────────────────────
        nov_score = float(novelty_scores[i])
        nov_class = classify_novelty(nov_score)

        records.append({
            "sample_id":         str(sample_id),
            "cluster_id":        int(labels[i]),
            "umap_x":            round(float(X_umap[i, 0]), 6),
            "umap_y":            round(float(X_umap[i, 1]), 6),
            "diversity_score":   round(div_score, 6),
            "diversity_class":   div_class,
            "novelty_score":     round(nov_score, 6),
            "novelty_class":     nov_class,
            "dominant_microbes": json.dumps(dominant),
            "phylum_profile":    json.dumps(phylum_profile),
            "n_dominant":        len(dominant),
        })

    fingerprints_df = pd.DataFrame(records)

    fingerprints_df.to_csv(FINGERPRINTS_CSV, index=False)
    logger.info("Fingerprints saved → %s  (%d rows)", FINGERPRINTS_CSV, len(fingerprints_df))

    return fingerprints_df


# ──────────────────────────────────────────────────────────────────────────────
# Cluster-level summary
# ──────────────────────────────────────────────────────────────────────────────

def _safe_value_counts(series: pd.Series, top_n: int = 10) -> dict[str, int]:
    """Return value counts as a dict, excluding NaN, limited to top_n."""
    return (
        series.dropna()
        .value_counts()
        .head(top_n)
        .to_dict()
    )


def build_cluster_summary(
    fingerprints_df: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    eval_results: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a comprehensive per-cluster summary dictionary.

    Parameters
    ----------
    fingerprints_df : output of compute_fingerprints()
    metadata        : cleaned metadata DataFrame
    labels          : (n_samples,) cluster labels
    eval_results    : evaluation results DataFrame (for scores)

    Returns
    -------
    summary : dict serialisable to JSON
    """
    ensure_dirs()

    # Merge fingerprints with metadata on sampleID
    merged = fingerprints_df.merge(
        metadata.rename(columns={"sampleID": "sample_id"}),
        on="sample_id",
        how="left",
    )

    unique_clusters = sorted([int(c) for c in np.unique(labels)])
    cluster_data: list[dict[str, Any]] = []

    for cluster_id in unique_clusters:
        cdf = merged[merged["cluster_id"] == cluster_id]
        if len(cdf) == 0:
            continue

        # Dominant phyla from phylum profiles
        phylum_agg: dict[str, float] = {}
        for row in cdf["phylum_profile"]:
            try:
                profile = json.loads(row) if isinstance(row, str) else row
                for k, v in profile.items():
                    phylum_agg[k] = phylum_agg.get(k, 0.0) + float(v)
            except (json.JSONDecodeError, TypeError):
                continue
        total_ph = sum(phylum_agg.values())
        if total_ph > 0:
            dominant_phyla = dict(
                sorted(
                    {k: round(v / total_ph, 4) for k, v in phylum_agg.items()}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:6]
            )
        else:
            dominant_phyla = {}

        # Collect dominant microbe names
        all_microbes: dict[str, float] = {}
        for row in cdf["dominant_microbes"]:
            try:
                microbes = json.loads(row) if isinstance(row, str) else row
                for m in microbes:
                    name = m.get("name", "Unknown")
                    abun = float(m.get("abundance", 0.0))
                    all_microbes[name] = all_microbes.get(name, 0.0) + abun
            except (json.JSONDecodeError, TypeError):
                continue
        top_microbes = dict(
            sorted(all_microbes.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Metadata distributions
        label_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"

        cluster_entry: dict[str, Any] = {
            "cluster_id":    cluster_id,
            "label":         label_name,
            "size":          int(len(cdf)),
            "dominant_phyla": dominant_phyla,
            "top_microbes":   top_microbes,
            "diversity": {
                "mean":  round(float(cdf["diversity_score"].mean()), 4),
                "std":   round(float(cdf["diversity_score"].std()), 4),
                "min":   round(float(cdf["diversity_score"].min()), 4),
                "max":   round(float(cdf["diversity_score"].max()), 4),
                "class_distribution": _safe_value_counts(cdf["diversity_class"]),
            },
            "novelty": {
                "mean":  round(float(cdf["novelty_score"].mean()), 4),
                "class_distribution": _safe_value_counts(cdf["novelty_class"]),
            },
            "metadata_distributions": {
                "disease":  _safe_value_counts(cdf.get("disease", pd.Series(dtype=str))),
                "country":  _safe_value_counts(cdf.get("country", pd.Series(dtype=str))),
                "gender":   _safe_value_counts(cdf.get("gender",  pd.Series(dtype=str))),
            },
        }

        # Add age stats if available
        if "age" in cdf.columns:
            age_clean = pd.to_numeric(cdf["age"], errors="coerce").dropna()
            if len(age_clean) > 0:
                cluster_entry["age_stats"] = {
                    "mean":   round(float(age_clean.mean()), 1),
                    "median": round(float(age_clean.median()), 1),
                    "std":    round(float(age_clean.std()), 1),
                    "min":    round(float(age_clean.min()), 1),
                    "max":    round(float(age_clean.max()), 1),
                }

        cluster_data.append(cluster_entry)

    # Top-level eval summary for the selected pipeline
    # Guard: 'selected' column exists when evaluate_pipelines() writes the CSV,
    # but may be absent if eval_results is loaded from disk before the column
    # was appended. Fall back to picking the row with highest silhouette.
    if "selected" in eval_results.columns:
        selected_row = eval_results[eval_results["selected"] == True]
    else:
        selected_row = eval_results.sort_values("silhouette", ascending=False).head(1)
    if not selected_row.empty:
        pipeline_scores = {
            "pipeline":       selected_row.iloc[0]["pipeline"],
            "silhouette":     float(selected_row.iloc[0]["silhouette"]),
            "davies_bouldin": float(selected_row.iloc[0]["davies_bouldin"]),
        }
    else:
        pipeline_scores = {}

    summary: dict[str, Any] = {
        "n_clusters":     len([c for c in unique_clusters if c >= 0]),
        "n_noise":        int((labels == -1).sum()),
        "pipeline_scores": pipeline_scores,
        "clusters":       cluster_data,
    }

    with open(CLUSTER_SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info("Cluster summary saved → %s", CLUSTER_SUMMARY_JSON)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_fingerprints() -> pd.DataFrame:
    """Load fingerprints CSV from disk."""
    if not FINGERPRINTS_CSV.exists():
        raise FileNotFoundError(
            f"Fingerprints not found: {FINGERPRINTS_CSV}. "
            "Run compute_fingerprints() first."
        )
    return pd.read_csv(FINGERPRINTS_CSV)