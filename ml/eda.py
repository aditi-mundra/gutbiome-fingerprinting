"""
ml/eda.py
=========
Medium-level Exploratory Data Analysis for the Microbiome Atlas.

Generates the following figures to docs/figures/:
  - dataset_overview.png
  - metadata_distributions.png
  - missing_values.png
  - top_microbes.png
  - most_variable_features.png
  - pca_variance.png
  - umap_preview.png

Run standalone:
    python -m ml.eda
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ABUNDANCE_CSV,
    CORE_META_COLS,
    DOCS_FIGURES_DIR,
    EDA,
    FEATURE_PREFIX,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    NULL_SENTINELS,
    PCA_PARAMS,
    RANDOM_SEED,
    UMAP_PARAMS,
    ensure_dirs,
)

warnings.filterwarnings("ignore")

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────
_PALETTE = [
    "#5C6BC0", "#26A69A", "#EF7C34", "#EC407A",
    "#AB47BC", "#42A5F5", "#66BB6A", "#FFA726",
    "#78909C", "#D4E157",
]


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to docs/figures/ and close it."""
    path = DOCS_FIGURES_DIR / f"{name}.{EDA['figure_format']}"
    fig.savefig(path, dpi=EDA["figure_dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


def _clean_meta_value(series: pd.Series) -> pd.Series:
    """Replace null sentinel strings with np.nan."""
    return series.replace(NULL_SENTINELS, np.nan)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset Overview
# ──────────────────────────────────────────────────────────────────────────────
def plot_dataset_overview(df: pd.DataFrame, feature_cols: list[str], meta_cols: list[str]) -> dict[str, Any]:
    """
    Render a text-heavy summary panel showing key dataset statistics.

    Returns a dict of overview stats (also logged).
    """
    n_samples   = len(df)
    n_features  = len(feature_cols)
    n_meta      = len(meta_cols)
    sparsity    = float((df[feature_cols].values == 0).mean())
    n_diseases  = df["disease"].nunique()
    n_countries = df["country"].nunique()

    stats = {
        "samples":          n_samples,
        "microbial_features": n_features,
        "metadata_columns": n_meta,
        "feature_sparsity": round(sparsity, 4),
        "unique_diseases":  n_diseases,
        "unique_countries": n_countries,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    rows = [
        ["Metric",                 "Value"],
        ["Total samples",          f"{n_samples:,}"],
        ["Microbial features",     f"{n_features:,}"],
        ["Metadata columns",       f"{n_meta:,}"],
        ["Feature sparsity",       f"{sparsity:.1%}"],
        ["Unique disease labels",  str(n_diseases)],
        ["Unique countries",       str(n_countries)],
    ]

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#5C6BC0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f4f8")
        else:
            cell.set_facecolor("white")

    ax.set_title("Microbiome Atlas — Dataset Overview", fontsize=15, pad=18, fontweight="bold")
    _save(fig, "dataset_overview")

    for k, v in stats.items():
        logger.info("  %-30s %s", k, v)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 2. Metadata distributions
# ──────────────────────────────────────────────────────────────────────────────
def plot_metadata_distributions(df: pd.DataFrame) -> None:
    """
    2×2 grid:  disease | country | age histogram | gender pie
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Metadata Distributions", fontsize=16, fontweight="bold", y=1.01)

    # — Disease distribution —
    disease_counts = (
        df["disease"]
        .replace(NULL_SENTINELS, np.nan)
        .dropna()
        .value_counts()
        .head(15)
    )
    ax = axes[0, 0]
    bars = ax.barh(disease_counts.index[::-1], disease_counts.values[::-1], color=_PALETTE[0])
    ax.set_title("Disease Distribution (top 15)", fontweight="bold")
    ax.set_xlabel("Sample count")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for bar, v in zip(bars, disease_counts.values[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # — Country distribution —
    country_counts = (
        df["country"]
        .replace(NULL_SENTINELS, np.nan)
        .dropna()
        .value_counts()
        .head(12)
    )
    ax = axes[0, 1]
    ax.barh(country_counts.index[::-1], country_counts.values[::-1], color=_PALETTE[1])
    ax.set_title("Country Distribution (top 12)", fontweight="bold")
    ax.set_xlabel("Sample count")
    ax.spines[["top", "right"]].set_visible(False)

    # — Age distribution —
    age_series = (
        pd.to_numeric(
            df["age"].replace(NULL_SENTINELS, np.nan),
            errors="coerce",
        ).dropna()
    )
    ax = axes[1, 0]
    ax.hist(age_series, bins=30, color=_PALETTE[2], edgecolor="white", linewidth=0.6)
    ax.set_title("Age Distribution", fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.axvline(age_series.median(), color="#d62728", linestyle="--", linewidth=1.2,
               label=f"Median: {age_series.median():.0f}")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # — Gender distribution —
    gender_map = {"female": "Female", "male": "Male"}
    gender_series = (
        df["gender"]
        .str.strip()
        .str.lower()
        .map(gender_map)
        .dropna()
    )
    gender_counts = gender_series.value_counts()
    ax = axes[1, 1]
    wedge_colors = [_PALETTE[3], _PALETTE[4]]
    ax.pie(
        gender_counts.values,
        labels=gender_counts.index,
        autopct="%1.1f%%",
        colors=wedge_colors[:len(gender_counts)],
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11},
    )
    ax.set_title("Gender Distribution (known)", fontweight="bold")

    plt.tight_layout()
    _save(fig, "metadata_distributions")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Missing Value Analysis
# ──────────────────────────────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame, meta_cols: list[str]) -> None:
    """
    Bar chart showing % missing values for each core metadata column.
    A value is considered missing if it is NaN or a null sentinel.
    """
    missing_pct: dict[str, float] = {}
    for col in meta_cols:
        if col not in df.columns:
            continue
        n_missing = (
            df[col]
            .replace(NULL_SENTINELS, np.nan)
            .isna()
            .sum()
        )
        missing_pct[col] = round(100 * n_missing / len(df), 2)

    series = pd.Series(missing_pct).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#ef5350" if v > 30 else "#FFA726" if v > 10 else "#66BB6A"
              for v in series.values]
    bars = ax.bar(series.index, series.values, color=colors, edgecolor="white")

    for bar, val in zip(bars, series.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_title("Missing Values — Core Metadata Columns", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Missing")
    ax.set_ylim(0, max(series.values) * 1.18 + 2)
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#ef5350", label=">30% missing"),
        Patch(facecolor="#FFA726", label="10–30% missing"),
        Patch(facecolor="#66BB6A", label="<10% missing"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="upper right")

    plt.tight_layout()
    _save(fig, "missing_values")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Top Microbial Abundance
# ──────────────────────────────────────────────────────────────────────────────
def _parse_species_label(full_name: str) -> str:
    """
    Extract the most specific taxonomic label from a pipe-delimited lineage string.
    e.g. 'k__Bacteria|p__Firmicutes|...|s__Faecalibacterium_prausnitzii'
         → 'Faecalibacterium prausnitzii'
    """
    parts = full_name.split("|")
    for part in reversed(parts):
        level, _, name = part.partition("__")
        name = name.strip()
        if name and name.lower() not in ("", "unclassified", "unknown"):
            return name.replace("_", " ")
    return full_name


def plot_top_microbes(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Horizontal bar chart of the top-N microbes by mean relative abundance.
    """
    n = EDA["top_microbes_n"]
    mean_abund = df[feature_cols].mean().sort_values(ascending=False).head(n)
    labels     = [_parse_species_label(f) for f in mean_abund.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n))
    bars = ax.barh(labels[::-1], mean_abund.values[::-1], color=colors[::-1])

    for bar, val in zip(bars, mean_abund.values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_title(f"Top {n} Microbes by Mean Relative Abundance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean relative abundance (across all samples)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "top_microbes")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Most Variable Features
# ──────────────────────────────────────────────────────────────────────────────
def plot_most_variable_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Bar chart of top-N features by coefficient of variation (CV).
    CV = std / mean — identifies biologically informative features.
    """
    n = EDA["top_variable_n"]
    feat_df = df[feature_cols]
    means   = feat_df.mean()
    stds    = feat_df.std()

    # Avoid division by zero
    cv = (stds / means.replace(0, np.nan)).dropna().sort_values(ascending=False).head(n)
    labels = [_parse_species_label(f) for f in cv.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, n))
    ax.barh(labels[::-1], cv.values[::-1], color=colors[::-1])
    ax.set_title(f"Top {n} Most Variable Microbial Features (by CV)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient of Variation (std / mean)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "most_variable_features")


# ──────────────────────────────────────────────────────────────────────────────
# 6. PCA Explained Variance
# ──────────────────────────────────────────────────────────────────────────────
def plot_pca_variance(X_scaled: np.ndarray) -> None:
    """
    Scree plot + cumulative explained variance for the first 50 PCA components.
    """
    n_components = min(PCA_PARAMS["n_components"], X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(X_scaled)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PCA Explained Variance Analysis", fontsize=14, fontweight="bold")

    # Scree plot
    ax1.bar(range(1, len(evr) + 1), evr * 100, color=_PALETTE[0], alpha=0.85)
    ax1.set_title("Per-Component Explained Variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.spines[["top", "right"]].set_visible(False)

    # Cumulative
    ax2.plot(range(1, len(cumulative) + 1), cumulative * 100,
             color=_PALETTE[2], linewidth=2, marker="o", markersize=3)
    for threshold in [50, 70, 90]:
        idx = np.searchsorted(cumulative, threshold / 100)
        if idx < len(cumulative):
            ax2.axhline(threshold, linestyle="--", linewidth=0.8, color="gray", alpha=0.7)
            ax2.text(len(cumulative) * 0.65, threshold + 0.8,
                     f"{threshold}% @ PC{idx+1}", fontsize=8, color="gray")
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_ylim(0, 101)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "pca_variance")
    logger.info("PCA: first 10 components explain %.1f%% variance",
                cumulative[9] * 100 if len(cumulative) >= 10 else cumulative[-1] * 100)


# ──────────────────────────────────────────────────────────────────────────────
# 7. UMAP Preview
# ──────────────────────────────────────────────────────────────────────────────
def plot_umap_preview(df: pd.DataFrame, X_scaled: np.ndarray) -> None:
    """
    Compute a quick 2-D UMAP embedding and colour points by disease label.
    """
    try:
        import umap as umap_lib
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP preview.")
        return

    logger.info("Computing UMAP for preview (this may take ~2 min) …")
    reducer = umap_lib.UMAP(**UMAP_PARAMS)
    embedding = reducer.fit_transform(X_scaled)

    disease = df["disease"].replace(NULL_SENTINELS, "unknown").fillna("unknown")
    unique_diseases = disease.unique()
    color_map = {d: _PALETTE[i % len(_PALETTE)] for i, d in enumerate(unique_diseases)}
    colors = [color_map[d] for d in disease]

    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colors, s=12, alpha=0.65, linewidths=0,
    )
    ax.set_title("UMAP Preview — coloured by Disease Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[d],
               markersize=8, label=d)
        for d in sorted(unique_diseases)[:14]
    ]
    ax.legend(handles=handles, title="Disease", fontsize=7,
              loc="upper right", framealpha=0.85)

    plt.tight_layout()
    _save(fig, "umap_preview")


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame, feature_cols: list[str], meta_cols: list[str]) -> dict[str, Any]:
    """
    Run the full EDA pipeline on the already-loaded DataFrame.

    Parameters
    ----------
    df           : raw DataFrame loaded from abundance CSV
    feature_cols : list of microbial feature column names (k__ prefix)
    meta_cols    : list of metadata column names

    Returns
    -------
    overview_stats : dict with basic dataset statistics
    """
    ensure_dirs()
    logger.info("═" * 55)
    logger.info("Starting EDA")
    logger.info("═" * 55)

    logger.info("1/7  Dataset overview …")
    stats = plot_dataset_overview(df, feature_cols, meta_cols)

    logger.info("2/7  Metadata distributions …")
    plot_metadata_distributions(df)

    logger.info("3/7  Missing values …")
    plot_missing_values(df, meta_cols)

    logger.info("4/7  Top microbial abundances …")
    plot_top_microbes(df, feature_cols)

    logger.info("5/7  Most variable features …")
    plot_most_variable_features(df, feature_cols)

    # Quick standard-scaled matrix for PCA + UMAP previews
    logger.info("6/7  PCA variance (scaling for preview) …")
    feat_matrix = df[feature_cols].fillna(0).values.astype(np.float64)
    scaler_tmp  = StandardScaler()
    X_scaled_tmp = scaler_tmp.fit_transform(feat_matrix)
    plot_pca_variance(X_scaled_tmp)

    logger.info("7/7  UMAP preview …")
    plot_umap_preview(df, X_scaled_tmp)

    logger.info("EDA complete. Figures saved to %s", DOCS_FIGURES_DIR)
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ml.preprocessing import load_raw_data, identify_columns

    logger.info("Loading raw data …")
    raw_df = load_raw_data()
    feat_cols, m_cols = identify_columns(raw_df)
    run_eda(raw_df, feat_cols, m_cols)