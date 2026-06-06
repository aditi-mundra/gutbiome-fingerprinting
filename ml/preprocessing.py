"""
ml/preprocessing.py
===================
Loads raw abundance data, separates metadata from microbial features,
cleans null sentinels, applies Centered Log-Ratio normalisation (CLR)
on microbial abundances, and persists processed artefacts.

CLR is the recommended transform for compositional microbiome data:
    clr(x_i) = log(x_i / geometric_mean(x))

A small pseudocount (1e-6) is added before log to handle zeros.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ABUNDANCE_CSV,
    CORE_META_COLS,
    DATA_PROCESSED_DIR,
    FEATURE_COLUMNS_TXT,
    FEATURE_PREFIX,
    FEATURES_NORMALIZED_NPY,
    GENDER_MAP,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    METADATA_CSV,
    NULL_SENTINELS,
    PREPROCESSING,
    RANDOM_SEED,
    SCALER_JOBLIB,
    ensure_dirs,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: Path = ABUNDANCE_CSV) -> pd.DataFrame:
    """
    Load the raw abundance CSV with robust type inference.

    Raises
    ------
    FileNotFoundError if the CSV does not exist.
    ValueError if required columns are absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Abundance CSV not found at {path}. "
            "Place abundance_stoolsubset.csv in data/raw/."
        )

    logger.info("Loading raw data from %s …", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d samples × %d columns", *df.shape)

    # Validate required columns
    required = {"sampleID", "disease", "country", "gender"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from CSV: {missing}")

    return df


def identify_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """
    Split columns into microbial feature columns and metadata columns.

    Returns
    -------
    feature_cols : columns whose name starts with FEATURE_PREFIX ('k__')
    meta_cols    : all other columns
    """
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    meta_cols    = [c for c in df.columns if not c.startswith(FEATURE_PREFIX)]

    if not feature_cols:
        raise ValueError(
            f"No microbial feature columns found "
            f"(expected prefix '{FEATURE_PREFIX}'). "
            "Check your CSV column names."
        )

    logger.info("Identified %d microbial features and %d metadata columns",
                len(feature_cols), len(meta_cols))
    return feature_cols, meta_cols


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean core metadata columns:
    - Replace null sentinels with NaN
    - Coerce age and bmi to numeric
    - Normalise gender values
    - Strip whitespace from string fields

    Returns a copy of the DataFrame with only the columns in CORE_META_COLS
    that actually exist in df.
    """
    existing = [c for c in CORE_META_COLS if c in df.columns]
    meta = df[existing].copy()

    # Replace sentinels across all string-typed columns
    for col in meta.select_dtypes(include=["object"]).columns:
        meta[col] = meta[col].astype(str).str.strip()
        meta[col] = meta[col].replace(NULL_SENTINELS, np.nan)

    # Coerce numeric columns
    for col in ["age", "bmi"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")

    # Normalise gender
    if "gender" in meta.columns:
        meta["gender"] = (
            meta["gender"]
            .str.strip()
            .str.lower()
            .map(GENDER_MAP)
        )

    logger.info("Metadata cleaned. Shape: %s", meta.shape)
    return meta


def filter_low_prevalence(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_prevalence: float,
) -> list[str]:
    """
    Drop microbial features that appear (non-zero) in fewer than
    `min_prevalence` fraction of samples.

    Returns
    -------
    Filtered feature column list.
    """
    feat_matrix = df[feature_cols].astype(np.float64)
    prevalence  = (feat_matrix > 0).mean(axis=0)
    kept        = prevalence[prevalence >= min_prevalence].index.tolist()

    n_dropped = len(feature_cols) - len(kept)
    logger.info(
        "Prevalence filter (>= %.1f%%): kept %d / %d features, dropped %d",
        min_prevalence * 100,
        len(kept),
        len(feature_cols),
        n_dropped,
    )
    return kept


def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """
    Centered Log-Ratio (CLR) transform for compositional microbiome data.

    Parameters
    ----------
    X           : (n_samples, n_features) abundance matrix, values >= 0
    pseudocount : small constant added before log to handle zeros

    Returns
    -------
    X_clr : (n_samples, n_features) CLR-transformed matrix
    """
    X_pseudo = X + pseudocount
    # Geometric mean per sample (row)
    log_X    = np.log(X_pseudo)
    geo_mean = log_X.mean(axis=1, keepdims=True)   # equivalent to log(gmean)
    X_clr    = log_X - geo_mean
    return X_clr


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler_type: str = "clr",
) -> Tuple[np.ndarray, StandardScaler | None]:
    """
    Normalise microbial abundance features.

    Pipeline:
      1. Fill NaN with 0  (absent microbe = 0 abundance)
      2. CLR transform    (or StandardScaler if scaler_type == "standard")
      3. StandardScaler   (zero-mean, unit-variance) — always applied after CLR

    Returns
    -------
    X_norm  : (n_samples, n_features) normalised matrix
    scaler  : fitted StandardScaler (saved to disk for inference)
    """
    X_raw = df[feature_cols].fillna(0).values.astype(np.float64)

    if scaler_type == "clr":
        logger.info("Applying CLR transform …")
        X_transformed = clr_transform(X_raw)
    else:
        logger.info("Skipping CLR (scaler_type='%s') …", scaler_type)
        X_transformed = X_raw

    logger.info("Applying StandardScaler …")
    scaler  = StandardScaler()
    X_norm  = scaler.fit_transform(X_transformed)

    logger.info("Normalization complete. Shape: %s, Mean≈%.4f, Std≈%.4f",
                X_norm.shape, X_norm.mean(), X_norm.std())
    return X_norm, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Main preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing() -> Tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Full preprocessing pipeline:
      1. Load raw CSV
      2. Identify columns
      3. Clean metadata
      4. Filter low-prevalence features
      5. Normalise features
      6. Persist all artefacts

    Returns
    -------
    X_norm       : (n_samples, n_features) normalised feature matrix
    metadata     : cleaned metadata DataFrame (includes sampleID)
    feature_cols : list of retained feature column names
    """
    ensure_dirs()

    # 1. Load
    df = load_raw_data()

    # 2. Split columns
    feature_cols, meta_cols = identify_columns(df)

    # 3. Clean metadata
    metadata = clean_metadata(df)

    # Ensure sampleID is present and has no nulls (used as primary key)
    if "sampleID" not in metadata.columns:
        raise ValueError("'sampleID' column is required but not found in metadata.")
    n_null_ids = metadata["sampleID"].isna().sum()
    if n_null_ids > 0:
        logger.warning("%d samples have null sampleID — dropping them.", n_null_ids)
        valid_mask = metadata["sampleID"].notna()
        metadata   = metadata[valid_mask].reset_index(drop=True)
        df         = df[valid_mask].reset_index(drop=True)

    # 4. Prevalence filter
    min_prev     = PREPROCESSING["min_prevalence"]
    feature_cols = filter_low_prevalence(df, feature_cols, min_prev)

    # 5. Normalise
    scaler_type  = PREPROCESSING["scaler"]
    X_norm, scaler = normalize_features(df, feature_cols, scaler_type)

    # 6. Persist artefacts
    logger.info("Persisting preprocessed artefacts …")

    np.save(FEATURES_NORMALIZED_NPY, X_norm)
    logger.info("  Saved normalised features → %s", FEATURES_NORMALIZED_NPY)

    FEATURE_COLUMNS_TXT.write_text("\n".join(feature_cols), encoding="utf-8")
    logger.info("  Saved feature column list  → %s", FEATURE_COLUMNS_TXT)

    metadata.to_csv(METADATA_CSV, index=False)
    logger.info("  Saved cleaned metadata     → %s", METADATA_CSV)

    joblib.dump(scaler, SCALER_JOBLIB)
    logger.info("  Saved scaler               → %s", SCALER_JOBLIB)

    logger.info("Preprocessing complete. Features: %d, Samples: %d",
                X_norm.shape[1], X_norm.shape[0])

    return X_norm, metadata, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# Convenience loaders (used by downstream modules)
# ──────────────────────────────────────────────────────────────────────────────

def load_processed() -> Tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Load already-processed artefacts from disk without re-running preprocessing.

    Raises
    ------
    FileNotFoundError if artefacts are not found (run run_preprocessing first).
    """
    for path in [FEATURES_NORMALIZED_NPY, METADATA_CSV, FEATURE_COLUMNS_TXT]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Processed artefact not found: {path}. "
                "Run preprocessing.run_preprocessing() first."
            )

    X_norm       = np.load(FEATURES_NORMALIZED_NPY)
    metadata     = pd.read_csv(METADATA_CSV, low_memory=False)
    feature_cols = FEATURE_COLUMNS_TXT.read_text(encoding="utf-8").splitlines()

    logger.info("Loaded processed data: X=%s, metadata=%s, features=%d",
                X_norm.shape, metadata.shape, len(feature_cols))
    return X_norm, metadata, feature_cols


if __name__ == "__main__":
    X, meta, feats = run_preprocessing()
    logger.info("Done. X shape: %s", X.shape)