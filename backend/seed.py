"""
backend/seed.py
===============
Standalone seeding script.

Reads ML pipeline output artifacts from data/results/ and data/processed/,
then batch-inserts them into the SQLite database via SQLAlchemy ORM.

Run from the project root:
    python -m backend.seed
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

# Allow running as `python -m backend.seed` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.database import SessionLocal, create_tables
from backend.models import (
    ClusterSummary,
    Embedding,
    FeatureImportance,
    Fingerprint,
    Sample,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("seed")

# ── Configuration ─────────────────────────────────────────────────────────────
BATCH_SIZE = 250    # rows per bulk INSERT


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nan_to_none(value: object) -> object:
    """Convert float NaN / pandas NA to Python None for SQLite compatibility."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _safe_json(value: object) -> str | None:
    """
    Ensure complex dicts/lists/strings are converted into perfectly formatted
    clean JSON text strings for SQLite storage, handling double-serialized data.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    
    # If it's a string, it might be double-serialized or contain escaped characters
    if isinstance(value, str):
        value_stripped = value.strip()
        
        # Check if it looks like a stringified JSON object or list
        if (value_stripped.startswith('{') and value_stripped.endswith('}')) or \
           (value_stripped.startswith('[') and value_stripped.endswith(']')):
            try:
                # Clean up nested string evaluations if any exist
                parsed = json.loads(value_stripped)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                return json.dumps(parsed)
            except (json.JSONDecodeError, ValueError):
                pass
                
        # Handle string fields that are explicitly double-escaped or corrupted
        if '\\"' in value:
            try:
                # Clean backslashes out and force re-evaluation
                cleaned = value.replace('\\"', '"').replace('\\\\', '\\')
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                parsed = json.loads(cleaned)
                return json.dumps(parsed)
            except (json.JSONDecodeError, ValueError):
                pass
                
        return json.dumps(value)
            
    # Convert native Python lists/dicts/numpy types to clean text strings
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return None


def _bulk_insert(db: Session, objects: list, label: str) -> None:
    """Batch-insert a list of ORM objects and commit each batch."""
    total = len(objects)
    for start in range(0, total, BATCH_SIZE):
        batch = objects[start : start + BATCH_SIZE]
        db.bulk_save_objects(batch)
        db.commit()
        logger.info("  %s: inserted %d / %d", label, min(start + BATCH_SIZE, total), total)


# ── Seed functions ────────────────────────────────────────────────────────────

def seed_samples(db: Session) -> dict[str, int]:
    """Load metadata.csv and insert into `samples`."""
    logger.info("Seeding samples …")

    db.query(Sample).delete()
    db.commit()

    path = settings.metadata_csv
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.info("  Loaded %d rows from metadata.csv", len(df))

    objects: list[Sample] = []
    for _, row in df.iterrows():
        objects.append(Sample(
            sample_id  = str(row.get("sampleID", "") or ""),
            subject_id = _nan_to_none(row.get("subjectID")),
            dataset    = _nan_to_none(row.get("dataset_name")),
            bodysite   = _nan_to_none(row.get("bodysite")),
            disease    = _nan_to_none(row.get("disease")),
            age        = _nan_to_none(row.get("age")),
            gender     = _nan_to_none(row.get("gender")),
            country    = _nan_to_none(row.get("country")),
            bmi        = _nan_to_none(row.get("bmi")),
        ))

    _bulk_insert(db, objects, "samples")

    rows = db.query(Sample.id, Sample.sample_id).order_by(Sample.id).all()
    sample_id_to_pk: dict[str, int] = {}
    for pk, sid in rows:
        if sid not in sample_id_to_pk:
            sample_id_to_pk[sid] = pk

    logger.info("  Samples seeded: %d rows (%d unique sampleIDs)",
                len(objects), len(sample_id_to_pk))
    return sample_id_to_pk


def seed_fingerprints(db: Session, sample_id_to_pk: dict[str, int]) -> None:
    """Load fingerprints.csv and insert into `fingerprints`."""
    logger.info("Seeding fingerprints …")

    db.query(Fingerprint).delete()
    db.commit()

    path = settings.fingerprints_csv
    if not path.exists():
        raise FileNotFoundError(f"fingerprints.csv not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.info("  Loaded %d rows from fingerprints.csv", len(df))

    missing_pks = 0
    objects: list[Fingerprint] = []
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        pk  = sample_id_to_pk.get(sid)
        if pk is None:
            missing_pks += 1
            continue

        objects.append(Fingerprint(
            sample_pk         = pk,
            sample_id         = sid,
            cluster_id        = int(row["cluster_id"]),
            umap_x            = float(row["umap_x"]),
            umap_y            = float(row["umap_y"]),
            diversity_score   = float(row["diversity_score"]),
            diversity_class   = str(row["diversity_class"]),
            novelty_score     = float(row["novelty_score"]),
            novelty_class     = str(row["novelty_class"]),
            n_dominant        = int(row.get("n_dominant", 0)),
            dominant_microbes = _safe_json(row.get("dominant_microbes")),
            phylum_profile    = _safe_json(row.get("phylum_profile")),
        ))

    if missing_pks:
        logger.warning("  %d fingerprint rows had no matching Sample PK (skipped)", missing_pks)

    _bulk_insert(db, objects, "fingerprints")
    logger.info("  Fingerprints seeded: %d rows", len(objects))


def seed_embeddings(db: Session, sample_id_to_pk: dict[str, int]) -> None:
    """Load embeddings.csv and insert into `embeddings`."""
    logger.info("Seeding embeddings …")

    db.query(Embedding).delete()
    db.commit()

    path = settings.embeddings_csv
    if not path.exists():
        raise FileNotFoundError(f"embeddings.csv not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.info("  Loaded %d rows from embeddings.csv", len(df))

    missing_pks = 0
    objects: list[Embedding] = []
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        pk  = sample_id_to_pk.get(sid)
        if pk is None:
            missing_pks += 1
            continue

        objects.append(Embedding(
            sample_pk  = pk,
            sample_id  = sid,
            cluster_id = int(row["cluster_id"]),
            umap_x     = float(row["umap_x"]),
            umap_y     = float(row["umap_y"]),
            pca_x      = float(row["pca_x"]),
            pca_y      = float(row["pca_y"]),
            disease    = _nan_to_none(row.get("disease")),
            country    = _nan_to_none(row.get("country")),
            gender     = _nan_to_none(row.get("gender")),
            age        = _nan_to_none(row.get("age")),
        ))

    if missing_pks:
        logger.warning("  %d embedding rows had no matching Sample PK (skipped)", missing_pks)

    _bulk_insert(db, objects, "embeddings")
    logger.info("  Embeddings seeded: %d rows", len(objects))


def seed_clusters(db: Session) -> None:
    """Load cluster_summary.json and insert into tables."""
    logger.info("Seeding cluster summaries …")

    db.query(FeatureImportance).delete()
    db.query(ClusterSummary).delete()
    db.commit()

    path = settings.cluster_summary_json
    if not path.exists():
        raise FileNotFoundError(f"cluster_summary.json not found: {path}")

    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)

    pipeline_scores = summary.get("pipeline_scores", {})
    clusters_data   = summary.get("clusters", [])
    logger.info("  Loaded %d clusters from cluster_summary.json", len(clusters_data))

    cluster_objects:   list[ClusterSummary]    = []
    importance_objects: list[FeatureImportance] = []

    for c in clusters_data:
        cid      = int(c["cluster_id"])
        is_noise = cid == -1

        cs = ClusterSummary(
            cluster_id       = cid,
            label            = str(c.get("label", f"Cluster {cid}")),
            size             = int(c.get("size", 0)),
            is_noise         = is_noise,
            narrative        = c.get("narrative"),
            dominant_microbes= _safe_json(c.get("top_microbes")),
            dominant_phyla   = _safe_json(c.get("dominant_phyla")),
            diversity_stats  = _safe_json(c.get("diversity")),
            novelty_stats    = _safe_json(c.get("novelty")),
            metadata_dist    = _safe_json(c.get("metadata_distributions")),
            age_stats        = _safe_json(c.get("age_stats")),
            top_features     = _safe_json(c.get("top_driving_features")),
            pipeline_scores  = _safe_json(pipeline_scores) if not is_noise else None,
            diversity_mean   = c.get("diversity", {}).get("mean") if c.get("diversity") else None,
            silhouette_score = (
                float(pipeline_scores["silhouette"])
                if pipeline_scores and not is_noise else None
            ),
        )
        cluster_objects.append(cs)

    _bulk_insert(db, cluster_objects, "cluster_summaries")
    logger.info("  ClusterSummaries seeded: %d rows", len(cluster_objects))

    # Feature importances from per-cluster top_driving_features
    for c in clusters_data:
        cid      = int(c["cluster_id"])
        features = c.get("top_driving_features") or []
        for feat in features:
            importance_objects.append(FeatureImportance(
                cluster_id       = cid,
                rank             = int(feat.get("rank", 0)),
                feature_col      = str(feat.get("feature_col", "")),
                display_name     = str(feat.get("display_name", "")),
                phylum           = feat.get("phylum"),
                importance_score = float(feat.get("importance_score", 0.0)),
                importance_pct   = feat.get("importance_pct"),
                importance_std   = None,
            ))

    # Seed global importances (cluster_id = -99)
    fi_path = settings.feature_importance_csv
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        for _, row in fi_df.iterrows():
            importance_objects.append(FeatureImportance(
                cluster_id       = -99,
                rank             = int(row["rank"]),
                feature_col      = str(row["feature_col"]),
                display_name     = str(row["display_name"]),
                phylum           = _nan_to_none(row.get("phylum")),
                importance_score = float(row["importance_score"]),
                importance_pct   = _nan_to_none(row.get("importance_pct")),
                importance_std   = _nan_to_none(row.get("importance_std")),
            ))
        logger.info("  Global feature importances: %d rows", len(fi_df))

    _bulk_insert(db, importance_objects, "feature_importances")
    logger.info("  FeatureImportances seeded: %d rows", len(importance_objects))


# ── Main ──────────────────────────────────────────────────────────────────────

def run_seed() -> None:
    """Execute the full seeding pipeline in dependency order."""
    import time
    t0 = time.perf_counter()

    logger.info("" * 60)
    logger.info("Microbiome Atlas — Database Seeder")
    logger.info("" * 60)

    logger.info("Creating tables if they do not exist …")
    create_tables()

    db = SessionLocal()
    try:
        sample_id_to_pk = seed_samples(db)
        seed_fingerprints(db, sample_id_to_pk)
        seed_embeddings(db, sample_id_to_pk)
        seed_clusters(db)

    except Exception as exc:
        logger.exception("Seeding failed: %s", exc)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()

    elapsed = time.perf_counter() - t0
    logger.info("" * 60)
    logger.info("Seeding complete in %.1f s", elapsed)
    logger.info("" * 60)


if __name__ == "__main__":
    run_seed()