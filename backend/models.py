"""
backend/models.py
=================
SQLAlchemy ORM table definitions for the Microbiome Atlas.

Tables
------
  samples            — one row per raw microbiome sample
  fingerprints       — ML-derived per-sample profile (FK → samples)
  embeddings         — UMAP + PCA 2-D coordinates (FK → samples)
  cluster_summaries  — per-cluster biological statistics
  feature_importances— top Random-Forest feature importance scores

Design decisions
----------------
- VARCHAR lengths are generous to handle long microbial identifiers.
- JSON columns (MySQL 5.7.8+) store the dominant_microbes and metrics
  blobs without needing a separate normalisation table.
- All FKs use ON DELETE CASCADE so that deleting a Sample row
  automatically removes its dependent fingerprint, embedding, etc.
- Indexes are added on every column that appears in WHERE / ORDER BY
  clauses in the API routes.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from backend.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# Samples
# ─────────────────────────────────────────────────────────────────────────────

class Sample(Base):
    """
    One row per microbiome sample from the original abundance CSV.

    sample_id is the natural key from the dataset (e.g. 'H10', 'MH0002').
    Because the upstream dataset contains duplicate sampleIDs across
    different datasets, we use a surrogate integer PK internally and
    expose sample_id as a non-unique indexed column.
    """

    __tablename__ = "samples"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    sample_id  = Column(String(128), nullable=False, index=True)
    subject_id = Column(String(128), nullable=True)
    dataset    = Column(String(128), nullable=True)
    bodysite   = Column(String(64),  nullable=True)
    disease    = Column(String(128), nullable=True, index=True)
    age        = Column(Float,       nullable=True)
    gender     = Column(String(16),  nullable=True, index=True)
    country    = Column(String(64),  nullable=True, index=True)
    bmi        = Column(Float,       nullable=True)

    # Relationships
    fingerprint = relationship(
        "Fingerprint", back_populates="sample", uselist=False, cascade="all, delete-orphan"
    )
    embedding = relationship(
        "Embedding", back_populates="sample", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_samples_disease_country", "disease", "country"),
    )

    def __repr__(self) -> str:
        return f"<Sample id={self.id} sample_id={self.sample_id!r} disease={self.disease!r}>"


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprints
# ─────────────────────────────────────────────────────────────────────────────

class Fingerprint(Base):
    """
    ML-derived microbiome profile for each sample.

    dominant_microbes : JSON list of {name, abundance, phylum, genus, species}
    phylum_profile    : JSON dict of {phylum_name: relative_abundance}
    """

    __tablename__ = "fingerprints"

    id               = Column(Integer,     primary_key=True, autoincrement=True)
    sample_pk        = Column(Integer,     ForeignKey("samples.id", ondelete="CASCADE"),
                              nullable=False, unique=True, index=True)
    sample_id        = Column(String(128), nullable=False, index=True)
    cluster_id       = Column(Integer,     nullable=False, index=True)
    umap_x           = Column(Float,       nullable=False)
    umap_y           = Column(Float,       nullable=False)
    diversity_score  = Column(Float,       nullable=False)
    diversity_class  = Column(String(16),  nullable=False)   # Low | Medium | High
    novelty_score    = Column(Float,       nullable=False)
    novelty_class    = Column(String(32),  nullable=False)   # Common | Moderately Unique | Highly Unique
    n_dominant       = Column(Integer,     nullable=False, default=0)
    dominant_microbes = Column(JSON,       nullable=True)    # list[dict]
    phylum_profile    = Column(JSON,       nullable=True)    # dict[str, float]

    # Relationship
    sample = relationship("Sample", back_populates="fingerprint")

    __table_args__ = (
        Index("ix_fingerprints_cluster_diversity", "cluster_id", "diversity_class"),
    )

    def __repr__(self) -> str:
        return (
            f"<Fingerprint sample_id={self.sample_id!r} "
            f"cluster={self.cluster_id} diversity={self.diversity_class!r}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class Embedding(Base):
    """
    2-D UMAP and PCA coordinates for the Microbiome Atlas visualisation.
    """

    __tablename__ = "embeddings"

    id         = Column(Integer,     primary_key=True, autoincrement=True)
    sample_pk  = Column(Integer,     ForeignKey("samples.id", ondelete="CASCADE"),
                        nullable=False, unique=True, index=True)
    sample_id  = Column(String(128), nullable=False, index=True)
    cluster_id = Column(Integer,     nullable=False, index=True)
    umap_x     = Column(Float,       nullable=False)
    umap_y     = Column(Float,       nullable=False)
    pca_x      = Column(Float,       nullable=False)
    pca_y      = Column(Float,       nullable=False)

    # Metadata columns denormalised here for fast atlas queries
    # (avoids joining to samples table on every atlas page load)
    disease    = Column(String(128), nullable=True)
    country    = Column(String(64),  nullable=True)
    gender     = Column(String(16),  nullable=True)
    age        = Column(Float,       nullable=True)

    # Relationship
    sample = relationship("Sample", back_populates="embedding")

    def __repr__(self) -> str:
        return f"<Embedding sample_id={self.sample_id!r} cluster={self.cluster_id}>"


# ─────────────────────────────────────────────────────────────────────────────
# Cluster Summaries
# ─────────────────────────────────────────────────────────────────────────────

class ClusterSummary(Base):
    """
    Per-cluster biological and statistical summary generated by the ML pipeline.

    dominant_microbes : JSON dict {microbe_name: mean_abundance}
    dominant_phyla    : JSON dict {phylum_name: relative_abundance}
    diversity_stats   : JSON dict {mean, std, min, max, class_distribution}
    novelty_stats     : JSON dict {mean, class_distribution}
    metadata_dist     : JSON dict {disease: {}, country: {}, gender: {}}
    age_stats         : JSON dict {mean, median, std, min, max} or null
    top_features      : JSON list [{rank, display_name, phylum, importance_score}]
    pipeline_scores   : JSON dict {pipeline, silhouette, davies_bouldin}
    """

    __tablename__ = "cluster_summaries"

    id               = Column(Integer,     primary_key=True, autoincrement=True)
    cluster_id       = Column(Integer,     nullable=False, unique=True, index=True)
    label            = Column(String(64),  nullable=False)    # e.g. "Cluster 0", "Noise"
    size             = Column(Integer,     nullable=False)
    is_noise         = Column(Boolean,     nullable=False, default=False)
    narrative        = Column(Text,        nullable=True)

    dominant_microbes = Column(JSON, nullable=True)
    dominant_phyla    = Column(JSON, nullable=True)
    diversity_stats   = Column(JSON, nullable=True)
    novelty_stats     = Column(JSON, nullable=True)
    metadata_dist     = Column(JSON, nullable=True)
    age_stats         = Column(JSON, nullable=True)
    top_features      = Column(JSON, nullable=True)
    pipeline_scores   = Column(JSON, nullable=True)

    # Convenience scalar metrics for sorting / filtering without JSON extraction
    diversity_mean    = Column(Float, nullable=True)
    silhouette_score  = Column(Float, nullable=True)

    # Relationships
    feature_importances = relationship(
        "FeatureImportance", back_populates="cluster", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ClusterSummary cluster_id={self.cluster_id} size={self.size}>"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importances
# ─────────────────────────────────────────────────────────────────────────────

class FeatureImportance(Base):
    """
    Random Forest feature importance scores per cluster.

    One row per (cluster_id, feature) combination.
    cluster_id = -1 represents the global (all-cluster) importance ranking.
    """

    __tablename__ = "feature_importances"

    id               = Column(Integer,     primary_key=True, autoincrement=True)
    cluster_id       = Column(Integer,     ForeignKey("cluster_summaries.cluster_id",
                                                       ondelete="CASCADE"),
                              nullable=False, index=True)
    rank             = Column(Integer,     nullable=False)
    feature_col      = Column(Text,        nullable=False)   # full k__ lineage string
    display_name     = Column(String(256), nullable=False, index=True)
    phylum           = Column(String(128), nullable=True,  index=True)
    importance_score = Column(Float,       nullable=False)
    importance_pct   = Column(Float,       nullable=True)
    importance_std   = Column(Float,       nullable=True)

    # Relationship
    cluster = relationship("ClusterSummary", back_populates="feature_importances")

    __table_args__ = (
        Index("ix_fi_cluster_rank", "cluster_id", "rank"),
        UniqueConstraint("cluster_id", "rank", name="uq_fi_cluster_rank"),
    )

    def __repr__(self) -> str:
        return (
            f"<FeatureImportance cluster={self.cluster_id} "
            f"rank={self.rank} feature={self.display_name!r}>"
        )