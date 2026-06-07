"""
backend/schemas.py
==================
Pydantic v2 schemas for request validation and response serialisation.

Naming convention:
  <Entity>Base    — shared fields
  <Entity>Create  — fields required when creating (POST body)
  <Entity>Out     — fields returned in API responses
  <Entity>Detail  — extended response including nested objects

All response schemas use model_config = ConfigDict(from_attributes=True)
so they can be constructed directly from SQLAlchemy ORM objects via
Pydantic's model_validate().
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# Shared / primitives
# ─────────────────────────────────────────────────────────────────────────────

class PaginationMeta(BaseModel):
    """Pagination metadata returned alongside paginated results."""
    total:   int = Field(..., description="Total matching records")
    page:    int = Field(..., description="Current page (1-based)")
    limit:   int = Field(..., description="Items per page")
    pages:   int = Field(..., description="Total number of pages")


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint
# ─────────────────────────────────────────────────────────────────────────────

class FingerprintOut(BaseModel):
    """Microbiome fingerprint for a single sample."""

    model_config = ConfigDict(from_attributes=True)

    sample_id:         str
    cluster_id:        int
    umap_x:            float
    umap_y:            float
    diversity_score:   float
    diversity_class:   str          # Low | Medium | High
    novelty_score:     float
    novelty_class:     str          # Common | Moderately Unique | Highly Unique
    n_dominant:        int
    dominant_microbes: list[dict[str, Any]] | None = None
    phylum_profile:    dict[str, float]     | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingOut(BaseModel):
    """2-D atlas coordinates for a single sample."""

    model_config = ConfigDict(from_attributes=True)

    sample_id:  str
    cluster_id: int
    umap_x:     float
    umap_y:     float
    pca_x:      float
    pca_y:      float
    disease:    str | None = None
    country:    str | None = None
    gender:     str | None = None
    age:        float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Sample
# ─────────────────────────────────────────────────────────────────────────────

class SampleOut(BaseModel):
    """Slim sample row for list/pagination endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id:         int
    sample_id:  str
    subject_id: str | None = None
    dataset:    str | None = None
    disease:    str | None = None
    age:        float | None = None
    gender:     str | None = None
    country:    str | None = None
    bmi:        float | None = None
    bodysite:   str | None = None


class SampleDetail(SampleOut):
    """Full sample detail including nested fingerprint and embedding."""

    fingerprint: FingerprintOut | None = None
    embedding:   EmbeddingOut   | None = None


class SampleListResponse(BaseModel):
    """Paginated list of samples."""
    data:       list[SampleOut]
    pagination: PaginationMeta


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

class FeatureImportanceOut(BaseModel):
    """Single feature importance row."""

    model_config = ConfigDict(from_attributes=True)

    rank:             int
    display_name:     str
    phylum:           str | None = None
    importance_score: float
    importance_pct:   float | None = None
    importance_std:   float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Cluster Summary
# ─────────────────────────────────────────────────────────────────────────────

class ClusterSummaryOut(BaseModel):
    """Cluster summary for the clusters list endpoint."""

    model_config = ConfigDict(from_attributes=True)

    cluster_id:       int
    label:            str
    size:             int
    is_noise:         bool
    narrative:        str | None = None
    diversity_mean:   float | None = None
    silhouette_score: float | None = None
    dominant_phyla:   dict[str, float]      | None = None
    diversity_stats:  dict[str, Any]        | None = None
    novelty_stats:    dict[str, Any]        | None = None
    metadata_dist:    dict[str, Any]        | None = None
    age_stats:        dict[str, Any]        | None = None
    pipeline_scores:  dict[str, Any]        | None = None
    top_features:     list[dict[str, Any]]  | None = None
    dominant_microbes: dict[str, float]     | None = None


class ClusterListResponse(BaseModel):
    """All cluster summaries + pipeline metadata."""
    pipeline:  str
    n_clusters: int
    n_noise:   int
    clusters:  list[ClusterSummaryOut]


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Network
# ─────────────────────────────────────────────────────────────────────────────

class NetworkNode(BaseModel):
    """A node in the similarity graph."""
    id:         str
    cluster:    int
    umap_x:     float
    umap_y:     float
    disease:    str | None = None
    country:    str | None = None
    gender:     str | None = None


class NetworkEdge(BaseModel):
    """A weighted edge in the similarity graph."""
    source: str
    target: str
    weight: float


class NetworkResponse(BaseModel):
    """Full similarity network for Plotly rendering."""
    n_nodes:   int
    n_edges:   int
    threshold: float
    nodes:     list[NetworkNode]
    edges:     list[NetworkEdge]


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint Prediction (POST /api/v1/fingerprint/predict)
# ─────────────────────────────────────────────────────────────────────────────

class FingerprintPredictRequest(BaseModel):
    """
    Body for the real-time fingerprint prediction endpoint.

    abundances : dict mapping microbial feature column names (k__ lineage
                 strings) to their relative abundance values (floats >= 0).
                 Features not in the trained feature set are silently ignored.
                 Missing features default to 0.0.

    Example:
        {
          "abundances": {
            "k__Bacteria|p__Firmicutes|...|s__Faecalibacterium_prausnitzii": 0.152,
            "k__Bacteria|p__Bacteroidetes": 0.082
          }
        }
    """
    abundances: dict[str, float] = Field(
        ...,
        description="Map of microbial feature name → relative abundance value.",
        min_length=1,
    )


class FingerprintPredictResponse(BaseModel):
    """
    Response for the real-time fingerprint prediction endpoint.

    cluster_id      : assigned cluster (-1 = noise / no cluster)
    cluster_label   : human-readable label (e.g. "Cluster 2")
    umap_x, umap_y  : 2-D atlas position of the predicted sample
    diversity_score : Shannon entropy of the provided abundance vector
    diversity_class : Low | Medium | High
    novelty_score   : distance from cluster centroid, normalised 0-1
    novelty_class   : Common | Moderately Unique | Highly Unique
    top_microbes    : top-10 species from the input abundances
    phylum_profile  : phylum-level abundance breakdown
    n_features_used : how many of the input features matched the trained set
    n_features_total: total features the model was trained on
    pipeline        : which ML pipeline was used (e.g. 'umap_hdbscan')
    """
    cluster_id:       int
    cluster_label:    str
    umap_x:           float
    umap_y:           float
    diversity_score:  float
    diversity_class:  str
    novelty_score:    float
    novelty_class:    str
    top_microbes:     list[dict[str, Any]]
    phylum_profile:   dict[str, float]
    n_features_used:  int
    n_features_total: int
    pipeline:         str


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    db_online:   bool
    pipeline:    str
    n_samples:   int
    n_clusters:  int
    models_loaded: bool