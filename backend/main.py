"""
backend/main.py
===============
Microbiome Atlas — FastAPI application entry-point.

Startup sequence
----------------
1. Load ML artifacts into memory (scaler, PCA, UMAP, HDBSCAN, feature list,
   UMAP training embeddings for centroid inference).
2. Create all database tables if they do not exist.
3. Mount routers and CORS middleware.
4. Serve.

Run (from project root):
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health
GET  /api/v1/samples               — paginated sample list with filters
GET  /api/v1/samples/{sample_id}   — sample detail + fingerprint
GET  /api/v1/clusters              — all cluster summaries
GET  /api/v1/clusters/{cluster_id} — single cluster detail
GET  /api/v1/network               — similarity network nodes + edges
POST /api/v1/fingerprint/predict   — real-time microbiome fingerprinting
"""

from __future__ import annotations

import json
import logging
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# ── Project path (run from project root; backend/ is a package) ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.database import check_connection, create_tables, get_db
from backend.models import ClusterSummary, Embedding, Fingerprint, Sample
from backend.schemas import (
    ClusterListResponse,
    ClusterSummaryOut,
    EmbeddingOut,
    FeatureImportanceOut,
    FingerprintOut,
    FingerprintPredictRequest,
    FingerprintPredictResponse,
    HealthResponse,
    NetworkEdge,
    NetworkNode,
    NetworkResponse,
    PaginationMeta,
    SampleDetail,
    SampleListResponse,
    SampleOut,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("main")

# ── ML model registry (populated at startup) ──────────────────────────────────

class ModelRegistry:
    """
    In-memory registry for all ML artifacts needed at inference time.

    Loaded once on startup; shared across all request handlers via module scope.
    """

    scaler:          Any = None    # sklearn StandardScaler
    pca:             Any = None    # sklearn PCA
    umap_model:      Any = None    # umap.UMAP (fitted)
    hdbscan_model:   Any = None    # hdbscan.HDBSCAN (fitted)
    feature_cols:    list[str] = []
    umap_embeddings: np.ndarray | None = None   # training UMAP coords (n, 2)
    umap_labels:     np.ndarray | None = None   # training cluster labels (n,)
    centroids:       dict[int, np.ndarray] = {}  # cluster_id → 2-D centroid
    pipeline:        str = "unknown"
    network_cache:   dict | None = None          # lazy-loaded similarity network
    n_clusters:      int = 0
    loaded:          bool = False


ml = ModelRegistry()


def _compute_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute mean 2-D position for every non-noise cluster."""
    centroids: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        if label < 0:
            continue
        centroids[int(label)] = embeddings[labels == label].mean(axis=0)
    return centroids


def load_ml_models() -> None:
    """
    Load all ML artifacts into the ModelRegistry.
    Raises RuntimeError if any required file is missing.
    """
    required = {
        "scaler":   settings.scaler_path,
        "pca":      settings.pca_path,
        "umap":     settings.umap_path,
        "model":    settings.best_model_path,
        "features": settings.feature_columns_path,
        "embeddings": settings.umap_embeddings_path,
        "pipeline": settings.best_pipeline_path,
    }
    for name, path in required.items():
        if not path.exists():
            raise RuntimeError(
                f"Required ML artifact missing: {path}. "
                "Run `python -m ml.train` before starting the backend."
            )

    logger.info("Loading ML artifacts …")

    ml.scaler        = joblib.load(settings.scaler_path)
    ml.pca           = joblib.load(settings.pca_path)
    ml.umap_model    = joblib.load(settings.umap_path)
    ml.hdbscan_model = joblib.load(settings.best_model_path)
    ml.feature_cols  = settings.feature_columns_path.read_text().splitlines()
    ml.pipeline      = settings.best_pipeline_path.read_text().strip()

    raw_embeddings   = np.load(settings.umap_embeddings_path)
    ml.umap_embeddings = raw_embeddings
    ml.umap_labels     = ml.hdbscan_model.labels_
    ml.centroids       = _compute_centroids(raw_embeddings, ml.umap_labels)
    ml.n_clusters      = len(ml.centroids)

    ml.loaded = True
    logger.info(
        "ML models loaded: pipeline=%s, features=%d, clusters=%d",
        ml.pipeline, len(ml.feature_cols), ml.n_clusters,
    )


# ── Inference helpers ─────────────────────────────────────────────────────────

def _clr_transform(x: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """Centered Log-Ratio — must match ml/preprocessing.py exactly."""
    x = x + pseudocount
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def _assign_cluster_by_centroid(umap_point: np.ndarray) -> tuple[int, float]:
    """
    Assign a new 2-D UMAP point to the nearest cluster centroid.

    Returns (cluster_id, distance_to_centroid).
    Falls back to (-1, inf) if no centroids exist.
    """
    if not ml.centroids:
        return -1, float("inf")

    best_id   = -1
    best_dist = float("inf")
    for cid, centroid in ml.centroids.items():
        dist = float(np.linalg.norm(umap_point - centroid))
        if dist < best_dist:
            best_dist = dist
            best_id   = cid
    return best_id, best_dist


def _novelty_score_from_distance(cluster_id: int, dist: float) -> float:
    """
    Normalise distance to [0, 1] using the max distance seen in training
    for that cluster.  Falls back to 1.0 (maximally novel) for noise.
    """
    if cluster_id < 0 or ml.umap_embeddings is None:
        return 1.0

    centroid = ml.centroids.get(cluster_id)
    if centroid is None:
        return 1.0

    mask      = ml.umap_labels == cluster_id
    members   = ml.umap_embeddings[mask]
    distances = np.linalg.norm(members - centroid, axis=1)
    d_max     = distances.max()
    d_min     = distances.min()
    if d_max <= d_min:
        return 0.0
    return float(min(1.0, (dist - d_min) / (d_max - d_min)))


def _shannon(abundances: np.ndarray) -> float:
    a = abundances[abundances > 0]
    if len(a) == 0:
        return 0.0
    p = a / a.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def _classify_diversity(score: float) -> str:
    if score < 1.5:
        return "Low"
    elif score < 3.0:
        return "Medium"
    return "High"


def _classify_novelty(score: float) -> str:
    if score <= 0.33:
        return "Common"
    elif score <= 0.66:
        return "Moderately Unique"
    return "Highly Unique"


def _parse_display_name(col: str) -> str:
    """Extract most specific taxon name from a k__… lineage string."""
    for part in reversed(col.split("|")):
        level, _, name = part.partition("__")
        name = name.strip().replace("_", " ")
        if name and name.lower() not in ("", "unclassified", "unknown"):
            return name
    return col


def _get_phylum(col: str) -> str:
    for part in col.split("|"):
        level, _, name = part.partition("__")
        if level.strip().lower() == "p":
            return name.strip().replace("_", " ")
    return "Unknown"


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving and shutdown tasks after."""
    logger.info("Microbiome Atlas backend starting up …")

    # Load ML models
    try:
        load_ml_models()
    except RuntimeError as exc:
        logger.error("Startup failed: %s", exc)
        # Proceed without models — /health will report models_loaded=False
        # Routes that require inference will return 503

    # Ensure DB tables exist
    try:
        create_tables()
    except Exception as exc:
        logger.error("Database table creation failed: %s", exc)

    logger.info("Backend ready.")
    yield

    logger.info("Backend shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Microbiome Atlas API",
    description=(
        "Unsupervised learning platform for human gut microbiome discovery. "
        "Provides cluster assignments, fingerprints, similarity networks, "
        "and real-time microbiome profiling."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global error handlers ─────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"detail": "Resource not found."})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Return server health, DB connectivity, and ML model status."""
    db_ok      = check_connection()
    n_samples  = db.query(Sample).count()
    n_clusters = db.query(ClusterSummary).filter(
        ClusterSummary.is_noise == False  # noqa: E712
    ).count()

    return HealthResponse(
        status        = "ok",
        db_online     = db_ok,
        pipeline      = ml.pipeline,
        n_samples     = n_samples,
        n_clusters    = n_clusters,
        models_loaded = ml.loaded,
    )


# ── Samples ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/samples", response_model=SampleListResponse, tags=["Samples"])
def list_samples(
    db:       Session = Depends(get_db),
    page:     int     = Query(1,    ge=1,  description="Page number (1-based)"),
    limit:    int     = Query(50,   ge=1,  le=500, description="Items per page"),
    disease:  str | None = Query(None, description="Filter by disease label"),
    country:  str | None = Query(None, description="Filter by country"),
    gender:   str | None = Query(None, description="Filter by gender"),
    cluster:  int | None = Query(None, description="Filter by cluster ID (via fingerprint)"),
) -> SampleListResponse:
    """
    Paginated list of microbiome samples with optional metadata filters.

    Filtering on cluster_id joins through the fingerprints table so that
    only samples with a computed fingerprint appear when cluster is provided.
    """
    query = db.query(Sample)

    if disease:
        query = query.filter(Sample.disease.ilike(f"%{disease}%"))
    if country:
        query = query.filter(Sample.country.ilike(f"%{country}%"))
    if gender:
        query = query.filter(Sample.gender == gender.lower())
    if cluster is not None:
        query = (
            query
            .join(Fingerprint, Fingerprint.sample_pk == Sample.id)
            .filter(Fingerprint.cluster_id == cluster)
        )

    total  = query.count()
    offset = (page - 1) * limit
    rows   = query.order_by(Sample.id).offset(offset).limit(limit).all()

    return SampleListResponse(
        data=[SampleOut.model_validate(r) for r in rows],
        pagination=PaginationMeta(
            total=total,
            page=page,
            limit=limit,
            pages=math.ceil(total / limit) if total > 0 else 1,
        ),
    )


@app.get(
    "/api/v1/samples/{sample_id}",
    response_model=SampleDetail,
    tags=["Samples"],
)
def get_sample(sample_id: str, db: Session = Depends(get_db)) -> SampleDetail:
    """
    Full sample detail including fingerprint and embedding.

    sample_id here is the natural string ID from the dataset (e.g. 'H10').
    When duplicates exist, the first database row for that sample_id is returned.
    """
    sample = (
        db.query(Sample)
        .filter(Sample.sample_id == sample_id)
        .order_by(Sample.id)
        .first()
    )
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    result = SampleDetail.model_validate(sample)
    if sample.fingerprint:
        result.fingerprint = FingerprintOut.model_validate(sample.fingerprint)
    if sample.embedding:
        result.embedding = EmbeddingOut.model_validate(sample.embedding)
    return result


# ── Clusters ──────────────────────────────────────────────────────────────────

@app.get(
    "/api/v1/clusters",
    response_model=ClusterListResponse,
    tags=["Clusters"],
)
def list_clusters(db: Session = Depends(get_db)) -> ClusterListResponse:
    """
    All discovered clusters with biological summaries and driving features.

    Results are ordered by cluster_id ascending; noise cluster (id=-1) is last.
    """
    clusters = (
        db.query(ClusterSummary)
        .order_by(
            ClusterSummary.is_noise.asc(),   # real clusters first
            ClusterSummary.cluster_id.asc(),
        )
        .all()
    )

    n_noise    = sum(1 for c in clusters if c.is_noise)
    n_clusters = len(clusters) - n_noise

    # Read pipeline name from the first non-noise cluster's pipeline_scores
    pipeline = ml.pipeline
    if not pipeline or pipeline == "unknown":
        for c in clusters:
            if not c.is_noise and c.pipeline_scores:
                pipeline = c.pipeline_scores.get("pipeline", ml.pipeline)
                break

    return ClusterListResponse(
        pipeline   = pipeline,
        n_clusters = n_clusters,
        n_noise    = n_noise,
        clusters   = [ClusterSummaryOut.model_validate(c) for c in clusters],
    )


@app.get(
    "/api/v1/clusters/{cluster_id}",
    response_model=ClusterSummaryOut,
    tags=["Clusters"],
)
def get_cluster(cluster_id: int, db: Session = Depends(get_db)) -> ClusterSummaryOut:
    """Single cluster detail by integer cluster ID."""
    cluster = (
        db.query(ClusterSummary)
        .filter(ClusterSummary.cluster_id == cluster_id)
        .first()
    )
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")
    return ClusterSummaryOut.model_validate(cluster)


# ── Similarity Network ────────────────────────────────────────────────────────

@app.get(
    "/api/v1/network",
    response_model=NetworkResponse,
    tags=["Network"],
)
def get_network(
    cluster: int | None = Query(None, description="Filter nodes to a single cluster ID"),
) -> NetworkResponse:
    """
    Similarity network for Plotly graph visualisation.

    The network JSON is cached in memory after the first request.
    If a cluster filter is applied, nodes are filtered but edges are
    restricted to those connecting two nodes within the filtered set.
    """
    # Lazy-load from disk on first call
    if ml.network_cache is None:
        path = settings.similarity_network_json
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail="similarity_network.json not found. Run ml.train first.",
            )
        with open(path, encoding="utf-8") as fh:
            ml.network_cache = json.load(fh)
        logger.info("Similarity network loaded from disk: %d nodes, %d edges",
                    ml.network_cache["n_nodes"], ml.network_cache["n_edges"])

    network = ml.network_cache

    nodes_raw: list[dict] = network.get("nodes", [])
    edges_raw: list[dict] = network.get("edges", [])

    # Optional cluster filter
    if cluster is not None:
        nodes_raw = [n for n in nodes_raw if n.get("cluster") == cluster]
        node_ids  = {n["id"] for n in nodes_raw}
        edges_raw = [
            e for e in edges_raw
            if e["source"] in node_ids and e["target"] in node_ids
        ]

    nodes = [
        NetworkNode(
            id      = n["id"],
            cluster = int(n.get("cluster", -1)),
            umap_x  = float(n.get("umap_x", 0.0)),
            umap_y  = float(n.get("umap_y", 0.0)),
            disease = n.get("disease"),
            country = n.get("country"),
            gender  = n.get("gender"),
        )
        for n in nodes_raw
    ]
    edges = [
        NetworkEdge(
            source = e["source"],
            target = e["target"],
            weight = float(e.get("weight", 0.0)),
        )
        for e in edges_raw
    ]

    return NetworkResponse(
        n_nodes   = len(nodes),
        n_edges   = len(edges),
        threshold = float(network.get("threshold", 0.9)),
        nodes     = nodes,
        edges     = edges,
    )


# ── Real-time Fingerprint Prediction ─────────────────────────────────────────

@app.post(
    "/api/v1/fingerprint/predict",
    response_model=FingerprintPredictResponse,
    tags=["Fingerprint"],
)
def predict_fingerprint(
    body: FingerprintPredictRequest,
) -> FingerprintPredictResponse:
    """
    Accept raw microbial abundance values and return a complete microbiome
    fingerprint in real time.

    Inference pipeline
    ------------------
    1. Align input abundances to the trained feature vector (1076 features).
    2. Apply CLR transform (must match ml/preprocessing.py).
    3. Apply fitted StandardScaler.
    4. Transform with fitted PCA (→ 50 dims).
    5. Transform with fitted UMAP (→ 2 dims).
    6. Assign to nearest cluster centroid in UMAP space.
    7. Compute diversity, novelty, and dominant microbe list.

    Missing features default to 0.0 (absent = not detected).
    Extra features not in the trained set are silently ignored.
    """
    if not ml.loaded:
        raise HTTPException(
            status_code=503,
            detail="ML models are not loaded. Start the server after running ml.train.",
        )

    raw_abundances: dict[str, float] = body.abundances

    # ── 1. Align to feature vector ────────────────────────────────────────────
    n_features   = len(ml.feature_cols)
    X_raw        = np.zeros((1, n_features), dtype=np.float64)
    n_matched    = 0

    for i, col in enumerate(ml.feature_cols):
        val = raw_abundances.get(col, 0.0)
        if val > 0:
            n_matched += 1
        X_raw[0, i] = max(0.0, float(val))   # clamp negatives to 0

    # ── 2. CLR transform ─────────────────────────────────────────────────────
    X_clr = _clr_transform(X_raw)

    # ── 3. StandardScaler ────────────────────────────────────────────────────
    X_scaled = ml.scaler.transform(X_clr)

    # ── 4. PCA ───────────────────────────────────────────────────────────────
    X_pca = ml.pca.transform(X_scaled)

    # ── 5. UMAP ──────────────────────────────────────────────────────────────
    try:
        X_umap = ml.umap_model.transform(X_pca)
    except Exception as exc:
        logger.error("UMAP transform failed: %s", exc)
        raise HTTPException(status_code=500, detail="UMAP transform failed.")

    umap_x = float(X_umap[0, 0])
    umap_y = float(X_umap[0, 1])

    # ── 6. Cluster assignment ─────────────────────────────────────────────────
    cluster_id, centroid_dist = _assign_cluster_by_centroid(X_umap[0])

    cluster_label = (
        "Noise" if cluster_id < 0 else f"Cluster {cluster_id}"
    )

    # ── 7a. Diversity ─────────────────────────────────────────────────────────
    diversity_score = _shannon(X_raw[0])
    diversity_class = _classify_diversity(diversity_score)

    # ── 7b. Novelty ───────────────────────────────────────────────────────────
    novelty_score  = _novelty_score_from_distance(cluster_id, centroid_dist)
    novelty_class  = _classify_novelty(novelty_score)

    # ── 7c. Dominant microbes ─────────────────────────────────────────────────
    abun_vec   = X_raw[0]
    top_idx    = np.argsort(abun_vec)[::-1][:10]
    top_microbes: list[dict[str, Any]] = []
    for idx in top_idx:
        val = float(abun_vec[idx])
        if val < 1e-6:
            break
        col = ml.feature_cols[idx]
        top_microbes.append({
            "name":      _parse_display_name(col),
            "abundance": round(val, 6),
            "phylum":    _get_phylum(col),
        })

    # ── 7d. Phylum profile ────────────────────────────────────────────────────
    phylum_agg: dict[str, float] = {}
    for i, col in enumerate(ml.feature_cols):
        ph  = _get_phylum(col)
        val = float(abun_vec[i])
        phylum_agg[ph] = phylum_agg.get(ph, 0.0) + val
    total_abun = sum(phylum_agg.values())
    phylum_profile: dict[str, float] = (
        {k: round(v / total_abun, 6) for k, v in phylum_agg.items()}
        if total_abun > 0 else {}
    )

    return FingerprintPredictResponse(
        cluster_id       = cluster_id,
        cluster_label    = cluster_label,
        umap_x           = round(umap_x, 6),
        umap_y           = round(umap_y, 6),
        diversity_score  = round(diversity_score, 6),
        diversity_class  = diversity_class,
        novelty_score    = round(novelty_score, 6),
        novelty_class    = novelty_class,
        top_microbes     = top_microbes,
        phylum_profile   = phylum_profile,
        n_features_used  = n_matched,
        n_features_total = n_features,
        pipeline         = ml.pipeline,
    )


# ── Development entry-point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info",
    )