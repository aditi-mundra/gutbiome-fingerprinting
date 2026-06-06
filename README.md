# Gutbiome Fingerprinting Platform

An advanced, unsupervised machine learning pipeline designed to discover microbial communities, generate individual microbiome fingerprints, calculate diversity metrics, and build population similarity networks from stool sample abundance data.

This project focuses entirely on data discovery, pattern extraction, and population analytics.

---

## 🚀 Project Status
- [x] **Module 1: Machine Learning & Analytics Pipeline** (Completed)
- [ ] **Module 2: Backend API Development** (Future Work)
- [ ] **Module 3: Relational Database Integration** (Future Work)
- [ ] **Module 4: Frontend UI Dashboard** (Future Work)

---

## 📊 Pipeline Architecture & ML Strategy

The machine learning framework acts as an isolated engine that processes raw files and outputs static artifacts (models, charts, and structural JSONs) to be consumed later by the backend APIs.

### 1. Exploratory Data Analysis (EDA)
Generates comprehensive distribution profiles and analysis charts saved directly to `docs/figures/`:
* `dataset_overview.png`: High-level metrics on features and metadata distributions.
* `missing_values.png`: Data integrity mapping.
* `top_microbes.png` & `most_variable_features.png`: Abundance spectrum profiles.
* `pca_variance.png`: Dimensionality analysis.
* `umap_preview.png`: Core geometric structure visualization.

### 2. Preprocessing & Normalization
* Filters and partitions clinical metadata from microbial abundance features (identified by the `k__` prefix).
* Handles missing data array elements and scales/normalizes abundance values for stable distance calculation.

### 3. Unsupervised Clustering Engine
Compares three distinct competitive pipelines to automatically isolate structural communities:
* **Pipeline A:** Principal Component Analysis (PCA) $\rightarrow$ K-Means
* **Pipeline B:** Principal Component Analysis (PCA) $\rightarrow$ Gaussian Mixture Model (GMM)
* **Pipeline C:** Uniform Manifold Approximation and Projection (UMAP) $\rightarrow$ HDBSCAN

### 4. Evaluation & Diagnostics
Pipelines are scored via **Silhouette Coefficients** and the **Davies-Bouldin Index**. The pipeline dynamically self-selects the champion model configuration, compiling the benchmark metrics into `evaluation_results.csv`.

### 5. Multi-Tier Bio-Fingerprinting
For every unique profile processed, the pipeline evaluates:
* **Alpha Diversity:** Calculates the *Shannon Diversity Index*, categorizing profiles into Low, Medium, or High diversity tiers.
* **Novelty Scoring:** Evaluates distance bounds from discovered cluster centroids to flag *Common*, *Moderately Unique*, or *Highly Unique* microbiome profiles.
* **Similarity Engine:** Builds a topological network graph using `NetworkX` (saved as `similarity_network.json`) based on Cosine Similarity metrics to uncover profile proximities.
* **Cluster Explainability:** Evaluates surrogate Random Forest models over generated cluster assignments to calculate feature importance ranks and isolate defining signature microbes.

---

## 📂 Project Structure

```text
gutbiome-fingerprinting/
│
├── data/                       # (Ignored by Git)
│   ├── raw/                    # Input datasets (abundance_stoolsubset.csv)
│   ├── processed/              # Normalized matrices and embeddings
│   └── results/                # CSV/JSON outputs (fingerprints, networks)
│
├── docs/                       # Project documentation
│   └── figures/                # (Ignored by Git) Generated EDA plots
│
├── ml/                         # Modular Machine Learning Pipeline
│   ├── config.py               # Hyperparameters, random seeds, and thresholds
│   ├── eda.py                  # Profiling and visual graphing logic
│   ├── preprocessing.py        # Feature isolation and transformation
│   ├── dimensionality_reduction.py # PCA and UMAP wrappers
│   ├── clustering.py           # Clustering pipeline initializations
│   ├── evaluation.py           # Model diagnostics and selection
│   ├── fingerprint.py          # Diversity, novelty, and signature compilation
│   ├── similarity.py           # NetworkX graph and similarity engines
│   ├── explainability.py       # Random Forest cluster feature extraction
│   └── train.py                # Pipeline execution script
│
├── models/                     # (Ignored by Git) Saved joblib artifacts
├── .gitignore                  # Keeps heavy data and environments out of git
├── README.md                   # Project documentation
└── requirements.txt            # Package dependencies 
```

##🛠️ Local Setup & Execution
1. Prerequisites
Ensure you have Python 3.10+ installed.

2. Environment Configuration
Clone this repository, initialize a local virtual environment, and install the required dependencies:

Bash
# Initialize Virtual Environment
python -m venv venv

# Activate Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate Environment (Mac/Linux)
source venv/bin/activate

# Install Core Framework Libraries
pip install -r requirements.txt
3. Execution
To run the complete machine learning pipeline from data ingestion to artifact generation, execute train.py as a module from the root directory:

Bash
python -m ml.train


## 🔮 Planned Enhancements
* **Database Layer (MySQL):** Design a relational database schema to migrate static JSON/CSV results into structured MySQL tables for persistent, scalable storage.
* **Backend Microservice (FastAPI):** Construct robust API endpoints to query the MySQL database and dynamically load `best_model.joblib` to serve real-time fingerprint lookups and cluster analytics.
* **Frontend UI Dashboard (React & Plotly):** Develop an interactive single-page web application to cleanly display individual bio-fingerprints and render the population similarity network dynamically from backend API responses.