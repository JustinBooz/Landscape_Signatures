# Graph-Based Embedding Analysis Pipeline: Swiss Landscape Signatures

This document defines the technical architecture and current status of the 14-step computational pipeline used to process, cluster, and validate **7.2 million** landscape image embeddings (**DINOv3 7B** features).

## 1. Project Context & Scale
*   **Dataset**: 7,215,142 landscape images covering all of Switzerland.
*   **Embeddings**: **DINOv3 7B** vision transformer features (4,096 dimensions).
*   **Coordinate System**: EPSG:2056 (Swiss LV95 / CH1903+).
*   **Hardware Profile**: Optimized for 125GB RAM nodes with high-core counts (24+ vCPUs) and GPU-accelerated FAISS.

## 2. Global Configuration & Performance
*   **Data Flow**: Checkpoint-driven with contiguous `memmap` storage to handle the ~60GB primary feature array.
*   **Resiliency**: Mid-fit checkpointing in Step 03 and segfault-recovery logic in Step 06 ensure robustness against system-level interrupts.

---

## 3. Pipeline Specification & Current Status

### Phase 1: Data Preparation & Compression
*   **Step 01: Manifest Construction (`s01_manifest.py`)** [COMPLETE]
    *   Builds the master index mapping 7.2M images to LV95 coordinates.
*   **Step 02: Embedding Normalization (`s02_normalize.py`)** [COMPLETE]
    *   L2-normalization for cosine-similarity equivalence.
*   **Step 03: PCA Compression & Diagnostics (`s03_pca.py`)** [COMPLETE]
    *   **Result**: Selected **256D** as the optimal dimension after testing 128/256/512 tiers. Achieved ~67% neighbor preservation recall@50.

### Phase 2: Graph Construction
*   **Step 04: FAISS kNN Search (`s04_faiss_knn.py`)** **[ACTIVE BOTTLENECK]**
    *   **Configuration**: GPU-accelerated `IVFFlat` index ($k=50, 100, 200$, $nlist=4096, nprobe=128$).
    *   **Status**: Currently searching `k=50` for 7.2M vectors (~380 vectors/second).
*   **Step 05: Graph Construction (`s05_graph.py`)** [READY]
    *   Generates shared-nearest-neighbor (SNN) sparse adjacency matrices.

### Phase 3: Community Detection & Stability
*   **Step 06: Leiden Community Detection (`leiden_parallel.py`)** [READY]
    *   Fractional grid of 13 Leiden runs (res 0.25 to 4.0) with multi-worker parallelism.
*   **Step 07: Cluster Stability Diagnostics (`s07_stability.py`)**
*   **Step 08: Medoid Extraction (`s08_medoids.py`)**

### Phase 4: Validation & Finalization
*   **Step 09-11**: Auxiliary HDBSCAN, UMAP 2D projections, and geographic continuity validation.
*   **Step 12: External Validation (`s12_external.py`)** **[STUB]**
    *   **Upcoming**: Validation against Arealstatistik and expert labels.
*   **Step 13-14**: Negative control statistical testing and Final Master Parquet assembly.

---

## 4. Advanced Enrichment Workflows (Sidecars)

### LABES Indicator Integration v2 (`labes_enrich.py`)
**[MATURE / PRODUCTION READY]**
The pipeline now includes a fully spec-driven enrichment system for socio-environmental indicators.
*   **Single Source of Truth**: `config/labes_indicators.yaml` defines 34 indicators.
*   **Multi-Phase Execution**:
    *   **Phase A (Social)**: Municipal survey data (perceived beauty, identity).
    *   **Phase B (Spatial)**: Polygon-based joins (mesh size, agricultural diversity).
    *   **Phase C (Raster/Grid)**: 100m Arealstatistik point lattice (forest, alpine pasture), elevation and sprawl sampling.
    *   **Phase D (Distance)**: Precise vector distances to infrastructure.
*   **Validation Logic**: Includes a `validation_report.md` with "rigor-checks" and landmark sampling against known ground-truth points.
*   **Stable Schema**: Emits a sidecar parquet with a fixed schema, ensuring downstream compatibility even if individual indicators fail or are skipped.

### Infrastructure Enrichment (`geodata_enrich_v2.py`)
*   Exact geometry distances (not centroids) to TLM3D/GWR features using `STRtree` spatial indexing.

---

## 5. Remaining High-Priority Tasks
1.  **Step 04 Completion**: High-priority kNN search for the 7.2M image set.
2.  **External Validation (Step 12)**: Implementing the final enrichment logic for land-use comparison.
3.  **Frontend Schema Finalization**: Consolidating the Master Parquet for interactive map deployment.
