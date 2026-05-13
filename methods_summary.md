# Methodology Summary: Graph-Based Landscape Signature Identification

## 1. Overview
This study utilizes a high-throughput computational pipeline to identify and categorize "landscape signatures" across Switzerland. By processing 7,215,142 high-resolution landscape images, we derive a data-driven typology that captures both visual aesthetics and socio-environmental contexts.

## 2. Dataset Engineering & Acquisition
### 2.1 Primary Data Source
The primary imagery for this study is sourced from **Mapillary**, a crowdsourced street-level imagery platform. The dataset captures a comprehensive spatial cross-section of the Swiss landscape, representing diverse environments from dense urban centers to high-alpine regions.

### 2.2 Spatial Sampling & Engineering
The raw dataset consists of 7,215,142 images, curated through a multi-stage engineering process:
*   **Geographic Filtering**: Imagery is strictly bounded to the Swiss national territory (CH1903+ / LV95 coordinate system).
*   **Burst & Redundancy Management**: High-frequency spatial sampling (bursts) is addressed during manifest construction by flagging images with identical geographic coordinates (`is_duplicate_coord` flag), allowing for statistical control of over-sampled locations.
*   **Panorama & Resolution Handling**: To account for variable aspect ratios and panoramic imagery, the extraction pipeline employs an **adaptive transform** logic. Images are dynamically resized along their shortest edge to 518 pixels and subsequently **center-cropped** to a standard 518x518 square before feature extraction.
*   **Dual-Image Ingestion**: The engineering pipeline processes image pairs (e.g., sequential frames or dual-camera captures) as unified WebDataset samples, ensuring consistent coordinate assignment across multi-image bursts.
*   **Manifest Integration**: A unified spatial manifest links each unique image ID to its precise LV95 coordinates and embedding shard location, supporting atomic resumption and distributed graph construction.

## 3. Feature Extraction & Data Preparation
### 3.1 Deep Visual Features
We employ the **DINOv3 7B** vision transformer to extract high-dimensional **visual representation features** from each landscape image. This model provides 4,096-dimensional embeddings ($D=4096$) that capture complex architectural styles, land-cover patterns, and topographical nuances.

### 3.2 Normalization & Storage
Embeddings are L2-normalized to ensure that cosine similarity is equivalent to Euclidean distance, facilitating downstream clustering. To manage the ~60GB primary dataset within a 125GB RAM constraint, features are stored as a contiguous 16-bit floating-point (`float16`) memory-mapped array, with on-the-fly 32-bit upcasting for computational precision.

## 4. Dimensionality Reduction & Compression
To mitigate the "curse of dimensionality" and optimize graph construction, we apply **Incremental Principal Component Analysis (IPCA)**. 
*   **Compression Tiers**: 128, 256, and 512 components.
*   **Selection Logic**: We evaluate the preservation of local neighborhoods by comparing k-Nearest Neighbor ($k=50$) overlap between the original 4096D space and compressed subspaces.
*   **Operational Baseline**: **256D** was selected as the optimal balance between computational efficiency and topological fidelity (~67% recall@50).

## 5. Manifold Learning & Graph Construction
### 5.1 Approximate Nearest Neighbor (ANN) Search
We utilize the **FAISS** (Facebook AI Similarity Search) library with a GPU-accelerated **IVFFlat** index. We compute the nearest neighbors for every point at three scales: $k \in \{50, 100, 200\}$. 

### 5.2 Shared-Nearest-Neighbor (SNN) Graph
The kNN results are transformed into a sparse adjacency matrix. We employ a **Shared-Nearest-Neighbor (SNN)** approach, where edge weights are defined by the Jaccard similarity of the neighbor sets between any two points. This method effectively filters noise and emphasizes robust local communities over spurious high-dimensional proximities.

## 6. Community Detection & Stability
### 6.1 Leiden Clustering
Community detection is performed using the **Leiden algorithm**, which optimizes modularity while avoiding the formation of disconnected communities. 
*   **Resolution Tuning**: We execute a multi-resolution grid ($ \gamma \in \{0.25, 0.5, 1.0, 2.0, 4.0\} $) to identify signatures at varying scales, from broad regional patterns to hyper-local architectural nuances.

### 6.2 Robustness Diagnostics
To ensure that discovered clusters are not artifacts of the algorithm, we run a **fractional grid of 13 perturbations** (varying seeds, $k$, and graph types). Cluster stability is quantified using the **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**.

## 7. Geospatial & Socio-Environmental Enrichment
### 7.1 Infrastructure Context
We perform high-precision spatial joins against Swiss national geodata (TLM3D, GWR, SwissALTI3D). Using **STRtree** spatial indexing, we calculate the exact geometric distance of each manifest point to roads, rail networks, water bodies, and building footprints.

### 7.2 LABES v2 Indicators
The dataset is enriched with 34 socio-environmental indicators from the **LABES** framework. This spec-driven workflow (v2) categorizes indicators into four phases:
*   **Survey-Based (Social)**: Perceived beauty, identity, and landscape quality.
*   **Structural (Spatial)**: Urban sprawl (WUP), mesh size, and agricultural diversity.
*   **Physical (Raster/Grid)**: Topographical remoteness, light emissions, and 100m Arealstatistik point-lattice classification (forest, pasture, settlement).
*   **Proximity (Vector)**: Precise geometric distances to transport infrastructure and renewable energy features.

## 8. Synthesis & Representative Extraction
For each validated cluster, we identify the **Medoid**—the single image embedding that is most central to the community in the high-dimensional feature space. These medoids serve as the canonical visual representatives for the identified landscape signatures in final reporting and visualization.
