# Pipeline Robustness Extension — Coding Agent Instructions

## Context
Extend the existing 14-step checkpoint-driven landscape embedding pipeline (DINOv3 → PCA → FAISS kNN → mutual-kNN graph → Leiden → validation). **Do not rewrite the pipeline.** Keep Leiden as the primary clustering method. Add targeted robustness checks, auxiliary comparators, per-cluster validation scores, and a pre-registered candidate-signature filter.

## Engineering Constraints (Hard)
- Do not rewrite the pipeline. Do not replace Leiden as the primary method.
- All new steps must be checkpoint-driven, resumable, configurable, and skippable.
- No dense all-pairs matrices. No full N × N consensus matrices.
- HDBSCAN and FINCH must be optional/skippable if the dependency is missing. **Do not add EVōC.**
- Use sampling (≤ 1000 points per cluster) for any O(n²) spatial metric. Never compute all-pairs geographic distances.
- UMAP islands are visualization only, never treated as final clusters.

---

## 1. Pre-Registration (NEW) — `config/candidate_signature_criteria.yaml`

Create and commit to git **before** running the extended pipeline. Numeric thresholds marked TBD are calibrated after Steps 7 and 13 complete; the *structure* must be committed up front.

```yaml
reference_configuration:
  pca_dim: 256
  k: 100
  graph_type: snn_jaccard
  resolution: 1.0
  seed: 0

candidate_signature_thresholds:
  seed_stability_min: 0.70         # mean best-match Jaccard across seeds
  k_stability_min: 0.60            # mean across k ∈ {50, 200}
  graph_stability_min: 0.60        # mean across {mutual_knn, mutual_snn_jaccard}
  resolution_stability_min: 0.50   # mean across adjacent resolutions
  spatial_coherence_min: TBD       # set after s13 null calibration
  geodata_purity_min: TBD          # set after s13 null calibration
  negative_control_p_max: 0.01

scope_caveat: |
  These criteria establish robustness conditional on the DINOv3 embedding.
  They do not validate the embedding itself against human perceptual structure.
  That validation is performed in WP2 (gaze tracking).
```

---

## 2. Graph Variants — extend `s05_graph.py`

From existing FAISS kNN outputs (k = 50, 100, 200), build and checkpoint three graph types:
- `mutual_knn` (already exists; keep as baseline)
- `snn_jaccard`
- `mutual_snn_jaccard`

For SNN/Jaccard, compute weights **only on existing kNN candidate edges**:
`Jaccard(i,j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|`. Never all-pairs.

Per graph variant save: sparse CSR file, edge count, mean/median degree, degree quantiles (5/25/50/75/95), connected components, largest component size, isolated node fraction, edge-weight summary (min/mean/median/max/std), runtime.

---

## 3. Leiden Grid — extend `s06_leiden.py` / `leiden_parallel.py`

Use a **fractional design** centered on the reference configuration, not a full factorial. Reference: `pca256, k=100, snn_jaccard, res=1.0`.

| Axis | Perturbations | Runs |
|------|---------------|------|
| Seed | reference × seeds {0,1,2,3,4} | 5 |
| k | reference with k ∈ {50, 200} × seed 0 | 2 |
| Graph | reference with graph ∈ {mutual_knn, mutual_snn_jaccard} × seed 0 | 2 |
| Resolution | reference with res ∈ {0.25, 0.5, 2.0, 4.0} × seed 0 | 4 |

**Total: 13 runs** (not 225). Add additional samples only if a specific stability score falls below threshold.

Per run save: cluster assignments (`global_index`, `cluster_id`), config metadata (`pca_dim`, `k`, `graph_type`, `resolution`, `seed`), num clusters, cluster size distribution, singleton count, largest cluster size, percent of points in top 10 clusters, modularity/CPM objective, runtime, graph used.

---

## 4. Stability Diagnostics — extend `s07_stability.py`

Compute ARI and NMI across the four perturbation axes:
- 5 seed runs (seed stability)
- k perturbations vs reference (k stability)
- graph perturbations vs reference (graph stability)
- resolution perturbations vs reference (resolution stability)

**Do not build a full N × N consensus matrix.**

Output `stability_summary.parquet` and `stability_summary.md` identifying: most stable graph/k/resolution, overfragmented configurations (excessive singletons), giant-cluster-dominated configurations (top cluster > 30% of points), recommended primary configuration (if reference fails, recommend alternative).

---

## 5. Per-Cluster Confidence — NEW `s07b_cluster_confidence.py`

For each cluster in the reference clustering, compute best-matching cluster Jaccard against every perturbation run. Aggregate per cluster:
- `seed_stability`, `k_stability`, `graph_stability`, `resolution_stability`
- `overall_cluster_confidence` = weighted mean of the four

Save `cluster_confidence.parquet` keyed by reference `cluster_id`. **No pairwise point consensus.**

---

## 6. Microclustering — NEW `s04b_microclusters.py`

FAISS spherical k-means on PCA-256 embeddings, `n_microclusters = 50000` (configurable).

Save: image→microcluster assignments, centroids, distance-to-centroid, size distribution, empty cluster count, runtime. Output enables tractable auxiliary clustering on 50k centroids rather than 7.2M points.

---

## 7. Auxiliary Comparators — NEW `s09b_aux_comparators.py`

Two optional comparators, each skips gracefully if the dependency is missing:
- **HDBSCAN** on microcluster centroids
- **FINCH** (Sarfraz et al. 2019) on microcluster centroids

**Do not add EVōC.**

Per comparator save: cluster labels (lifted from centroids to all 7.2M points via microcluster assignment), hierarchy levels (FINCH), noise labels (HDBSCAN), cluster size distribution, ARI/NMI agreement with the primary Leiden reference clustering.

---

## 8. Per-Cluster Spatial / Geodata Validation — extend `s11_spatial.py`, `s12_external.py`

For each reference cluster compute (sampling ≤ 1000 points per cluster for O(n²) metrics):
- cluster size
- geographic dispersion (median pairwise distance on the sample)
- spatial extent (bounding box area, convex hull area)
- elevation summary from SwissALTI3D (mean, std, quantiles)
- distance to {road, rail, water, forest, building}: mean, median, quantiles
- dominant geodata categories (top 3 by frequency)
- geodata entropy (Shannon entropy over category frequencies)
- geodata purity (frequency of the dominant category)

Save `cluster_spatial_validation.parquet`.

---

## 9. Calibrated Negative Controls — extend `s13_negative_controls.py`

Replace binary pass/fail with empirical p-values from a null distribution.

For each of 100 random permutations of cluster labels (preserving cluster size distribution): recompute spatial coherence (median pairwise distance per cluster) and geodata purity per cluster.

For each real cluster, compute an empirical p-value of its `spatial_coherence_score` and `geodata_purity_score` against the null. Save `negative_control_pvalues.parquet`.

**After this step runs**, fill in the TBD threshold values in `config/candidate_signature_criteria.yaml` such that random clusters would not pass (e.g., set `spatial_coherence_min` at the 99th percentile of the null distribution).

---

## 10. Final Assembly — extend `s14_final_assembly.py`

Per-image master table `landscape_embedding_analysis_master.parquet` must include:
- `primary_leiden_cluster_id` (from reference configuration)
- `primary_clustering_config` (string identifier)
- `cluster_confidence` (joined from s07b)
- `microcluster_id`
- `hdbscan_label`, `finch_label_L1..LN` (NaN if comparator skipped)
- `spatial_coherence_score`
- `geodata_purity_score`, `geodata_entropy`
- `negative_control_pvalue`
- `medoid_flag` (boolean)
- `candidate_signature_flag` (boolean: passes **all** pre-registered thresholds)

Cluster-level summary `cluster_signatures_summary.parquet`:
- `method`, `config`, `cluster_id`, `cluster_size`, `medoid_global_index`
- `seed_stability`, `k_stability`, `graph_stability`, `resolution_stability`
- `spatial_coherence_score`, `geodata_purity_score`, `geodata_entropy`
- `negative_control_pvalue`
- `candidate_signature_flag`
- `overall_candidate_signature_score` (composite scalar for ranking)

---

## 11. Reporting Language

The generated methodology summary **must not** say "Leiden discovered landscape signatures." Use:

> Leiden identifies visual-embedding communities. Communities that remain stable across graph construction choices, clustering parameters, auxiliary methods, spatial validation, geodata validation, and negative controls are treated as **candidate landscape signatures**. The robustness checks in this pipeline establish that, conditional on the DINOv3 embedding, the identified communities are not artifacts of clustering parameters. Validation that the embedding itself corresponds to human perceptual structure is addressed in WP2 (gaze tracking).

The report must include a section "Candidate Landscape Signatures" stating how many of N reference clusters passed the pre-registered thresholds in `config/candidate_signature_criteria.yaml`, with the criteria reproduced inline for auditability.

---

## Execution Order
1. Commit `config/candidate_signature_criteria.yaml` (with TBD spatial/geodata thresholds).
2. Implement `s04b_microclusters.py`.
3. Implement `s05_graph.py` graph variants.
4. Implement `s06_leiden.py` fractional grid.
5. Implement `s07_stability.py` + `s07b_cluster_confidence.py`.
6. Implement `s09b_aux_comparators.py` (optional, skip-on-missing-dep).
7. Implement `s11_spatial.py` + `s12_external.py` per-cluster scores.
8. Implement `s13_negative_controls.py` calibrated nulls → **then** fill in TBD thresholds in the YAML.
9. Implement `s14_final_assembly.py` with `candidate_signature_flag`.
10. Regenerate final report using the required language.
