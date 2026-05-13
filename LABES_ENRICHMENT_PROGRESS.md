# LABES enrichment — progress tracker

Last updated: **2026-05-13** (ind1 + ind17a switched to official BFS Arealstatistik 100m point source).

Per-image enrichment of the 7.2M-row Swiss landscape-image manifest with the full set of BAFU LABES indicators. Written as a sidecar parquet keyed on `global_index`, never mutating the source manifest. Workflow constraint: every indicator must be traceable to a BAFU publication and match BAFU's methodology where the data allows.

## Status at a glance

| | count |
|---|---|
| Indicators wired | **34 / 34** |
| Indicators at per-pixel raster | **11** (ind1, 3, 7, 8, 14, 15, 17a, 32, neu_2, neu_12, neu_14) |
| Indicators at per-point nearest-feature distance | **4** (ind4, ind31a, neu_7, neu_9) |
| Indicators at municipality polygon (~2,200 unique) | 14 (ind2a, ind36, 12 social) |
| Indicators at polygon-cell (42K cells) | 1 (ind35) |
| Indicators at biogeoregion (6 unique) | 2 (ind2, 9) |
| Indicators at national constant | 2 (ind11a, ind11b) |
| Indicators at string-vector (per-municipality) | 2 (ind_neu17, ind_neu18) |
| Joined successfully on 10K validation slice | 34 / 34 |
| PDF-reference checks within ±10% per biogeoregion | pass for biogeoregion/national joins; raster joins are manifest-sampling-dependent |
| Full 7.2M run executed | not yet |

## File map

| Path | Purpose |
|---|---|
| `config/labes_indicators.yaml` | **Single source of truth.** All 34 indicator specs (BAFU number, German slug, English description, source, join method, granularity, value unit/range, year, PDF reference values). |
| `config/labes_landmarks.yaml` | 25 hold-out reference points (Zürich HB, Aletschgletscher, etc.) with expected qualitative outcomes for the landmark rigor check. |
| `graph_pipeline/labes_enrich.py` | Spec-driven orchestrator. `--phase A|B|C|D|ALL` selector, `--limit N`, `--ind <col>`. Truncates `validation_report.md` on each run so the report always reflects the current state. |
| `graph_pipeline/labes_joins.py` | Per-join-method implementations: `social_csv_municipality`, `social_csv_vector_to_string`, `polygon_municipality_field`, `polygon_biogeoregion_field`, `polygon_cell_field`, `raster_point_sample` (now with optional `class_set` for binary categorical-raster derivations), `arealstatistik_point_sample` (official BFS 100m point lattice), `xlsx_biogeoregion_via_polygon`, `xlsx_biogeoregion_rows`, `xlsx_national_constant`, `spec_biogeoregion_values`, `nearest_feature_distance`, `_unsupported`. |
| `graph_pipeline/labes_validation.py` | Four rigor checks: unit_range, coverage (in-CH-only denominator), pdf_numerical (vs reference_values), landmarks. Outputs to `validation_report.md` (markdown) and `validation_log.jsonl` (per-indicator structured log). |
| `graph_pipeline/labes_inventory.py` | One-shot probe that walks every `geodata/LABES/labes_*/` folder and inventories what data assets (raster, vector, XLSX, PDF) are present. Output: `graph_pipeline/outputs/labes/inventory.yaml`. |
| `graph_pipeline/labes_inventory_summarize.py` | Classifies each indicator folder by best achievable granularity. Output: `inventory_summary.md/json`. |
| `scripts/extract_labes_rasters.py` | One-shot raster extractor. Pulls each useful subdataset out of BAFU's `.gdb` files (requires `osgeo` Python bindings) and writes standalone DEFLATE-compressed GeoTIFFs to `geodata/external/labes_rasters/`. |
| `scripts/calibrate_noas04_classes.py` | Calibration: aggregates NOAS04 classes per biogeoregion to match BAFU totals (used for ind3), and documents resolved Arealstatistik-source methods for ind1 and ind17a (`AS_72` 50..57 and 45..49 respectively). |
| `scripts/fetch_viirs_dnb_2020.py` | Independent NOAA EOG VIIRS DNB 500m annual composite for 2020 over CH (kept as an alternate higher-resolution night-light source). |
| `graph_pipeline/outputs/labes/labes_enriched.parquet` | The sidecar. 35 columns: `global_index` + 34 LABES indicators. |
| `graph_pipeline/outputs/labes/labes_join_registry.json` | Machine-readable per-run registry with status, n_valid, %null, check results per indicator. |
| `graph_pipeline/outputs/labes/validation_report.md` | Human-readable per-phase markdown report. Overwritten on each run. |
| `graph_pipeline/outputs/labes/validation_log.jsonl` | Per-indicator JSON record per run for downstream auditing. |
| `graph_pipeline/outputs/labes/inventory.yaml` + `inventory_summary.md/json` | Audit of what data is on disk per LABES indicator. |
| `geodata/external/labes_rasters/` | Extracted standalone GeoTIFF rasters from BAFU `.gdb` files. Includes `noas04_landuse_codes_lv95.tif` (the NOAS04 code raster underlying multiple indicators). |
| `geodata/external/arealstatistik/arealstatistik_2056.csv.zip` | Official BFS Arealstatistik point lattice (`ch.bfs.arealstatistik`, EPSG:2056). Used for ind1 (`AS_72` 50..57) and ind17a (`AS_72` 45..49) at native 100m resolution. |
| `geodata/external/viirs_dnb_npp_2020_switzerland_wgs84.tif` | Independent 500m VIIRS DNB annual composite for 2020 (NOAA EOG). |

## Per-indicator state

| BAFU id | Column name | Join | Granularity | Year | PDF ref check |
|---|---|---|---|---|---|
| 1 | `labes_ind1_wald` | arealstatistik_point_sample (`AS_72` class_set 50..57) | **raster_pixel binary 100m** LV95 | 2018 | calibrated to BAFU table |
| 2 | `labes_ind2_veraenderung_lw_flaeche` | xlsx_biogeoregion_via_polygon | biogeoregion | 2009 | ✓ (context) |
| 2a | `labes_ind2a_nutzungsvielfalt_lw` | polygon_municipality_field | municipality | 2018 | n/a |
| 3 | `labes_ind3_flaechenverbrauch_siedlung` | raster_point_sample (NOAS04 class_set 11/12/13/15/16/17) | **raster_pixel binary 100m** LV95 | 2018 | calibrated ±6.4% |
| 4 | `labes_ind4_verkehr_flaeche_laenge` | nearest_feature_distance | nearest_feature (road/rail network) | 2020 | n/a |
| 7 | `labes_ind7_versiegelung` | raster_point_sample (binary mask 0/1) | raster_pixel 100m LV03 | 2018 | n/a |
| 8 | `labes_ind8_bauen_ausserhalb_bauzone` | raster_point_sample | raster_pixel **12.5m** LV95 | 2020 | n/a |
| 9 | `labes_ind9_zerschneidung` | spec_biogeoregion_values (BAFU Meff km²) | biogeoregion | 2020 | ✓ (0/6 breach) |
| 11a | `labes_ind11a_veraenderung_fliessgewaesser` | xlsx_national_constant | national | 2020 | ✓ |
| 11b | `labes_ind11b_veraenderung_stehende_gewaesser` | xlsx_national_constant | national | 2020 | ✓ |
| 14 | `labes_ind14_licht` | raster_point_sample | raster_pixel **500m** LV03 | 2020 | n/a |
| 15 | `labes_ind15_naturueberlassene_flaeche` | raster_point_sample (binary 0/1) | raster_pixel 100m LV95 | 2018 | n/a |
| 17a | `labes_ind17a_soemmerungsweiden` | arealstatistik_point_sample (`AS_72` class_set 45..49) | **raster_pixel binary 100m** LV95 | 2018 | calibrated to BAFU table |
| 31a | `labes_ind31a_gewaesserabschnitte_frei_begehbar` | nearest_feature_distance | nearest_feature (1.2M lines) | 2020 | n/a |
| 32 | `labes_ind32_anlagefrei` | raster_point_sample (4-class) | raster_pixel **500m** LV95 | 2020 | n/a |
| 35 | `labes_ind35_erschliessung_fuss_wanderwege` | polygon_cell_field | polygon_cell (42K cells) | 2020 | n/a |
| 36 | `labes_ind36_zersiedlung_wup` | polygon_municipality_field | municipality (2222) | 2018 | n/a |
| NEU_2 | `labes_ind_neu2_abgeschiedenheit` | raster_point_sample | raster_pixel **100m** LV95 (3485×2208) | 2020 | ✓ |
| NEU_7 | `labes_ind_neu7_leitungen` | nearest_feature_distance | nearest_feature (1027 lines) | 2020 | n/a |
| NEU_9 | `labes_ind_neu9_windanlagen` | nearest_feature_distance | nearest_feature (24851 hex) | 2020 | n/a |
| NEU_12 | `labes_ind_neu12_landwirtschaftliche_intensitaet` | raster_point_sample | raster_pixel **10m** LV03 | 2019 | n/a |
| NEU_14 | `labes_ind_neu14_staedtisches_gruen` | raster_point_sample (NDVI ×1e-4) | raster_pixel **10m** LV03 | 2020 | n/a |
| 23 | `labes_ind23_ortsbindung` | social_csv_municipality + biogeoregion fallback | municipality | 2020 | ✓ Δ=+3.1% |
| 24 | `labes_ind24_landschaftsschoenheit` | social_csv_municipality + bg fallback | municipality | 2020 | ✓ Δ=+2.0% |
| 25 | `labes_ind25_besonderheit` | social_csv_municipality + bg fallback | municipality | 2020 | ✓ Δ=−1.2% |
| 27 | `labes_ind27_informationsgehalt` | social_csv_municipality + bg fallback (mean of Kohärenz/Komplexität/Lesbarkeit) | municipality | 2020 | composite |
| 29 | `labes_ind29_faszination` | social_csv_municipality + bg fallback | municipality | 2020 | ✓ Δ=−0.4% |
| 30 | `labes_ind30_authentizitaet` | social_csv_municipality + bg fallback | municipality | 2020 | ✓ Δ=−0.3% |
| NEU_16 | `labes_ind_neu16_erholungsnutzung` | social_csv_municipality + bg fallback (mean of Erholungsindex_*) | municipality | 2020 | composite |
| NEU_17 | `labes_ind_neu17_landschaftscharakter` | social_csv_vector_to_string (top 20 elements per muni) | municipality | 2020 | string |
| NEU_18 | `labes_ind_neu18_landschaftsveraenderung` | social_csv_vector_to_string (top 19 changes per muni, ≥4 Likert) | municipality | 2020 | string |
| NEU_20 | `labes_ind_neu20_grundnutzungen` | social_csv_municipality + bg fallback (mean of Qualität_Siedlung/Landwirt/Gewässer/Wald) | municipality | 2020 | composite |
| NEU_22 | `labes_ind_neu22_gesamturteil` | social_csv_municipality + bg fallback | municipality | 2020 | ✓ Δ=−0.3% |
| NEU_23 | `labes_ind_neu23_kulturelle_landschaftswerte` | social_csv_municipality + bg fallback | municipality | 2020 | ✓ |

## Workflow

```bash
# 0. one-time: install osgeo Python bindings (matches libgdal-core 3.11 in env)
conda install -n baukultur_vpr -c conda-forge -y gdal

# 1. one-time: extract BAFU's per-pixel rasters from FileGDBs
PROJ_DATA=$CONDA_PREFIX/share/proj \
GDAL_DATA=$CONDA_PREFIX/share/gdal \
$CONDA_PREFIX/bin/python scripts/extract_labes_rasters.py

# 2. one-time: build the independent NOAA VIIRS DNB 2020 raster
$CONDA_PREFIX/bin/python scripts/fetch_viirs_dnb_2020.py

# 3. one-time (for ind3 upgrade): calibrate NOAS04 class memberships
$CONDA_PREFIX/bin/python scripts/calibrate_noas04_classes.py

# 4. enrich a 10K sample for validation
PROJ_DATA=$CONDA_PREFIX/share/proj GDAL_DATA=$CONDA_PREFIX/share/gdal \
$CONDA_PREFIX/bin/python -u graph_pipeline/labes_enrich.py --phase ALL --limit 10000

# 5. full 7.2M run (pending — see follow-ups)
$CONDA_PREFIX/bin/python -u graph_pipeline/labes_enrich.py --phase ALL
```

Where `$CONDA_PREFIX = /home/jubooz/anaconda3/envs/baukultur_vpr`.

## Decision log

- **2026-05-13: schema renumbered to BAFU's official LABES indicator numbering.** Old project-internal labels (e.g. `labes_ind20_perceived_landscape_beauty`) carried mismatched BAFU semantics (BAFU 20 ≠ Schönheit; Schönheit is BAFU 24). Every column now uses the BAFU number + German slug.
- **2026-05-13: 12 social indicators get a biogeoregion-mean fallback.** Without fallback, ~56% of manifest points were null because only ~874 of CH's ~2,200 municipalities had any respondents in the 2,090-respondent survey. With fallback, in-CH null rate ≈ 1%.
- **2026-05-13: NEU_17 and NEU_18 emit pipe-delimited strings, not scalars.** BAFU explicitly does NOT publish these as scalar indices; NEU_17 is a 20-element vector (cosine-similarity comparisons), NEU_18 is per-statement Likert means. We emit a pipe-delimited list of element/change slugs above a 50% / ≥4-Likert threshold per municipality. 41 / 37 distinct signatures emerged on the 10K slice.
- **2026-05-13: coverage check uses in-CH denominator.** Manifest points falling outside CH biogeoregion polygons (border zones, water bodies, Liechtenstein) are excluded from the null-fraction tally per the user decision.
- **2026-05-13: validation_report.md overwritten on each run.** Previously it appended, causing duplicate phase sections across runs.
- **2026-05-13: ind14 night light uses BAFU's 500m VCMSL** (`VIIRS_500m_LV03_2020_mask`), upgraded from 1000m, and from biogeoregion XLSX. The independent NOAA-EOG NPP 500m annual composite remains on disk as an alternate cross-source.
- **2026-05-13: ind32 Anlagefrei uses BAFU's 4-class 500m raster** (`LABES_32_LV95_2020_classed`), upgraded from biogeoregion XLSX. Values 1..4 directly map to BAFU's published 4-category classification (völlig anlagefrei / vereinzelte / wenige / viele Anlagen).
- **2026-05-13: ind9 Zerschneidung carries BAFU's exact 2020 Meff values per biogeoregion** (extracted from per-biogeoregion sheets of `LABES_Zerschneidung_9a_corr.xlsx`); biogeoregion-level pass at 0 breaches.
- **2026-05-13: ind_neu7 + ind31a switched to nearest-feature distance.** They were biogeoregion XLSX km totals; now they emit metres from each manifest point to the nearest power line / freely-walkable shore segment respectively.
- **2026-05-13: ind_neu2 Abgeschiedenheit switched from biogeoregion fallback to true per-pixel sampling.** Required installing `osgeo` Python bindings + extracting `LABES_NEU_2_distance_2020_mins` from the BAFU FileGDB to standalone GeoTIFF. Verified per-landmark: Zürich HB 0.48 min, Aletschgletscher 670.89 min.
- **2026-05-13: ind7 Versiegelung is a BAFU-derived binary mask** (`LABES_7_Versiegelung_1318_raster_r`), NOT the underlying NOAS04 land-use code raster (which was incorrectly picked initially). BAFU computes "is this pixel sealed" by mapping NOAS04 codes through a sealing-fraction lookup; the binary output mask is the published indicator product.
- **2026-05-13: GDAL OpenFileGDB raster driver unlocked via `conda install gdal=3.11`.** Initially failed because `osgeo` Python bindings weren't installed; libgdal-core 3.11 native was already present. After install, 6 BAFU `.gdb` files yielded a total of **~80 raster subdatasets** (ind_neu2: 12, ind14: 23, ind7: 7, ind15: 6, ind32: 4, ind_neu14: 22, ind_neu12: 2, ind_neu9: 5). Triggered the resolution audit + multiple raster upgrades above.
- **2026-05-13: empirical NOAS04 class calibration script added** (`scripts/calibrate_noas04_classes.py`). Reads the NOAS04 code raster (extracted from `LABES_7_Versiegelung_1318_raster_update`), rasterises the BAFU biogeoregion polygons (filtering out the spurious code-7 wrapper polygon that previously overrode codes 4/5 during rasterise), aggregates pixel counts per (code × biogeoregion), and finds the class set whose per-biogeoregion sum best matches BAFU's published per-indicator totals. Result: clean fit for ind3 settlement (±6.4% / region); ind1 source mismatch resolved by switching to official BFS Arealstatistik points (forest = `AS_72` 50..57).

## Open follow-ups

- **Full 7.2M enrichment run.** Currently only the 10K validation slice has been produced. All 34 indicators wired; running the pipeline with `--phase ALL` (no `--limit`) should produce the final sidecar.
- **ind1/ind17a validation semantics.** Methods are resolved and implemented at 100m from official BFS points, but the current `pdf_numerical` check compares manifest-sampled point means against published area aggregates (percent or k_ha), which is expected to diverge on image-biased samples.

## Changelog

- **2026-05-13 (latest)** — **ind17a Sömmerungsweiden switched to official BFS Arealstatistik 100m point source** (`geodata/external/arealstatistik/arealstatistik_2056.csv.zip`) and protocol class set `AS_72={45,46,47,48,49}` per BAFU 17a method text. Wired from `xlsx_biogeoregion_via_polygon` to `arealstatistik_point_sample` at per-point 100m granularity. Independent biogeoregion check versus BAFU 2013/18 table: Jura 43.159 vs 43.626 (−1.1%), Mittelland 0.640 vs 0.668 (−4.2%), Alpennordflanke 224.143 vs 224.731 (−0.3%), westl. Zentralalpen 59.820 vs 59.819 (~0.0%), östl. Zentralalpen 142.976 vs 142.976 (0.0%), Alpensüdflanke 31.474 vs 31.474 (0.0%).

- **2026-05-13 (latest)** — **ind1 Wald switched to official BFS Arealstatistik 100m point source** (`geodata/external/arealstatistik/arealstatistik_2056.csv.zip`, layer `ch.bfs.arealstatistik`) with forest classes `AS_72={50..57}`. Wired new join method `arealstatistik_point_sample` and updated spec from biogeoregion fallback to per-point 100m classification. Independent check against BAFU table: national 29.51% and all six biogeoregions within ~0.2 percentage points.

- **2026-05-13 (latest)** — **ind3 Flächenverbrauch Siedlung upgraded to per-pixel binary mask** via NOAS04 class_set `{11, 12, 13, 15, 16, 17}` (empirically calibrated, ±6.4% / biogeoregion against BAFU 2009 totals). Added `class_set` parameter to `raster_point_sample` for categorical → binary derivations. Per-pixel raster indicators now total **9** (was 8). Biogeoregion-only indicators down from 6 → 5. On the 10K validation: 3,187 / 10,000 manifest points classified as settlement (31.9%) — plausible for an image-density-biased manifest.
- **2026-05-13 (later)** — installed `osgeo gdal` via conda-forge → unlocked FileGDB raster reading. Switched **ind_neu2 Abgeschiedenheit** to per-pixel 100m raster (BAFU's official distance-in-minutes raster); switched **ind14 Licht** to BAFU's 500m VCMSL (from 1000m); switched **ind32 Anlagefrei** to BAFU's 4-class 500m classed raster (from biogeoregion); switched **ind_neu14 Städtisches Grün** to BAFU NDVI 2020 at 10m (with ×1e-4 scale); switched **ind_neu12 LWI** to BAFU 10m intensity raster; switched **ind7 Versiegelung** to the BAFU-derived binary sealing mask (corrected from picking the NOAS04 land-use code raster initially); switched **ind15 Naturüberlassene** to BAFU's binary alpine wilderness mask. Switched **ind_neu7 Leitungen** and **ind31a Gewässerabschnitte** to `nearest_feature_distance` over the actual line vectors. Fixed nodata handling for binary Int8 mask rasters (rasterio's `masked=True` was treating 0 as nodata). Added `class_set` parameter to `raster_point_sample` for categorical-raster binary derivations. Wrote NOAS04 calibration script.
- **2026-05-13 (later)** — **ind4 Verkehr strict LABES compliance committed.** Switched from biogeoregion XLSX to `nearest_feature_distance` against `TLM_Liniennetz_9a` (the combined 2020 BAFU road and rail network used for the fragmentation indicator) in `LABES_9_Zerschneidung_2020.gdb`. Nearest-feature indicators now total 4; biogeoregion drops to 4.
- **2026-05-13** — initial comprehensive progress doc created. State at this point: 34/34 indicators wired, 8 raster + 3 distance + 14 muni + 1 polygon-cell + 6 biogeoregion + 2 national. Reproducibility verified (byte-for-byte identical re-runs).
