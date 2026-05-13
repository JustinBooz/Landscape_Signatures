"""
Extract per-indicator rasters from BAFU's LABES File Geodatabases (.gdb).

Several LABES indicators ship the canonical Switzerland-wide raster INSIDE
their .gdb (e.g. labes_neu_2_abgeschiedenheit/.../LABES_NEU_2_Abgeschiedenheit.gdb).
GDAL ≥ 3.7 with the OpenFileGDB driver can read these but rasterio cannot
open the subdataset URI directly. One-shot: pull each useful subdataset out
to a standalone DEFLATE-compressed GeoTIFF in geodata/external/labes_rasters/,
keyed by indicator id, so the enrichment pipeline can use plain
raster_point_sample.

This script is idempotent — existing output GeoTIFFs are skipped unless
--force is passed.

Run:
    /home/jubooz/anaconda3/envs/baukultur_vpr/bin/python scripts/extract_labes_rasters.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from osgeo import gdal

gdal.UseExceptions()

OUT_DIR = Path("/home/jubooz/landscape_signatures/geodata/external/labes_rasters")

# (indicator_id, source_gdb_relative, subdataset_name, out_filename, description)
EXTRACTIONS = [
    (
        "neu_2",
        "geodata/LABES/labes_neu_2_abgeschiedenheit/LABES_NEU_2_GIS/LABES_NEU_2_Abgeschiedenheit.gdb",
        "LABES_NEU_2_distance_2020_mins",
        "labes_ind_neu2_abgeschiedenheit_2020_lv95.tif",
        "Remoteness in minutes (BAFU LABES NEU_2, 1 ha LV95 raster, 2020)",
    ),
    (
        "14",
        "geodata/LABES/labes_14_licht/LABES_14_GIS/LABES_14_Licht_VCMSL.gdb",
        "VIIRS_500m_LV03_2020_mask",
        "labes_ind14_licht_2020_lv03_500m.tif",
        "Night light emissions, BAFU LABES 14 VCMSL product at the FINER 500m LV03 grid (preferred over the 1000m DMSP-rescaled version for resolution).",
    ),
    (
        "7",
        "geodata/LABES/labes_7_versiegelung/LABES_7_GIS/LABES_7_Versiegelung.gdb",
        "LABES_7_Versiegelung_1318_raster_r",
        "labes_ind7_versiegelung_1318_lv03_mask.tif",
        "Soil-sealing BINARY MASK for 2013-2018 (BAFU LABES 7, 1=pixel sealed, 0=not sealed; 1 ha LV03). BAFU derives this from the NOAS04 land-use codes (also stored alongside as *_update in LV95). Per-biogeoregion totals in hectares are obtained by summing 1-pixels and reported in LABES_7_Results.xlsx.",
    ),
    (
        "15",
        "geodata/LABES/labes_15_naturuberlasseneflache/LABES_15_GIS/LABES_15_NaturuberlasseneFlache.gdb",
        "NOAS04_18_alpine_raster_recl_update",
        "labes_ind15_naturueberlassene_18_lv95.tif",
        "Nature-overlooked area, 2018 NOAS04-derived raster (BAFU LABES 15).",
    ),
    (
        "neu_14",
        "geodata/LABES/labes_neu_14_staedtischesgruen/LABES_NEU_14_GIS/LABES_NEU_14_StadtischesGrun.gdb",
        "NDVI_2020",
        "labes_ind_neu14_ndvi_2020_lv95.tif",
        "NDVI 2020 raster, BAFU LABES NEU_14 (proxy for urban greenness).",
    ),
    (
        "neu_12",
        "geodata/LABES/labes_neu_12_landwirtschaftliche_intensitaet/LABES_NEU_12_GIS/LABES_NEU_12_LWI.gdb",
        "LABES_LWI_2016_2019_new_std",
        "labes_ind_neu12_lwi_2019_lv95.tif",
        "Agricultural intensity index LWI 2016-2019 standardised, BAFU LABES NEU_12.",
    ),
    (
        "32",
        "geodata/LABES/labes_32_anlagefrei/LABES_32_GIS/LABES_32_Anlagefreie_2020.gdb",
        "LABES_32_LV95_2020_classed",
        "labes_ind32_anlagefrei_2020_lv95_classed.tif",
        "Anlagefrei (installation-free) per-pixel BAFU 4-class raster at 500m LV95, 2020. Values: 1=völlig anlagefrei (0% installations), 2=vereinzelte Anlagen (0.1-5%), 3=wenige Anlagen (5.1-10%), 4=viele Anlagen (>10%). Per-biogeoregion shares in LABES_32_Statistik.xlsx.",
    ),
]

PROJECT_ROOT = Path("/home/jubooz/landscape_signatures")


def extract(idx: int, src_gdb: Path, subdataset: str, out_path: Path, force: bool) -> Optional[str]:
    if out_path.exists() and not force:
        print(f"[{idx}] {out_path.name}: skip (already exists, {out_path.stat().st_size/1e6:.1f} MB)", file=sys.stderr)
        return None
    if not src_gdb.exists():
        return f"src missing: {src_gdb}"
    src_uri = f'OpenFileGDB:"{src_gdb}":{subdataset}'
    print(f"[{idx}] translating {subdataset} → {out_path.name}", file=sys.stderr)
    opts = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    ds = gdal.Translate(str(out_path), src_uri, options=opts)
    if ds is None:
        return f"gdal translate returned None for {subdataset}"
    ds = None
    size_mb = out_path.stat().st_size / 1e6
    print(f"     wrote {size_mb:.1f} MB", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-extract even if output exists")
    parser.add_argument("--only", action="append", help="Restrict to indicator id(s) (repeatable)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    for i, (ind_id, gdb_rel, sub, out_name, _desc) in enumerate(EXTRACTIONS, 1):
        if args.only and ind_id not in args.only:
            continue
        gdb = PROJECT_ROOT / gdb_rel
        out = OUT_DIR / out_name
        err = extract(i, gdb, sub, out, args.force)
        if err:
            failures.append((ind_id, err))

    if failures:
        print("\n=== failures ===", file=sys.stderr)
        for ind, err in failures:
            print(f"  ind {ind}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
