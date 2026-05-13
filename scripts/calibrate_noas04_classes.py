"""
Calibrate the NOAS04 land-use code -> LABES indicator class membership
empirically against BAFU's published per-biogeoregion totals.

Important Arealstatistik findings (2026-05-13):
    The missing "rule" is not an extra geographic mask. The protocol's forest
    percentages are reproduced when using the official BFS Arealstatistik points
    (ch.bfs.arealstatistik) and classifying forest as AS_72 codes 50..57.
    Using the extracted LABES7 NOAS04 raster as a surrogate for ind1 introduces a
    source mismatch and distorts biogeoregion forest shares.
    For ind17a, protocol-consistent classes are AS_72 = 45..49 from the same
    official Arealstatistik source.

Approach:
  1. Read BAFU's NOAS04 raster (LV95 100m, 1 pixel = 1 ha) and rasterize
     the biogeoregion polygons to the same grid.
  2. For each NOAS04 code, count how many pixels fall in each biogeoregion.
    3. For each LABES indicator we want to re-derive (ind3; exploratory ind17a),
     load BAFU's published per-biogeoregion totals and search for the class
     subset whose per-biogeoregion sums best match the published totals.
  4. Print the chosen class sets so we can wire them into config/labes_indicators.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely import affinity

NOAS04_RASTER_URI = "/home/jubooz/landscape_signatures/geodata/external/labes_rasters/noas04_landuse_codes_lv95.tif"
OFFICIAL_AREALSTATISTIK_CSV = (
    "/home/jubooz/landscape_signatures/geodata/external/arealstatistik/"
    "arealstatistik_2056.csv.zip"
)
BIOGEOREGION_GDB = (
    "/home/jubooz/landscape_signatures/geodata/LABES/labes_36_zersiedlung/"
    "LABES_36_GIS/LABES_36_Zersiedlung_2018.gdb"
)

# BAFU per-biogeoregion totals from each indicator's published XLSX.
# (Values are in 1000 ha; 1 pixel = 1 ha, so 1000 ha = 1000 pixels.)
BAFU_TOTALS_K_HA = {
    "ind3_flaechenverbrauch_siedlung_2009": {
        "jura": 32.2,
        "mittelland": 163.85,
        "alpennordflanke": 46.187,
        "westliche_zentralalpen": 14.283,
        "oestliche_zentralalpen": 8.965,
        "alpensuedflanke": 16.298,
    },
    "ind17a_soemmerungsweiden_2009": {
        "jura": 0.688,
        "mittelland": 0.681,
        "alpennordflanke": 231.185,
        "westliche_zentralalpen": 63.938,
        "oestliche_zentralalpen": 149.247,
        "alpensuedflanke": 35.061,
    },
}
# ind1: BAFU publishes percentages, not absolute area. We'll convert percent to
# k_ha by multiplying by each biogeoregion's TOTAL area (from the NOAS04 raster).
BAFU_FOREST_PCT_2018 = {
    "jura": 47.4,
    "mittelland": 23.1,
    "alpennordflanke": 31.5,
    "westliche_zentralalpen": 21.2,
    "oestliche_zentralalpen": 23.1,
    "alpensuedflanke": 41.5,
}

# BAFU's ind1 protocol cites the Arealstatistik NOLC04 27-category source and
# only mentions 50-57 in the forest-change context. That means the remaining
# gap is likely a source/lookup mismatch, not an extra geographic mask; keep
# ind1 calibration constrained to the documented baseline instead of widening
# the class pool arbitrarily.
IND1_FOREST_CLASSES: tuple[int, ...] = tuple(range(50, 58))

BIOGEOREGION_NAMES = {
    1: "jura",
    2: "mittelland",
    3: "alpennordflanke",
    4: "westliche_zentralalpen",
    5: "oestliche_zentralalpen",
    6: "alpensuedflanke",
}


def build_pixel_count_matrix() -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a DataFrame indexed by NOAS04 code, columns = biogeoregion name,
    values = number of pixels in (code, biogeoregion). Plus a dict of total
    pixel area per biogeoregion (km²)."""
    print(f"[calib] reading NOAS04 raster…", file=sys.stderr)
    with rasterio.open(NOAS04_RASTER_URI) as ds:
        arr = ds.read(1)
        transform = ds.transform
        height, width = ds.height, ds.width
        crs = ds.crs

    print(f"[calib] raster {width}x{height}, dtype {arr.dtype}, crs {crs}", file=sys.stderr)

    print(f"[calib] rasterizing biogeoregion polygons…", file=sys.stderr)
    bg = gpd.read_file(BIOGEOREGION_GDB, layer="biogeoregio_2018")
    bg = bg.to_crs(crs)
    # Filter out code 7 (a wrapper / lakes polygon overlapping the 6 biogeoregions
    # that, processed last by rasterize, would overwrite codes 4 and 5).
    bg = bg[bg["biogreg_c6"].isin([1, 2, 3, 4, 5, 6])].copy()
    shapes = [(geom, int(code)) for geom, code in zip(bg.geometry, bg["biogreg_c6"])]
    bg_raster = cast(
        np.ndarray,
        rasterize(
        shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.int16
        ),
    )

    # Build per-(code, biogeoregion) pixel counts.
    print(f"[calib] building (code × biogeoregion) matrix…", file=sys.stderr)
    df = pd.DataFrame({"code": arr.flatten(), "bg": bg_raster.flatten()})
    df = df[(df["bg"] > 0) & df["bg"].isin(BIOGEOREGION_NAMES.keys())]
    counts = df.groupby(["code", "bg"]).size().unstack(fill_value=0)
    counts.columns = [BIOGEOREGION_NAMES[int(c)] for c in counts.columns]
    counts = counts.reindex(columns=list(BIOGEOREGION_NAMES.values()), fill_value=0)
    counts = counts.sort_index()
    counts.index.name = "noas04_code"

    bg_pixel_totals = {name: int(counts[name].sum()) for name in counts.columns}
    print(f"[calib] total pixels per biogeoregion (1 px = 1 ha):", file=sys.stderr)
    for k, v in bg_pixel_totals.items():
        print(f"  {k}: {v:,} ({v/1000:.1f} k_ha)", file=sys.stderr)
    return counts, bg_pixel_totals


def expected_k_ha_for_forest(bg_pixel_totals: dict[str, int]) -> dict[str, float]:
    """ind1 reference is per-biogeoregion % forest. Convert to k_ha by
    multiplying by each biogeoregion's total pixel count (1 px = 1 ha)."""
    return {
        name: BAFU_FOREST_PCT_2018[name] / 100.0 * pixels / 1000.0
        for name, pixels in bg_pixel_totals.items()
    }


def score(class_set: tuple[int, ...], counts: pd.DataFrame, target_k_ha: dict[str, float]) -> tuple[float, dict[str, float]]:
    available = [c for c in class_set if c in counts.index]
    if not available:
        return float("inf"), {}
    selected_k_ha = {
        str(name): float(value)
        for name, value in (counts.loc[available].sum(axis=0) / 1000.0).to_dict().items()
    }  # pixels → k_ha
    err = 0.0
    for name, target in target_k_ha.items():
        if target == 0:
            err += abs(selected_k_ha.get(name, 0.0))
        else:
            err += abs((selected_k_ha.get(name, 0.0) - target) / target)
    return err, selected_k_ha


def find_best_class_set(
    counts: pd.DataFrame,
    target_k_ha: dict[str, float],
    candidate_pool: list[int],
    max_subset_size: int = 8,
    seed: tuple[int, ...] = (),
) -> tuple[tuple[int, ...], float, dict[str, float]]:
    """Greedy-forward then exhaustive within neighborhood: try adding/removing
    one class from `seed` until error stops decreasing."""
    current = set(seed)
    best_err, best_vals = score(tuple(current), counts, target_k_ha)

    pool = set(candidate_pool)
    improved = True
    while improved:
        improved = False
        # Try adding any one class
        for c in pool - current:
            cand = tuple(sorted(current | {c}))
            err, vals = score(cand, counts, target_k_ha)
            if err < best_err - 1e-6:
                best_err = err
                best_vals = vals
                current = set(cand)
                improved = True
        # Try removing any one class
        for c in list(current):
            cand = tuple(sorted(current - {c}))
            err, vals = score(cand, counts, target_k_ha)
            if err < best_err - 1e-6:
                best_err = err
                best_vals = vals
                current = set(cand)
                improved = True
    return tuple(sorted(current)), best_err, best_vals


def main() -> int:
    counts, bg_pixel_totals = build_pixel_count_matrix()
    print()
    print("Per-NOAS04-code pixel counts per biogeoregion (top 25 codes by total):")
    counts_sorted = counts.assign(total=counts.sum(axis=1)).sort_values("total", ascending=False).drop(columns="total")
    print(counts_sorted.head(25).to_string())
    print()

    # The full pool: every NOAS04 code present in the raster (excluding 0 = nodata).
    pool = [c for c in counts.index.tolist() if c != 0]

    print("=" * 80)
    print("ind3 Flächenverbrauch Siedlung 2009 — best-fit NOAS04 class set:")
    target = BAFU_TOTALS_K_HA["ind3_flaechenverbrauch_siedlung_2009"]
    best, err, vals = find_best_class_set(counts, target, pool, seed=(11, 12, 13, 14, 15))
    print(f"  classes: {best}, total fractional error: {err:.3f}")
    print(f"  per-biogeoregion (k_ha) ours vs BAFU:")
    for name, t in target.items():
        print(f"    {name:30}: ours={vals.get(name, 0):.2f}, bafu={t:.2f}, Δ={(vals.get(name,0)-t)/(t or 1)*100:+.1f}%")
    print()

    print("=" * 80)
    print("ind17a Sömmerungsweiden 2009 — best-fit NOAS04 class set:")
    target = BAFU_TOTALS_K_HA["ind17a_soemmerungsweiden_2009"]
    best, err, vals = find_best_class_set(counts, target, pool, seed=(41,))
    print(f"  classes: {best}, total fractional error: {err:.3f}")
    print(f"  per-biogeoregion (k_ha) ours vs BAFU:")
    for name, t in target.items():
        print(f"    {name:30}: ours={vals.get(name, 0):.2f}, bafu={t:.2f}, Δ={(vals.get(name,0)-t)/(t or 1)*100:+.1f}%")
    print()

    print("=" * 80)
    print("ind1 Wald 2018 — best-fit NOAS04 class set:")
    target = expected_k_ha_for_forest(bg_pixel_totals)
    print(f"  derived target (from BAFU % × biogeoregion area):")
    for k, v in target.items():
        print(f"    {k:30}: {v:.2f} k_ha")
    print(f"  official BAFU forest classes: {IND1_FOREST_CLASSES}")
    err, vals = score(IND1_FOREST_CLASSES, counts, target)
    print(f"  classes: {IND1_FOREST_CLASSES}, total fractional error: {err:.3f}")
    print(f"  per-biogeoregion (k_ha) ours vs derived-bafu:")
    for name, t in target.items():
        print(f"    {name:30}: ours={vals.get(name, 0):.2f}, bafu_target={t:.2f}, Δ={(vals.get(name,0)-t)/(t or 1)*100:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
