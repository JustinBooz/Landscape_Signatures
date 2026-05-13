"""Produce a clean per-indicator summary from inventory.yaml.

For each labes_* folder, classify what's available for per-point enrichment:
- COMPREHENSIVE_VECTOR: a single Switzerland-wide vector layer with the indicator value
- COMPREHENSIVE_RASTER: a Switzerland-wide raster the manifest point can be sampled against
- MUNICIPALITY_TABLE: a per-municipality XLSX value (join via point-in-municipality polygon)
- BIOGEOREGION_TABLE: a per-biogeoregion XLSX value (join via point-in-biogeoregion polygon)
- SAMPLE_BASED_ONLY: the .gdb contains sample monitoring sites, not Switzerland-wide coverage
- NATIONAL_ONLY: only a single national scalar is published

Outputs a markdown table to stdout + JSON summary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

INV_PATH = Path("/home/jubooz/landscape_signatures/graph_pipeline/outputs/labes/inventory.yaml")
OUT_MD = Path("/home/jubooz/landscape_signatures/graph_pipeline/outputs/labes/inventory_summary.md")
OUT_JSON = Path("/home/jubooz/landscape_signatures/graph_pipeline/outputs/labes/inventory_summary.json")


SAMPLE_GDB_RE = re.compile(r"LABES_\d{3}\.gdb$")  # per-sample-site geodatabases


def classify_indicator(ind: dict) -> dict:
    folder = ind["indicator_dir"]
    assets = ind["assets"]
    out = {
        "folder": folder,
        "classification": None,
        "best_source_path": None,
        "best_source_kind": None,
        "best_source_layer": None,
        "best_source_columns": None,
        "best_source_features": None,
        "notes": [],
    }

    # 1) Switzerland-wide raster present?
    if assets["rasters"]:
        for r in assets["rasters"]:
            if "error" in r:
                continue
            w = r.get("width", 0) or 0
            h = r.get("height", 0) or 0
            if w >= 1000 and h >= 1000:
                out["classification"] = "COMPREHENSIVE_RASTER"
                out["best_source_path"] = r["path"]
                out["best_source_kind"] = "raster"
                out["notes"].append(
                    f"{w}x{h} cells, res={r.get('res')}, crs={r.get('crs','?')[:40]}"
                )
                return out

    # 2) Switzerland-wide vector layer in .gdb (excluding per-sample LABES_NNN.gdbs)
    main_gdbs = [
        g for g in assets["geodatabases"]
        if not SAMPLE_GDB_RE.search(g["path"])
        and not g.get("likely_raster_only_unreadable")
        and g.get("vector_layers")
    ]
    municipality_layer = None
    biogeoregion_layer = None
    countrywide_layer = None
    for gdb in main_gdbs:
        for L in gdb["vector_layers"]:
            cols_lc = " ".join(str(c).lower() for c in L.get("columns", []))
            name_lc = L["name"].lower()
            n = L.get("features", 0) or 0
            if any(k in name_lc for k in ("gemeinde", "_gem_", "municipality")) and n > 1000:
                municipality_layer = (gdb["path"], L)
            elif "biogeoreg" in name_lc:
                biogeoregion_layer = (gdb["path"], L)
            elif "bfs" in cols_lc and n > 1000:
                municipality_layer = (gdb["path"], L)
            elif n > 1000 and L.get("geometry_type", "").startswith("MultiPolygon"):
                countrywide_layer = (gdb["path"], L)
    if municipality_layer:
        out["classification"] = "COMPREHENSIVE_VECTOR"
        out["best_source_path"] = municipality_layer[0]
        out["best_source_kind"] = "gdb_layer"
        out["best_source_layer"] = municipality_layer[1]["name"]
        out["best_source_columns"] = municipality_layer[1].get("columns", [])
        out["best_source_features"] = municipality_layer[1].get("features")
        out["notes"].append("municipality-granularity polygon coverage")
        return out
    if countrywide_layer:
        out["classification"] = "COMPREHENSIVE_VECTOR"
        out["best_source_path"] = countrywide_layer[0]
        out["best_source_kind"] = "gdb_layer"
        out["best_source_layer"] = countrywide_layer[1]["name"]
        out["best_source_columns"] = countrywide_layer[1].get("columns", [])
        out["best_source_features"] = countrywide_layer[1].get("features")
        out["notes"].append(f"polygon coverage n={countrywide_layer[1].get('features')}")
        return out

    # 3) Standalone shapefile with BFS column?
    for shp in assets["shapefiles"]:
        cols_lc = " ".join(str(c).lower() for c in shp.get("columns", []))
        if shp.get("features", 0) > 1000 and ("bfs" in cols_lc or "gemnr" in cols_lc):
            out["classification"] = "COMPREHENSIVE_VECTOR"
            out["best_source_path"] = shp["path"]
            out["best_source_kind"] = "shapefile"
            out["best_source_columns"] = shp.get("columns", [])
            out["best_source_features"] = shp.get("features")
            out["notes"].append("shapefile with municipality key")
            return out

    # 4) XLSX with municipality / biogeoregion key in column names?
    for xl in assets["xlsx"]:
        for sheet in xl.get("sheets", []):
            cols_lc = " ".join(str(c).lower() for c in sheet.get("columns", []))
            if "biogreg" in cols_lc or "biogeo" in cols_lc:
                out["classification"] = out["classification"] or "BIOGEOREGION_TABLE"
                if out["classification"] == "BIOGEOREGION_TABLE":
                    out["best_source_path"] = xl["path"]
                    out["best_source_kind"] = "xlsx_sheet"
                    out["best_source_layer"] = sheet["name"]
                    out["best_source_columns"] = sheet.get("columns")
            elif "gemnr" in cols_lc or "bfs" in cols_lc or "gemeinde" in cols_lc:
                out["classification"] = "MUNICIPALITY_TABLE"
                out["best_source_path"] = xl["path"]
                out["best_source_kind"] = "xlsx_sheet"
                out["best_source_layer"] = sheet["name"]
                out["best_source_columns"] = sheet.get("columns")
                out["notes"].append("municipality-granularity xlsx")
                return out

    if out["classification"]:
        out["notes"].append("biogeoregion-granularity xlsx only")
        return out

    # 5) Sample-based .gdb only (LABES_NNN.gdbs)
    sample_gdbs = [g for g in assets["geodatabases"] if SAMPLE_GDB_RE.search(g["path"])]
    if sample_gdbs:
        out["classification"] = "SAMPLE_BASED_ONLY"
        out["notes"].append(f"{len(sample_gdbs)} sample-site .gdb databases; aggregate stats only via XLSX")
        # fallback to a national xlsx if available
        if assets["xlsx"]:
            out["best_source_path"] = assets["xlsx"][0]["path"]
            out["best_source_kind"] = "xlsx_national"
        return out

    if assets["xlsx"]:
        out["classification"] = "NATIONAL_ONLY"
        out["best_source_path"] = assets["xlsx"][0]["path"]
        out["best_source_kind"] = "xlsx_national"
        out["notes"].append("only XLSX present; granularity TBD by sheet inspection")
        return out

    out["classification"] = "NO_DATA"
    return out


def main() -> int:
    with open(INV_PATH) as f:
        inv = yaml.safe_load(f)

    rows = [classify_indicator(ind) for ind in inv["indicators"]]
    rows.sort(key=lambda r: r["folder"])

    md_lines = ["# LABES inventory summary", ""]
    md_lines.append(
        "| Folder | Classification | Best source | Layer / sheet | Features | Notes |"
    )
    md_lines.append("|---|---|---|---|---|---|")
    for r in rows:
        path = ""
        if r["best_source_path"]:
            path = "/".join(Path(r["best_source_path"]).parts[-2:])
        md_lines.append(
            f"| {r['folder']} | **{r['classification']}** | `{path}` "
            f"| {r['best_source_layer'] or ''} "
            f"| {r['best_source_features'] or ''} "
            f"| {'; '.join(r['notes'])} |"
        )
    md_lines.append("")

    # Count classifications
    counts: dict = {}
    for r in rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    md_lines.append("## Counts by classification")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        md_lines.append(f"- `{k}`: {v}")
    md_lines.append("")

    md_text = "\n".join(md_lines)
    OUT_MD.write_text(md_text, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(md_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
