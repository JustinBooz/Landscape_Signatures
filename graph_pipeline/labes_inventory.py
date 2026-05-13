"""
LABES data inventory probe.

Walks every geodata/LABES/labes_*/ folder and reports actual data assets:
- File geodatabase layers + their geometry type + feature count + key columns
- Standalone shapefiles + columns
- Raster grids (Arc/Info .adf, .tif) + dimensions + CRS
- Precomputed statistics workbooks (XLSX) + sheet names + the first row
- PDF protocol document path

Output is YAML written to graph_pipeline/outputs/labes/inventory.yaml. The YAML
becomes the factual basis for config/labes_indicators.yaml — we only claim a
spatial granularity for an indicator that this probe verified.

Run via the project conda env:
    conda run -n baukultur_vpr python -u graph_pipeline/labes_inventory.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path

import yaml

LABES_DIR = Path("/home/jubooz/landscape_signatures/geodata/LABES")
OUT_PATH = Path("/home/jubooz/landscape_signatures/graph_pipeline/outputs/labes/inventory.yaml")


def _probe_gdb(gdb_path: Path) -> dict:
    import geopandas as gpd

    result = {"path": str(gdb_path), "vector_layers": [], "errors": []}
    try:
        # geopandas / pyogrio surfaces the vector layers; raster sub-datasets are not
        # listed here. If pyogrio can't open the .gdb at all (e.g. raster-only or
        # missing OpenFileGDB driver build), we record that and move on.
        import pyogrio

        layers = pyogrio.list_layers(str(gdb_path))
        for entry in layers:
            name, geom_type = entry[0], entry[1]
            info = {"name": name, "geometry_type": geom_type}
            try:
                gdf = gpd.read_file(gdb_path, layer=name, rows=1)
                info["columns"] = list(gdf.columns)
                info["crs"] = str(gdf.crs)
                # Cheap feature count via pyogrio info
                try:
                    meta = pyogrio.read_info(str(gdb_path), layer=name)
                    info["features"] = int(meta.get("features", 0) or 0)
                except Exception:
                    pass
            except Exception as e:
                info["read_error"] = str(e).splitlines()[0]
            result["vector_layers"].append(info)
    except Exception as e:
        result["errors"].append(f"layer_listing: {str(e).splitlines()[0]}")

    # Try raster open via rasterio. Most ESRI raster-only .gdb files won't open via
    # the OpenFileGDB driver in this env; if vector_layers is empty AND rasterio
    # rejects it, we mark the .gdb as "likely_raster_only_unreadable" so the spec
    # has to fall back to the precomputed zonal-stats XLSX.
    if not result["vector_layers"] and not result["errors"]:
        result["errors"].append("layer_listing: no vector layers (likely raster-only)")
    try:
        import rasterio  # noqa: F401

        with __import__("rasterio").open(str(gdb_path)) as src:
            result["raster"] = {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs),
                "res": list(src.res),
            }
    except Exception as e:
        if not result["vector_layers"]:
            result["likely_raster_only_unreadable"] = str(e).splitlines()[0]

    return result


def _probe_shp(shp_path: Path) -> dict:
    import geopandas as gpd

    result = {"path": str(shp_path)}
    try:
        gdf = gpd.read_file(shp_path, rows=1)
        result["columns"] = list(gdf.columns)
        result["crs"] = str(gdf.crs)
        result["geometry_type"] = gdf.geometry.iloc[0].geom_type if len(gdf) else None
        # Lazy feature count via pyogrio
        try:
            import pyogrio

            meta = pyogrio.read_info(str(shp_path))
            result["features"] = int(meta.get("features", 0) or 0)
        except Exception:
            pass
    except Exception as e:
        result["error"] = str(e).splitlines()[0]
    return result


def _probe_raster_dir(adf_or_tif_path: Path) -> dict:
    """Probe an Arc/Info grid directory (parent of w001001.adf) or a standalone tif."""
    import rasterio

    result = {"path": str(adf_or_tif_path)}
    try:
        with rasterio.open(adf_or_tif_path) as src:
            result["width"] = src.width
            result["height"] = src.height
            result["bands"] = src.count
            result["crs"] = str(src.crs)
            result["dtype"] = str(src.dtypes[0])
            result["bounds"] = list(src.bounds)
            result["res"] = list(src.res)
            try:
                result["nodata"] = src.nodata
            except Exception:
                pass
            # Sample a tiny window to confirm readable
            window_data = src.read(1, window=((0, min(8, src.height)), (0, min(8, src.width))))
            result["sample_min"] = float(window_data.min())
            result["sample_max"] = float(window_data.max())
    except Exception as e:
        result["error"] = str(e).splitlines()[0]
    return result


def _probe_xlsx(xlsx_path: Path) -> dict:
    import pandas as pd

    result = {"path": str(xlsx_path)}
    try:
        xf = pd.ExcelFile(xlsx_path)
        result["sheets"] = []
        for sheet in xf.sheet_names:
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=2)
                result["sheets"].append(
                    {
                        "name": sheet,
                        "columns": list(map(str, df.columns)),
                        "n_preview_rows": len(df),
                    }
                )
            except Exception as e:
                result["sheets"].append({"name": sheet, "error": str(e).splitlines()[0]})
    except Exception as e:
        result["error"] = str(e).splitlines()[0]
    return result


def _probe_csv(csv_path: Path) -> dict:
    import pandas as pd

    result = {"path": str(csv_path)}
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", nrows=2, encoding="utf-8-sig")
        result["columns"] = list(map(str, df.columns))
        result["n_columns"] = len(df.columns)
    except Exception as e:
        result["error"] = str(e).splitlines()[0]
    return result


def _infer_granularity(folder_report: dict) -> str:
    """Best-guess granularity tag based on the data assets present."""
    has_raster = bool(folder_report["assets"]["rasters"]) or any(
        gdb.get("raster") or gdb.get("raster_subdatasets")
        for gdb in folder_report["assets"]["geodatabases"]
    )
    if has_raster:
        return "raster_pixel"
    layer_names = []
    for gdb in folder_report["assets"]["geodatabases"]:
        layer_names.extend(L["name"] for L in gdb.get("vector_layers", []))
    layer_names_lc = " ".join(layer_names).lower()
    if "gemeinde" in layer_names_lc or "bfs" in layer_names_lc or "municipality" in layer_names_lc:
        return "municipality_polygon"
    if "biogeoreg" in layer_names_lc:
        return "biogeoregion_polygon"
    for shp in folder_report["assets"]["shapefiles"]:
        cols = " ".join(shp.get("columns", [])).lower()
        if "bfs" in cols or "gemnr" in cols or "municipality" in cols:
            return "municipality_polygon"
    if folder_report["assets"]["xlsx"]:
        return "table_only"
    return "unknown"


def probe_folder(folder: Path) -> dict:
    report: dict = {
        "folder": str(folder),
        "indicator_dir": folder.name,
        "assets": {
            "pdfs": [],
            "geodatabases": [],
            "shapefiles": [],
            "rasters": [],
            "xlsx": [],
            "csvs": [],
        },
    }
    for child in sorted(folder.rglob("*")):
        if child.is_dir():
            if child.suffix == ".gdb":
                report["assets"]["geodatabases"].append(_probe_gdb(child))
        elif child.is_file():
            name = child.name.lower()
            if name.endswith(".pdf"):
                report["assets"]["pdfs"].append(str(child))
            elif name.endswith(".shp"):
                # Skip example shapefiles bundled with third-party tools
                if "USM Tool" in str(child) or "Behnisch Tool" in str(child) or "Example" in str(child):
                    continue
                report["assets"]["shapefiles"].append(_probe_shp(child))
            elif name == "hdr.adf":
                # Arc/Info grid directory marker — probe the containing dir
                grid_dir = child.parent
                if "USM Tool" in str(grid_dir):
                    continue
                report["assets"]["rasters"].append(_probe_raster_dir(grid_dir))
            elif name.endswith(".tif") or name.endswith(".tiff"):
                if "USM Tool" in str(child) or "Behnisch Tool" in str(child) or "Example" in str(child):
                    continue
                report["assets"]["rasters"].append(_probe_raster_dir(child))
            elif name.endswith(".xlsx") or name.endswith(".xls"):
                report["assets"]["xlsx"].append(_probe_xlsx(child))
            elif name.endswith(".csv"):
                report["assets"]["csvs"].append(_probe_csv(child))
    report["inferred_granularity"] = _infer_granularity(report)
    return report


def main() -> int:
    folders = sorted(d for d in LABES_DIR.iterdir() if d.is_dir() and d.name.startswith("labes_"))
    inventory = {"labes_dir": str(LABES_DIR), "indicators": []}
    for folder in folders:
        print(f"[probe] {folder.name} ...", file=sys.stderr)
        try:
            inventory["indicators"].append(probe_folder(folder))
        except Exception:
            inventory["indicators"].append(
                {"folder": str(folder), "fatal_error": traceback.format_exc(limit=2)}
            )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(inventory, f, sort_keys=False, allow_unicode=True, width=120)
    print(f"[probe] wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
