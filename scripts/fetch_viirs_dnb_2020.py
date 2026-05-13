"""
Fetch a 2020 annual VIIRS DNB composite over Switzerland.

BAFU's LABES 14 (Lichtemissionen) is computed on a 500 m VIIRS Day/Night Band
grid, but the raw raster is not shipped in geodata/LABES/labes_14_licht/ —
only the yearly statistics XLSX. We build a local Switzerland-extent annual
composite from NOAA EOG monthly products mirrored on the public AWS bucket
`globalnightlight` (no auth required).

For each of 12 months of 2020 we:
  1. Open the global avg_rade9.tif via `/vsicurl/` (HTTP range reads).
  2. Read just the Switzerland geographic window.
  3. Accumulate sum and valid-pixel count.

Final output: the mean monthly radiance over 2020 over Switzerland, saved as
a GeoTIFF in WGS84 (EPSG:4326) at the native ~500 m product resolution.

Output:
  geodata/external/viirs_dnb_npp_2020_switzerland_wgs84.tif

Run from project root:
  conda run -n baukultur_vpr python -u scripts/fetch_viirs_dnb_2020.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# Switzerland extent in WGS84 with a small margin.
# CH bbox approx: lon 5.96-10.49, lat 45.82-47.81.
CH_BOUNDS_WGS84 = (5.7, 45.7, 10.6, 47.9)

YEAR = 2020
SENSOR = "npp"
PROCESSING = "ops"  # "rp2" replaces "ops" for older NPP files; 2020 uses "ops"
VARIANT = "ecm-slcorr"  # empirical-cloud-mask + stray-light-corrected; the standard product

OUT_DIR = Path("/home/jubooz/landscape_signatures/geodata/external")
OUT_PATH = OUT_DIR / f"viirs_dnb_{SENSOR}_{YEAR}_switzerland_wgs84.tif"


def monthly_url(year: int, month: int) -> str:
    yyyymm = f"{year}{month:02d}"
    folder = f"composites/{SENSOR}_{yyyymm}_{PROCESSING}"
    days_in_month = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    fname = (
        f"DNB_{SENSOR}_{yyyymm}01-{yyyymm}{days_in_month:02d}_global_"
        f"{VARIANT}_v10_{PROCESSING}.avg_rade9.tif"
    )
    return f"/vsicurl/https://globalnightlight.s3.amazonaws.com/{folder}/{fname}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Make rasterio use a longer read timeout and small block size for ranged GETs.
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("VSI_CACHE", "TRUE")

    accum = None
    count = None
    profile = None
    window = None

    for month in range(1, 13):
        url = monthly_url(YEAR, month)
        t0 = time.time()
        try:
            with rasterio.open(url) as src:
                if window is None:
                    window = from_bounds(*CH_BOUNDS_WGS84, transform=src.transform)
                    window = window.round_offsets().round_lengths()
                    print(
                        f"[fetch] window {int(window.width)}x{int(window.height)} "
                        f"at offset ({int(window.col_off)},{int(window.row_off)})",
                        file=sys.stderr,
                    )
                    profile = src.profile.copy()
                    profile.update(
                        width=int(window.width),
                        height=int(window.height),
                        transform=src.window_transform(window),
                        compress="deflate",
                        predictor=2,
                        tiled=True,
                        BIGTIFF="IF_SAFER",
                    )

                arr = src.read(1, window=window).astype(np.float64)
                # vcmsl / ecm-slcorr products use nodata or near-zero floors; trust the
                # file's nodata if set, otherwise treat NaN-like values as missing.
                if src.nodata is not None:
                    valid = arr != src.nodata
                else:
                    valid = np.isfinite(arr)
        except Exception as e:
            print(f"[fetch] {YEAR}-{month:02d} FAILED: {e}", file=sys.stderr)
            continue

        if accum is None:
            accum = np.zeros_like(arr)
            count = np.zeros_like(arr, dtype=np.int32)
        accum[valid] += arr[valid]
        count[valid] += 1
        dt = time.time() - t0
        print(
            f"[fetch] {YEAR}-{month:02d} ok, "
            f"min={arr[valid].min() if valid.any() else 'na'} "
            f"max={arr[valid].max() if valid.any() else 'na'} "
            f"valid_frac={valid.mean():.3f}  dt={dt:.1f}s",
            file=sys.stderr,
        )

    if accum is None:
        print("[fetch] no months succeeded", file=sys.stderr)
        return 1

    mean = np.where(count > 0, accum / np.maximum(count, 1), np.nan).astype(np.float32)

    profile.update(dtype="float32", count=1, nodata=float("nan"))
    with rasterio.open(OUT_PATH, "w", **profile) as dst:
        dst.write(mean, 1)

    print(
        f"[fetch] wrote {OUT_PATH}  shape={mean.shape}  "
        f"mean_radiance_overall={np.nanmean(mean):.3f}  "
        f"frac_with_data={np.mean(count > 0):.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
