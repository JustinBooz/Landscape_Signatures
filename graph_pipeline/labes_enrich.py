"""
LABES sidecar enrichment for the manifest, version 2.

Reads `config/labes_indicators.yaml` as single source of truth and writes:
  graph_pipeline/outputs/labes/labes_enriched.parquet      sidecar (global_index + indicators)
  graph_pipeline/outputs/labes/labes_join_registry.json    machine-readable run registry
  graph_pipeline/outputs/labes/validation_report.md        human-readable rigor-check log
  graph_pipeline/outputs/labes/validation_log.jsonl        per-indicator structured log

Indicators are wired phase by phase; this script can be invoked with
`--phase A` or `--ind <column>` to run only a subset. The schema is stable
across runs; indicators not yet implemented (or that fail in this run) are
emitted as a null column so downstream code sees the canonical 34-column shape.

Run via:
    conda run -n baukultur_vpr python -u graph_pipeline/labes_enrich.py \
        --phase A --limit 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # type: ignore  # noqa: E402
import labes_joins  # type: ignore  # noqa: E402
import labes_validation as labes_val  # type: ignore  # noqa: E402

logger = config.setup_logging(
    "labes_enrich",
    log_file=os.path.join(config.BASE_DIR, "labes_enrich.log"),
)

PROJECT_ROOT = Path("/home/jubooz/landscape_signatures")
SPEC_PATH = PROJECT_ROOT / "config/labes_indicators.yaml"
LANDMARKS_PATH = PROJECT_ROOT / "config/labes_landmarks.yaml"
OUTPUT_DIR = Path(config.OUTPUT_DIR) / "labes"
OUTPUT_PATH = OUTPUT_DIR / "labes_enriched.parquet"
REGISTRY_PATH = OUTPUT_DIR / "labes_join_registry.json"

MANIFEST_COLUMNS = ["global_index", "lv95_easting", "lv95_northing"]


PHASE_GROUPS = {
    "A": [
        "social_csv_municipality",
        "social_csv_vector_to_string",
        "unsupported_vector_parameter",
        "unsupported_no_single_index",
    ],
    "B": [
        "polygon_municipality_field",
        "polygon_biogeoregion_field",
        "polygon_cell_field",
        "xlsx_biogeoregion_via_polygon",
        "xlsx_biogeoregion_rows",
        "sample_based_biogeoregion",
        "xlsx_national_constant",
        "spec_biogeoregion_values",
    ],
    "C": ["raster_point_sample", "arealstatistik_point_sample"],
    "D": ["nearest_feature_distance"],
}


def load_spec() -> dict:
    with open(SPEC_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_landmarks() -> list[Mapping]:
    if not LANDMARKS_PATH.exists():
        return []
    with open(LANDMARKS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("landmarks") or [])


def load_manifest(limit: Optional[int]) -> pd.DataFrame:
    logger.info("Loading manifest coordinates...")
    df = pd.read_parquet(config.manifest_path(), columns=MANIFEST_COLUMNS)
    if limit is not None:
        df = df.head(limit).copy()
    logger.info("  Loaded %s rows", f"{len(df):,}")
    return df


def select_indicators(spec: dict, phase: Optional[str], ind_filter: Optional[list[str]]) -> list[Mapping]:
    selected: list[Mapping] = []
    allowed_methods = set(PHASE_GROUPS.get(phase, [])) if phase else None
    for ind in spec["indicators"]:
        if ind_filter and ind["column_name"] not in ind_filter:
            continue
        if allowed_methods is not None and ind.get("join_method") not in allowed_methods:
            continue
        selected.append(ind)
    return selected


def run_phase(
    spec: dict,
    manifest: pd.DataFrame,
    indicators: list[Mapping],
    phase_label: str,
    landmarks: list[Mapping],
) -> tuple[pd.DataFrame, dict]:
    ctx = labes_joins.build_context(manifest, spec)
    output = manifest[["global_index"]].copy()
    status: dict = {"phase": phase_label, "indicators": []}
    validations = []

    for ind in indicators:
        col = ind["column_name"]
        record: dict = {"column": col, "method": ind.get("join_method")}
        try:
            series = labes_joins.dispatch(ctx, ind)
            series.index = ctx.manifest["global_index"].values
            output[col] = series.reindex(output["global_index"].values).values
            record["status"] = "joined"
            # Landmark sampling (only meaningful for raster indicators in this rev).
            landmark_values = labes_joins.sample_landmarks(ind, landmarks, series)
            val = labes_val.validate_indicator(
                enriched=pd.DataFrame(
                    {
                        "global_index": output["global_index"].values,
                        col: output[col].values,
                        "biogreg_c6": ctx.biogeoregion_per_point.reindex(
                            output["global_index"].values
                        ).values,
                    }
                ),
                biogeoregion_id_col="biogreg_c6",
                indicator_spec=ind,
                biogeoregion_keys=spec.get("biogeoregion_keys", {}),
                landmarks_with_expectations=landmarks,
                values_at_landmarks=landmark_values,
            )
            validations.append(val)
            record["n_valid"] = int(val.n_valid)
            record["null_fraction"] = float(val.null_fraction)
            record["checks"] = {c.name: {"passed": c.passed, "summary": c.summary} for c in val.checks}
        except Exception as exc:
            logger.exception("[%s] FAILED", col)
            output[col] = np.nan
            record["status"] = "failed"
            record["error"] = str(exc).splitlines()[0]
        status["indicators"].append(record)

    labes_val.append_report(phase_label, validations)
    return output, status


def write_outputs(enriched: pd.DataFrame, runs: list[dict], args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".parquet.tmp")
    enriched.to_parquet(tmp, index=False)
    os.rename(tmp, OUTPUT_PATH)
    logger.info("Wrote %s (rows=%s cols=%s)", OUTPUT_PATH, f"{len(enriched):,}", len(enriched.columns))

    registry = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec_path": str(SPEC_PATH),
        "manifest": config.manifest_path(),
        "output": str(OUTPUT_PATH),
        "args": vars(args),
        "runs": runs,
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", REGISTRY_PATH)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LABES sidecar enrichment v2 (spec-driven).")
    p.add_argument("--phase", choices=["A", "B", "C", "D", "ALL"], default="ALL",
                   help="Phase A=social, B=polygon/biogeoregion/national, C=raster, D=nearest-feature.")
    p.add_argument("--ind", action="append", help="Restrict to this column_name (repeatable).")
    p.add_argument("--limit", type=int, default=None, help="Only process first N manifest rows.")
    p.add_argument("--no-overwrite", action="store_true",
                   help="If labes_enriched.parquet exists, merge new indicator columns instead of overwriting.")
    return p.parse_args()


def main(args: argparse.Namespace) -> int:
    spec = load_spec()
    manifest = load_manifest(args.limit)
    landmarks = load_landmarks()

    phases = ["A", "B", "C", "D"] if args.phase == "ALL" else [args.phase]

    # Truncate the validation report + log at the start of each run so the
    # files always reflect the current run's state (rather than accumulating
    # across runs).
    labes_val.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# LABES enrichment — validation report\n\n"
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} · "
        f"phase={args.phase} · limit={args.limit} · ind={args.ind} · "
        f"no_overwrite={args.no_overwrite}\n"
    )
    labes_val.REPORT_PATH.write_text(header, encoding="utf-8")
    labes_val.LOG_PATH.write_text("", encoding="utf-8")

    runs: list[dict] = []
    enriched_all: Optional[pd.DataFrame] = None
    if args.no_overwrite and OUTPUT_PATH.exists():
        enriched_all = pd.read_parquet(OUTPUT_PATH)
        # restrict to current limit
        enriched_all = enriched_all[enriched_all["global_index"].isin(manifest["global_index"])].copy()
    else:
        enriched_all = manifest[["global_index"]].copy()
        # initialize every spec'd column as null (stable schema)
        for ind in spec["indicators"]:
            enriched_all[ind["column_name"]] = np.nan

    for phase in phases:
        indicators = select_indicators(spec, phase=phase, ind_filter=args.ind)
        if not indicators:
            logger.info("[phase %s] no matching indicators; skipping", phase)
            continue
        logger.info("[phase %s] %s indicators: %s", phase, len(indicators),
                    ", ".join(i["column_name"] for i in indicators))
        phase_out, status = run_phase(spec, manifest, indicators, f"Phase {phase}", landmarks)
        runs.append(status)
        for col in phase_out.columns:
            if col == "global_index":
                continue
            enriched_all[col] = phase_out.set_index("global_index")[col].reindex(
                enriched_all["global_index"]
            ).values

    write_outputs(enriched_all, runs, args)
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
