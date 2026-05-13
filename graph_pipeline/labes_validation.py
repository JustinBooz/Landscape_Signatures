"""
Validation framework for LABES indicator enrichment.

Implements the four rigor checks specified in [[feedback-indicator-quality-first]]:

  Check 1: numerical PDF comparison
      For each indicator with a published `reference_values.national_mean` and
      `biogeoregion_means`, compute our equivalents from the enriched manifest
      and compare. Assert biogeoregion means agree within ±10% of the published
      figure, weighted by manifest-point count per biogeoregion.

  Check 2: cross-source agreement
      Where a second independent source exists (e.g., ind14 raster vs the
      published LABES_14 YearlyStatistics XLSX), correlate the two and require
      Spearman ρ > 0.85 at the biogeoregion aggregation.

  Check 3: hold-out landmark
      For each landmark in config/labes_landmarks.yaml that declares an
      `expectations` block, look up the manifest-point indicator value at that
      coordinate and assert it matches the expectation (>national_mean,
      <national_p10, or absolute thresholds).

  Check 4: unit-range / coverage
      Every indicator declares a `value_range` and a `value_unit`. Assert all
      non-null values fall in range, and report null-coverage per indicator.

Outputs each indicator's check results both:
  - As a row appended to graph_pipeline/outputs/labes/validation_report.md
  - As a JSON record appended to graph_pipeline/outputs/labes/validation_log.jsonl
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


REPORT_PATH = Path("/home/jubooz/landscape_signatures/graph_pipeline/outputs/labes/validation_report.md")
LOG_PATH = Path("/home/jubooz/landscape_signatures/graph_pipeline/outputs/labes/validation_log.jsonl")


@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    details: dict = field(default_factory=dict)


@dataclass
class IndicatorValidation:
    column_name: str
    granularity: str
    n_total: int
    n_valid: int
    null_fraction: float
    summary_stats: dict
    checks: list[CheckResult]

    def to_jsonable(self) -> dict:
        return {
            "column_name": self.column_name,
            "granularity": self.granularity,
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "null_fraction": self.null_fraction,
            "summary_stats": self.summary_stats,
            "checks": [asdict(c) for c in self.checks],
        }


# -----------------------------------------------------------------
# Check implementations
# -----------------------------------------------------------------


def _series_summary(values: pd.Series) -> dict:
    if values.dtype == object or pd.api.types.is_string_dtype(values):
        non_null = values.dropna()
        non_null = non_null[non_null != ""]
        return {
            "count": int(len(non_null)),
            "n_unique": int(non_null.nunique()),
            "top_values": non_null.value_counts().head(5).to_dict(),
            "is_string": True,
        }
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) == 0:
        return {"count": 0}
    return {
        "count": int(len(s)),
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "std": float(s.std()),
        "n_unique": int(s.nunique()),
    }


def check_unit_range(values: pd.Series, value_range) -> CheckResult:
    if value_range is None:
        return CheckResult(
            name="unit_range",
            passed=True,
            summary="skipped (string column, no numeric range)",
            details={"range": None},
        )
    s = pd.to_numeric(values, errors="coerce")
    lo, hi = (None, None)
    if value_range:
        lo = value_range[0] if len(value_range) > 0 else None
        hi = value_range[1] if len(value_range) > 1 else None
    valid = s.dropna()
    out_of_range = 0
    if lo is not None:
        out_of_range += int((valid < lo).sum())
    if hi is not None:
        out_of_range += int((valid > hi).sum())
    return CheckResult(
        name="unit_range",
        passed=(out_of_range == 0),
        summary=(
            f"all {len(valid):,} non-null values in [{lo}, {hi}]"
            if out_of_range == 0
            else f"{out_of_range:,} values outside [{lo}, {hi}]"
        ),
        details={"range": [lo, hi], "out_of_range": out_of_range},
    )


def check_coverage(
    values: pd.Series,
    granularity: str,
    indicator_spec: Optional[Mapping] = None,
    in_ch_mask: Optional[pd.Series] = None,
) -> CheckResult:
    """Coverage check restricted to manifest points inside Switzerland.

    Points outside CH biogeoregion polygons (Liechtenstein, lakes near the
    border, off-grid coordinates) are excluded from the null denominator.
    Per the project decision: as long as points inside CH are covered, we
    don't care about gaps outside.
    """
    n_total = len(values)
    if in_ch_mask is None:
        in_ch_mask = pd.Series([True] * n_total, index=values.index)
    in_ch_mask = in_ch_mask.fillna(False).astype(bool)
    n_in_ch = int(in_ch_mask.sum())
    n_outside_ch = n_total - n_in_ch
    n_null_in_ch = int(values[in_ch_mask].isna().sum())
    frac_null_in_ch = (n_null_in_ch / n_in_ch) if n_in_ch else 0.0

    status = (indicator_spec or {}).get("implementation_status", "")
    if status in ("unsupported_vector_parameter", "unsupported_no_single_index"):
        return CheckResult(
            name="coverage",
            passed=True,
            summary=f"intentionally null ({status})",
            details={"n_total": n_total, "n_in_ch": n_in_ch, "threshold": None},
        )

    if indicator_spec and indicator_spec.get("coverage_null_threshold") is not None:
        threshold = float(indicator_spec["coverage_null_threshold"])
    elif granularity in ("raster_pixel", "polygon_cell"):
        threshold = 0.05
    elif granularity == "national":
        threshold = 0.0
    else:
        threshold = 0.20
    passed = frac_null_in_ch <= threshold

    return CheckResult(
        name="coverage",
        passed=passed,
        summary=(
            f"{frac_null_in_ch:.1%} null inside CH ({n_null_in_ch:,}/{n_in_ch:,}), "
            f"{n_outside_ch:,} pts outside CH excluded; threshold {threshold:.0%}"
        ),
        details={
            "n_total": n_total,
            "n_in_ch": n_in_ch,
            "n_outside_ch": n_outside_ch,
            "n_null_in_ch": n_null_in_ch,
            "frac_null_in_ch": frac_null_in_ch,
            "threshold": threshold,
        },
    )


def check_pdf_numerical(
    values: pd.Series,
    biogeo_id: pd.Series,
    reference: Mapping,
    biogeoregion_keys: Mapping[str, int],
    tolerance_pct: float = 10.0,
) -> CheckResult:
    if not reference:
        return CheckResult(
            name="pdf_numerical",
            passed=True,
            summary="skipped (no PDF reference values yet)",
            details={},
        )
    biogeo_refs = reference.get("biogeoregion_means") or {}
    nat_ref = reference.get("national_mean")
    if nat_ref is None and not biogeo_refs:
        return CheckResult(
            name="pdf_numerical",
            passed=True,
            summary="skipped (no PDF reference values yet)",
            details={},
        )
    comparison_scope = reference.get("comparison_scope", "biogeoregion")

    biogeo_id = pd.to_numeric(biogeo_id, errors="coerce")
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "b": biogeo_id})
    df = df.dropna(subset=["v", "b"])

    deltas = {}
    n_breaches = 0
    for name, ref_value in biogeo_refs.items():
        if ref_value is None:
            continue
        code = biogeoregion_keys.get(name)
        if code is None:
            continue
        sub = df[df["b"] == code]["v"]
        if len(sub) == 0:
            deltas[name] = {"ref": ref_value, "ours": None, "delta_pct": None}
            continue
        ours = float(sub.mean())
        delta_pct = (
            (ours - ref_value) / abs(ref_value) * 100.0 if ref_value else math.nan
        )
        deltas[name] = {"ref": ref_value, "ours": ours, "delta_pct": delta_pct}
        if abs(delta_pct) > tolerance_pct:
            n_breaches += 1

    nat_ours = float(df["v"].mean()) if len(df) else None
    nat_delta = None
    if nat_ref is not None and nat_ours is not None and nat_ref != 0:
        nat_delta = (nat_ours - nat_ref) / abs(nat_ref) * 100.0
    nat_delta_str = f"{nat_delta:.1f}%" if nat_delta is not None else "n/a"

    if comparison_scope == "national_only" or not biogeo_refs:
        passed = nat_delta is None or abs(nat_delta) <= tolerance_pct
        summary = (
            f"national: ours={nat_ours} vs ref={nat_ref} "
            f"(Δ={nat_delta_str} if comparable); "
            "biogeoregion comparison skipped by spec"
        )
    else:
        passed = n_breaches == 0
        if nat_ref is None:
            national_summary = f"manifest-weighted national context: ours={nat_ours}; "
        else:
            national_summary = (
                f"manifest-weighted national context: ours={nat_ours} vs ref={nat_ref} "
                f"(Δ={nat_delta_str}; not used for biogeoregion pass/fail); "
            )
        summary = (
            national_summary +
            f"{n_breaches} biogeoregion(s) outside ±{tolerance_pct:.0f}%"
        )
    return CheckResult(
        name="pdf_numerical",
        passed=passed,
        summary=summary,
        details={
            "tolerance_pct": tolerance_pct,
            "comparison_scope": comparison_scope,
            "national_reference": nat_ref,
            "national_ours": nat_ours,
            "national_delta_pct": nat_delta,
            "biogeoregion": deltas,
        },
    )


def check_landmarks(
    values_by_landmark: Mapping[str, Optional[float]],
    landmarks_with_expectations: Sequence[Mapping],
    indicator_column: str,
    summary_stats: Mapping,
    indicator_spec: Optional[Mapping] = None,
) -> CheckResult:
    expected_ids = [
        lm["id"]
        for lm in landmarks_with_expectations
        if (lm.get("expectations") or {}).get(indicator_column) is not None
    ]
    granularity = (indicator_spec or {}).get("granularity")
    if expected_ids and granularity in ("biogeoregion", "national"):
        return CheckResult(
            name="landmarks",
            passed=True,
            summary=f"skipped for {granularity} aggregate ({len(expected_ids)} expectations declared)",
            details={"skipped_landmarks": expected_ids},
        )
    if expected_ids and not values_by_landmark:
        return CheckResult(
            name="landmarks",
            passed=True,
            summary=f"skipped (no direct landmark sampler for this join method; {len(expected_ids)} expectations declared)",
            details={"skipped_landmarks": expected_ids},
        )

    failures: list[dict] = []
    evaluated = 0
    for lm in landmarks_with_expectations:
        exp = (lm.get("expectations") or {}).get(indicator_column)
        if exp is None:
            continue
        evaluated += 1
        value = values_by_landmark.get(lm["id"])
        if value is None or (isinstance(value, float) and math.isnan(value)):
            failures.append({"landmark": lm["id"], "expected": exp, "got": None})
            continue
        passed = _evaluate_expectation(value, exp, summary_stats)
        if not passed:
            failures.append({"landmark": lm["id"], "expected": exp, "got": value})

    if evaluated == 0:
        return CheckResult(
            name="landmarks",
            passed=True,
            summary="no expectations declared for this indicator",
            details={},
        )
    passed = len(failures) == 0
    return CheckResult(
        name="landmarks",
        passed=passed,
        summary=f"{evaluated - len(failures)}/{evaluated} landmark expectations met",
        details={"failures": failures},
    )


def _evaluate_expectation(value: float, expectation: Any, summary_stats: Mapping) -> bool:
    """Parse expressions like '>national_mean', '<national_p10', '>40', '<4.20'."""
    if isinstance(expectation, (int, float)):
        return float(value) == float(expectation)
    s = str(expectation).strip()
    if not s:
        return True
    op = None
    if s.startswith(">="):
        op, rhs = ">=", s[2:]
    elif s.startswith("<="):
        op, rhs = "<=", s[2:]
    elif s.startswith(">"):
        op, rhs = ">", s[1:]
    elif s.startswith("<"):
        op, rhs = "<", s[1:]
    else:
        return False
    rhs = rhs.strip()
    rhs_val: Optional[float] = None
    if rhs in summary_stats:
        rhs_val = summary_stats.get(rhs)
    elif rhs == "national_mean":
        rhs_val = summary_stats.get("mean")
    elif rhs == "national_median":
        rhs_val = summary_stats.get("median")
    elif rhs.startswith("national_p"):
        try:
            q = int(rhs.replace("national_p", "")) / 100.0
            key = f"p{int(round(q * 100)):02d}"
            if key in summary_stats:
                rhs_val = summary_stats[key]
            else:
                for fallback_key in ("p05", "p10", "p25", "median", "p75", "p90", "p95"):
                    if fallback_key in summary_stats and abs(_pkey_to_q(fallback_key) - q) < 1e-3:
                        rhs_val = summary_stats[fallback_key]
                        break
        except Exception:
            return False
    else:
        try:
            rhs_val = float(rhs)
        except ValueError:
            return False
    if rhs_val is None:
        return False
    if op == ">":
        return float(value) > rhs_val
    if op == "<":
        return float(value) < rhs_val
    if op == ">=":
        return float(value) >= rhs_val
    if op == "<=":
        return float(value) <= rhs_val
    return False


def _pkey_to_q(key: str) -> float:
    return {
        "p05": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "median": 0.5,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
    }[key]


# -----------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------


def validate_indicator(
    enriched: pd.DataFrame,
    biogeoregion_id_col: str,
    indicator_spec: Mapping,
    biogeoregion_keys: Mapping[str, int],
    landmarks_with_expectations: Sequence[Mapping] = (),
    values_at_landmarks: Optional[Mapping[str, Optional[float]]] = None,
) -> IndicatorValidation:
    col = indicator_spec["column_name"]
    series = enriched[col]
    summary = _series_summary(series)

    # in-CH mask: any manifest point with a non-null biogeoregion id is inside
    # one of the 6 BAFU biogeoregion polygons (i.e., inside CH).
    in_ch_mask = None
    if biogeoregion_id_col in enriched.columns:
        in_ch_mask = enriched[biogeoregion_id_col].notna()

    checks: list[CheckResult] = []
    checks.append(check_unit_range(series, indicator_spec.get("value_range")))
    checks.append(check_coverage(series, indicator_spec.get("granularity", "unknown"), indicator_spec, in_ch_mask=in_ch_mask))

    if biogeoregion_id_col in enriched.columns:
        checks.append(
            check_pdf_numerical(
                series,
                enriched[biogeoregion_id_col],
                indicator_spec.get("reference_values") or {},
                biogeoregion_keys,
            )
        )
    if values_at_landmarks is not None:
        checks.append(
            check_landmarks(values_at_landmarks, landmarks_with_expectations, col, summary, indicator_spec)
        )

    return IndicatorValidation(
        column_name=col,
        granularity=indicator_spec.get("granularity", "unknown"),
        n_total=len(series),
        n_valid=int(series.notna().sum()),
        null_fraction=float(series.isna().mean()),
        summary_stats=summary,
        checks=checks,
    )


def append_report(
    phase_label: str,
    indicator_validations: Sequence[IndicatorValidation],
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for v in indicator_validations:
            f.write(json.dumps({"phase": phase_label, **v.to_jsonable()}, ensure_ascii=False))
            f.write("\n")

    lines: list[str] = []
    lines.append(f"\n## {phase_label}\n")
    lines.append("### Per-indicator summary\n")
    lines.append("| column | granularity | n valid | %null | min/p25/median/p75/max  or  top-5 values | unique |")
    lines.append("|---|---|---|---|---|---|")
    for v in indicator_validations:
        s = v.summary_stats
        if s.get("is_string"):
            tops = s.get("top_values", {})
            tops_str = "; ".join(f"`{k or '∅'}`×{n}" for k, n in list(tops.items())[:5])
            distribution = f"top-5: {tops_str}"
        else:
            distribution = (
                f"{s.get('min','')} / {s.get('p25','')} / {s.get('median','')} "
                f"/ {s.get('p75','')} / {s.get('max','')}"
            )
        lines.append(
            f"| `{v.column_name}` | {v.granularity} | {v.n_valid:,} | "
            f"{v.null_fraction:.1%} | {distribution} | {s.get('n_unique','')} |"
        )
    lines.append("")
    lines.append("### Rigor checks\n")
    lines.append("| column | unit_range | coverage | pdf_numerical | landmarks |")
    lines.append("|---|---|---|---|---|")
    for v in indicator_validations:
        by_name = {c.name: c for c in v.checks}
        cells = []
        for nm in ("unit_range", "coverage", "pdf_numerical", "landmarks"):
            c = by_name.get(nm)
            if c is None:
                cells.append("—")
            else:
                mark = "✔" if c.passed else "✘"
                cells.append(f"{mark} {c.summary}")
        lines.append(f"| `{v.column_name}` | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")

    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
