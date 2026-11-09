"""Output contract for ``lam_x`` and its uncertainty (prop-1104 consumer).

The campaign requires the correlation length to stop being a literature anchor
and become an *auditable estimator output*: a value, an uncertainty, the
sampling adequacy, the estimator provenance and an explicit evidence grade.
This module pins that payload as a versioned schema (``lambda_x_estimate.v1``,
mirrored in ``schemas/lambda_x_estimate.v1.json``) and provides the flat view
that a downstream consumer (prop-1104) can ingest without re-deriving anything.

Payload design rules
--------------------
* One physical quantity per record: ``value_m`` (metres) plus the dimensionless
  ``value_over_thickness`` because the literature band (1.5--3) is quoted as
  ``lam_x/h``; both are always emitted so the consumer never has to divide.
* Uncertainty is a 95 % interval, not a standard deviation alone, and carries
  the *method* it came from (block-bootstrap for the DEM band, Monte Carlo for
  the synthetic calibration).
* ``quality`` makes the record self-policing: ``sufficient`` is only true when
  the record passes the calibrated threshold ``L/lam >= K*`` **and** the
  estimator regime is ``ok``; ``regime`` reports white-noise / periodic
  records that must be rejected instead of being silently accepted.
* ``evidence_grade`` distinguishes ``REAL`` (native DEM estimate, calibrated
  estimator), ``CALIBRATED`` (estimator bias model applied, native simulation)
  and ``WITHHELD`` (record not sufficient -- a value is still emitted for
  traceability, with ``value`` marked non-design).

No number enters this payload by hand: every field is produced by
:func:`build_estimate` from the study JSON.
"""
from __future__ import annotations

import math

__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_KEYS",
    "LITERATURE_BAND",
    "consensus",
    "build_estimate",
    "to_prop_1104",
    "validate",
]

SCHEMA_VERSION = "lambda_x_estimate.v1"
LITERATURE_BAND = (1.5, 3.0)

REQUIRED_KEYS = (
    "schema_version", "quantity", "unit", "value", "ci95", "estimator",
    "sampling", "quality", "evidence_grade", "sources", "warnings",
)


def consensus(values: dict) -> dict:
    """Median and relative spread of the per-method estimates."""
    finite = {k: float(v) for k, v in values.items() if v is not None and math.isfinite(float(v))}
    if not finite:
        return {"value_m": float("nan"), "method_spread": float("nan"),
                "methods": sorted(values), "agreement": "none"}
    vals = sorted(finite.values())
    med = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
    spread = (max(vals) - min(vals)) / med if med > 0 else float("inf")
    return {
        "value_m": float(med),
        "method_spread": float(spread),
        "methods": sorted(finite),
        "agreement": "consensus" if spread <= 0.30 else "disagreement",
    }


def build_estimate(
    *,
    value_m: float,
    thickness_m: float,
    ci95_m: tuple[float, float] | None,
    estimator: dict,
    record_length_m: float,
    dx_m: float,
    n_samples: int,
    threshold_K: float,
    bias_corrected: bool,
    regime: str = "ok",
    evidence_grade: str = "CALIBRATED",
    sources: list | None = None,
    warnings: list | None = None,
) -> dict:
    """Assemble a schema-valid ``lam_x`` record from computed ingredients."""
    warn = list(warnings or [])
    L_over_lam = record_length_m / value_m if value_m and value_m > 0 else float("inf")
    sufficient = bool(regime == "ok" and math.isfinite(L_over_lam)
                      and L_over_lam >= float(threshold_K))
    if not sufficient:
        msg = (f"record L/lam = {L_over_lam:.2f} below the sufficiency threshold "
               f"K* = {threshold_K:.2f} (or regime = {regime})")
        if msg not in warn:
            warn.append(msg)
    over_h = value_m / thickness_m if thickness_m else float("nan")
    ci_over_h = (
        [ci95_m[0] / thickness_m, ci95_m[1] / thickness_m]
        if ci95_m is not None and thickness_m else None
    )
    band = None
    if math.isfinite(over_h):
        band = {
            "literature_band_over_thickness": list(LITERATURE_BAND),
            "inside_band": bool(LITERATURE_BAND[0] <= over_h <= LITERATURE_BAND[1]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "quantity": "lambda_x",
        "unit": "m",
        "value": float(value_m),
        "value_over_thickness": float(over_h),
        "thickness_m": float(thickness_m),
        "ci95": list(ci95_m) if ci95_m is not None else None,
        "ci95_over_thickness": ci_over_h,
        "estimator": dict(estimator),
        "sampling": {
            "record_length_m": float(record_length_m),
            "dx_m": float(dx_m),
            "n_samples": int(n_samples),
            "L_over_lambda": float(L_over_lam),
        },
        "quality": {
            "sufficient": sufficient,
            "threshold_K": float(threshold_K),
            "bias_corrected": bool(bias_corrected),
            "regime": regime,
            "method_agreement": estimator.get("method_agreement", "n/a"),
            "design_usable": bool(sufficient and evidence_grade in ("REAL", "CALIBRATED")),
        },
        "literature_comparison": band,
        "evidence_grade": evidence_grade,
        "sources": list(sources or []),
        "warnings": warn,
    }


def to_prop_1104(record: dict) -> dict:
    """Flat consumer view of a ``lambda_x_estimate.v1`` record.

    The consumer interface is pinned by key names only (no units conversion, no
    nesting) so that a schema change upstream cannot silently change the meaning
    of a downstream input.  ``lambda_x_h`` is the literature-normalised value the
    1.5--3 band is quoted in.
    """
    errors = validate(record)
    if errors:
        raise ValueError("invalid lambda_x record: " + "; ".join(errors))
    ci = record.get("ci95") or [float("nan"), float("nan")]
    return {
        "lambda_x_m": record["value"],
        "lambda_x_h": record["value_over_thickness"],
        "lambda_x_ci95_m": [float(ci[0]), float(ci[1])],
        "lambda_x_ci95_h": record.get("ci95_over_thickness"),
        "lambda_x_estimator": record["estimator"].get("primary", "unknown"),
        "lambda_x_L_over_lambda": record["sampling"]["L_over_lambda"],
        "lambda_x_sufficient": record["quality"]["sufficient"],
        "lambda_x_threshold_K": record["quality"]["threshold_K"],
        "lambda_x_evidence": record["evidence_grade"],
        "lambda_x_inside_literature_band": (
            record.get("literature_comparison") or {}
        ).get("inside_band"),
        "lambda_x_schema": record["schema_version"],
    }


def validate(record: dict) -> list[str]:
    """Structural validation of a ``lam_x`` record (no external dependency)."""
    errs: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in record:
            errs.append(f"missing key {key!r}")
    if errs:
        return errs
    if record["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {record['schema_version']!r} != {SCHEMA_VERSION!r}")
    if record["quantity"] != "lambda_x":
        errs.append("quantity must be 'lambda_x'")
    if record["unit"] != "m":
        errs.append("unit must be 'm'")
    v = record["value"]
    if not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0.0:
        errs.append("value must be a finite non-negative number")
    ci = record["ci95"]
    if ci is not None:
        if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
            errs.append("ci95 must be a 2-element interval or null")
        elif not (float(ci[0]) <= float(v) <= float(ci[1])):
            errs.append("ci95 must bracket the value")
    q = record["quality"]
    for key in ("sufficient", "threshold_K", "bias_corrected", "regime"):
        if key not in q:
            errs.append(f"quality missing {key!r}")
    if q.get("sufficient") and q.get("regime") != "ok":
        errs.append("sufficient=True requires regime='ok'")
    if record["evidence_grade"] not in ("REAL", "CALIBRATED", "WITHHELD", "MOCK", "STUB"):
        errs.append(f"unknown evidence_grade {record['evidence_grade']!r}")
    est = record["estimator"]
    if "primary" not in est:
        errs.append("estimator.primary is required")
    return errs