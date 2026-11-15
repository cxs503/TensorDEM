"""Sampling-stride convergence of the granular-column arrival definitions.

Why this module exists
----------------------
``CASE-COLUMN-COLLAPSE`` compares a *measured* emplacement time ``t`` with the
Lube et al. (2004) prediction ``t_inf = 3 sqrt(h_i/g)``.  ``t`` is not a
primitive of the simulation: it is an **estimator**, and there are two
independent choices in it:

1. *the arrival definition* -- implemented in
   :mod:`tensordem.granular_column.arrival` (``fraction`` / ``velocity`` /
   ``length`` families, 13 pre-registered members);
2. *the sampling stride of the stored front history* -- every archived case of
   this card stored the history every 5 or 20 integration steps, so the
   sampling interval (2-8 ms at ``dt ~ 4e-4 s``) is the same order as the
   arrival module's own smoothing window (10 ms).

Choice (2) has never been audited, yet it can masquerade as physics: a coarse
stride delays the first sample that meets a threshold, inflating ``t`` by up to
one sampling interval.  This module makes that coupling **measurable and
testable**:

* :func:`decimate` -- subsample a dense history exactly the way the run would
  have stored it (deterministic, last sample always kept);
* :func:`stride_scan` -- evaluate every pre-registered definition at a ladder
  of strides from **one** trajectory (no new physics, no re-integration);
* :func:`sampling_bound_violations` -- check the *analytic invariant* of the
  ``fraction`` / ``length`` estimators on a monotone front,
  ``t_dense <= t_k <= t_dense + k * sample_dt``; a violation is an
  implementation defect, not a physical effect;
* :func:`definition_band` / :func:`adjudicate` -- separate the
  *definition-induced* spread of ``t`` from the residual against the reference,
  so a verdict that flips across definitions can be reported as such instead of
  being quoted as a single number.

Everything here is pure ``numpy`` and closed-form testable: the invariant above
is a theorem for a non-decreasing front, so the unit tests assert it on
analytically known histories (and assert that it *fails* for a receding front,
which is why the underlying :mod:`...arrival` module rejects those).
"""

from __future__ import annotations

import math

import numpy as np

from .arrival import (FRACTION_LEVELS, LENGTH_TOLERANCES_D, ArrivalError,
                      VELOCITY_FRACTIONS, PRIMARY_FRACTION,
                      PRIMARY_LENGTH_TOLERANCE_D, PRIMARY_VELOCITY_FRACTION,
                      arrival_report, time_to_fraction, time_to_length_tolerance,
                      time_to_velocity_threshold)

__all__ = [
    "ArrivalSamplingError",
    "DEFAULT_STRIDES",
    "SAMPLING_TOL",
    "DEFINITION_BAND_TOL",
    "PRIMARY_KEY",
    "decimate",
    "definition_keys",
    "flatten_arrivals",
    "arrival_grid",
    "stride_scan",
    "sampling_bound_violations",
    "definition_band",
    "adjudicate",
    "emplacement_errors",
]

#: Stride ladder used by the pre-registered scan (``docs/PREREG_..._m21945.md``).
DEFAULT_STRIDES = (1, 2, 5, 10, 20, 40)
#: A stride change must move ``t`` by less than this fraction of ``t_ref``.
SAMPLING_TOL = 0.03
#: A definition family spread wider than the card tolerance (15 %) dominates it.
DEFINITION_BAND_TOL = 0.15
#: The card's historical primary definition (rounds 1-6, m17917 onward).
PRIMARY_KEY = f"fraction_{PRIMARY_FRACTION:g}"


class ArrivalSamplingError(ValueError):
    """Raised for any input that would make the scan silently meaningless."""


def _as_pair(time, front) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=np.float64).ravel()
    f = np.asarray(front, dtype=np.float64).ravel()
    if t.size != f.size:
        raise ArrivalSamplingError("time and front must have equal length")
    if t.size < 2:
        raise ArrivalSamplingError("history needs at least 2 samples to decimate")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)):
        raise ArrivalSamplingError("history must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ArrivalSamplingError("time must be strictly increasing")
    return t, f


def _as_stride(stride) -> int:
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
        raise ArrivalSamplingError("stride must be an integer number of steps")
    stride = int(stride)
    if stride < 1:
        raise ArrivalSamplingError("stride must be >= 1")
    return stride


def _as_strides(strides) -> tuple[int, ...]:
    out = tuple(_as_stride(s) for s in strides)
    if not out:
        raise ArrivalSamplingError("stride ladder must not be empty")
    if len(set(out)) != len(out):
        raise ArrivalSamplingError("stride ladder must not repeat a stride")
    return out


def decimate(time, front, stride):
    """Subsample a dense history at ``stride``, always keeping the last sample.

    The last sample is kept because the final front defines ``r_inf``: dropping
    it would make the runout itself depend on the stride, which is a different
    (and much larger) effect than the arrival-time quantisation studied here.
    """
    t, f = _as_pair(time, front)
    k = _as_stride(stride)
    if k == 1:
        return t.copy(), f.copy()
    idx = np.arange(0, t.size, k, dtype=int)
    if idx[-1] != t.size - 1:
        idx = np.append(idx, t.size - 1)
    return t[idx], f[idx]


def definition_keys() -> tuple[str, ...]:
    """Every pre-registered arrival definition, in a stable order."""
    return tuple(
        [f"fraction_{f:g}" for f in FRACTION_LEVELS]
        + [f"velocity_{f:g}" for f in VELOCITY_FRACTIONS]
        + [f"length_{n:g}d" for n in LENGTH_TOLERANCES_D]
    )


def flatten_arrivals(report: dict) -> dict:
    """Flatten an :func:`...arrival.arrival_report` into ``{key: time}``."""
    out = {}
    for f, value in report["fraction"].items():
        out[f"fraction_{f}"] = float(value)
    for f, value in report["velocity"].items():
        out[f"velocity_{f}"] = float(value)
    for n, value in report["length"].items():
        out[f"length_{n}d"] = float(value)
    missing = [k for k in definition_keys() if k not in out]
    if missing:
        raise ArrivalSamplingError(f"arrival report missing definitions: {missing}")
    return out


def _select(key: str) -> tuple[str, float]:
    family, _, level = key.partition("_")
    if family == "length":
        return family, float(level[:-1])
    return family, float(level)


def arrival_grid(time, front, *, radius: float, grain_diameter: float,
                 stride: int = 1, **report_kwargs) -> dict:
    """All 13 arrival definitions of one history, subsampled at ``stride``."""
    t, f = decimate(time, front, stride)
    report = arrival_report(t, f, radius=radius, grain_diameter=grain_diameter,
                            **report_kwargs)
    grid = flatten_arrivals(report)
    grid["__sample_dt"] = float(np.median(np.diff(t)))
    grid["__n_samples"] = int(t.size)
    grid["__stride"] = _as_stride(stride)
    return grid


def stride_scan(time, front, *, radius: float, grain_diameter: float,
                strides=DEFAULT_STRIDES, strict: bool = False,
                **report_kwargs) -> dict:
    """Evaluate every definition along a stride ladder from one trajectory.

    ``strict=True`` re-raises :class:`...arrival.ArrivalError`; the default
    (``strict=False``) records the failure per stride as
    ``nan`` + ``failed_strides`` so one degenerate stride cannot destroy a whole
    scan.  Failures are never interpolated over.
    """
    t, f = _as_pair(time, front)
    ladder = _as_strides(strides)
    dense_dt = float(np.median(np.diff(t)))
    keys = definition_keys()
    table: dict[str, dict[str, float]] = {key: {} for key in keys}
    failed: dict[str, str] = {}
    sample_dt: dict[int, float] = {}
    n_samples: dict[int, int] = {}
    for stride in ladder:
        try:
            grid = arrival_grid(t, f, radius=radius,
                                grain_diameter=grain_diameter, stride=stride,
                                **report_kwargs)
        except ArrivalError as exc:
            if strict:
                raise
            failed[str(stride)] = f"{type(exc).__name__}: {exc}"
            sample_dt[stride] = stride * dense_dt
            n_samples[stride] = int(np.arange(0, t.size, stride).size)
            continue
        sample_dt[stride] = grid["__sample_dt"]
        n_samples[stride] = grid["__n_samples"]
        for key in keys:
            table[key][stride] = float(grid[key])
    base_stride = ladder[0]
    definitions: dict[str, dict] = {}
    for key in keys:
        values = table[key]
        good = [(s, values[s]) for s in ladder
                if s in values and math.isfinite(values[s])]
        reference = values.get(base_stride, math.nan)
        biases = {str(s): (v - reference) / reference
                  if math.isfinite(reference) and reference > 0.0 else math.nan
                  for s, v in good}
        times = [v for _, v in good]
        spread = (max(times) - min(times)) if len(times) >= 2 else 0.0
        definitions[key] = {
            "times": {str(s): (values.get(s, math.nan)) for s in ladder},
            "reference_time": reference,
            "reference_stride": base_stride,
            "n_valid": len(good),
            "bias_rel": biases,
            "max_abs_bias_rel": max((abs(b) for b in biases.values()
                                     if math.isfinite(b)), default=math.nan),
            "spread_s": spread,
            "spread_rel": spread / reference
            if math.isfinite(reference) and reference > 0.0 else math.nan,
            "budget_s": {str(s): s * dense_dt for s in ladder},
        }
    return {
        "strides": list(ladder),
        "dense_sample_dt": dense_dt,
        "sample_dt": {str(s): sample_dt[s] for s in ladder},
        "n_samples": {str(s): n_samples[s] for s in ladder},
        "failed_strides": failed,
        "definitions": definitions,
        "radius_initial": float(radius),
        "grain_diameter": float(grain_diameter),
        "front_final_dense": float(f[-1]),
    }


def sampling_bound_violations(scan: dict, keys=None, tol_factor: float = 1.0) -> list[dict]:
    """Analytic invariant: ``t_1 <= t_k <= t_1 + k * sample_dt`` (monotone front).

    Only the two *threshold-attained* families (``fraction`` and ``length``) are
    checked: the ``velocity`` family is a smoothed-derivative estimate and is
    not subject to the one-sample quantisation bound.  A violation beyond
    ``tol_factor`` sampling intervals means the decimation is not a faithful
    resampling of the stored history.
    """
    if not math.isfinite(tol_factor) or tol_factor <= 0.0:
        raise ArrivalSamplingError("tol_factor must be finite and positive")
    keys = definition_keys() if keys is None else tuple(keys)
    unknown = [k for k in keys if k not in scan["definitions"]]
    if unknown:
        raise ArrivalSamplingError(f"unknown definitions: {unknown}")
    dense_dt = float(scan["dense_sample_dt"])
    out = []
    for key in keys:
        if not (key.startswith("fraction_") or key.startswith("length_")):
            continue
        entry = scan["definitions"][key]
        ref = entry["reference_time"]
        if not math.isfinite(ref):
            continue
        for stride_str, value in entry["times"].items():
            stride = int(stride_str)
            if not math.isfinite(value):
                continue
            upper = ref + tol_factor * stride * dense_dt
            if value < ref - 1e-12 or value > upper + 1e-12:
                out.append({"definition": key, "stride": stride,
                            "time": value, "dense_time": ref,
                            "budget_s": stride * dense_dt,
                            "excursion_s": value - ref})
    return out


def definition_band(grid: dict) -> dict:
    """Spread of the arrival time across the 13 definitions at one stride."""
    times = {k: float(v) for k, v in grid.items() if not k.startswith("__")}
    finite = {k: v for k, v in times.items() if math.isfinite(v)}
    if not finite:
        raise ArrivalSamplingError("no finite arrival time at this stride")
    lo_key = min(finite, key=finite.get)
    hi_key = max(finite, key=finite.get)
    lo, hi = finite[lo_key], finite[hi_key]
    spread = hi - lo
    return {
        "min_definition": lo_key, "max_definition": hi_key,
        "min_time": lo, "max_time": hi,
        "spread_s": spread,
        "spread_rel": spread / lo if lo > 0.0 else math.nan,
        "n_finite": len(finite),
        "n_nan": len(times) - len(finite),
        "definition_dominated": (spread / lo > DEFINITION_BAND_TOL
                                 if lo > 0.0 else math.nan),
    }


def emplacement_errors(grid: dict, reference_time: float) -> dict:
    """Relative error of every definition against the reference ``t_inf``."""
    if not math.isfinite(reference_time) or reference_time <= 0.0:
        raise ArrivalSamplingError("reference_time must be finite and positive")
    return {k: (float(v) - reference_time) / reference_time
            for k, v in grid.items() if not k.startswith("__")}


def adjudicate(errors: dict, tolerance: float = 0.15,
               primary: str = PRIMARY_KEY) -> dict:
    """Which definitions pass the card tolerance, and how many flips that implies.

    ``flip`` = a definition whose in-band verdict differs from the pre-registered
    primary one.  A single flip means the anchor's verdict is not a property of
    the model but of the (undeclared) definition, which is exactly what has to be
    reported instead of a bare error number.
    """
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ArrivalSamplingError("tolerance must be finite and positive")
    if primary not in errors:
        raise ArrivalSamplingError(f"primary definition {primary!r} not in errors")
    in_band = {k: (abs(v) <= tolerance if math.isfinite(v) else None)
               for k, v in errors.items()}
    if in_band[primary] is None:
        raise ArrivalSamplingError("primary definition error is not finite")
    flips = sorted(k for k, v in in_band.items() if v is not None
                   and v != in_band[primary])
    best = min((k for k, v in errors.items() if math.isfinite(v)),
               key=lambda k: abs(errors[k]))
    worst = max((k for k, v in errors.items() if math.isfinite(v)),
                key=lambda k: abs(errors[k]))
    return {
        "tolerance": tolerance, "primary": primary,
        "primary_error": errors[primary],
        "primary_in_band": in_band[primary],
        "in_band": in_band,
        "n_in_band": sum(1 for v in in_band.values() if v),
        "n_definitions": len(errors),
        "flips": flips,
        "n_flips": len(flips),
        "best": best, "best_error": errors[best],
        "worst": worst, "worst_error": errors[worst],
        "any_within_three_percent": any(abs(v) < 0.03
                                        for v in errors.values()
                                        if math.isfinite(v)),
    }