"""Extreme-load scaling law and load envelope for ridge keel interaction.

The DEM crushing model (:mod:`tensordem.myi_ridge.crushing`) delivers a
steady crushing pressure ``p`` on the hull plate for a rubble layer of depth
``h_keel``.  The extreme ice load on the ship is that pressure acting on the
contact patch of the channel the hull cuts:

    F_max = p * h_keel * L_c

``L_c`` is the loaded length of hull in contact with the ridge (a ship
parameter; the contact patch is ``h_keel x L_c``).  Because ``p`` is measured
by the DEM at each ``(sigma_ice, h_keel, V, n)`` point, the exponents of

    F_max ~ sigma_ice^a * h_keel^b

are *measured*, not assumed: ``a`` comes from the pressure law
``p ~ sigma_ice^a`` and ``b`` from the residual height dependence of the
pressure plus the contact-area factor.

The module also provides the monotonicity checks and the envelope assembly used
by the acceptance criteria.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "fit_power_law",
    "fit_exponents",
    "monotone_increasing",
    "monotonicity_report",
    "extreme_load",
    "load_envelope",
]


def _vector(points, keys) -> tuple[np.ndarray, ...]:
    columns = []
    for key in keys:
        values = []
        for point in points:
            if key not in point:
                raise KeyError(f"missing field {key!r}")
            value = float(point[key])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"field {key!r} must be positive and finite")
            values.append(value)
        columns.append(np.asarray(values, dtype=float))
    return tuple(columns)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares fit of ``y = c x^a`` in log space.

    Returns the exponent, prefactor, coefficient of determination and the
    standard error of the exponent.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional and of equal length")
    if x.size < 3:
        raise ValueError("at least three points are needed")
    if not np.all(x > 0.0) or not np.all(y > 0.0):
        raise ValueError("power-law fitting requires positive data")
    lx = np.log(x)
    ly = np.log(y)
    design = np.vstack([np.ones_like(lx), lx]).T
    coefficients, *_ = np.linalg.lstsq(design, ly, rcond=None)
    residual = ly - design @ coefficients
    dof = max(lx.size - 2, 1)
    variance = float((residual ** 2).sum() / dof)
    covariance = variance * np.linalg.inv(design.T @ design)
    total = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / total if total > 0.0 else 1.0
    return {
        "exponent": float(coefficients[1]),
        "exponent_stderr": float(math.sqrt(max(covariance[1, 1], 0.0))),
        "prefactor": float(math.exp(coefficients[0])),
        "r_squared": r2,
        "points": int(lx.size),
        "rms_log_residual": float(math.sqrt(variance)),
    }


def fit_exponents(points, *, value_key: str = "F_max",
                  sigma_key: str = "sigma_ice", depth_key: str = "h_keel") -> dict:
    """Fit ``F_max = C sigma_ice^a h_keel^b`` over a grid of DEM cases.

    The fit is a two-regressor least-squares problem in log space; the
    interaction is not resolved (the grid must therefore be balanced enough for
    the exponents to be identifiable).
    """
    sigma, depth = _vector(points, (sigma_key, depth_key))
    values = np.asarray([float(point[value_key]) for point in points], dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("F_max must be positive")
    design = np.vstack([np.ones_like(values), np.log(sigma), np.log(depth)]).T
    coefficients, *_ = np.linalg.lstsq(design, np.log(values), rcond=None)
    residual = np.log(values) - design @ coefficients
    dof = max(values.size - 3, 1)
    variance = float((residual ** 2).sum() / dof)
    covariance = variance * np.linalg.inv(design.T @ design)
    total = float(((np.log(values) - np.log(values).mean()) ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / total if total > 0.0 else 1.0
    return {
        "prefactor": float(math.exp(coefficients[0])),
        "exponent_sigma": float(coefficients[1]),
        "exponent_sigma_stderr": float(math.sqrt(max(covariance[1, 1], 0.0))),
        "exponent_depth": float(coefficients[2]),
        "exponent_depth_stderr": float(math.sqrt(max(covariance[2, 2], 0.0))),
        "r_squared": r2,
        "points": int(values.size),
        "rms_log_residual": float(math.sqrt(variance)),
    }


def monotone_increasing(values, *, tolerance: float = 0.0) -> bool:
    """True if the sequence never decreases by more than ``tolerance``."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("need a one-dimensional sequence of at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return bool(np.all(np.diff(array) >= -abs(float(tolerance))))


def monotonicity_report(sweeps: dict, *, tolerance: float = 0.0) -> dict:
    """Check monotone increase along each named one-factor-at-a-time sweep."""
    report = {}
    for name, values in sweeps.items():
        flags = monotone_increasing(values, tolerance=tolerance)
        array = np.asarray(values, dtype=float)
        report[name] = {
            "values": [float(v) for v in array],
            "monotone_increasing": flags,
            "min_step": float(np.diff(array).min()) if array.size > 1 else 0.0,
        }
    report["all_monotone"] = all(item["monotone_increasing"] for key, item in report.items()
                                if key != "all_monotone")
    report["tolerance"] = float(tolerance)
    return report


def extreme_load(pressure: float, keel_depth: float, loaded_length: float) -> float:
    """``F_max = p * h_keel * L_c`` -- the extreme ice load on the ship [N]."""
    for name, value in (("pressure", pressure), ("keel_depth", keel_depth),
                        ("loaded_length", loaded_length)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    return float(pressure) * float(keel_depth) * float(loaded_length)


def load_envelope(cases, *, loaded_length: float) -> dict:
    """Assemble the extreme-load envelope from DEM crushing cases.

    Each case must provide ``sigma_ice`` [Pa], ``h_keel`` [m], ``hull_velocity``
    [m/s], ``pressure_steady`` [Pa] and ``porosity``.  The returned dictionary
    contains the per-case loads, the fitted exponents, the monotonicity report
    of the one-factor sweeps and the envelope bounds.
    """
    if not cases:
        raise ValueError("at least one case is required")
    rows = []
    for case in cases:
        pressure = float(case["pressure_steady"])
        load = extreme_load(pressure, float(case["h_keel"]), loaded_length)
        rows.append({
            "sigma_ice": float(case["sigma_ice"]),
            "h_keel": float(case["h_keel"]),
            "hull_velocity": float(case["hull_velocity"]),
            "porosity": float(case.get("porosity", 0.30)),
            "pressure_steady": pressure,
            "F_max": load,
            "energy_closure": float(case.get("closure_of_exchanged", 0.0)),
            "seed": int(case.get("seed", 0)),
        })
    loads = np.asarray([row["F_max"] for row in rows], dtype=float)
    exponents = fit_exponents(rows) if len({row["sigma_ice"] for row in rows}) > 1 \
        and len({row["h_keel"] for row in rows}) > 1 else None

    def _sweep(field: str, fixed: dict) -> list[float]:
        selected = [row for row in rows
                    if all(abs(row[key] - value) < 1e-12 for key, value in fixed.items())]
        selected.sort(key=lambda row: row[field])
        return [row["F_max"] for row in selected]

    # one-factor-at-a-time sweeps must hold *every* other factor fixed, porosity
    # included, otherwise a case that differs in porosity leaks into a strength
    # or depth sweep and destroys the monotonicity test
    modal_porosity = max({row["porosity"] for row in rows},
                         key=lambda value: sum(1 for row in rows
                                               if abs(row["porosity"] - value) < 1e-12))
    base = {"sigma_ice": 7.5e6, "h_keel": 20.0, "hull_velocity": 2.0,
            "porosity": modal_porosity}
    sweeps = {}
    sigma_values = sorted({row["sigma_ice"] for row in rows})
    if len(sigma_values) > 1:
        sweeps["sigma_ice"] = _sweep("sigma_ice", {"h_keel": 20.0, "hull_velocity": 2.0,
                                                   "porosity": modal_porosity})
    depth_values = sorted({row["h_keel"] for row in rows})
    if len(depth_values) > 1:
        sweeps["h_keel"] = _sweep("h_keel", {"sigma_ice": 7.5e6, "hull_velocity": 2.0,
                                             "porosity": modal_porosity})
    speed_values = sorted({row["hull_velocity"] for row in rows})
    if len(speed_values) > 1:
        sweeps["hull_velocity"] = _sweep("hull_velocity", {"sigma_ice": 7.5e6,
                                                           "h_keel": 20.0,
                                                           "porosity": modal_porosity})
    monotonicity = monotonicity_report(sweeps) if sweeps else {"all_monotone": None}
    return {
        "loaded_length": float(loaded_length),
        "cases": rows,
        "case_count": len(rows),
        "load_min": float(loads.min()),
        "load_max": float(loads.max()),
        "pressure_min": float(min(row["pressure_steady"] for row in rows)),
        "pressure_max": float(max(row["pressure_steady"] for row in rows)),
        "exponents": exponents,
        "monotonicity": monotonicity,
        "sweep_fixed_point": base,
    }