"""Operator adjudication for the biaxial shear-band case (CASE-BIAXIAL-SHEAR-BAND).

Why this module exists
----------------------
The delivered rounds report the band inclination with **one** operator as the
primary (Radon projection focus, ``tensordem.biaxial_shear_band``) and two more
as "secondary" numbers: a structure tensor (``tensordem.biaxial_xval``,
Desrues & Viggiani's experimental route) and a one-step incremental Radon
variant.  On the R7 late-window fields those numbers are

    primary (cumulative Radon)   55.00 deg
    peak-relative Radon          56.25 deg
    single-increment Radon       53.91 deg
    structure tensor             67.75 deg

-- a 12.75 deg spread (23 % of the measured value) on the *same* displacement
fields, which is the same order as the 15 % acceptance tolerance the anchor is
judged with.  The verdict ``|mean - 60| / 60 = 8.33 % < 15 %`` therefore rests on
which operator is believed, and that question was never settled: the existing
validation of the secondary operators
(:func:`tensordem.biaxial_xval.cross_validate_synthetic`,
:func:`tensordem.biaxial_band_diagnostics.estimator_bias_sweep`) plants a known
band into a **clean** lattice, i.e. it measures operator behaviour on a field
that contains nothing but the band.  A real late-window field is not clean: it
carries the diffuse background of a bonded specimen, boundary bands nucleated
at the rough walls, and a finite number of grains.

What this module adds
---------------------
``planted_field_audit``  a **known-angle band is superposed on the real measured
    shear-strain field** (contamination ``c``: ``c = 1`` pure planted ridge,
    ``c = 0`` the real field, anything between keeps the real background at
    weight ``1 - c``), and all three operators are run on the resulting field.
    The recovery error is the operator's error *on that specimen's* field
    structure, so it can be used to bias-correct the operator's reading of the
    real field instead of trusting one of them a priori.

``operator_angles``  the three operators on **one shared** ``(particles,
    shear-strain)`` pair, so the only difference between them is the orientation
    operator (no window, no pair, no field difference).

``bias_corrected_angle``  subtract the planted-band bias (interpolated at the
    measured angle) from a measured angle.

``law_test``  the three classical inclination laws evaluated with quantities
    measured *in the same run* -- Mohr-Coulomb ``45 + phi/2``, Roscoe (1970)
    ``45 + psi_band/2`` and Arthur et al. (1977) ``45 + (phi + psi_band)/4`` --
    plus the relative error of the measured angle against each.  Roscoe and
    Arthur take the **in-band** dilation angle (the band's own kinematics),
    which is the quantity the earlier rounds withheld.

Everything is SI, ``numpy``/``scipy`` only; nothing here steps the DEM.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .biaxial_band_diagnostics import band_frame
from .biaxial_shear_band import (BandMeasurementError, fold_angle_deg,
                                 radon_band_angle, theory_angles_deg)
from .biaxial_xval import second_moment_angle, structure_tensor_angle

__all__ = [
    "OPERATORS",
    "ANCHOR_DEG",
    "TOLERANCE",
    "band_ridge_weight",
    "lattice_spacing",
    "mixed_field",
    "operator_angles",
    "planted_field_audit",
    "signed_angle_diff_deg",
    "bias_corrected_angle",
    "law_test",
]

#: orientation operators compared here (all consume the same scalar field)
OPERATORS = ("radon", "structure_tensor", "second_moment")

#: acceptance anchor of the case (midpoint of the 55-60 deg literature band)
ANCHOR_DEG = 60.0
#: acceptance tolerance of the case
TOLERANCE = 0.15


# ---------------------------------------------------------------------------
# field construction
# ---------------------------------------------------------------------------
def signed_angle_diff_deg(measured_deg: float, true_deg: float) -> float:
    """Signed difference of two band inclinations on the folded 180 deg circle.

    A band angle lives on ``[0, 180)`` with ``alpha`` and ``180 - alpha`` the
    same inclination, so the difference must be taken modulo 180 and folded
    back into ``(-90, 90]``; a plain subtraction would report a 100 deg error
    for a 10 deg mistake across the fold.
    """
    d = (float(measured_deg) - float(true_deg)) % 180.0
    return d - 180.0 if d > 90.0 else d


def _normalise(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    scale = float(np.std(v))
    if not scale > 0.0:
        raise BandMeasurementError("field has zero spread; cannot be mixed")
    return (v - float(np.mean(v))) / scale


def lattice_spacing(pos: np.ndarray) -> float:
    """Mean inter-particle spacing of a 2-D point set (area / count).

    Used to set a *physical* band width for the planted ridge so that the
    injected band has the same width as the real localisation instead of an
    arbitrary grid-dependent one.
    """
    points = np.asarray(pos, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 10:
        raise BandMeasurementError("pos must be a (N>=10, 2) array")
    width = float(np.ptp(points[:, 0]))
    height = float(np.ptp(points[:, 1]))
    if not (width > 0.0 and height > 0.0):
        raise BandMeasurementError("degenerate point set")
    return math.sqrt(width * height / float(points.shape[0]))


def band_ridge_weight(pos: np.ndarray, *, angle_deg: float,
                      half_width: float,
                      centre: Sequence[float] | None = None) -> np.ndarray:
    """Analytic ridge weight of a planar band at known ``angle_deg``.

    ``w = exp(-0.5 (s / half_width)^2)`` with ``s`` the distance from the band
    axis, ``s = (x - centre) . n`` and ``n`` the band **normal**
    (:func:`tensordem.biaxial_band_diagnostics.band_frame`).  ``angle_deg`` is
    measured from the ``sigma_3`` (horizontal) axis, i.e. the same convention
    as the production estimators.  The ridge is normalised to unit peak so the
    caller controls the amplitude through ``contamination``.
    """
    points = np.asarray(pos, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise BandMeasurementError("pos must be (N, 2)")
    if not float(half_width) > 0.0:
        raise BandMeasurementError("half_width must be positive")
    origin = (np.asarray(centre, dtype=np.float64) if centre is not None
              else points.mean(axis=0))
    _tangent, normal = band_frame(float(angle_deg))
    s = (points - origin) @ normal
    return np.exp(-0.5 * (s / float(half_width)) ** 2)


def mixed_field(gamma_real: Sequence[float], ridge: Sequence[float], *,
                contamination: float) -> np.ndarray:
    """Superpose a known-angle ridge on the real field at weight ``contamination``.

    ``contamination = 1`` returns the planted ridge alone (unit spread),
    ``contamination = 0`` the real field (unit spread), in between the
    variance-normalised mix -- so the two components contribute comparable
    amplitude and the contamination parameter has a meaning independent of the
    arbitrary units of a shear-strain field.
    """
    c = float(contamination)
    if not 0.0 <= c <= 1.0:
        raise BandMeasurementError("contamination must be in [0, 1]")
    real = _normalise(gamma_real)
    planted = _normalise(ridge)
    if real.shape != planted.shape:
        raise BandMeasurementError("gamma_real and ridge must have the same length")
    return (1.0 - c) * real + c * planted


# ---------------------------------------------------------------------------
# the three operators on one shared field
# ---------------------------------------------------------------------------
def operator_angles(pos: np.ndarray, gamma: Sequence[float], *,
                    cells: int = 40) -> dict[str, Any]:
    """Run every orientation operator on one shared ``(pos, gamma)`` pair.

    The three operators are structurally different: a projection/Radon focus on
    the point cloud, a structure tensor of a rasterised field (gradient
    eigendecomposition, Desrues & Viggiani's experimental route) and a weighted
    second moment of the high-strain point set (``regionprops`` route).  An
    operator that raises is reported with ``"error"`` instead of aborting the
    comparison, so a failure is visible rather than silently dropped.
    """
    points = np.asarray(pos, dtype=np.float64)
    values = np.asarray(gamma, dtype=np.float64).ravel()
    if points.ndim != 2 or points.shape[1] != 2:
        raise BandMeasurementError("pos must be (N, 2)")
    if values.shape[0] != points.shape[0]:
        raise BandMeasurementError("gamma and pos must have the same length")
    out: dict[str, Any] = {"particles": int(points.shape[0]),
                           "gamma_mean": float(values.mean()),
                           "localisation_index": float(values.std() / values.mean())
                           if values.mean() > 0 else math.nan}
    try:
        rad = radon_band_angle(points[:, 0], points[:, 1], values)
        out["radon"] = {"angle_deg": float(rad["folded_angle_deg"]),
                        "eta2": float(rad["eta2"]),
                        "z_score": float(rad["z_score"])}
    except Exception as exc:                                  # pragma: no cover
        out["radon"] = {"error": repr(exc)}
    try:
        tensor = structure_tensor_angle(points[:, 0], points[:, 1], values,
                                        cells=int(cells))
        out["structure_tensor"] = {"angle_deg": float(tensor["folded_angle_deg"]),
                                   "spread_deg": float(tensor["spread_deg"]),
                                   "cells_used": int(tensor["cells_used"])}
    except Exception as exc:                                  # pragma: no cover
        out["structure_tensor"] = {"error": repr(exc)}
    try:
        moment = second_moment_angle(points[:, 0], points[:, 1], values)
        out["second_moment"] = {"angle_deg": float(moment["folded_angle_deg"]),
                                "anisotropy": float(moment["anisotropy"]),
                                "particles_used": int(moment["particles_used"])}
    except Exception as exc:                                  # pragma: no cover
        out["second_moment"] = {"error": repr(exc)}
    return out


# ---------------------------------------------------------------------------
# planted-band audit on the real field
# ---------------------------------------------------------------------------
def planted_field_audit(pos: np.ndarray, gamma_real: Sequence[float], *,
                        angles_deg: Sequence[float] = tuple(range(30, 81, 5)),
                        contaminations: Sequence[float] = (1.0, 0.5, 0.25),
                        half_width: float | None = None,
                        cells: int = 40) -> dict[str, Any]:
    """Recovery error of every operator for known injected band angles.

    For each ``(contamination, angle)`` the field
    ``mixed_field(gamma_real, band_ridge_weight(angle))`` is measured with all
    three operators; the *true* angle is known exactly, so
    ``error = measured - true`` is a pure operator error **on this specimen's
    background structure** rather than on a clean lattice.

    Returns ``{"grid": [...], "bias": {operator: {c: {angle: error}}},
    "summary": {operator: {c: {"bias_mean", "bias_abs_max", "rmse"}}}}``.
    """
    points = np.asarray(pos, dtype=np.float64)
    half = (float(half_width) if half_width is not None
            else 2.5 * lattice_spacing(points))
    rows: list[dict[str, Any]] = []
    for c in contaminations:
        for angle in angles_deg:
            ridge = band_ridge_weight(points, angle_deg=float(angle),
                                      half_width=half)
            field = mixed_field(gamma_real, ridge, contamination=float(c))
            measured = operator_angles(points, field, cells=cells)
            row: dict[str, Any] = {"contamination": float(c),
                                   "angle_true_deg": float(angle)}
            for name in OPERATORS:
                entry = measured.get(name) or {}
                value = entry.get("angle_deg")
                row[name] = (float(value) if value is not None and
                             np.isfinite(value) else None)
                row[name + "_error"] = (
                    float(signed_angle_diff_deg(float(value), float(angle)))
                    if value is not None and np.isfinite(value) else None)
            rows.append(row)
    summary: dict[str, Any] = {}
    for name in OPERATORS:
        summary[name] = {}
        for c in contaminations:
            errs = [r[name + "_error"] for r in rows
                    if r["contamination"] == float(c) and r[name + "_error"] is not None]
            if errs:
                arr = np.asarray(errs, dtype=np.float64)
                summary[name][f"{float(c):.2f}"] = {
                    "n": int(arr.size),
                    "bias_mean_deg": float(arr.mean()),
                    "bias_median_deg": float(np.median(arr)),
                    "bias_abs_max_deg": float(np.max(np.abs(arr))),
                    "rmse_deg": float(np.sqrt(np.mean(arr ** 2))),
                }
            else:                                             # pragma: no cover
                summary[name][f"{float(c):.2f}"] = {"n": 0}
    return {"half_width": float(half),
            "lattice_spacing": float(lattice_spacing(points)),
            "contaminations": [float(c) for c in contaminations],
            "angles_deg": [float(a) for a in angles_deg],
            "cells": int(cells),
            "rows": rows,
            "summary": summary}


def bias_corrected_angle(measured_deg: float, audit: dict[str, Any], *,
                         operator: str, contamination: float = 1.0) -> dict[str, Any]:
    """Subtract the planted-band bias of ``operator`` from a measured angle.

    The bias curve ``measured - true`` from :func:`planted_field_audit` is
    interpolated at the *measured* angle (the correction is evaluated where the
    measurement actually landed) and subtracted.  Angles are handled on the
    folded ``[0, 90]`` interval.
    """
    if operator not in OPERATORS:
        raise BandMeasurementError(f"unknown operator {operator!r}")
    key = f"{float(contamination):.2f}"
    table = (audit.get("summary") or {}).get(operator, {}).get(key)
    if not table or not table.get("n"):
        raise BandMeasurementError(
            f"no bias sample for operator {operator!r} at contamination {key}")
    xs, ys = [], []
    for row in audit["rows"]:
        if row["contamination"] != float(contamination):
            continue
        err = row.get(operator + "_error")
        if err is None:
            continue
        xs.append(float(row["angle_true_deg"]))
        ys.append(float(err))
    if len(xs) < 2:
        raise BandMeasurementError("bias curve needs at least two points")
    order = np.argsort(xs)
    bias = float(np.interp(float(measured_deg), np.asarray(xs)[order],
                           np.asarray(ys)[order]))
    corrected = float(fold_angle_deg(float(measured_deg) - bias))
    return {"operator": operator,
            "contamination": float(contamination),
            "measured_deg": float(measured_deg),
            "bias_deg": bias,
            "corrected_deg": corrected,
            "relative_change": (abs(corrected - measured_deg) / measured_deg
                                if measured_deg else math.nan)}


# ---------------------------------------------------------------------------
# classical inclination laws with in-run parameters
# ---------------------------------------------------------------------------
def law_test(theta_measured_deg: float, *, phi_deg: float,
             psi_band_deg: float) -> dict[str, Any]:
    """Compare a measured band angle with the three classical laws.

    ``theta_MC = 45 + phi/2`` (Mohr-Coulomb plasticity), ``theta_Roscoe =
    45 + psi/2`` (Roscoe 1970, Geotechnique 20:129-146) and ``theta_Arthur =
    45 + (phi + psi)/4`` (Arthur et al. 1977, Geotechnique 27:53-66).  Both
    ``phi`` (secant friction angle at the deviator peak) and ``psi`` (the
    **in-band** dilation angle) come from the same run, so no parameter is
    borrowed from the law being tested.
    """
    theta = float(theta_measured_deg)
    phi = float(phi_deg)
    psi = float(psi_band_deg)
    if not math.isfinite(theta):
        raise BandMeasurementError("theta_measured_deg must be finite")
    predictions = theory_angles_deg(phi, psi)
    out: dict[str, Any] = {"theta_measured_deg": theta,
                           "phi_deg": phi,
                           "psi_band_deg": psi,
                           "predictions_deg": predictions,
                           "relative_error": {}, "within_tolerance": {}}
    for name, predicted in predictions.items():
        err = abs(theta - predicted) / abs(predicted) if predicted else math.inf
        out["relative_error"][name] = float(err)
        out["within_tolerance"][name] = bool(err <= TOLERANCE)
    out["laws_within_tolerance"] = [k for k, v in out["within_tolerance"].items() if v]
    out["anchor_relative_error"] = float(abs(theta - ANCHOR_DEG) / ANCHOR_DEG)
    out["anchor_within_tolerance"] = bool(out["anchor_relative_error"] <= TOLERANCE)
    return out