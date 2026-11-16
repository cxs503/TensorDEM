"""State-matched test of the classical shear-band inclination laws.

Why this module exists
----------------------
``docs/BIAXIAL_SHEAR_BAND.md`` §① lists the three classical relations that
predict a shear band's inclination ``theta`` (deg from the ``sigma_3`` axis)
from the friction angle ``phi`` and the dilation angle ``psi``::

    Mohr-Coulomb (plasticity bound)   theta = 45 + phi / 2
    Roscoe (Geotechnique 20, 1970)    theta = 45 + psi / 2
    Arthur et al. (Geotechnique 27, 1977)
                                      theta = 45 + (phi + psi) / 4

``tensordem.biaxial_shear_band.theory_angles_deg`` already evaluates them, and
every specimen result carries ``theory`` / ``theory_relative_error``.  R2-R7
never used those numbers for the verdict, for two stated reasons
(``docs/ADVERSARIAL_REVIEW_BIAXIAL_SHEAR_BAND_R7.md`` §4.2):

1. the dilation angle of a bonded specimen looked unphysical, so the
   ``psi``-based laws (Roscoe, Arthur) were withheld;
2. the only published comparison was against a *borrowed* band (55-60 deg,
   Desrues & Viggiani 2004, midpoint 60 deg), whose provenance was never
   re-derived.

Both objections are addressed here without touching the DEM cell:

* **State matching.**  The band angle is measured over the *late* post-peak
  window, but the friction angle reported next to it (``peak.friction_angle_deg``)
  is the secant angle at the *deviator peak*.  Those are different material
  states.  This module rebuilds ``phi`` and ``psi`` from the recorded stress
  path **over exactly the late window used for the band angle**, so the two
  sides of the comparison describe one state.  ``phi`` comes from the wall
  stress trace, ``psi`` from the volumetric-vs-axial slope, both with the
  conventions of ``biaxial_shear_band`` (``sin(phi) = q / (2 p)``,
  ``psi = asin(-d eps_v / d eps_a)``).
* **Dilation without a bonded dilation reading.**  The stress-dilatancy
  (Rowe 1962; Bolton 1986) route ``sin psi = (sin phi - sin phi_cv) /
  (1 - sin phi sin phi_cv)`` supplies ``psi`` from the measured ``phi`` and a
  declared critical-state friction angle ``phi_cv``, so the ``psi``-based laws
  can be tested against a *physical* dilation angle instead of being withheld.
  Sensitivity to ``phi_cv`` is reported, never silently fixed.
* **Anchor provenance is arithmetic.**  ``anchor_implied_angles`` inverts the
  borrowed anchor: 60 deg is Mohr-Coulomb at ``phi = 30 deg``
  (``2 (60 - 45)``), which is not the specimen's friction angle.  The anchor is
  therefore a fixed material assumption of the source paper, not a prediction
  for this specimen.

Everything here is pure arithmetic on numbers the caller supplies; nothing is
fitted to the measured band angle.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .biaxial_shear_band import theory_angles_deg

__all__ = [
    "LAWS",
    "secant_friction_deg",
    "dilation_angle_deg",
    "stress_dilatancy_psi_deg",
    "late_window_state",
    "late_window_band_angle",
    "law_errors",
    "adjudicate",
    "anchor_implied_angles",
]

#: the laws evaluated by :func:`tensordem.biaxial_shear_band.theory_angles_deg`
LAWS = ("mohr_coulomb", "roscoe", "arthur")

#: laws whose prediction contains ``psi`` (withheld by R2-R7)
PSI_LAWS = ("roscoe", "arthur")


def _clip_sin(value: float) -> float:
    return float(max(min(float(value), 0.95), -0.95))


def _clip_sin_positive(value: float) -> float:
    """Clip to ``[0, 0.95]``: the friction-angle convention of the DEM cell."""
    return float(max(min(float(value), 0.95), 0.0))


def secant_friction_deg(q_over_p: float) -> float:
    """Secant friction angle from the stress ratio ``q / p`` (degrees).

    ``sin(phi) = (sigma_1 - sigma_3) / (sigma_1 + sigma_3) = q / (2 p)`` with
    ``p = (sigma_1 + sigma_3) / 2``, i.e. the same definition
    ``biaxial_shear_band`` uses at the peak, evaluated here on any state.
    """
    return math.degrees(math.asin(_clip_sin_positive(0.5 * float(q_over_p))))


def dilation_angle_deg(axial_strain: Sequence[float],
                       volumetric_strain: Sequence[float]) -> tuple[float, float]:
    """Dilation angle (deg) and ``d eps_v / d eps_a`` from a strain segment.

    Same convention as ``biaxial_shear_band``: ``eps_v`` is the *log*
    volumetric strain with compaction positive, so dilation gives a negative
    slope and ``psi = asin(-slope)``.  Fewer than five samples, or a strain
    segment with no axial span, raise ``ValueError`` instead of returning a
    meaningless number.
    """
    eps_a = np.asarray(axial_strain, dtype=float)
    eps_v = np.asarray(volumetric_strain, dtype=float)
    if eps_a.size < 5:
        raise ValueError(f"need >= 5 samples for a dilation fit, got {eps_a.size}")
    if float(np.ptp(eps_a)) <= 0.0:
        raise ValueError("axial strain segment has zero span; dilation undefined")
    slope = float(np.polyfit(eps_a, eps_v, 1)[0])
    return math.degrees(math.asin(_clip_sin(-slope))), slope


def stress_dilatancy_psi_deg(phi_deg: float, phi_cv_deg: float) -> float:
    """Dilation angle from stress-dilatancy (Rowe 1962 / Bolton 1986).

    ``sin psi = (sin phi - sin phi_cv) / (1 - sin phi sin phi_cv)``.

    Returns 0 for ``phi <= phi_cv`` (no dilation, the critical state), never a
    negative dilation angle: the relation is only used to supply the *dilative*
    branch that the ``psi``-based band laws assume.
    """
    s_phi = math.sin(math.radians(float(phi_deg)))
    s_cv = math.sin(math.radians(float(phi_cv_deg)))
    denom = 1.0 - s_phi * s_cv
    if abs(denom) < 1e-12:
        raise ValueError("degenerate stress-dilatancy denominator")
    s_psi = (s_phi - s_cv) / denom
    return math.degrees(math.asin(_clip_sin(max(s_psi, 0.0))))


def late_window_state(stress_path: Mapping[str, Iterable[float]], *,
                      late_window_fraction: float = 0.8,
                      peak_tolerance_rows: int = 0) -> dict:
    """``phi`` / ``psi`` over the late post-peak window of a recorded path.

    ``stress_path`` is ``result["stress_path"]`` of
    ``biaxial_shear_band.run_biaxial_shear_band``.  The peak is the (optionally
    smoothed) maximum of the deviator stress, the post-peak record starts
    there, and the late window is its last ``late_window_fraction`` — the same
    window the cumulative band estimator uses, so band angle and material state
    are read at one state point.

    Keys of the returned dict: ``phi_deg`` (mean secant angle in the window),
    ``phi_peak_deg`` (secant angle at the deviator peak, for reference),
    ``psi_deg`` (dilation fit in the window), ``slope``,
    ``sigma_ratio`` (mean ``q / p``), ``strain_span`` (log axial strain span of
    the window), ``rows``, ``peak_index``, ``late_start_index``.
    """
    if not 0.0 < float(late_window_fraction) <= 1.0:
        raise ValueError("late_window_fraction must lie in (0, 1]")
    q = np.asarray(stress_path["deviator"], dtype=float)
    eps_a = np.asarray(stress_path["axial_strain_log"], dtype=float)
    eps_v = np.asarray(stress_path["volumetric_strain_log"], dtype=float)
    ratio = np.asarray(stress_path["stress_ratio"], dtype=float)
    if q.size < 10:
        raise ValueError(f"stress path too short ({q.size} rows)")

    window = max(1, int(peak_tolerance_rows) or int(round(0.01 * q.size)))
    smooth = np.convolve(q, np.ones(window) / window, mode="same")
    peak = int(np.argmax(smooth))
    tail = q.size - peak
    start = peak + int(math.floor((1.0 - float(late_window_fraction)) * tail))
    start = min(max(start, peak + 1), q.size - 1)

    seg_a = eps_a[start:]
    seg_v = eps_v[start:]
    psi, slope = dilation_angle_deg(seg_a, seg_v)
    phi_late = secant_friction_deg(float(np.mean(ratio[start:])))
    phi_peak = secant_friction_deg(float(ratio[peak]))
    return {
        "phi_deg": float(phi_late),
        "phi_peak_deg": float(phi_peak),
        "psi_deg": float(psi),
        "slope": float(slope),
        "sigma_ratio": float(np.mean(ratio[start:])),
        "strain_span": float(seg_a[-1] - seg_a[0]),
        "rows": int(q.size - start),
        "peak_index": peak,
        "late_start_index": int(start),
    }


def late_window_band_angle(band_history: Iterable[Mapping], *,
                           tail_start_strain: float,
                           resolved_std_deg: float = 10.0,
                           min_snapshots: int = 3) -> dict:
    """Late-window band angle from ``result["band"]["history"]``.

    Reproduces the published estimator: snapshots at or after
    ``tail_start_strain`` (linear axial strain) are kept, the incremental
    ``folded_angle_deg`` values are averaged, and the window is *resolved* when
    at least ``min_snapshots`` agree within ``resolved_std_deg``.  The
    cumulative variant (``block["cumulative"]["folded_angle_deg"]``) is
    returned alongside so the two published statistics can be compared.
    """
    late = [b for b in band_history
            if float(b.get("axial_strain", -1.0)) >= float(tail_start_strain)
            and np.isfinite(b.get("folded_angle_deg", np.nan))]
    incremental = np.array([float(b["folded_angle_deg"]) for b in late], dtype=float)
    cumulative = np.array([float((b.get("cumulative") or {}).get("folded_angle_deg", np.nan))
                           for b in late], dtype=float)
    cumulative = cumulative[np.isfinite(cumulative)]
    out = {
        "n_snapshots": int(incremental.size),
        "incremental_mean_deg": float(incremental.mean()) if incremental.size else float("nan"),
        "incremental_std_deg": float(incremental.std()) if incremental.size else float("nan"),
        "cumulative_mean_deg": float(cumulative.mean()) if cumulative.size else float("nan"),
        "cumulative_std_deg": float(cumulative.std()) if cumulative.size else float("nan"),
        "resolved": bool(incremental.size >= int(min_snapshots)
                         and (incremental.size > 0
                              and float(incremental.std()) < float(resolved_std_deg))),
    }
    return out


def law_errors(theta_meas_deg: float, phi_deg: float,
               psi_deg: float) -> dict[str, dict]:
    """Signed and relative error of ``theta_meas`` against each law."""
    theta = float(theta_meas_deg)
    predictions = theory_angles_deg(float(phi_deg), float(psi_deg))
    out: dict[str, dict] = {}
    for law in LAWS:
        pred = float(predictions[law])
        out[law] = {
            "predicted_deg": pred,
            "signed_error_deg": theta - pred,
            "relative_error": (theta - pred) / pred if pred else float("nan"),
            "abs_relative_error": abs(theta - pred) / pred if pred else float("nan"),
        }
    return out


def adjudicate(theta_meas_deg: float, phi_deg: float, psi_deg: float, *,
               tolerance: float = 0.15, precision: float = 0.03) -> dict:
    """Which laws the measured angle is consistent with.

    A law is *tolerance-consistent* when ``|theta_meas - theta_law| / theta_law
    <= tolerance`` and *precision-consistent* when it meets ``precision``.  The
    closest law is reported with the margin to the runner-up, so a verdict of
    "law X" is only accepted when it beats the runner-up by
    ``discrimination_deg`` — otherwise the comparison does not discriminate and
    the caller must say so.
    """
    errors = law_errors(theta_meas_deg, phi_deg, psi_deg)
    order = sorted(LAWS, key=lambda k: errors[k]["abs_relative_error"])
    best, second = order[0], order[1]
    margin = (errors[second]["abs_relative_error"] - errors[best]["abs_relative_error"])
    return {
        "errors": errors,
        "closest_law": best,
        "runner_up": second,
        "margin_relative": float(margin),
        "discrimination_deg": abs(errors[second]["predicted_deg"]
                                  - errors[best]["predicted_deg"]),
        "within_tolerance": [k for k in LAWS
                             if errors[k]["abs_relative_error"] <= float(tolerance)],
        "within_precision": [k for k in LAWS
                             if errors[k]["abs_relative_error"] <= float(precision)],
        "tolerance": float(tolerance),
        "precision": float(precision),
    }


def anchor_implied_angles(theta_anchor_deg: float = 60.0) -> dict:
    """What the borrowed anchor *assumes*, inverted through each law.

    ``theta = 45 + phi / 2`` inverts to ``phi = 2 (theta - 45)``; likewise the
    ``psi`` laws.  For ``theta = 60`` this gives ``phi = 30`` — the material
    assumption inside the 55-60 deg literature anchor, not a prediction for a
    specimen whose own secant friction angle is measured in the run.
    """
    theta = float(theta_anchor_deg)
    return {
        "theta_anchor_deg": theta,
        "mohr_coulomb_implied_phi_deg": 2.0 * (theta - 45.0),
        "roscoe_implied_psi_deg": 2.0 * (theta - 45.0),
        "arthur_implied_phi_plus_psi_deg": 4.0 * (theta - 45.0),
        "note": ("the anchor fixes a material state (phi = 30 deg for the "
                 "Mohr-Coulomb reading); a specimen with a different measured "
                 "phi is not predicted to sit at the same angle"),
    }