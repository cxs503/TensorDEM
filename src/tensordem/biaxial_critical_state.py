"""Critical-state detection for the plane-strain biaxial stress path.

Motivation (CASE-DEM-BIAXIAL-FRICTION, round 2)
----------------------------------------------
The production biaxial engine :func:`tensordem.biaxial_shear_band.run_biaxial_shear_band`
records the full stress path (``sigma_v``, ``sigma_h``, ``deviator``,
``volumetric_strain`` vs ``axial_strain``) but only reports the **peak**
Mohr-Coulomb angle.  A peak angle is a function of the preparation history
(density, cementation, specimen size) and therefore cannot be compared with a
*critical-state* reference.  This module adds the missing, separately testable
post-processing step: locate the critical state on the recorded path and
measure

    sin(phi_cv) = (sigma_1 - sigma_3) / (sigma_1 + sigma_3),   c = 0

there, together with the invariants that make the word "critical state"
meaningful:

1. **isochoric flow** -- the volumetric strain rate vanishes,
   ``|d eps_v / d eps_a| -> 0`` (no further dilation or compaction);
2. **stationary stress ratio** -- the deviator stress has plateaued
   (relative drift over the window below a threshold), i.e. no softening;
3. **window independence** -- the angle measured over the last 10 %, 20 % and
   30 % of the sheared path must agree within a declared spread, so the answer
   does not hinge on the arbitrary choice of averaging window;
4. **confining-stress servo closure** -- the measured lateral stress must
   still sit on the prescribed ``sigma_3`` (a drifting servo invalidates the
   Mohr circle).

References
----------
- Schofield & Wroth (1968) *Critical State Soil Mechanics*, McGraw-Hill:
  critical state = stationary stress ratio at constant volume.
- Rowe, P. W. (1962) "The stress-dilatancy relation for static equilibrium of
  an assembly of particles in contact", *Proc. R. Soc. Lond. A* **269**:
  500-527.  ``sigma_1/sigma_3 = tan^2(45 + phi_mu/2) * (1 - d eps_v/d eps_1)``;
  at the critical state (``d eps_v/d eps_1 = 0``) the mobilised angle equals
  the interparticle friction angle, ``phi_cv = arctan(mu)``.
- Bolton, M. D. (1986) "The strength and dilatancy of sands", *Geotechnique*
  **36**(1): 65-78: for angular quartz sands ``phi_crit ~ 33 deg`` and the
  peak is ``phi_peak = phi_crit + A * I_R`` (A = 3 triaxial, 5 plane strain).

All angles in degrees, stresses in Pa, strains compression-positive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "CriticalStateResult",
    "phi_deg_from_stresses",
    "critical_state_from_stress_path",
    "rowe_stress_dilatancy",
    "judge_against_reference",
]

# Default detector settings (declared *before* looking at any result).
DEFAULT_MIN_STRAIN = 0.20      # ignore the pre-peak branch entirely
DEFAULT_WINDOW_FRACTION = 0.20  # primary window = last 20 % of the sheared path
DEFAULT_VOL_TOLERANCE = 0.02   # |d eps_v / d eps_a| <= 2 % counts as isochoric
DEFAULT_DEVIATOR_TOLERANCE = 0.05  # q drift over window <= 5 %
DEFAULT_SERVO_TOLERANCE = 0.10     # sigma_h vs sigma_3 within 10 %
DEFAULT_WINDOW_SPREAD_TOLERANCE = 1.5  # deg, spread across 10/20/30 % windows


def phi_deg_from_stresses(sigma_v: Any, sigma_h: Any) -> np.ndarray:
    """Mohr-Coulomb angle of a cohesionless assembly from axial/lateral stress."""
    s1 = np.asarray(sigma_v, dtype=float)
    s3 = np.asarray(sigma_h, dtype=float)
    denom = s1 + s3
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_phi = np.where(denom > 0.0, (s1 - s3) / denom, np.nan)
    sin_phi = np.clip(sin_phi, -0.95, 0.95)
    return np.degrees(np.arcsin(sin_phi))


def _window_stats(eps: np.ndarray, phi: np.ndarray, q: np.ndarray,
                  ev: np.ndarray, sh: np.ndarray,
                  lo: float, hi: float) -> dict[str, float]:
    mask = (eps >= lo) & (eps <= hi)
    n = int(mask.sum())
    if n < 5:
        return {"n": n, "valid": False}
    e = eps[mask]
    slope = float(np.polyfit(e, ev[mask], 1)[0])
    q_win = q[mask]
    q_mean = float(q_win.mean())
    # relative change of the deviator across the window, from a linear fit:
    # |dq/deps| * window_width / q_mean  (softening/tail plateau test)
    q_slope = float(np.polyfit(e, q_win, 1)[0]) if n >= 5 else 0.0
    q_drift = float(abs(q_slope) * (e[-1] - e[0]) / max(abs(q_mean), 1e-30))
    return {
        "n": n, "valid": True,
        "strain": [float(e[0]), float(e[-1])],
        "phi_deg": float(np.nanmean(phi[mask])),
        "phi_std_deg": float(np.nanstd(phi[mask])),
        "q_Pa": q_mean,
        "q_drift": q_drift,
        "sigma_h_Pa": float(sh[mask].mean()),
        "d_eps_v_d_eps_a": slope,
    }


@dataclass
class CriticalStateResult:
    """Critical-state measurement on one stress path."""

    phi_cv_deg: float
    q_cv_Pa: float
    sigma_h_cv_Pa: float
    sigma3_target_Pa: float
    d_eps_v_d_eps_a: float
    q_drift: float
    servo_relative_error: float
    window: tuple[float, float]
    window_fraction: float
    min_strain: float
    window_sensitivity: dict[str, dict[str, float]] = field(default_factory=dict)
    window_spread_deg: float = float("nan")
    converged: bool = False
    flags: dict[str, bool] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    peak_phi_deg: float = float("nan")
    peak_strain: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "phi_cv_deg": self.phi_cv_deg,
            "q_cv_Pa": self.q_cv_Pa,
            "sigma_h_cv_Pa": self.sigma_h_cv_Pa,
            "sigma3_target_Pa": self.sigma3_target_Pa,
            "d_eps_v_d_eps_a": self.d_eps_v_d_eps_a,
            "q_drift": self.q_drift,
            "servo_relative_error": self.servo_relative_error,
            "window": list(self.window),
            "window_fraction": self.window_fraction,
            "min_strain": self.min_strain,
            "window_sensitivity": self.window_sensitivity,
            "window_spread_deg": self.window_spread_deg,
            "converged": self.converged,
            "flags": dict(self.flags),
            "thresholds": dict(self.thresholds),
            "peak_phi_deg": self.peak_phi_deg,
            "peak_strain": self.peak_strain,
        }


def critical_state_from_stress_path(
    stress_path: dict[str, Sequence[float]],
    *,
    sigma3_target: float | None = None,
    min_strain: float = DEFAULT_MIN_STRAIN,
    window_fraction: float = DEFAULT_WINDOW_FRACTION,
    vol_tolerance: float = DEFAULT_VOL_TOLERANCE,
    deviator_tolerance: float = DEFAULT_DEVIATOR_TOLERANCE,
    servo_tolerance: float = DEFAULT_SERVO_TOLERANCE,
    window_spread_tolerance: float = DEFAULT_WINDOW_SPREAD_TOLERANCE,
) -> CriticalStateResult:
    """Locate the critical state on a recorded biaxial stress path.

    ``stress_path`` must contain ``axial_strain``, ``sigma_v``, ``sigma_h``,
    ``deviator`` and ``volumetric_strain`` (the keys returned by
    :func:`tensordem.biaxial_shear_band.run_biaxial_shear_band`).  The critical
    state is taken as the **last** ``window_fraction`` of the record above
    ``min_strain`` -- a declared rule fixed before the run, so the window is not
    chosen to fit the data.
    """
    eps = np.asarray(stress_path["axial_strain"], dtype=float)
    sv = np.asarray(stress_path["sigma_v"], dtype=float)
    sh = np.asarray(stress_path["sigma_h"], dtype=float)
    q = np.asarray(stress_path.get("deviator", sv - sh), dtype=float)
    ev = np.asarray(stress_path["volumetric_strain"], dtype=float)
    if eps.size < 10:
        raise ValueError("stress path too short for a critical-state window")

    phi = phi_deg_from_stresses(sv, sh)
    shear = eps >= min_strain
    if int(shear.sum()) < 10:
        raise ValueError(
            f"no samples above min_strain={min_strain} (max strain {eps.max():.3f})")
    eps_s, phi_s, q_s, ev_s, sh_s = eps[shear], phi[shear], q[shear], ev[shear], sh[shear]

    hi = float(eps_s[-1])
    lo = hi - window_fraction * (hi - min_strain)
    primary = _window_stats(eps_s, phi_s, q_s, ev_s, sh_s, lo, hi)
    if not primary["valid"]:
        raise ValueError("primary critical-state window has too few samples")

    sensitivity: dict[str, dict[str, float]] = {}
    for frac in (0.10, 0.20, 0.30):
        lo_f = hi - frac * (hi - min_strain)
        sensitivity[f"last_{int(round(frac * 100))}pct"] = _window_stats(
            eps_s, phi_s, q_s, ev_s, sh_s, lo_f, hi)
    valid_angles = [v["phi_deg"] for v in sensitivity.values() if v.get("valid")]
    spread = float(max(valid_angles) - min(valid_angles)) if len(valid_angles) >= 2 \
        else float("nan")

    sigma3_meas = primary["sigma_h_Pa"]
    servo = (abs(sigma3_meas - sigma3_target) / sigma3_target
             if sigma3_target else float("nan"))

    flags = {
        "isochoric": bool(abs(primary["d_eps_v_d_eps_a"]) <= vol_tolerance),
        "deviator_stationary": bool(primary["q_drift"] <= deviator_tolerance),
        "window_independent": bool(
            np.isfinite(spread) and spread <= window_spread_tolerance),
        "servo_on_target": bool(
            not np.isfinite(servo) or servo <= servo_tolerance),
    }
    # peak for reference (not the benchmark quantity)
    peak_mask = eps >= min_strain * 0.0
    peak_index = int(np.argmax(np.where(peak_mask, q, -np.inf)))

    return CriticalStateResult(
        phi_cv_deg=primary["phi_deg"],
        q_cv_Pa=primary["q_Pa"],
        sigma_h_cv_Pa=sigma3_meas,
        sigma3_target_Pa=float(sigma3_target) if sigma3_target else float("nan"),
        d_eps_v_d_eps_a=primary["d_eps_v_d_eps_a"],
        q_drift=primary["q_drift"],
        servo_relative_error=float(servo),
        window=(lo, hi),
        window_fraction=window_fraction,
        min_strain=min_strain,
        window_sensitivity=sensitivity,
        window_spread_deg=spread,
        converged=all(flags.values()),
        flags=flags,
        thresholds={
            "vol_tolerance": vol_tolerance,
            "deviator_tolerance": deviator_tolerance,
            "servo_tolerance": servo_tolerance,
            "window_spread_tolerance_deg": window_spread_tolerance,
        },
        peak_phi_deg=float(phi[peak_index]),
        peak_strain=float(eps[peak_index]),
    )


def rowe_stress_dilatancy(
    stress_path: dict[str, Sequence[float]],
    *,
    mu: float,
    min_strain: float = DEFAULT_MIN_STRAIN,
) -> dict[str, Any]:
    """Rowe (1962) stress-dilatancy cross-check on the sheared branch.

    ``sigma_1/sigma_3 = tan^2(45 + phi_mu/2) * (1 - d eps_v/d eps_1)`` with
    ``phi_mu = arctan(mu)``.  The *predicted* critical-state angle is obtained by
    evaluating the relation at the measured dilatancy of the critical-state
    window; at a true critical state (``D = 0``) it collapses to ``arctan(mu)``,
    which makes the classical prediction and the measurement independently
    comparable.
    """
    eps = np.asarray(stress_path["axial_strain"], dtype=float)
    sv = np.asarray(stress_path["sigma_v"], dtype=float)
    sh = np.asarray(stress_path["sigma_h"], dtype=float)
    ev = np.asarray(stress_path["volumetric_strain"], dtype=float)
    mask = eps >= min_strain
    if int(mask.sum()) < 10:
        raise ValueError("stress path too short for the Rowe check")
    e, s1, s3, v = eps[mask], sv[mask], sh[mask], ev[mask]
    # local dilatancy rate (smoothed central difference)
    k = max(3, int(round(0.01 * e.size)))
    kernel = np.ones(k) / k
    d_ev = np.convolve(np.gradient(v, e), kernel, mode="same")
    ratio_meas = s1 / np.maximum(s3, 1e-30)
    phi_mu = math.degrees(math.atan(mu))
    kappa = math.tan(math.radians(45.0 + 0.5 * phi_mu)) ** 2
    ratio_rowe = kappa * (1.0 - d_ev)
    # trim the convolution edges before summarising
    core = slice(k, -k) if e.size > 2 * k + 5 else slice(0, e.size)
    rel = (ratio_meas[core] - ratio_rowe[core]) / np.maximum(ratio_rowe[core], 1e-30)
    sin_pred = (ratio_rowe[core] - 1.0) / (ratio_rowe[core] + 1.0)
    phi_pred = np.degrees(np.arcsin(np.clip(sin_pred, -0.95, 0.95)))
    return {
        "mu": float(mu),
        "phi_mu_deg": float(phi_mu),
        "kappa_tan2_45_plus_phi_mu_over_2": float(kappa),
        "phi_cv_predicted_deg": float(phi_mu),   # D -> 0 limit
        "measured_minus_rowe_mean": float(np.nanmean(rel)),
        "measured_minus_rowe_std": float(np.nanstd(rel)),
        "phi_from_rowe_relation_mean_deg": float(np.nanmean(phi_pred)),
        "phi_from_rowe_relation_std_deg": float(np.nanstd(phi_pred)),
        "window_strain": [float(e[core][0]), float(e[core][-1])],
        "note": ("Rowe (1962) sigma1/sigma3 = tan^2(45+phi_mu/2)(1 - dEv/dE1); "
                 "at critical state D=0 the prediction is phi_cv = arctan(mu)"),
    }


def judge_against_reference(phi_got_deg: float, phi_ref_deg: float,
                            tolerance: float) -> dict[str, Any]:
    """Relative-error judgement with an explicit tolerance."""
    rel = abs(phi_got_deg - phi_ref_deg) / phi_ref_deg
    return {
        "got_deg": float(phi_got_deg),
        "ref_deg": float(phi_ref_deg),
        "relative_error": float(rel),
        "tolerance": float(tolerance),
        "passed": bool(rel <= tolerance),
    }