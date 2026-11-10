"""The mu(I) closure and its calibration against DEM steady-shear data.

Law (Jop, Forterre & Pouliquen 2006)
------------------------------------
.. math:: \\tau = \\mu(I) P,\\qquad
          \\mu(I) = \\mu_s + \\frac{\\mu_2-\\mu_s}{I_0/I+1},\\qquad
          I = \\frac{\\dot\\gamma d}{\\sqrt{P/\\rho}} .

``I -> 0`` gives ``mu_s`` (quasi-static), ``I -> inf`` gives ``mu_2``
(collisional).  The three constants are *not* free in the resistance model: the
standard Jop values (0.36, 0.84, 0.30) are used, and this module measures how
well they can be recovered from the repository's own DEM steady-shear device
(:func:`tensordem.mu_i_rheology.simulate_servo_shear`, constant-stress servo
shear cell) - i.e. criterion (a) of the campaign node is a *reproducibility*
check on the constitutive calibration, not a fitting knob for the resistance
model.

calibrate: 5-8 inertial-number targets, each ~25 s of DEM on CPU, so the sweep is
run once and cached in ``artifacts/brash_ice_rheology/mu_i_servo_sweep.json``
(:func:`run_shear_sweep` writes it, :func:`calibration_report` re-fits it in
milliseconds).  Tests re-run a two-point subset to prove the cache is live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "JopParameters",
    "LITERATURE_JOP",
    "inertial_number",
    "mu_of_I",
    "run_shear_sweep",
    "fit_sweep",
    "calibration_report",
    "SWEEP_TARGETS",
]

#: Literature constants (Jop et al. 2006), identical to
#: ``tensordem.granular_column.scaling.JOP_MU_S/_MU_2/_I_0``.
LITERATURE_JOP: Tuple[float, float, float] = (0.36, 0.84, 0.30)

#: Inertial-number targets of the calibration sweep (log-spaced over the dense
#: to collisional transition).
SWEEP_TARGETS: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 0.6)


@dataclass(frozen=True)
class JopParameters:
    """The three mu(I) constants."""

    mu_s: float = LITERATURE_JOP[0]
    mu_2: float = LITERATURE_JOP[1]
    i_0: float = LITERATURE_JOP[2]

    def __post_init__(self) -> None:
        if not 0.0 < self.mu_s < self.mu_2:
            raise ValueError("need 0 < mu_s < mu_2")
        if not self.i_0 > 0.0:
            raise ValueError("i_0 must be positive")

    def mu(self, inertial: float) -> float:
        """``mu(I)``; the ``I -> 0`` limit is ``mu_s`` (no division by zero)."""
        i = max(float(inertial), 1e-30)
        return self.mu_s + (self.mu_2 - self.mu_s) / (self.i_0 / i + 1.0)

    def relative_errors(self, reference: "JopParameters" = None) -> Dict[str, float]:
        """Relative error of each constant against ``reference`` (default: literature)."""
        ref = reference or JopParameters(*LITERATURE_JOP)
        return {
            "mu_s": (self.mu_s - ref.mu_s) / ref.mu_s,
            "mu_2": (self.mu_2 - ref.mu_2) / ref.mu_2,
            "i_0": (self.i_0 - ref.i_0) / ref.i_0,
        }

    def as_dict(self) -> Dict[str, float]:
        return {"mu_s": self.mu_s, "mu_2": self.mu_2, "i_0": self.i_0}


def inertial_number(shear_rate: float, diameter: float, pressure: float,
                    density: float) -> float:
    """``I = gamma_dot d / sqrt(P/rho)`` (SI), guarding ``P <= 0``."""
    if pressure <= 0.0:
        raise ValueError("pressure must be positive")
    if density <= 0.0:
        raise ValueError("density must be positive")
    if shear_rate < 0.0:
        raise ValueError("shear_rate must be non-negative")
    return shear_rate * diameter / math.sqrt(pressure / density)


def mu_of_I(inertial: float, params: JopParameters = None) -> float:
    """Module-level shortcut for ``JopParameters.mu``."""
    return (params or JopParameters()).mu(inertial)


def run_shear_sweep(targets: Sequence[float] = SWEEP_TARGETS, *,
                    contact_friction: float = LITERATURE_JOP[0],
                    rolling_friction: float = 0.0,
                    normal_stress_pa: float = 2.0e3,
                    shear_strain: float = 0.15,
                    settle_steps: int = 1500,
                    min_shear_steps: int = 1000,
                    seed: int = 5279) -> List[Dict[str, float]]:
    """Run the in-repository DEM servo shear cell at each inertial-number target.

    Returns a list of measurement records ``{"inertial_number_target",
    "inertial_number_measured", "mu_measured", "normal_stress_pa",
    "shear_stress_pa", "wall_clock_s"}``.  ~25 s per point on one CPU core, so
    callers should cache the result (see :mod:`tensordem.brash_ice_rheology.study`).
    """
    from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear

    records: List[Dict[str, float]] = []
    for target in targets:
        case = ServoShearCase(inertial_number=float(target),
                              mu=float(contact_friction),
                              mu_rolling=float(rolling_friction),
                              normal_stress=float(normal_stress_pa),
                              shear_strain=float(shear_strain),
                              settle_steps=int(settle_steps),
                              min_shear_steps=int(min_shear_steps),
                              seed=int(seed))
        result = simulate_servo_shear(case)
        records.append({
            "inertial_number_target": float(target),
            "inertial_number_measured": float(result["inertial_number"]),
            "mu_measured": float(result["mu_effective"]),
            "normal_stress_pa": float(result["normal_stress_measured_pa"]),
            "shear_stress_pa": float(result["shear_stress_pa"]),
            "contact_friction": float(contact_friction),
            "rolling_friction": float(rolling_friction),
            "coordination_number": float(result["coordination_number"]),
            "solid_fraction": float(result["solid_fraction"]),
            "wall_clock_s": float(result["wall_clock_s"]),
        })
    return records


def fit_sweep(records: Iterable[Dict[str, float]]) -> Dict[str, object]:
    """Bounded least-squares fit of ``mu(I)`` to measured steady-shear points.

    Wraps :func:`tensordem.granular_column.scaling.fit_jop_law` (the
    repository's bounded non-linear least-squares implementation of the fit;
    :func:`tensordem.mu_i_rheology.fit_jop_law` is a grid search and is only
    used by that module's own self-check).
    """
    from tensordem.granular_column.scaling import fit_jop_law

    rows = list(records)
    if len(rows) < 3:
        raise ValueError("need at least 3 measured points to fit 3 constants")
    i_values = [r["inertial_number_measured"] for r in rows]
    mu_values = [r["mu_measured"] for r in rows]
    fit = fit_jop_law(i_values, mu_values)
    params = JopParameters(mu_s=float(fit["mu_s"]),
                           mu_2=float(fit["mu_2"]),
                           i_0=float(fit["i_0"]))
    predicted = [params.mu(i) for i in i_values]
    measured_mean = sum(mu_values) / len(mu_values)
    rms = math.sqrt(sum((p - m) ** 2 for p, m in zip(predicted, mu_values))
                    / len(mu_values))
    return {
        "fit": fit,
        "parameters": params.as_dict(),
        "n_points": len(rows),
        "rms_residual": rms,
        "rms_relative": rms / measured_mean,
        "measured_mean_mu": measured_mean,
        "predicted": predicted,
        "literature": JopParameters(*LITERATURE_JOP).as_dict(),
        "literature_relative_errors": params.relative_errors(),
    }


def calibration_report(records: Iterable[Dict[str, float]],
                       *, tolerance: float = 0.20) -> Dict[str, object]:
    """Criterion (a): fit the law, then confront the constants with literature.

    Verdicts
    --------
    ``a1_fit_quality``   PASS if ``rms/mean(mu) < tolerance``: the law
                         reproduces the measured steady-shear points.
    ``a2_literature``    PASS if *every* constant is within ``tolerance`` of the
                         Jop et al. (2006) value; this is a genuinely external
                         check of the DEM contact model, not of the fitter.
    """
    out = fit_sweep(records)
    rel = out["literature_relative_errors"]
    worst = max(abs(v) for v in rel.values())
    out["a1_fit_quality"] = {
        "value": out["rms_relative"],
        "threshold": tolerance,
        "verdict": "PASS" if out["rms_relative"] < tolerance else "FAIL",
    }
    out["a2_literature"] = {
        "relative_errors": rel,
        "worst_abs_relative_error": worst,
        "threshold": tolerance,
        "verdict": "PASS" if worst < tolerance else "FAIL",
    }
    return out
