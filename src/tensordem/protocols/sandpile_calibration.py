"""Rolling-friction calibration surface on the frozen sandpile protocol.

Why this module exists
----------------------
Wensrich & Katterfeld (2012), *Powder Technol.* **217**:409-417, treat the
rolling-friction coefficient of a **sphere** as a *surrogate for particle
shape*: an assembly of angular or otherwise non-spherical grains is replaced by
spheres whose single rolling-friction coefficient ``mu_r`` is **calibrated on a
bulk observable** (the angle of repose), and it is that calibrated value -- not
a single-contact geometric argument -- that carries the shape effect into the
simulation.  The correspondence registered in this repository
(:mod:`tensordem.protocols.sandpile`'s literature table, row
``Wensrich & Katterfeld (2012)``) is monotone in ``mu_r``:
``mu_r = 0 -> 0.5`` maps to a repose angle of ``21 deg -> 32 deg``.

:mod:`tensordem.protocols.sandpile` deliberately refuses to calibrate: its
baseline is prior-guarded (``assert_priors``) and a hand-tuned ``mu_r`` chasing
a target repose angle is rejected before the first step, so a *validation*
anchor cannot be fished.  The guard also means the surrogate's calibration
curve **cannot be produced at all** -- the one object the literature procedure
needs.  Previous rounds of this anchor therefore compared a single prior run
(``mu_r = 0.05``) against the literature band and had no mapping to invert.

This module is the separate, explicitly flagged calibration surface:

* every point runs the *same* frozen protocol and the *same* measurement code
  (:class:`tensordem.protocols.sandpile.SandpileRun`) with
  ``calibration_sweep=True``; the record is tagged ``kind="calibration_sweep"``,
  ``calibration_disabled=False`` and ``prior_guard="bypassed:calibration_sweep"``
  so a calibration point can never be read as a guarded baseline;
* :func:`fit_saturation` / :func:`invert_target` turn the surface into a
  *predictor*, and :func:`transfer_error` scores it on conditions that were not
  used for the fit (different seed / grain count), i.e. the surrogate is tested
  out of sample rather than only fitted;
* :func:`parse_wk2012_reference` reads the literature correspondence out of the
  in-repository table at run time (no hand-copied constants), so the comparison
  target stays traceable to the file and its sha256.

Conventions: SI units throughout, angles in degrees at the interface (the
protocol measures ``atan(-slope)`` of the flank), coefficients dimensionless.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .sandpile import PRIORS, SandpileConfig, SandpileProtocolError, SandpileRun

__all__ = [
    "UnattainableTarget",
    "CalibrationCondition",
    "CalibrationPoint",
    "SaturationFit",
    "WK2012Reference",
    "run_point",
    "sweep",
    "fit_saturation",
    "invert_target",
    "transfer_error",
    "parse_wk2012_reference",
]

#: In-repository literature table holding the registered Wensrich & Katterfeld
#: (2012) rolling-friction -> repose correspondence.
REFERENCE_DOC = "docs/GAP_DEM_CLUMP_ROUND2.md"

#: Regex for that table row: ``mu_r 0->0.5`` mapped to ``21 deg -> 32 deg``.
_WK2012_ROW = re.compile(r"Wensrich\s*&\s*Katterfeld")
_WK2012_MAP = re.compile(
    r"μ_r\s*([0-9.]+)\s*(?:→|->)\s*([0-9.]+)\s*对应\s*([0-9.]+)\s*°\s*(?:→|->)\s*([0-9.]+)\s*°")


class UnattainableTarget(ValueError):
    """The requested repose angle lies outside the measured calibration range."""

    def __init__(self, target_deg: float, attainable: tuple[float, float]) -> None:
        self.target_deg = float(target_deg)
        self.attainable = (float(attainable[0]), float(attainable[1]))
        super().__init__(
            f"target repose angle {target_deg:.3f} deg is outside the measured "
            f"attainable range [{attainable[0]:.3f}, {attainable[1]:.3f}] deg")


# ---------------------------------------------------------------------------
# Protocol condition (everything except the swept coefficients)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationCondition:
    """The fixed part of a calibration point's protocol.

    ``pour_rate``/``drop_height`` set the feed: ``injection_velocity`` is the
    nozzle exit speed implied by the mass flow rate (clamped to [0.05, 3] m/s by
    the protocol), so a quiet feed is a low rate, not a hand-set velocity.
    """

    n_particles: int = 2400
    radius: float = 0.005
    density: float = 2500.0
    seed: int = 11
    pour_rate: float = 890.0
    drop_height: float = 0.005
    settle_time: float = 0.15
    dt: float | None = None
    restitution: float = 0.4

    def config(self, mu_r: float, *, mu_s: float = PRIORS["mu_s"],
               **overrides: Any) -> SandpileConfig:
        """Frozen protocol config for one calibration point."""
        kwargs: dict[str, Any] = dict(
            n_particles=self.n_particles,
            radius=self.radius,
            density=self.density,
            seed=self.seed,
            pour_rate=self.pour_rate,
            drop_height=self.drop_height,
            settle_time=self.settle_time,
            restitution=self.restitution,
            mu_s=mu_s,
            mu_r=float(mu_r),
            calibration_sweep=True,
        )
        if self.dt is not None:
            kwargs["dt"] = self.dt
        kwargs.update(overrides)
        return SandpileConfig(**kwargs)

    def as_record(self) -> dict[str, Any]:
        return {
            "n_particles": self.n_particles,
            "radius_m": self.radius,
            "density_kg_m3": self.density,
            "seed": self.seed,
            "pour_rate_grains_s": self.pour_rate,
            "drop_height_m": self.drop_height,
            "settle_time_s": self.settle_time,
            "dt_s": self.dt,
            "restitution": self.restitution,
        }


# ---------------------------------------------------------------------------
# One calibration point
# ---------------------------------------------------------------------------

@dataclass
class CalibrationPoint:
    """One ``(mu_r -> repose angle)`` point of the calibration surface."""

    mu_r: float
    mu_s: float
    theta_deg: float
    theta_stderr_deg: float
    r2: float
    packing_bulk: float
    apex_z: float
    toe_radius: float
    radius: float
    steps: int
    simulated_time_s: float
    injection_velocity_m_s: float
    wall_seconds: float
    condition: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def height_over_radius(self) -> float:
        """Apex height in grain radii -- how many layers the flank has."""
        return float(self.apex_z / self.radius) if self.radius > 0 else float("nan")

    def as_row(self) -> dict[str, Any]:
        return {
            "mu_r": self.mu_r,
            "mu_s": self.mu_s,
            "theta_deg": self.theta_deg,
            "theta_stderr_deg": self.theta_stderr_deg,
            "r2": self.r2,
            "packing_bulk": self.packing_bulk,
            "apex_z_m": self.apex_z,
            "toe_radius_m": self.toe_radius,
            "height_over_radius": self.height_over_radius,
            "steps": self.steps,
            "simulated_time_s": self.simulated_time_s,
            "injection_velocity_m_s": self.injection_velocity_m_s,
            "wall_seconds": self.wall_seconds,
            "condition": self.condition,
        }

    @classmethod
    def from_record(cls, mu_r: float, mu_s: float, record: dict[str, Any],
                    wall_seconds: float) -> "CalibrationPoint":
        fit = record["repose_fit"]
        packing = record.get("packing", {})
        bulk = float(packing.get("bulk", {}).get("packing_fraction", float("nan")))
        return cls(
            mu_r=float(mu_r),
            mu_s=float(mu_s),
            theta_deg=float(record["repose_deg"]),
            theta_stderr_deg=float(fit["theta_stderr_deg"]),
            r2=float(fit["r2"]),
            packing_bulk=bulk,
            apex_z=float(fit["apex_z"]),
            toe_radius=float(fit["toe_radius"]),
            radius=float(record["radius_m"]),
            steps=int(record["steps"]),
            simulated_time_s=float(record["simulated_time_s"]),
            injection_velocity_m_s=float(record["injection_velocity_m_s"]),
            wall_seconds=float(wall_seconds),
            condition={
                "n_particles": record["n_particles"],
                "seed": record["seed"],
                "pour_rate_grains_s": record["pour_rate_grains_s"],
                "drop_height_m": record["drop_height_m"],
                "settle_time_s": record["settle_time_s"],
                "dt_s": record["dt_s"],
                "restitution": record["restitution"],
            },
            record=record,
        )


def run_point(mu_r: float, condition: CalibrationCondition, *,
              mu_s: float = PRIORS["mu_s"], **overrides: Any) -> CalibrationPoint:
    """Run one calibration point through the frozen sandpile protocol."""
    config = condition.config(mu_r, mu_s=mu_s, **overrides)
    started = time.time()
    run = SandpileRun(config)
    record = run.run()
    wall = time.time() - started
    if record.get("kind") != "calibration_sweep":
        raise AssertionError(
            "calibration point was not tagged as a calibration_sweep run: "
            f"kind={record.get('kind')!r}")
    return CalibrationPoint.from_record(mu_r, mu_s, record, wall)


def sweep(mu_rs: Iterable[float], condition: CalibrationCondition, *,
          mu_s: float = PRIORS["mu_s"], **overrides: Any) -> list[CalibrationPoint]:
    """Run the surface at each ``mu_r`` in ``mu_rs`` (ascending order kept)."""
    return [run_point(mu_r, condition, mu_s=mu_s, **overrides)
            for mu_r in mu_rs]


# ---------------------------------------------------------------------------
# Surrogate: saturation fit, inversion, out-of-sample transfer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SaturationFit:
    """``theta(mu_r) = theta_inf - amplitude * exp(-mu_r / mu_c)``.

    The saturating form is the qualitative content of the literature
    correspondence (repose angle rises with ``mu_r`` and flattens once the
    rolling limit stops being the binding constraint); ``theta_inf`` is the
    plateau the sphere surrogate can reach at fixed ``mu_s``.
    """

    theta_inf: float
    amplitude: float
    mu_c: float
    rmse_deg: float
    n_points: int

    def __call__(self, mu_r: float | np.ndarray) -> Any:
        return self.theta_inf - self.amplitude * np.exp(-np.asarray(mu_r) / self.mu_c)

    def as_record(self) -> dict[str, Any]:
        return {
            "form": "theta_inf - amplitude*exp(-mu_r/mu_c)",
            "theta_inf_deg": self.theta_inf,
            "amplitude_deg": self.amplitude,
            "mu_c": self.mu_c,
            "rmse_deg": self.rmse_deg,
            "n_points": self.n_points,
        }


def _sorted_xy(points: Sequence[CalibrationPoint]) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ValueError("need at least two calibration points")
    pts = sorted(points, key=lambda p: p.mu_r)
    x = np.array([p.mu_r for p in pts], dtype=np.float64)
    y = np.array([p.theta_deg for p in pts], dtype=np.float64)
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("duplicate mu_r values in the calibration surface")
    if not np.all(np.isfinite(y)):
        raise ValueError("non-finite repose angle in the calibration surface")
    return x, y


def fit_saturation(points: Sequence[CalibrationPoint]) -> SaturationFit:
    """Least-squares saturation fit of ``theta(mu_r)`` (bounded, deterministic)."""
    from scipy.optimize import curve_fit

    x, y = _sorted_xy(points)
    if len(points) < 3:
        raise ValueError("saturation fit needs at least three points")

    def model(m, theta_inf, amplitude, mu_c):
        return theta_inf - amplitude * np.exp(-m / mu_c)

    lower = [float(y.max()) - 1e-6, 0.0, 1e-3]
    upper = [max(float(y.max()) + 90.0, 180.0), 180.0, 50.0]
    p0 = [float(y.max()) + 5.0, max(float(y.max() - y.min()), 1.0), 0.3]
    popt, _ = curve_fit(model, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
    resid = y - model(x, *popt)
    return SaturationFit(
        theta_inf=float(popt[0]), amplitude=float(popt[1]),
        mu_c=float(popt[2]),
        rmse_deg=float(np.sqrt(np.mean(resid ** 2))), n_points=len(points))


def invert_target(points: Sequence[CalibrationPoint], theta_target_deg: float) -> dict:
    """Invert the measured surface for the ``mu_r`` that gives a target angle.

    Inversion is piecewise linear on the *measured* points (no extrapolation):
    a target outside the measured range raises :class:`UnattainableTarget` with
    the attainable interval, which is the honest answer to "can a sphere
    surrogate represent this material at all".
    """
    x, y = _sorted_xy(points)
    lo, hi = float(y.min()), float(y.max())
    target = float(theta_target_deg)
    if not (lo <= target <= hi):
        raise UnattainableTarget(target, (lo, hi))
    order = np.argsort(y, kind="stable")
    ys = y[order]
    xs = x[order]
    # monotone inversion: collapse plateaus by taking the mean mu_r of the bracket
    idx = int(np.searchsorted(ys, target, side="left"))
    if idx <= 0:
        mu_r_star = float(xs[0])
        bracket = (float(xs[0]), float(xs[1]))
        theta_bracket = (float(ys[0]), float(ys[1]))
    elif idx >= len(ys):
        mu_r_star = float(xs[-1])
        bracket = (float(xs[-2]), float(xs[-1]))
        theta_bracket = (float(ys[-2]), float(ys[-1]))
    else:
        y0, y1 = float(ys[idx - 1]), float(ys[idx])
        x0, x1 = float(xs[idx - 1]), float(xs[idx])
        bracket = (x0, x1)
        theta_bracket = (y0, y1)
        if y1 == y0:
            mu_r_star = 0.5 * (x0 + x1)
        else:
            frac = (target - y0) / (y1 - y0)
            mu_r_star = x0 + frac * (x1 - x0)
    return {
        "theta_target_deg": target,
        "mu_r_star": mu_r_star,
        "mu_r_bracket": list(bracket),
        "theta_bracket_deg": list(theta_bracket),
        "method": "piecewise_linear_on_measured_points",
        "attainable_deg": [lo, hi],
    }


def transfer_error(calibration: Sequence[CalibrationPoint],
                   holdout: Sequence[CalibrationPoint], *,
                   fit: SaturationFit | None = None) -> dict:
    """Score the surface fitted on ``calibration`` against held-out points.

    This is the only non-tautological accuracy statement available for a
    surrogate: the held-out points differ in seed and/or grain count, so the
    prediction is out of sample.
    """
    fit = fit_saturation(calibration) if fit is None else fit
    rows = []
    for point in sorted(holdout, key=lambda p: p.mu_r):
        pred = float(fit(point.mu_r))
        rel = abs(pred - point.theta_deg) / abs(point.theta_deg) if point.theta_deg else float("inf")
        rows.append({
            "mu_r": point.mu_r,
            "theta_holdout_deg": point.theta_deg,
            "theta_predicted_deg": pred,
            "abs_error_deg": pred - point.theta_deg,
            "rel_error": rel,
            "condition": point.condition,
        })
    worst = max((r["rel_error"] for r in rows), default=float("nan"))
    worst_abs = max((abs(r["abs_error_deg"]) for r in rows), default=float("nan"))
    return {
        "fit": fit.as_record(),
        "points": rows,
        "worst_rel_error": worst,
        "worst_abs_error_deg": worst_abs,
    }


# ---------------------------------------------------------------------------
# Literature reference (read, never retyped)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WK2012Reference:
    """The registered Wensrich & Katterfeld (2012) correspondence."""

    mu_r: tuple[float, float]
    theta_deg: tuple[float, float]
    source_path: str
    source_sha256: str
    source_line: int

    @property
    def sensitivity_deg_per_unit_mu_r(self) -> float:
        d_mu = self.mu_r[1] - self.mu_r[0]
        return (self.theta_deg[1] - self.theta_deg[0]) / d_mu

    def as_record(self) -> dict[str, Any]:
        return {
            "mu_r": list(self.mu_r),
            "theta_deg": list(self.theta_deg),
            "sensitivity_deg_per_unit_mu_r": self.sensitivity_deg_per_unit_mu_r,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_line": self.source_line,
            "note": ("registered literature correspondence (Wensrich & Katterfeld "
                     "2012, Powder Technol. 217:409-417), parsed from the "
                     "in-repository literature table; the paper itself is not "
                     "in this environment"),
        }


def parse_wk2012_reference(root: str | Path = ".", *,
                           relpath: str = REFERENCE_DOC) -> WK2012Reference:
    """Parse the ``mu_r 0 -> 0.5 == 21 deg -> 32 deg`` row (zero hand-copy)."""
    path = Path(root) / relpath
    if not path.is_file():
        raise FileNotFoundError(f"literature table not found: {path}")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not _WK2012_ROW.search(line):
            continue
        match = _WK2012_MAP.search(line)
        if match is None:
            continue
        mu0, mu1, th0, th1 = (float(g) for g in match.groups())
        return WK2012Reference(
            mu_r=(mu0, mu1), theta_deg=(th0, th1),
            source_path=str(relpath),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_line=lineno)
    raise SandpileProtocolError(
        f"Wensrich & Katterfeld row with a mu_r->theta correspondence not found "
        f"in {relpath}")