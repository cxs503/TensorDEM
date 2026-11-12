"""Rotary-drum flow-regime map: Froude classification, free surface, power ledger.

Physics background
------------------
The transverse motion of a granular bed in a rotating cylinder is described by
the **Froude number** of the shell

    Fr = omega^2 R / g,                                (1)

the ratio of centrifugal to gravitational acceleration.  Two limits are exact:

* ``Fr >= 1``: the radial force balance for a particle on the bare shell,
  ``N = m (omega^2 R + g cos theta)`` (``theta`` measured from the 6 o'clock
  position), can no longer become zero anywhere, so the load is pinned to the
  wall (centrifuging);
* ``Fr < 1``: a particle loses wall contact at the *release angle*

      theta_d = arccos(-Fr)                              (2)

  measured from the bottom, and thereafter follows a ballistic arc.  Above
  ``theta_d`` the shell carries the particle with no support from the bed, which
  is the cataracting (throwing) mechanism.

The regime thresholds used by the machine classifier are therefore *not*
arbitrary: ``Fr = 1`` is exact, and ``Fr = cos(135 deg) = 0.7071`` is the value
at which the release point sits 45 deg above the horizontal (mid-upper
quadrant), i.e. the classical cataracting onset.

The four regimes followed here are

============  =====================================================
``slipping``  bed slides on the shell, wall layer lags the wall a lot
``rolling``   bed is carried round, free surface flows back (avalanche)
``cataracting`` particles released from the shell fly across the void
``centrifuging`` load pinned as an annulus, no free surface
============  =====================================================

Power ledger
------------
For the mechanical closure, ``P_in = tau*omega`` is the power the shell feeds
into the load and ``Phi_diss`` is the sum of the local contact dissipations
(normal dashpot + Coulomb friction slip + rolling resistance).  Over a
measurement window the book-keeping identity is

    d(K + U_g + U_el)/dt = P_in - Phi_diss ,

so in a statistically steady state ``P_in = Phi_diss``.
"""
from __future__ import annotations

import math

import numpy as np

from .kernel import DrumConfig, make_drum_state, run_drum

__all__ = [
    "fill_to_chord_angle",
    "drum_state",
    "fit_free_surface",
    "analytic_surface_curvature",
    "free_surface_fit_quality",
    "detachment_angle_deg",
    "cataract_onset_froude",
    "froude",
    "omega_for_froude",
    "classify_froude",
    "classify_kinematic",
    "simulate_point",
    "FR_BANDS",
    "MELLMANN_BANDS",
    "MEASURED_REGIME_THRESHOLDS",
    "classify_froude_literature",
    "classify_measured",
    "REGIMES",
]

REGIMES = ("slipping", "rolling", "cataracting", "centrifuging")

#: Fr-band classifier prescribed by the campaign node (IN32-D01).
FR_BANDS = ((0.0, 0.3, "slipping"), (0.3, 0.7, "rolling"),
            (0.7, 1.0, "cataracting"), (1.0, math.inf, "centrifuging"))

#: thresholds of the *kinematic* (measurement based) classifier
KINEMATIC_THRESHOLDS = {
    "airborne_cataracting": 0.02,   # fraction of load with no contact at all
    "annulus_centrifuging": 0.60,   # 5th percentile of r/R (no free surface if large)
    "wall_slip_slipping": 0.35,     # wall-layer tangential speed / (omega*R)
    "wall_grip_centrifuging": 0.80,
}


def froude(omega: float, radius: float, gravity: float = 9.81) -> float:
    """Fr = omega^2 R / g."""
    return float(omega) ** 2 * float(radius) / float(gravity)


def omega_for_froude(fr: float, radius: float, gravity: float = 9.81) -> float:
    """Inverse of :func:`froude` (takes the positive root)."""
    if fr < 0.0:
        raise ValueError("Fr must be non-negative")
    return math.sqrt(float(fr) * float(gravity) / float(radius))


def fill_to_chord_angle(fill: float) -> float:
    """Half-angle ``alpha`` (rad from the bottom) of a circular segment of area fraction ``fill``."""
    from scipy.optimize import brentq

    def resid(a):
        return (2.0 * a - math.sin(2.0 * a)) / (2.0 * math.pi) - fill

    return brentq(resid, 1e-9, math.pi - 1e-9)


def detachment_angle_deg(fr: float) -> float:
    """Release angle theta_d = arccos(-Fr) from the 6 o'clock position (deg).

    Returns 180 deg for Fr >= 1 (no release below the top of the shell).
    """
    if fr >= 1.0:
        return 180.0
    return math.degrees(math.acos(-float(fr)))


def cataract_onset_froude(theta_d_deg: float = 135.0) -> float:
    """Fr at which the release angle reaches ``theta_d_deg``: Fr = -cos(theta_d)."""
    return -math.cos(math.radians(theta_d_deg))


def classify_froude(fr: float) -> str:
    """Regime label from the prescribed Froude bands."""
    if fr < 0.0 or not math.isfinite(fr):
        raise ValueError("Fr must be finite and non-negative")
    for lo, hi, name in FR_BANDS:
        if lo <= fr < hi:
            return name
    return "centrifuging"


def classify_kinematic(obs: dict, thresholds: dict | None = None) -> str:
    """Regime label from measured bed kinematics (independent of Fr).

    ``obs`` needs ``airborne_fraction``, ``r_p05_over_R`` and
    ``wall_velocity_ratio`` (all produced by :func:`simulate_point`).  The rules
    follow the physics of the mechanisms, in order of exclusivity:

    1. a load with a substantial detached (ballistic) population is
       *cataracting*;
    2. a load that no longer touches the core (large ``r_p05``) and is carried
       with the shell (wall layer co-rotating) is *centrifuging*; the annulus
       test is what separates it from rolling, because in a rolling bed the free
       surface reaches close to the axis;
    3. a load whose wall layer lags the shell badly is *slipping*;
    4. otherwise *rolling*.
    """
    th = dict(KINEMATIC_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    airborne = float(obs["airborne_fraction"])
    p05 = float(obs["r_p05_over_R"])
    wall = float(obs["wall_velocity_ratio"])
    if airborne > th["airborne_cataracting"]:
        return "cataracting"
    if p05 > th["annulus_centrifuging"] and wall > th["wall_grip_centrifuging"]:
        return "centrifuging"
    if wall < th["wall_slip_slipping"]:
        return "slipping"
    return "rolling"


# ---------------------------------------------------------------------------
# free surface
# ---------------------------------------------------------------------------
def fit_free_surface(pos: np.ndarray, n_bins: int = 16, min_count: int = 2) -> dict:
    """Fit ``y = a + b x + c x^2`` to the upper boundary of a 2-D particle cloud.

    For each vertical strip the *highest* particle marks the free surface; strips
    with fewer than ``min_count`` particles are dropped.  Returns the
    coefficients, the coefficient of determination ``r2`` and the RMS residual.
    """
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] < 4:
        raise ValueError("pos must be an (N, 2) array with N >= 4")
    x = pos[:, 0]
    y = pos[:, 1]
    edges = np.linspace(x.min() - 1e-12, x.max() + 1e-12, n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if int(mask.sum()) >= min_count:
            xs.append(float(x[mask].mean()))
            ys.append(float(y[mask].max()))
    if len(xs) < 4:
        raise ValueError("not enough populated strips to fit a parabola")
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    coef = np.polyfit(xs, ys, 2)          # [c, b, a]
    pred = np.polyval(coef, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "a": float(coef[2]), "b": float(coef[1]), "c": float(coef[0]),
        "r2": float(r2),
        "rms": float(np.sqrt(ss_res / xs.size)),
        "n_strips": int(xs.size),
        "x_range": (float(xs.min()), float(xs.max())),
        "x": xs.tolist(), "y": ys.tolist(),
    }


def analytic_surface_curvature(omega_bed: float, gravity: float, y_surface: float) -> float:
    """Curvature coefficient ``c`` of the centrifugal free surface (1/m).

    In the frame rotating with the load the effective potential is
    ``Phi = g y - 0.5*omega_bed^2*(x^2 + y^2)``; a pressure-free surface is an
    equipotential, which is a circular arc of centre ``y = g/omega_bed^2``.
    Expanding around the surface apex ``y_s`` gives the parabola

        y(x) = y_s + c x^2,     c = omega_bed^2 / (2 (g - omega_bed^2 y_s)).

    ``c`` is positive (the surface is concave up: the shell-side edges are
    lifted by the centrifugal field).  ``omega_bed -> 0`` recovers the straight
    surface of a slowly rotating bed.
    """
    w2 = float(omega_bed) ** 2
    denom = 2.0 * (float(gravity) - w2 * float(y_surface))
    if abs(denom) < 1e-12:
        return math.inf
    return w2 / denom


def free_surface_fit_quality(fit: dict, analytic_c: float) -> dict:
    """Compare a fitted parabola with the analytic centrifugal curvature."""
    dev = abs(fit["c"] - analytic_c) / abs(analytic_c) if analytic_c not in (0.0, math.inf) else math.nan
    return {"c_fit": fit["c"], "c_analytic": analytic_c, "rel_deviation": float(dev)}


# ---------------------------------------------------------------------------
# one phase-map cell
# ---------------------------------------------------------------------------
def drum_state(cfg: DrumConfig, fill: float, fr: float, seed: int = 0):
    """Build a drum state for a target Froude number."""
    omega = omega_for_froude(fr, cfg.radius, cfg.gravity)
    return make_drum_state(cfg, fill, omega, seed=seed)


def simulate_point(
    fill: float,
    fr: float,
    cfg: DrumConfig | None = None,
    revolutions: float = 2.0,
    measure_last: float = 1.0,
    settle_time: float = 0.15,
    seed: int = 0,
    sample_every: int = 200,
) -> dict:
    """Run one phase-map cell and return regimes/ledger/surface diagnostics."""
    cfg = cfg or DrumConfig()
    state = drum_state(cfg, fill, fr, seed=seed)
    out = run_drum(state, revolutions=revolutions, measure_last=measure_last,
                   sample_every=sample_every, settle_time=settle_time)
    mean = out["mean"]
    ledger = out["ledger"]
    obs = {
        "airborne_fraction": mean.get("airborne_fraction", 0.0),
        "r_p05_over_R": mean.get("r_p05_over_R", 0.0),
        "wall_velocity_ratio": mean.get("wall_velocity_ratio", 0.0),
    }
    result = {
        "fill": float(fill),
        "fill_realised": float(out["fill_realised"]),
        "fr": float(fr),
        "omega": float(state.omega),
        "n_particles": int(out["n_particles"]),
        "steps": int(out["steps"]),
        "observables": obs,
        "omega_bulk_over_omega": mean.get("omega_bulk", 0.0) / state.omega,
        "omega_bed_over_omega": mean.get("omega_bed", 0.0) / state.omega,
        "surface_speed_ratio": mean.get("surface_speed_ratio", 0.0),
        "contact_count": mean.get("n_pair", 0.0),
        "height_com": mean.get("height_com", 0.0),
        "ledger": ledger,
        "regime_froude": classify_froude(fr),
        "regime_kinematic": classify_kinematic(obs),
        "detachment_angle_deg": detachment_angle_deg(fr),
        "chord_half_angle_deg": math.degrees(fill_to_chord_angle(fill)),
    }
    result["flight_upper_fraction"] = mean.get("flight_upper_fraction", 0.0)
    snap = out.get("snapshot")
    if snap is not None:
        try:
            # robust fit: use only particles that carry contacts (the bed); ballistic
            # particles above the surface would bias the "highest per strip" rule
            loaded = snap["nn_contacts"] > 0.0
            bed = snap["pos"][loaded] if int(loaded.sum()) >= 8 else snap["pos"]
            fit_all = fit_free_surface(snap["pos"])
            fit = fit_free_surface(bed)
            c_an = analytic_surface_curvature(mean.get("omega_bulk", 0.0), cfg.gravity, fit["a"])
            result["surface"] = {
                "fit": {k: fit[k] for k in ("a", "b", "c", "r2", "rms", "n_strips", "x_range")},
                "analytic": free_surface_fit_quality(fit, c_an),
                "slope_angle_deg": math.degrees(math.atan(fit["b"])),
                "r2_all_particles": fit_all["r2"],
            }
        except ValueError as exc:
            result["surface"] = {"error": str(exc)}
    return result

# ---------------------------------------------------------------------------
# Literature Froude bands and a purely measured (kinematic) classifier
# ---------------------------------------------------------------------------
#: Form-of-motion bands of the classical rotary-drum diagram as a function of
#: ``Fr = omega^2 R / g`` (Mellmann 2001, "The transverse motion of solids in
#: rotating cylinders - forms of motion and transition behavior", Powder
#: Technology 118:251-270).  Only the two endpoints are exact: ``Fr -> 0`` is
#: pure sliding and ``Fr = 1`` is the centrifuging onset (the normal force at
#: the top of the shell vanishes).  The intermediate boundaries are the
#: literature's order-of-magnitude transitions; this table is used for
#: QUALITATIVE labels only.  Quantitative anchors of the rotating-drum case are
#: ``detachment_angle_deg`` and ``Fr = 1``.
MELLMANN_BANDS = (
    (0.0, 1.0e-4, "sliding"),
    (1.0e-4, 1.0e-3, "surging"),
    (1.0e-3, 1.0e-2, "rolling"),
    (1.0e-2, 1.0e-1, "cascading"),
    (1.0e-1, 1.0, "cataracting"),
    (1.0, math.inf, "centrifuging"),
)

#: thresholds of the measured classifier (frozen before the m15649 runs; see
#: ``prereg/CASE_ROTATING_DRUM_R2.json``).  They act on observables that do not
#: contain Fr, so the resulting label is an independent measurement.
MEASURED_REGIME_THRESHOLDS = {
    "pinned_r_p05_over_R": 0.75,      # load pinned as an annulus -> centrifuging
    "ballistic_upper_fraction": 0.05,  # free flight above the centre -> cataracting
    "wall_slip_ratio": 0.35,           # bed lags the shell -> slipping
    "cascading_wall_ratio": 0.70,      # partial wall coupling -> cascading
}


def classify_froude_literature(fr: float) -> str:
    """Qualitative regime label from the literature Froude diagram."""
    if fr < 0.0 or not math.isfinite(fr):
        raise ValueError("Fr must be finite and non-negative")
    for lo, hi, name in MELLMANN_BANDS:
        if lo <= fr < hi:
            return name
    return "centrifuging"


def classify_measured(obs: dict, thresholds: dict | None = None) -> str:
    """Regime label from measured drum kinematics alone (no Fr input).

    ``obs`` must provide ``r_p05_over_R``, ``upper_flight_fraction`` and
    ``wall_velocity_ratio`` as produced by
    :func:`tensordem.drum_segregation.kernel2d.simulate`.  Rules are applied in
    the frozen order of ``prereg/CASE_ROTATING_DRUM_R2.json``.
    """
    th = thresholds or MEASURED_REGIME_THRESHOLDS
    r_p05 = float(obs.get("r_p05_over_R", float("nan")))
    upper = float(obs.get("upper_flight_fraction", float("nan")))
    wall = float(obs.get("wall_velocity_ratio", float("nan")))
    if math.isfinite(r_p05) and r_p05 >= th["pinned_r_p05_over_R"]:
        return "centrifuging"
    if math.isfinite(upper) and upper >= th["ballistic_upper_fraction"]:
        return "cataracting"
    if math.isfinite(wall) and wall < th["wall_slip_ratio"]:
        return "slipping"
    if math.isfinite(wall) and wall < th["cascading_wall_ratio"]:
        return "cascading"
    return "rolling"
