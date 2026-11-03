"""Finite-amplitude heave of a floating 2-D ice floe: exact energy-quadrature reference.

Motivation
----------
``docs/ice_floe_physics_validation.md`` (merged) verifies the repository's
buoyancy force law and the *small-amplitude* heave period, and explicitly leaves
one item open (§6):

    "Small-amplitude only.  The period anchor uses 2e-3 R; the nonlinear
     (finite-amplitude) heave period is not verified and is expected to differ."

This module closes that item for the 2-D extruded-disk floe (``model="2d"``).
The reference is a *closed-form potential energy*, obtained from the analytic
primitive of the circular-segment submerged area, so the reference shares no
code path with ``tensordem.buoyancy_wet`` (which differentiates an arccos-based
segment formula numerically).  The nonlinear period then follows from an exact
energy quadrature

    T(E) = 2 * int_{y_min}^{y_max} dy / sqrt(2 (E - V(y)) / m)

evaluated with a cosine substitution ``y = c + d cos(theta)`` that removes the
inverse-square-root singularity at both turning points.

Why the floe is strongly nonlinear here
---------------------------------------
With rho_i/rho_w = 0.8946 the disk floats almost fully submerged
(y_eq ~ -0.338 R-units), so the waterplane width ``w(y) = 2 sqrt(R^2 - y^2)``
— the instantaneous heave stiffness ``k(y) = rho_w g t w(y)`` — varies by tens
of percent over an amplitude of 0.3 R.  The restoring force is therefore
asymmetric: heaving up stiffens, heaving down softens, and the oscillation
acquires (i) an amplitude-dependent period and (ii) a rectified mean draft even
for a symmetric release.  Both effects are measured against the quadrature.

Usage
-----
    python benchmarks/floe_heave_finite_amplitude.py            # full campaign
    python benchmarks/floe_heave_finite_amplitude.py --fast      # reduced
    python benchmarks/floe_heave_finite_amplitude.py --out DIR

Writes ``DIR/heave_report.json`` (default ``artifacts/floe_heave/``).  No number
in ``docs/floe_heave_finite_amplitude.md`` is transcribed by hand.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:      # several worktrees may share one environment
    sys.path.insert(0, str(_SRC))

from tensordem.buoyancy_wet import (   # noqa: E402
    WetCouplingConfig,
    compute_buoyancy_force,
    heave_restoring_stiffness,
    solve_floating_equilibrium,
)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Case definition (SI units)
# ---------------------------------------------------------------------------
RADIUS = 0.5            # R, m
THICKNESS = 1.0         # t, m (out-of-plane extrusion)
RHO_ICE = 917.0         # kg/m^3
RHO_WATER = 1025.0      # kg/m^3
GRAVITY = 9.81          # m/s^2


def case_config() -> WetCouplingConfig:
    """Repository configuration: 2-D disk, sea water, dry damping (pure heave)."""
    return WetCouplingConfig(
        enabled=True,
        water_density=RHO_WATER,
        gravity=GRAVITY,
        water_level=0.0,
        wet_drag_coeff=0.0,   # undamped: the energy reference is conservative
        model="2d",
    )


def floe_mass(radius: float = RADIUS, thickness: float = THICKNESS) -> float:
    return RHO_ICE * math.pi * radius ** 2 * thickness


# ---------------------------------------------------------------------------
# Independent closed forms (no call into tensordem.buoyancy_wet)
# ---------------------------------------------------------------------------
def submerged_area(y: float, radius: float = RADIUS) -> float:
    """Submerged area of a disk whose centre sits at height ``y`` (0 = surface).

    Independent derivation used by ``docs/ice_floe_physics_validation.md``:
    substitute ``y' = R sin(phi)`` in ``A = int_{-R}^{y} 2 sqrt(R^2 - y'^2) dy'``.
    """
    if y <= -radius:            # fully submerged
        return math.pi * radius ** 2
    if y >= radius:             # fully dry (no buoyancy)
        return 0.0
    phi = math.asin(-y / radius)
    return radius ** 2 * (math.pi / 2.0 + phi + math.sin(phi) * math.cos(phi))


def waterplane_width(y: float, radius: float = RADIUS) -> float:
    """Chord length at the free surface: ``w(y) = 2 sqrt(R^2 - y^2)``."""
    return 2.0 * math.sqrt(max(radius ** 2 - y ** 2, 0.0))


def area_primitive(y: float, radius: float = RADIUS) -> float:
    """Exact ``int A_sub dy`` = ``y A_sub(y) - (2/3) (R^2 - y^2)^{3/2}``.

    Because ``dA_sub/dy = -w(y) = -2 sqrt(R^2 - y^2)``,
    ``d/dy [ y A_sub - (2/3)(R^2-y^2)^{3/2} ] = A_sub - y w + y w = A_sub``
    (the sign of the correction term is the one easy slip: ``+`` gives
    ``A_sub - 2 y w``).  ``tests/test_floe_heave_finite_amplitude.py`` checks the
    identity numerically, so the primitive is not an assumption.
    """
    full_area = math.pi * radius ** 2
    if y <= -radius:            # fully submerged: A_sub is constant -> linear continuation
        return -math.pi * radius ** 3 + full_area * (y + radius)
    if y >= radius:             # fully dry: A_sub = 0 -> flat continuation
        return math.pi * radius ** 3
    return y * submerged_area(y, radius) - (2.0 / 3.0) * (radius ** 2 - y ** 2) ** 1.5


def net_force_reference(y: float, radius: float = RADIUS, thickness: float = THICKNESS,
                        cfg: WetCouplingConfig | None = None) -> float:
    """Upward-positive net force on the disk at centre height ``y`` (reference)."""
    cfg = cfg or case_config()
    buoy = cfg.water_density * cfg.gravity * submerged_area(y, radius) * thickness
    return buoy - floe_mass(radius, thickness) * cfg.gravity


def equilibrium_reference(radius: float = RADIUS, thickness: float = THICKNESS,
                          cfg: WetCouplingConfig | None = None,
                          tol: float = 1e-15) -> float:
    """Bisection on the closed form for the floating centre height."""
    cfg = cfg or case_config()
    target_area = (RHO_ICE / cfg.water_density) * math.pi * radius ** 2
    lo, hi = -radius, radius
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if (submerged_area(mid, radius) - target_area) > 0.0:
            lo = mid          # too deep -> raise
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def potential(y: float, y_eq: float, radius: float = RADIUS, thickness: float = THICKNESS,
              cfg: WetCouplingConfig | None = None) -> float:
    """Conservative potential energy per unit mass-scale, referenced to ``V(y_eq) = 0``.

    ``V(y) = m g (y - y_eq) - rho_w g t [Pi(y) - Pi(y_eq)]`` with ``Pi`` the exact
    area primitive; ``-dV/dy`` reproduces the net force by construction.
    """
    cfg = cfg or case_config()
    m = floe_mass(radius, thickness)
    return (m * cfg.gravity * (y - y_eq)
            - cfg.water_density * cfg.gravity * thickness
            * (area_primitive(y, radius) - area_primitive(y_eq, radius)))


# ---------------------------------------------------------------------------
# Exact nonlinear period by energy quadrature
# ---------------------------------------------------------------------------
def _legendre(n: int, x: float):
    '''(P_n(x), P_n'(x)) by the three-term recurrence.'''
    p0, p1 = 1.0, x
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    if abs(x) < 1.0:
        dp1 = n * (x * p1 - p0) / (x * x - 1.0)
    else:
        dp1 = n * (n + 1) / 2.0
    return p1, dp1


def _gauss_legendre(order: int):
    '''Nodes and weights of the Gauss-Legendre rule on [-1, 1] (Newton iteration).'''
    nodes, weights = [], []
    for i in range(1, order + 1):
        x = math.cos(math.pi * (i - 0.25) / (order + 0.5))
        for _ in range(100):
            leg, dleg = _legendre(order, x)
            dx = -leg / dleg
            x += dx
            if abs(dx) < 1e-16:
                break
        leg, dleg = _legendre(order, x)
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dleg * dleg))
    return nodes, weights


def turning_points(y0: float, y_eq: float, radius: float = RADIUS,
                   thickness: float = THICKNESS, cfg: WetCouplingConfig | None = None,
                   ) -> tuple[float, float, float]:
    """Return (y_min, y_max, E) for a release from rest at ``y0``."""
    cfg = cfg or case_config()
    energy = potential(y0, y_eq, radius, thickness, cfg)
    # V is monotone on each side of the minimum -> bracket and bisect.
    if y0 >= y_eq:                      # released above equilibrium: find y_min < y_eq
        lo, hi = y_eq - 1e-3 * radius, y_eq
        for _ in range(200):
            if potential(lo, y_eq, radius, thickness, cfg) > energy:
                break
            lo -= (hi - lo) if (hi - lo) > 0 else 1e-9 * radius
            lo = max(lo, y_eq - 3.0 * radius)
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if potential(mid, y_eq, radius, thickness, cfg) > energy:
                lo = mid
            else:
                hi = mid
        y_min = 0.5 * (lo + hi)
        y_max = y0
    else:                               # released below equilibrium: find y_max > y_eq
        lo, hi = y_eq, y_eq + 1e-3 * radius
        for _ in range(200):
            if potential(hi, y_eq, radius, thickness, cfg) > energy:
                break
            lo, hi = hi, hi + (hi - y_eq)
            hi = min(hi, y_eq + 3.0 * radius)
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if potential(mid, y_eq, radius, thickness, cfg) > energy:
                hi = mid
            else:
                lo = mid
        y_min = y0
        y_max = 0.5 * (lo + hi)
    return y_min, y_max, energy


def period_and_mean_quadrature(y0: float, y_eq: float, radius: float = RADIUS,
                               thickness: float = THICKNESS, cfg=None,
                               order: int = 64) -> dict:
    '''Exact conservative period and time-averaged draft by energy quadrature.

    ``T = 2 int dy / sqrt(2 (E - V)/m)`` is evaluated with ``y = c + d cos(theta)``,
    which makes the integrand analytic at a turning point (the ``1/sqrt`` and
    ``sin`` factors cancel there).  The interval is **split at** ``y = +/-R``:
    there the disk becomes fully submerged / fully dry, where ``d3V/dy3`` is
    singular, so an unsplit rule degrades to first order exactly in the amplitude
    range where the nonlinearity is interesting (observed: a 1e-4 plateau at
    ``A/R = 0.3`` that is pure quadrature error, not a model error).  On each
    piece a Gauss-Legendre rule applies, whose nodes never touch the singular
    endpoints.
    '''
    cfg = cfg or case_config()
    m = floe_mass(radius, thickness)
    y_min, y_max, energy = turning_points(y0, y_eq, radius, thickness, cfg)

    breaks = {y_min, y_max}
    for x in (-radius, radius):
        if y_min < x < y_max:
            breaks.add(x)
    bounds = sorted(breaks)

    nodes, weights = _gauss_legendre(order)
    theta = [0.5 * math.pi * (x + 1.0) for x in nodes]
    wtheta = [0.5 * math.pi * w for w in weights]

    def inverse_velocity(y: float) -> float:
        e_kin = energy - potential(y, y_eq, radius, thickness, cfg)
        if e_kin <= 0.0:
            force = abs(net_force_reference(y, radius, thickness, cfg))
            return math.sqrt(m / force) if force > 0.0 else 0.0
        return math.sqrt(m / (2.0 * e_kin))

    half_period = 0.0
    first_moment = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        cm, dm = 0.5 * (a + b), 0.5 * (b - a)
        for t, w in zip(theta, wtheta):
            y = cm + dm * math.cos(t)
            g = w * dm * math.sin(t) * inverse_velocity(y)
            half_period += g
            first_moment += g * y

    period = 2.0 * half_period
    mean_y = first_moment / half_period
    return {
        "period": period,
        "mean_y": mean_y,
        "mean_offset": mean_y - y_eq,
        "y_min": y_min,
        "y_max": y_max,
        "energy": energy,
        "pieces": bounds,
        "order": order,
    }


def linear_period(k_heave: float, radius: float = RADIUS, thickness: float = THICKNESS) -> float:
    """Small-amplitude limit ``T0 = 2 pi sqrt(m / k_heave)``; NaN if ``k_heave = 0``."""
    if k_heave <= 0.0:      # fully submerged / fully dry: no waterplane restoring
        return float("nan")
    return 2.0 * math.pi * math.sqrt(floe_mass(radius, thickness) / k_heave)


# ---------------------------------------------------------------------------
# Repository-law simulation (symplectic Euler, the repository's update order)
# ---------------------------------------------------------------------------
def repo_net_force(y: torch.Tensor, cfg: WetCouplingConfig) -> torch.Tensor:
    """Net vertical force from the *repository* buoyancy law minus the weight.

    ``y`` is a batch of centre heights, so many independent lanes are integrated
    in one call (the force law is unchanged; this is purely a batching of the
    same per-particle evaluation).
    """
    positions = torch.zeros((y.numel(), 2), dtype=y.dtype)
    positions[:, 1] = y
    buoyancy = compute_buoyancy_force(positions, RADIUS, THICKNESS, RHO_ICE, cfg)[:, 1]
    return buoyancy - floe_mass() * cfg.gravity


def _turning_times(times, v):
    """Times at which the velocity of one lane changes sign (linear interpolation)."""
    turns = []
    for i in range(1, len(v)):
        if v[i - 1] * v[i] < 0.0:
            frac = v[i - 1] / (v[i - 1] - v[i])
            turns.append(times[i - 1] + frac * (times[i] - times[i - 1]))
    return turns


def _lane_metrics(times, ys, vs):
    turns = _turning_times(times, vs)
    periods = [turns[i + 2] - turns[i] for i in range(len(turns) - 2)]
    t_start = turns[0] if turns else times[0]
    t_end = turns[-1] if turns else times[-1]
    idx = [i for i, t in enumerate(times) if t_start <= t <= t_end]
    mean_y = float("nan")
    if len(idx) > 2:
        area = 0.0
        for j in range(1, len(idx)):
            area += 0.5 * (ys[idx[j]] + ys[idx[j - 1]]) * (times[idx[j]] - times[idx[j - 1]])
        mean_y = area / (times[idx[-1]] - times[idx[0]])
    return {
        "period": sum(periods) / len(periods) if periods else float("nan"),
        "period_spread": (max(periods) - min(periods)) if periods else float("nan"),
        "mean_y": mean_y,
        "y_max": float(max(ys)),
        "y_min": float(min(ys)),
        "n_turnings": len(turns),
        "turn_times": turns,
    }


def simulate_heave_batch(y0s, dt: float, n_periods: float = 4.0,
                         cfg: WetCouplingConfig | None = None, force_fn=None,
                         keep_history: bool = False) -> list[dict]:
    """Integrate the repository force law with the repository's Euler ordering.

    ``v += a dt ; x += v dt`` — the same sequence used by the 2-D DEM integrator
    and by ``benchmarks/ice_floe_physics.py`` anchor H.  All lanes start from
    rest at their own release height and share one time step.
    """
    cfg = cfg or case_config()
    force_fn = force_fn or (lambda y: repo_net_force(y, cfg))
    m = floe_mass()
    if not isinstance(y0s, (list, tuple)):
        y0s = [y0s]

    t_ref = linear_period(heave_restoring_stiffness(RADIUS, THICKNESS, RHO_ICE, cfg)[0])
    n_steps = int(math.ceil(n_periods * t_ref / dt))
    y = torch.tensor([float(v) for v in y0s], dtype=torch.float64)
    v = torch.zeros_like(y)

    times = [0.0]
    hist_y = [y.clone()]
    hist_v = [v.clone()]
    for step in range(1, n_steps + 1):
        a = force_fn(y) / m
        v = v + a * dt
        y = y + v * dt
        times.append(step * dt)
        hist_y.append(y.clone())
        hist_v.append(v.clone())

    Y = torch.stack(hist_y).numpy()
    V = torch.stack(hist_v).numpy()
    out = []
    for lane in range(Y.shape[1]):
        metrics = _lane_metrics(times, list(Y[:, lane]), list(V[:, lane]))
        metrics["dt"] = dt
        metrics["n_steps"] = n_steps
        if keep_history:
            metrics["times"] = times
            metrics["ys"] = [float(v) for v in Y[:, lane]]
        out.append(metrics)
    return out


def simulate_heave(y0: float, dt: float, n_periods: float = 4.0, **kwargs) -> dict:
    return simulate_heave_batch([y0], dt, n_periods=n_periods, **kwargs)[0]


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------
@dataclass
class Row:
    amplitude_ratio: float
    amplitude: float
    release_y0: float
    period_quadrature: float
    period_sim_dt: float
    period_sim_refined: float
    rel_err_dt: float
    rel_err_refined: float
    period_linear: float
    nonlinear_shift_refined: float
    mean_quadrature: float
    mean_sim_refined: float
    mean_err_refined: float
    ymin_quadrature: float
    ymax_quadrature: float
    ymin_sim: float
    ymax_sim: float


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("nan")


def run_campaign(fast: bool = False, amplitude_ratios=None, dt: float = 2e-3) -> dict:
    cfg = case_config()
    y_eq_module, f_eq_module, converged = solve_floating_equilibrium(RADIUS, THICKNESS, RHO_ICE, cfg)
    y_eq_ref = equilibrium_reference(RADIUS, THICKNESS, cfg)
    k_module, _stable = heave_restoring_stiffness(RADIUS, THICKNESS, RHO_ICE, cfg)
    k_analytic = RHO_WATER * GRAVITY * THICKNESS * waterplane_width(y_eq_ref, RADIUS)
    t_lin = linear_period(k_module)

    ratios = amplitude_ratios or ([0.002, 0.02, 0.05, 0.10, 0.20, 0.30] if not fast
                                  else [0.002, 0.05, 0.30])
    lanes = []          # (signed ratio, release height)
    for sign in (1.0, -1.0):
        for ratio in ratios:
            y0 = y_eq_module + sign * ratio * RADIUS
            if -RADIUS < y0 < RADIUS:
                lanes.append((sign * ratio, y0))

    y0s = [y0 for _, y0 in lanes]
    sims = simulate_heave_batch(y0s, dt, n_periods=4.0, cfg=cfg)
    sims_fine = simulate_heave_batch(y0s, dt / 4.0, n_periods=4.0, cfg=cfg, keep_history=True)

    rows = []
    series = {}
    for (ratio, y0), sim, fine in zip(lanes, sims, sims_fine):
        quad = period_and_mean_quadrature(y0, y_eq_module)
        rows.append(Row(
            amplitude_ratio=ratio,
            amplitude=y0 - y_eq_module,
            release_y0=y0,
            period_quadrature=quad["period"],
            period_sim_dt=sim["period"],
            period_sim_refined=fine["period"],
            rel_err_dt=_rel(sim["period"], quad["period"]),
            rel_err_refined=_rel(fine["period"], quad["period"]),
            period_linear=t_lin,
            nonlinear_shift_refined=(fine["period"] - t_lin) / t_lin,
            mean_quadrature=quad["mean_y"],
            mean_sim_refined=fine["mean_y"],
            mean_err_refined=abs(fine["mean_y"] - quad["mean_y"]),
            ymin_quadrature=quad["y_min"],
            ymax_quadrature=quad["y_max"],
            ymin_sim=fine["y_min"],
            ymax_sim=fine["y_max"],
        ))
        if abs(abs(ratio) - 0.05) < 1e-12 or abs(abs(ratio) - 0.30) < 1e-12:
            series[f"{'+' if ratio > 0 else '-'}{abs(ratio):g}"] = {
                "t": fine["times"][::10],
                "y": fine["ys"][::10],
                "mean_quadrature": quad["mean_y"],
                "period_quadrature": quad["period"],
                "y_eq": y_eq_module,
            }

    # Restoring-force curve on a grid (code vs closed form) -- the source of the asymmetry.
    grid = torch.tensor([-0.95 * RADIUS + 1.9 * RADIUS * i / 40.0 for i in range(41)],
                        dtype=torch.float64)
    f_code = repo_net_force(grid, cfg)
    force_curve = [{
        "y": float(grid[i].item()),
        "F_code": float(f_code[i].item()),
        "F_reference": net_force_reference(float(grid[i].item())),
        "k_local": RHO_WATER * GRAVITY * THICKNESS * waterplane_width(float(grid[i].item()), RADIUS),
    } for i in range(grid.numel())]

    return {
        "schema": "tensordem.floe-heave-finite-amplitude/1",
        "task": "m2843",
        "closes": "docs/ice_floe_physics_validation.md section 6, item 'Small-amplitude only'",
        "case": {
            "radius_m": RADIUS,
            "thickness_m": THICKNESS,
            "rho_ice": RHO_ICE,
            "rho_water": RHO_WATER,
            "gravity": GRAVITY,
            "mass_kg": floe_mass(),
            "submerged_fraction": RHO_ICE / RHO_WATER,
            "model": "2d",
            "wet_drag_coeff": 0.0,
        },
        "equilibrium": {
            "y_eq_module": y_eq_module,
            "y_eq_reference": y_eq_ref,
            "abs_diff": abs(y_eq_module - y_eq_ref),
            "module_converged": converged,
            "f_sub_module": f_eq_module,
        },
        "stiffness": {
            "k_module": k_module,
            "k_analytic": k_analytic,
            "rel_err": _rel(k_module, k_analytic),
            "period_linear": t_lin,
        },
        "dt": dt,
        "rows": [asdict(r) for r in rows],
        "force_curve": force_curve,
        "series": series,
        "quadrature": {"order": 64, "rule": "per-segment Gauss-Legendre on the cosine substitution"},
    }


def _negative_controls(fast: bool = False) -> dict:
    """Controls: the anchor must fail when the physics under test is removed."""
    cfg = case_config()
    y_eq, _, _ = solve_floating_equilibrium(RADIUS, THICKNESS, RHO_ICE, cfg)
    y0 = y_eq + 0.05 * RADIUS
    quad = period_and_mean_quadrature(y0, y_eq)
    dt = 4e-3 if fast else 2e-3
    out = {}

    # C1: buoyancy -> 0 (free fall): no oscillation at all.
    zero = lambda y: torch.zeros_like(y)
    try:
        sim = simulate_heave(y0, dt, n_periods=2.0, force_fn=zero)
        out["C1_zero_buoyancy"] = {
            "period": sim["period"],
            "verdict": "no oscillation (expected)" if sim["n_turnings"] == 0 else "UNEXPECTED rotation",
            "failed_as_expected": sim["n_turnings"] == 0,
        }
    except Exception as exc:                                  # pragma: no cover - defensive
        out["C1_zero_buoyancy"] = {"exception": repr(exc), "failed_as_expected": True}

    # C2: linearised (constant-stiffness) law: removes the amplitude dependence.
    # Judged at A/R = 0.30, where the nonlinear shift is 6.3 %; at A/R = 0.05 the
    # shift is only 0.096 % and a linearised law is *not* discriminating, so that
    # amplitude would produce a control that cannot fail (see the report).
    k_lin = RHO_WATER * GRAVITY * THICKNESS * waterplane_width(y_eq, RADIUS)
    lin = lambda y: -k_lin * (y - y_eq)
    y0_big = y_eq + 0.30 * RADIUS
    quad_big = period_and_mean_quadrature(y0_big, y_eq)
    sim_lin = simulate_heave(y0_big, dt, n_periods=4.0, force_fn=lin)
    out["C2_linearised_law"] = {
        "amplitude_ratio": 0.30,
        "period": sim_lin["period"],
        "quadrature_period": quad_big["period"],
        "rel_err": _rel(sim_lin["period"], quad_big["period"]),
        "failed_as_expected": _rel(sim_lin["period"], quad_big["period"]) > 5e-2,
    }

    # C3: 1.3x buoyancy -> stiffer heave than the quadrature predicts.
    # (0.5x is degenerate: rho_i/rho_w > 1 and the disk has no waterplane at all.)
    stiff_cfg = WetCouplingConfig(enabled=True, water_density=1.3 * RHO_WATER,
                                  gravity=GRAVITY, water_level=0.0,
                                  wet_drag_coeff=0.0, model="2d")
    sim_stiff = simulate_heave(y0, dt, n_periods=4.0, cfg=stiff_cfg)
    out["C3_over_buoyancy"] = {
        "period": sim_stiff["period"],
        "quadrature_period": quad["period"],
        "rel_err": _rel(sim_stiff["period"], quad["period"]),
        "failed_as_expected": _rel(sim_stiff["period"], quad["period"]) > 0.05,
    }
    return out


def _json_default(obj):
    """Serialise torch/numpy scalars that leak into the metrics dicts."""
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced amplitude set")
    parser.add_argument("--out", default="artifacts/floe_heave", help="output directory")
    parser.add_argument("--dt", type=float, default=2e-3, help="base time step, s")
    args = parser.parse_args()

    report = run_campaign(fast=args.fast, dt=args.dt)
    report["negative_controls"] = _negative_controls(fast=args.fast)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "heave_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default))

    print(f"y_eq module {report['equilibrium']['y_eq_module']:.6f} m  "
          f"reference {report['equilibrium']['y_eq_reference']:.6f} m")
    print(f"k_heave module {report['stiffness']['k_module']:.3f} N/m  "
          f"analytic {report['stiffness']['k_analytic']:.3f} N/m  "
          f"rel {report['stiffness']['rel_err']:.2e}")
    print(f"T0 = {report['stiffness']['period_linear']:.6f} s")
    print(f"{'A/R':>7} {'T_quad':>10} {'T_sim(dt/4)':>12} {'rel':>10} {'shift vs T0':>12} "
          f"{'mean-draft err':>14}")
    for r in report["rows"]:
        print(f"{r['amplitude_ratio']:7.3f} {r['period_quadrature']:10.6f} "
              f"{r['period_sim_refined']:12.6f} {r['rel_err_refined']:10.2e} "
              f"{r['nonlinear_shift_refined']:12.4%} {r['mean_err_refined']:14.2e}")
    worst = max(r["rel_err_refined"] for r in report["rows"])
    print(f"worst period error vs quadrature: {worst:.3e}")
    print(f"report written to {out_dir / 'heave_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())