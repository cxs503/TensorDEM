"""Beverloo silo-discharge benchmark for TensorDEM granular flow.

Physical background
-------------------
When a granular material drains from a silo through a bottom orifice, the mass
flow rate ``W`` follows the Beverloo correlation (Beverloo et al., 1961):

    W = C · ρ_b · √g · (D - k·d)^(5/2)

with ``D`` the orifice diameter, ``d`` the particle diameter and ``ρ_b`` the
bulk density; literature values for glass beads are ``C ≈ 0.58`` and
``k ≈ 1.4 … 1.6``.

Implementation contract
-----------------------
This module owns the *case definition* and the analytic correlation; the DEM
integration is done by the verified flat-bottom-orifice solver
``tensordem.silo_discharge`` (real contact dynamics, particles removed once
they pass below the plate, per-window mass bookkeeping).  ``simulate_discharge``
therefore returns a **measured** flow rate obtained from a particle count
balance, never the correlation value re-labelled as a measurement:

* ``flow_rate_kg_s`` = mean over ``n_windows`` windows of
  ``Δm_discharged / window`` (discrete-element mass balance);
* ``flow_rate_windows_kg_s`` / ``flow_rate_std_kg_s`` = the per-window series
  and its ± 1σ spread;
* ``r2`` = coefficient of determination of cumulative discharged mass against
  time (a *steadiness* check of the flow, not an agreement check);
* ``rel_err_vs_beverloo`` / ``recovered_C`` = the agreement with the
  correlation, reported separately so the measurement and the prediction can
  differ.

The regression that motivated this contract (a former implementation returned
the analytic ``W`` as ``flow_rate_kg_s`` with ``n_discharged = 0`` and
``r2 = 1.0``) is pinned by ``tests/test_beverloo_discharge_measurement.py``.
See ``docs/beverloo_discharge_measurement_audit.md``.

All quantities are SI (m, kg, s, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

DTYPE_NAME = "float64"


class BeverlooError(ValueError):
    pass


# Empirical stability guard for the underlying solver: the contact kernel
# integrates a spring-dashpot law, and a *generated* packing with deep initial
# interpenetration (a jittered HCP lattice can be born overlapped) diverges to
# non-finite positions instead of relaxing.  Measured on this bench
# (``docs/beverloo_discharge_measurement_audit.md`` §6): configurations with a
# worst-case initial interpenetration above ~0.25 r blew up, below ~0.15 r they
# ran; 0.25 r is therefore used as a refuse threshold, not a physical limit.
MAX_INITIAL_OVERLAP_OVER_RADIUS = 0.25


@dataclass(frozen=True)
class BeverlooCase:
    """Flat-bottom circular-orifice silo case (SI)."""

    # material
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    # geometry
    silo_radius: float = 0.04
    fill_height: float = 0.12
    orifice_diameter: float = 0.02
    gravity: float = 9.81
    packing_gap: float = 0.02
    jitter: float = 0.01
    # numerics
    dt: Optional[float] = None
    seed: int = 5278
    max_time: float = 2.0            # total simulated time = settle + discharge
    discharge_check_interval: int = 100  # deprecated, kept for API stability
    # discharge protocol (forwarded to the silo solver)
    settle_time: float = 0.15        # pre-discharge relaxation
    window: float = 0.2              # flow-rate window length
    n_windows: int = 5               # >= 5 windows for a steady mean
    plate_thickness: Optional[float] = None   # default: 2 r
    wall_mu: float = 0.0             # side-wall friction (0 = smooth)

    @property
    def particle_mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.particle_mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def discharge_time(self) -> float:
        return self.max_time - self.settle_time

    @property
    def resolved_plate_thickness(self) -> float:
        return 2.0 * self.radius if self.plate_thickness is None else self.plate_thickness

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        period = 2.0 * math.pi * math.sqrt((0.5 * self.particle_mass) / self.kn)
        return period / 20.0

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "kt_ratio",
                     "silo_radius", "fill_height", "orifice_diameter", "max_time",
                     "settle_time", "window"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0.0):
                raise BeverlooError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "restitution", "packing_gap", "jitter",
                     "wall_mu"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0.0):
                raise BeverlooError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.gravity) or self.gravity < 0.0:
            raise BeverlooError("gravity must be finite and nonnegative")
        if not 0.0 <= self.restitution < 1.0:
            raise BeverlooError("restitution must lie in [0, 1)")
        if self.orifice_diameter >= 2.0 * self.silo_radius:
            raise BeverlooError("orifice_diameter must be less than silo diameter")
        if self.orifice_diameter < 3.0 * self.radius:
            raise BeverlooError("orifice must be at least 3 particle diameters")
        if isinstance(self.n_windows, bool) or not isinstance(self.n_windows, int):
            raise BeverlooError("n_windows must be an integer")
        if self.n_windows < 5:
            raise BeverlooError("need at least 5 flow-rate windows")
        if self.plate_thickness is not None:
            t = self.plate_thickness
            if not (isinstance(t, (int, float)) and math.isfinite(t) and t >= 0.0):
                raise BeverlooError("plate_thickness must be finite and nonnegative")
        if self.settle_time + self.window * self.n_windows <= 0.0:
            raise BeverlooError("settle_time + n_windows*window must be positive")

    def replace(self, **kwargs) -> "BeverlooCase":
        """Create a new case with modified parameters."""
        return BeverlooCase(**{f.name: getattr(self, f.name) for f in fields(self) if f.name not in kwargs}, **kwargs)


def to_silo_case(case: BeverlooCase):
    """Map a :class:`BeverlooCase` onto the verified flat-orifice silo case."""
    from .silo_discharge import SiloCase

    needed = case.settle_time + case.window * case.n_windows
    if case.max_time < needed:
        raise BeverlooError(
            f"max_time={case.max_time} cannot hold settle_time={case.settle_time} "
            f"plus {case.n_windows} windows of {case.window} s "
            f"(needs >= {needed:.6g} s): raise max_time or shrink the windows")
    return SiloCase(
        radius=case.radius,
        density=case.density,
        mu=case.mu,
        mu_rolling=case.mu_rolling,
        restitution=case.restitution,
        kn=case.kn,
        kt_ratio=case.kt_ratio,
        gravity=case.gravity,
        silo_radius=case.silo_radius,
        fill_height=case.fill_height,
        orifice_diameter=case.orifice_diameter,
        plate_thickness=case.resolved_plate_thickness,
        wall_mu=case.wall_mu,
        packing_gap=case.packing_gap,
        jitter=case.jitter,
        seed=case.seed,
        dt=case.dt,
        settle_time=case.settle_time,
        discharge_time=case.discharge_time,
        window=case.window,
        n_windows=case.n_windows,
    )


def initial_interpenetration(case: BeverlooCase) -> float:
    """Worst-case initial interpenetration of the generated packing, in units of r.

    The packing is the solver's own deterministic HCP column
    (``silo_discharge._initial_packing``); a ``jitter`` larger than the lattice
    gap can make particles *born* overlapped.
    """
    from .granular3d_core import build_pairs
    from .silo_discharge import _initial_packing

    pos = _initial_packing(to_silo_case(case))
    if len(pos) < 2:
        return 0.0
    iu, ju = build_pairs(pos, case.radius)
    if len(iu) == 0:
        return 0.0
    sep = ((pos[iu] - pos[ju]) ** 2).sum(axis=1) ** 0.5
    overlap = 2.0 * case.radius - sep
    worst = float(overlap.max()) if len(overlap) else 0.0
    return max(worst, 0.0) / case.radius


def _steadiness_r2(rates: list[float], window: float, settle_time: float) -> float:
    """R² of cumulative discharged mass vs time over the measured windows.

    A steady discharge gives constant window rates, so the cumulative mass is a
    straight line: this statistic measures *flow steadiness*, not agreement with
    the Beverloo correlation.
    """
    n = len(rates)
    if n < 2:
        return 0.0
    times, cum, acc = [], [], 0.0
    for i, rate in enumerate(rates, start=1):
        acc += rate * window
        times.append(settle_time + i * window)
        cum.append(acc)
    t_mean = sum(times) / n
    c_mean = sum(cum) / n
    sxx = sum((t - t_mean) ** 2 for t in times)
    if sxx <= 0.0:
        return 0.0
    sxy = sum((t - t_mean) * (c - c_mean) for t, c in zip(times, cum))
    slope = sxy / sxx
    ss_tot = sum((c - c_mean) ** 2 for c in cum)
    ss_res = sum((c - (c_mean + slope * (t - t_mean))) ** 2
                 for t, c in zip(times, cum))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def simulate_discharge(case: BeverlooCase, *, return_history: bool = False,
                       C: float = 0.58, k: float = 1.5) -> dict:
    """Run a real 3-D DEM orifice discharge and return the *measured* flow rate.

    Parameters
    ----------
    case:
        Case definition; ``max_time`` is the total simulated time (settle +
        discharge) and must cover ``settle_time + n_windows * window``.
    return_history:
        Also return the final positions / velocities / alive mask.
    C, k:
        Beverloo constants used only for the *reference* branch of the result
        (``beverloo_prediction``, ``rel_err_vs_beverloo``, ``recovered_C``).

    Returns
    -------
    dict
        ``flow_rate_kg_s`` is the measured mass-balance rate (kg/s) over
        ``n_windows`` windows, ``n_discharged`` the number of particles that
        left the silo, ``rel_err_vs_beverloo`` the signed relative difference to
        the correlation.  ``outside_window_discharged_mass_kg`` is the mass that
        left the silo outside the measurement windows (normally the settling
        phase); it is *not* folded into the rate, so the reported rate is a
        window average and not an average over the whole run.
    """
    from .silo_discharge import simulate_silo

    silo_case = to_silo_case(case)
    worst_overlap = initial_interpenetration(case)
    if worst_overlap > MAX_INITIAL_OVERLAP_OVER_RADIUS:
        raise BeverlooError(
            "initial packing interpenetrates by "
            f"{worst_overlap:.3f} r (> {MAX_INITIAL_OVERLAP_OVER_RADIUS} r): "
            "jitter exceeds the lattice gap (reduce jitter or raise "
            "packing_gap); the solver diverges on such a start state")
    sim = simulate_silo(silo_case, return_history=return_history)

    mass = case.particle_mass
    windows = [float(x) for x in sim["flow_rate_windows_kg_s"]]
    W = float(sim["flow_rate_kg_s"])
    W_std = float(sim["flow_rate_std_kg_s"])
    n_discharged = int(sim["n_discharged"])
    discharged_mass = float(sim["discharged_mass_kg"])
    rho_b = float(sim["bulk_density_kg_m3"])

    pred = beverloo_prediction(case, C=C, k=k)
    W_lit = float(pred["W_kg_s"])
    if W_lit > 0.0 and math.isfinite(W_lit):
        rel_err = (W - W_lit) / W_lit
    else:
        rel_err = float("nan")

    out = {
        "n_initial": int(sim["n_initial"]),
        "n_remaining": int(sim["n_initial"]) - n_discharged,
        "n_discharged": n_discharged,
        "total_discharged_mass_kg": discharged_mass,
        "mass_balance_residual_kg": discharged_mass - n_discharged * mass,
        "window_span_discharged_mass_kg": sum(windows) * case.window,
        "outside_window_discharged_mass_kg":
            discharged_mass - sum(windows) * case.window,
        "flow_rate_kg_s": W,
        "flow_rate_std_kg_s": W_std,
        "flow_rate_windows_kg_s": windows,
        "flow_rate_source": "dem_mass_balance",
        "t_settle_s": case.settle_time,
        "t_end_s": case.max_time,
        "r2": _steadiness_r2(windows, case.window, case.settle_time),
        "r2_basis": "steady_flow_cumulative_mass_vs_time",
        "rel_err_vs_beverloo": rel_err,
        "recovered_C": recovered_C(case, W, k=k, rho_b=rho_b),
        "recovered_C_at_reference_rho": recovered_C(case, W, k=k),
        "initial_max_overlap_over_radius": worst_overlap,
        "dt": float(sim["dt"]),
        "steps": int(sim["steps"]),
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
        "gravity": case.gravity,
        "orifice_diameter": case.orifice_diameter,
        "silo_radius": case.silo_radius,
        "radius": case.radius,
        "density": case.density,
        "bulk_density_kg_m3": rho_b,
        "packing_fraction": float(sim["packing_fraction"]),
        "D_over_d": float(sim["D_over_d"]),
        "plate_thickness": case.resolved_plate_thickness,
        "window": case.window,
        "n_windows": case.n_windows,
        "beverloo_prediction": pred,
    }
    if return_history:
        out["positions"] = sim["positions"]
        out["velocities"] = sim["velocities"]
        out["alive"] = sim["alive"]
    return out


def beverloo_prediction(case: BeverlooCase, *, C: float = 0.58, k: float = 1.5) -> dict:
    """Analytic Beverloo correlation for the case (a *prediction*, not a measurement)."""
    d = 2.0 * case.radius
    D = case.orifice_diameter
    g = case.gravity
    rho_p = case.density
    phi = 0.64
    rho_b = phi * rho_p
    D_eff = D - k * d
    if D_eff <= 0 or g <= 0:
        return {"W_kg_s": 0.0, "D_eff": D_eff, "C": C, "k": k, "rho_b": rho_b,
                "valid": g > 0 and D_eff > 0}
    W = C * rho_b * math.sqrt(g) * D_eff ** 2.5
    return {"W_kg_s": W, "D_eff": D_eff, "C": C, "k": k, "rho_b": rho_b, "valid": True}


def recovered_C(case: BeverlooCase, W_measured: float, *, k: float = 1.5,
                rho_b: Optional[float] = None) -> float:
    """Invert the correlation with a *measured* rate to recover the constant C.

    ``rho_b`` defaults to the literature packing reference ``0.64 ρ_p``; pass
    the bulk density *measured* on the packing to factor the packing fraction
    out of the comparison.  Returns ``nan`` when ``D - k d <= 0``.
    """
    d_eff = case.orifice_diameter - k * 2.0 * case.radius
    if d_eff <= 0.0 or case.gravity <= 0.0:
        return float("nan")
    if rho_b is None:
        rho_b = 0.64 * case.density
    if rho_b <= 0.0:
        return float("nan")
    return W_measured / (rho_b * math.sqrt(case.gravity) * d_eff ** 2.5)


def beverloo_2d_prediction(width: float, d: float, rho_b: float, g: float,
                           *, C_2d: float = 0.65, k: float = 1.5) -> dict:
    W_eff = width - k * d
    if W_eff <= 0:
        return {"W_kg_ms": 0.0, "W_eff": W_eff, "valid": False}
    W_2d = C_2d * rho_b * math.sqrt(g) * W_eff ** 1.5
    return {"W_kg_ms": W_2d, "W_eff": W_eff, "C_2d": C_2d, "k": k, "valid": True}


__all__ = [
    "BeverlooError", "BeverlooCase", "simulate_discharge", "to_silo_case",
    "beverloo_prediction", "beverloo_2d_prediction", "recovered_C",
]