"""Analytic verification of the 2-D buoyancy / wet-drag force law.

The module :mod:`tensordem.buoyancy_wet` provides a buoyancy force derived from
circular-segment geometry plus a linear (Stokes-like) wet drag weighted by the
submerged fraction.  Its existing unit tests check force values at hand-picked
points.  This module supplies the *independent analytic references* needed to
verify the law as a dynamical model:

1. **Geometry** — the closed-form submerged area fraction is compared with an
   independent composite-Simpson quadrature of the chord width.  This is a
   code-independent re-derivation, so a transcription error in the circular
   segment formula cannot survive the check.
2. **Restoring stiffness** — ``dF_b/dh`` measured by central differences of the
   *actual* force routine is compared with the analytic chord-width identity
   ``dF_b/dh = -rho_w * g * thickness * 2*sqrt(R**2 - h**2)``.  This verifies
   that buoyancy is a consistent conservative restoring force (sign and
   magnitude of its gradient), which the force-value tests cannot see.
3. **Archimedes equilibrium** — a free-floating disk must settle with submerged
   area fraction ``f_sub = rho_ice / rho_water``.
4. **Free-float dynamics** — a released disk performs damped heave about that
   draft.  Linearising about the draft gives a harmonic oscillator with
   ``k = rho_w * g * thickness * W(h_eq)``, ``c = C_wet * f_eq``, hence
   ``omega_n = sqrt(k/m)``, ``zeta = c / (2 m omega_n)``,
   ``omega_d = omega_n * sqrt(1 - zeta**2)``; the analytic trajectory and its
   exponential envelope are compared with the integrated motion.
5. **Pure drag decay** — with ``rho_ice == rho_water`` and the particle fully
   submerged the net quasi-static force vanishes, so the velocity must follow
   the exact exponential ``v(t) = v0 * exp(-C_wet * t / m)``.
6. **Timestep convergence** — the measurements above are shown to be
   timestep-converged (RK4 refinement study), so a residual gap against the
   analytic reference is attributable to the model's nonlinearity rather than
   to the integrator.
7. **Linearisation limit** — the gap between simulated and linearised response
   must shrink with the excitation amplitude, which demonstrates that the
   finite residual is the expected O(amplitude) nonlinearity of the exact
   circular-segment law rather than an implementation defect.

Everything is 2-D (per-unit-thickness) and single particle: the campaign is a
verification of the force law, not a physical qualification of ice heave.  In
particular there is **no hydrodynamic added mass**, so the natural frequency
reported here is a property of the code model and must not be compared with a
measured ice-floe heave period.

All quantities are SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .buoyancy_wet import (
    WetCouplingConfig,
    compute_buoyancy_force,
    compute_wet_drag_force,
)

PROTOCOL = "tensordem-buoyancy-free-float-verification-v1"


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------

def chord_width(h: float, radius: float) -> float:
    """Width of the water-plane chord of a disk whose centre is at height *h*.

    For ``|h| < R`` the chord at the free surface has length ``2*sqrt(R**2-h**2)``;
    outside that range the disk does not pierce the surface and the width is 0.
    """
    if abs(h) >= radius:
        return 0.0
    return 2.0 * math.sqrt(max(0.0, radius * radius - h * h))


def submerged_fraction_quadrature(
    h: float, radius: float, panels: int = 2000
) -> float:
    """Submerged *area* fraction from composite Simpson quadrature.

    The disk is centred at ``y = h`` and the water surface is ``y = 0``.  In the
    absolute height variable the chord width ``2*sqrt(R**2 - (y - h)**2)`` has
    square-root endpoints, and Simpson's rule then converges only as
    ``O(n**-1.5)``: at 64 panels the bias is already ``3e-4`` in ``f``, which is
    large enough to shift a bisected equilibrium draft by ``6e-4 m`` and to be
    mistaken for a solver defect.  The substitution ``y = h + R sin(phi)``
    removes the singularity: ``dA = 2 R**2 cos**2(phi) dphi`` with
    ``phi in [-pi/2, asin(-h/R)]``, a smooth integrand that Simpson integrates
    to ``O(n**-4)``.  This is still an independent numerical route: the
    closed-form segment formula is never evaluated here.
    """
    if h >= radius:
        return 0.0
    if h <= -radius:
        return 1.0
    phi_max = math.asin(max(-1.0, min(1.0, -h / radius)))
    if phi_max <= -0.5 * math.pi:
        return 0.0
    if panels % 2:
        panels += 1
    lo = -0.5 * math.pi
    step = (phi_max - lo) / panels
    integrand = lambda phi: 2.0 * radius * radius * math.cos(phi) ** 2  # noqa: E731
    total = integrand(lo) + integrand(phi_max)
    for k in range(1, panels):
        total += (4.0 if k % 2 else 2.0) * integrand(lo + k * step)
    area = total * step / 3.0
    return area / (math.pi * radius * radius)


def buoyancy_stiffness_analytic(
    h: float, radius: float, rho_water: float, gravity: float, thickness: float
) -> float:
    """Analytic ``dF_b/dh`` = ``-rho_w * g * thickness * W(h)`` (per unit depth m)."""
    return -rho_water * gravity * thickness * chord_width(h, radius)


def solve_equilibrium_height(
    radius: float, rho_ice: float, rho_water: float, tol: float = 1e-13
) -> float:
    """Centre height at which ``f_sub = rho_ice / rho_water`` (bisection).

    ``f_sub`` decreases monotonically with ``h`` on ``[-R, R]``, so the root is
    unique.  This reference is independent of the force routine: it inverts the
    geometry, and only later is the force routine required to agree with it.
    """
    target = rho_ice / rho_water
    if target <= 0.0:
        return radius
    if target >= 1.0:
        return -radius
    lo, hi = -radius, radius
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if submerged_fraction_quadrature(mid, radius, panels=2000) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class FreeFloatParameters:
    """Single-disk free-float fixture (2-D, per-unit-thickness)."""

    radius: float = 0.5
    thickness: float = 1.0
    rho_ice: float = 917.0
    rho_water: float = 1025.0
    gravity: float = 9.81
    wet_drag_coeff: float = 516.4
    amplitude: float = 5e-3
    duration: float = 6.0
    dt: float = 1e-2

    @property
    def mass(self) -> float:
        return self.rho_ice * math.pi * self.radius ** 2 * self.thickness

    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    def config(self) -> WetCouplingConfig:
        return WetCouplingConfig(
            enabled=True,
            water_density=self.rho_water,
            gravity=self.gravity,
            water_level=0.0,
            wet_drag_coeff=self.wet_drag_coeff,
        )

    def equilibrium_height(self) -> float:
        return solve_equilibrium_height(self.radius, self.rho_ice, self.rho_water)

    def stiffness(self) -> float:
        return -buoyancy_stiffness_analytic(
            self.equilibrium_height(), self.radius,
            self.rho_water, self.gravity, self.thickness,
        )

    def omega_n(self) -> float:
        return math.sqrt(self.stiffness() / self.mass)

    def damping_ratio(self) -> float:
        f_eq = self.rho_ice / self.rho_water
        return self.wet_drag_coeff * f_eq / (2.0 * self.mass * self.omega_n())

    def omega_d(self) -> float:
        zeta = self.damping_ratio()
        return self.omega_n() * math.sqrt(max(1e-30, 1.0 - zeta * zeta))


# ---------------------------------------------------------------------------
# Force law wrapper (uses the production routines under verification)
# ---------------------------------------------------------------------------

def vertical_acceleration(
    y: float,
    v: float,
    params: FreeFloatParameters,
    *,
    apply_gravity: bool = True,
) -> float:
    """Return ``d2y/dt2`` from the production buoyancy/drag force routines."""
    cfg = params.config()
    positions = torch.tensor([[0.0, y]], dtype=torch.float64)
    velocities = torch.tensor([[0.0, v]], dtype=torch.float64)
    f_buoy = compute_buoyancy_force(
        positions, params.radius, params.thickness, params.rho_ice, cfg
    )[0, 1].item()
    f_drag = compute_wet_drag_force(
        positions, velocities, params.radius, cfg
    )[0, 1].item()
    force = f_buoy + f_drag
    if apply_gravity:
        force -= params.weight
    return force / params.mass


def integrate_free_float(
    params: FreeFloatParameters,
    y0: float,
    v0: float = 0.0,
    *,
    apply_gravity: bool = True,
    sample_every: int = 1,
) -> dict[str, list[float]]:
    """Classical fixed-step RK4 on the 1-DOF motion driven by the real force law."""
    dt = params.dt
    steps = int(round(params.duration / dt))
    times: list[float] = []
    ys: list[float] = []
    vs: list[float] = []
    y, v = y0, v0
    acc = lambda yy, vv: vertical_acceleration(  # noqa: E731 - local shorthand
        yy, vv, params, apply_gravity=apply_gravity
    )
    for n in range(steps + 1):
        if n % sample_every == 0:
            times.append(n * dt)
            ys.append(y)
            vs.append(v)
        if n == steps:
            break
        k1y, k1v = v, acc(y, v)
        k2y, k2v = v + 0.5 * dt * k1v, acc(y + 0.5 * dt * k1y, v + 0.5 * dt * k1v)
        k3y, k3v = v + 0.5 * dt * k2v, acc(y + 0.5 * dt * k2y, v + 0.5 * dt * k2v)
        k4y, k4v = v + dt * k3v, acc(y + dt * k3y, v + dt * k3v)
        y = y + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        v = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return {"time": times, "height": ys, "velocity": vs}


# ---------------------------------------------------------------------------
# Analytic linearised trajectory
# ---------------------------------------------------------------------------

def analytic_displacement(t: float | Sequence[float], params: FreeFloatParameters,
                          amplitude: float | None = None) -> Any:
    """Linearised heave about ``h_eq`` released from rest at ``h_eq + A``."""
    amp = params.amplitude if amplitude is None else amplitude
    zeta = params.damping_ratio()
    omega_n = params.omega_n()
    omega_d = params.omega_d()
    phase = (zeta * omega_n / omega_d)
    if isinstance(t, (int, float)):
        return amp * math.exp(-zeta * omega_n * t) * (
            math.cos(omega_d * t) + phase * math.sin(omega_d * t)
        )
    return [
        amp * math.exp(-zeta * omega_n * ti) * (
            math.cos(omega_d * ti) + phase * math.sin(omega_d * ti)
        )
        for ti in t
    ]


def analytic_envelope(t: float, params: FreeFloatParameters,
                      amplitude: float | None = None) -> float:
    """Magnitude bound ``A * exp(-zeta * omega_n * t)`` of the linearised response."""
    amp = params.amplitude if amplitude is None else amplitude
    return amp * math.exp(-params.damping_ratio() * params.omega_n() * t)


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------

def _zero_crossings(times: Sequence[float], values: Sequence[float]) -> list[float]:
    crossings: list[float] = []
    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if a == 0.0:
            crossings.append(times[i - 1])
        elif a * b < 0.0:
            frac = abs(a) / (abs(a) + abs(b))
            crossings.append(times[i - 1] + frac * (times[i] - times[i - 1]))
    return crossings


def _extrema(times: Sequence[float], values: Sequence[float]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for i in range(1, len(values) - 1):
        a, b, c = values[i - 1], values[i], values[i + 1]
        if (b - a) * (c - b) <= 0.0 and abs(b) > abs(a) and abs(b) > abs(c):
            out.append((times[i], b))
    return out


def measure_oscillation(
    history: dict[str, list[float]], equilibrium: float
) -> dict[str, Any]:
    """Measure damped frequency and envelope decay rate from a time history."""
    times = history["time"]
    dev = [y - equilibrium for y in history["height"]]
    crossings = _zero_crossings(times, dev)
    omega_d = float("nan")
    if len(crossings) >= 3:
        gaps = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
        omega_d = math.pi / (sum(gaps) / len(gaps))
    extrema = _extrema(times, dev)
    decay = float("nan")
    if len(extrema) >= 2:
        ts = [e[0] for e in extrema]
        ln_a = [math.log(abs(e[1])) for e in extrema if e[1] != 0.0]
        if len(ln_a) == len(ts) and len(ts) >= 2:
            n = len(ts)
            mean_t = sum(ts) / n
            mean_l = sum(ln_a) / n
            num = sum((t - mean_t) * (l - mean_l) for t, l in zip(ts, ln_a))
            den = sum((t - mean_t) ** 2 for t in ts)
            decay = -(num / den) if den else float("nan")
    return {
        "omega_d": omega_d,
        "decay_rate": decay,          # zeta * omega_n
        "crossings": len(crossings),
        "extrema": extrema,
        "deviation": dev,
    }


def _rel_err(value: float, reference: float) -> float:
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

def geometry_check(radius: float = 0.5, samples: int = 61) -> dict[str, Any]:
    """Closed-form submerged fraction vs independent quadrature."""
    from .buoyancy_wet import submerged_fraction_tensor

    worst = 0.0
    worst_h = 0.0
    for i in range(samples):
        h = -1.2 * radius + 2.4 * radius * i / (samples - 1)
        closed = submerged_fraction_tensor(
            torch.tensor([h], dtype=torch.float64), radius, 0.0
        )[0].item()
        quad = submerged_fraction_quadrature(h, radius, panels=2000)
        err = abs(closed - quad)
        if err > worst:
            worst, worst_h = err, h
    return {
        "name": "geometry_vs_quadrature",
        "samples": samples,
        "max_abs_error": worst,
        "worst_height": worst_h,
        "tolerance": 1e-9,
        "passed": worst <= 1e-9,
    }


def stiffness_check(params: FreeFloatParameters, heights: Sequence[float] | None = None,
                    delta: float = 1e-4) -> dict[str, Any]:
    """Central-difference dF_b/dh of the real routine vs the chord-width identity."""
    cfg = params.config()
    if heights is None:
        h_eq = params.equilibrium_height()
        heights = [h_eq - 0.1, h_eq, h_eq + 0.1, 0.0]
    rows: list[dict[str, float]] = []
    worst = 0.0
    for h in heights:
        def f_b(height: float) -> float:
            pos = torch.tensor([[0.0, height]], dtype=torch.float64)
            return compute_buoyancy_force(
                pos, params.radius, params.thickness, params.rho_ice, cfg
            )[0, 1].item()

        numeric = -(f_b(h + delta) - f_b(h - delta)) / (2.0 * delta)
        analytic = -buoyancy_stiffness_analytic(
            h, params.radius, params.rho_water, params.gravity, params.thickness
        )
        err = _rel_err(numeric, analytic)
        worst = max(worst, err)
        rows.append({"height": h, "numeric": numeric, "analytic": analytic,
                     "rel_error": err})
    return {
        "name": "stiffness_identity",
        "rows": rows,
        "max_rel_error": worst,
        "tolerance": 1e-6,
        "passed": worst <= 1e-6,
    }


def archimedes_check(params: FreeFloatParameters,
                     settle_duration: float = 22.0) -> dict[str, Any]:
    """Free-float settling must reproduce f_sub = rho_ice / rho_water."""
    from .buoyancy_wet import submerged_fraction_tensor

    h_eq = params.equilibrium_height()
    f_target = params.rho_ice / params.rho_water
    long = FreeFloatParameters(
        radius=params.radius, thickness=params.thickness, rho_ice=params.rho_ice,
        rho_water=params.rho_water, gravity=params.gravity,
        wet_drag_coeff=params.wet_drag_coeff, amplitude=params.amplitude,
        duration=settle_duration, dt=2e-2,
    )
    history = integrate_free_float(long, h_eq + params.amplitude, sample_every=50)
    y_final = history["height"][-1]
    v_final = history["velocity"][-1]
    f_final = submerged_fraction_tensor(
        torch.tensor([y_final], dtype=torch.float64), params.radius, 0.0
    )[0].item()
    draft_analytic = params.radius - h_eq
    draft_sim = params.radius - y_final
    draft_err = _rel_err(draft_sim, draft_analytic)
    return {
        "name": "archimedes_draft",
        "f_target": f_target,
        "f_final": f_final,
        "f_abs_error": abs(f_final - f_target),
        "h_eq_analytic": h_eq,
        "h_final_simulated": y_final,
        "v_final": v_final,
        "draft_analytic": draft_analytic,
        "draft_simulated": draft_sim,
        "draft_rel_error": draft_err,
        "tolerance_rel": 3e-3,
        "passed": draft_err <= 3e-3 and abs(f_final - f_target) <= 1e-3,
    }


def oscillator_check(params: FreeFloatParameters,
                     amplitude: float | None = None) -> dict[str, Any]:
    """Damped heave frequency, decay rate and waveform vs the linearised solution."""
    amp = params.amplitude if amplitude is None else amplitude
    h_eq = params.equilibrium_height()
    history = integrate_free_float(params, h_eq + amp, sample_every=1)
    measured = measure_oscillation(history, h_eq)
    omega_analytic = params.omega_d()
    decay_analytic = params.damping_ratio() * params.omega_n()
    analytic = analytic_displacement(history["time"], params, amplitude=amp)
    dev = measured["deviation"]
    waveform_err = max(abs(d - a) for d, a in zip(dev, analytic)) / amp
    omega_err = _rel_err(measured["omega_d"], omega_analytic)
    decay_err = _rel_err(measured["decay_rate"], decay_analytic)
    return {
        "name": "damped_heave",
        "amplitude": amp,
        "dt": params.dt,
        "omega_d_measured": measured["omega_d"],
        "omega_d_analytic": omega_analytic,
        "omega_rel_error": omega_err,
        "decay_measured": measured["decay_rate"],
        "decay_analytic": decay_analytic,
        "decay_rel_error": decay_err,
        "waveform_max_rel_error": waveform_err,
        "crossings": measured["crossings"],
        "tolerance_rel": 3e-2,
        "passed": omega_err <= 3e-2 and decay_err <= 3e-2 and waveform_err <= 3e-2,
        "history": history,
        "analytic": analytic,
    }


def drag_decay_check(params: FreeFloatParameters, v0: float = 0.05) -> dict[str, Any]:
    """Neutral-density, fully-submerged particle must decelerate exponentially."""
    neutral = FreeFloatParameters(
        radius=params.radius, thickness=params.thickness, rho_ice=params.rho_water,
        rho_water=params.rho_water, gravity=params.gravity,
        wet_drag_coeff=params.wet_drag_coeff, amplitude=params.amplitude,
        duration=2.0, dt=2e-2,
    )
    # Fully submerged (centre at -1.5 R) and neutrally buoyant: net quasi-static
    # force vanishes, so the only force is the linear wet drag.
    y0 = -1.5 * params.radius
    history = integrate_free_float(neutral, y0, v0, sample_every=1)
    times = history["time"]
    velocities = history["velocity"]
    ln_v = [math.log(v / v0) for v in velocities]
    n = len(times)
    mean_t = sum(times) / n
    mean_l = sum(ln_v) / n
    slope = sum((t - mean_t) * (l - mean_l) for t, l in zip(times, ln_v)) / sum(
        (t - mean_t) ** 2 for t in times
    )
    rate_measured = -slope
    rate_analytic = neutral.wet_drag_coeff / neutral.mass
    err = _rel_err(rate_measured, rate_analytic)
    return {
        "name": "drag_exponential_decay",
        "rate_measured": rate_measured,
        "rate_analytic": rate_analytic,
        "rel_error": err,
        "v_end": velocities[-1],
        "tolerance_rel": 3e-2,
        "passed": err <= 3e-2,
    }


def timestep_convergence(params: FreeFloatParameters,
                         dts: Sequence[float] = (8e-2, 4e-2, 2e-2)) -> dict[str, Any]:
    """Refine the timestep and check the free-float measurement is converged."""
    h_eq = params.equilibrium_height()
    states: list[float] = []
    freqs: list[float] = []
    for dt in dts:
        local = FreeFloatParameters(
            radius=params.radius, thickness=params.thickness, rho_ice=params.rho_ice,
            rho_water=params.rho_water, gravity=params.gravity,
            wet_drag_coeff=params.wet_drag_coeff, amplitude=params.amplitude,
            duration=params.duration, dt=dt,
        )
        history = integrate_free_float(local, h_eq + params.amplitude, sample_every=1)
        states.append(history["height"][-1])
        freqs.append(measure_oscillation(history, h_eq)["omega_d"])
    diffs = [abs(states[i + 1] - states[i]) for i in range(len(states) - 1)]
    ratios = [diffs[i] / diffs[i + 1] for i in range(len(diffs) - 1)
              if diffs[i + 1] > 0.0]
    order = math.log(ratios[0], 2.0) if ratios else float("nan")
    freq_spread = (max(freqs) - min(freqs)) / freqs[-1]
    return {
        "name": "timestep_convergence",
        "dts": list(dts),
        "final_heights": states,
        "omega_d": freqs,
        "abs_differences": diffs,
        "observed_order": order,
        "omega_d_spread_rel": freq_spread,
        "tolerance_order": 3.0,
        "tolerance_spread": 1e-2,
        "passed": (order >= 3.0) and (freq_spread <= 1e-2),
    }


def linearisation_check(params: FreeFloatParameters,
                        amplitudes: Sequence[float] = (1e-2, 1e-3)) -> dict[str, Any]:
    """The linearised-vs-exact gap must shrink as the amplitude is reduced."""
    rows: list[dict[str, float]] = []
    for amp in amplitudes:
        result = oscillator_check(params, amplitude=amp)
        rows.append({
            "amplitude": amp,
            "waveform_max_rel_error": result["waveform_max_rel_error"],
            "omega_rel_error": result["omega_rel_error"],
        })
    shrinking = rows[-1]["waveform_max_rel_error"] < rows[0]["waveform_max_rel_error"]
    return {
        "name": "linearisation_limit",
        "rows": rows,
        "shrinking": shrinking,
        "passed": shrinking and all(r["waveform_max_rel_error"] <= 3e-2 for r in rows),
    }


def run_campaign(params: FreeFloatParameters | None = None,
                 *, include_histories: bool = True,
                 quick: bool = False) -> dict[str, Any]:
    """Execute every check and return a deterministic JSON-serialisable report.

    ``quick`` shortens the settling run so the campaign fits a CI budget; the
    tolerances are unchanged, only the observation window is shortened.
    """
    p = params or FreeFloatParameters()
    if quick:
        p = FreeFloatParameters(
            radius=p.radius, thickness=p.thickness, rho_ice=p.rho_ice,
            rho_water=p.rho_water, gravity=p.gravity,
            wet_drag_coeff=p.wet_drag_coeff, amplitude=p.amplitude,
            duration=4.0, dt=2e-2,
        )
    geometry = geometry_check(p.radius)
    stiffness = stiffness_check(p)
    archimedes = archimedes_check(p, settle_duration=14.0 if quick else 22.0)
    oscillator = oscillator_check(p)
    drag = drag_decay_check(p)
    timestep = timestep_convergence(p)
    linearisation = linearisation_check(p)

    checks = [geometry, stiffness, archimedes, oscillator, drag, timestep, linearisation]
    if not include_histories:
        oscillator = {k: v for k, v in oscillator.items() if k not in ("history", "analytic")}

    h_eq = p.equilibrium_height()
    summary = {
        "protocol": PROTOCOL,
        "fixture": {
            "radius_m": p.radius,
            "thickness_m": p.thickness,
            "rho_ice_kg_m3": p.rho_ice,
            "rho_water_kg_m3": p.rho_water,
            "gravity_m_s2": p.gravity,
            "wet_drag_coeff_N_s_m": p.wet_drag_coeff,
            "mass_kg": p.mass,
            "amplitude_m": p.amplitude,
            "duration_s": p.duration,
            "dt_s": p.dt,
            "f_eq_expected": p.rho_ice / p.rho_water,
            "h_eq_analytic_m": h_eq,
            "draft_analytic_m": p.radius - h_eq,
            "stiffness_N_m": p.stiffness(),
            "omega_n_rad_s": p.omega_n(),
            "zeta": p.damping_ratio(),
            "omega_d_rad_s": p.omega_d(),
            "period_d_s": 2.0 * math.pi / p.omega_d(),
        },
        "checks": {c["name"]: c for c in checks},
        "all_passed": all(c["passed"] for c in checks),
    }
    if not include_histories:
        for name in ("damped_heave",):
            summary["checks"][name] = {
                k: v for k, v in summary["checks"][name].items()
                if k not in ("history", "analytic")
            }
    return summary