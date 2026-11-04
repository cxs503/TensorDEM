"""Reference solution and corrected mapping for the *Hertzian* restitution target.

Why this module exists
----------------------
``tensordem.contact_restitution`` closed the gap between the requested and the
measured coefficient of restitution (COR) for the **linear** spring-dashpot
contact law.  The Hertzian branch of :class:`tensordem.dem3d.IceDEM3D` was
explicitly left out of scope there ("Adhesive or Hertzian contacts are out of
scope - this validates the linear law only"), and the shipped code reuses the
*linear free-oscillator* identity

    zeta = |ln e| / sqrt(pi^2 + ln(e)^2),      e = exp(-pi zeta / sqrt(1-zeta^2))

to turn a requested ``restitution_coefficient`` into a damping coefficient

    gamma(delta) = 2 zeta sqrt(k_s(delta) m_eff),     k_s(delta) = A sqrt(delta)

with ``A = (4/3) E* sqrt(R*)`` the Hertz prefactor and ``k_s`` the *secant*
stiffness at the current overlap.  That identity is derived for a **constant**
stiffness and a free (unclamped) oscillator; neither assumption holds here:

* the damper scales as ``delta^(1/4)`` (through ``sqrt(k_s)``), and
* the shipped contact force is truncated at zero (``max(0, ...)``), so the
  contact is released while the bodies still overlap.

Consequences measured in this module (all reproducible, see
``docs/dem3d_hertz_restitution_verification.md``):

* the COR produced by the shipped mapping is **independent of the impact
  speed** - the dimensionless impact problem is invariant under
  ``delta -> s^2 delta, t -> s^-1 t`` because both the elastic and the damping
  term scale as ``s^3``.  The bias therefore depends on ``e_target`` alone, so
  a single scalar re-mapping is enough to remove it;
* that bias is +1.3 % at ``e_target = 0.9``, +3.2 % at 0.8, **+16.5 % at 0.5**
  and +43.7 % at 0.3 - i.e. the requested COR is systematically *overshot* and
  the 3 % accuracy band is exceeded already at ``e_target ~ 0.82``;
* inverting the reference integration gives ``zeta_eff(e_target)`` which, fed
  back into the same solver, reproduces the requested COR to well below 1 %.

What is (and is not) claimed
----------------------------
The reference model integrated here is *the model the solver implements*, not a
physical experiment: a rigid Hertzian spring in parallel with a linear viscous
damper, compressive-only.  Recovering the requested COR of that model is a
numerical-consistency property.  It does **not** validate Hertzian damping
physics against a measured coefficient (that needs an impact experiment), and
it does not apply to adhesive (JKR/DMT) or elasto-plastic contacts.

All quantities are SI (N, m, s, kg).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Iterable

__all__ = [
    "HertzImpactSpec",
    "hertz_prefactor",
    "classic_zeta_from_restitution",
    "secant_damping_zeta",
    "hertz_impact_restitution",
    "hertz_zeta_for_target",
    "hertz_restitution_bias_table",
    "measure_solver_impact",
]


def hertz_prefactor(e_star: float, r_eff: float) -> float:
    """Hertz prefactor ``A`` in ``F = A delta^(3/2)`` [N m^-3/2]."""
    if not math.isfinite(e_star) or e_star <= 0.0:
        raise ValueError("e_star must be finite and positive")
    if not math.isfinite(r_eff) or r_eff <= 0.0:
        raise ValueError("r_eff must be finite and positive")
    return (4.0 / 3.0) * e_star * math.sqrt(r_eff)


def classic_zeta_from_restitution(e: float) -> float:
    """Inverse of the linear free-oscillator identity ``e = exp(-pi z / sqrt(1-z^2))``."""
    if not (math.isfinite(e) and 0.0 < e <= 1.0):
        raise ValueError("restitution coefficient must be in (0, 1]")
    if e == 1.0:
        return 0.0
    ln_e = abs(math.log(e))
    return ln_e / math.sqrt(math.pi**2 + ln_e**2)


@dataclass(frozen=True)
class HertzImpactSpec:
    """Reference impact problem: two equal spheres, or a sphere on a rigid plane.

    ``e_star``/``r_eff`` are the Hertz effective values, ``mass`` is the mass of
    one sphere (the reduced mass ``mass/2`` enters the relative-coordinate ODE),
    ``v_in`` is the closing speed.  ``steps`` is the RK4 step count over roughly
    five elastic contact durations.
    """

    e_star: float = 1.0 / (2.0 * (1.0 - 0.3**2) / 1.0e9)
    r_eff: float = 0.0125
    mass: float = 917.0 * (4.0 / 3.0) * math.pi * 0.025**3
    v_in: float = 1.0
    steps: int = 2_000

    def __post_init__(self) -> None:
        for name in ("e_star", "r_eff", "mass", "v_in"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 200:
            raise ValueError("steps must be an integer >= 200")

    @property
    def reduced_mass(self) -> float:
        return 0.5 * self.mass

    @property
    def prefactor(self) -> float:
        return hertz_prefactor(self.e_star, self.r_eff)

    def peak_overlap(self) -> float:
        """Elastic (undamped) peak overlap ``delta_max = (15 m_eff v^2 / (16 A))^(2/5)``.

        From ``U(delta_max) = (8/15) A delta^(5/2) = m_eff v^2 / 2``.
        """
        a = self.prefactor
        return (15.0 * self.reduced_mass * self.v_in**2 / (16.0 * a)) ** 0.4


def secant_damping_zeta(zeta: float) -> float:
    """Validate a damping ratio used with the secant-stiffness damper."""
    if not math.isfinite(zeta) or zeta < 0.0:
        raise ValueError("zeta must be finite and nonnegative")
    return float(zeta)


def hertz_impact_restitution(
    zeta: float,
    spec: HertzImpactSpec | None = None,
    *,
    record: bool = False,
) -> dict[str, Any]:
    """Reference COR of the shipped Hertzian contact law for a given damping ratio.

    Integrates the compressive-only relative motion

        m_eff * d2(delta)/dt2 = -(A delta^(3/2) + gamma(delta) * d(delta)/dt)
        gamma(delta)          = 2 zeta sqrt(k_s(delta) m_eff),  k_s = A sqrt(delta)

    with classical RK4 and a fixed step count spanning ~five elastic contact
    durations.  ``delta = 0`` is the first contact instant, ``d(delta)/dt = v_in``
    the closing speed, and the returned COR is ``|d(delta)/dt| / v_in`` at the
    moment the overlap returns to zero.  The force is clipped at zero, matching
    ``dem3d`` (no tensile/adhesive contact).

    Returns a dict with ``e``, ``zeta``, ``contact_time_s``, ``steps``,
    ``peak_overlap_m`` and ``dt_s`` so that callers can record convergence
    evidence instead of a bare number.  With ``record=True`` the dict also
    carries ``history``: ``(t_s, overlap_m, force_N)`` samples of the contact,
    used to overlay the solver trace in the verification figure.
    """
    spec = HertzImpactSpec() if spec is None else spec
    zeta = secant_damping_zeta(zeta)
    a = spec.prefactor
    m_eff = spec.reduced_mass
    v_in = spec.v_in
    delta_max = spec.peak_overlap()
    # Elastic Hertz contact time is ~2.94 delta_max / v_in; integrate 5 of those.
    dt = 5.0 * 2.94 * delta_max / v_in / spec.steps

    def gamma(delta: float) -> float:
        if zeta == 0.0:
            return 0.0
        return 2.0 * zeta * math.sqrt(a * math.sqrt(delta) * m_eff)

    def force(delta: float, rate: float) -> float:
        delta = delta if delta > 0.0 else 0.0
        return max(0.0, a * delta**1.5 + gamma(delta) * rate)

    def rhs(delta: float, rate: float) -> tuple[float, float]:
        return rate, -force(delta, rate) / m_eff

    delta = 0.0
    rate = v_in
    t = 0.0
    history: list[tuple[float, float, float]] = []
    for step in range(spec.steps * 12):
        k1 = rhs(delta, rate)
        k2 = rhs(delta + 0.5 * dt * k1[0], rate + 0.5 * dt * k1[1])
        k3 = rhs(delta + 0.5 * dt * k2[0], rate + 0.5 * dt * k2[1])
        k4 = rhs(delta + dt * k3[0], rate + dt * k3[1])
        delta_new = delta + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        rate_new = rate + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        t += dt
        if record:
            history.append((t, max(delta, 0.0), force(delta, rate)))
        if step > 2 and delta > 0.0 and delta_new <= 0.0:
            payload = {
                "e": abs(rate_new) / v_in,
                "zeta": zeta,
                "contact_time_s": t,
                "steps": spec.steps,
                "dt_s": dt,
                "peak_overlap_m": delta_max,
            }
            if record:
                payload["history"] = history
            return payload
        delta, rate = delta_new, rate_new
    raise RuntimeError(
        "Hertz impact did not separate within the integration window; "
        "increase steps or check the damping ratio"
    )


@lru_cache(maxsize=256)
def _zeta_for_target(e_target: float, steps: int) -> tuple[float, float]:
    """Cached bisection of ``hertz_impact_restitution`` (monotone decreasing in zeta)."""
    spec = HertzImpactSpec(steps=steps)
    if hertz_impact_restitution(0.0, spec)["e"] < e_target:
        raise ValueError("elastic limit never falls below the requested restitution")
    lo, hi = 0.0, 1.0
    e_hi = hertz_impact_restitution(hi, spec)["e"]
    # The clipped law can hold the contact open once damping exceeds the elastic
    # term during approach (zeta >~ 1/sqrt(2)); widen only if the target demands it.
    while e_hi > e_target and hi < 32.0:
        hi *= 2.0
        e_hi = hertz_impact_restitution(hi, spec)["e"]
    if e_hi > e_target:
        raise ValueError(
            f"no damping ratio up to zeta={hi:g} reaches e_target={e_target:g}"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        e_mid = hertz_impact_restitution(mid, spec)["e"]
        if abs(e_mid - e_target) < 1e-5:
            return mid, e_mid
        if e_mid > e_target:
            lo = mid
        else:
            hi = mid
    return mid, e_mid


def hertz_zeta_for_target(e_target: float, *, steps: int = 2_000) -> dict[str, Any]:
    """Damping ratio that reproduces ``e_target`` under the shipped Hertz law.

    This is the *corrected* counterpart of :func:`classic_zeta_from_restitution`
    for ``contact_model="hertz"``.  The returned ``zeta`` is meant to enter the
    same ``gamma(delta) = 2 zeta sqrt(k_s(delta) m_eff)`` formula the solver
    already uses, so only one scalar changes.  The mapping is speed independent
    by construction (see the module docstring).
    """
    if not (math.isfinite(e_target) and 0.0 < e_target <= 1.0):
        raise ValueError("e_target must be in (0, 1]")
    if e_target == 1.0:
        return {"e_target": 1.0, "zeta_eff": 0.0, "e_verified": 1.0,
                "zeta_classic": 0.0, "bias_pct": 0.0, "steps": steps}
    zeta_eff, e_verified = _zeta_for_target(float(e_target), int(steps))
    zeta_classic = classic_zeta_from_restitution(e_target)
    return {
        "e_target": float(e_target),
        "zeta_eff": zeta_eff,
        "e_verified": e_verified,
        "zeta_classic": zeta_classic,
        "bias_pct": 100.0 * (e_verified / e_target - 1.0),
        "steps": steps,
    }


def hertz_restitution_bias_table(
    e_targets: Iterable[float] = (0.95, 0.9, 0.8, 0.7, 0.5, 0.3),
    *,
    speeds: Iterable[float] = (0.1, 0.5, 1.0, 2.0),
    steps: int = 2_000,
) -> list[dict[str, Any]]:
    """Bias of the *shipped* (classic-zeta) mapping and the corrected mapping.

    One row per ``(e_target, v_in)``: the classical zeta, the COR it actually
    produces, its relative bias, and the same two numbers for the corrected
    zeta.  Speeds are included to demonstrate the speed independence instead of
    asserting it.
    """
    rows: list[dict[str, Any]] = []
    for e_target in e_targets:
        zeta_classic = classic_zeta_from_restitution(e_target)
        corrected = hertz_zeta_for_target(e_target, steps=steps)
        for v_in in speeds:
            spec = HertzImpactSpec(v_in=float(v_in), steps=steps)
            shipped = hertz_impact_restitution(zeta_classic, spec)
            fixed = hertz_impact_restitution(corrected["zeta_eff"], spec)
            rows.append(
                {
                    "e_target": float(e_target),
                    "v_in": float(v_in),
                    "zeta_classic": zeta_classic,
                    "e_shipped": shipped["e"],
                    "bias_shipped_pct": 100.0 * (shipped["e"] / e_target - 1.0),
                    "zeta_eff": corrected["zeta_eff"],
                    "e_corrected": fixed["e"],
                    "bias_corrected_pct": 100.0 * (fixed["e"] / e_target - 1.0),
                }
            )
    return rows

# ---------------------------------------------------------------------------
# Solver-level measurement (deferred import: dem3d imports this module)
# ---------------------------------------------------------------------------
def measure_solver_impact(
    *,
    e_target: float,
    verified: bool,
    v0: float = 0.5,
    dt_scale: float = 0.35,
    radius: float = 0.025,
    density: float = 917.0,
    young_modulus: float = 1.0e9,
    poisson_ratio: float = 0.3,
    record: bool = False,
) -> dict[str, Any]:
    """Impact two free ``IceDEM3D`` spheres on the Hertz branch and measure the COR.

    The pair is bond-free and spin-free: particle 0 and particle 1 face each other
    along z and every other lattice site is parked and fixed, so the only active
    force is the normal contact under test.  ``v0`` is the speed of *each*
    sphere (closing speed ``2 v0``); the measured COR is ``|v_out| / v0`` and is
    therefore directly comparable with :func:`hertz_impact_restitution` at
    ``v_in = 2 v0``.

    ``record=True`` additionally returns ``history`` as ``(t_s, overlap_m,
    force_N)`` samples, where the force is recovered from the momentum change of
    particle 0 over one step.
    """
    import torch  # noqa: PLC0415 - keep the numeric core import-light

    from .dem3d import DEM3DConfig, IceDEM3D  # noqa: PLC0415 - deferred cycle

    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=radius, density=density,
        contact_model="hertz", young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        contact_damping_model="restitution", restitution_coefficient=e_target,
        hertz_restitution_verified=verified,
        drag=0.0, platen_damping=0.0,
        bond_stiffness=1.0, breaking_strain=1e-12, shear_breaking_strain=1e-12,
        fix_x_edges=False, fix_bottom=False, tool_gap=1.0e3, tool_speed=1.0e-9,
    )
    dem = IceDEM3D(cfg)
    n = len(dem.positions)
    positions = torch.zeros_like(dem.positions)
    for i in range(n):
        positions[i] = torch.tensor([10.0 * i, 500.0, 500.0], dtype=dem.dtype)
    positions[0] = torch.tensor([0.0, 0.0, -(radius + 1.0e-6)], dtype=dem.dtype)
    positions[1] = torch.tensor([0.0, 0.0, radius + 1.0e-6], dtype=dem.dtype)
    dem.positions = positions.clone()
    dem.initial_positions = positions.clone()
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    mask[1] = False
    dem.fixed = mask
    # Break every lattice bond so the pair is genuinely free-flying.
    dem.forces()
    if int(dem.alive.sum().item()) != 0:
        raise RuntimeError("bonded lattice did not fully break; harness assumption violated")
    dem.dt = dem.dt * dt_scale
    dem.velocities = torch.zeros_like(dem.positions)
    dem.velocities[0] = torch.tensor([0.0, 0.0, v0], dtype=dem.dtype)
    dem.velocities[1] = torch.tensor([0.0, 0.0, -v0], dtype=dem.dtype)

    mass = float(cfg.mass)
    dt = float(dem.dt)
    history: list[tuple[float, float, float]] = []
    peak_overlap = 0.0
    contact_steps = 0
    for step in range(200_000):
        v_prev = float(dem.velocities[0, 2])
        dem.step()
        v_now = float(dem.velocities[0, 2])
        separation = float(dem.positions[1, 2] - dem.positions[0, 2])
        overlap = 2.0 * radius - separation
        if overlap > 0.0:
            contact_steps += 1
            peak_overlap = max(peak_overlap, overlap)
        if record and step % 2 == 0:
            history.append((float(dem.time), max(overlap, 0.0),
                            mass * (v_prev - v_now) / dt))
        if separation > 2.0 * radius and step > 5:
            payload: dict[str, Any] = {
                "e_measured": abs(v_now) / v0,
                "e_target": e_target,
                "verified": verified,
                "steps_to_separation": step,
                "contact_steps": contact_steps,
                "dt_s": dt,
                "peak_overlap_m": peak_overlap,
            }
            if record:
                payload["history"] = history
            return payload
    raise RuntimeError("impact did not separate; check dt_scale / initial clearance")
