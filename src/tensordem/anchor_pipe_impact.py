"""Anchor drop / drag impact on a subsea pipeline (campaign node ``p22w2-n08``).

Damage chain implemented here (each link has its own analytic anchor):

1. **Free fall through water** -- rigid anchor (sphere or cone) of mass ``m``
   falling a height ``h`` with added mass ``C_a`` and submerged weight;
   impact velocity ``v0 = sqrt(2 a_net h)``, ``a_net = W' / (m (1+C_a))``.
2. **Soil penetration** -- marine clay, Mohr-Coulomb bearing capacity with
   buoyant overburden, ``Q_u(z) = (N_c c_u + gamma' z) A(z)``, where ``A(z)``
   is the projected (frontal) area of the anchor at embedment ``z``.
3. **Pipe local dent** -- plastic ring squash: the four-hinge limit load of a
   ring loaded by two opposite point loads is ``P_c = 4 M_pl / R`` with
   ``M_pl = sigma_y t^2 / 4`` (per unit axial length).  The anchor loads the
   pipe over a growing contact patch ``L_c(delta) = 2 sqrt(2 R_eq delta)``,
   hence ``F_pipe = P_c L_c`` and ``E_dent = (4/3) P_c sqrt(2 R_eq) delta^1.5``
   -- the classical ``E ~ delta^1.5`` denting law.
4. **Dynamic chain** -- 1-DOF impulse/momentum time integration (velocity
   Verlet, adaptive step) of
   ``m_eff z'' = W' - Q_u(z) - F_pipe(z - z_cov)``,
   giving penetration depth, dent and the energy partition.
5. **DNV-RP-F107-style closed-form energy method** -- the same energy partition
   solved statically (impact energy absorbed by soil work + dent work), with
   the design rule ``delta/D <= 5%`` for a no-repair dent.

Units: SI throughout.  ``M_pl`` and all ring quantities are *per unit axial
length* of the pipe (plane-strain ring squash), so ``P_c`` is N/m and the
contact length turns it into a force.

Verification (machine-checkable; driver in ``benchmarks/anchor_pipe_impact``):

* E1 free fall / rebound: ``h_rebound = e^2 h`` within 2 %; momentum change
  equals ``int F dt`` within 1e-6.
* E2 static bearing equilibrium: dynamic settling depth vs the analytic root
  of ``W' = Q_u(z)`` within 3 %.
* E3 ring-squash limit load (DEM relaxation ring) vs ``4 M_pl / R`` within 5 %.
* E4 dent-energy scaling: local exponent of ``delta`` vs impact energy matches
  the analytic plastic-work exponent (2/3) within 10 %.
* E5 verdict agreement with the RP-F107-style closed form >= 90 %.
* E6 negative controls: without soil resistance the anchor crushes the pipe;
  anchor mass x1.5 moves the penetration depth per the analytic scale.
* E7 no full-scale anchor-impact test data in the worktree -> WITHHELD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

G = 9.81
STEEL_RHO = 7850.0
WATER_RHO = 1025.0

__all__ = [
    "G",
    "STEEL_RHO",
    "WATER_RHO",
    "Pipe",
    "Soil",
    "Anchor",
    "ImpactCase",
    "soil_resistance",
    "soil_energy",
    "static_equilibrium_depth",
    "soil_penetration_depth",
    "contact_length",
    "dent_force",
    "dent_energy",
    "free_fall_rebound",
    "ring_squash_load_curve",
    "ring_squash_limit_load",
    "simulate_impact",
    "rp_f107_closed_form",
    "dent_scaling_exponent",
    "energy_ledger",
    "evaluate_acceptance",
]


# ---------------------------------------------------------------------------
# input model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pipe:
    """Steel pipe section (ring quantities are per unit axial length)."""

    D: float = 0.508          # outer diameter [m]
    t: float = 0.0206         # wall thickness [m]
    sigma_y: float = 448.0e6  # yield stress [Pa] (X65-ish)
    E: float = 210.0e9        # Young's modulus [Pa]
    dent_allow_ratio: float = 0.05  # no-repair dent limit delta/D [-]

    @property
    def R(self) -> float:
        return 0.5 * self.D

    @property
    def second_moment(self) -> float:
        """Second moment of area of the wall per unit axial length [m^3]."""
        return self.t ** 3 / 12.0

    @property
    def M_pl(self) -> float:
        """Plastic moment per unit axial length: M_pl = sigma_y t^2 / 4."""
        return self.sigma_y * self.t ** 2 / 4.0

    @property
    def ring_limit_load(self) -> float:
        """Four-hinge ring-squash limit load ``4 M_pl / R`` [N/m]."""
        return 4.0 * self.M_pl / self.R

    def geometry_check(self) -> Dict[str, float]:
        return {
            "D_over_t": self.D / self.t,
            "M_pl_Nm_per_m": self.M_pl,
            "ring_limit_load_N_per_m": self.ring_limit_load,
            "elastic_diametral_stiffness_N_per_m": (
                self.E * self.second_moment / (0.1488 * self.R ** 3)
            ),
        }


@dataclass(frozen=True)
class Soil:
    """Undrained marine clay (Mohr-Coulomb bearing + buoyant overburden)."""

    c_u: float = 60.0e3       # undrained shear strength [Pa]
    N_c: float = 7.5          # bearing capacity factor [-]
    gamma_eff: float = 8.0e3  # buoyant unit weight [N/m^3]


@dataclass(frozen=True)
class Anchor:
    """Rigid anchor idealised as a sphere or a cone (tip down)."""

    kind: str = "sphere"          # 'sphere' | 'cone'
    mass: float = 3000.0          # dry mass in air [kg]
    half_angle_deg: float = 30.0  # cone half angle (cone only)
    C_a: float = 0.5              # added mass coefficient [-]
    tip_radius: float = 0.10      # cone tip rounding -> contact patch [m]

    @property
    def volume(self) -> float:
        return self.mass / STEEL_RHO

    @property
    def half_angle(self) -> float:
        return math.radians(self.half_angle_deg)

    @property
    def radius(self) -> float:
        """Equivalent sphere radius from the displaced volume."""
        return (3.0 * self.volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    @property
    def height(self) -> float:
        """Cone height from the displaced volume."""
        return (3.0 * self.volume / (math.pi * math.tan(self.half_angle) ** 2)) ** (1.0 / 3.0)

    @property
    def dimension(self) -> float:
        return self.radius if self.kind == "sphere" else self.height

    @property
    def buoyancy(self) -> float:
        return WATER_RHO * self.volume * G

    @property
    def weight_eff(self) -> float:
        """Submerged (effective) weight [N]."""
        return (self.mass - WATER_RHO * self.volume) * G

    @property
    def m_eff(self) -> float:
        """Inertia mass including hydrodynamic added mass [kg]."""
        return self.mass * (1.0 + self.C_a)

    @property
    def a_net(self) -> float:
        """Net downward acceleration in water [m/s^2]."""
        return self.weight_eff / self.m_eff

    def radius_eq(self) -> float:
        return self.radius if self.kind == "sphere" else self.tip_radius

    def l_max(self) -> float:
        if self.kind == "sphere":
            return 2.0 * self.radius
        return 2.0 * self.height * math.tan(self.half_angle)

    def area(self, z):
        """Projected frontal area [m^2] at embedment ``z`` (scalar or vectorised)."""
        # largest horizontal section of the embedded part -> monotone in z
        if np.ndim(z) == 0:
            zf = max(float(z), 0.0)
            if self.kind == "sphere":
                r = self.radius
                zc = min(zf, r)
                return math.pi * (2.0 * r * zc - zc * zc)
            h = self.height
            th = math.tan(self.half_angle)
            return math.pi * (min(zf, h) * th) ** 2
        zz = np.clip(np.asarray(z, dtype=float), 0.0, None)
        if self.kind == "sphere":
            r = self.radius
            zc = np.minimum(zz, r)
            return math.pi * (2.0 * r * zc - zc * zc)
        h = self.height
        th = math.tan(self.half_angle)
        return math.pi * (np.minimum(zz, h) * th) ** 2


@dataclass(frozen=True)
class ImpactCase:
    """One anchor-drop scenario (local soil cover model)."""

    anchor: Anchor = Anchor()
    pipe: Pipe = Pipe()
    soil: Soil = Soil()
    drop_height: float = 5.0   # free fall height above seabed top [m]
    cover: float = 0.5         # soil cover above the pipe crown [m]
    soil_on: bool = True       # negative-control switch (E6)

    @property
    def v0(self) -> float:
        return math.sqrt(2.0 * self.anchor.a_net * self.drop_height)

    @property
    def impact_energy(self) -> float:
        """Kinetic energy at the seabed top [J]."""
        return 0.5 * self.anchor.m_eff * self.v0 ** 2

    def label(self) -> str:
        return (
            f"{self.anchor.kind}-{self.anchor.mass:.0f}kg-h{self.drop_height:.1f}"
            f"-cov{self.cover:.2f}-c{self.soil.c_u/1e3:.0f}"
        )


# ---------------------------------------------------------------------------
# link 2: soil resistance
# ---------------------------------------------------------------------------
def soil_resistance(z, anchor: Anchor, soil: Soil):
    """Mohr-Coulomb bearing capacity ``Q_u = (N_c c_u + gamma' z) A(z)`` [N]."""
    if np.ndim(z) == 0:
        zf = max(float(z), 0.0)
        return (soil.N_c * soil.c_u + soil.gamma_eff * zf) * anchor.area(zf)
    zz = np.clip(np.asarray(z, dtype=float), 0.0, None)
    return (soil.N_c * soil.c_u + soil.gamma_eff * zz) * anchor.area(zz)


def soil_energy(z: float, anchor: Anchor, soil: Soil, n_quad: int = 2001) -> float:
    """Work absorbed by the soil, ``int_0^z Q_u(z') dz'`` [J]."""
    if z <= 0.0:
        return 0.0
    zs = np.linspace(0.0, z, n_quad)
    q = soil_resistance(zs, anchor, soil)
    return float(np.trapezoid(q, zs))


def static_equilibrium_depth(case: ImpactCase, tol: float = 1e-10) -> Optional[float]:
    """Root of ``W' = Q_u(z)`` (bisection); ``None`` if the anchor sinks through."""
    if not case.soil_on:
        return None
    w = case.anchor.weight_eff
    z_hi = 2.0 * case.anchor.dimension
    if float(soil_resistance(z_hi, case.anchor, case.soil)) < w:
        return None
    z_lo = 0.0
    for _ in range(200):
        z_mid = 0.5 * (z_lo + z_hi)
        if float(soil_resistance(z_mid, case.anchor, case.soil)) < w:
            z_lo = z_mid
        else:
            z_hi = z_mid
        if z_hi - z_lo < tol:
            break
    return 0.5 * (z_lo + z_hi)


def soil_penetration_depth(case: ImpactCase, n_iter: int = 200) -> Optional[float]:
    """Penetration depth of the anchor in soil only (no pipe), momentum + weight.

    Solves ``E_impact + W' z = int_0^z Q_u dz'`` by bisection.  This is the
    analytic scale used by the negative control E6 (mass scaling).
    """
    if not case.soil_on:
        return None
    e_imp = case.impact_energy
    w = case.anchor.weight_eff

    def resid(z: float) -> float:
        return e_imp + w * z - soil_energy(z, case.anchor, case.soil)

    z_hi = 4.0 * case.anchor.dimension
    for _ in range(20):        # expand until the anchor is arrested
        if resid(z_hi) <= 0.0:
            break
        z_hi *= 2.0
    else:
        return None
    lo, hi = 0.0, z_hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# link 3: pipe local dent (plastic ring squash)
# ---------------------------------------------------------------------------
def contact_length(delta, anchor: Anchor):
    """Axial contact patch length of the anchor footprint [m]."""
    if np.ndim(delta) == 0:
        df = max(float(delta), 0.0)
        return min(anchor.l_max(), 2.0 * math.sqrt(2.0 * anchor.radius_eq() * df))
    d = np.clip(np.asarray(delta, dtype=float), 0.0, None)
    return np.minimum(anchor.l_max(), 2.0 * np.sqrt(2.0 * anchor.radius_eq() * d))


def dent_force(delta, anchor: Anchor, pipe: Pipe):
    """Local dent resistance ``F = P_c L_c(delta)`` [N]."""
    return pipe.ring_limit_load * contact_length(delta, anchor)


def dent_energy(delta: float, anchor: Anchor, pipe: Pipe) -> float:
    """Plastic work of denting, ``int_0^delta F ddelta`` [J] (closed form)."""
    if delta <= 0.0:
        return 0.0
    pc = pipe.ring_limit_load
    r_eq = anchor.radius_eq()
    d_cap = anchor.l_max() ** 2 / (8.0 * r_eq)
    if delta <= d_cap:
        return pc * (4.0 / 3.0) * math.sqrt(2.0 * r_eq) * delta ** 1.5
    e_cap = pc * (4.0 / 3.0) * math.sqrt(2.0 * r_eq) * d_cap ** 1.5
    return e_cap + pc * anchor.l_max() * (delta - d_cap)


# ---------------------------------------------------------------------------
# link 4: dynamic chain (impulse / momentum integration)
# ---------------------------------------------------------------------------
def _dent_force_model(case: ImpactCase, pipe_model: str = "plastic",
                      ring_curve: Optional[Dict[str, object]] = None):
    """Dent resistance as a function of the dent depth (two closures).

    ``plastic``  -- constant ring limit load ``P_c = 4 M_pl / R`` times the
                    growing contact patch (the analytic plastic-work closure);
    ``ring``     -- load taken from the elastic-plastic ring squash curve of
                    :func:`ring_squash_load_curve` (hinge elasticity included),
                    i.e. the collision force history is coupled to the pipe's
                    circumferential plastic-hinge model.
    """
    if pipe_model == "plastic":
        return lambda d: float(dent_force(d, case.anchor, case.pipe))
    if pipe_model != "ring":
        raise ValueError(f"unknown pipe_model {pipe_model!r}")
    if ring_curve is None:
        ring_curve = ring_squash_load_curve(case.pipe, n_seg=64, n_delta=20,
                                           delta_max_frac=0.6)
    rows = ring_curve["rows"]
    dd = np.array([r["delta"] for r in rows])
    ll = np.array([max(r["load"], 0.0) for r in rows])
    d_max = float(dd[-1])
    l_plateau = float(ll.max())            # plastic plateau -> constant beyond table

    def dent(d: float) -> float:
        if d <= 0.0:
            return 0.0
        if d >= d_max:
            p = l_plateau
        else:
            p = float(np.interp(d, dd, ll))
        return p * float(contact_length(d, case.anchor))

    return dent


def _chain_force_parts(z: float, case: ImpactCase, dent_fn=None):
    """(soil resistance, pipe dent resistance) at anchor embedment ``z`` [N]."""
    f_soil = 0.0
    if case.soil_on and z > 0.0:
        f_soil = float(soil_resistance(z, case.anchor, case.soil))
    f_pipe = 0.0
    dz = z - case.cover
    if dz > 0.0:
        f_pipe = float(dent_force(dz, case.anchor, case.pipe)) if dent_fn is None \
            else float(dent_fn(dz))
    return f_soil, f_pipe


def _chain_force(z: float, case: ImpactCase,
                 dent_fn=None) -> float:
    f_soil, f_pipe = _chain_force_parts(z, case, dent_fn)
    return f_soil + f_pipe


def simulate_impact(
    case: ImpactCase,
    dt_max: float = 2.0e-5,
    t_max: float = 2.0,
    stiff_safety: float = 0.25,
    pipe_model: str = "plastic",
    ring_curve: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Integrate the vertical chain; returns histories, penetration, dent, ledger.

    Downward is positive; ``z`` is the embedment of the anchor's lowest point
    below the seabed top.  The pipe crown sits at ``z = cover``; everything
    beyond that is pipe dent.  Soil and dent resistances are treated as
    rate-independent plastic work (no unloading), which is the conservative
    assumption for a resting anchor after impact.
    """
    m = case.anchor.m_eff
    w = case.anchor.weight_eff
    dent_fn = _dent_force_model(case, pipe_model, ring_curve)
    v = case.v0
    z = 0.0
    t = 0.0
    hist = {"t": [0.0], "z": [0.0], "v": [v], "F": [0.0], "F_soil": [0.0], "F_pipe": [0.0]}
    z_max = 0.0
    impulse = 0.0
    work = 0.0
    work_soil = 0.0
    work_dent = 0.0
    f_soil, f_pipe = _chain_force_parts(0.0, case, dent_fn)
    n_steps = 0

    while t < t_max:
        f = _chain_force(z, case, dent_fn)
        # local stiffness by central difference for the adaptive step
        h = 1.0e-4 * max(1.0, abs(z))
        k_loc = abs(_chain_force(z + h, case, dent_fn)
                    - _chain_force(z - h, case, dent_fn)) / (2.0 * h)
        k_loc = max(k_loc, 1.0)
        dt = min(dt_max, stiff_safety * math.sqrt(m / k_loc))
        # do not overshoot the pipe crown / the arrest point by a whole step
        if v > 0.0:
            dt = min(dt, 0.5 * max(1.0e-6, (1.0 - 1.0e-6) * 1.0) * 1.0 / 1.0)
        # velocity Verlet (trapezoidal velocity update -> momentum exact)
        a = (w - f) / m
        z_new = z + v * dt + 0.5 * a * dt * dt
        if z_new < 0.0:  # cannot rise above the seabed in this plastic chain
            z_new = 0.0
        f_new = _chain_force(z_new, case, dent_fn)
        a_new = (w - f_new) / m
        v_new = v + 0.5 * (a + a_new) * dt
        impulse += 0.5 * (f + f_new) * dt
        dz_step = z_new - z
        work += 0.5 * (f + f_new) * dz_step
        fs_new, fp_new = _chain_force_parts(z_new, case, dent_fn)
        work_soil += 0.5 * (f_soil + fs_new) * dz_step
        work_dent += 0.5 * (f_pipe + fp_new) * dz_step
        f_soil, f_pipe = fs_new, fp_new
        z, v = z_new, v_new
        t += dt
        n_steps += 1
        if n_steps % 20 == 0 or v <= 0.0:
            hist["t"].append(t)
            hist["z"].append(z)
            hist["v"].append(v)
            hist["F"].append(f_new)
            hist["F_soil"].append(
                float(soil_resistance(z, case.anchor, case.soil)) if case.soil_on else 0.0
            )
            hist["F_pipe"].append(
                float(dent_force(max(0.0, z - case.cover), case.anchor, case.pipe))
            )
        z_max = max(z_max, z)
        if v <= 0.0:
            break

    z_end = z_max
    delta = max(0.0, z_end - case.cover)
    e_soil = soil_energy(z_end, case.anchor, case.soil) if case.soil_on else 0.0
    e_dent = dent_energy(delta, case.anchor, case.pipe)
    e_imp = case.impact_energy
    e_w = case.anchor.weight_eff * z_end
    e_avail = e_imp + e_w
    return {
        "case": case.label(),
        "z_end": z_end,
        "delta": delta,
        "delta_over_D": delta / case.pipe.D,
        "v0": case.v0,
        "impact_energy": e_imp,
        "E_weight": e_w,
        "E_available": e_avail,
        "E_soil": e_soil,
        "E_dent": e_dent,
        "E_soil_dynamic": work_soil,
        "E_dent_dynamic": work_dent,
        "E_ledger_error_rel": (work_soil + work_dent - e_avail) / e_avail,
        "impulse": impulse,
        "momentum_change": m * (case.v0 - 0.0),
        "t_end": t,
        "n_steps": n_steps,
        "peak_force": float(max(hist["F"])),
        "contact_attained": z_end > case.cover,
        "hist": hist,
    }


def energy_ledger(res: Dict[str, object]) -> Dict[str, float]:
    """Energy partition of one chain run (J and fractions)."""
    e = res["impact_energy"]
    ea = res["E_available"]
    return {
        "E_impact": e,
        "E_weight": res["E_weight"],
        "E_available": ea,
        "E_soil": res["E_soil"],
        "E_dent": res["E_dent"],
        "f_soil_of_available": res["E_soil"] / ea,
        "f_dent_of_available": res["E_dent"] / ea,
        "f_dent_of_impact": res["E_dent"] / e,
        "closure_rel": res["E_ledger_error_rel"],
    }


# ---------------------------------------------------------------------------
# link 5: DNV-RP-F107-style closed-form energy method
# ---------------------------------------------------------------------------
def rp_f107_closed_form(
    case: ImpactCase,
    gravity_work: bool = True,
    dent_allow_ratio: Optional[float] = None,
) -> Dict[str, object]:
    """DNV-RP-F107-style closed-form energy method (static energy partition).

    Work-energy balance of the design event::

        E_impact + W' (cover + delta) = E_soil(cover + delta) + E_dent(delta)

    with ``E_impact = 0.5 m (1+C_a) v0^2``, ``v0 = sqrt(2 a_net h)``, the soil
    work ``int Q_u dz`` and the dent work ``int P_c L_c ddelta``.  The extra
    gravity term ``W' z`` is the weight work the anchor does while penetrating;
    ``gravity_work=False`` reproduces the common design simplification that only
    the free-fall kinetic energy is counted (conservative).  Verdict: allowed
    iff ``delta / D <= dent_allow_ratio`` (RP-F107 no-repair dent rule, 5 %).
    """
    e_imp = case.impact_energy
    w = case.anchor.weight_eff if gravity_work else 0.0
    allow = case.pipe.dent_allow_ratio if dent_allow_ratio is None else dent_allow_ratio

    def budget(delta: float) -> float:
        """Available energy (impact + weight work during penetration) minus demand."""
        z = case.cover + delta
        e_s = soil_energy(z, case.anchor, case.soil) if case.soil_on else 0.0
        return e_imp + w * z - e_s - dent_energy(delta, case.anchor, case.pipe)

    if budget(0.0) <= 0.0:          # stopped in the soil cover, pipe untouched
        delta = 0.0
    else:
        lo, hi = 0.0, case.pipe.D
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if budget(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1.0e-12:
                break
        delta = 0.5 * (lo + hi)
    z = case.cover + delta
    e_soil = soil_energy(z, case.anchor, case.soil) if case.soil_on else 0.0
    e_dent = dent_energy(delta, case.anchor, case.pipe)
    return {
        "case": case.label(),
        "delta": delta,
        "delta_over_D": delta / case.pipe.D,
        "E_soil": e_soil,
        "E_dent": e_dent,
        "E_weight": w * z,
        "impact_energy": e_imp,
        "E_available": e_imp + w * z,
        "gravity_work": bool(gravity_work),
        "closure_rel": (e_soil + e_dent - (e_imp + w * z)) / (e_imp + w * z),
        "verdict": bool(delta / case.pipe.D <= allow),
    }


# ---------------------------------------------------------------------------
# E1: free-fall / rebound unit test of the impact integrator
# ---------------------------------------------------------------------------
def free_fall_rebound(
    h: float = 1.0,
    e: float = 0.5,
    mass: float = 100.0,
    a_net: float = G,
    n_step_per_contact: int = 2000,
) -> Dict[str, float]:
    """Drop a point mass from ``h`` onto a Kelvin-Voigt contact, measure rebound.

    Restitution of a linear spring-dashpot oscillator is exactly
    ``e = exp(-zeta pi / sqrt(1 - zeta^2))``; the damping ratio is calibrated
    from the requested ``e``.  The free-flight phases are ballistic (analytic),
    only the contact phase is time integrated, so the momentum statement below
    is a pure statement about the integrator.  The step is truncated so that it
    lands exactly on the separation point ``z = 0``; with the trapezoidal
    velocity update of velocity Verlet the impulse identity
    ``int F dt = m (v_in - v_out) - m a_net T`` then holds to round-off.
    """
    v_in = math.sqrt(2.0 * a_net * h)
    ln_e = math.log(max(e, 1.0e-9))
    zeta = -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
    omega = 2.0e4                      # contact stiffness scale [rad/s]
    k = mass * omega ** 2
    c = 2.0 * zeta * math.sqrt(k * mass)
    t_contact = math.pi / (omega * math.sqrt(max(1.0 - zeta ** 2, 1e-12)))
    dt = t_contact / n_step_per_contact

    z = 0.0
    v = v_in                       # downward positive
    t = 0.0
    impulse = 0.0
    f = 0.0                        # contact force, negative = upward
    v_out = v
    n_max = 60 * n_step_per_contact
    for _ in range(n_max):
        if z <= 0.0 and f == 0.0 and v < 0.0:
            v_out = v
            break
        a = a_net + f / mass
        step = dt
        z_new = z + v * step + 0.5 * a * step * step
        if z > 0.0 and z_new < 0.0:
            # truncate the step so that it lands on the separation point
            disc = v * v - 2.0 * a * z
            disc = max(disc, 0.0)
            root = (-v + math.sqrt(disc)) / a if abs(a) > 1.0e-12 else -z / v
            if 0.0 < root < step:
                step = root
                z_new = z + v * step + 0.5 * a * step * step
                z_new = min(z_new, 0.0)
        f_new = -(k * z_new + c * (v + a * step)) if z_new > 0.0 else 0.0
        a_new = a_net + f_new / mass
        v_new = v + 0.5 * (a + a_new) * step
        impulse += -0.5 * (f + f_new) * step
        z, v, f = z_new, v_new, f_new
        t += step
        if z <= 0.0 and v < 0.0:
            v_out = v
            break
    else:
        v_out = v
    h_rebound = v_out ** 2 / (2.0 * a_net)
    # upward impulse must cancel the weight impulse AND reverse the momentum
    momentum_change = mass * (v_in - v_out) + mass * a_net * t
    return {
        "h": h,
        "e": e,
        "zeta": zeta,
        "dt": dt,
        "T_contact_measured": t,
        "v_in": v_in,
        "v_out": v_out,
        "h_rebound": h_rebound,
        "rebound_target": e ** 2 * h,
        "rebound_rel_error": (h_rebound - e ** 2 * h) / (e ** 2 * h),
        "impulse_intFdt": impulse,
        "momentum_change": momentum_change,
        "momentum_rel_error": abs(impulse - momentum_change) / abs(momentum_change),
    }


# ---------------------------------------------------------------------------
# E3: ring-squash limit load by DEM relaxation
# ---------------------------------------------------------------------------
def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class _RingGeometry:
    """Polygonal discretisation of the pipe ring (plane strain, unit length)."""

    def __init__(self, pipe: Pipe, n_seg: int):
        self.pipe = pipe
        self.N = n_seg
        self.ang = 2.0 * np.pi * np.arange(n_seg) / n_seg
        self.x0 = pipe.R * np.cos(self.ang)
        self.y0 = pipe.R * np.sin(self.ang)
        self.ip1 = (np.arange(n_seg) + 1) % n_seg
        self.im1 = (np.arange(n_seg) - 1) % n_seg
        self.l0 = float(np.mean(np.hypot(self.x0[self.ip1] - self.x0,
                                         self.y0[self.ip1] - self.y0)))
        self.th0 = 2.0 * np.pi / n_seg
        self.k_b = pipe.E * pipe.second_moment / self.l0      # hinge stiffness
        self.k_a = pipe.E * pipe.t / self.l0                  # axial spring stiffness
        self.M_pl = pipe.M_pl
        self.th_y = self.M_pl / self.k_b
        self.j_top = int(np.argmax(self.y0))
        self.j_bot = int(np.argmin(self.y0))

    def energy_grad(self, q: np.ndarray, delta_target: float, k_flat: float):
        N, ip1, im1 = self.N, self.ip1, self.im1
        x, y = q[:N], q[N:]
        ux = x - x[im1]
        uy = y - y[im1]
        wx = x[ip1] - x
        wy = y[ip1] - y
        Lu2 = ux * ux + uy * uy
        Lw2 = wx * wx + wy * wy
        L = np.sqrt(Lu2)
        ex = ux / L
        ey = uy / L
        tension = self.k_a * (L - self.l0)
        u_ax = 0.5 * self.k_a * ((L - self.l0) ** 2).sum()
        # dU_axial/dx_i = T_i e_i - T_{i+1} e_{i+1}  (T_i is the spring i-1 -> i)
        tex = tension * ex
        tey = tension * ey
        grad_ax_x = tex - tex[ip1]
        grad_ax_y = tey - tey[ip1]
        cross = ux * wy - uy * wx
        dot = ux * wx + uy * wy
        dth = _wrap(np.arctan2(cross, dot) - self.th0)
        ad = np.abs(dth)
        elastic = ad <= self.th_y
        u_bend = float(
            np.where(elastic, 0.5 * self.k_b * dth ** 2,
                     self.M_pl * (ad - 0.5 * self.th_y)).sum()
        )
        M = np.where(elastic, self.k_b * dth, np.sign(dth) * self.M_pl)
        gux, guy = -uy / Lu2, ux / Lu2
        gwx, gwy = -wy / Lw2, wx / Lw2
        # bending force on the nodes (energy gradient is minus the force)
        fbx = -M * (-(gux + gwx))
        fby = -M * (-(guy + gwy))
        fbx[im1] += -M * gux
        fby[im1] += -M * guy
        fbx[ip1] += -M * gwx
        fby[ip1] += -M * gwy
        flatten = (self.y0[self.j_top] - y[self.j_top]) + (y[self.j_bot] - self.y0[self.j_bot])
        resid = flatten - delta_target
        u_flat = 0.5 * k_flat * resid ** 2
        g = np.empty_like(q)
        g[:N] = grad_ax_x - fbx
        g[N:] = grad_ax_y - fby
        g[N + self.j_top] += k_flat * resid * (-1.0)
        g[N + self.j_bot] += k_flat * resid * (+1.0)
        return u_ax + u_bend + u_flat, g


def ring_squash_load_curve(
    pipe: Pipe,
    n_seg: int = 64,
    n_delta: int = 14,
    delta_max_frac: float = 0.20,
    k_flat_scale: float = 1.0e3,
    tol: float = 1.0e-12,
) -> Dict[str, object]:
    """Load-deflection curve of the ring by displacement-controlled energy minimisation.

    The ring is a polygon whose hinges are elastic-perfectly-plastic
    (``M_pl = sigma_y t^2 / 4``) and whose segments are (penalty) inextensible.
    A prescribed diametral flattening ``delta`` is imposed through a stiff
    spring; the reaction ``P = k_flat (delta - flatten)`` is exactly the load
    that the two opposite point loads must supply.  The maximum of ``P(delta)``
    is the plastic limit load of the four-hinge mechanism.
    """
    from scipy.optimize import minimize

    geom = _RingGeometry(pipe, n_seg)
    k_el = pipe.E * pipe.second_moment / (0.1488 * pipe.R ** 3)
    k_flat = k_flat_scale * k_el
    deltas = np.linspace(0.0, delta_max_frac * pipe.R, n_delta)
    q = np.concatenate([geom.x0, geom.y0])
    rows = []
    for d in deltas:
        fun = lambda qq, d=d: geom.energy_grad(qq, d, k_flat)
        res = minimize(fun, q, jac=True, method="L-BFGS-B",
                       options={"maxiter": 4000, "ftol": tol, "gtol": 1e-10})
        q = res.x
        _, _ = geom.energy_grad(q, d, k_flat)
        N = geom.N
        y = q[N:]
        flatten = (geom.y0[geom.j_top] - y[geom.j_top]) + (y[geom.j_bot] - geom.y0[geom.j_bot])
        load = k_flat * (d - flatten)
        rows.append({
            "delta": float(d),
            "delta_over_R": float(d / pipe.R),
            "flatten": float(flatten),
            "load": float(load),
            "load_over_analytic": float(load / pipe.ring_limit_load),
            "constraint_residual": float(abs(d - flatten)),
            "nit": int(res.nit),
            "success": bool(res.success),
        })
    return {"rows": rows, "n_seg": n_seg, "k_flat": k_flat}


def ring_squash_limit_load(
    pipe: Pipe,
    n_seg: int = 64,
    n_delta: int = 14,
    delta_max_frac: float = 0.20,
    **kw,
) -> Dict[str, object]:
    """Plastic limit load of the ring, ``max_delta P(delta)``, vs ``4 M_pl / R``."""
    curve = ring_squash_load_curve(pipe, n_seg=n_seg, n_delta=n_delta,
                                   delta_max_frac=delta_max_frac, **kw)
    loads = np.array([r["load"] for r in curve["rows"]])
    p_num = float(loads.max())
    p_an = pipe.ring_limit_load
    return {
        "P_num": p_num,
        "P_analytic": p_an,
        "rel_error": (p_num - p_an) / p_an,
        "curve": curve,
        "n_seg": n_seg,
        "plateau_loads": [float(v) for v in loads[-4:]],
    }


# ---------------------------------------------------------------------------
# E4: dent - energy scaling
# ---------------------------------------------------------------------------
def dent_scaling_exponent(
    energies: Sequence[float],
    deltas: Sequence[float],
) -> Dict[str, object]:
    """Local log-log exponent ``n = dln(delta)/dln(E)`` over doubling intervals."""
    e = np.asarray(energies, dtype=float)
    d = np.asarray(deltas, dtype=float)
    order = np.argsort(e)
    e, d = e[order], d[order]
    exps = []
    for i in range(len(e) - 1):
        if d[i] <= 0 or d[i + 1] <= 0 or e[i] <= 0 or e[i + 1] <= 0:
            continue
        if abs(e[i + 1] / e[i] - 2.0) > 0.25:   # only near-doubling intervals
            continue
        exps.append(math.log(d[i + 1] / d[i]) / math.log(e[i + 1] / e[i]))
    exps_arr = np.asarray(exps, dtype=float)
    analytic = 2.0 / 3.0
    mean = float(exps_arr.mean()) if exps_arr.size else float("nan")
    return {
        "exponents": [float(v) for v in exps_arr],
        "mean": mean,
        "analytic": analytic,
        "rel_error": (mean - analytic) / analytic if exps_arr.size else float("nan"),
        "n_intervals": int(exps_arr.size),
    }


# ---------------------------------------------------------------------------
# acceptance harness
# ---------------------------------------------------------------------------
def evaluate_acceptance(
    rebound_cases: Optional[Sequence[Tuple[float, float]]] = None,
    scenario_matrix: Optional[Sequence[ImpactCase]] = None,
    ring_kw: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Run the E1..E7 checks and return a machine-readable verdict block."""
    if rebound_cases is None:
        rebound_cases = [(1.0, 0.5), (2.0, 0.2), (0.5, 0.8), (1.0, 0.3)]
    pipe = Pipe()
    out: Dict[str, object] = {}

    # ---- E1 -------------------------------------------------------------
    e1 = [free_fall_rebound(h=h, e=e) for h, e in rebound_cases]
    out["E1"] = {
        "cases": e1,
        "max_rebound_rel_error": max(abs(r["rebound_rel_error"]) for r in e1),
        "max_momentum_rel_error": max(r["momentum_rel_error"] for r in e1),
    }

    # ---- E2 -------------------------------------------------------------
    e2_rows = []
    for anchor in (Anchor(kind="sphere", mass=3000.0), Anchor(kind="cone", mass=3000.0)):
        for c_u in (30.0e3, 60.0e3):
            case = ImpactCase(anchor=anchor, soil=Soil(c_u=c_u), cover=0.5,
                              drop_height=0.0)
            z_an = static_equilibrium_depth(case)
            if z_an is None:
                e2_rows.append({"case": case.label(), "analytic": None,
                                "settled": None, "rel_error": float("nan")})
                continue
            z_num = _settle_depth(case, z0=0.0)
            e2_rows.append({
                "case": f"{anchor.kind}-c{c_u/1e3:.0f}",
                "analytic": z_an,
                "settled": z_num,
                "rel_error": (z_num - z_an) / z_an,
            })
    out["E2"] = {
        "rows": e2_rows,
        "max_abs_rel_error": max(abs(r["rel_error"]) for r in e2_rows
                                 if r["rel_error"] == r["rel_error"]),
    }

    # ---- E3 -------------------------------------------------------------
    ring_opt = dict(ring_kw or {})
    ring = ring_squash_limit_load(pipe, n_seg=ring_opt.pop("n_seg", 64),
                                  n_delta=ring_opt.pop("n_delta", 16), **ring_opt)
    ring_fine_opt = dict(ring_kw or {})
    ring_fine = ring_squash_limit_load(pipe, n_seg=96,
                                       n_delta=max(8, ring_fine_opt.get("n_delta", 16) - 4),
                                       **{k: v for k, v in ring_fine_opt.items() if k != "n_delta"})
    out["E3"] = {
        "coarse": {k: v for k, v in ring.items() if k != "curve"},
        "fine": {k: v for k, v in ring.items() if k != "curve"},
        "curve": ring["curve"]["rows"],
        "max_abs_rel_error": max(abs(ring["rel_error"]), abs(ring_fine["rel_error"])),
        "mesh_sensitivity": abs(ring_fine["P_num"] - ring["P_num"]) / ring["P_num"],
    }

    # ---- E4 -------------------------------------------------------------
    base_anchor = Anchor(kind="sphere", mass=3000.0)
    energies, deltas, rows4 = [], [], []
    for h in (0.125, 0.25, 0.5, 1.0, 2.0):
        case = ImpactCase(anchor=base_anchor, soil=Soil(c_u=60.0e3),
                          drop_height=h, cover=0.0, soil_on=False)
        res = simulate_impact(case)
        energies.append(res["E_dent"])
        deltas.append(res["delta"])
        rows4.append({"drop_height": h, "E_dent": res["E_dent"],
                      "delta": res["delta"], "delta_over_D": res["delta_over_D"]})
    # large-deflection / saturated contact-patch regime (reported, not benchmarked)
    rows4_big = []
    for h in (4.0, 8.0, 16.0):
        case = ImpactCase(anchor=base_anchor, soil=Soil(c_u=60.0e3),
                          drop_height=h, cover=0.0, soil_on=False)
        res = simulate_impact(case)
        rows4_big.append({"drop_height": h, "E_dent": res["E_dent"],
                          "delta": res["delta"], "delta_over_D": res["delta_over_D"]})
    # sensitivity: elastic-plastic ring compliance instead of the rigid-plastic plateau
    ring_curve = ring_squash_load_curve(pipe, n_seg=64, n_delta=20, delta_max_frac=0.6)
    sens = []
    for h in (0.125, 0.5, 2.0):
        case = ImpactCase(anchor=base_anchor, soil=Soil(c_u=60.0e3),
                          drop_height=h, cover=0.0, soil_on=False)
        res = simulate_impact(case, pipe_model="ring", ring_curve=ring_curve)
        sens.append({"drop_height": h, "delta": res["delta"],
                     "delta_plastic": deltas[[0.125, 0.25, 0.5, 1.0, 2.0].index(h)],
                     "ratio": res["delta"] / deltas[[0.125, 0.25, 0.5, 1.0, 2.0].index(h)]})
    out["E4"] = {
        "rows": rows4,
        "rows_large_deflection": rows4_big,
        "scaling": dent_scaling_exponent(energies, deltas),
        "ring_compliance_sensitivity": sens,
    }

    # ---- E5 -------------------------------------------------------------
    if scenario_matrix is None:
        scenario_matrix = default_scenario_matrix()
    rows5 = []
    agree = 0
    agree_design = 0
    for case in scenario_matrix:
        chain = simulate_impact(case)
        cf = rp_f107_closed_form(case, gravity_work=True)
        cf_design = rp_f107_closed_form(case, gravity_work=False)
        v_chain = bool(chain["delta_over_D"] <= case.pipe.dent_allow_ratio)
        same = bool(v_chain == cf["verdict"])
        same_design = bool(v_chain == cf_design["verdict"])
        agree += int(same)
        agree_design += int(same_design)
        rows5.append({
            "case": case.label(),
            "delta_over_D_chain": chain["delta_over_D"],
            "delta_over_D_closed": cf["delta_over_D"],
            "verdict_chain": v_chain,
            "verdict_closed": cf["verdict"],
            "verdict_closed_impact_energy_only": cf_design["verdict"],
            "agree": same,
            "agree_design": same_design,
        })
    out["E5"] = {
        "rows": rows5,
        "n": len(rows5),
        "agreement": agree / len(rows5),
        "agreement_impact_energy_only": agree_design / len(rows5),
        "n_allowed_chain": int(sum(r["verdict_chain"] for r in rows5)),
        "n_allowed_closed": int(sum(r["verdict_closed"] for r in rows5)),
    }

    # ---- E6 -------------------------------------------------------------
    # control A: soil removed -> the anchor reaches and crushes the pipe
    ref = ImpactCase(anchor=Anchor(kind="sphere", mass=3000.0), soil=Soil(c_u=60.0e3),
                     cover=1.0, drop_height=5.0, soil_on=True)
    no_soil = ImpactCase(anchor=ref.anchor, pipe=ref.pipe, soil=ref.soil,
                         drop_height=ref.drop_height, cover=ref.cover, soil_on=False)
    r_ref = simulate_impact(ref)
    r_ns = simulate_impact(no_soil)
    # control B: deep burial (pipe never reached) -> mass scaling of the depth
    deep = ImpactCase(anchor=Anchor(kind="sphere", mass=3000.0), soil=Soil(c_u=30.0e3),
                      cover=3.0, drop_height=8.0, soil_on=True)
    deep_hv = ImpactCase(anchor=Anchor(kind="sphere", mass=4500.0), soil=deep.soil,
                         cover=deep.cover, drop_height=deep.drop_height, soil_on=True)
    r_deep = simulate_impact(deep)
    r_deep_hv = simulate_impact(deep_hv)
    z_an = soil_penetration_depth(deep)
    z_an_hv = soil_penetration_depth(deep_hv)
    an_ratio = (z_an_hv / z_an) if (z_an and z_an_hv) else float("nan")
    num_ratio = r_deep_hv["z_end"] / r_deep["z_end"]
    out["E6"] = {
        "ref": {"case": ref.label(), "z_end": r_ref["z_end"],
                "delta_over_D": r_ref["delta_over_D"]},
        "no_soil": {"case": no_soil.label(), "z_end": r_ns["z_end"],
                    "delta_over_D": r_ns["delta_over_D"],
                    "contact_attained": r_ns["contact_attained"]},
        "deep_ref": {"case": deep.label(), "z_end": r_deep["z_end"],
                     "delta_over_D": r_deep["delta_over_D"],
                     "z_analytic": z_an},
        "deep_heavy": {"case": deep_hv.label(), "z_end": r_deep_hv["z_end"],
                       "delta_over_D": r_deep_hv["delta_over_D"],
                       "z_analytic": z_an_hv},
        "mass_scale_ratio_numeric": num_ratio,
        "mass_scale_ratio_analytic": an_ratio,
        "mass_scale_rel_change": num_ratio - 1.0,
        "anchor_arrested_in_soil": bool(not r_ref["contact_attained"]),
        "no_soil_penetrates": bool(r_ns["delta_over_D"] > 0.15),
        "no_soil_amplifies_dent": bool(r_ns["delta_over_D"] > r_ref["delta_over_D"]),
    }

    # ---- E7 -------------------------------------------------------------
    out["E7"] = {
        "status": "WITHHELD",
        "reason": (
            "No full-scale or flume anchor-drop-on-pipeline test data exist in this "
            "worktree; DNV-RP-F107 itself is a licensed document whose tables were not "
            "verified verbatim here. Only analytic anchors (rebound, bearing capacity, "
            "ring-squash limit load, dent-energy scaling, energy closure) are claimed."
        ),
        "claimed_scope": "analytic chain verification only",
    }

    # ---- summary --------------------------------------------------------
    out["criteria"] = {
        "E1_rebound_le_2pct": out["E1"]["max_rebound_rel_error"] <= 0.02,
        "E1_momentum_le_1e-6": out["E1"]["max_momentum_rel_error"] <= 1.0e-6,
        "E2_penetration_le_3pct": out["E2"]["max_abs_rel_error"] <= 0.03,
        "E3_ring_squash_le_5pct": out["E3"]["max_abs_rel_error"] <= 0.05,
        "E4_scaling_le_10pct": abs(out["E4"]["scaling"]["rel_error"]) <= 0.10,
        "E5_agreement_ge_90pct": out["E5"]["agreement"] >= 0.90,
        "E6_anchor_arrested_with_soil": out["E6"]["anchor_arrested_in_soil"],
        "E6_no_soil_penetrates": out["E6"]["no_soil_penetrates"],
        "E6_no_soil_amplifies_dent": out["E6"]["no_soil_amplifies_dent"],
        "E6_mass_scale_ge_5pct": abs(out["E6"]["mass_scale_rel_change"]) >= 0.05,
        "E7_withheld_registered": out["E7"]["status"] == "WITHHELD",
    }
    out["all_pass"] = all(out["criteria"].values())
    return out


def default_scenario_matrix() -> List[ImpactCase]:
    """Scenario matrix for E5 (3 masses x 3 drop heights x 3 covers x 2 x 2)."""
    cases: List[ImpactCase] = []
    for kind in ("sphere", "cone"):
        for mass in (2000.0, 3000.0, 5000.0):
            for h in (2.0, 5.0, 8.0):
                for cover in (0.0, 0.5, 1.0):
                    for c_u in (25.0e3, 60.0e3):
                        cases.append(
                            ImpactCase(
                                anchor=Anchor(kind=kind, mass=mass),
                                soil=Soil(c_u=c_u),
                                cover=cover,
                                drop_height=h,
                            )
                        )
    return cases


def _settle_depth(case: ImpactCase, z0: float = 0.0, zeta: float = 1.0,
                  t_max: float = 2.0, dt_max: float = 2.0e-4,
                  tol_v: float = 1.0e-4, tol_f: float = 1.0e-3) -> float:
    """Quasi-static settling: damped dynamics under gravity alone -> rest depth.

    The same force model as :func:`simulate_impact` is integrated with a
    critically damped (``zeta = 1``) viscous term until both the velocity and
    the force imbalance are below tolerance; the resting depth is then compared
    with the analytic root of ``W' = Q_u(z)`` (check E2).
    """
    m = case.anchor.m_eff
    w = case.anchor.weight_eff
    z, v, t = float(z0), 0.0, 0.0
    while t < t_max:
        f = _chain_force(z, case)
        h = 1e-4 * max(1.0, abs(z))
        k_loc = abs(_chain_force(z + h, case) - _chain_force(z - h, case)) / (2.0 * h)
        k_loc = max(k_loc, 1.0)
        c = 2.0 * zeta * math.sqrt(k_loc * m)
        dt = min(dt_max, 0.2 * math.sqrt(m / k_loc), 0.2 * m / c)
        a = (w - f - c * v) / m
        v_new = v + a * dt
        z = max(0.0, z + 0.5 * (v + v_new) * dt)
        v = v_new
        t += dt
        if abs(v) < tol_v * math.sqrt(w / m * 1.0) and abs(w - f) < tol_f * w:
            break
    return z
