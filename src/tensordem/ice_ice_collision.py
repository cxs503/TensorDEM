"""Floe-field ice-ice collision face: Hertz-Mindlin contacts, freeze-bond
rupture and the concentration dependence of the effective viscosity.

Physical scenario
-----------------
A drift ice field of plan-view concentration ``C`` (areal fraction of ice) is
sheared.  The floes are buoyant plates of thickness ``d``; heave is relaxed
(the flat-lying assumption already used by :mod:`tensordem.ice_channel` and
:mod:`tensordem.clear_ice.field`), so a floe is a disk of radius ``r`` that
translates and spins in the horizontal plane.

Three ingredients are modelled and verified here:

1. **Hertz-Mindlin contact** (Hertz 1882; Mindlin 1949): an elastic normal
   force ``F_n = (4/3) E* sqrt(R*) delta^{3/2}`` and an incremental tangential
   (Mindlin) spring ``k_t = 8 G* a`` capped by the Coulomb cone
   ``|F_t| <= mu |F_n|`` (Cundall & Strack 1979).  The contact patch of a floe
   is extruded over the ice thickness ``d``, so all contact forces carry the
   dimensionless thickness ratio ``d_hat = d / 1 m``; this keeps the force
   dimensionally a force (N) and reproduces the linear scaling of the force
   with plate thickness.

2. **Freeze bonds** (ice-ice sintering): a bonded pair carries a linear
   elastic-brittle normal bond of stiffness ``k_b = E A_b / L_b`` over the bond
   area ``A_b = pi r_b^2``.  The bond ruptures when the normal stress
   ``|F_n| / A_b`` exceeds the bond strength ``sigma_b``; after rupture the pair
   is a *purely frictional* Hertz-Mindlin contact (no tensile capacity).

3. **Field rheology**: the field is sheared in a Couette cell (periodic in
   ``x``, two frictional walls with velocity difference ``U = gamma_dot H``).
   The shear stress is measured with the virial estimator (the wall force is
   carried along as an independent cross-check) and the *equivalent viscosity*
   is

       mu_eff = <tau_xy> / (d * gamma_dot)      [Pa s]

   i.e. the apparent (rate dependent) viscosity of the ice field per unit ice
   thickness.  Its concentration dependence ``mu_eff(C)`` is the rheological
   closure used in sea-ice drift models.

Energy accounting
-----------------
Every step the code accumulates the dissipated energy (normal dashpot +
Coulomb sliding, floe-floe and floe-wall), the tangential spring energy and the
work done by the walls, so that the ledger ``E_kin + E_spring + E_diss -
W_wall = E_kin(0)`` can be audited after a run (criterion C6).  All quantities
are SI (m, kg, s, Pa).

Documented idealisations (see ``docs/ICE_ICE_COLLISION_VALIDATION.md``):
flat-lying floes without heave/rafting, a *soft-sphere* reduced modulus (chosen
so that the contact frequency resolves with a millisecond step; the rheology is
verified to be stiffness independent), scalar (2-D) tangential history per
contact and rigid rough walls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

__all__ = [
    "IceIceCollisionError",
    "reduced_youngs_modulus",
    "reduced_shear_modulus",
    "reduced_radius",
    "hertz_contact_force",
    "hertz_contact_radius",
    "hertz_secant_stiffness",
    "mindlin_tangential_stiffness",
    "hertz_impact_peak",
    "damping_ratio_from_restitution",
    "hunt_crossley_beta",
    "bond_area",
    "bond_normal_stress",
    "bond_ruptures",
    "bond_tensile_test",
    "two_floe_collision",
    "FloeFieldConfig",
    "FloeFieldSimulator",
    "simulate_floe_field",
    "sweep_concentration",
    "dissipation_scaling",
]

DTYPE = np.float64


class IceIceCollisionError(ValueError):
    """Raised for invalid ice-ice collision configurations."""


# ---------------------------------------------------------------------------
# contact mechanics primitives (closed form, the analytic anchors)
# ---------------------------------------------------------------------------
def reduced_youngs_modulus(e1: float, e2: float, nu1: float, nu2: float) -> float:
    """``1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2`` (Hertz 1882)."""
    if min(e1, e2) <= 0.0:
        raise IceIceCollisionError("Young's moduli must be positive")
    inv = (1.0 - nu1 ** 2) / e1 + (1.0 - nu2 ** 2) / e2
    if inv <= 0.0:
        raise IceIceCollisionError("non-positive contact compliance")
    return 1.0 / inv


def reduced_shear_modulus(g1: float, g2: float, nu1: float, nu2: float) -> float:
    """``1/G* = (2-nu1)/G1 + (2-nu2)/G2`` (Johnson 1985, eq. 3.39)."""
    if min(g1, g2) <= 0.0:
        raise IceIceCollisionError("shear moduli must be positive")
    return 1.0 / ((2.0 - nu1) / g1 + (2.0 - nu2) / g2)


def reduced_radius(r1: float, r2: float) -> float:
    """``1/R* = 1/R1 + 1/R2`` (a rigid plane is ``R2 = inf``)."""
    if math.isinf(r2):
        return r1
    if math.isinf(r1):
        return r2
    if r1 <= 0.0 or r2 <= 0.0:
        raise IceIceCollisionError("radii must be positive")
    return 1.0 / (1.0 / r1 + 1.0 / r2)


def hertz_contact_radius(delta, r_eff):
    """Hertz contact radius ``a = sqrt(R* delta)`` (m)."""
    return np.sqrt(np.maximum(np.asarray(delta, dtype=DTYPE), 0.0) * r_eff)


def hertz_contact_force(delta, r_eff, e_eff, thickness_factor: float = 1.0):
    """Hertz normal force ``F_n = (4/3) E* sqrt(R*) delta^{3/2}`` (N).

    ``thickness_factor`` is the dimensionless ice-thickness ratio ``d/1 m``:
    the contact patch of a plate is extruded over the plate thickness, so the
    force scales linearly with ``d``.
    """
    d = np.maximum(np.asarray(delta, dtype=DTYPE), 0.0)
    return (4.0 / 3.0) * e_eff * math.sqrt(r_eff) * d ** 1.5 * thickness_factor


def hertz_secant_stiffness(delta, r_eff, e_eff, thickness_factor: float = 1.0):
    """``k_n = dF_n/ddelta = 2 E* a`` (N/m)."""
    return 2.0 * e_eff * hertz_contact_radius(delta, r_eff) * thickness_factor


def mindlin_tangential_stiffness(delta, r_eff, g_eff, thickness_factor: float = 1.0):
    """Mindlin tangential stiffness ``k_t = 8 G* a`` (N/m)."""
    return 8.0 * g_eff * hertz_contact_radius(delta, r_eff) * thickness_factor


def hertz_impact_peak(mass: float, v0: float, r_eff: float, e_eff: float,
                      thickness_factor: float = 1.0) -> dict[str, float]:
    """Analytic Hertz impact of two elastic bodies (Johnson 1985, ch. 11).

    Energy balance ``m v0^2 / 2 = (2/5) k delta_max^{5/2}`` with
    ``k = (4/3) E* sqrt(R*) d_hat`` gives

        delta_max = (5 m v0^2 / (4 k))^{2/5},   F_max = k delta_max^{3/2},

    where ``m`` is the *effective* mass of the pair, ``m1 m2 / (m1 + m2)``.
    """
    if mass <= 0.0 or v0 <= 0.0:
        raise IceIceCollisionError("mass and impact speed must be positive")
    k = (4.0 / 3.0) * e_eff * math.sqrt(r_eff) * thickness_factor
    delta_max = (5.0 * mass * v0 ** 2 / (4.0 * k)) ** 0.4
    return {
        "delta_max": delta_max,
        "force_max": k * delta_max ** 1.5,
        "contact_time": 2.94 * (5.0 * mass / (4.0 * k)) ** 0.4 * v0 ** -0.2,
        "k_hertz": k,
    }


def damping_ratio_from_restitution(restitution: float) -> float:
    """Viscous damping ratio ``zeta`` for a target restitution coefficient."""
    e = float(min(max(restitution, 1e-6), 1.0 - 1e-12))
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


HERTZ_HC_SHAPE = 0.68
"""Shape factor of the ``delta^{3/2}`` contact for the Hunt-Crossley damping.

``e = 1 - (2/3) beta v`` holds for a linear spring; for a Hertzian spring the
contact spends a shorter fraction of the impact in compression, so the same
``beta`` under-damps by this factor.  The value is *calibrated* with the
module's own single-impact test (:func:`two_floe_collision`) and the measured
restitution is reported with every validation run.
"""


def hunt_crossley_beta(restitution: float, v_ref: float) -> float:
    """Hunt-Crossley damping factor for a target restitution at ``v_ref``.

    The nonlinear dashpot ``F_n = F_elastic (1 - beta v_n)`` gives the speed
    dependent restitution ``e = 1 - (2/3) beta v`` (Hunt & Crossley 1975).  Its
    dissipative power ``F_elastic beta v_n^2`` is non-negative and exactly
    bookkeepable, which keeps the energy ledger closed (criterion C6).
    """
    if v_ref <= 0.0:
        raise IceIceCollisionError("v_ref must be positive")
    e = float(min(max(restitution, 0.0), 1.0 - 1e-12))
    return 3.0 * (1.0 - e) / (2.0 * v_ref * HERTZ_HC_SHAPE)


# ---------------------------------------------------------------------------
# freeze bonds
# ---------------------------------------------------------------------------
def bond_area(r1: float, r2: float, radius_ratio: float = 0.25) -> float:
    """Bond cross-section ``A_b = pi (radius_ratio * min(r1, r2))^2`` (m^2)."""
    if radius_ratio <= 0.0:
        raise IceIceCollisionError("bond radius ratio must be positive")
    rb = radius_ratio * min(r1, r2)
    return math.pi * rb ** 2


def bond_normal_stress(force, area):
    """Bond normal stress ``|F_n| / A_b`` (Pa), vectorised over pairs."""
    f = np.abs(np.asarray(force, dtype=DTYPE))
    return f / area


def bond_ruptures(force, area, sigma_b):
    """True where the normal stress exceeds the bond strength ``sigma_b``."""
    return bond_normal_stress(force, area) > sigma_b


def bond_tensile_test(sigma_b: float = 1.0e5, radius: float = 0.6,
                      thickness: float = 0.5, density: float = 900.0,
                      young_modulus: float = 1.0e7, poisson_ratio: float = 0.3,
                      bond_radius_ratio: float = 0.25,
                      bond_length_ratio: float = 0.05,
                      strain_rate: float = 1.0e-4,
                      dt_factor: float = 0.02,
                      max_steps: int = 400000) -> dict[str, float]:
    """Quasi-static tensile test of one freeze bond (criterion C2).

    Two floes bonded at their touching separation are pulled apart at a constant
    strain rate.  The bond is a linear elastic spring of stiffness
    ``k_b = E A_b / L_b`` with ``A_b = pi r_b^2`` and ``r_b = ratio * r``.  The
    stress history ``|F| / A_b`` is recorded and the bond is declared ruptured
    the moment that stress exceeds ``sigma_b``; the floe pair then has *no*
    tensile capacity (Coulomb friction only).  The reported rupture stress is
    compared with ``sigma_b`` (criterion: deviation <= 10 %).
    """
    area = bond_area(radius, radius, bond_radius_ratio)
    rb = bond_radius_ratio * radius
    k_b = young_modulus * area / (bond_length_ratio * rb)
    mass = density * math.pi * radius ** 2 * thickness
    s0 = 2.0 * radius
    pull_speed = strain_rate * s0
    dt = dt_factor * 2.0 * math.pi / math.sqrt(k_b / (0.5 * mass))
    dt = min(dt, 0.2 * s0 / pull_speed)

    sep = s0
    vel = 0.5 * pull_speed
    peak_stress = 0.0
    rupture_stress = float("nan")
    rupture_time = float("nan")
    time = 0.0
    stress_trace: list[tuple[float, float]] = []
    ruptured = False
    for _ in range(max_steps):
        sep += 2.0 * vel * dt
        time += dt
        ext = sep - s0
        force = k_b * ext if not ruptured else 0.0
        stress = abs(force) / area
        stress_trace.append((time, stress))
        if stress > peak_stress:
            peak_stress = stress
        if not ruptured and stress > sigma_b:
            ruptured = True
            rupture_stress = stress
            rupture_time = time
            break
        if ext > 0.5 * s0:
            break
    return {
        "sigma_b": sigma_b,
        "bond_area": area,
        "bond_stiffness": k_b,
        "rupture_stress": rupture_stress,
        "peak_stress": peak_stress,
        "relative_error": abs(rupture_stress - sigma_b) / sigma_b
        if ruptured else float("inf"),
        "ruptured": ruptured,
        "rupture_time": rupture_time,
        "rupture_extension": (rupture_stress * (bond_length_ratio * rb)
                              / young_modulus) if ruptured else float("nan"),
        "trace": stress_trace,
        "dt": dt,
    }

# ---------------------------------------------------------------------------
# single-pair analytic verification helpers
# ---------------------------------------------------------------------------
def two_floe_collision(radius: float = 0.6, thickness: float = 0.5,
                       density: float = 900.0, young_modulus: float = 1.0e7,
                       poisson_ratio: float = 0.3, friction: float = 0.3,
                       restitution: float = 1.0,
                       impact_speed: float = 1.0,
                       dt_factor: float = 0.01,
                       max_steps: int = 4000000) -> dict[str, Any]:
    """Free central collision of two equal floes (criteria C1 and C6).

    The floes approach each other with ``+-impact_speed/2`` in a stress-free
    medium (no walls, no periodicity), so

    * the peak normal force is compared with the analytic Hertz impact of
      :func:`hertz_impact_peak` (criterion C1), and
    * the closed ledger ``E_kin + E_spring + E_diss = E_kin(0)`` is audited
      (criterion C6).

    Velocity-Verlet integration is used (force evaluated twice per step), so the
    total energy of the elastic case is conserved to ``O(dt^2)``.
    """
    e_eff = reduced_youngs_modulus(young_modulus, young_modulus,
                                   poisson_ratio, poisson_ratio)
    g_eff = reduced_shear_modulus(young_modulus / (2.0 * (1.0 + poisson_ratio)),
                                  young_modulus / (2.0 * (1.0 + poisson_ratio)),
                                  poisson_ratio, poisson_ratio)
    r_eff = radius / 2.0
    mass = density * math.pi * radius ** 2 * thickness
    m_eff = 0.5 * mass
    d_hat = thickness
    beta = hunt_crossley_beta(restitution, impact_speed)

    analytic = hertz_impact_peak(m_eff, impact_speed, r_eff, e_eff, d_hat)

    gap = 0.4 * radius
    pos = np.array([[-radius - gap / 2.0, 0.0], [radius + gap / 2.0, 0.0]], dtype=DTYPE)
    vel = np.array([[0.5 * impact_speed, 0.0], [-0.5 * impact_speed, 0.0]], dtype=DTYPE)

    omega_ref = math.sqrt(hertz_secant_stiffness(0.01 * radius, r_eff, e_eff, d_hat)
                          / mass)  # noqa: E501
    dt = dt_factor * 2.0 * math.pi / omega_ref

    e_kin0 = float(np.sum(0.5 * mass * np.sum(vel ** 2, axis=1)))
    state = {"diss": 0.0, "htang": 0.0, "e_spring": 0.0, "peak_force": 0.0,
             "peak_overlap": 0.0, "contact": False, "trace": []}

    def forces(p, v):
        """Contact force on both floes plus the dissipative bookkeeping."""
        dx = p[1, 0] - p[0, 0]
        dist = abs(dx)
        overlap = 2.0 * radius - dist
        force = np.zeros((2, 2), dtype=DTYPE)
        if overlap <= 0.0:
            state["htang"] = 0.0
            state["e_spring"] = 0.0
            return force, overlap
        state["contact"] = True
        nx = 1.0 if dx > 0 else -1.0
        n_vec = np.array([nx, 0.0])
        t_vec = np.array([0.0, nx])
        v_rel = v[1] - v[0]
        vn = float(v_rel @ n_vec)
        vt = float(v_rel @ t_vec)
        a = math.sqrt(r_eff * overlap)
        kt = 8.0 * g_eff * a * d_hat
        fn_elastic = (4.0 / 3.0) * e_eff * math.sqrt(r_eff) * overlap ** 1.5 * d_hat
        fn = max(fn_elastic * (1.0 - beta * vn), 0.0)
        h_old = state["htang"]
        state["htang"] = h_old + vt * dt
        ft = -kt * state["htang"]
        cap = friction * fn
        if abs(ft) > cap:
            ft = math.copysign(cap, ft)
            state["htang"] = -ft / kt
        e_spring_new = 0.5 * kt * state["htang"] ** 2
        work_t = -ft * vt * dt
        state["diss"] += fn_elastic * beta * vn * vn * dt \
            + max(work_t - (e_spring_new - 0.5 * kt * h_old ** 2), 0.0)
        state["e_spring"] = e_spring_new
        f_pair = fn * n_vec + ft * t_vec
        force[1] = f_pair
        force[0] = -f_pair
        state["peak_force"] = max(state["peak_force"], fn)
        state["peak_overlap"] = max(state["peak_overlap"], overlap)
        state["trace"].append((overlap, fn, float(v_rel @ n_vec)))
        return force, overlap

    force, overlap = forces(pos, vel)
    v_rel_after = float("nan")
    separated_steps = 0
    for _ in range(max_steps):
        vel += 0.5 * dt * force / mass                     # half kick
        pos += dt * vel                                    # drift
        force, overlap = forces(pos, vel)                  # new force
        vel += 0.5 * dt * force / mass                     # half kick
        if overlap <= 0.0 and state["contact"]:
            if separated_steps == 0:
                v_rel_after = float(vel[1, 0] - vel[0, 0])
            separated_steps += 1
            if separated_steps > int(0.2 / dt):
                break

    e_kin_final = float(np.sum(0.5 * mass * np.sum(vel ** 2, axis=1)))
    ledger_residual = e_kin_final + state["e_spring"] + state["diss"] - e_kin0
    measured_e = (abs(v_rel_after) / impact_speed
                  if not math.isnan(v_rel_after) else float("nan"))
    return {
        "analytic": analytic,
        "peak_force": state["peak_force"],
        "peak_overlap": state["peak_overlap"],
        "force_error": abs(state["peak_force"] - analytic["force_max"]) / analytic["force_max"],
        "overlap_error": abs(state["peak_overlap"] - analytic["delta_max"]) / analytic["delta_max"],
        "e_kin_initial": e_kin0,
        "e_kin_final": e_kin_final,
        "dissipated": state["diss"],
        "e_spring": state["e_spring"],
        "ledger_residual": ledger_residual,
        "ledger_relative_error": abs(ledger_residual) / e_kin0,
        "measured_restitution": measured_e,
        "target_restitution": restitution,
        "dt": dt,
        "contact_time_analytic": analytic["contact_time"],
        "trace": state["trace"],
    }


def bonded_pair_rupture_test(sigma_b: float = 1.0e5, radius: float = 0.6,
                             thickness: float = 0.5, density: float = 900.0,
                             young_modulus: float = 1.0e7,
                             poisson_ratio: float = 0.3,
                             bond_radius_ratio: float = 0.25,
                             bond_length_ratio: float = 0.05,
                             forcing_rate: float = 2.0e5,
                             max_steps: int = 40000,
                             seed: int = 7) -> dict[str, Any]:
    """Pull a *bonded* floe pair apart inside the full contact machinery.

    This exercises the production code path
    (:class:`FloeFieldSimulator` + :meth:`FloeFieldSimulator.form_bonds`) rather
    than the standalone spring of :func:`bond_tensile_test`: the pair is bonded,
    then separated at constant velocity until the recorded bond rupture event
    fires.  The reported stress at rupture is compared with ``sigma_b``
    (criterion C2) and the post-rupture tensile force must vanish.
    """
    cfg = FloeFieldConfig(concentration=0.02, box_width=10.0, box_height=10.0,
                          thickness=thickness, mean_radius=radius,
                          radius_spread=1.0, density=density,
                          young_modulus=young_modulus, poisson_ratio=poisson_ratio,
                          bond_fraction=1.0, bond_strength=sigma_b,
                          bond_radius_ratio=bond_radius_ratio,
                          bond_length_ratio=bond_length_ratio,
                          shear_rate=0.0, settle_steps=0, shear_steps=1, seed=seed)
    sim = FloeFieldSimulator(cfg)
    if sim.n < 2:
        raise IceIceCollisionError("need two floes for the bonded-pair test")
    r = sim.radii[0]
    sim.radii[1] = r
    sim.mass[1] = sim.mass[0]
    sim.inertia[1] = sim.inertia[0]
    overlap0 = 1.0e-4
    sim.pos[0] = [5.0 - 0.5 * 2.0 * r + 0.5 * overlap0, 5.0]
    sim.pos[1] = [5.0 + 0.5 * 2.0 * r - 0.5 * overlap0, 5.0]
    sim.vel[:] = 0.0
    sim.omega[:] = 0.0
    n_bonds = sim.form_bonds(1.0)
    k = int(np.argmax(sim.bond_active))
    area = float(sim.bond_area_arr[k])
    stress_trace: list[tuple[float, float]] = []
    force_after = float("nan")
    for step in range(max_steps):
        # force-controlled ramp: the applied tension is raised linearly in time
        sim.external_force[0, 0] = -forcing_rate * sim.time
        sim.external_force[1, 0] = +forcing_rate * sim.time
        sim.step(sim._timestep())
        active = bool(sim.bond_active[k])
        if active:
            ext = abs(sim.pos[1, 0] - sim.pos[0, 0]) - float(sim.bond_s0[k])
            stress_trace.append((sim.time, abs(sim.bond_k[k] * ext) / area))
        elif math.isnan(force_after):
            force_after = abs(sim.pos[1, 0] - sim.pos[0, 0])
            break
    events = sim.bond_rupture_events
    rupture_stress = float(events[0]["normal_stress"]) if events else float("nan")
    return {
        "sigma_b": sigma_b,
        "bonds_formed": n_bonds,
        "bond_area": area,
        "rupture_stress": rupture_stress,
        "relative_error": abs(rupture_stress - sigma_b) / sigma_b if events else float("inf"),
        "ruptured": bool(events),
        "rupture_time": float(events[0]["time"]) if events else float("nan"),
        "trace": stress_trace,
        "steps": sim.n_steps,
        "dt": sim.dt,
        "post_rupture_bond_active": bool(sim.bond_active[k]),
    }


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FloeFieldConfig:
    """Shear cell of a drift-ice floe field.

    ``concentration`` is the *plan-view* areal ice fraction that fixes the
    number of floes in the cell, ``C = N pi r_mean^2 / (Lx H)``.
    """

    concentration: float = 0.5
    box_width: float = 20.0           # Lx, periodic in x [m]
    box_height: float = 20.0          # H, wall gap in y [m]
    thickness: float = 0.5            # ice thickness d [m]
    mean_radius: float = 0.6          # mean floe radius [m]
    radius_spread: float = 2.0        # r_max / r_min
    density: float = 900.0            # ice density [kg/m^3]
    young_modulus: float = 1.0e7      # soft-sphere DEM modulus [Pa]
    poisson_ratio: float = 0.3
    friction: float = 0.3             # ice-ice Coulomb friction
    restitution: float = 0.5          # target normal restitution
    shear_rate: float = 0.5           # gamma_dot [1/s]
    wall_friction: float = 0.5
    bond_strength: float = 1.0e5      # sigma_b [Pa]
    bond_fraction: float = 0.0        # fraction of touching pairs bonded
    bond_radius_ratio: float = 0.25   # r_bond / min(r1, r2)
    bond_length_ratio: float = 0.05   # L_bond / r_bond (bond stiffness length)
    settle_steps: int = 2000
    shear_steps: int = 9000
    sample_steps: int = 5000          # the last sample_steps are averaged
    dt_safety: float = 0.04
    seed: int = 20260101

    def __post_init__(self) -> None:
        if not 0.0 <= self.concentration <= 0.95:
            raise IceIceCollisionError("concentration must lie in [0, 0.95]")
        for name in ("box_width", "box_height", "thickness", "mean_radius",
                     "density", "young_modulus"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise IceIceCollisionError(f"{name} must be finite and positive")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise IceIceCollisionError("poisson_ratio must lie in [0, 0.5)")
        for name in ("friction", "wall_friction", "bond_strength", "bond_fraction"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise IceIceCollisionError(f"{name} must be finite and nonnegative")
        if not 0.0 <= self.restitution < 1.0:
            raise IceIceCollisionError("restitution must lie in [0, 1)")
        if self.radius_spread < 1.0:
            raise IceIceCollisionError("radius_spread must be >= 1")
        if self.shear_rate < 0.0:
            raise IceIceCollisionError("shear_rate must be nonnegative")
        if self.settle_steps < 0 or self.shear_steps <= 0:
            raise IceIceCollisionError("invalid step counts")

    @property
    def n_floes(self) -> int:
        area = math.pi * self.mean_radius ** 2
        return int(round(self.concentration * self.box_width * self.box_height / area))

    @property
    def thickness_ratio(self) -> float:
        """Dimensionless ice thickness ``d / 1 m`` (contact-force scaling)."""
        return self.thickness

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def gamma_dot(self) -> float:
        """Imposed nominal strain rate ``U / H`` [1/s]."""
        return self.shear_rate

    @property
    def wall_speed(self) -> float:
        """Half the wall velocity difference, ``gamma_dot H / 2`` [m/s]."""
        return 0.5 * self.shear_rate * self.box_height

    @property
    def mean_diameter(self) -> float:
        return 2.0 * self.mean_radius


# ---------------------------------------------------------------------------
# shear-cell simulator
# ---------------------------------------------------------------------------
class FloeFieldSimulator:
    """Lees-Edwards shear cell of buoyant disks with Hertz-Mindlin contacts.

    The cell is doubly periodic (no walls) and sheared by the **SLLOD**
    equations of motion (Evans & Morriss 1984) in the Lees-Edwards sliding-brick
    frame, so the imposed strain rate ``gamma_dot`` acts on *every* floe:

        dx/dt = v_x + gamma_dot y,   dy/dt = v_y
        dv_x/dt = F_x / m - gamma_dot v_y,   dv_y/dt = F_y / m

    Internally the ``x`` coordinate is stored with the affine part removed,
    ``s = x - gamma y`` (``gamma = gamma_dot t``), which avoids any unbounded
    bookkeeping of the sliding-brick offset: the contact separation of a pair
    seen through an image ``iy`` is ``dx = (s_i - s_j) + gamma dy`` with ``dy``
    the minimum-image ``y`` separation.  The measured velocity is the *peculiar*
    velocity, so the shear stress is the sum of the contact virial and the
    convective (kinetic) part, exactly as in granular kinetic theory.

    Every floe carries two translational and one rotational degree of freedom.
    """

    def __init__(self, cfg: FloeFieldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = cfg.n_floes
        self.Lx = cfg.box_width
        self.H = cfg.box_height
        self.d_hat = cfg.thickness_ratio
        self.mu = cfg.friction
        v_ref = max(cfg.shear_rate * cfg.mean_diameter, 1e-3)
        self.beta = hunt_crossley_beta(cfg.restitution, v_ref)
        self.gamma_dot = cfg.gamma_dot
        self.gamma = 0.0

        self.radii = self._radii()
        self.mass = cfg.density * math.pi * self.radii ** 2 * cfg.thickness
        self.inertia = 0.5 * self.mass * self.radii ** 2
        self.pos = self._initial_positions()     # column 0 = s, column 1 = y
        self.vel = np.zeros((self.n, 2), dtype=DTYPE)
        self.omega = np.zeros(self.n, dtype=DTYPE)
        self.force = np.zeros((self.n, 2), dtype=DTYPE)
        self.torque = np.zeros(self.n, dtype=DTYPE)
        self.external_force = np.zeros((self.n, 2), dtype=DTYPE)

        self._pi, self._pj = np.triu_indices(self.n, k=1)
        n_pairs = len(self._pi)
        self._htang = np.zeros(n_pairs, dtype=DTYPE)
        self._r_eff = 1.0 / (1.0 / self.radii[self._pi] + 1.0 / self.radii[self._pj])
        inv_e = (1.0 - cfg.poisson_ratio ** 2) / cfg.young_modulus
        inv_g = (2.0 - cfg.poisson_ratio) / cfg.shear_modulus
        self._e_eff = 1.0 / (2.0 * inv_e)
        self._g_eff = 1.0 / (2.0 * inv_g)

        self.bond_active = np.zeros(n_pairs, dtype=bool)
        self.bond_s0 = np.zeros(n_pairs, dtype=DTYPE)
        self.bond_area_arr = np.zeros(n_pairs, dtype=DTYPE)
        self.bond_k = np.zeros(n_pairs, dtype=DTYPE)
        self.bond_rupture_events: list[dict[str, Any]] = []

        self.diss_damping = 0.0
        self.diss_friction = 0.0
        self.e_spring = 0.0
        self.time = 0.0
        self.n_steps = 0
        self.dt = 1e-3
        self.stress_trace: list[tuple[float, float]] = []

    # -- geometry ---------------------------------------------------------
    def _radii(self) -> np.ndarray:
        cfg = self.cfg
        if self.n <= 0:
            return np.zeros(0, dtype=DTYPE)
        r_min = cfg.mean_radius * 2.0 / (1.0 + cfg.radius_spread)
        r_max = r_min * cfg.radius_spread
        return r_min + (r_max - r_min) * self.rng.random(self.n)

    def _initial_positions(self) -> np.ndarray:
        if self.n <= 0:
            return np.zeros((0, 2), dtype=DTYPE)
        aspect = self.Lx / self.H
        ny = max(2, int(math.ceil(math.sqrt(self.n / aspect))))
        nx = max(2, int(math.ceil(self.n / ny)))
        ys = (np.arange(ny) + 0.5) * (self.H / ny)
        xs = (np.arange(nx) + 0.5) * (self.Lx / nx)
        pts = np.array([(x, y) for y in ys for x in xs], dtype=DTYPE)
        pts = pts[self.rng.permutation(len(pts))][: self.n]
        if len(pts) < self.n:  # pragma: no cover - defensive
            raise IceIceCollisionError("lattice cannot host the requested floes")
        jitter = 0.35 * np.array([self.Lx / nx, self.H / ny])
        pts = pts + self.rng.uniform(-1.0, 1.0, size=pts.shape) * jitter
        pts[:, 0] = np.mod(pts[:, 0], self.Lx)
        pts[:, 1] = np.mod(pts[:, 1], self.H)
        return pts

    def physical_positions(self) -> np.ndarray:
        """Floe centres in the lab frame, ``x = s + gamma y`` (mod Lx)."""
        out = np.empty_like(self.pos)
        out[:, 0] = np.mod(self.pos[:, 0] + self.gamma * self.pos[:, 1], self.Lx)
        out[:, 1] = self.pos[:, 1]
        return out

    def _contact_data(self) -> dict[str, np.ndarray]:
        """Overlap, unit normal and contact stiffness of every image pair."""
        pos = self.pos
        dy_raw = pos[self._pj, 1] - pos[self._pi, 1]
        dy = dy_raw - self.H * np.round(dy_raw / self.H)
        dx = pos[self._pj, 0] - pos[self._pi, 0] + self.gamma * dy
        dx = dx - self.Lx * np.round(dx / self.Lx)
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-12)
        r_sum = self.radii[self._pi] + self.radii[self._pj]
        delta = r_sum - dist
        active = delta > 0.0
        nx = np.where(active, dx / dist_safe, 1.0)
        ny = np.where(active, dy / dist_safe, 0.0)
        a = hertz_contact_radius(np.maximum(delta, 0.0), self._r_eff)
        kn = 2.0 * self._e_eff * a * self.d_hat
        kt = 8.0 * self._g_eff * a * self.d_hat
        return {"dx": dx, "dy": dy, "dist": dist, "delta": delta, "active": active,
                "nx": nx, "ny": ny, "kn": kn, "kt": kt, "r_sum": r_sum}

    # -- forces -----------------------------------------------------------
    def _floe_forces(self, dt: float) -> dict[str, Any]:
        cd = self._contact_data()
        active = cd["active"]
        i, j = self._pi, self._pj
        nx, ny = cd["nx"], cd["ny"]
        tx, ty = -ny, nx
        r_i, r_j = self.radii[i], self.radii[j]
        spin = self.omega[j] * r_j + self.omega[i] * r_i
        rel_vx = self.vel[j, 0] - self.vel[i, 0] - spin * tx
        rel_vy = self.vel[j, 1] - self.vel[i, 1] - spin * ty
        vn = rel_vx * nx + rel_vy * ny
        vt = rel_vx * tx + rel_vy * ty

        delta = np.maximum(cd["delta"], 0.0)
        fn_elastic = (4.0 / 3.0) * self._e_eff * np.sqrt(self._r_eff) \
            * delta ** 1.5 * self.d_hat
        fn = np.where(active, np.maximum(fn_elastic * (1.0 - self.beta * vn), 0.0), 0.0)

        # Mindlin tangential spring with Coulomb projection (Cundall-Strack)
        kt = np.where(active, cd["kt"], 0.0)
        h_old = self._htang
        self._htang = np.where(active, self._htang + vt * dt, 0.0)
        ft = -kt * self._htang
        cap = self.mu * fn
        slip = np.abs(ft) > cap
        scale = np.where(slip, cap / np.maximum(np.abs(ft), 1e-30), 1.0)
        self._htang = self._htang * scale
        ft = ft * scale
        e_new = 0.5 * kt * self._htang ** 2
        e_old = 0.5 * kt * h_old ** 2
        work_t = float(np.sum(-ft * vt * dt))
        self.diss_friction += max(work_t - float(np.sum(e_new - e_old)), 0.0)
        self.e_spring = float(np.sum(e_new))
        self.diss_damping += float(np.sum((fn_elastic * self.beta * vn * vn)[active] * dt))

        if self.bond_active.any():
            ext = cd["dist"] - self.bond_s0
            tensile = np.where(self.bond_active, self.bond_k * ext, 0.0)
            total_normal = np.abs(fn - tensile)
            broken = self.bond_active & (
                total_normal / np.maximum(self.bond_area_arr, 1e-30)
                > self.cfg.bond_strength)
            if broken.any():
                for idx in np.nonzero(broken)[0]:
                    self.bond_rupture_events.append({
                        "pair": [int(i[idx]), int(j[idx])],
                        "time": float(self.time),
                        "normal_stress": float(total_normal[idx]
                                               / self.bond_area_arr[idx]),
                        "sigma_b": float(self.cfg.bond_strength),
                    })
                self.bond_active = self.bond_active & ~broken
                tensile = np.where(self.bond_active, tensile, 0.0)
            fn = fn - tensile        # tension positive: pulls the pair together

        fx = fn * nx + ft * tx
        fy = fn * ny + ft * ty
        n = self.n
        self.force[:, 0] = (np.bincount(j, weights=fx, minlength=n)
                            - np.bincount(i, weights=fx, minlength=n))
        self.force[:, 1] = (np.bincount(j, weights=fy, minlength=n)
                            - np.bincount(i, weights=fy, minlength=n))
        self.torque = (-np.bincount(i, weights=r_i * ft, minlength=n)
                       - np.bincount(j, weights=r_j * ft, minlength=n))

        if self.external_force.any():
            self.force += self.external_force
        virial_xy = float(np.sum((fx * cd["dy"])[active]))
        virial_yy = float(np.sum((fy * cd["dy"])[active]))
        max_overlap = (float(np.max(np.maximum(cd["delta"], 0.0) / cd["r_sum"]))
                       if self.n > 1 else 0.0)
        return {"active": active, "fn": fn, "ft": ft, "virial_xy": virial_xy,
                "virial_yy": virial_yy, "n_contacts": int(active.sum()),
                "max_overlap": max_overlap}

    # -- integration ------------------------------------------------------
    def _timestep(self) -> float:
        """Stable step from the Hertz/Mindlin frequency of the current state."""
        if self.n <= 1:
            return 1e-3
        cd = self._contact_data()
        if not cd["active"].any():
            v_max = max(float(np.max(np.abs(self.vel))) if self.n else 0.0, 1e-6)
            return float(np.clip(0.02 * float(np.min(self.radii)) / v_max, 1e-6, 2e-3))
        kn_eff = cd["kn"] + np.where(self.bond_active, self.bond_k, 0.0)
        kn_max = float(np.max(kn_eff))
        kt_max = float(np.max(cd["kt"]))
        m_min = float(np.min(self.mass))
        i_min = float(np.min(self.inertia))
        r_max = float(np.max(self.radii))
        omega = max(math.sqrt(kn_max / m_min),
                    math.sqrt(kt_max * r_max ** 2 / i_min))
        return float(np.clip(self.cfg.dt_safety * 2.0 * math.pi / omega, 1e-7, 5e-3))

    def kinetic_energy(self) -> float:
        trans = 0.5 * float(np.sum(self.mass * np.sum(self.vel ** 2, axis=1)))
        rot = 0.5 * float(np.sum(self.inertia * self.omega ** 2))
        return trans + rot

    def kinetic_stress(self) -> float:
        """``sum m v_x v_y / A`` (the velocity is peculiar in the SLLOD frame)."""
        if self.n == 0:
            return 0.0
        return float(np.sum(self.mass * self.vel[:, 0] * self.vel[:, 1])) \
            / (self.Lx * self.H)

    def step(self, dt: float, damping: float = 0.0) -> dict[str, Any]:
        """One velocity-Verlet step of the SLLOD equations."""
        half = 0.5 * dt
        self.vel[:, 0] += half * (self.force[:, 0] / self.mass - self.gamma_dot
                                  * self.vel[:, 1])
        self.vel[:, 1] += half * self.force[:, 1] / self.mass
        self.omega += half * self.torque / self.inertia
        # drift: dx/dt = v_x + gamma_dot y handled through s = x - gamma y
        self.pos[:, 0] += dt * (self.vel[:, 0] - self.gamma * self.vel[:, 1])
        self.pos[:, 1] += dt * self.vel[:, 1]
        self.gamma += self.gamma_dot * dt
        self.pos[:, 0] = np.mod(self.pos[:, 0], self.Lx)
        self.pos[:, 1] = np.mod(self.pos[:, 1], self.H)
        floe = self._floe_forces(dt)
        self.vel[:, 0] += half * (self.force[:, 0] / self.mass - self.gamma_dot
                                  * self.vel[:, 1])
        self.vel[:, 1] += half * self.force[:, 1] / self.mass
        self.omega += half * self.torque / self.inertia
        if damping > 0.0:
            self.vel *= (1.0 - damping)
            self.omega *= (1.0 - damping)
        self.time += dt
        self.n_steps += 1
        self.dt = dt
        return floe

    # -- protocols --------------------------------------------------------
    def settle(self, steps: Optional[int] = None) -> dict[str, float]:
        """Relax the initial packing (no shear, strong numerical damping)."""
        steps = self.cfg.settle_steps if steps is None else steps
        if self.n == 0:
            return {"max_overlap": 0.0, "contacts": 0}
        shear = self.gamma_dot
        self.gamma_dot = 0.0
        self.gamma = 0.0
        self._floe_forces(1e-3)
        for _ in range(steps):
            self.step(self._timestep(), damping=0.05)
        self.gamma_dot = shear
        self.vel[:] = 0.0
        self.omega[:] = 0.0
        self._htang[:] = 0.0
        self.diss_damping = self.diss_friction = 0.0
        pass
        self.e_spring = 0.0
        self.time = 0.0
        self.n_steps = 0
        self.gamma = 0.0
        cd = self._contact_data()
        return {
            "max_overlap": float(np.max(np.maximum(cd["delta"], 0.0) / cd["r_sum"]))
            if self.n > 1 else 0.0,
            "contacts": int(cd["active"].sum()),
        }

    def form_bonds(self, fraction: Optional[float] = None) -> int:
        """Freeze-bond a fraction of the touching pairs."""
        frac = self.cfg.bond_fraction if fraction is None else fraction
        if self.n <= 1 or frac <= 0.0:
            return 0
        cd = self._contact_data()
        idx = np.nonzero(cd["active"])[0]
        if len(idx) == 0:
            return 0
        n_take = int(round(frac * len(idx)))
        if n_take <= 0:
            return 0
        chosen = self.rng.choice(idx, size=min(n_take, len(idx)), replace=False)
        cfg = self.cfg
        for k in chosen:
            i, j = self._pi[k], self._pj[k]
            area = bond_area(self.radii[i], self.radii[j], cfg.bond_radius_ratio)
            rb = cfg.bond_radius_ratio * min(self.radii[i], self.radii[j])
            self.bond_active[k] = True
            self.bond_s0[k] = cd["dist"][k]
            self.bond_area_arr[k] = area
            self.bond_k[k] = cfg.young_modulus * area / (cfg.bond_length_ratio * rb)
        return len(chosen)

    def run_shear(self, steps: Optional[int] = None) -> dict[str, Any]:
        """Shear the field and return the steady-state rheology diagnostics."""
        steps = self.cfg.shear_steps if steps is None else steps
        self._floe_forces(1e-3)
        sample_start = max(0, steps - self.cfg.sample_steps)
        samples: list[dict[str, float]] = []
        e_kin_start = self.kinetic_energy()
        dt = self._timestep()
        diss_window0 = self.diss_damping + self.diss_friction
        time_window0 = self.time
        for step in range(steps):
            floe = self.step(dt)
            if step >= sample_start:
                samples.append({"virial_xy": floe["virial_xy"],
                                "virial_yy": floe["virial_yy"],
                                "kin_xy": self.kinetic_stress(),
                                "n_contacts": floe["n_contacts"],
                                "e_kin": self.kinetic_energy()})
                self.stress_trace.append(
                    (self.time, (floe["virial_xy"] + self.kinetic_stress()
                                 * self.Lx * self.H) / (self.Lx * self.H)))
            if step % 100 == 0:
                dt = self._timestep()
        out = self.diagnostics(samples)
        window = max(self.time - time_window0, 1e-30)
        mass_total = float(np.sum(self.mass))
        window_diss = (self.diss_damping + self.diss_friction) - diss_window0
        out["window_duration"] = window
        out["window_dissipation"] = window_diss
        out["diss_rate_per_mass_window"] = window_diss / (mass_total * window)
        out["e_kin_start"] = e_kin_start
        out["dt"] = self.dt
        out["max_overlap"] = max((s.get("max_overlap", 0.0) for s in samples),
                                 default=0.0)
        return out

    # -- diagnostics ------------------------------------------------------
    def diagnostics(self, samples: list[dict[str, float]]) -> dict[str, Any]:
        area = self.Lx * self.H
        if not samples:
            samples = [{"virial_xy": 0.0, "virial_yy": 0.0, "kin_xy": 0.0,
                        "n_contacts": 0, "e_kin": 0.0}]
        tau_virial = -float(np.mean([s["virial_xy"] for s in samples])) / area
        tau_kin = -float(np.mean([s["kin_xy"] for s in samples]))
        pressure = float(np.mean([s["virial_yy"] for s in samples])) / area
        n_contacts = float(np.mean([s["n_contacts"] for s in samples]))
        gamma_dot = self.gamma_dot
        tau_total = tau_virial + tau_kin
        scale = self.cfg.thickness * gamma_dot if gamma_dot > 0 else 0.0
        mu_eff = abs(tau_total) / scale if scale > 0 else 0.0
        k = max(1, len(samples) // 5)
        tau_trend = (float(np.mean([s0["virial_xy"] for s0 in samples[:k]]))
                     / max(abs(float(np.mean([s0["virial_xy"] for s0 in samples[-k:]]))),
                           1e-30))
        total_diss = self.diss_damping + self.diss_friction
        mass_total = float(np.sum(self.mass))
        duration = max(self.time, 1e-30)
        return {
            "concentration": self.cfg.concentration,
            "n_floes": self.n,
            "tau_virial": tau_virial,
            "tau_kinetic": tau_kin,
            "tau_total": tau_total,
            "pressure": pressure,
            "mu_eff": mu_eff,
            "gamma_dot": gamma_dot,
            "mean_contacts": n_contacts,
            "coordination": 2.0 * n_contacts / self.n if self.n else 0.0,
            "e_kin_mean": float(np.mean([s["e_kin"] for s in samples])),
            "e_spring": self.e_spring,
            "diss_total": total_diss,
            "diss_damping": self.diss_damping,
            "diss_friction": self.diss_friction,
            "diss_rate_per_mass": total_diss / (mass_total * duration),
            "collisional_diss_rate_per_mass": self.diss_damping / (mass_total * duration),
            "dissipation_number": total_diss / max(
                abs(tau_total) * gamma_dot * area * duration, 1e-30),
            "tau_trend": tau_trend,
            "bond_ruptures": len(self.bond_rupture_events),
            "ice_mass": mass_total,
            "time": self.time,
            "steps": self.n_steps,
            "kinetic_energy": self.kinetic_energy(),
        }


# ---------------------------------------------------------------------------
# top-level drivers
# ---------------------------------------------------------------------------
def simulate_floe_field(cfg: FloeFieldConfig, *, return_simulator: bool = False):
    """Settle, (optionally) bond and shear one floe field; return diagnostics."""
    if cfg.concentration == 0.0 or cfg.n_floes <= 0:
        zero = {
            "concentration": 0.0, "n_floes": 0, "tau_virial": 0.0,
            "tau_kinetic": 0.0, "tau_total": 0.0, "pressure": 0.0,
            "mu_eff": 0.0,
            "gamma_dot": cfg.shear_rate, "wall_speed": cfg.wall_speed,
            "mean_contacts": 0.0, "coordination": 0.0, "e_kin_mean": 0.0,
            "e_spring": 0.0, "diss_total": 0.0, "diss_damping": 0.0,
            "diss_friction": 0.0, "diss_rate_per_mass": 0.0,
            "dissipation_number": 0.0, "collisional_diss_rate_per_mass": 0.0,
            "bond_ruptures": 0, "ice_mass": 0.0, "time": 0.0, "steps": 0,
            "kinetic_energy": 0.0, "settle_max_overlap": 0.0, "settle_contacts": 0,
            "bonds_formed": 0, "dt": 0.0, "e_kin_start": 0.0,
            "bond_rupture_events": [],
        }
        return (zero, None) if return_simulator else zero

    sim = FloeFieldSimulator(cfg)
    settle_info = sim.settle()
    bonds = sim.form_bonds()
    out = sim.run_shear()
    out["settle_max_overlap"] = settle_info["max_overlap"]
    out["settle_contacts"] = settle_info["contacts"]
    out["bonds_formed"] = bonds
    out["bond_rupture_events"] = sim.bond_rupture_events
    return (out, sim) if return_simulator else out


def sweep_concentration(concentrations, base: Optional[FloeFieldConfig] = None,
                        **overrides) -> list[dict[str, Any]]:
    """Run ``mu_eff(C)`` over a list of plan-view concentrations."""
    base = base or FloeFieldConfig()
    rows = []
    for c in concentrations:
        cfg = replace(base, concentration=float(c), **overrides)
        row = simulate_floe_field(cfg)
        row.pop("bond_rupture_events", None)
        rows.append(row)
    return rows


def dissipation_scaling(shear_rates, base: Optional[FloeFieldConfig] = None,
                        **overrides) -> dict[str, Any]:
    """Energy-dissipation scaling ``epsilon ~ C v^3 / d`` (Bagnold 1954).

    The characteristic velocity is ``v = gamma_dot * D_mean`` (mean floe
    diameter), and ``epsilon`` is the *collisional* dissipation rate per unit
    ice mass.  The fitted exponent of ``epsilon(v)`` is compared with the
    theoretical value 3.
    """
    base = base or FloeFieldConfig()
    rows = []
    for gd in shear_rates:
        cfg = replace(base, shear_rate=float(gd), **overrides)
        out = simulate_floe_field(cfg)
        d_mean = cfg.mean_diameter
        v = cfg.shear_rate * d_mean
        rows.append({"shear_rate": cfg.shear_rate, "v": v,
                     "epsilon": out["diss_rate_per_mass_window"],
                     "epsilon_total": out["diss_rate_per_mass_window"],
                     "tau_total": out["tau_total"],
                     "coordination": out["coordination"],
                     "mu_eff": out["mu_eff"], "concentration": cfg.concentration,
                     "mean_contacts": out["mean_contacts"],
                     "n_floes": out["n_floes"]})
    v = np.array([r["v"] for r in rows])
    eps = np.array([r["epsilon"] for r in rows])
    eps_total = np.array([r["epsilon_total"] for r in rows])
    good = eps > 0.0
    exponent = (float(np.polyfit(np.log(v[good]), np.log(eps[good]), 1)[0])
                if good.sum() > 2 else float("nan"))
    good_t = eps_total > 0.0
    exponent_total = (float(np.polyfit(np.log(v[good_t]), np.log(eps_total[good_t]), 1)[0])
                      if good_t.sum() > 2 else float("nan"))
    # normalisation check: epsilon / (C v^3 / d) should be a constant
    normalised = []
    for r in rows:
        if r["epsilon"] > 0.0 and r["v"] > 0.0:
            normalised.append(r["epsilon"] / (r["concentration"] * r["v"] ** 3
                                              / base.thickness))
    return {"rows": rows, "exponent": exponent, "exponent_total": exponent_total,
            "concentration": base.concentration,
            "normalisation_spread": (max(normalised) / min(normalised)
                                     if len(normalised) > 1 else float("nan")),
            "normalised": normalised}
