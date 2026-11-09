"""The three verification cases of the F(t) export contract.

Every case returns a :class:`~tensordem.fatigue_chain.contract.ContactForceHistory`
in ``physics_complete`` mode (force + energy group), so the contract's energy
budget check has something real to bite on.

``single_ball_impact``
    One ball against a rigid wall, ``F_n = k delta + c v_n`` (Kelvin-Voigt,
    ``c = 2 zeta sqrt(k m)``).  The exported record is the **closed-form**
    damped-oscillator solution; :func:`integrate_impact_ode` integrates the same
    contact law with an independent semi-implicit scheme so the export can be
    checked against an integrator (peak force, contact duration, restitution
    ``e = exp(-pi zeta / sqrt(1 - zeta^2))``).

``two_ball_collision``
    Two balls, unequal masses, head-on.  Exported from the closed form of the
    *relative* coordinate (effective mass ``m1 m2 / (m1 + m2)``); the body
    velocities are recovered by integrating the contact force, so linear
    momentum is a check on the export itself, not an input.

``floe_field_bow``
    A 2D reduced-scale DEM: rigid circular floes (disks with thickness) driven
    by water drag, a rigid stem-circle bow advancing into the field, Hertz-type
    floe-floe contacts and Coulomb friction.  Explicit integration, no
    clamping of the normal force while ``delta > 0`` (so the energy identity
    holds exactly), contacts recorded per episode at the sub-step rate.

Scope note: the floe-field case is a **reduced-scale, translation-only 2D
model** of a broken-ice field, not a fully resolved 3D DEM of a ship in ice.
Its contact laws are the same as the certified impact cases, but its absolute
force/duration levels are model scale; the node therefore Froude-scales it
(``lambda_L = 20`` documented in the case metadata) before it enters the
fatigue chain, and every reported criterion on it is a ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .contract import HULL, ContactForceHistory

__all__ = [
    "ImpactCase",
    "single_ball_impact",
    "integrate_impact_ode",
    "two_ball_collision",
    "floe_field_bow",
    "FLOE_FIELD_DEFAULTS",
]

FLOE_FIELD_DEFAULTS = dict(
    n_floes=45,
    r_min_m=0.020,
    r_max_m=0.045,
    thickness_m=0.020,
    ice_density_kg_m3=917.0,
    bow_radius_m=0.20,
    bow_speed_m_s=0.60,
    stiffness_N_m15=2.0e5,
    damping_ratio=0.05,
    friction=0.30,
    water_drag_N_s_m=3.0,
    dt_s=2.5e-5,
    duration_s=2.0,
    seed=7,
    lambda_L=20.0,
)


@dataclass
class ImpactCase:
    """A verified case: contract history plus its analytic/reference data."""

    name: str
    history: ContactForceHistory
    analytic: Dict[str, float]
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# closed-form damped impact (shared by the two impact cases)
# ---------------------------------------------------------------------------

def _impact_closed_form(t: np.ndarray, *, mass: float, velocity: float, stiffness: float,
                        damping_ratio: float) -> Dict[str, np.ndarray]:
    wn = math.sqrt(stiffness / mass)
    zeta = float(damping_ratio)
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    c = 2.0 * zeta * math.sqrt(stiffness * mass)
    e = np.exp(-zeta * wn * t)
    delta = (velocity / wd) * e * np.sin(wd * t)
    vn = velocity * e * (np.cos(wd * t) - (zeta / math.sqrt(1.0 - zeta * zeta)) * np.sin(wd * t))
    force = stiffness * delta + c * vn
    return {
        "delta": delta,
        "v_n": vn,
        "F_n": force,
        "e_spring": 0.5 * stiffness * delta * delta,
        "stiffness": np.full_like(t, stiffness),
        "damping": np.full_like(t, c),
    }


def _cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    if y.size > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def integrate_impact_ode(*, mass: float, velocity: float, stiffness: float,
                         damping_ratio: float, dt: float = 2.0e-7,
                         max_steps: int = 2_000_000) -> Dict[str, object]:
    """Independent semi-implicit integration of the same one-contact law."""
    wn = math.sqrt(stiffness / mass)
    c = 2.0 * damping_ratio * math.sqrt(stiffness * mass)
    x, v = 0.0, float(velocity)
    t = 0.0
    ts, fs, vs, ds = [0.0], [], [], []
    peak_force = 0.0
    while t < max_steps * dt:
        force = stiffness * x + c * v
        if x <= 0.0 and t > 0.0:
            break
        peak_force = max(peak_force, force)
        ts.append(t)
        fs.append(force)
        vs.append(v)
        ds.append(x)
        a = -force / mass
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        v = v_half + 0.5 * dt * (-(stiffness * x + c * v_half) / mass)
        t += dt
    ts = np.asarray(ts)
    fs = np.asarray(fs)
    vs = np.asarray(vs)
    return {
        "t": ts,
        "F_n": fs,
        "v_n": vs,
        "delta": np.asarray(ds),
        "contact_duration_s": float(ts[-1]),
        "peak_force_N": float(peak_force),
        "restitution": float(-vs[-1] / velocity),
        "omega_n": wn,
        "analytic_restitution": math.exp(-math.pi * damping_ratio
                                         / math.sqrt(1.0 - damping_ratio ** 2)),
        "analytic_contact_duration_s": math.pi / (wn * math.sqrt(1.0 - damping_ratio ** 2)),
    }


def single_ball_impact(*, mass: float = 0.25, velocity: float = 0.5,
                       stiffness: float = 2.0e5, damping_ratio: float = 0.05,
                       rate_hz: float = 50_000.0, contact_id: int = 0) -> ImpactCase:
    """Ball against a rigid wall; export the closed-form contact episode."""
    wn = math.sqrt(stiffness / mass)
    wd = wn * math.sqrt(1.0 - damping_ratio ** 2)
    t_end = math.pi / wd                       # delta returns to zero
    dt = 1.0 / rate_hz
    n = max(int(math.floor(t_end / dt)), 8)
    t = np.arange(n) * dt                      # interior of the contact, delta > 0
    cf = _impact_closed_form(t, mass=mass, velocity=velocity, stiffness=stiffness,
                             damping_ratio=damping_ratio)
    # an unclamped Kelvin-Voigt force turns tensile just before delta -> 0; the
    # exported episode stops at that crossing so the energy identity is exact
    tensile = np.flatnonzero(cf["F_n"] < 0.0)
    n_rec = int(tensile[0]) if tensile.size else t.size
    t = t[:n_rec]
    cf = {key: value[:n_rec] for key, value in cf.items()}
    force = cf["F_n"]
    e_diss = _cumulative_trapz(2.0 * damping_ratio * math.sqrt(stiffness * mass) * cf["v_n"] ** 2, t)
    ids = np.full(t.size, contact_id)
    ball_radius = 0.05
    history = ContactForceHistory(
        t=t, contact_id=ids, floe_i=np.zeros(t.size, dtype=int),
        floe_j=np.full(t.size, HULL), F_n=force, F_t=np.zeros(t.size),
        patch_area=math.pi * ball_radius * np.maximum(cf["delta"], 0.0),
        delta=cf["delta"], v_n=cf["v_n"], e_spring=cf["e_spring"], e_diss=e_diss,
        meta={"case": "single_ball_impact", "mass_kg": mass, "velocity_m_s": velocity,
              "stiffness_N_m": stiffness, "damping_ratio": damping_ratio,
              "sampling_rate_hz": rate_hz, "contact_law": "kelvin_voigt_linear",
              "ball_radius_m": ball_radius,
              "truncated_at_tensile_crossing": bool(tensile.size)},
    )
    analytic = {
        "peak_force_N": float(force.max()),
        "contact_duration_s": float(t_end),
        "restitution": math.exp(-math.pi * damping_ratio
                                / math.sqrt(1.0 - damping_ratio ** 2)),
        "max_penetration_m": float(cf["delta"].max()),
        "peak_spring_energy_J": float(cf["e_spring"].max()),
        "dissipated_energy_J": float(e_diss[-1]),
        "initial_kinetic_energy_J": 0.5 * mass * velocity ** 2,
        "omega_n_rad_s": wn,
    }
    return ImpactCase(name="single_ball_impact", history=history, analytic=analytic)


def two_ball_collision(*, mass_i: float = 0.40, mass_j: float = 0.60,
                       velocity_i: float = 0.30, velocity_j: float = -0.10,
                       stiffness: float = 2.0e5, damping_ratio: float = 0.05,
                       rate_hz: float = 50_000.0) -> ImpactCase:
    """Head-on collision of two balls (relative coordinate, effective mass)."""
    m_eff = mass_i * mass_j / (mass_i + mass_j)
    v_rel = velocity_i - velocity_j
    wn = math.sqrt(stiffness / m_eff)
    wd = wn * math.sqrt(1.0 - damping_ratio ** 2)
    t_end = math.pi / wd
    dt = 1.0 / rate_hz
    n = max(int(math.floor(t_end / dt)), 8)
    t = np.arange(n) * dt
    cf = _impact_closed_form(t, mass=m_eff, velocity=v_rel, stiffness=stiffness,
                             damping_ratio=damping_ratio)
    tensile = np.flatnonzero(cf["F_n"] < 0.0)
    n_rec = int(tensile[0]) if tensile.size else t.size
    t = t[:n_rec]
    cf = {key: value[:n_rec] for key, value in cf.items()}
    force = cf["F_n"]
    impulse = _cumulative_trapz(force, t)
    v_i = velocity_i - impulse / mass_i
    v_j = velocity_j + impulse / mass_j
    e_diss = _cumulative_trapz(2.0 * damping_ratio * math.sqrt(stiffness * m_eff) * cf["v_n"] ** 2, t)
    history = ContactForceHistory(
        t=t, contact_id=np.zeros(t.size, dtype=int),
        floe_i=np.zeros(t.size, dtype=int), floe_j=np.ones(t.size, dtype=int),
        F_n=force, F_t=np.zeros(t.size),
        patch_area=math.pi * 0.05 * np.maximum(cf["delta"], 0.0),
        delta=cf["delta"], v_n=cf["v_n"], e_spring=cf["e_spring"], e_diss=e_diss,
        meta={"case": "two_ball_collision", "mass_i_kg": mass_i, "mass_j_kg": mass_j,
              "velocity_i_m_s": velocity_i, "velocity_j_m_s": velocity_j,
              "stiffness_N_m": stiffness, "damping_ratio": damping_ratio,
              "sampling_rate_hz": rate_hz, "contact_law": "kelvin_voigt_linear",
              "effective_mass_kg": m_eff},
    )
    analytic = {
        "peak_force_N": float(force.max()),
        "contact_duration_s": float(t_end),
        "restitution": math.exp(-math.pi * damping_ratio
                                / math.sqrt(1.0 - damping_ratio ** 2)),
        "momentum_initial_kg_m_s": float(mass_i * velocity_i + mass_j * velocity_j),
        "momentum_final_kg_m_s": float(mass_i * v_i[-1] + mass_j * v_j[-1]),
        "relative_velocity_initial_m_s": float(v_rel),
        "relative_velocity_final_m_s": float(-v_rel * math.exp(
            -math.pi * damping_ratio / math.sqrt(1.0 - damping_ratio ** 2))),
    }
    return ImpactCase(name="two_ball_collision", history=history, analytic=analytic,
                      extra={"v_i": v_i, "v_j": v_j, "impulse": impulse})


def floe_field_bow(**overrides) -> ImpactCase:
    """Reduced-scale 2D DEM of a broken-ice field crossed by a rigid bow."""
    cfg = dict(FLOE_FIELD_DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"unknown floe-field parameters: {sorted(unknown)}")
    cfg.update(overrides)
    rng = np.random.default_rng(int(cfg["seed"]))
    n = int(cfg["n_floes"])
    radius = rng.uniform(cfg["r_min_m"], cfg["r_max_m"], n)
    mass = cfg["ice_density_kg_m3"] * math.pi * radius ** 2 * cfg["thickness_m"]
    bow_r = float(cfg["bow_radius_m"])
    bow_x0 = -0.15                       # bow starts clear of the field
    # rejection placement: no initial overlap (neither floe-floe nor floe-bow)
    pos = np.zeros((n, 2))
    placed = 0
    for _ in range(200_000):
        if placed == n:
            break
        cand = np.array([rng.uniform(0.10, 1.10), rng.uniform(-0.34, 0.34)])
        if math.hypot(cand[0] - bow_x0, cand[1]) < bow_r + radius[placed] + 0.01:
            continue
        if placed and np.any(np.linalg.norm(pos[:placed] - cand, axis=1)
                             < radius[:placed] + radius[placed] + 0.002):
            continue
        pos[placed] = cand
        placed += 1
    if placed != n:
        raise RuntimeError(f"could not place {n} non-overlapping floes ({placed} placed)")
    vel = rng.normal(0.0, 0.03, (n, 2))
    bow_v = float(cfg["bow_speed_m_s"])
    k = float(cfg["stiffness_N_m15"])
    zeta = float(cfg["damping_ratio"])
    mu = float(cfg["friction"])
    drag = float(cfg["water_drag_N_s_m"])
    dt = float(cfg["dt_s"])
    steps = int(round(cfg["duration_s"] / dt))
    thickness = float(cfg["thickness_m"])

    # per-contact damping coefficient (using the smallest floe as conservative scale)
    m_ref = float(np.min(mass))
    c_n = 2.0 * zeta * math.sqrt(k * m_ref)

    active: Dict[tuple, int] = {}
    next_id = 0
    rows: Dict[str, List[float]] = {key: [] for key in
                                    ("t", "contact_id", "floe_i", "floe_j", "F_n", "F_t",
                                     "patch_area", "delta", "v_n", "e_spring", "e_diss")}
    spring_energy_state: Dict[int, float] = {}
    diss_state: Dict[int, float] = {}
    peak_bow_force = 0.0
    n_contacts = 0
    iu = np.triu_indices(n, k=1)

    v_prev: Dict[int, float] = {}

    def emit(cid: int, floe_i: int, floe_j: int, t: float, f_n: float, f_t: float,
             delta: float, v_n: float) -> None:
        e_s = 0.5 * k * delta * delta
        # trapezoid-consistent damping work: the contract's budget compares the
        # trapezoidal reduction of F_n v_n with (delta e_spring + e_diss), so
        # e_diss must be reduced with the same rule (a rectangle rule leaves an
        # O(dt) mismatch at the velocity reversal, measured 7 % at 25 kHz)
        prev = v_prev.get(cid, 0.0)
        diss = 0.5 * c_n * (v_n * v_n + prev * prev) * dt
        v_prev[cid] = v_n
        diss_state[cid] = diss_state.get(cid, 0.0) + diss
        rows["t"].append(t)
        rows["contact_id"].append(cid)
        rows["floe_i"].append(floe_i)
        rows["floe_j"].append(floe_j)
        rows["F_n"].append(f_n)
        rows["F_t"].append(f_t)
        rows["patch_area"].append(2.0 * math.sqrt(2.0 * radius[floe_i] * max(delta, 0.0))
                                 * thickness)
        rows["delta"].append(delta)
        rows["v_n"].append(v_n)
        rows["e_spring"].append(e_s)
        rows["e_diss"].append(diss_state[cid])

    bow_x = bow_x0
    for step in range(steps):
        t = step * dt
        force = np.zeros((n, 2))
        # floe-floe contacts
        diff = pos[iu[1]] - pos[iu[0]]
        dist = np.linalg.norm(diff, axis=1)
        dist_safe = np.maximum(dist, 1e-12)
        overlap = radius[iu[0]] + radius[iu[1]] - dist
        touching = overlap > 0.0
        for idx in np.flatnonzero(touching):
            i, j = int(iu[0][idx]), int(iu[1][idx])
            d = float(overlap[idx])
            nvec = diff[idx] / dist_safe[idx]
            v_rel = vel[j] - vel[i]
            v_n = float(-np.dot(v_rel, nvec))
            f_n = k * d + (c_n * v_n if v_n > 0.0 else 0.0)
            f_t = np.zeros(2)
            v_t = v_rel - np.dot(v_rel, nvec) * nvec
            v_t_norm = float(np.linalg.norm(v_t))
            if v_t_norm > 1e-12:
                f_t = -mu * f_n * v_t / v_t_norm
            force[i] -= f_n * nvec + f_t
            force[j] += f_n * nvec + f_t
            key = ("ff", i, j)
            if key not in active:
                active[key] = next_id
                spring_energy_state[next_id] = 0.0
                diss_state[next_id] = 0.0
                next_id += 1
                n_contacts += 1
            emit(active[key], i, j, t, f_n, float(np.linalg.norm(f_t)), d, v_n)
        # bow contacts (stem circle advancing in +x)
        rel = pos - np.array([bow_x, 0.0])
        dist_b = np.linalg.norm(rel, axis=1)
        dist_b_safe = np.maximum(dist_b, 1e-12)
        overlap_b = bow_r + radius - dist_b
        for i in np.flatnonzero(overlap_b > 0.0):
            i = int(i)
            d = float(overlap_b[i])
            nvec = rel[i] / dist_b_safe[i]
            v_rel = vel[i] - np.array([bow_v, 0.0])
            v_n = float(-np.dot(v_rel, nvec))
            f_n = k * d + (c_n * v_n if v_n > 0.0 else 0.0)
            v_t = v_rel - np.dot(v_rel, nvec) * nvec
            v_t_norm = float(np.linalg.norm(v_t))
            f_t = -mu * f_n * v_t / v_t_norm if v_t_norm > 1e-12 else np.zeros(2)
            # nvec points from the bow centre to the floe: a solid contact
            # pushes the floe outward, i.e. along +nvec
            force[i] += f_n * nvec + f_t
            peak_bow_force = max(peak_bow_force, f_n)
            key = ("bow", i)
            if key not in active:
                active[key] = next_id
                spring_energy_state[next_id] = 0.0
                diss_state[next_id] = 0.0
                next_id += 1
                n_contacts += 1
            emit(active[key], i, HULL, t, f_n, float(np.linalg.norm(f_t)), d, v_n)
        # remove closed contacts
        if active:
            for key in list(active):
                if key[0] == "ff":
                    i, j = key[1], key[2]
                    if radius[i] + radius[j] - np.linalg.norm(pos[j] - pos[i]) <= 0.0:
                        active.pop(key)
                else:
                    i = key[1]
                    if bow_r + radius[i] - np.linalg.norm(pos[i] - np.array([bow_x, 0.0])) <= 0.0:
                        active.pop(key)
        # integrate (semi-implicit Euler, drag + contact)
        acc = force / mass[:, None] - (drag / mass)[:, None] * vel
        vel += acc * dt
        pos += vel * dt
        bow_x += bow_v * dt
        if not (np.all(np.isfinite(vel)) and np.all(np.abs(vel) < 200.0)):
            raise RuntimeError(f"DEM diverged at t = {t:.4f} s; reduce dt_s or stiffness")

    history = ContactForceHistory(
        t=np.asarray(rows["t"]), contact_id=np.asarray(rows["contact_id"], dtype=int),
        floe_i=np.asarray(rows["floe_i"], dtype=int), floe_j=np.asarray(rows["floe_j"], dtype=int),
        F_n=np.asarray(rows["F_n"]), F_t=np.asarray(rows["F_t"]),
        patch_area=np.asarray(rows["patch_area"]),
        delta=np.asarray(rows["delta"]), v_n=np.asarray(rows["v_n"]),
        e_spring=np.asarray(rows["e_spring"]), e_diss=np.asarray(rows["e_diss"]),
        meta={"case": "floe_field_bow", "config": cfg, "n_contact_episodes": n_contacts,
              "sampling_rate_hz": 1.0 / dt, "contact_law": "hertz_linear_spring_dashpot",
              "note": "reduced-scale 2D DEM, translation only, froude lambda_L="
                      f"{cfg['lambda_L']}"},
    )
    analytic = {
        "peak_bow_force_N": float(peak_bow_force),
        "n_contact_episodes": float(n_contacts),
        "duration_s": float(steps * dt),
        "n_floes": float(n),
        "bow_travel_m": float(bow_x),
        "total_floe_mass_kg": float(mass.sum()),
        "mean_floe_radius_m": float(radius.mean()),
    }
    return ImpactCase(name="floe_field_bow", history=history, analytic=analytic)