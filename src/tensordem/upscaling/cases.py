"""Macro-anchor test cases for the coarse-graining campaign.

Three standard granular macro-anchors measure the *observable* consequences
of a coarse-graining law:

1. ``shear``   - periodic direct-shear cell with a **dead-load** top platen
                 (constant normal stress by construction): macroscopic
                 friction coefficient ``mu_macro = tau / sigma_n`` and
                 friction angle ``phi = atan(mu)``.
2. ``repose``  - rotating drum: dynamic angle of repose ``theta_r`` from the
                 free surface of the steady avalanche.
3. ``ucs``     - uniaxial compression between rough platens: effective
                 Young's modulus ``E_eff`` and the first force peak.

Two modes:

* ``similar``      - all lengths scaled by ``f`` (self-similar replica, same
                     particle count): isolates the *scaling-law* error.
* ``fixed_domain`` - specimen kept at its f=1 size while the grains are f
                     times coarser (``N -> N/f^3``): adds the *resolution /
                     RVE* error of real coarse-graining.

The dimensionless protocol is held fixed across ``f``: Froude number
``omega^2 R_d / g`` for the drum; engineering strain rate for UCS and the
shear cell (the inertial number ``I = gamma_dot d sqrt(rho/sigma)`` is
invariant because ``d/L`` and ``sigma`` are invariant).

All quantities SI (m, kg, s, N, Pa, rad).
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

from .engine import Sim, PlaneWall, CylinderWall, hcp_fill_box, hcp_fill_cylinder
from .law import BaseMaterial, ScaledMaterial, scale_material

# ---------------------------------------------------------------------------
# Reference (f = 1) geometry and protocol
# ---------------------------------------------------------------------------

SHEAR_BASE = dict(
    lx=0.030,                # shear direction (periodic)
    ly=0.012,                # width (periodic)
    h=0.012,                 # height
    sigma_n=200.0,           # Pa, applied normal stress (dead load)
    shear_velocity=0.030,    # m/s -> strain rate 1.0 1/s (I ~ 0.007)
    shear_strain=0.18,       # accumulated strain over which mu is averaged
    settle_time=0.030,       # s, consolidation before shearing
)

DRUM_BASE = dict(
    drum_radius=0.014,       # m
    width=0.010,             # m
    omega=6.0,               # rad/s (Froude = omega^2 R / g = 0.051)
    revolutions=1.0,
    settle_time=0.060,       # s, gravity fill before rotation
)

UCS_BASE = dict(
    lx=0.014, ly=0.014, h=0.020,
    strain_rate=0.40,        # 1/s engineering strain rate
    strain_target=0.020,
    settle_time=0.060,
)


# Only *geometric lengths* (and velocities tied to them by the invariant
# dimensionless protocol) are scaled.  Dimensionless or physical-set quantities
# (revolutions, strain rate, confining stress, restitution, ...) stay invariant.
GEOM_KEYS = {
    "shear": ("lx", "ly", "h", "shear_velocity"),
    "repose": ("drum_radius", "width"),
    "ucs": ("lx", "ly", "h"),
}


def _geom(scale: float, d: Dict[str, float], case: str) -> Dict[str, float]:
    keys = GEOM_KEYS[case]
    return {k: (v * scale if (k in keys and isinstance(v, float)) else v)
            for k, v in d.items()}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: str
    approach: str
    f: float
    mode: str
    n_particles: int
    n_steps: int
    dt: float
    metrics: Dict[str, float]
    stability: Dict[str, float]
    wall_time: float
    mat: Dict[str, float]
    seed: int = 0
    traces: Dict[str, List[float]] = None

    def __post_init__(self) -> None:
        if self.traces is None:
            self.traces = {}

    def to_dict(self) -> dict:
        return asdict(self)


def _mat_dict(mat: ScaledMaterial) -> Dict[str, float]:
    return {
        "f": float(mat.f), "radius": float(mat.radius), "mass": float(mat.mass),
        "kn": float(mat.kn), "kt": float(mat.kt), "kr": float(mat.kr),
        "gamma_pair": float(mat.gamma_pair), "gamma_wall": float(mat.gamma_wall),
        "contact_time": float(mat.contact_time), "timestep": float(mat.timestep),
        "stiffness_ratio": float(mat.stiffness_ratio),
        "wave_speed": float(mat.wave_speed),
        "force_exponent": float(mat.force_exponent),
        "stress_exponent": float(mat.stress_exponent),
    }


def _stability(sim: Sim, v_ref: float, energy_trace: List[float],
               n_escaped: int = 0) -> Dict[str, float]:
    """Criterion ⑤: no particle fly-off, dissipative energy decay when settling."""
    vmax_ratio = sim.max_velocity / v_ref if v_ref > 0 else float("inf")
    if len(energy_trace) >= 3:
        e = np.asarray(energy_trace, dtype=float)
        span = max(e.max() - e.min(), 1e-30)
        rises = np.diff(e) / span
        max_rise = float(rises.max()) if len(rises) else 0.0
        monotone_ok = bool(max_rise <= 0.02)     # 2 % of the total energy span
        decay_frac = float((e[-1] - e[0]) / span)
    else:
        max_rise, monotone_ok, decay_frac = 0.0, True, 0.0
    return {
        "v_max": float(sim.max_velocity),
        "v_ref": float(v_ref),
        "v_max_ratio": float(vmax_ratio),
        "flyoff_ok": bool(vmax_ratio < 10.0),
        "energy_max_rise_frac": max_rise,
        "energy_monotone_ok": monotone_ok,
        "energy_decay_frac": decay_frac,
        "n_escaped": int(n_escaped),
    }


# ---------------------------------------------------------------------------
# Wall traction read-out (uses the contact history committed by the last step)
# ---------------------------------------------------------------------------


def _wall_forces(sim: Sim, wall) -> np.ndarray:
    """Recompute the current wall contact force on each particle."""
    mat = sim.mat
    n = len(sim.pos)
    out = np.zeros((n, 3), dtype=np.float64)
    gap = wall.gap(sim.pos, mat.radius)
    active = gap < 0.0
    if not active.any():
        return out
    idx = np.nonzero(active)[0]                      # ascending -> keys sorted
    widx = [i for i, w in enumerate(sim.walls) if w.name == wall.name][0]
    nvec = wall.normal_at(sim.pos[idx])
    overlap = -gap[idx]
    cpoint = sim.pos[idx] + nvec * mat.radius
    v_rel = sim.vel[idx] - wall.surface_velocity(cpoint)
    v_n = (v_rel * nvec).sum(axis=1)
    f_n = np.maximum(mat.kn * overlap - mat.gamma_wall * v_n, 0.0)
    keys = widx * n + idx
    tang, _ = sim.wall_history[wall.name].lookup(keys)
    out[idx] = f_n[:, None] * nvec - mat.kt * tang
    return out


def _wall_traction(sim: Sim, wall) -> tuple[float, float]:
    f = _wall_forces(sim, wall)
    return float(f[:, 0].sum()), float(f[:, 2].sum())


def _top_normal_stress(sim: Sim, wall: PlaneWall, area: float) -> float:
    return float(-_wall_forces(sim, wall)[:, 2].sum()) / area


def _settle(sim: Sim, v_tol: float, t_max: float,
            energy_trace: List[float] | None = None) -> int:
    """Run until the bed is quiescent (``v_max < 0.01 v_ref``) or ``t_max``.

    A gravity-deposited packing keeps creeping for a while; macro anchors are
    only meaningful once the initial free-fall/collapse transient has decayed.
    ``v_tol`` is an absolute velocity tolerance (typically 1e-3 R/t_c).
    """
    n_max = max(int(round(t_max / sim.dt)), 100)
    quiet_streak = 0
    for k in range(n_max):
        sim.step()
        if energy_trace is not None and k % 10 == 0:
            energy_trace.append(sim.total_energy())
        v = float(np.max(np.linalg.norm(sim.vel, axis=1))) if len(sim.vel) else 0.0
        quiet_streak = quiet_streak + 1 if v < v_tol else 0
        if quiet_streak >= 100 and k > 200:
            return k + 1
    return n_max


# ---------------------------------------------------------------------------
# 1. Direct shear cell (dead-load platen)
# ---------------------------------------------------------------------------


def run_shear(mat: ScaledMaterial, mode: str = "similar", seed: int = 11,
              geom_scale: float | None = None) -> CaseResult:
    t0 = _time.time()
    s = geom_scale if geom_scale is not None else (mat.f if mode == "similar" else 1.0)
    g = _geom(s, SHEAR_BASE, "shear")
    pos = hcp_fill_box(mat.radius, g["lx"], g["ly"], g["h"], seed=seed, jitter=0.03)
    area = g["lx"] * g["ly"]
    # a dimensionless-similar shear test needs sigma_n / (rho g H) invariant, i.e.
    # the imposed stress must scale with f exactly like the self-weight stress
    sigma_n = g["sigma_n"] * s
    z_top0 = float(pos[:, 2].max()) + mat.radius * 1.02

    bottom = PlaneWall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), name="bottom")
    top = PlaneWall((0.0, 0.0, z_top0), (0.0, 0.0, -1.0),
                    velocity=(g["shear_velocity"], 0.0, 0.0), name="top")
    sim = Sim(pos=pos, mat=mat, box=np.array([g["lx"], g["ly"], 0.0]),
              walls=[bottom, top],
              gravity=np.array([0.0, 0.0, -mat.gravity]), dt=mat.timestep)

    # dead-load platen dynamics: mass = particle mass of the bed, critically damped
    m_wall = mat.mass * len(pos)
    f_load = sigma_n * area                 # pure dead load (platen weight excluded)
    n_contacts = max(int(0.1 * len(pos)), 1)
    c_wall = 2.0 * 0.7 * math.sqrt(m_wall * n_contacts * mat.kn)
    v_wall_z = 0.0

    def platen_step() -> None:
        nonlocal v_wall_z
        f_p = _wall_forces(sim, top)
        react_z = -float(f_p[:, 2].sum())          # upward reaction on the platen
        a = (react_z - f_load - c_wall * v_wall_z) / m_wall
        v_wall_z += a * mat.timestep
        top.point[2] += v_wall_z * mat.timestep
        top.velocity[2] = v_wall_z
        top.velocity[0] = g["shear_velocity"]

    v_ref = math.sqrt(mat.gravity * g["h"])
    energy_trace: List[float] = []
    n_settle = int(round(g["settle_time"] / mat.timestep))
    for k in range(n_settle):
        platen_step()
        sim.step()
        if k % 10 == 0:
            energy_trace.append(sim.total_energy())
    v_tol = 1e-3 * mat.radius / mat.contact_time
    n_settle += _settle(sim, v_tol, 0.25 * math.sqrt(s), energy_trace)
    # the dead-load platen must actually be loaded before the shear phase
    # starts, otherwise the measured normal stress is zero (mu undefined)
    for _ in range(6000):
        platen_step()
        sim.step()
        n_settle += 1
        if _top_normal_stress(sim, top, area) >= 0.7 * sigma_n:
            break

    n_shear = max(int(round(g["shear_strain"] * g["lx"] / g["shear_velocity"]
                            / mat.timestep)), 200)
    win_start = int(0.4 * n_shear)
    tau_hist, sig_hist, z_hist, h_hist = [], [], [], []
    for k in range(n_shear):
        platen_step()
        sim.step()
        if k >= win_start:
            fx, fz = _wall_traction(sim, top)
            tau_hist.append(fx / area)
            sig_hist.append(-fz / area)
            z_hist.append(float(sim.pos[:, 2].mean()))
            h_hist.append(float(top.point[2]))
    tau = float(np.mean(tau_hist)) if tau_hist else float("nan")
    sigma = float(np.mean(sig_hist)) if sig_hist else float("nan")
    mu_macro = tau / sigma if (sigma == sigma and sigma > 0) else float("nan")
    phi = math.degrees(math.atan(mu_macro)) if mu_macro == mu_macro else float("nan")
    stab = _stability(sim, v_ref, energy_trace)
    return CaseResult(
        case="shear", approach=mat.approach, f=mat.f, mode=mode,
        n_particles=int(len(sim.pos)), n_steps=n_settle + n_shear, dt=mat.timestep,
        metrics={
            "tau_pa": tau, "sigma_n_pa": sigma, "sigma_applied_pa": sigma_n,
            "mu_macro": mu_macro, "friction_angle_deg": phi,
            "bulk_z_m": float(np.mean(z_hist)) if z_hist else float("nan"),
            "platen_z_m": float(np.mean(h_hist)) if h_hist else float("nan"),
            "shear_strain_window": float(0.6 * g["shear_strain"]),
        },
        stability=stab, wall_time=_time.time() - t0, mat=_mat_dict(mat), seed=seed,
        traces={"tau_pa": tau_hist[::4], "sigma_pa": sig_hist[::4]},
    )


# ---------------------------------------------------------------------------
# 2. Rotating drum -> angle of repose
# ---------------------------------------------------------------------------


def run_repose(mat: ScaledMaterial, mode: str = "similar", seed: int = 21,
               geom_scale: float | None = None) -> CaseResult:
    t0 = _time.time()
    s = geom_scale if geom_scale is not None else (mat.f if mode == "similar" else 1.0)
    g = _geom(s, DRUM_BASE, "repose")
    Rd, W = g["drum_radius"], g["width"]
    pos = hcp_fill_cylinder(mat.radius, (0.0, 0.0), Rd, W,
                            fill_fraction=0.45, seed=seed, jitter=0.03)
    omega = DRUM_BASE["omega"] / math.sqrt(s)     # Froude-number similarity

    cyl = CylinderWall((0.0, 0.0, 0.0), Rd, omega=0.0, name="drum")

    def co_rotate(p: np.ndarray) -> np.ndarray:
        v = np.zeros_like(p)
        v[:, 0] = cyl.omega * p[:, 2]
        v[:, 2] = -cyl.omega * p[:, 0]
        return v

    end_a = PlaneWall((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), velocity=co_rotate, name="end_a")
    end_b = PlaneWall((0.0, W, 0.0), (0.0, -1.0, 0.0), velocity=co_rotate, name="end_b")
    sim = Sim(pos=pos, mat=mat, box=None, walls=[cyl, end_a, end_b],
              gravity=np.array([0.0, 0.0, -mat.gravity]), dt=mat.timestep)

    v_ref = math.sqrt(mat.gravity * Rd)
    energy_trace: List[float] = []
    n_settle = max(int(round(g["settle_time"] / mat.timestep)), 50)
    for k in range(n_settle):
        sim.step()
        if k % 10 == 0:
            energy_trace.append(sim.total_energy())
    n_settle += _settle(sim, 1e-3 * mat.radius / mat.contact_time,
                        0.20 * math.sqrt(s), energy_trace)

    cyl.omega = omega
    period = 2.0 * math.pi / omega
    n_rot = max(int(round(g["revolutions"] * period / mat.timestep)), 200)
    snapshots: List[np.ndarray] = []
    n_escaped = 0
    snap_from = int(0.45 * n_rot)
    snap_every = max(1, (n_rot - snap_from) // 30)
    for k in range(n_rot):
        sim.step()
        if k >= snap_from and (k - snap_from) % snap_every == 0:
            snapshots.append(sim.pos.copy())
        if k % 50 == 0:
            d = np.hypot(sim.pos[:, 0], sim.pos[:, 2])
            n_escaped = max(n_escaped, int(np.sum(d > Rd + 3.0 * mat.radius)))

    theta = _repose_angle(snapshots, Rd, mat.radius)
    stab = _stability(sim, v_ref, energy_trace, n_escaped)
    return CaseResult(
        case="repose", approach=mat.approach, f=mat.f, mode=mode,
        n_particles=int(len(sim.pos)), n_steps=n_settle + n_rot, dt=mat.timestep,
        metrics={
            "theta_r_deg": theta,
            "omega_rad_s": omega,
            "froude": omega ** 2 * Rd / mat.gravity,
            "drum_radius_m": Rd,
            "revolutions": float(g["revolutions"]),
            "n_snapshots": float(len(snapshots)),
        },
        stability=stab, wall_time=_time.time() - t0, mat=_mat_dict(mat), seed=seed,
        traces=_surface_trace(snapshots),
    )


def _surface_trace(snapshots: List[np.ndarray]) -> Dict[str, List[float]]:
    if not snapshots:
        return {}
    snap = snapshots[-1]
    return {"x": snap[:, 0].tolist(), "z": snap[:, 2].tolist()}


def _repose_angle(snapshots: List[np.ndarray], Rd: float, radius: float) -> float:
    """Fit the free surface of the drum -> dynamic angle of repose (deg)."""
    if not snapshots:
        return float("nan")
    nbins = max(8, int(2.0 * Rd / (2.5 * radius)))
    edges = np.linspace(-Rd, Rd, nbins + 1)
    ztop = np.full(nbins, np.nan)
    for snap in snapshots:
        x, z = snap[:, 0], snap[:, 2] - radius
        idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
        for b in range(nbins):
            sel = idx == b
            if sel.any():
                val = float(z[sel].max())
                ztop[b] = val if np.isnan(ztop[b]) else max(ztop[b], val)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(ztop) & (np.abs(centres) < 0.78 * Rd)
    if ok.sum() < 4:
        return float("nan")
    x, z = centres[ok], ztop[ok]
    for _ in range(2):
        a, b = np.polyfit(x, z, 1)
        resid = z - (a * x + b)
        sd = float(resid.std())
        keep = np.abs(resid) <= max(2.0 * sd, 1e-12)
        if keep.sum() < 4:
            break
        x, z = x[keep], z[keep]
    a, _ = np.polyfit(x, z, 1)
    return float(math.degrees(math.atan(abs(a))))


# ---------------------------------------------------------------------------
# 3. Uniaxial compression -> E_eff, first force peak
# ---------------------------------------------------------------------------


def run_ucs(mat: ScaledMaterial, mode: str = "similar", seed: int = 31,
            geom_scale: float | None = None) -> CaseResult:
    """Uniaxial (oedometric) compression of a gravity-settled specimen.

    Protocol: the specimen is deposited under gravity until quiescent, the platen
    is placed on the free surface, the contact state is equilibrated, and the
    specimen is then compressed kinematically at a fixed engineering strain rate.
    ``E_eff`` is the least-squares slope of the axial stress-strain curve over
    ``eps in [5e-4, 3e-3]`` (contact overlap there is < 2 %); the first force
    peak is the maximum platen force over the whole compression.

    Note (declared in the docs): because the specimen is gravity supported, the
    measured modulus also contains the self-weight pre-compaction.  That is
    negligible at f=1 but *grows with f under口径B* (overlap ∝ f), which is
    precisely the deformability failure boundary quantified in this campaign.
    """
    t0 = _time.time()
    s = geom_scale if geom_scale is not None else (mat.f if mode == "similar" else 1.0)
    g = _geom(s, UCS_BASE, "ucs")
    pos = hcp_fill_box(mat.radius, g["lx"], g["ly"], g["h"], seed=seed, jitter=0.03)
    area0 = g["lx"] * g["ly"]
    v_platen = g["strain_rate"] * g["h"]

    bottom = PlaneWall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), name="bottom")
    top = PlaneWall((0.0, 0.0, float(pos[:, 2].max()) + 5.0 * mat.radius),
                    (0.0, 0.0, -1.0), name="top")
    sim = Sim(pos=pos, mat=mat, box=np.array([g["lx"], g["ly"], 0.0]),
              walls=[bottom, top],
              gravity=np.array([0.0, 0.0, -mat.gravity]), dt=mat.timestep)

    v_ref = math.sqrt(mat.gravity * g["h"])
    energy_trace: List[float] = []
    n_settle = max(int(round(g["settle_time"] / mat.timestep)), 50)
    for k in range(n_settle):
        sim.step()
        if k % 10 == 0:
            energy_trace.append(sim.total_energy())
    n_settle += _settle(sim, 1e-3 * mat.radius / mat.contact_time,
                        0.40 * math.sqrt(s), energy_trace)

    top.point[2] = float(sim.pos[:, 2].max()) + mat.radius * 0.999
    for _ in range(400):                       # contact equilibration
        sim.step()
        n_settle += 1
        energy_trace.append(sim.total_energy())
    h0 = float(top.point[2])
    sigma_pre = -float(_wall_forces(sim, top)[:, 2].sum()) / area0

    top.velocity = np.array([0.0, 0.0, -v_platen])
    n_comp = max(int(round(g["strain_target"] / g["strain_rate"] / mat.timestep)), 200)
    eps_l, sig_l, fz_l = [], [], []
    for k in range(n_comp):
        top.point[2] -= v_platen * mat.timestep
        sim.step()
        eps = (h0 - top.point[2]) / h0
        _, fz = _wall_traction(sim, top)
        eps_l.append(float(eps))
        fz_l.append(float(-fz))
        sig_l.append(float(-fz / area0))

    eps, sig, fz_arr = np.asarray(eps_l), np.asarray(sig_l), np.asarray(fz_l)
    win = (eps >= 5e-4) & (eps <= 3e-3)        # linear window (overlap < 2 %)
    e_eff = float(np.polyfit(eps[win], sig[win], 1)[0]) if win.sum() >= 5 else float("nan")
    peak = int(np.argmax(fz_arr)) if len(fz_arr) else 0
    stab = _stability(sim, v_ref, energy_trace)
    return CaseResult(
        case="ucs", approach=mat.approach, f=mat.f, mode=mode,
        n_particles=int(len(sim.pos)), n_steps=n_settle + n_comp, dt=mat.timestep,
        metrics={
            "e_eff_pa": e_eff,
            "first_peak_force_n": float(fz_arr[peak]),
            "first_peak_strain": float(eps[peak]),
            "peak_stress_pa": float(sig[peak]),
            "final_strain": float(eps[-1]),
            "strain_rate": g["strain_rate"],
            "platen_velocity": v_platen,
            "specimen_height_m": h0,
            "initial_stress_pa": sigma_pre,
        },
        stability=stab, wall_time=_time.time() - t0, mat=_mat_dict(mat), seed=seed,
        traces={"eps": eps_l[::4], "sigma_pa": sig_l[::4], "force_n": fz_l[::4]},
    )


RUNNERS = {"shear": run_shear, "repose": run_repose, "ucs": run_ucs}


BASE_SEEDS = {"shear": 11, "repose": 21, "ucs": 31}


def run_case(case: str, base: BaseMaterial, approach: str, f: float,
             mode: str = "similar", geom_scale: float | None = None,
             seed: int | None = None) -> CaseResult:
    mat = scale_material(base, f, approach)
    if seed is None:
        seed = BASE_SEEDS[case]
    return RUNNERS[case](mat, mode=mode, geom_scale=geom_scale, seed=seed)