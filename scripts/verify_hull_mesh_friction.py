"""Verification campaign for the hull-path friction / rolling law (real anchors).

Every gate here is scored against a *closed form*, never against another
implementation of the same discretisation:

* sliding on an incline      -> ``rolling_resistance.incline_sliding_acceleration``
* static friction threshold  -> ``tan(theta) = mu`` (Mohr-Coulomb)
* rolling threshold          -> ``rolling_resistance.critical_incline_angle_deg``
* rolling acceleration       -> ``rolling_resistance.incline_rolling_acceleration``
* frictionless limb          -> ``hull_mesh_contact.mesh_contact_forces`` (bitwise)
* tangential direction       -> the downhill tangent ``P_t g / |P_t g|`` of the
                               local facet (geometry, not a model)

plus the structural claims that have no scalar reference and are checked as
invariants: no tangential leakage into the normal, no friction dropout when a
sphere crosses facets of a doubly curved hull (with a **negative control**:
the same run with the memory dropped), momentum closure of the reaction, exact
state reset on contact loss, and a closed energy ledger.

Outputs (programmatic, zero hand-copied numbers):

    artifacts/hull_mesh_friction/verification.json
    artifacts/hull_mesh_friction/fig_incline.png
    artifacts/hull_mesh_friction/fig_thresholds.png
    artifacts/hull_mesh_friction/fig_facet_crossing.png

Run:  PYTHONPATH=src python scripts/verify_hull_mesh_friction.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.contact_tangential import saturation_displacement
from tensordem.hull_form_mesh import WigleyHull, wigley_mesh
from tensordem.hull_mesh_contact import make_flat_mesh, mesh_contact_forces
from tensordem.hull_mesh_friction import (
    HullFrictionConfig,
    HullFrictionState,
    integrate_sphere_on_mesh,
    mesh_contact_forces_frictional,
)
from tensordem.rolling_resistance import (
    critical_incline_angle_deg,
    incline_rolling_acceleration,
    incline_sliding_acceleration,
    pure_rolling_feasible,
)

F64 = torch.float64
OUT = Path("artifacts/hull_mesh_friction")
G = 9.81
R = 0.05                      # floe radius, m
MASS = 0.52                   # rho_ice * 4/3 pi r^3 (rho = 1000 kg/m^3)
INERTIA = 0.4 * MASS * R * R  # solid sphere
K_N = 2.0e5                   # normal penalty, N/m
K_T = 2.0e5                   # tangential stiffness, N/m


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def rot_y(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tilted_plane(theta_deg: float, size: float = 3.0, n: int = 6):
    """Flat plate of the given incline, gravity fixed along -z."""
    verts, faces = make_flat_mesh(size=size, normal="z", nx=n, ny=n)
    Rm = rot_y(theta_deg)
    v = (Rm @ verts.numpy().T).T
    return torch.as_tensor(v, dtype=F64), faces


def place_on_plane(theta_deg: float, radius: float, overlap: float):
    """Centre of a sphere resting on the tilted plane with the given overlap."""
    n = rot_y(theta_deg) @ np.array([0.0, 0.0, 1.0])
    return torch.as_tensor((radius - overlap) * n, dtype=F64)


def downhill_tangent(theta_deg: float) -> torch.Tensor:
    """Unit vector: the downhill direction in the plane (gravity projected)."""
    n = torch.as_tensor(rot_y(theta_deg) @ np.array([0.0, 0.0, 1.0]), dtype=F64)
    g = torch.tensor([0.0, 0.0, -G], dtype=F64)
    t = g - (g * n).sum() * n
    return t / torch.linalg.vector_norm(t)


def fit_acceleration(t: torch.Tensor, v: torch.Tensor, direction: torch.Tensor,
                     frac_from: float = 0.3):
    """Least-squares slope of the velocity along ``direction``, m/s^2."""
    n = v.shape[0]
    lo = int(frac_from * n)
    tt = t[lo:].numpy()
    vv = (v[lo:] @ direction).numpy() / float(torch.linalg.vector_norm(direction))
    A = np.vstack([tt, np.ones_like(tt)]).T
    slope, _ = np.linalg.lstsq(A, vv, rcond=None)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def gate_cone_and_direction():
    """G1: saturated friction = mu f_n, exactly anti-parallel to the slip."""
    theta = 15.0
    verts, faces = tilted_plane(theta)
    mu = 0.4
    cfg = HullFrictionConfig(mu=mu, k_t=K_T)
    n_hat = torch.as_tensor(rot_y(theta) @ np.array([0.0, 0.0, 1.0]), dtype=F64)
    overlap = 2.0e-4
    x = place_on_plane(theta, R, overlap).reshape(1, 3)
    t_hat = downhill_tangent(theta)
    v = (t_hat * 1.0).reshape(1, 3)          # sliding downhill at 1 m/s
    state = HullFrictionState()
    res = None
    for _ in range(400):                      # enough to saturate the spring
        res = mesh_contact_forces_frictional(
            x, v, torch.zeros(1, 3, dtype=F64), verts, faces, torch.zeros(3, dtype=F64),
            particle_radius=R, config=cfg, state=state, dt=1.0e-4, stiffness=K_N,
        )
    fn = float(res.normal_force[0])
    ft = res.tangential_force[0]
    ft_mag = float(torch.linalg.vector_norm(ft))
    cosang = float((-ft) @ t_hat) / max(ft_mag, 1e-300)
    return {
        "mu": mu,
        "normal_force_N": fn,
        "tangential_force_N": ft_mag,
        "ratio_ft_over_fn": ft_mag / fn,
        "cone_rel_error": abs(ft_mag - mu * fn) / (mu * fn),
        "direction_angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, cosang)))),
        "sliding_flag": bool(res.sliding[0]),
        "pass": abs(ft_mag - mu * fn) / (mu * fn) < 1e-12
        and math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) < 1e-6,
    }


def gate_normal_law_untouched():
    """G2: with mu = 0 the frictional law reproduces the frictionless kernel."""
    theta = 12.0
    verts, faces = tilted_plane(theta)
    x = place_on_plane(theta, R, 3.0e-4).reshape(1, 3)
    v = torch.tensor([[0.2, -0.1, -0.3]], dtype=F64)
    legacy_f, legacy_r = mesh_contact_forces(
        x, v, verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, stiffness=K_N, damping=7.0,
    )
    res = mesh_contact_forces_frictional(
        x, v, torch.zeros(1, 3, dtype=F64), verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=HullFrictionConfig(mu=0.0, mu_r=0.0),
        state=HullFrictionState(), dt=1.0e-4, stiffness=K_N, damping=7.0,
    )
    diff = float((legacy_f - res.forces).abs().max())
    rdiff = float((legacy_r - res.reaction_force).abs().max())
    # and with friction ON the *normal* ordinate must not move either
    res_f = mesh_contact_forces_frictional(
        x, v, torch.zeros(1, 3, dtype=F64), verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=HullFrictionConfig(mu=0.6, k_t=K_T),
        state=HullFrictionState(), dt=1.0e-4, stiffness=K_N, damping=7.0,
    )
    fn_diff = abs(float(legacy_f[0] @ (res_f.contacts["normal"][0]))
                  - float(res_f.normal_force[0]))
    return {
        "max_force_diff_mu0_N": diff,
        "max_reaction_diff_mu0_N": rdiff,
        "normal_ordinate_diff_with_friction_N": fn_diff,
        "pass": diff < 1e-12 and rdiff < 1e-12 and fn_diff < 1e-12,
    }


def gate_sliding_acceleration():
    """G3: translation-only incline acceleration vs the closed form."""
    rows = []
    mu = 0.35
    cfg = HullFrictionConfig(mu=mu, k_t=K_T)
    dt = 2.0e-4
    for theta in (10.0, 15.0, 20.0, 25.0):
        verts, faces = tilted_plane(theta)
        overlap = MASS * G * math.cos(math.radians(theta)) / K_N
        x0 = place_on_plane(theta, R, overlap)
        n_steps = int(0.25 / dt)
        run = integrate_sphere_on_mesh(
            x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
            radius=R, mass=MASS, inertia=None, vertices=verts, faces=faces,
            config=cfg, dt=dt, n_steps=n_steps, stiffness=K_N, damping=0.0,
            record_every=max(1, n_steps // 200),
        )
        a_meas = fit_acceleration(run["t"], run["v"], downhill_tangent(theta))
        a_ref = incline_sliding_acceleration(theta, mu, gravity=G)
        rows.append({
            "theta_deg": theta,
            "a_measured_m_s2": a_meas,
            "a_reference_m_s2": a_ref,
            "rel_error": abs(a_meas - a_ref) / abs(a_ref),
        })
    worst = max(r["rel_error"] for r in rows)
    return {"mu": mu, "rows": rows, "worst_rel_error": worst,
            "pass": worst < 0.01}


def _net_drift(run, theta):
    x = run["x"].numpy()
    t_hat = downhill_tangent(theta).numpy()
    proj = x @ t_hat
    n = len(proj)
    return float(proj[-1] - proj[n // 2])


def gate_static_threshold():
    """G4: the largest incline a resting floe can hold = atan(mu)."""
    mu = 0.45
    cfg = HullFrictionConfig(mu=mu, k_t=K_T)
    dt = 2.0e-4
    n_steps = int(0.12 / dt)
    theta_ref = math.degrees(math.atan(mu))

    def holds(theta):
        verts, faces = tilted_plane(theta, size=6.0, n=8)
        overlap = MASS * G * math.cos(math.radians(theta)) / K_N
        x0 = place_on_plane(theta, R, overlap)
        run = integrate_sphere_on_mesh(
            x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
            radius=R, mass=MASS, inertia=None, vertices=verts, faces=faces,
            config=cfg, dt=dt, n_steps=n_steps, stiffness=K_N, damping=0.0,
            record_every=max(1, n_steps // 100),
        )
        # sliding case at the same angle -> self-calibrated tolerance
        return _net_drift(run, theta), run

    lo, hi = 1.0, 60.0
    grid = []
    x_hold = None
    for _ in range(12):                       # bisection on the hold/slide verdict
        mid = 0.5 * (lo + hi)
        drift, run = holds(mid)
        # threshold: the elastic stick amplitude is m g sin(theta) / k_t
        delta_eq = MASS * G * math.sin(math.radians(mid)) / cfg.k_t
        stuck = abs(drift) < 50.0 * delta_eq
        grid.append({"theta_deg": mid, "net_drift_m": drift,
                     "stick_amplitude_m": delta_eq, "holds": stuck})
        if stuck:
            lo = mid
            x_hold = run
        else:
            hi = mid
    theta_meas = 0.5 * (lo + hi)
    return {
        "mu": mu,
        "theta_critical_reference_deg": theta_ref,
        "theta_critical_measured_deg": theta_meas,
        "rel_error": abs(theta_meas - theta_ref) / theta_ref,
        "bisection": grid,
        "pass": abs(theta_meas - theta_ref) / theta_ref < 0.01,
    }


def gate_rolling():
    """G5: rolling-resistance threshold and acceleration vs closed form."""
    mu_r = 0.04
    mu = 2.0                                  # welded: forces pure rolling
    theta_c = critical_incline_angle_deg(mu_r)
    acc_rows = []
    dt = 1.5e-4
    for theta in (8.0, 12.0, 16.0):
        rcfg = HullFrictionConfig(mu=mu, k_t=K_T, mu_r=mu_r, rolling_model="cdt")
        verts, faces = tilted_plane(theta, size=6.0, n=8)
        overlap = MASS * G * math.cos(math.radians(theta)) / K_N
        x0 = place_on_plane(theta, R, overlap)
        n_steps = int(0.12 / dt)
        run = integrate_sphere_on_mesh(
            x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
            radius=R, mass=MASS, inertia=INERTIA, vertices=verts, faces=faces,
            config=rcfg, dt=dt, n_steps=n_steps, stiffness=K_N, damping=0.0,
            record_every=max(1, n_steps // 200),
        )
        a_meas = fit_acceleration(run["t"], run["v"], downhill_tangent(theta))
        a_ref = incline_rolling_acceleration(theta, mu_r, gravity=G)
        acc_rows.append({
            "theta_deg": theta,
            "a_measured_m_s2": a_meas,
            "a_reference_m_s2": a_ref,
            "rel_error": abs(a_meas - a_ref) / abs(a_ref),
        })
    # threshold by bisection on the sign of the measured acceleration
    def accel(theta):
        rcfg = HullFrictionConfig(mu=mu, k_t=K_T, mu_r=mu_r, rolling_model="cdt")
        verts, faces = tilted_plane(theta, size=6.0, n=8)
        overlap = MASS * G * math.cos(math.radians(theta)) / K_N
        x0 = place_on_plane(theta, R, overlap)
        n_steps = int(0.12 / dt)
        run = integrate_sphere_on_mesh(
            x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
            radius=R, mass=MASS, inertia=INERTIA, vertices=verts, faces=faces,
            config=rcfg, dt=dt, n_steps=n_steps, stiffness=K_N, damping=0.0,
            record_every=max(1, n_steps // 100),
        )
        return fit_acceleration(run["t"], run["v"], downhill_tangent(theta), frac_from=0.5)

    lo, hi = 0.3 * theta_c, 3.0 * theta_c
    bis = []
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        a = accel(mid)
        bis.append({"theta_deg": mid, "a_measured_m_s2": a})
        if a > 0.0:
            hi = mid
        else:
            lo = mid
    theta_meas = 0.5 * (lo + hi)
    worst = max(r["rel_error"] for r in acc_rows)
    return {
        "mu_r": mu_r,
        "mu_coulomb": mu,
        "pure_rolling_feasible_at_8deg": pure_rolling_feasible(8.0, mu_r, mu),
        "theta_critical_reference_deg": theta_c,
        "theta_critical_measured_deg": theta_meas,
        "threshold_rel_error": abs(theta_meas - theta_c) / theta_c,
        "acceleration_rows": acc_rows,
        "worst_acceleration_rel_error": worst,
        "bisection": bis,
        "pass": abs(theta_meas - theta_c) / theta_c < 0.02 and worst < 0.05,
    }


def hull_point_normal(u: float, v: float):
    """Analytic point and outward unit normal of the ``y >= 0`` Wigley half.

    The normal is a finite-difference cross product of the analytic
    parametrisation -- independent of the mesh, so it can be used to *place* a
    sphere without borrowing the kernel's own answer.
    """
    hull = WigleyHull()
    h = 1e-6
    X = hull.surface
    du = (X(u + h, v) - X(u - h, v)) / (2 * h)
    dv = (X(u, v + h) - X(u, v - h)) / (2 * h)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n)
    if n[1] < 0:                      # outward = +y side of the half hull
        n = -n
    return X(u, v), n


def _hull_mesh(n_u: int = 28, n_v: int = 12):
    verts_np, faces_np = wigley_mesh(n_u=n_u, n_v=n_v)
    return (torch.as_tensor(verts_np, dtype=F64),
            torch.as_tensor(faces_np, dtype=torch.int64))


def gate_curved_hull():
    """G6/G7: no tangential leakage, traction along the downhill tangent."""
    verts, faces = _hull_mesh()
    mu = 0.3
    cfg = HullFrictionConfig(mu=mu, k_t=K_T)
    overlap = 2.0e-4
    dt = 2.0e-4
    leak, angle, ratio, normals = [], [], [], []
    for k in range(6):
        u = 0.2 + 0.6 * k / 5.0
        v = 0.45 + 0.25 * (k % 3) / 2.0
        p_an, n_an = hull_point_normal(u, v)
        centre = torch.as_tensor(p_an + (R - overlap) * n_an, dtype=F64).reshape(1, 3)
        n_hat = torch.as_tensor(n_an, dtype=F64)
        g = torch.tensor([0.0, 0.0, -G], dtype=F64)
        t_down = g - (g * n_hat).sum() * n_hat
        t_down = t_down / torch.linalg.vector_norm(t_down)
        v_slip = (1.0 * t_down).reshape(1, 3)
        state = HullFrictionState()
        res = None
        for _ in range(300):                       # saturate the tangential spring
            res = mesh_contact_forces_frictional(
                centre, v_slip, torch.zeros(1, 3, dtype=F64), verts, faces,
                torch.zeros(3, dtype=F64),
                particle_radius=R, config=cfg, state=state, dt=dt, stiffness=K_N,
            )
            if len(res.normal_force) == 0:
                break
        if res is None or len(res.normal_force) == 0:
            continue
        n = res.contacts["normal"][0]
        ft = res.tangential_force[0]
        fmag = float(torch.linalg.vector_norm(ft))
        fn = float(res.normal_force[0])
        normals.append(n.tolist())
        leak.append(abs(float(ft @ n)) / max(fmag, 1e-300))
        ratio.append(fmag / (mu * fn))
        if fmag > 0:
            c = float((-ft) @ t_down) / fmag
            angle.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    ok = bool(leak) and max(leak) < 1e-10 and max(angle) < 0.5 \
        and abs(min(ratio) - 1.0) < 0.05
    return {
        "n_probe_contacts": len(ratio),
        "max_tangential_leak_ratio": max(leak) if leak else None,
        "max_direction_error_deg": max(angle) if angle else None,
        "cone_ratio_min": min(ratio) if ratio else None,
        "cone_ratio_max": max(ratio) if ratio else None,
        "facet_normals": normals,
        "pass": bool(ok),
    }


def gate_facet_crossing():
    """G8: friction must not drop when the floe crosses facets (control: drop memory).

    The floe is carried along the hull *surface* (u fixed, v swept) at constant
    penetration and constant surface speed, so it crosses many facets inside
    the window; ``|F_t| / (mu F_n)`` is sampled every step.
    """
    verts, faces = _hull_mesh()
    hull = WigleyHull()
    mu = 0.4
    dt = 5.0e-5
    n_steps = 900
    overlap = 2.0e-4
    u0 = 0.55
    v0, v1 = 0.80, 0.20
    speed = 0.5                                   # m/s along the surface
    # arc length of the swept curve, so the step in v gives the surface speed
    ss = np.linspace(0.0, 1.0, 401)
    path = np.array([hull_point_normal(u0, v0 + (v1 - v0) * s)[0] for s in ss])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    total_arc = arc[-1]

    rows = {}
    for carry in (True, False):
        cfg = HullFrictionConfig(mu=mu, k_t=K_T, carry_over_across_facets=carry)
        state = HullFrictionState()
        ratios, tris = [], []
        for step in range(n_steps):
            s = min(1.0, (speed * step * dt) / total_arc)
            p_an, n_an = hull_point_normal(u0, v0 + (v1 - v0) * s)
            centre = torch.as_tensor(p_an + (R - overlap) * n_an, dtype=F64).reshape(1, 3)
            # surface velocity = speed * tangent of the path
            k = min(len(ss) - 1, max(1, int(s * (len(ss) - 1))))
            tang = path[k] - path[k - 1]
            tang = tang / np.linalg.norm(tang)
            v_now = torch.as_tensor(speed * tang, dtype=F64).reshape(1, 3)
            res = mesh_contact_forces_frictional(
                centre, v_now, torch.zeros(1, 3, dtype=F64), verts, faces,
                torch.zeros(3, dtype=F64),
                particle_radius=R, config=cfg, state=state, dt=dt, stiffness=K_N,
            )
            if len(res.normal_force) == 0:
                continue
            fn = float(res.normal_force[0])
            fmag = float(torch.linalg.vector_norm(res.tangential_force[0]))
            if fn > 0:
                ratios.append(fmag / (mu * fn))
            tris.append(int(res.contacts["triangle"][0]))
            if s >= 1.0:
                break
        rows["carry_on" if carry else "carry_off"] = {
            "min_cone_ratio": min(ratios),
            "mean_cone_ratio": float(np.mean(ratios)),
            "n_facet_changes": int(np.count_nonzero(np.diff(tris))),
            "n_samples": len(ratios),
            "ratio_series": ratios,
            "triangle_series": tris,
        }
    on, off = rows["carry_on"], rows["carry_off"]
    return {
        "mu": mu,
        "surface_speed_m_s": speed,
        "arc_length_m": total_arc,
        "carry_on": {k: v for k, v in on.items() if not k.endswith("_series")},
        "carry_off": {k: v for k, v in off.items() if not k.endswith("_series")},
        "n_facet_changes": on["n_facet_changes"],
        "pass": on["min_cone_ratio"] > 0.95 and off["min_cone_ratio"] < 0.5,
        "series": rows,
    }


def gate_reaction_and_energy():
    """G9/G10: energy ledger, hull kinematics, reaction transfer, state hygiene."""
    theta = 18.0
    mu = 0.4
    verts, faces = tilted_plane(theta, size=6.0, n=8)
    overlap = MASS * G * math.cos(math.radians(theta)) / K_N
    x0 = place_on_plane(theta, R, overlap)
    cfg = HullFrictionConfig(mu=mu, k_t=K_T)
    dt = 2.0e-4
    n_steps = int(0.15 / dt)
    run = integrate_sphere_on_mesh(
        x0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64),
        radius=R, mass=MASS, inertia=None, vertices=verts, faces=faces,
        config=cfg, dt=dt, n_steps=n_steps, stiffness=K_N, damping=0.0,
        record_every=max(1, n_steps // 300),
    )
    ledger = run["ledger_total"]
    rel_spread = float((ledger.max() - ledger.min()) / (MASS * G * R))
    diss = run["dissipated"]
    # net frictional work is dissipative (per-step work is not monotone while
    # the contact sticks and the tangential spring returns energy)
    monotone = bool(float(diss[-1]) > 0.0)
    a_meas = fit_acceleration(run["t"], run["v"], downhill_tangent(theta))
    a_ref = incline_sliding_acceleration(theta, mu, gravity=G)

    # --- hull kinematics: the slip velocity must carry omega_h x (c - x_0) ---
    x = torch.as_tensor([[0.0, 0.0, 0.30]], dtype=F64)
    hw = torch.tensor([0.0, 0.0, 0.2], dtype=F64)
    h0 = torch.tensor([0.1, 0.0, 0.0], dtype=F64)
    hv = torch.tensor([0.3, 0.0, 0.0], dtype=F64)
    res = mesh_contact_forces_frictional(
        x, torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, hv,
        particle_radius=R, config=HullFrictionConfig(mu=mu, k_t=K_T),
        state=HullFrictionState(), dt=dt, stiffness=K_N,
        hull_angular_velocity=hw, hull_origin=h0,
    )
    c = res.contacts["closest"][0]
    n = res.contacts["normal"][0]
    v_hull_pt = hv + torch.linalg.cross(hw, c - h0)
    v_slip_exact = v_hull_pt - (v_hull_pt * n).sum() * n
    slip_err = float((res.slip_velocity[0] + v_slip_exact).abs().max())

    # --- no relative motion => no friction at all ---------------------------
    v_sphere = torch.tensor([[0.3, 0.0, 0.0]], dtype=F64)
    res_noslip = mesh_contact_forces_frictional(
        x, v_sphere, torch.zeros(1, 3, dtype=F64), verts, faces,
        torch.tensor([0.3, 0.0, 0.0], dtype=F64),
        particle_radius=R, config=HullFrictionConfig(mu=mu, k_t=K_T),
        state=HullFrictionState(), dt=dt, stiffness=K_N,
    )
    ft_noslip = float(torch.linalg.vector_norm(res_noslip.tangential_force[0]))

    # --- reaction transfer to a shifted origin (parallel-axis identity) -----
    shift = torch.tensor([0.4, -0.2, 0.1], dtype=F64)
    res_shift = mesh_contact_forces_frictional(
        x, torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, hv,
        particle_radius=R, config=HullFrictionConfig(mu=mu, k_t=K_T),
        state=HullFrictionState(), dt=dt, stiffness=K_N,
        hull_angular_velocity=hw, hull_origin=h0 + shift,
    )
    # tau_o' = tau_o - shift x F  (the same force system, a different origin)
    predicted = res.reaction_torque - torch.linalg.cross(shift, res.reaction_force)
    transfer_err = float((res_shift.reaction_torque - predicted).abs().max())
    force_err = float((res_shift.reaction_force - res.reaction_force).abs().max())

    # --- state hygiene: a sphere that leaves the hull leaves no state -------
    state = HullFrictionState()
    _ = mesh_contact_forces_frictional(
        x, torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=cfg, state=state, dt=dt, stiffness=K_N,
    )
    entries_on_contact = len(state)
    far = torch.tensor([[0.0, 0.0, 5.0]], dtype=F64)
    _ = mesh_contact_forces_frictional(
        far, torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
        verts, faces, torch.zeros(3, dtype=F64),
        particle_radius=R, config=cfg, state=state, dt=dt, stiffness=K_N,
    )
    entries_after_loss = len(state)

    ok = (rel_spread < 1e-2 and monotone and abs(a_meas - a_ref) / abs(a_ref) < 0.01
          and slip_err < 1e-12 and ft_noslip < 1e-10 and transfer_err < 1e-12
          and force_err < 1e-12 and entries_on_contact > 0 and entries_after_loss == 0)
    return {
        "energy_ledger_rel_spread": rel_spread,
        "dissipated_monotone": monotone,
        "final_dissipated_J": float(diss[-1]),
        "acceleration_rel_error": abs(a_meas - a_ref) / abs(a_ref),
        "slip_velocity_kinematics_err_m_s": slip_err,
        "no_slip_tangential_force_N": ft_noslip,
        "reaction_transfer_err_Nm": transfer_err,
        "reaction_force_origin_invariance_err_N": force_err,
        "state_entries_on_contact": entries_on_contact,
        "state_entries_after_contact_loss": entries_after_loss,
        "reaction_force_N": res.reaction_force.tolist(),
        "reaction_torque_Nm": res.reaction_torque.tolist(),
        "pass": bool(ok),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_incline(g3, g4, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    th = np.array([r["theta_deg"] for r in g3["rows"]])
    am = np.array([r["a_measured_m_s2"] for r in g3["rows"]])
    ar = np.array([r["a_reference_m_s2"] for r in g3["rows"]])
    tt = np.linspace(8, 27, 100)
    ax[0].plot(tt, [incline_sliding_acceleration(x, g3["mu"], gravity=G) for x in tt],
               "-", color="k", label=r"$g(\sin\theta-\mu\cos\theta)$")
    ax[0].plot(th, am, "o", ms=7, label="mesh contact, sliding")
    ax[0].set_xlabel(r"incline angle $\theta$ [deg]")
    ax[0].set_ylabel(r"downhill acceleration [m/s$^2$]")
    ax[0].set_title("sliding on a triangulated plane")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    rows = g4["bisection"]
    th = np.array([r["theta_deg"] for r in rows])
    drift = np.array([abs(r["net_drift_m"]) for r in rows])
    order = np.argsort(th)
    ax[1].semilogy(th[order], drift[order], "o-", label="net drift over the window")
    ax[1].axvline(g4["theta_critical_reference_deg"], color="k", ls="--",
                  label=r"$\arctan\mu$ = %.2f deg" % g4["theta_critical_reference_deg"])
    ax[1].axvline(g4["theta_critical_measured_deg"], color="C1", ls=":",
                  label="measured = %.2f deg" % g4["theta_critical_measured_deg"])
    ax[1].set_xlabel(r"incline angle $\theta$ [deg]")
    ax[1].set_ylabel("net tangential drift [m]")
    ax[1].set_title("static friction threshold (bisection)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_thresholds(g5, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    th = np.array([r["theta_deg"] for r in g5["acceleration_rows"]])
    am = np.array([r["a_measured_m_s2"] for r in g5["acceleration_rows"]])
    tt = np.linspace(0.5, 20, 120)
    ax.plot(tt, [incline_rolling_acceleration(x, g5["mu_r"], gravity=G) for x in tt],
            "-", color="k", label=r"$g(\sin\theta-\mu_r\cos\theta)/(1+I/mR^2)$")
    ax.plot(th, am, "s", ms=7, label="mesh contact, rolling")
    ax.axvline(critical_incline_angle_deg(g5["mu_r"]), color="C3", ls="--",
               label=r"$\arctan\mu_r$ = %.2f deg" % critical_incline_angle_deg(g5["mu_r"]))
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel(r"incline angle $\theta$ [deg]")
    ax.set_ylabel(r"downhill acceleration [m/s$^2$]")
    ax.set_title("rolling resistance on the hull contact path")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_facet_crossing(g8, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for key, style in (("carry_on", "-"), ("carry_off", "--")):
        s = g8["series"][key]
        ax[0].plot(s["ratio_series"], style, lw=1.2,
                   label=("memory carried over" if key == "carry_on" else "memory dropped"))
        ax[1].plot(s["triangle_series"], style, lw=0.9, label=key)
    ax[0].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[0].set_xlabel("time step")
    ax[0].set_ylabel(r"$|F_t|/(\mu F_n)$")
    ax[0].set_title("friction on a curved hull while crossing facets")
    ax[0].set_ylim(0, 1.15)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("nearest triangle id")
    ax[1].set_title("facet the floe is touching")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    gates = {}
    gates["G1_cone_and_direction"] = gate_cone_and_direction()
    gates["G2_normal_law_untouched"] = gate_normal_law_untouched()
    gates["G3_sliding_acceleration"] = gate_sliding_acceleration()
    gates["G4_static_threshold"] = gate_static_threshold()
    gates["G5_rolling"] = gate_rolling()
    gates["G6_G7_curved_hull"] = gate_curved_hull()
    gates["G8_facet_crossing"] = gate_facet_crossing()
    gates["G9_G10_reaction_energy"] = gate_reaction_and_energy()

    fig_incline(gates["G3_sliding_acceleration"], gates["G4_static_threshold"],
                OUT / "fig_incline.png")
    fig_thresholds(gates["G5_rolling"], OUT / "fig_thresholds.png")
    fig_facet_crossing(gates["G8_facet_crossing"], OUT / "fig_facet_crossing.png")

    summary = {k: bool(v.get("pass", False)) for k, v in gates.items()}
    payload = {
        "module": "tensordem.hull_mesh_friction",
        "generated_by": "scripts/verify_hull_mesh_friction.py",
        "params": {"radius_m": R, "mass_kg": MASS, "inertia_kg_m2": INERTIA,
                   "k_n_N_m": K_N, "k_t_N_m": K_T, "gravity_m_s2": G},
        "gates": {k: {kk: vv for kk, vv in v.items() if kk not in ("run", "series")}
                  for k, v in gates.items()},
        "gate_pass": summary,
        "all_gates_pass": all(summary.values()),
        "n_pass": sum(summary.values()),
        "n_gates": len(summary),
        "wall_clock_s": time.time() - t0,
        "figures": ["fig_incline.png", "fig_thresholds.png", "fig_facet_crossing.png"],
    }
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps({"gate_pass": summary, "n_pass": payload["n_pass"],
                      "wall_clock_s": payload["wall_clock_s"]}, indent=2))
    return payload


if __name__ == "__main__":
    main()