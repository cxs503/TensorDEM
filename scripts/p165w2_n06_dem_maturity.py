"""P165-W2-N06: curvature / integrator / torque evidence for non-spherical DEM.

Runs the cheap acceptance experiments of the campaign node and writes one JSON
artifact (all report numbers are read from it, never hand-copied):

    artifacts/dem/p165w2_n06/maturity_results.json
    artifacts/dem/p165w2_n06/figures/*.png

Groups
------
A  analytic curvature vs textbook (ellipsoid limit) and vs an independent
   mesh-based estimator (general superquadrics); refinement orders;
   legacy finite-difference baselines.
B  curvature-improvement neutrality: contact torque / effective radius change.
D  integrator: order of accuracy (harmonic + Duffing), rotational damping-rate
   fit, conservative energy drift (velocity-Verlet vs Euler negative control).

F  column collapse (particle-number / time-step convergence, friction scan) is
   the expensive part and lives in ``scripts/p165w2_n06_column_collapse.py``;
   this script only merges its JSON if present.

Usage::

    PYTHONPATH=src python scripts/p165w2_n06_dem_maturity.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.flat_punch import reduced_modulus                      # noqa: E402
from tensordem.superquadric import principal_curvatures_superquadric as legacy_curv  # noqa: E402
from tensordem.superquadric_curvature import (                        # noqa: E402
    curvature_table,
    ellipsoid_gaussian_curvature,
    fd_refinement_order,
    gaussian_curvature,
    mesh_refinement_order,
    principal_curvatures_analytic,
    principal_curvatures_fd,
    principal_curvatures_mesh,
    surface_points,
)
from tensordem.superquadric_dem import (                              # noqa: E402
    SQSystem,
    SQSystemConfig,
    Wall,
    energy_drift_per_1000,
    make_superquadric_body,
)
from tensordem.superquadric_integrators import (                      # noqa: E402
    duffing_accel,
    harmonic_accel,
    integrate,
    order_from_errors,
)
from tensordem.superquadric_kernel import (                           # noqa: E402
    ContactMaterial,
    contact_geometry_batch,
    effective_radius_hertz,
    principal_curvatures_local,
    rolling_resistance_torque,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts", "dem", "p165w2_n06")
FIG = os.path.join(OUT, "figures")

A_, B_, C_ = 0.03, 0.02, 0.01
SHAPES = [(A_, B_, C_, 2.0, 2.0), (A_, B_, C_, 3.0, 2.0), (A_, B_, C_, 4.0, 3.0),
          (A_, B_, C_, 5.0, 4.0), (A_, B_, C_, 8.0, 8.0), (A_, B_, C_, 2.0, 8.0)]
ANGLES = [(30.0, 20.0), (-25.0, 110.0), (50.0, 200.0), (70.0, 45.0),
          (10.0, 300.0), (-60.0, 80.0), (80.0, 10.0), (-40.0, 250.0)]


# ---------------------------------------------------------------------------
# A: curvature
# ---------------------------------------------------------------------------

def group_a() -> Dict[str, object]:
    # A1 ellipsoid limit vs textbook closed forms
    rng = np.random.default_rng(0)
    a, b, c = A_, B_, C_
    th = rng.uniform(0.05, math.pi - 0.05, 40)
    ph = rng.uniform(0.0, 2.0 * math.pi, 40)
    pts = np.column_stack([a * np.sin(th) * np.cos(ph),
                           b * np.sin(th) * np.sin(ph),
                           c * np.cos(th)])
    K = gaussian_curvature(pts, a, b, c, 2, 2)
    K_ref = ellipsoid_gaussian_curvature(pts, a, b, c)
    ellipsoid_K_err = float(np.max(np.abs(K - K_ref) / np.abs(K_ref)))

    # pole: k1 = c/a^2, k2 = c/b^2 (textbook)
    k1p, k2p = principal_curvatures_analytic(np.array([[0.0, 0.0, c]]), a, b, c, 2, 2)
    pole_ref = sorted([c / a ** 2, c / b ** 2])
    pole_err = max(abs(float(k1p[0]) - pole_ref[1]) / pole_ref[1],
                   abs(float(k2p[0]) - pole_ref[0]) / pole_ref[0])

    # sphere limit: k1 = k2 = 1/R
    R = 0.01
    sph = np.array([rng.standard_normal(3) for _ in range(20)])
    sph = R * sph / np.linalg.norm(sph, axis=1)[:, None]
    ks1, ks2 = principal_curvatures_analytic(sph, R, R, R, 2, 2)
    sphere_err = float(max(np.max(np.abs(ks1 - 1.0 / R)), np.max(np.abs(ks2 - 1.0 / R))) * R)
    gauss_sphere_err = float(max(abs(float(gaussian_curvature(sph[:1], R, R, R, 2, 2)[0])
                                     - 1.0 / R ** 2) * R ** 2, 0.0))

    # A2 general superquadrics: analytic vs independent mesh estimator
    table = curvature_table(SHAPES, ANGLES, n_rings=256)
    mesh_err = table["max_rel_err"]

    # A3 legacy estimators (baseline): scalar four-point FD on F, kernel FD on grad
    legacy_rows = []
    kernel_fd_rows = []
    for shape in SHAPES:
        a, b, c, n1, n2 = shape
        worst_legacy = 0.0
        worst_kernel_fd = 0.0
        worst_legacy_pt = None
        for eta_deg, omega_deg in ANGLES:
            eta, omega = math.radians(eta_deg), math.radians(omega_deg)
            p = surface_points(np.array([eta]), np.array([omega]), a, b, c, n1, n2)[0]
            ka1, ka2 = principal_curvatures_analytic(p[None, :], a, b, c, n1, n2)
            scale = max(abs(float(ka1[0])), abs(float(ka2[0])), 1e-30)
            l1, l2 = legacy_curv(p, a, b, c, n1, n2, method="fd")
            e_leg = max(abs(float(ka1[0]) - l1), abs(float(ka2[0]) - l2)) / scale
            if e_leg > worst_legacy:
                worst_legacy = e_leg
                worst_legacy_pt = [eta_deg, omega_deg]
            f2a, f2b = principal_curvatures_local(p[None, :], a, b, c, n1, n2,
                                                 method="fd")
            e_fd = max(abs(float(ka1[0]) - float(f2b[0])),
                       abs(float(ka2[0]) - float(f2a[0]))) / scale
            worst_kernel_fd = max(worst_kernel_fd, e_fd)
        legacy_rows.append({"shape": list(shape), "max_rel_err": worst_legacy,
                            "worst_point_deg": worst_legacy_pt})
        kernel_fd_rows.append({"shape": list(shape), "max_rel_err": worst_kernel_fd})

    # A4 refinement orders
    mesh_orders = [mesh_refinement_order(30.0, 20.0, s, rings=(4, 8, 16, 32, 64))
                   for s in SHAPES[:5]]
    fd_orders = [fd_refinement_order(30.0, 20.0, s) for s in SHAPES[:5]]

    return {
        "ellipsoid_gaussian_curvature_max_rel_err": ellipsoid_K_err,
        "ellipsoid_pole_principal_max_rel_err": pole_err,
        "sphere_limit_principal_max_rel_err": sphere_err,
        "sphere_limit_gaussian_max_rel_err": gauss_sphere_err,
        "mesh_table": table,
        "general_mesh_max_rel_err": mesh_err,
        "legacy_scalar_fd": legacy_rows,
        "kernel_fd": kernel_fd_rows,
        "mesh_refinement": mesh_orders,
        "fd_step_refinement": fd_orders,
        "mesh_order_min": float(min(o["order"] for o in mesh_orders)),
    }


# ---------------------------------------------------------------------------
# B: curvature change is torque-neutral for the production contact path
# ---------------------------------------------------------------------------

def group_b() -> Dict[str, object]:
    """Contact torque / effective radius with analytic vs FD curvature."""
    mat = ContactMaterial(e_star=reduced_modulus(1e9, 0.3, 1e9, 0.3), g_star=4e8,
                          mu=0.4, mu_rolling=0.15)
    rng = np.random.default_rng(11)
    torques_an, torques_fd, r_an, r_fd = [], [], [], []
    mu_r = 0.15
    sat_an, sat_fd = [], []
    for shape in SHAPES:
        a, b, c, n1, n2 = shape
        for eta_deg, omega_deg in ANGLES:
            p = surface_points(np.array([math.radians(eta_deg)]),
                               np.array([math.radians(omega_deg)]), a, b, c, n1, n2)
            k1a, k2a = principal_curvatures_local(p, a, b, c, n1, n2)
            k1f, k2f = principal_curvatures_local(p, a, b, c, n1, n2, method="fd")
            ra = effective_radius_hertz(k1a, k2a, np.zeros(1), np.zeros(1))
            rf = effective_radius_hertz(k1f, k2f, np.zeros(1), np.zeros(1))
            f_n = 1.0 + 5.0 * rng.random()
            omega_rel = rng.standard_normal((1, 3))
            m_a = rolling_resistance_torque(np.array([f_n]), ra, omega_rel,
                                           np.array([mu_r]), np.array([1e-3]))
            m_f = rolling_resistance_torque(np.array([f_n]), rf, omega_rel,
                                            np.array([mu_r]), np.array([1e-3]))
            torques_an.append(float(np.linalg.norm(m_a)))
            torques_fd.append(float(np.linalg.norm(m_f)))
            r_an.append(float(ra[0]))
            r_fd.append(float(rf[0]))
            sat_an.append(float(np.linalg.norm(m_a)) / (mu_r * float(ra[0]) * f_n))
            sat_fd.append(float(np.linalg.norm(m_f)) / (mu_r * float(rf[0]) * f_n))
    ta = np.array(torques_an)
    tf = np.array(torques_fd)
    torque_rel = float(np.max(np.abs(ta - tf) / np.maximum(ta, 1e-30)))
    ra = np.array(r_an)
    rf = np.array(r_fd)
    reff_rel = float(np.max(np.abs(ra - rf) / np.maximum(ra, 1e-30)))

    # parent criterion: rolling friction saturates at mu_r R_eff F_n (ratio 1.0)
    sat_ratio_an = float(np.mean(sat_an))
    sat_ratio_fd = float(np.mean(sat_fd))

    # legacy scalar-curvature path: same torque, i.e. how much the parent's
    # reference contact path moves when the 15-97 % curvature error is removed
    legacy_ratio = []
    for shape in SHAPES:
        a, b, c, n1, n2 = shape
        for eta_deg, omega_deg in ANGLES:
            p = surface_points(np.array([math.radians(eta_deg)]),
                               np.array([math.radians(omega_deg)]), a, b, c, n1, n2)[0]
            l1, l2 = legacy_curv(p, a, b, c, n1, n2, method="fd")
            k1, k2 = principal_curvatures_local(p[None, :], a, b, c, n1, n2)
            r_l = 1.0 / max((abs(l1) + abs(l2)) / 2.0, 1e-30)
            r_a = float(effective_radius_hertz(k1, k2, np.zeros(1), np.zeros(1))[0])
            legacy_ratio.append(abs(r_l - r_a) / max(r_a, 1e-30))
    return {
        "rolling_torque_max_rel_change": torque_rel,
        "r_eff_max_rel_change": reff_rel,
        "saturation_ratio_analytic": sat_ratio_an,
        "saturation_ratio_fd": sat_ratio_fd,
        "legacy_scalar_r_eff_max_rel_change": float(np.max(legacy_ratio)),
        "mu_rolling": mu_r,
        "n_cases": len(ta),
    }


# ---------------------------------------------------------------------------
# D: integrators
# ---------------------------------------------------------------------------

def group_d(quick: bool = False) -> Dict[str, object]:
    # D1 harmonic oscillator order (phase/position error at t = 10 periods)
    omega = 2.0 * math.pi
    period = 1.0 / (omega / (2 * math.pi))
    # deliberately NOT an integer number of periods: at t = k*T the phase error
    # of a symplectic scheme cancels to leading order and the measured order is
    # spuriously high (measured 3.0/4.0 for Euler/Verlet at exactly 10 T).
    t_end = 10.37 * period
    x0, v0 = 1.0, 0.0
    dts = [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4]
    exact = lambda t: (x0 * math.cos(omega * t), -x0 * omega * math.sin(omega * t))
    harm = {}
    for name in ("euler", "velocity_verlet"):
        errs = []
        for dt in dts:
            n = int(round(t_end / dt))
            x, v = integrate(name, harmonic_accel(omega), [x0], [v0], dt, n)
            xe, ve = exact(n * dt)
            errs.append(float(abs(x[0] - xe)))
        harm[name] = {"dts": dts, "err": errs, "order": order_from_errors(dts, errs)}

    # D2 Duffing (nonlinear conservative) order vs a fine reference solution
    beta = 2.0
    acc = duffing_accel(omega, beta)
    dt_ref = 2.5e-5
    n_ref = int(round(t_end / dt_ref))
    xr, vr = integrate("velocity_verlet", acc, [1.0], [0.0], dt_ref, n_ref)
    duff = {}
    for name in ("euler", "velocity_verlet"):
        errs = []
        for dt in dts:
            n = int(round(t_end / dt))
            x, v = integrate(name, acc, [1.0], [0.0], dt, n)
            errs.append(float(abs(x[0] - xr[0])))
        duff[name] = {"dts": dts, "err": errs, "order": order_from_errors(dts, errs)}

    # D3 rotational damping-rate fit: omega' = -gamma omega, gamma dt = 0.1
    gamma = 1000.0
    decay = {}
    for dt in (1e-4, 5e-5, 2.5e-5):
        n = int(round(0.02 / dt))            # 20 decay e-foldings * 0.1 ... ~0.02 s
        rows = {}
        for name, kw in (("euler", {}), ("velocity_verlet_kdk", {"velocity_predictor": False}),
                         ("velocity_verlet", {})):
            base = "euler" if name == "euler" else "velocity_verlet"
            omega_t = 50.0
            ts, ws = [0.0], [omega_t]
            for k in range(n):
                # scalar damped spin with the same step function as the DEM
                # (-gamma*omega acts like the background spin damping)
                x, v, _ = __import__(
                    "tensordem.superquadric_integrators", fromlist=["step"],
                ).step(base, np.zeros(1), np.array([omega_t]), dt,
                       lambda _x, _v: -gamma * _v, **kw)
                omega_t = float(v[0])
                ts.append((k + 1) * dt)
                ws.append(omega_t)
            ts = np.array(ts)
            ws = np.array(ws)
            slope = float(np.polyfit(ts, np.log(np.maximum(ws, 1e-300)), 1)[0])
            rows[name] = {"gamma_fit": -slope,
                          "rel_err": abs(-slope - gamma) / gamma,
                          "omega_end": float(ws[-1])}
        decay[f"{dt:.2e}"] = rows

    # D4 conservative energy drift over 1e4 steps (VV vs Euler negative control)
    drift = {}
    for scale in (0.5, 1.0, 2.0, 4.0):
        for name in ("euler", "velocity_verlet"):
            drift.setdefault(name, {})
            radius, rho = 0.005, 2500.0
            mat = ContactMaterial(e_star=reduced_modulus(1e7, 0.3, 1e7, 0.3),
                                  g_star=1e7 / 5.2, mu=0.0, mu_rolling=0.0,
                                  restitution=1.0)
            body = make_superquadric_body(radius, radius, radius, 2.0, 2.0,
                                          [0.0, 0.0, 0.02], rho)
            body.velocity = np.array([0.0, 0.0, -0.25])
            sysm = SQSystem([body], [Wall(np.array([0.0, 0.0, 1.0]))],
                            SQSystemConfig(material=mat, integrator=name,
                                           gravity=np.zeros(3)))
            dt0 = sysm.dt
            sysm.dt = dt0 * scale
            sysm.energy_log.clear()
            e0 = sysm.energy()
            maxdev = 0.0
            n_steps = 10000
            for k in range(n_steps):
                sysm.step()
                if k % 5 == 0:
                    maxdev = max(maxdev, abs(sysm.energy() - e0) / abs(e0))
            drift[name][f"{scale:.1f}"] = {
                "dt": sysm.dt, "dt_over_default": scale,
                "max_rel_energy_dev": maxdev,
                "end_rel_drift": abs(sysm.energy() - e0) / abs(e0),
                "drift_per_1000": energy_drift_per_1000(
                    [e0, sysm.energy()], n_steps),
                "max_penetration_ratio": sysm.max_penetration_ratio,
            }
    return {"harmonic": harm, "duffing": duff, "damping_decay": decay,
            "energy_drift_1e4": drift}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(res: Dict[str, object]) -> List[Dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    figs: List[Dict[str, str]] = []

    # fig 1: mesh refinement of the discrete curvature estimator
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for row in res["A"]["mesh_refinement"]:
        rings = [h["n_rings"] for h in row["history"]]
        err = [max(h["rel_err"], 1e-12) for h in row["history"]]
        label = f"n1={row['shape'][3]:g}, n2={row['shape'][4]:g} (p={row['order']:.2f})"
        ax.loglog(rings, err, "o-", label=label)
    ax.loglog([8, 64], [1e-2, 1e-2 * (8 / 64) ** 2], "k--", lw=1, label="slope -2")
    ax.set_xlabel("mesh resolution N (rings per pi)")
    ax.set_ylabel("|k_analytic - k_mesh| / max|k|")
    ax.set_title("Curvature: discrete estimator converges at 2nd order")
    ax.legend(fontsize=6)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(FIG, "fig1_curvature_mesh_order.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig1_curvature_mesh_order.png",
                 "caption": "Analytic vs mesh-based principal curvature, mesh refinement",
                 "path": path})

    # fig 2: error by shape family (analytic vs mesh), legacy vs new
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    shapes = [f"n1={r['shape'][3]:g}\nn2={r['shape'][4]:g}" for r in res["A"]["legacy_scalar_fd"]]
    legacy = [r["max_rel_err"] for r in res["A"]["legacy_scalar_fd"]]
    new = [r["max_rel_err"] for r in res["A"]["kernel_fd"]]
    mesh = []
    for r in res["A"]["legacy_scalar_fd"]:
        key = tuple(r["shape"])
        row = [t for t in res["A"]["mesh_table"]["rows"] if tuple(
            [t["a"], t["b"], t["c"], t["n1"], t["n2"]]) == key]
        mesh.append(max(t["rel_err_max"] for t in row) if row else 0.0)
    idx = np.arange(len(shapes))
    ax.bar(idx - 0.27, legacy, 0.27, label="legacy 4-pt FD on F (parent baseline)")
    ax.bar(idx, new, 0.27, label="kernel FD on grad (parent DEM path)")
    ax.bar(idx + 0.27, np.maximum(mesh, 1e-8), 0.27, label="analytic vs mesh (this node)")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(shapes, fontsize=7)
    ax.set_ylabel("max relative curvature error")
    ax.axhline(0.01, color="r", ls="--", lw=1, label="1% criterion")
    ax.set_title("Principal curvature error: baselines vs analytic")
    ax.legend(fontsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(FIG, "fig2_curvature_error_bars.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig2_curvature_error_bars.png",
                 "caption": "Curvature error by shape family: legacy FD vs analytic",
                 "path": path})

    # fig 3: integrator order
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for key, style in (("harmonic", "o-"), ("duffing", "s--")):
        for name, label in (("euler", "Euler"), ("velocity_verlet", "velocity-Verlet")):
            d = res["D"][key][name]
            ax.loglog(d["dts"], d["err"], style,
                      label=f"{key} / {label}: p={d['order']:.2f}")
    ax.set_xlabel("dt (s)")
    ax.set_ylabel("final position error (m)")
    ax.set_title("Integrator order of accuracy")
    ax.legend(fontsize=6)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(FIG, "fig3_integrator_order.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig3_integrator_order.png",
                 "caption": "Position error vs dt: symplectic Euler (1st) vs velocity-Verlet (2nd)",
                 "path": path})

    # fig 4: conservative energy deviation vs dt scale
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    scales = sorted(res["D"]["energy_drift_1e4"]["euler"],
                    key=lambda s: res["D"]["energy_drift_1e4"]["euler"][s]["dt"])
    for name, style in (("euler", "o-"), ("velocity_verlet", "s-")):
        y = [max(res["D"]["energy_drift_1e4"][name][s]["max_rel_energy_dev"], 1e-16)
             for s in scales]
        x = [res["D"]["energy_drift_1e4"][name][s]["dt"] for s in scales]
        ax.loglog(x, y, style, label=name)
    ax.axhline(1e-3, color="g", ls="--", lw=1, label="VV criterion 1e-3")
    ax.axhline(1e-2, color="r", ls=":", lw=1, label="Euler control 1e-2")
    ax.set_xlabel("dt (s)")
    ax.set_ylabel("max |E-E0|/|E0| over 1e4 steps")
    ax.set_title("Conservative elastic impact: energy deviation vs dt")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(FIG, "fig4_energy_drift.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig4_energy_drift.png",
                 "caption": "Energy deviation of a conservative bounce for Euler vs velocity-Verlet",
                 "path": path})

    # fig 5: damping-rate fit error
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    dts = sorted(res["D"]["damping_decay"], key=lambda s: float(s))
    for name in ("euler", "velocity_verlet_kdk", "velocity_verlet"):
        y = [max(res["D"]["damping_decay"][s][name]["rel_err"], 1e-8) for s in dts]
        x = [float(s) for s in dts]
        ax.loglog(x, y, "o-", label=name)
    ax.axhline(0.01, color="r", ls="--", lw=1, label="1% criterion")
    ax.set_xlabel("dt (s)")
    ax.set_ylabel("|gamma_fit - gamma| / gamma")
    ax.set_title("Rotational damping rate fit (log-linear regression)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(FIG, "fig5_damping_fit.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig5_damping_fit.png",
                 "caption": "Fitted rotational damping rate error vs dt",
                 "path": path})
    return figs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    res: Dict[str, object] = {"campaign_node": "P165-W2-N06", "repo": "TensorDEM"}
    print("group A (curvature) ...", flush=True)
    res["A"] = group_a()
    print("  ellipsoid K err %.3e | mesh err %.3e | legacy max %.3f | mesh order %.2f"
          % (res["A"]["ellipsoid_gaussian_curvature_max_rel_err"],
             res["A"]["general_mesh_max_rel_err"],
             max(r["max_rel_err"] for r in res["A"]["legacy_scalar_fd"]),
             res["A"]["mesh_order_min"]), flush=True)
    print("group B (torque neutrality) ...", flush=True)
    res["B"] = group_b()
    print("  rolling torque rel change %.3e | R_eff rel change %.3e"
          % (res["B"]["rolling_torque_max_rel_change"],
             res["B"]["r_eff_max_rel_change"]), flush=True)
    print("group D (integrators) ...", flush=True)
    res["D"] = group_d(quick=args.quick)
    print("  harmonic orders: euler %.2f verlet %.2f"
          % (res["D"]["harmonic"]["euler"]["order"],
             res["D"]["harmonic"]["velocity_verlet"]["order"]), flush=True)
    print("  duffing orders: euler %.2f verlet %.2f"
          % (res["D"]["duffing"]["euler"]["order"],
             res["D"]["duffing"]["velocity_verlet"]["order"]), flush=True)
    for scale in res["D"]["energy_drift_1e4"]["velocity_verlet"]:
        e = res["D"]["energy_drift_1e4"]["euler"][scale]["max_rel_energy_dev"]
        v = res["D"]["energy_drift_1e4"]["velocity_verlet"][scale]["max_rel_energy_dev"]
        print("  energy dev dt x%s: euler %.3e verlet %.3e" % (scale, e, v), flush=True)
    figs = make_figures(res)
    res["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]
    res["wall_clock_s"] = time.perf_counter() - t0
    path = os.path.join(OUT, "maturity_results.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()