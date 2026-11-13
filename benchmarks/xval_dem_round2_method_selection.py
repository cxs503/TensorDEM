#!/usr/bin/env python
"""XVAL · CASE-COLUMN-COLLAPSE round-2 method selection vs LIGGGHTS-PUBLIC.

Anchor ``XVAL-DEM-ROUND2-METHOD-SELECTION`` (repo TensorDEM).  The round-2
method selection (``oss_solutions``::

    TensorDEM|CASE-COLUMN-COLLAPSE|round2-method-selection

) projects grain angularity onto a single rolling coefficient ``mu_r`` (item 4).
That projection is exactly where our contact law and the mature open-source
implementation (LIGGGHTS-PUBLIC rolling models) can disagree, so this harness
cross-validates the *law* on closed single-contact scenes whose reference value
is analytic, and quantifies the parameter-mapping factor between the codes.

Everything is written to ``artifacts/xval_dem_round2_method_selection/``
(``results.json``, ``table_main.md``, ``figures/*.png``); no number in the
report is typed by hand.  Predeclared thresholds: ``prereg/XVAL_DEM_ROUND2_METHOD_SELECTION.json``.

Usage::

    PYTHONPATH=src python benchmarks/xval_dem_round2_method_selection.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular3d_core import Material3D                       # noqa: E402
from tensordem.granular_column.kernel import (WallSpringState,        # noqa: E402
                                             wall_contact_forces)
from tensordem.xval.oss_rolling import (cdt_torque,                   # noqa: E402
                                        hertz_equivalent_shear_modulus,
                                        oss_epsd2, rayleigh_timestep,
                                        tensordem_kernel)
from tensordem.xval.scenes import (pair_spin_relax, roll_on_plane,    # noqa: E402
                                   slice_theory_deceleration)

OUT = os.path.join(ROOT, "artifacts", "xval_dem_round2_method_selection")
FIG = os.path.join(OUT, "figures")
TOL = 0.15                     # owner precision standard (15 %)

#: material of the frozen CASE-COLUMN-COLLAPSE wave-5 surviving configuration
BASE = dict(radius=0.03, density=2500.0, kn=6.0e4, kt_ratio=0.5,
            mu=0.5, mu_rolling=0.5, mu_wall=0.8, gravity=9.81,
            dt_scale=1.0, poisson=0.25)

#: scene integration step: 5e-5 s resolves the stiffest contact frequency by a
#: wide margin (dt * sqrt(kt/m) = 1.6e-2) and is sweep-tested in E7.
DT = 5.0e-5


class CDTModel:
    """Adapter exposing :func:`cdt_torque` through the scene protocol."""

    name = "oss_cdt"

    def reset(self, n: int) -> None:
        self.n = int(n)

    def step(self, spin_rel, fn, reff, kt, mu_r, dt, normal=None) -> dict:
        torque = cdt_torque(spin_rel, fn, reff, mu_r, normal=normal)
        spin = np.atleast_2d(np.asarray(spin_rel, dtype=float))
        slip = spin * float(dt)
        return {"torque": torque, "roll": np.zeros_like(spin), "slip": slip,
                "dissipation": float(-np.einsum("ij,ij->i", torque, slip).sum()),
                "energy": 0.0}


class KernelWallModel:
    """The *production* kernel law through ``wall_contact_forces`` (plane z=0).

    The sphere is held at a fixed penetration ``delta = N/kn`` so the normal
    force equals the scene's ``N``.  Sliding friction is switched off
    (``mu_wall = 0``) so the kernel returns the *rolling* torque only and the
    scene keeps ownership of the tangential spring — otherwise the two
    tangential springs would be counted twice.
    """

    name = "tensordem_kernel_wall_contact_forces"

    def __init__(self, radius: float, kn: float, mat: Material3D,
                 mu_wall: float | None) -> None:
        self.radius = radius
        self.kn = kn
        self.mat = mat
        self.mu_wall = mu_wall

    def reset(self, n: int) -> None:
        self.state = WallSpringState(int(n))

    def step(self, spin_rel, fn, reff, kt, mu_r, dt, normal=None) -> dict:
        n_force = float(np.asarray(fn).reshape(-1)[0])
        delta = n_force / self.kn
        pos = np.array([[0.0, 0.0, self.radius - delta]])
        vel = np.zeros((1, 3))
        omega = np.atleast_2d(np.asarray(spin_rel, dtype=float)).copy()
        out = wall_contact_forces(pos, vel, omega, self.mat, float(dt),
                                  mu_wall=0.0, mu_rolling_wall=mu_r,
                                  state=self.state)
        torque = out["torque"].copy()
        return {"torque": torque, "roll": self.state.roll.copy(),
                "slip": np.zeros_like(torque),
                "dissipation": -float(out["dissip_rolling"]), "energy": 0.0}


def rel_err(got, ref) -> float:
    if ref == 0.0:
        return float("nan")
    return float((got - ref) / ref)


def rel_diff(a, b) -> float:
    scale = max(abs(a), abs(b), 1e-300)
    return float(abs(a - b) / scale)


# ---------------------------------------------------------------------------
# E1 — is the kernel law identical to the OSS EPSD2 law on wall contacts?
# ---------------------------------------------------------------------------
def experiment_kernel_identity() -> dict:
    mat = Material3D(radius=BASE["radius"], density=BASE["density"],
                     mu=BASE["mu"], mu_rolling=BASE["mu_rolling"],
                     kn=BASE["kn"], kt_ratio=BASE["kt_ratio"],
                     gravity=BASE["gravity"])
    kw = dict(radius=BASE["radius"], density=BASE["density"],
              kn=BASE["kn"], kt_ratio=BASE["kt_ratio"], mu_s=BASE["mu"],
              mu_r=BASE["mu_rolling"], gravity=BASE["gravity"],
              dt=DT, v0=0.30, t_max=0.02, sample_every=1)
    # production kernel (tangential + rolling both owned by the kernel)
    prod = roll_on_plane(KernelWallModel(BASE["radius"], BASE["kn"], mat,
                                         BASE["mu"]), **kw)
    # transcription of rolling_model_epsd2.h with the wall lever reff = r
    oss = roll_on_plane(oss_epsd2(), reff=BASE["radius"], **kw)
    hp, ho = prod["history"], oss["history"]
    return {
        "t_end": float(hp["t"][-1]),
        "max_rel_diff_omega": rel_diff(float(hp["omega"][-1]), float(ho["omega"][-1])),
        "max_abs_diff_omega": float(np.max(np.abs(hp["omega"] - ho["omega"]))),
        "omega_start": float(hp["omega"][0]), "omega_end_kernel": float(hp["omega"][-1]),
        "omega_end_oss": float(ho["omega"][-1]),
        "note": "kernel wall_contact_forces vs transcription of rolling_model_epsd2.h",
    }


# ---------------------------------------------------------------------------
# E2 — spin-down on a horizontal plane: stopping time vs analytic + vs OSS
# ---------------------------------------------------------------------------
def experiment_spin_down(mu_grid) -> dict:
    rows = []
    for mu_r in mu_grid:
        arms = {}
        for name, factory, reff in (
                ("kernel", tensordem_kernel, BASE["radius"]),
                ("oss_epsd2", oss_epsd2, BASE["radius"]),
                ("oss_cdt", CDTModel, BASE["radius"])):
            t_stop_est = 0.30 / slice_theory_deceleration(
                mu_r, BASE["gravity"], 0.0)
            out = roll_on_plane(factory(), radius=BASE["radius"],
                                density=BASE["density"], kn=BASE["kn"],
                                kt_ratio=BASE["kt_ratio"], mu_s=BASE["mu"],
                                mu_r=mu_r, gravity=BASE["gravity"], dt=DT,
                                v0=0.30, t_max=1.6 * t_stop_est + 0.02,
                                reff=reff, sample_every=20)
            arms[name] = out
        a_ref = slice_theory_deceleration(mu_r, BASE["gravity"], 0.0)
        t_ref = 0.30 / a_ref
        rows.append({
            "mu_r": float(mu_r),
            "t_stop_theory_s": float(t_ref),
            "a_theory_m_s2": float(a_ref),
            "t_stop": {k: (None if v["t_stop"] is None else float(v["t_stop"]))
                       for k, v in arms.items()},
            "x_stop": {k: (None if v["x_stop"] is None else float(v["x_stop"]))
                       for k, v in arms.items()},
            "err_vs_theory": {k: (None if v["t_stop"] is None else
                                  rel_err(v["t_stop"], t_ref))
                              for k, v in arms.items()},
            "err_kernel_vs_oss_epsd2": (None if (arms["kernel"]["t_stop"] is None or
                                                arms["oss_epsd2"]["t_stop"] is None)
                                        else rel_diff(arms["kernel"]["t_stop"],
                                                      arms["oss_epsd2"]["t_stop"])),
            "err_kernel_vs_oss_cdt": (None if (arms["kernel"]["t_stop"] is None or
                                              arms["oss_cdt"]["t_stop"] is None)
                                      else rel_diff(arms["kernel"]["t_stop"],
                                                    arms["oss_cdt"]["t_stop"])),
            "_traces": {k: v["history"] for k, v in arms.items()},
        })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# E3 — grain-grain pair: the lever arm (reff) convention and its mapping factor
# ---------------------------------------------------------------------------
def experiment_pair_lever(mu_r: float = 0.5, fn: float = 1.0) -> dict:
    r = BASE["radius"]
    kw = dict(radius=r, density=BASE["density"], kn=BASE["kn"],
              kt_ratio=BASE["kt_ratio"], mu_r=mu_r, w0=20.0, fn=fn,
              dt=DT, t_max=0.1)
    arms = {
        # TensorDEM kernel: lever = particle radius r
        "kernel_reff_r": pair_spin_relax(tensordem_kernel(), reff=r, **kw),
        # LIGGGHTS EPSD2 on a contact of two equal spheres: reff = r/2
        "oss_epsd2_reff_r2": pair_spin_relax(oss_epsd2(), reff=0.5 * r, **kw),
        # parameter mapping mu_r -> mu_r * (reff_oss / reff_kernel) = mu_r/2
        "kernel_mapped_mu_half": pair_spin_relax(tensordem_kernel(),
                                                 mu_r=0.5 * mu_r, reff=r,
                                                 **{k: v for k, v in kw.items()
                                                    if k != "mu_r"}),
        "oss_cdt_reff_r2": pair_spin_relax(CDTModel(), reff=0.5 * r, **kw),
    }
    t_theory = {}
    for key, out in arms.items():
        rate = out["theory_rate"]          # |dw_rel/dt| at the Coulomb cap
        # the couple saturates, so the decay is linear: half the initial spin
        # is lost after 0.5 * w0 / rate (an exponential law would be ln2/rate)
        t_theory[key] = 0.5 * 20.0 / rate if rate > 0 else float("inf")
    t_oss = arms["oss_epsd2_reff_r2"]["t_half"]
    return {
        "mu_r": float(mu_r), "fn_N": float(fn), "reff_ratio_oss_over_kernel": 0.5,
        "t_half": {k: (None if v["t_half"] is None else float(v["t_half"]))
                   for k, v in arms.items()},
        "t_half_theory": {k: float(v) for k, v in t_theory.items()},
        "err_vs_oss": {k: (None if (v["t_half"] is None or t_oss is None) else
                           rel_err(v["t_half"], t_oss))
                       for k, v in arms.items()},
        "omega_half_life_ratio_oss_over_kernel":
            (None if (t_oss is None or arms["kernel_reff_r"] ["t_half"] is None)
             else float(t_oss / arms["kernel_reff_r"]["t_half"])),
        "lever_factor_analytic": 2.0,
        "_traces": {k: v["history"] for k, v in arms.items()},
    }


# ---------------------------------------------------------------------------
# E4 — static hold on an incline: does a rolling law resist a sub-threshold drive?
# ---------------------------------------------------------------------------
def experiment_incline(mu_r: float = 0.5, slopes=(5, 15, 20, 25, 30),
                       dt: float = DT) -> dict:
    theta_c = math.degrees(math.atan(mu_r))       # tan(theta_c) = mu_r (rigid static)
    rows = []
    for slope in slopes:
        d = {}
        for name, factory in (("kernel", tensordem_kernel),
                              ("oss_epsd2", oss_epsd2),
                              ("oss_cdt", CDTModel)):
            out = roll_on_plane(factory(), radius=BASE["radius"],
                                density=BASE["density"], kn=BASE["kn"],
                                kt_ratio=BASE["kt_ratio"], mu_s=BASE["mu"],
                                mu_r=mu_r, gravity=BASE["gravity"], dt=dt,
                                v0=0.0, slope_deg=slope, t_max=0.2,
                                reff=BASE["radius"], sample_every=200)
            v = out["history"]["v"]
            n = max(3, v.size // 10)
            d[name] = {"v_final": float(np.mean(v[-n:])),
                       "x_final": float(out["history"]["x"][-1])}
        rows.append({"slope_deg": float(slope),
                     "tan_theta": float(math.tan(math.radians(slope))),
                     "below_static_threshold": bool(
                         math.tan(math.radians(slope)) < mu_r),
                     **d})
    return {"mu_r": float(mu_r), "dt_s": float(dt), "theta_c_deg": float(theta_c),
            "rows": rows}


def experiment_incline_dt(mu_r: float = 0.5, slope: float = 30.0) -> dict:
    out = {}
    for dt in (1.0e-5, DT, 2.0e-4):
        d = {}
        for name, factory in (("kernel", tensordem_kernel),
                              ("oss_epsd2", oss_epsd2),
                              ("oss_cdt", CDTModel)):
            run = roll_on_plane(factory(), radius=BASE["radius"],
                                density=BASE["density"], kn=BASE["kn"],
                                kt_ratio=BASE["kt_ratio"], mu_s=BASE["mu"],
                                mu_r=mu_r, gravity=BASE["gravity"], dt=dt,
                                v0=0.0, slope_deg=slope, t_max=0.2,
                                reff=BASE["radius"], sample_every=1)
            v = run["history"]["v"]
            n = max(3, v.size // 10)
            d[name] = float(np.mean(v[-n:]))
        out[f"dt_{dt:g}"] = d
    return {"mu_r": float(mu_r), "slope_deg": float(slope), "creep_v": out}


# ---------------------------------------------------------------------------
# E5 — drilling (torsion) spin: the kernel lumps it into rolling, OSS does not
# ---------------------------------------------------------------------------
def experiment_torsion(mu_r: float = 0.5, w0: float = 20.0) -> dict:
    r = BASE["radius"]
    m = BASE["density"] * 4.0 / 3.0 * math.pi * r ** 3
    n_load = m * BASE["gravity"]
    kt = BASE["kt_ratio"] * BASE["kn"]
    inertia = 0.4 * m * r ** 2

    def decay(model, torsion: bool):
        model.torsion = torsion
        model.reset(1)
        w = w0
        dt, t = DT, 0.0
        hist = []
        while t < 0.05:
            spin = np.array([[0.0, 0.0, w]])
            out = model.step(spin, np.array([n_load]), r, kt, mu_r, dt,
                             normal=np.array([[0.0, 0.0, 1.0]]))
            w += dt * float(out["torque"][0, 2]) / inertia
            t += dt
            if len(hist) == 0 or t - hist[-1][0] > 5.0e-4:
                hist.append((t, w))
        return w, hist

    mat = Material3D(radius=r, density=BASE["density"], mu=BASE["mu"],
                     mu_rolling=mu_r, kn=BASE["kn"],
                     kt_ratio=BASE["kt_ratio"], gravity=BASE["gravity"])
    w_kernel, h_kernel = decay(tensordem_kernel(), True)
    w_oss, h_oss = decay(tensordem_kernel(), False)

    # the same measurement through the *production* kernel entry point
    prod = KernelWallModel(r, BASE["kn"], mat, None)
    prod.reset(1)
    w_prod, dt, t, hist_prod = w0, DT, 0.0, []
    while t < 0.05:
        out = prod.step(np.array([[0.0, 0.0, w_prod]]), np.array([n_load]), r,
                        kt, mu_r, dt, normal=np.array([[0.0, 0.0, 1.0]]))
        w_prod += dt * float(out["torque"][0, 2]) / inertia
        t += dt
        if len(hist_prod) == 0 or t - hist_prod[-1][0] > 5.0e-4:
            hist_prod.append((t, w_prod))
    return {"mu_r": float(mu_r), "w0": float(w0), "t_end_s": 0.05,
            "w_end_kernel_torsion_on": float(w_kernel),
            "w_end_oss_default": float(w_oss),
            "w_end_production_kernel": float(w_prod),
            "spin_lost_fraction_kernel": float(1.0 - w_kernel / w0),
            "spin_lost_fraction_oss": float(1.0 - w_oss / w0),
            "spin_lost_fraction_production": float(1.0 - w_prod / w0),
            "_traces": {"t": [x[0] for x in h_kernel],
                        "kernel": [x[1] for x in h_kernel],
                        "oss": [x[1] for x in h_oss],
                        "production": [x[1] for x in hist_prod]}}


# ---------------------------------------------------------------------------
# E6 — timestep criterion: Rayleigh (OSS) vs "20 steps per normal period" (ours)
# ---------------------------------------------------------------------------
def experiment_timestep() -> dict:
    mat = Material3D(radius=BASE["radius"], density=BASE["density"],
                     mu=BASE["mu"], mu_rolling=BASE["mu_rolling"],
                     kn=BASE["kn"], kt_ratio=BASE["kt_ratio"],
                     gravity=BASE["gravity"])
    m = mat.mass
    g_eq = hertz_equivalent_shear_modulus(m * BASE["gravity"], BASE["kn"],
                                          BASE["radius"], BASE["poisson"])
    dt_r = rayleigh_timestep(BASE["radius"], BASE["density"], g_eq,
                             BASE["poisson"])
    dt_case = float(mat.timestep) * BASE["dt_scale"]
    return {"kn_N_m": BASE["kn"], "mass_kg": float(m),
            "hertz_equivalent_G_Pa": float(g_eq), "poisson": BASE["poisson"],
            "dt_rayleigh_s": float(dt_r), "dt_case_s": dt_case,
            "fraction_of_rayleigh": float(dt_case / dt_r),
            "oss_recommended_band": [0.1, 0.2],
            "note": ("OSS fixes the admissible fraction as a user input "
                     "(fraction_rayleigh_lim, fix_check_timestep_gran.cpp:83); "
                     "0.1-0.2 is the usual practice, not a source constant")}


# ---------------------------------------------------------------------------
# E7 — step-size sensitivity of the two laws (the stopping-time metric)
# ---------------------------------------------------------------------------
def experiment_dt_sensitivity(mu_r: float = 0.5) -> dict:
    out = {}
    ref = None
    for dt in (1.0e-5, DT, 1.0e-4, 2.0e-4):
        row = {}
        for name, factory in (("kernel", tensordem_kernel),
                              ("oss_epsd2", oss_epsd2),
                              ("oss_cdt", CDTModel)):
            run = roll_on_plane(factory(), radius=BASE["radius"],
                                density=BASE["density"], kn=BASE["kn"],
                                kt_ratio=BASE["kt_ratio"], mu_s=BASE["mu"],
                                mu_r=mu_r, gravity=BASE["gravity"], dt=dt,
                                v0=0.30, t_max=0.4, reff=BASE["radius"],
                                sample_every=50)
            row[name] = (None if run["t_stop"] is None else float(run["t_stop"]))
        out[f"dt_{dt:g}"] = row
        if abs(dt - DT) < 1e-12:
            ref = row
    spread = {}
    for name in ("kernel", "oss_epsd2", "oss_cdt"):
        vals = [v[name] for v in out.values() if v[name] is not None]
        spread[name] = (float((max(vals) - min(vals)) / ref[name])
                        if vals and ref and ref.get(name) else None)
    return {"mu_r": float(mu_r), "t_stop": out,
            "spread_fraction_over_dt_sweep": spread,
            "theory_s": float(0.30 / slice_theory_deceleration(
                mu_r, BASE["gravity"], 0.0))}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(res: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    made = []

    # 1 — spin-down traces at mu_r = 0.5
    rows = res["spin_down"]["rows"]
    row = min(rows, key=lambda r: abs(r["mu_r"] - 0.5))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=130)
    for key, style in (("kernel", dict(color="#1f77b4", lw=1.8)),
                       ("oss_epsd2", dict(color="#2ca02c", lw=1.2, ls="--")),
                       ("oss_cdt", dict(color="#d62728", lw=1.2, ls=":"))):
        tr = row["_traces"][key]
        ax.plot(tr["t"], tr["omega"], label=key, **style)
    t_th = row["t_stop_theory_s"]
    ax.plot([0, t_th], [0.30 / BASE["radius"], 0.0], color="k", lw=0.9,
            ls="-.", label="theory (5/7)mu_r g")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\omega_y$ [rad/s]")
    ax.set_title(f"rolling spin-down, $\\mu_r$={row['mu_r']:.2f} (plane contact)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    f = os.path.join(FIG, "fig1_spin_down.png")
    fig.tight_layout(); fig.savefig(f); plt.close(fig); made.append(f)

    # 2 — pair relaxation (lever convention)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=130)
    for key, style in (("kernel_reff_r", dict(color="#1f77b4", lw=1.8)),
                       ("oss_epsd2_reff_r2", dict(color="#2ca02c", lw=1.2, ls="--")),
                       ("kernel_mapped_mu_half", dict(color="#9467bd", lw=1.2, ls=":")),
                       ("oss_cdt_reff_r2", dict(color="#d62728", lw=1.0, ls="-."))):
        tr = res["pair_lever"]["_traces"][key]
        ax.plot(tr["t"], tr["w_rel"], label=key, **style)
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$|\omega_i-\omega_j|$ [rad/s]")
    ax.set_title("equal-sphere pair: rolling lever arm $r$ (ours) vs $r/2$ (OSS)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    f = os.path.join(FIG, "fig2_pair_lever.png")
    fig.tight_layout(); fig.savefig(f); plt.close(fig); made.append(f)

    # 3 — error vs mu_r
    mus = [r["mu_r"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=130)
    for key, style in (("kernel", dict(color="#1f77b4", marker="o")),
                       ("oss_epsd2", dict(color="#2ca02c", marker="s")),
                       ("oss_cdt", dict(color="#d62728", marker="^"))):
        y = [100.0 * (r["err_vs_theory"][key] or float("nan")) for r in rows]
        ax.plot(mus, y, label=f"{key} vs theory", **style)
    ax.axhspan(-15, 15, color="0.85", label="15 % band")
    ax.set_xlabel(r"$\mu_r$ [-]"); ax.set_ylabel("stopping-time error [%]")
    ax.set_title("rolling spin-down stopping time: law vs analytic")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    f = os.path.join(FIG, "fig3_error_vs_mur.png")
    fig.tight_layout(); fig.savefig(f); plt.close(fig); made.append(f)

    # 4 — incline creep vs slope
    inc = res["incline"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=130)
    slopes = [r["slope_deg"] for r in inc["rows"]]
    for key, style in (("kernel", dict(color="#1f77b4", marker="o")),
                       ("oss_epsd2", dict(color="#2ca02c", marker="s")),
                       ("oss_cdt", dict(color="#d62728", marker="^"))):
        ax.plot(slopes, [abs(r[key]["v_final"]) for r in inc["rows"]],
                label=key, **style)
    ax.axvline(inc["theta_c_deg"], color="k", ls="--", lw=1.0,
               label=rf"$\theta_c=\arctan\mu_r$={inc['theta_c_deg']:.1f}$\deg$")
    ax.set_xlabel(r"slope $\theta$ [deg]"); ax.set_ylabel(r"residual $|v|$ [m/s]")
    ax.set_yscale("log")
    ax.set_title(rf"static hold on an incline, $\mu_r$={inc['mu_r']:.2f}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    f = os.path.join(FIG, "fig4_incline_hold.png")
    fig.tight_layout(); fig.savefig(f); plt.close(fig); made.append(f)

    # 5 — torsion decay
    tor = res["torsion"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=130)
    ax.plot(tor["_traces"]["t"], tor["_traces"]["kernel"],
            color="#1f77b4", lw=1.8, label="kernel (torsion in rolling spring)")
    ax.plot(tor["_traces"]["t"], tor["_traces"]["production"],
            color="#ff7f0e", lw=1.2, ls=":", label="production kernel (wall_contact_forces)")
    ax.plot(tor["_traces"]["t"], tor["_traces"]["oss"],
            color="#2ca02c", lw=1.2, ls="--", label="OSS default (torsion removed)")
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\omega_z$ [rad/s]")
    ax.set_title("axial (drilling) spin on a plane")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    f = os.path.join(FIG, "fig5_torsion.png")
    fig.tight_layout(); fig.savefig(f); plt.close(fig); made.append(f)
    return [os.path.relpath(x, ROOT) for x in made]


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
def write_table(res: dict) -> str:
    lines = ["### E2 · spin-down on a plane (v0 = 0.30 m/s, dt = 2e-6 s)", "",
             "| mu_r | t_stop theory [ms] | kernel [ms] | OSS EPSD2 [ms] | OSS CDT [ms] "
             "| err kernel vs theory | err kernel vs OSS EPSD2 | <15% |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["spin_down"]["rows"]:
        def ms(x):
            return "—" if x is None else f"{1e3 * x:.1f}"
        def pc(x):
            return "—" if x is None else f"{100.0 * x:+.2f}%"
        err = r["err_kernel_vs_oss_epsd2"]
        ok = "✅" if (err is not None and err < TOL) else "❌"
        lines.append(f"| {r['mu_r']:.2f} | {ms(r['t_stop_theory_s'])} "
                     f"| {ms(r['t_stop']['kernel'])} | {ms(r['t_stop']['oss_epsd2'])} "
                     f"| {ms(r['t_stop']['oss_cdt'])} | {pc(r['err_vs_theory']['kernel'])} "
                     f"| {pc(err)} | {ok} |")
    pl = res["pair_lever"]
    lines += ["", "### E3 · grain–grain pair spin relaxation (reff = r vs r/2)", "",
              "| arm | t_half [ms] | theory [ms] | err vs OSS |", "|---|---|---|---|"]
    for k in ("kernel_reff_r", "oss_epsd2_reff_r2", "kernel_mapped_mu_half",
              "oss_cdt_reff_r2"):
        th = pl['t_half'][k]; tt = pl['t_half_theory'][k]
        e = pl["err_vs_oss"][k]
        lines.append(f"| {k} | {'—' if th is None else f'{1e3*th:.2f}'} "
                     f"| {1e3*tt:.2f} | {'—' if e is None else f'{100*e:+.2f}%'} |")
    inc = res["incline"]
    lines += ["", f"### E4 · static hold on an incline (mu_r = {inc['mu_r']:.2f}, "
              f"dt = {inc['dt_s']:g} s, theta_c = {inc['theta_c_deg']:.2f} deg)", "",
              "| slope [deg] | tan(theta) | below theta_c | kernel |v| | OSS EPSD2 |v| "
              "| OSS CDT |v| |", "|---|---|---|---|---|---|"]
    for r in inc["rows"]:
        lines.append(f"| {r['slope_deg']:.0f} | {r['tan_theta']:.3f} "
                     f"| {'yes' if r['below_static_threshold'] else 'no'} "
                     f"| {abs(r['kernel']['v_final']):.2e} "
                     f"| {abs(r['oss_epsd2']['v_final']):.2e} "
                     f"| {abs(r['oss_cdt']['v_final']):.2e} |")
    tor = res["torsion"]
    lines += ["", "### E5 · drilling spin (mu_r = %.2f, w0 = %.0f rad/s, 50 ms)" %
              (tor["mu_r"], tor["w0"]), "",
              "| arm | w_end [rad/s] | spin lost |", "|---|---|---|",
              f"| kernel (torsion in rolling) | {tor['w_end_kernel_torsion_on']:.3f} "
              f"| {100*tor['spin_lost_fraction_kernel']:.2f}% |",
              f"| OSS default (torsion removed) | {tor['w_end_oss_default']:.3f} "
              f"| {100*tor['spin_lost_fraction_oss']:.2f}% |",
              f"| production kernel (wall_contact_forces) | "
              f"{tor['w_end_production_kernel']:.3f} "
              f"| {100*tor['spin_lost_fraction_production']:.2f}% |"]
    ds = res["dt_sensitivity"]
    lines += ["", f"### E7 · step-size sensitivity of t_stop (mu_r = {ds['mu_r']:.2f}, "
              f"theory {1e3*ds['theory_s']:.1f} ms)", "",
              "| dt [s] | kernel [ms] | OSS EPSD2 [ms] | OSS CDT [ms] |", "|---|---|---|---|"]
    for k, v in ds["t_stop"].items():
        cells = " | ".join("—" if v[m] is None else f"{1e3*v[m]:.2f}"
                           for m in ("kernel", "oss_epsd2", "oss_cdt"))
        lines.append(f"| {k.replace('dt_', '')} | {cells} |")
    lines.append("")
    lines.append("* relative spread over the sweep: " + ", ".join(
        f"{k} {100*v:+.2f}%" for k, v in ds["spread_fraction_over_dt_sweep"].items()
        if v is not None))
    ts = res["timestep"]
    lines += ["", "### E6 · timestep criterion", "",
              f"* our criterion (20 steps / normal period): dt = {ts['dt_case_s']:.3e} s",
              f"* OSS Rayleigh criterion at the Hertz-equivalent shear modulus "
              f"G = {ts['hertz_equivalent_G_Pa']:.3e} Pa: dt_R = {ts['dt_rayleigh_s']:.3e} s",
              f"* dt / dt_R = {ts['fraction_of_rayleigh']:.2f} "
              f"(OSS practice band {ts['oss_recommended_band']})",
              "", "### E1 · kernel law vs OSS EPSD2 transcription (wall contact)", "",
              f"* max |omega_kernel - omega_oss| = "
              f"{res['kernel_identity']['max_abs_diff_omega']:.3e} rad/s "
              f"over {res['kernel_identity']['t_end']:.3f} s"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "table_main.md"), "w") as fh:
        fh.write(text)
    return text


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    res = {"tolerance": TOL, "base_material": BASE}
    res["kernel_identity"] = experiment_kernel_identity()
    res["spin_down"] = experiment_spin_down((0.1, 0.2, 0.3, 0.5, 0.8))
    res["pair_lever"] = experiment_pair_lever()
    res["incline"] = experiment_incline()
    res["incline_dt"] = experiment_incline_dt()
    res["torsion"] = experiment_torsion()
    res["timestep"] = experiment_timestep()
    res["dt_sensitivity"] = experiment_dt_sensitivity()
    res["figures"] = make_figures(res)

    traces = {}
    for key in ("spin_down", "pair_lever"):
        blob = res.get(key, {})
        if "_traces" in blob:
            traces[key] = {k: {kk: list(map(float, vv)) for kk, vv in v.items()}
                           for k, v in blob.pop("_traces").items()}
    if "_traces" in res.get("torsion", {}):
        blob = res["torsion"].pop("_traces")
        traces["torsion"] = {k: [float(x) for x in v] for k, v in blob.items()}
    if traces:
        with open(os.path.join(OUT, "traces.json"), "w") as fh:
            json.dump(traces, fh)
    # strip nested traces from the summary file
    for row in res["spin_down"]["rows"]:
        row.pop("_traces", None)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    table = write_table(res)
    print(table)
    print("artifacts ->", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()