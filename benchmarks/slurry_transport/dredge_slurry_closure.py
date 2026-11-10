#!/usr/bin/env python
"""Dredged-slurry pipeline transport closure benchmark (campaign p246w2-n03).

Three run modes
---------------
``static``   A1..A5 + A7 acceptance faces, all analytic closures, figures, JSON.
``dem``      one shard of the DEM-CFD sweep (6 cases over 3 shards).
``assemble`` merge the DEM shards, evaluate A6, write the final report JSON.

Usage
-----
    python dredge_slurry_closure.py --mode static --out runs/slurry
    python dredge_slurry_closure.py --mode dem --shard 0 --nshards 3 --out runs/slurry
    python dredge_slurry_closure.py --mode assemble --out runs/slurry

Every number in the output JSON is produced here (no hand transcription).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensordem.slurry_transport import verification as V          # noqa: E402
from tensordem.slurry_transport.closures import (                # noqa: E402
    MU_W, RHO_W, terminal_velocity_numeric, terminal_velocity_rubey,
    terminal_velocity_stokes)
from tensordem.slurry_transport.critical_velocity import (       # noqa: E402
    cleanaway_deposition_velocity, durand_deposition_velocity,
    durand_fl)
from tensordem.slurry_transport.pressure_drop import (           # noqa: E402
    colebrook_friction, haaland_friction, mixture_gradient, pressure_drop,
    swamee_jain_friction)
from tensordem.slurry_transport.pump_pipe import (               # noqa: E402
    PipeSpec, PumpSpec, SlurrySpec, operating_point, pump_head, system_head)

#: DEM sweep definition: (label, V/Vc, cv, Ss)
DEM_CASES = [
    ("vv1p2_cv10", 1.20, 0.10, 2.65),
    ("vv1p6_cv10", 1.60, 0.10, 2.65),
    ("vv2p0_cv10", 2.00, 0.10, 2.65),
    ("vv1p6_cv05", 1.60, 0.05, 2.65),
    ("vv1p6_cv20", 1.60, 0.20, 2.65),
    ("vv1p6_neutral", 1.60, 0.10, 1.00),
]


# ---------------------------------------------------------------------------
# static faces
# ---------------------------------------------------------------------------
def run_static(out):
    t0 = time.time()
    res = {}
    res["A1"] = V.face_a1()
    res["A2"] = V.face_a2()
    res["A3"] = V.face_a3()
    res["A4"] = V.face_a4()
    res["A5"] = V.face_a5()
    res["A7"] = V.face_a7()

    # design-case table (prototype line) for the report
    c = V.DESIGN_CASE
    pipe = PipeSpec(c["D"], c["L"], eps_D=c["eps_D"], zeta_minor=c["zeta_minor"],
                    H_static=c["H_static"], L_suction=c["L_suction"],
                    h_suction=c["h_suction"])
    pump = PumpSpec(**c["pump"])
    slurry = SlurrySpec(c["Ss"], c["cv"], c["d50"], rho_f=c["rho_f"],
                        mu=c["mu"], n_void=c["n_void"])
    op = operating_point(pipe, pump, slurry)
    # sensitivity of the operating point to kappa (bed-load enhancement)
    sens = {}
    for kappa in (1.0, 1.5, 2.0, 3.0):
        s = operating_point(pipe, pump, slurry, kappa=kappa)
        sens[str(kappa)] = {"Q": s["Q"], "H": s["H"], "V": s["V"],
                            "V_over_Vc": s["V_over_Vc"],
                            "P_shaft": s["P_shaft"],
                            "Q_bank_m3h": s["production"]["Q_bank_m3h"]}
    # pump speed sweep (affinity laws)
    speed = {}
    for n in (0.85, 0.90, 1.00, 1.05):
        try:
            s = operating_point(pipe, pump, slurry, n=n)
            speed[f"{n:.2f}"] = {"Q": s["Q"], "H": s["H"], "V": s["V"],
                                 "P_shaft": s["P_shaft"],
                                 "npsh_margin": s["npsh_margin"],
                                 "Q_bank_m3h": s["production"]["Q_bank_m3h"]}
        except ValueError as exc:
            speed[f"{n:.2f}"] = {"error": str(exc)}
    res["design"] = {
        "case": c, "w_s": slurry.w_s, "rho_m": slurry.rho_m,
        "operating_point": op, "kappa_sensitivity": sens,
        "pump_speed_sweep": speed,
        "elapsed_s": time.time() - t0,
    }
    write_json(os.path.join(out, "slurry_static.json"), res)
    make_static_figures(res, out)
    return res


# ---------------------------------------------------------------------------
# DEM shard
# ---------------------------------------------------------------------------
def run_dem(out, shard, nshards, t_end=3.5, dt=1e-4, labels=None):
    from tensordem.slurry_transport.pipe_dem import dem_transport_case

    D, W, d = V.DEM_CASE["D"], V.DEM_CASE["W"], V.DEM_CASE["d"]
    rho, mu = V.DEM_CASE["rho_f"], V.DEM_CASE["mu"]
    cases = [c for k, c in enumerate(DEM_CASES) if k % nshards == shard]
    if labels:
        keep = set(labels.split(","))
        cases = [c for c in cases if c[0] in keep]
    results = {}
    for label, ratio, cv, Ss in cases:
        # the reference deposition velocity is always evaluated at the sand
        # density (Ss = 1 has no deposition velocity by construction)
        vc = float(durand_deposition_velocity(D, 2.65, cv, d))
        V_flow = ratio * vc
        t0 = time.time()
        res = dem_transport_case(V_flow, D=D, W=W, d=d, Ss=Ss, cv=cv,
                                 rho_f=rho, mu=mu, t_end=t_end, dt=dt, seed=0)
        res["label"] = label
        res["V_over_Vc_design"] = ratio
        res["elapsed_s"] = time.time() - t0
        results[label] = res
        print(f"[shard {shard}] {label}: V={V_flow:.3f} Vc={vc:.3f} "
              f"i_e={res['i_e']:.5f} kappa={res['kappa']:.3f} "
              f"ratio={res['bottom_top_ratio']:.3f} "
              f"drift={res.get('i_e_drift', float('nan')):.3f} "
              f"({res['elapsed_s']:.0f}s)", flush=True)
        write_json(os.path.join(out, f"dem_{label}.json"), res)
    write_json(os.path.join(out, f"dem_shard{shard}.json"),
               {"cases": list(results), "nshards": nshards, "shard": shard})
    return results


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def run_assemble(out):
    static = read_json(os.path.join(out, "slurry_static.json"))
    dem = {}
    for label, *_ in DEM_CASES:
        path = os.path.join(out, f"dem_{label}.json")
        if os.path.exists(path):
            dem[label] = read_json(path)
    if "vv1p6_cv10" not in dem:
        raise SystemExit("primary DEM case vv1p6_cv10 missing - run --mode dem")
    res = dict(static)
    res["A6"] = V.face_a6(dem["vv1p6_cv10"])
    res["A6"]["all_cases"] = {
        label: {
            "V": r["U"], "cv": r["cv"], "Ss": r.get("Ss", 2.65),
            "i_e_dem": r["i_e"], "i_e_drag": r.get("i_e_drag"),
            "i_m": r.get("i_m"), "i_w": r.get("i_w"), "kappa": r["kappa"],
            "i_e_ideal": r["i_e_ideal"],
            "bottom_top_ratio": r["bottom_top_ratio"],
            "profile_skewness": r["profile_skewness"],
            "bed_fraction": r["bed_fraction"],
            "u_p_mean": r["u_p_mean"], "slip": r["slip"],
            "i_e_drift": r.get("i_e_drift"),
            "samples": r["samples"],
        } for label, r in dem.items()}
    # A7 DEM leg: neutral-density particles
    if "vv1p6_neutral" in dem:
        neu = dem["vv1p6_neutral"]
        # two admissible definitions of the "excess" are reported:
        #  * closure-consistent (Durand style): i_m - i_w^clear water
        #  * drag work: the momentum the carrier gives up to the solids
        res["A7"]["dem"] = {
            "i_e_closure_consistent": neu["i_e"], "i_w": neu["i_w"],
            "relative_excess_closure_consistent": float(neu["i_e"] / neu["i_w"]),
            "i_e_drag": neu.get("i_e_drag"),
            "relative_excess_drag": float(neu.get("i_e_drag", float("nan"))
                                          / neu["i_w"]),
            "i_e_drift": neu.get("i_e_drift"),
            "samples": neu.get("samples"),
            "relative_excess": float(neu.get("i_e_drag", float("nan"))
                                     / neu["i_w"]),
        }
        res["A7"]["pass_dem"] = bool(abs(neu.get("i_e_drag", 1.0)
                                         / neu["i_w"]) < 0.01)
    res["A7"]["pass_analytic"] = static["A7"]["pass_analytic"]
    gate = {"A1": bool(res["A1"]["pass_closed_form"] and res["A1"]["pass_stokes_slope"]),
            "A2": bool(res["A2"]["pass_moody"] and res["A2"]["pass_laminar"]),
            "A3": bool(res["A3"]["pass_dimensional"] and res["A3"]["pass_sqrt_D"]
                       and res["A3"]["pass_adjudication"]),
            "A4": bool(res["A4"]["pass"]),
            "A5": bool(res["A5"]["pass_residual"] and res["A5"]["pass_power"]
                       and res["A5"]["pass_npsh"]),
            "A6": bool(res["A6"]["pass_pressure"] and res["A6"]["pass_asymmetry"]),
            "A7": bool(res["A7"]["pass_analytic"]
                       and res["A7"].get("pass_dem", True))}
    res["gate"] = gate
    res["all_pass"] = all(gate.values())
    res["dem_raw"] = {k: {kk: vv for kk, vv in v.items()
                          if kk not in ("history",)} for k, v in dem.items()}
    write_json(os.path.join(out, "slurry_report.json"), res)
    make_dem_figures(dem, out)
    return res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_static_figures(res, out):
    plt = _mpl()
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Fig 1: suspension velocity vs diameter
    ds = np.geomspace(1e-6, 3e-3, 200)
    w_num = np.array([terminal_velocity_numeric(d, 2650.0, RHO_W, MU_W)["w"] for d in ds])
    w_rub = np.array([terminal_velocity_rubey(d, 2650.0, RHO_W, MU_W) for d in ds])
    w_stk = terminal_velocity_stokes(ds, 2650.0, RHO_W, MU_W)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(ds * 1e3, w_num, "k-", label="force balance (Schiller-Naumann)")
    ax[0].loglog(ds * 1e3, w_rub, "b--", label="Rubey closed form")
    ax[0].loglog(ds * 1e3, w_stk, "r:", label="Stokes $d^2$ limit")
    for r in res["A1"]["rows"]:
        ax[0].plot(r["d_mm"], r["w_numeric"], "ko", ms=5)
    ax[0].set_xlabel("particle diameter $d$ [mm]")
    ax[0].set_ylabel("terminal velocity $w_s$ [m/s]")
    ax[0].set_title("A1 suspension velocity: numeric vs closed forms")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].grid(alpha=0.3, which="both")
    # slope check in the Stokes limit
    ds_s = np.geomspace(1e-6, 60e-6, 40)
    ws_s = np.array([terminal_velocity_numeric(d, 2650.0, RHO_W, MU_W)["w"] for d in ds_s])
    ax[1].loglog(ds_s * 1e6, ws_s, "k.-")
    ax[1].set_xlabel("particle diameter $d$ [um]")
    ax[1].set_ylabel("$w_s$ [m/s]")
    ax[1].set_title(f"Stokes-limit slope = {res['A1']['stokes_slope']:.3f} (limit 2.0$\\pm$0.1)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_suspension_velocity.png"), dpi=140)
    plt.close(fig)

    # Fig 2: friction factor
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Re = np.geomspace(1e3, 1e7, 200)
    for rr, style in ((1e-4, "-"), (1e-3, "--"), (5e-3, ":")):
        f = np.array([colebrook_friction(r, rr)["f"] for r in Re])
        ax[0].loglog(Re, f, style, label=f"Colebrook $\\epsilon/D$={rr:g}")
        f_sj = np.array([swamee_jain_friction(r, rr) for r in Re])
        ax[0].loglog(Re, f_sj, ".", ms=2, alpha=0.5)
    Re_lam = np.geomspace(10, 2000, 50)
    ax[0].loglog(Re_lam, 64.0 / Re_lam, "k-", lw=2, label="64/Re")
    ax[0].set_xlabel("Re"); ax[0].set_ylabel("Darcy friction factor $f$")
    ax[0].set_title("A2 Colebrook implicit solve (lines) vs Swamee-Jain (dots)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    pts = res["A2"]["moody_points"]
    idx = np.arange(len(pts))
    ax[1].bar(idx - 0.2, [p["err_swamee_jain"] * 100 for p in pts], 0.4,
              label="Swamee-Jain")
    ax[1].bar(idx + 0.2, [p["err_haaland"] * 100 for p in pts], 0.4,
              label="Haaland")
    ax[1].axhline(1.0, color="r", ls="--", label="1% gate")
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels([f"Re={p['Re']:.0e}\n$\\epsilon/D$={p['eps_D']:g}" for p in pts],
                          fontsize=8)
    ax[1].set_ylabel("relative deviation from Colebrook [%]")
    ax[1].set_title("A2 three representative Moody points")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_friction_factor.png"), dpi=140)
    plt.close(fig)

    # Fig 3: deposition velocity vs diameter
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Ds = np.geomspace(0.03, 1.2, 60)
    for cv, style in ((0.05, ":"), (0.15, "-"), (0.30, "--")):
        vd = [float(durand_deposition_velocity(D, 2.585, cv, 0.5e-3)) for D in Ds]
        vc = [float(cleanaway_deposition_velocity(D, 2.585, cv, 0.5e-3)) for D in Ds]
        ax.loglog(Ds, vd, "k" + style, label=f"Durand, $C_v$={cv}")
        ax.loglog(Ds, vc, "r" + style, alpha=0.7, label=f"Cleanaway-type, $C_v$={cv}")
    ax.set_xlabel("pipe diameter $D$ [m]"); ax.set_ylabel("$V_c$ [m/s]")
    ax.set_title(f"A3 $V_c\\propto\\sqrt{{D}}$, slope={res['A3']['slope_durand']['slope']:.3f}")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_deposition_velocity.png"), dpi=140)
    plt.close(fig)

    # Fig 4: dP vs Cv
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    c = V.DESIGN_CASE
    cvs = np.linspace(0.02, 0.32, 60)
    ws = terminal_velocity_numeric(c["d50"], c["Ss"] * c["rho_f"], c["rho_f"],
                                   c["mu"])["w"]
    for V_, style in ((res["A4"]["V"], "-"), (0.7 * res["A4"]["V"], "--"),
                      (1.3 * res["A4"]["V"], ":")):
        dp = [pressure_drop(mixture_gradient(V_, c["D"], cv, c["Ss"], ws,
                                             rho=c["rho_f"], mu=c["mu"])["i_m"],
                            1000.0, rho=c["rho_f"]) / 1e3 for cv in cvs]
        ax.plot(cvs, dp, "k" + style, label=f"V={V_:.2f} m/s")
    ax.axvspan(0.05, 0.30, color="0.85", zorder=0, label="$C_v$ gate band")
    ax.set_xlabel("delivered concentration $C_v$ [-]")
    ax.set_ylabel("$\\Delta P$ over 1000 m [kPa]")
    ax.set_title(f"A4 linearity $R^2$={res['A4']['R2_dense']:.5f} (gate 0.98)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_pressure_vs_concentration.png"), dpi=140)
    plt.close(fig)

    # Fig 5: pump-pipe matching
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pipe = PipeSpec(c["D"], c["L"], eps_D=c["eps_D"], zeta_minor=c["zeta_minor"],
                    L_suction=c["L_suction"], h_suction=c["h_suction"])
    pump = PumpSpec(**c["pump"])
    slurry = SlurrySpec(c["Ss"], c["cv"], c["d50"], rho_f=c["rho_f"],
                        mu=c["mu"], n_void=c["n_void"])
    Qs = np.linspace(0.02, 1.02, 300)
    Hp = [system_head(Q, pipe, slurry) for Q in Qs]
    Hm = [pump_head(Q, pump) for Q in Qs]
    ax.plot(Qs, Hp, "k-", label="pipeline demand $H_{pipe}(Q)$")
    ax.plot(Qs, Hm, "b--", label="pump head $H_{pump}(Q)$")
    op = res["design"]["operating_point"]
    ax.plot(op["Q"], op["H"], "ro", ms=8, label=f"operating point Q*={op['Q']:.3f} m$^3$/s")
    Vc = op["V_c"]
    Qc = Vc * pipe.area
    ax.axvline(Qc, color="g", ls=":", label=f"$V_c$ flow ({Qc:.3f} m$^3$/s)")
    ax.set_xlabel("flow rate $Q$ [m$^3$/s]"); ax.set_ylabel("head [m]")
    ax.set_title("A5 pump-pipeline matching")
    ax.set_ylim(0, 1.15 * max(max(Hp[:60]), pump.H0))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5_pump_matching.png"), dpi=140)
    plt.close(fig)


def make_dem_figures(dem, out):
    plt = _mpl()
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    primary = dem.get("vv1p6_cv10")
    if primary is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for label, r in dem.items():
        z = np.array(r["z_centers"]) / r["geometry"]["D"]
        cvz = np.array(r["cv_z"])
        ax[0].plot(cvz, z, "-", label=label)
    ax[0].axhline(0.5, color="0.6", ls=":", label="mid-height")
    ax[0].set_xlabel("local concentration $C(z)$ [-]")
    ax[0].set_ylabel("$z/D$ [-]")
    ax[0].set_title("A6 DEM concentration profile (asymmetry vs height)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    for label, r in dem.items():
        z = np.array(r["z_centers"]) / r["geometry"]["D"]
        vx = np.array(r["vx_z"]) / r["U"]
        ax[1].plot(vx, z, "-", label=label)
    ax[1].set_xlabel("$u_p(z)/U$ [-]"); ax[1].set_ylabel("$z/D$ [-]")
    ax[1].set_title("A6 particle velocity profile")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig6_dem_profiles.png"), dpi=140)
    plt.close(fig)

    # excess gradient: DEM vs derived suspension closure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    labels = list(dem.keys())
    x = np.arange(len(labels))
    dem_ie = [dem[l]["i_e"] for l in labels]
    ideal = [dem[l]["i_e_ideal"] for l in labels]
    ax[0].bar(x - 0.2, dem_ie, 0.4, label="DEM-CFD measured $i_e$")
    ax[0].bar(x + 0.2, ideal, 0.4, label="suspension closure $C_v(S_s-1)w_s/V$")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=25, fontsize=7)
    ax[0].set_ylabel("excess gradient $i_e$ [-]")
    ax[0].set_title("A6 excess pressure gradient: DEM vs empirical")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
    kappa = [dem[l]["kappa"] for l in labels]
    ax[1].bar(x, kappa, 0.5, color="tab:green")
    ax[1].axhline(1.0, color="k", ls="--", label="$\\kappa=1$ (ideal suspension)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=25, fontsize=7)
    ax[1].set_ylabel("$\\kappa = i_e^{DEM}/i_e^{ideal}$ [-]")
    ax[1].set_title("bed-load enhancement factor")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig7_dem_excess_gradient.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=_default)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("static", "dem", "assemble"), required=True)
    ap.add_argument("--out", default="runs/slurry_p246w2_n03")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--labels", default=None,
                    help="comma separated subset of DEM case labels")
    ap.add_argument("--t-end", type=float, default=3.5)
    ap.add_argument("--dt", type=float, default=1e-4)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "static":
        res = run_static(args.out)
        print(json.dumps({"gate": {k: v["pass"] if isinstance(v, dict) and "pass" in v else None
                                   for k, v in res.items() if k.startswith("A")}},
                         indent=1))
    elif args.mode == "dem":
        run_dem(args.out, args.shard, args.nshards, t_end=args.t_end, dt=args.dt,
                labels=args.labels)
    else:
        res = run_assemble(args.out)
        print(json.dumps(res["gate"], indent=1))


if __name__ == "__main__":
    main()