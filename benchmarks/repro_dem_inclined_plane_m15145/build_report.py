#!/usr/bin/env python
"""Aggregate the chute-flow campaign into the paper-vs-got comparison table.

Reads ``runs/chute/*.json`` (produced by ``run_point.py``), writes
``runs/chute/summary.json`` + ``runs/chute/report.md`` and the figures used by
the mission report.  All numbers in the outputs are read from the run files --
nothing is typed by hand.

Reference values (Silbert et al., Phys. Rev. E 64, 051302 (2001)):

* Bagnold exponent of the velocity profile: Eq. (22) has ``p = 3/2``; the
  paper's measured height scaling is ``v_x(z) ~ H^alpha`` with
  ``alpha = 1.52 +/- 0.05`` (abstract / Sec. IV).
* Flow-cessation angle (3-D, mu = 0.5, e = 0.88, Model L3): ``theta_r,3D ~=
  19.4 deg``; bulk packing ``phi_max,3D = 0.590(5)`` at ``theta_r`` (Fig. 12).
* Bulk packing fraction decreases with tilt angle and is depth-independent
  over ~40 layers (Sec. III A, Fig. 4b).
"""
from __future__ import annotations

import base64
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(ROOT, "runs", "chute")

PAPER = {
    "doi": "10.1103/PhysRevE.64.051302",
    "arxiv": "cond-mat/0105071",
    "p_velocity": 1.5,          # Eq. (22): v = V[1-(1-y/h)^(3/2)]
    "p_tol": 0.20,              # +/-20 % acceptance band (mission criterion)
    "alpha_H": 1.52,
    "alpha_H_sigma": 0.05,
    "theta_r_3d": 19.4,         # deg, Fig. 12 text
    "theta_tol": 2.0,           # +/-2 deg (mission criterion)
    "phi_max_3d": 0.590,
    "model": "L3: kn=2e5, kt/kn=2/7, mu=0.50, e=0.88, no rolling friction",
}


def load_runs() -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name in ("summary",):
            continue
        try:
            with open(path) as fh:
                out[name] = json.load(fh)
        except Exception as exc:                      # pragma: no cover
            out[name] = {"error": str(exc)}
    return out


def _num(d, *keys, default=float("nan")):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarise(runs: dict) -> dict:
    rows = {}
    for name, r in runs.items():
        if "error" in r or "case" not in r:
            rows[name] = {"error": r.get("error", "not a chute run")}
            continue
        case = r["case"]
        fits = r.get("fits", {})
        bulk = r.get("bulk", {})
        met = r.get("metrics", {})
        rows[name] = {
            "theta": case["theta_deg"],
            "H": case["height"],
            "kn": case["kn"],
            "dt": r.get("dt"),
            "n_particles": r.get("n_particles"),
            "steps": r.get("steps"),
            "wall_s": r.get("wall_seconds"),
            "flowing": met.get("flowing"),
            "ke_steady": met.get("ke_steady"),
            "ke_decay_ratio": met.get("ke_decay_ratio"),
            "vx_mean": met.get("vx_mean_steady"),
            "vx_surface": met.get("vx_max_steady"),
            "y_p95": met.get("y_p95_steady"),
            "p_bagnold": fits.get("p"),
            "p_r2": fits.get("r2_best", fits.get("r2")),
            "p_r2_fixed_15": fits.get("r2_p15"),
            "near_base_q": _num(fits, "near_base", "q"),
            "stress_rate_n": _num(fits, "stress_rate", "n"),
            "phi_bulk": bulk.get("phi_bulk"),
            "phi_surface": bulk.get("phi_surface"),
            "phi_max": bulk.get("phi_max"),
            "y_surface": bulk.get("y_surface"),
        }
    return rows


def bagnold_table(rows: dict) -> list[dict]:
    """Velocity-profile check for the steady-flow runs at H = 20."""
    out = []
    for name, r in rows.items():
        if "error" in r or "H" not in r or r["H"] != 20 or r.get("flowing") is not True:
            continue
        if "sens_" in name:
            continue
        p = r["p_bagnold"]
        out.append({
            "run": name,
            "theta_deg": r["theta"],
            "p_got": p,
            "p_ref": PAPER["p_velocity"],
            "rel_error": None if p is None else abs(p - 1.5) / 1.5,
            "verdict": None if p is None else
            ("PASS" if abs(p - 1.5) / 1.5 <= PAPER["p_tol"] else "FAIL"),
            "r2_best": r["p_r2"],
            "r2_fixed_p15": r["p_r2_fixed_15"],
            "near_base_q": r["near_base_q"],
            "stress_rate_n": r["stress_rate_n"],
            "vx_surface": r["vx_surface"],
            "phi_bulk": r["phi_bulk"],
        })
    return out


def alpha_H_table(rows: dict, theta: float = 22.0) -> dict:
    pts = [(r["H"], r["vx_surface"]) for n, r in rows.items()
           if "error" not in r and r["theta"] == theta and "stop_" not in n
           and "sens_" not in n and r.get("flowing") and r["vx_surface"]]
    pts.sort()
    if len(pts) < 2:
        return {"n_points": len(pts), "alpha_got": None, "points": pts}
    h = np.log([p[0] for p in pts])
    v = np.log([p[1] for p in pts])
    a, b = np.polyfit(h, v, 1)
    pred = a * h + b
    var = float(((v - v.mean()) ** 2).sum())
    r2 = 1.0 - float(((v - pred) ** 2).sum()) / var if var > 0 else float("nan")
    return {"n_points": len(pts), "alpha_got": float(a), "r2": r2,
            "alpha_ref": PAPER["alpha_H"],
            "alpha_ref_sigma": PAPER["alpha_H_sigma"],
            "rel_error": abs(a - PAPER["alpha_H"]) / PAPER["alpha_H"],
            "verdict": ("PASS" if abs(a - PAPER["alpha_H"]) <= PAPER["alpha_H_sigma"]
                        else "FAIL"),
            "points": [(float(x), float(y)) for x, y in pts]}


def stop_curve(rows: dict) -> dict:
    """h_stop(theta): largest pile height that does not sustain flow."""
    by_theta: dict[float, list] = {}
    for name, r in rows.items():
        if "error" in r or "stop_" not in name:
            continue
        by_theta.setdefault(r["theta"], []).append((r["H"], r["flowing"],
                                                    r["ke_steady"]))
    curve = {}
    for theta, pts in sorted(by_theta.items()):
        pts.sort()
        flowing_h = [h for h, f, _ in pts if f]
        stopped_h = [h for h, f, _ in pts if not f]
        if flowing_h and stopped_h:
            h_stop = 0.5 * (max(stopped_h) + min(flowing_h))
        elif flowing_h and not stopped_h:
            h_stop = 0.0            # flows down to the thinnest pile tried
        else:
            h_stop = float(max(h for h, _, _ in pts))
        curve[theta] = {"h_stop": h_stop,
                        "points": [{"H": h, "flowing": f, "ke": k}
                                   for h, f, k in pts]}
    thetas = sorted(curve)
    flowing_thetas = [t for t in thetas if any(p["flowing"] for p in curve[t]["points"])]
    return {
        "curve": {str(k): v for k, v in curve.items()},
        "theta_min_flowing": min(flowing_thetas) if flowing_thetas else None,
        "theta_ref": PAPER["theta_r_3d"],
        "theta_tol": PAPER["theta_tol"],
        "verdict": ("PASS" if flowing_thetas
                    and abs(min(flowing_thetas) - PAPER["theta_r_3d"]) <= PAPER["theta_tol"]
                    else "FAIL"),
        "h_stop_monotonic": all(
            curve[a]["h_stop"] >= curve[b]["h_stop"]
            for a, b in zip(thetas, thetas[1:])),
    }


def make_figures(runs: dict, rows: dict, outdir: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                    # pragma: no cover
        return []
    figs = []
    # ---- velocity + packing fraction profiles --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colours = {20: "tab:blue", 22: "tab:orange", 24: "tab:green"}
    for name, r in sorted(runs.items()):
        if "error" in r or "case" not in r or r["case"]["height"] != 20 or "sens_" in name:
            continue
        th = int(round(r["case"]["theta_deg"]))
        prof = r["profiles"]
        y = np.asarray(prof["y_mid"])
        vx = np.asarray(prof["vx"])
        phi = np.asarray(prof["phi"])
        c = colours.get(th, "gray")
        axes[0].plot(vx, y, "o-", ms=3, color=c, label=f"theta={th} deg")
        axes[1].plot(phi, y, "o-", ms=3, color=c, label=f"theta={th} deg")
        fit = r.get("fits", {})
        if fit.get("p"):
            y0, h, V, p = fit["y0"], fit["h"], fit["V_surface"], fit["p"]
            yy = np.linspace(y0, min(h + y0, y.max()), 100)
            shape = 1.0 - np.clip(1.0 - (yy - y0) / h, 0, 1) ** p
            axes[0].plot(V * shape, yy, "--", lw=1, color=c,
                         label=f"fit p={p:.2f}" if th == 20 else None)
    axes[0].set_xlabel("$v_x$ [$\\sqrt{gd}$]")
    axes[0].set_ylabel("y [d]")
    axes[0].set_title("velocity profile (H=20)")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.5, color="k", lw=0.5, ls=":")
    axes[1].set_xlabel("packing fraction $\\phi$")
    axes[1].set_ylabel("y [d]")
    axes[1].set_title("packing-fraction profile (H=20)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(outdir, "profiles.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    figs.append(p1)

    # ---- h_stop curve --------------------------------------------------------
    stop = stop_curve(rows)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for theta, d in sorted(stop["curve"].items(), key=lambda kv: float(kv[0])):
        for pt in d["points"]:
            ax.plot(float(theta), pt["H"], "o" if pt["flowing"] else "x",
                    color="tab:green" if pt["flowing"] else "tab:red", ms=7)
    ax.axvline(PAPER["theta_r_3d"], color="k", ls="--", lw=1,
               label=f"paper $\\theta_r$={PAPER['theta_r_3d']} deg")
    ax.set_xlabel("inclination $\\theta$ [deg]")
    ax.set_ylabel("pile height H [d]")
    ax.set_title("flow / no-flow map (o flowing, x stopped)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(outdir, "h_stop.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    figs.append(p2)

    # ---- KE history ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for name, r in sorted(runs.items()):
        if "error" in r or "case" not in r or "stop_" in name:
            continue
        h = np.array([x["ke_per_particle"] for x in r["history"]])
        t = np.array([x["t"] for x in r["history"]])
        ax.plot(t, h, lw=1, label=name)
    ax.set_xlabel("t [$\\sqrt{d/g}$]")
    ax.set_ylabel("KE per particle")
    ax.set_yscale("log")
    ax.set_title("kinetic-energy history")
    ax.legend(fontsize=6)
    fig.tight_layout()
    p3 = os.path.join(outdir, "ke_history.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    figs.append(p3)
    return figs


def fmt(x, nd=3):
    if x is None:
        return "not_run"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_markdown(rows, bagnold, alpha, stop, figs) -> str:
    L = []
    L.append("# Silbert et al. (2001) chute flow — reproduction summary\n")
    L.append(f"reference: Phys. Rev. E 64, 051302 ({PAPER['doi']}), "
             f"arXiv:{PAPER['arxiv']}; model {PAPER['model']}\n")
    L.append("## Steady-flow runs (H = 20 d, kn = 1e4, dt = T/20)\n")
    L.append("| run | theta | H | N | steps | flowing | vx_surface | p | p rel.err | verdict | phi_bulk |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in sorted(rows.items()):
        if "error" in r:
            L.append(f"| {name} | error: {r['error']} |" + " |" * 9)
            continue
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, fmt(r["theta"], 1), fmt(r["H"], 1), fmt(r["n_particles"]),
            fmt(r["steps"]), fmt(r["flowing"]), fmt(r["vx_surface"], 2),
            fmt(r["p_bagnold"], 3),
            "not_run" if r["p_bagnold"] is None else fmt(abs(r["p_bagnold"] - 1.5) / 1.5),
            "not_run" if r["p_bagnold"] is None else
            ("PASS" if abs(r["p_bagnold"] - 1.5) / 1.5 <= PAPER["p_tol"] else "FAIL"),
            fmt(r["phi_bulk"], 4)))
    L.append("\n## Criterion 1 — Bagnold exponent (v ∝ y^p, paper p = 3/2 ± 20 %)\n")
    for row in bagnold:
        L.append(f"* {row['run']}: p = {fmt(row['p_got'], 3)} "
                 f"(rel.err {fmt(row['rel_error'])}, R²(fit) {fmt(row['r2_best'], 4)}, "
                 f"R²(p=1.5 fixed) {fmt(row['r2_fixed_p15'], 4)}) -> **{row['verdict']}**")
    L.append(f"\n* height scaling at theta = 22 deg: alpha = {fmt(alpha.get('alpha_got'), 3)}"
             f" vs paper {PAPER['alpha_H']}±{PAPER['alpha_H_sigma']} -> "
             f"{alpha.get('verdict')} (n={alpha.get('n_points')})")
    L.append(f"\n## Criterion 2 — flow / no-flow (h_stop), paper theta_r = "
             f"{PAPER['theta_r_3d']} deg ± {PAPER['theta_tol']}\n")
    for theta, d in sorted(stop["curve"].items(), key=lambda kv: float(kv[0])):
        pts = ", ".join(f"H={p['H']:.0f}:{'flow' if p['flowing'] else 'stop'}"
                        for p in d["points"])
        L.append(f"* theta = {theta}: h_stop ≈ {fmt(d['h_stop'], 1)} d  [{pts}]")
    L.append(f"\n* lowest flowing inclination {stop['theta_min_flowing']} deg vs paper "
             f"{PAPER['theta_r_3d']} deg -> **{stop['verdict']}**; "
             f"h_stop monotone in theta: {stop['h_stop_monotonic']}")
    L.append("\n## Criterion 3 — packing-fraction profiles (trend)\n")
    for name, r in sorted(rows.items()):
        if "error" in r or r["H"] != 20:
            continue
        L.append(f"* {name}: phi_bulk = {fmt(r['phi_bulk'], 4)}, "
                 f"phi_surface = {fmt(r['phi_surface'], 4)}, "
                 f"phi_max = {fmt(r['phi_max'], 4)}")
    L.append(f"\n* paper: phi_max,3D = {PAPER['phi_max_3d']} at theta_r and phi decreasing "
             "with tilt (Fig. 12); depth-independent over the bulk (Fig. 4b)")
    L.append("\n## Figures\n")
    for f in figs:
        L.append(f"* `{os.path.relpath(f, ROOT)}`")
    return "\n".join(L) + "\n"


def fig_payloads(figs):
    out = []
    for f in figs:
        with open(f, "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        out.append({"name": os.path.basename(f),
                    "caption": {
                        "profiles.png": "velocity and packing-fraction profiles at H=20 for theta=20/22/24 deg (dashed: Bagnold fit)",
                        "h_stop.png": "flow / no-flow map: o = steady flow, x = arrested; dashed line = paper theta_r = 19.4 deg",
                        "ke_history.png": "kinetic energy per particle vs time for the steady runs (log scale)",
                    }.get(os.path.basename(f), os.path.basename(f)),
                    "png_base64": data})
    return out


def main() -> int:
    runs = load_runs()
    rows = summarise(runs)
    bagnold = bagnold_table(rows)
    alpha = alpha_H_table(rows)
    stop = stop_curve(rows)
    figs = make_figures(runs, rows, RUNS)
    summary = {"paper": PAPER, "runs": rows, "bagnold": bagnold,
               "alpha_H": alpha, "stop": stop,
               "figures": [os.path.relpath(f, ROOT) for f in figs]}
    with open(os.path.join(RUNS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    md = write_markdown(rows, bagnold, alpha, stop, figs)
    with open(os.path.join(RUNS, "report.md"), "w") as fh:
        fh.write(md)
    payload = {"summary": summary, "markdown": md,
               "figures": fig_payloads(figs)}
    with open(os.path.join(RUNS, "report_payload.json"), "w") as fh:
        json.dump(payload, fh)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())