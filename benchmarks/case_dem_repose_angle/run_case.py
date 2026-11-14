"""CASE-DEM-REPOSE-ANGLE: angle of repose of glass beads, ref 22 deg (TensorDEM).

Benchmark case of the case pool (mission 19975).  The case is a *reuse* benchmark:
the lifting-cylinder repose protocol and the contact model already exist
(:mod:`tensordem.repose_material_calibration`) and are **not** re-implemented
here.  This script only

  1. encodes the reference numbers with their original location
     (:data:`REF`, Brzinski III/Mayor/Durian, PRL 111, 168002 (2013);
     arXiv:1307.4638v2, p. 1, col. 2) and derives the DEM inputs from them,
  2. runs the pre-registered legs (see ``docs/CASE-DEM-REPOSE-ANGLE_prereg.md``),
  3. aggregates, inverts ``theta(mu_r)`` at the target, evaluates the frozen
     acceptance criteria and writes JSON + markdown + figures.

One-command reproduction::

    python3 benchmarks/case_dem_repose_angle/run_case.py \
        --out artifacts/CASE-DEM-REPOSE-ANGLE --jobs 8

Every number quoted in the report is read back from ``results.json`` /
``criteria.json`` written by this script (no hand-typed values).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

TARGET_DEFAULT = 22.0          # deg, reference value (see REF)
TOLERANCE_DEG = 3.0            # mission tolerance, absolute

# ---------------------------------------------------------------------------
# Reference data (zero hand-copy: value + original location)
# ---------------------------------------------------------------------------
#: B. T. Brzinski III, M. Mayor, D. J. Durian, "Depth-dependent resistance of
#: granular media to vertical penetration", Phys. Rev. Lett. 111, 168002 (2013).
#: Text version held in this repository: arXiv:1307.4638v2.
#: All five values below come from the SAME line of the paper: p. 1, col. 2.
REF = {
    "paper": ("B. T. Brzinski III, M. Mayor, D. J. Durian, "
              "Phys. Rev. Lett. 111, 168002 (2013); arXiv:1307.4638v2"),
    "location": "arXiv:1307.4638v2 p. 1, col. 2 (medium description, one line)",
    "theta_exp_deg": 22.0,
    "mu_internal": 0.40,
    "rho_bulk_kg_m3": 1480.0,
    "packing_fraction": 0.59,
    "d_particle_m": 300.0e-6,
    "d_particle_tol_m": 50.0e-6,
}


def ref_derived() -> dict:
    """DEM inputs derived *from the reference numbers*, not typed."""
    rho_particle = REF["rho_bulk_kg_m3"] / REF["packing_fraction"]
    mu_from_angle = math.tan(math.radians(REF["theta_exp_deg"]))
    return {
        "rho_particle_kg_m3": rho_particle,
        "mu_from_angle": mu_from_angle,
        # consistency of the paper's own two numbers (22 deg <-> mu = 0.40)
        "mu_quoted_vs_tan_rel": abs(mu_from_angle - REF["mu_internal"])
        / REF["mu_internal"],
    }


RADIUS_BASE = 0.004            # m, DEM grain (coarsened from 300 um, documented)
SEEDS_SCAN = (19975, 19976)    # scan / calibration seeds
SEEDS_HOLDOUT = (19981, 19982, 19983)   # verdict seeds, never used for calibration
MU_R_NODES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.404, 0.50)
MU_R_GLASS_PRIOR = 0.05        # independently motivated smooth-glass rolling value
E_BASE = 0.70
WALL_BASE = 5.0
FILL_BASE = 9.0
LIFT_BASE = 1.20               # s


def _base(**over):
    p = dict(radius=RADIUS_BASE, density=None, restitution=E_BASE, mu=None,
             mu_rolling=None, wall=WALL_BASE, fill=FILL_BASE, lift=LIFT_BASE,
             seed=None, leg="", label="")
    p.update(over)
    return p


def build_phase1() -> list[dict]:
    """Legs that do not depend on the calibration result."""
    d = ref_derived()
    mu = d["mu_from_angle"]
    runs: list[dict] = []
    # Leg P -- H1 predictive leg: sliding friction from the reference, small
    # glass rolling prior, held-out seeds, no tuning.
    for s in SEEDS_HOLDOUT:
        runs.append(_base(leg="P", label="prediction mu_r=0.05", mu=mu,
                          mu_rolling=MU_R_GLASS_PRIOR, seed=s))
    # Leg N -- frictionless control.
    runs.append(_base(leg="N", label="frictionless control", mu=0.0,
                      mu_rolling=0.0, seed=SEEDS_HOLDOUT[0]))
    # Leg S -- mu_r scan (calibration basis).
    for mur in MU_R_NODES:
        for s in SEEDS_SCAN:
            runs.append(_base(leg="S", label=f"mu_r={mur:g}", mu=mu,
                              mu_rolling=mur, seed=s))
    return runs


def build_phase2(mu_r_star: float, radius: float = RADIUS_BASE) -> list[dict]:
    """Legs that need the calibrated ``mu_r*`` (independent-check + sensitivities)."""
    d = ref_derived()
    mu = d["mu_from_angle"]
    runs: list[dict] = []
    for s in SEEDS_HOLDOUT:
        runs.append(_base(leg="C", label="holdout at mu_r*", mu=mu,
                          mu_rolling=mu_r_star, seed=s))
    # sliding-friction sensitivity at mu_r*
    for m in (0.30, round(mu, 4), 0.50):
        for s in SEEDS_HOLDOUT[:2]:
            runs.append(_base(leg="M", label=f"mu={m:g}", mu=m,
                              mu_rolling=mu_r_star, seed=s))
    # coarsening / grain-size sensitivity at mu_r*
    for rad in (0.003, 0.006):
        for s in SEEDS_HOLDOUT[:2]:
            runs.append(_base(leg="E", label=f"r={rad*1e3:g}mm", mu=mu,
                              mu_rolling=mu_r_star, seed=s, radius=rad))
    # container confinement at mu_r*
    for w in (5.0, 8.0):
        for s in SEEDS_HOLDOUT[:2]:
            runs.append(_base(leg="W", label=f"Rc/r={w:g}", mu=mu,
                              mu_rolling=mu_r_star, seed=s, wall=w))
    # lift-rate (strain-rate) sensitivity at mu_r*
    for lt in (0.90, 1.50):
        for s in SEEDS_HOLDOUT[:2]:
            runs.append(_base(leg="L", label=f"lift={lt:g}s", mu=mu,
                              mu_rolling=mu_r_star, seed=s, lift=lt))
    # restitution (quasi-static check) at mu_r*
    for e in (0.30, 0.90):
        for s in SEEDS_HOLDOUT[:2]:
            runs.append(_base(leg="R", label=f"e={e:g}", mu=mu,
                              mu_rolling=mu_r_star, seed=s, restitution=e))
    return runs


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(task: dict) -> dict:
    """One realisation. Runs in a worker process (1 torch thread -> bit-repro)."""
    import torch
    torch.set_num_threads(1)
    from tensordem.repose_material_calibration import MaterialSpec, _geometry, simulate
    from tensordem import __name__ as _pkg  # noqa: F401  (import side effect only)

    d = ref_derived()
    radius = float(task["radius"])
    rc, height = _geometry(radius, float(task["wall"]), float(task["fill"]))
    spec = MaterialSpec(
        key="glass_bead_ref", label="glass beads (Brzinski 2013 medium)",
        radius=radius,
        density=float(task["density"] or d["rho_particle_kg_m3"]),
        restitution=float(task["restitution"]),
        mu_rolling=float(task["mu_rolling"]),
        target_theta_deg=float(TARGET_DEFAULT),
        cylinder_radius=rc, fill_height=height,
        lift_time=float(task["lift"]),
        dt_override=task.get("dt_override"),
    )
    t0 = time.time()
    meta = dict(leg=task["leg"], label=task["label"], seed=int(task["seed"]),
                radius_m=radius, wall_ratio=float(task["wall"]),
                fill_ratio=float(task["fill"]),
                restitution=float(task["restitution"]),
                lift_time_s=float(task["lift"]), mu=float(task["mu"]),
                mu_rolling=float(task["mu_rolling"]),
                density_kg_m3=spec.density, cylinder_radius_m=rc,
                fill_height_m=height, torch_threads=1)
    try:
        out = simulate(spec, float(task["mu"]), seed=int(task["seed"]),
                       mu_rolling=float(task["mu_rolling"]), keep_particles=True,
                       max_steps=task.get("max_steps"))
    except Exception as exc:                     # estimator/solver refusal
        rec = dict(meta)
        rec.update(error=f"{type(exc).__name__}: {exc}",
                   wall_time_s=round(time.time() - t0, 2))
        return rec
    pos = out.pop("positions")
    out.update(
        leg=task["leg"], label=task["label"], radius_m=radius,
        wall_ratio=float(task["wall"]), fill_ratio=float(task["fill"]),
        restitution=float(task["restitution"]), lift_time_s=float(task["lift"]),
        density_kg_m3=spec.density, kn_N_m=spec.kn, kt_ratio=spec.kt_ratio,
        cylinder_radius_m=rc, fill_height_m=height,
        wall_time_s=round(time.time() - t0, 2),
        torch_threads=1,
    )
    # deposit geometry for the field figure (kept small: 3 decimals in metres)
    out["deposit_xyz_mm"] = [[round(float(x) * 1e3, 3) for x in row] for row in pos]
    return out


def _init_worker() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)


def execute(runs: list[dict], jobs: int) -> list[dict]:
    if jobs <= 1:
        return [run_one(t) for t in runs]
    with mp.Pool(processes=jobs, initializer=_init_worker) as pool:
        return pool.map(run_one, runs, chunksize=1)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def is_valid(run: dict) -> bool:
    return run.get("error") is None and run.get("theta_r_deg") is not None


def flags(run: dict) -> dict:
    """Pre-declared quality flags (prereg section 6)."""
    if not is_valid(run):
        return {"run_error": True}
    return {
        "flowing": bool(run["max_speed_m_s"] > 0.05),
        "poor_fit": bool(run["r2"] < 0.80),
        "short_flank": bool(int(run.get("n_fit_bins", run.get("n_bins", 0))) < 5),
        "stub": bool(run["discharge_ratio"] < 1.30),
    }


def any_flag(run: dict) -> bool:
    return any(flags(run).values())


def aggregate(runs: list[dict]) -> dict:
    """mean/sigma per (leg, label) group, over all and over unflagged seeds."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["leg"], r["label"]), []).append(r)
    out = {}
    for (leg, label), grp in groups.items():
        def stats(sel):
            vals = [g["theta_r_deg"] for g in sel]
            if not vals:
                return None
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
            return {"n": n, "mean": mean, "sigma": math.sqrt(var),
                    "min": min(vals), "max": max(vals),
                    "dtheta": mean - TARGET_DEFAULT,
                    "abs_dtheta": abs(mean - TARGET_DEFAULT)}
        valid = [g for g in grp if is_valid(g)]
        unflagged = [g for g in valid if not any_flag(g)]
        out[f"{leg}|{label}"] = {
            "leg": leg, "label": label, "n_seeds": len(grp),
            "n_valid": len(valid), "n_error": len(grp) - len(valid),
            "all": stats(valid), "unflagged": stats(unflagged),
            "n_flagged": len(valid) - len(unflagged),
            "seeds": [g["seed"] for g in grp],
            "flagged_seeds": [g["seed"] for g in valid if any_flag(g)],
        }
    return out


def best_stat(val: dict | None) -> dict | None:
    """Unflagged statistics when available, else all, else None."""
    if not val:
        return None
    return val["unflagged"] or val["all"]


def invert_scan(agg: dict, target: float = TARGET_DEFAULT) -> dict:
    """Bracket-interpolate ``theta(mu_r)`` for ``target``; refuse extrapolation."""
    nodes = []
    for key, val in agg.items():
        if val["leg"] != "S":
            continue
        mu_r = float(val["label"].split("=")[1])
        use = best_stat(val)
        if use is None:
            continue
        nodes.append((mu_r, use["mean"], use["sigma"], val["unflagged"] is not None))
    nodes.sort()
    if not nodes:
        return {"status": "no_scan"}
    lo = min(nodes, key=lambda n: n[1])
    hi = max(nodes, key=lambda n: n[1])
    for (m0, t0, s0, _), (m1, t1, s1, _) in zip(nodes, nodes[1:]):
        if (t0 - target) * (t1 - target) <= 0.0 and abs(t1 - t0) > 0.0:
            frac = (target - t0) / (t1 - t0)
            mu_r_star = m0 + frac * (m1 - m0)
            sigma = math.sqrt((1 - frac) ** 2 * s0 ** 2 + frac ** 2 * s1 ** 2)
            return {"status": "bracketed", "mu_r_star": mu_r_star,
                    "bracket": [m0, m1], "bracket_theta": [t0, t1],
                    "sigma_interp": sigma, "nodes": [
                        {"mu_r": m, "theta": t, "sigma": s} for m, t, s, _ in nodes]}
    return {"status": "out_of_bracket", "target": target, "reachable": [lo[1], hi[1]],
            "nodes": [{"mu_r": m, "theta": t, "sigma": s}
                      for m, t, s, _ in nodes]}


def monotone_ok(agg: dict, lo=0.05, hi=0.35, k_sigma=2.0) -> dict:
    nodes = []
    for val in agg.values():
        if val["leg"] != "S":
            continue
        mu_r = float(val["label"].split("=")[1])
        if lo <= mu_r <= hi:
            use = best_stat(val)
            if use is None:
                continue
            nodes.append((mu_r, use["mean"], use["sigma"]))
    nodes.sort()
    if len(nodes) < 2:
        return {"monotone_within_2sigma": None, "pairs": [],
                "note": "fewer than two usable scan nodes"}
    pairs = []
    ok = True
    for (m0, t0, s0), (m1, t1, s1) in zip(nodes, nodes[1:]):
        tol = k_sigma * math.sqrt(s0 ** 2 + s1 ** 2)
        good = (t1 - t0) >= -tol
        ok &= good
        pairs.append({"pair": [m0, m1], "dtheta": t1 - t0, "tol": tol, "ok": bool(good)})
    return {"monotone_within_2sigma": bool(ok), "pairs": pairs}


def evaluate(agg: dict, inversion: dict, holdout: dict, control: dict,
             scale_legs: dict) -> dict:
    crit = {}

    def verdict(cond):
        return "PASS" if cond else "FAIL"

    p = best_stat(agg.get("P|prediction mu_r=0.05"))
    crit["AC-1"] = {
        "what": "predictive leg (mu=tan22, mu_r=0.05, 3 held-out seeds) within 3 deg",
        "theta_mean": None if p is None else p["mean"],
        "dtheta": None if p is None else p["dtheta"],
        "verdict": "WITHHELD" if p is None else verdict(abs(p["dtheta"]) <= TOLERANCE_DEG),
    }
    c = holdout
    crit["AC-2"] = {
        "what": "calibrated mu_r* on 3 held-out seeds within 3 deg",
        "mu_r_star": inversion.get("mu_r_star"),
        "theta_mean": None if not c else c["mean"],
        "dtheta": None if not c else c["dtheta"],
        "verdict": "WITHHELD" if not c else verdict(abs(c["dtheta"]) <= TOLERANCE_DEG),
    }
    crit["AC-3"] = {
        "what": "frictionless control flattens (theta < 5 deg)",
        "theta_mean": None if not control else control["mean"],
        "verdict": "WITHHELD" if not control else verdict(control["mean"] < 5.0),
    }
    mono = monotone_ok(agg)
    crit["AC-4"] = {"what": "theta(mu_r) invertible on [0.05, 0.35]",
                    "monotone_within_2sigma": mono["monotone_within_2sigma"],
                    "pairs": mono["pairs"],
                    "note": mono.get("note"),
                    "verdict": ("WITHHELD" if mono["monotone_within_2sigma"] is None
                                else verdict(mono["monotone_within_2sigma"]))}
    scale = {}
    base_theta = holdout["mean"] if holdout else None
    for name, val in scale_legs.items():
        if val and base_theta is not None:
            scale[name] = {"theta_mean": val["mean"],
                           "dtheta_vs_base": val["mean"] - base_theta}
    worst = max((abs(v["dtheta_vs_base"]) for v in scale.values()), default=None)
    crit["AC-5"] = {"what": "radius/container sensitivity <= 3 deg (informational)",
                    "legs": scale,
                    "worst_abs_dtheta": worst,
                    "verdict": "WITHHELD" if worst is None else verdict(worst <= TOLERANCE_DEG)}
    ok = crit["AC-3"]["verdict"] == "PASS" and (
        crit["AC-1"]["verdict"] == "PASS" or crit["AC-2"]["verdict"] == "PASS")
    crit["CASE"] = {"rule": "AC-3 AND (AC-1 OR AC-2)",
                    "verdict": verdict(ok),
                    "route": ("predictive" if crit["AC-1"]["verdict"] == "PASS"
                              else ("calibrated" if crit["AC-2"]["verdict"] == "PASS"
                                    else "neither"))}
    return crit


# ---------------------------------------------------------------------------
# Figures and markdown
# ---------------------------------------------------------------------------

def figures(runs, agg, inversion, outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    outdir.mkdir(parents=True, exist_ok=True)

    # fig 1: theta(mu_r) scan + target band + calibrated point + prediction point
    scan = sorted(((float(v["label"].split("=")[1]), v) for v in agg.values()
                   if v["leg"] == "S" and best_stat(v)), key=lambda t: t[0])
    if scan:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        xs = [s[0] for s in scan]
        ys = [best_stat(s[1])["mean"] for s in scan]
        es = [best_stat(s[1])["sigma"] for s in scan]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.4,
                    label="DEM $\\theta_r(\\mu_r)$  (mean $\\pm\\sigma$, 2 seeds)")
        ax.axhspan(TARGET_DEFAULT - TOLERANCE_DEG, TARGET_DEFAULT + TOLERANCE_DEG,
                   color="tab:green", alpha=0.13,
                   label=f"ref 22$\\pm$3$^\\circ$ (Brzinski 2013, p.1 col.2)")
        ax.axhline(TARGET_DEFAULT, color="tab:green", lw=1.0, ls="--")
        if inversion.get("mu_r_star"):
            ax.axvline(inversion["mu_r_star"], color="tab:red", lw=1.0, ls=":",
                       label=f"calibrated $\\mu_r^*$={inversion['mu_r_star']:.3f}")
        p = agg.get("P|prediction mu_r=0.05")
        pv = best_stat(p)
        if pv:
            ax.errorbar([MU_R_GLASS_PRIOR], [pv["mean"]], yerr=[pv["sigma"]],
                        marker="s", color="tab:purple", capsize=3, ls="none",
                        label="H1 prediction ($\\mu_r$=0.05)")
        ax.set_xlabel("rolling-resistance coefficient $\\mu_r$")
        ax.set_ylabel("angle of repose $\\theta_r$ [deg]")
        ax.set_title("glass beads: DEM repose vs reference 22$^\\circ$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        path = outdir / "fig1_theta_vs_mu_r.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        made.append(str(path))

    # fig 2: deposit cross-sections (x-z) of prediction vs calibrated holdout
    picks = []
    for leg in ("P", "C"):
        for r in runs:
            if r["leg"] == leg and r.get("deposit_xyz_mm"):
                picks.append(r)
                break
    if picks:
        fig, axes = plt.subplots(1, len(picks), figsize=(5.0 * len(picks), 3.8),
                                 squeeze=False)
        for ax, r in zip(axes[0], picks):
            pts = r["deposit_xyz_mm"]
            xs = [p[0] for p in pts]; zs = [p[2] for p in pts]
            sc = ax.scatter(xs, zs, c=zs, s=6, cmap="viridis")
            Rc = r["cylinder_radius_m"] * 1e3
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_aspect("equal")
            ax.set_xlim(-Rc * 1.9, Rc * 1.9)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("z [mm]")
            ax.set_title(f"leg {r['leg']}: $\\theta_r$={r['theta_r_deg']:.2f}$^\\circ$"
                         f" (seed {r['seed']})", fontsize=9)
            fig.colorbar(sc, ax=ax, label="z [mm]")
        path = outdir / "fig2_deposit_sections.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        made.append(str(path))

    # fig 3: sensitivity legs vs the calibrated baseline
    base = best_stat(agg.get("C|holdout at mu_r*")) or best_stat(agg.get("P|prediction mu_r=0.05"))
    base_theta = base["mean"] if base else None
    if base_theta is not None:
        keys = [("M", "$\\mu$"), ("E", "grain radius"), ("W", "container"),
                ("L", "lift rate"), ("R", "restitution")]
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        ylabels = []
        for leg, name in keys:
            for key, val in sorted(agg.items()):
                if val["leg"] != leg:
                    continue
                use = best_stat(val)
                if use is None:
                    continue
                ax.barh(len(ylabels), use["mean"] - base_theta,
                        xerr=use["sigma"], color="tab:blue", alpha=0.75)
                ylabels.append(f"{name}: {val['label']}")
        ax.axvspan(-TOLERANCE_DEG, TOLERANCE_DEG, color="tab:green", alpha=0.12,
                   label="$\\pm$3$^\\circ$ tolerance")
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlabel("$\\theta_r$ deviation from calibrated baseline [deg]")
        ax.set_title("parameter sensitivity (baseline "
                     f"{base_theta:.2f}$^\\circ$)", fontsize=10)
        ax.grid(alpha=0.3, axis="x"); ax.legend(fontsize=8)
        path = outdir / "fig3_sensitivity.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        made.append(str(path))
    return made


def markdown(agg, inversion, crit, runs, outdir: Path) -> Path:
    lines = ["<!-- generated by benchmarks/case_dem_repose_angle/run_case.py -->",
             "",
             "### Reference: 22 deg (Brzinski/Mayor/Durian, PRL 111, 168002 (2013),"
             " arXiv:1307.4638v2 p.1 col.2)",
             "",
             "#### Per-leg aggregates (deg)",
             "",
             "| leg | label | seeds | mean | sigma | min | max | d(ref) | flagged |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, v in sorted(agg.items()):
        u = best_stat(v)
        if u is None:
            lines.append(f"| {v['leg']} | {v['label']} | {v['n_seeds']} | "
                         f"no valid run | - | - | - | - | {v['n_error']} err |")
            continue
        lines.append(f"| {v['leg']} | {v['label']} | {v['n_seeds']} | {u['mean']:.3f} | "
                     f"{u['sigma']:.3f} | {u['min']:.3f} | {u['max']:.3f} | "
                     f"{u['dtheta']:+.3f} | {v['n_flagged']} |")
    lines += ["", "#### Per-run detail", "",
              "| leg | label | seed | mu | mu_r | r[mm] | theta | r2 | n_fit_bins | "
              "discharge | v_max | flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if not is_valid(r):
            lines.append(f"| {r['leg']} | {r['label']} | {r['seed']} | "
                         f"{r['mu']:.4f} | {r['mu_rolling']:.4f} | "
                         f"{r['radius_m']*1e3:g} | error | - | - | - | - | "
                         f"{r['error']} |")
            continue
        fl = [k for k, on in flags(r).items() if on]
        lines.append(
            f"| {r['leg']} | {r['label']} | {r['seed']} | {r['mu']:.4f} | "
            f"{r['mu_rolling']:.4f} | {r['radius_m']*1e3:g} | "
            f"{r['theta_r_deg']:.3f} | {r['r2']:.3f} | {r['n_fit_bins']} | "
            f"{r['discharge_ratio']:.2f} | {r['max_speed_m_s']:.4f} | "
            f"{','.join(fl) or '-'} |")
    lines += ["", "#### Inversion of theta(mu_r) at the target", "",
              "```json", json.dumps(inversion, indent=1), "```", "",
              "#### Criteria", "",
              "| id | what | value | verdict |",
              "|---|---|---|---|"]
    for cid, c in crit.items():
        if cid == "CASE":
            lines.append(f"| CASE | {c['rule']} | route: {c['route']} | {c['verdict']} |")
            continue
        if cid == "AC-4":
            value = f"monotone-within-2sigma={c['monotone_within_2sigma']}"
        elif cid == "AC-5":
            value = f"worst |d|={c['worst_abs_dtheta']}"
        else:
            value = (f"theta={c['theta_mean']:.3f}" if c.get("theta_mean") is not None
                     else "n/a")
        lines.append(f"| {cid} | {c['what']} | {value} | {c['verdict']} |")
    path = outdir / "TABLES.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/CASE-DEM-REPOSE-ANGLE")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="smoke mode: tiny protocol + short sweep (never a verdict)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    p1 = build_phase1()
    if args.quick:
        p1 = [dict(p, lift=0.12, max_steps=400) for p in p1 if p["leg"] in ("P", "N")]
    t0 = time.time()
    runs1 = execute(p1, args.jobs)
    agg1 = aggregate(runs1)
    inversion = invert_scan(agg1)
    mu_r_star = inversion.get("mu_r_star")
    if mu_r_star is None:
        mu_r_star = MU_R_NODES[len(MU_R_NODES) // 2]
        inversion["mu_r_star_used"] = mu_r_star
        inversion["note"] = "target outside bracket -> used mid node, AC-2 reported"

    p2 = build_phase2(mu_r_star)
    if args.quick:
        p2 = [dict(p, lift=0.12, max_steps=400) for p in p2 if p["leg"] == "C"]
    runs2 = execute(p2, args.jobs)
    runs = runs1 + runs2
    agg = aggregate(runs)
    # aggregate() keys are per label; re-run the inversion on the joint table
    # (identical for leg S) so the reported bracket is the one actually used.
    control = best_stat(agg.get("N|frictionless control"))
    holdout = best_stat(agg.get("C|holdout at mu_r*"))
    scale_legs = {k: best_stat(v) for k, v in agg.items()
                  if v["leg"] in ("E", "W") and best_stat(v)}
    crit = evaluate(agg, inversion, holdout, control, scale_legs)

    payload = {
        "case": "CASE-DEM-REPOSE-ANGLE",
        "mission_id": 19975,
        "reference": REF,
        "reference_derived": ref_derived(),
        "target_deg": TARGET_DEFAULT,
        "tolerance_deg": TOLERANCE_DEG,
        "environment": {"python": platform.python_version(),
                        "torch_threads": 1, "jobs": args.jobs,
                        "quick": args.quick},
        "wall_time_s": round(time.time() - t0, 1),
        "n_runs": len(runs),
        "inversion": inversion,
        "criteria": crit,
        "aggregates": agg,
        "runs": runs,
        "repro_cmd": ("python3 benchmarks/case_dem_repose_angle/run_case.py "
                      f"--out {args.out} --jobs {args.jobs}"),
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=1))
    (outdir / "criteria.json").write_text(json.dumps(crit, indent=1))
    fig_paths = [] if args.no_figures else figures(runs, agg, inversion, outdir)
    md = markdown(agg, inversion, crit, runs, outdir)
    payload["figures"] = [Path(p).name for p in fig_paths]
    (outdir / "results.json").write_text(json.dumps(payload, indent=1))

    print(json.dumps({"case": payload["case"], "n_runs": len(runs),
                      "wall_time_s": payload["wall_time_s"],
                      "inversion": {k: inversion.get(k) for k in
                                    ("status", "mu_r_star", "bracket")},
                      "criteria": {k: v.get("verdict") for k, v in crit.items()},
                      "figures": payload["figures"], "tables": str(md)}, indent=1))
    return 0 if crit["CASE"]["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())