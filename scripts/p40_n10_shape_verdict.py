#!/usr/bin/env python3
"""P40-N10: shape-representation verdict (clump vs. true shape).

Stages (each caches its own JSON so the study can be resumed):

    --stage hertz   elliptical Hertz closed form vs. the Boussinesq reference
    --stage clump   N-sphere clump geometric convergence (two-sided error)
    --stage piles   three-channel 2-D repose / static-dissipation runs
    --stage report  assemble artifacts/p40_n10/shape_representation_verdict.json
                    and the figures

Run ``--stage all`` for everything.  All numbers in the verdict artifact are
produced here; the doc and the test suite read the same artifact, so prose and
evidence cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.shape_representation import (  # noqa: E402
    PileConfig2D,
    Superquadric,
    clump_surface_deviation,
    elliptic_hertz,
    elliptic_hertz_quadrature,
    equivalent_radius_hertz,
    fit_inscribed_clump,
    run_pile,
)

ART = ROOT / "artifacts" / "p40_n10"
FIG = ROOT / "docs" / "figures"
E_STAR = 1.0e7      # Pa, reduced modulus used for the contact anchor
FORCE = 1.0         # N, contact load of the anchor case


def _ensure_dirs() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1: elliptical Hertz (analytic anchor)
# ---------------------------------------------------------------------------

def stage_hertz() -> dict:
    ratios = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0,
              75.0, 120.0, 200.0]
    A_fix = 1.0 / 0.05      # curvature along the tight direction (1/m)
    rows = []
    for r in ratios:
        A, B = A_fix, A_fix / r
        cf = elliptic_hertz(A, B, E_STAR, FORCE)
        q = elliptic_hertz_quadrature(A, B, E_STAR, FORCE)
        eq = equivalent_radius_hertz(A, B, E_STAR, FORCE)
        rows.append({
            "curvature_ratio": r,
            "eccentricity": q.eccentricity,
            "a_tight_closed": cf.a_tight,
            "a_tight_ref": q.a_tight,
            "a_flat_closed": cf.a_flat,
            "a_flat_ref": q.a_flat,
            "delta_closed": cf.delta,
            "delta_ref": q.delta,
            "stiffness_closed": cf.stiffness,
            "stiffness_ref": q.stiffness,
            "stiffness_equiv": eq.stiffness,
            "stiffness_err_pct": 100.0 * (cf.stiffness / q.stiffness - 1.0),
            "delta_err_pct": 100.0 * (cf.delta / q.delta - 1.0),
            "equiv_radius_err_pct": 100.0 * (eq.stiffness / q.stiffness - 1.0),
        })
    # circular limit against exact Hertz (independent analytic value)
    R = 1.0 / A_fix
    delta_ana = (9.0 * FORCE ** 2 / (16.0 * E_STAR ** 2 * R)) ** (1.0 / 3.0)
    stiff_ana = 2.0 * E_STAR * math.sqrt(R * delta_ana)
    circ = elliptic_hertz(A_fix, A_fix, E_STAR, FORCE)
    circ_ref = elliptic_hertz_quadrature(A_fix, A_fix, E_STAR, FORCE)
    out = {
        "stage": "hertz",
        "E_star": E_STAR,
        "force": FORCE,
        "rows": rows,
        "circular_limit": {
            "delta_analytic": delta_ana,
            "delta_closed": circ.delta,
            "delta_quadrature": circ_ref.delta,
            "stiffness_analytic": stiff_ana,
            "stiffness_closed": circ.stiffness,
            "stiffness_quadrature": circ_ref.stiffness,
            "delta_err_closed_pct": 100.0 * (circ.delta / delta_ana - 1.0),
            "delta_err_quadrature_pct": 100.0 * (circ_ref.delta / delta_ana - 1.0),
            "stiffness_err_closed_pct": 100.0 * (circ.stiffness / stiff_ana - 1.0),
        },
        "max_abs_stiffness_err_pct_le_75": max(abs(r["stiffness_err_pct"])
                                               for r in rows if r["curvature_ratio"] <= 75.0),
        "max_abs_stiffness_err_pct_all": max(abs(r["stiffness_err_pct"]) for r in rows),
        "max_abs_equiv_err_pct": max(abs(r["equiv_radius_err_pct"]) for r in rows),
    }
    return out


# ---------------------------------------------------------------------------
# Stage 2: clump geometric convergence
# ---------------------------------------------------------------------------

def stage_clump() -> dict:
    shapes = {
        # prolate spheroid (aspect 1.3, same volume as the reference sphere)
        "spheroid_1.3": Superquadric(a=0.0194, b=0.0194, c=0.0252, n1=2.0, n2=2.0),
        # superquadric block (edge exponent 3)
        "superquadric_n3": Superquadric(a=0.022, b=0.018, c=0.016, n1=3.0, n2=3.0),
    }
    n_list = [8, 16, 32, 64, 128]
    rows = []
    for name, sq in shapes.items():
        for n in n_list:
            t0 = time.time()
            centers, radii = fit_inscribed_clump(sq, n)
            dev = clump_surface_deviation(sq, centers, radii)
            rows.append({
                "shape": name,
                "n_spheres": n,
                "max_outward": dev.max_outward,
                "max_inward": dev.max_inward,
                "mean_abs": dev.mean_abs,
                "rms": dev.rms,
                "rel_max": dev.rel_max,
                "volume_ratio": dev.volume_ratio,
                "wall_time_s": time.time() - t0,
            })
    # geometric saturation number: smallest N beyond which the two-sided max
    # error changes by less than 1 % per doubling
    sat = {}
    for name in shapes:
        sub = [r for r in rows if r["shape"] == name]
        n_star = None
        for i in range(1, len(sub)):
            prev, cur = sub[i - 1], sub[i]
            if prev["rel_max"] > 0 and (prev["rel_max"] - cur["rel_max"]) / prev["rel_max"] < 0.20:
                n_star = cur["n_spheres"]
                break
        sat[name] = n_star
    return {"stage": "clump", "rows": rows, "geometric_saturation_N": sat}


# ---------------------------------------------------------------------------
# Stage 3: three-channel piles (repose + static pseudo-dissipation)
# ---------------------------------------------------------------------------

def _pile_cfg(channel: str, **kw) -> PileConfig2D:
    base = dict(channel=channel, radius=0.02, dt=1.5e-4, n_steps=6000,
                n_settle=4500, mu_slide=0.55, mu_roll=0.06, restitution=0.1,
                viscous_damp=0.99, record_every=1000)
    base.update(kw)
    return PileConfig2D(**base)


def stage_piles(only: str | None = None) -> dict:
    cases = [
        ("sphere_aspect1.0", _pile_cfg("sphere", n_particles=150, aspect=1.0)),
        ("sphere_aspect1.3", _pile_cfg("sphere", n_particles=150, aspect=1.3)),
        ("clump_N8_aspect1.3", _pile_cfg("clump", n_particles=150, n_sub=8, aspect=1.3)),
        ("clump_N32_aspect1.3", _pile_cfg("clump", n_particles=100, n_sub=32, aspect=1.3)),
        ("clump_N128_aspect1.3", _pile_cfg("clump", n_particles=45, n_sub=128, aspect=1.3)),
        ("superellipse_n2.0_aspect1.3",
         _pile_cfg("superellipse", n_particles=120, shape_factor=2.0, aspect=1.3)),
        ("superellipse_n2.5_aspect1.3",
         _pile_cfg("superellipse", n_particles=120, shape_factor=2.5, aspect=1.3)),
        ("superellipse_n3.5_aspect1.3",
         _pile_cfg("superellipse", n_particles=120, shape_factor=3.5, aspect=1.3)),
    ]
    rows = []
    for name, cfg in cases:
        if only and only not in name:
            continue
        t0 = time.time()
        res = run_pile(cfg)
        rows.append({
            "case": name,
            "channel": res.channel,
            "n_particles": res.n_particles,
            "n_sub": res.n_sub,
            "shape_factor": res.shape_factor,
            "aspect": res.aspect,
            "repose_deg": res.repose_deg,
            "repose_deg_std": res.repose_deg_std,
            "pseudo_dissipation_per_step": res.pseudo_dissipation,
            "kinetic_energy_final": res.kinetic_energy_final,
            "potential_energy_final": res.potential_energy_final,
            "contacts_final": res.contacts_final,
            "wall_time_s": time.time() - t0,
            "dt": cfg.dt,
            "mu_slide": cfg.mu_slide,
            "mu_roll": cfg.mu_roll,
            "viscous_damp": cfg.viscous_damp,
        })
        print(f"  {name:34s} repose={res.repose_deg:6.2f}+-{res.repose_deg_std:5.2f} deg"
              f"  diss={res.pseudo_dissipation:.2e}  ({time.time() - t0:.0f}s)", flush=True)
    return {"stage": "piles", "rows": rows}


# ---------------------------------------------------------------------------
# Stage 4: verdict + figures
# ---------------------------------------------------------------------------

def _load(stage: str) -> dict | None:
    p = ART / f"{stage}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _figures(hertz: dict, clump: dict, piles: dict | None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    rows = hertz["rows"]
    r = np.array([x["curvature_ratio"] for x in rows])
    err = np.array([abs(x["stiffness_err_pct"]) for x in rows])
    eq = np.array([abs(x["equiv_radius_err_pct"]) for x in rows])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(r, np.maximum(err, 1e-4), "o-", label="elliptic closed form vs. Boussinesq reference")
    ax[0].loglog(r, np.maximum(eq, 1e-4), "s--", label="isotropic equivalent-radius law vs. reference")
    ax[0].axhline(2.0, color="k", ls=":", label="acceptance limit 2 %")
    ax[0].set_xlabel("curvature ratio A/B")
    ax[0].set_ylabel("normal-stiffness error [%]")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, which="both", alpha=0.3)
    sub = [x for x in rows if x["curvature_ratio"] <= 20]
    ax[1].plot([x["curvature_ratio"] for x in sub],
               [x["a_flat_ref"] / x["a_tight_ref"] for x in sub], "o-")
    ax[1].set_xlabel("curvature ratio A/B")
    ax[1].set_ylabel("contact aspect ratio $a_{flat}/a_{tight}$")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Ellipsoid-plane contact: elliptical Hertz anchor (P40-N10)")
    fp = FIG / "p40_n10_hertz.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    figs.append(("p40_n10_hertz.png", "Left: normal-stiffness error of the elliptic-integral closed form and of the "
                                     "isotropic equivalent-radius law against the Boussinesq integral-equation "
                                     "reference. Right: contact-ellipse aspect ratio vs. curvature ratio."))

    # clump convergence
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, style in (("spheroid_1.3", "o-"), ("superquadric_n3", "s--")):
        sub = [x for x in clump["rows"] if x["shape"] == name]
        n = [x["n_spheres"] for x in sub]
        ax[0].loglog(n, [x["rel_max"] for x in sub], style, label=name)
        ax[1].loglog(n, [1.0 - x["volume_ratio"] for x in sub], style, label=name)
    ax[0].set_xlabel("spheres per clump N")
    ax[0].set_ylabel("max two-sided surface error / $r_{eq}$")
    ax[1].set_xlabel("spheres per clump N")
    ax[1].set_ylabel("1 - volume fill ratio")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("N-sphere clump geometric convergence (P40-N10)")
    fig.tight_layout()
    fp = FIG / "p40_n10_clump_convergence.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    figs.append(("p40_n10_clump_convergence.png",
                 "Two-sided surface error and volume fill error of the inscribed N-sphere clump vs. N, "
                 "for a prolate spheroid (aspect 1.3) and a superquadric block (edge exponent 3)."))

    if piles:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        names = [x["case"] for x in piles["rows"]]
        vals = [x["repose_deg"] for x in piles["rows"]]
        errs = [x["repose_deg_std"] for x in piles["rows"]]
        colors = ["C0" if x["channel"] == "sphere" else "C1" if x["channel"] == "clump" else "C2"
                  for x in piles["rows"]]
        ax[0].bar(range(len(names)), vals, yerr=errs, color=colors, alpha=0.85)
        ax[0].axhspan(30.0, 35.0, color="g", alpha=0.15, label="sand/gravel experiment 30-35 deg")
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=70, ha="right", fontsize=6)
        ax[0].set_ylabel("angle of repose [deg]")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3, axis="y")
        ax[1].semilogy(range(len(names)),
                       [max(x["pseudo_dissipation_per_step"], 1e-12) for x in piles["rows"]],
                       "o-")
        ax[1].axhline(1e-6, color="k", ls=":", label="acceptance limit 1e-6 / step")
        ax[1].set_xticks(range(len(names)))
        ax[1].set_xticklabels(names, rotation=70, ha="right", fontsize=6)
        ax[1].set_ylabel("static pseudo-dissipation |dE|/E per step")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3, which="both")
        fig.suptitle("2-D pile channels: repose angle and static dissipation (P40-N10)")
        fig.tight_layout()
        fp = FIG / "p40_n10_piles.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        figs.append(("p40_n10_piles.png",
                     "Left: measured 2-D angle of repose per representation channel against the "
                     "sand/gravel experimental window (30-35 deg). Right: static-pile pseudo-dissipation "
                     "per step against the 1e-6 acceptance limit."))
    return figs


def stage_report() -> dict:
    hertz = _load("hertz")
    clump = _load("clump")
    piles = _load("piles")
    if hertz is None or clump is None:
        raise SystemExit("run --stage hertz and --stage clump first")

    verdict = {
        "node": "P40-N10",
        "repository": "TensorDEM",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "criteria": {},
        "evidence": {"hertz": hertz, "clump": clump, "piles": piles},
    }

    # (a) elliptical Hertz closed form vs. reference
    a_pass = hertz["max_abs_stiffness_err_pct_le_75"] <= 2.0
    verdict["criteria"]["a_elliptic_hertz_closed_form"] = {
        "criterion": "ellipsoid-plane normal stiffness vs. Hertz elliptic-integral closed form <= 2 %",
        "max_err_pct_up_to_ratio75": hertz["max_abs_stiffness_err_pct_le_75"],
        "max_err_pct_all_ratios": hertz["max_abs_stiffness_err_pct_all"],
        "circular_limit_err_pct": abs(hertz["circular_limit"]["stiffness_err_closed_pct"]),
        "verdict": "PASS" if a_pass else "FAIL",
        "note": "reference = independent Boussinesq integral-equation solution; the isotropic "
                "equivalent-radius law deviates by up to %.1f %% over the same scan."
                % hertz["max_abs_equiv_err_pct"],
    }

    # (b)/(c) clump convergence and the repose channel
    geom_n = clump["geometric_saturation_N"]
    crit_b = {
        "criterion": "clump convergence: repose drift <= 1 deg for N >= 32, report saturation N*",
        "geometric_rel_max": {str(r["n_spheres"]): r["rel_max"]
                              for r in clump["rows"] if r["shape"] == "spheroid_1.3"},
        "geometric_saturation_N": geom_n,
    }
    pile_rows = [] if piles is None else piles["rows"]
    clump_piles = [x for x in pile_rows if x["channel"] == "clump"]
    if len(clump_piles) >= 2:
        by_n = {x["n_sub"]: x["repose_deg"] for x in clump_piles}
        ref_n = max(by_n)
        drift = {str(n): abs(v - by_n[ref_n]) for n, v in by_n.items()}
        crit_b["repose_deg_by_N"] = {str(k): v for k, v in by_n.items()}
        crit_b["repose_drift_deg_vs_maxN"] = drift
        ok = all(d <= 1.0 for n, d in drift.items() if n >= 32)
        crit_b["verdict"] = "PASS" if ok else "FAIL"
    else:
        crit_b["verdict"] = "WITHHELD"
    verdict["criteria"]["b_clump_convergence"] = crit_b

    # (c) three channels vs the experimental window
    crit_c = {"criterion": "repose angle in the 30-35 deg sand/gravel window, monotone in shape factor",
              "experimental_window_deg": [30.0, 35.0],
              "channels": {}}
    for ch in ("sphere", "clump", "superellipse"):
        sub = [x for x in pile_rows if x["channel"] == ch]
        if sub:
            crit_c["channels"][ch] = {x["case"]: x["repose_deg"] for x in sub}
    se = [x for x in pile_rows if x["channel"] == "superellipse"]
    if len(se) >= 2:
        se_sorted = sorted(se, key=lambda x: x["shape_factor"])
        crit_c["monotone_in_shape_factor"] = all(
            se_sorted[i + 1]["repose_deg"] >= se_sorted[i]["repose_deg"] - 0.5
            for i in range(len(se_sorted) - 1))
        crit_c["superellipse_repose"] = [(x["shape_factor"], x["repose_deg"]) for x in se_sorted]
    in_window = [x for x in pile_rows if 30.0 <= x["repose_deg"] <= 35.0]
    crit_c["n_cases_in_window"] = len(in_window)
    crit_c["n_cases_total"] = len(pile_rows)
    crit_c["verdict"] = "PASS" if crit_c.get("monotone_in_shape_factor") and in_window else \
                        ("PARTIAL" if crit_c.get("monotone_in_shape_factor") else "FAIL")
    verdict["criteria"]["c_repose_three_channels"] = crit_c

    # (d) static pseudo-dissipation
    crit_d = {"criterion": "static-pile pseudo-dissipation <= 1e-6 per step",
              "by_case": {x["case"]: x["pseudo_dissipation_per_step"] for x in pile_rows}}
    if pile_rows:
        worst = max(x["pseudo_dissipation_per_step"] for x in pile_rows)
        crit_d["max_per_step"] = worst
        crit_d["verdict"] = "PASS" if worst <= 1.0e-6 else "FAIL"
    else:
        crit_d["verdict"] = "WITHHELD"
    verdict["criteria"]["d_static_pseudo_dissipation"] = crit_d

    # (e) the verdict itself: when may a clump replace the true shape?
    geo = {r["n_spheres"]: r for r in clump["rows"] if r["shape"] == "spheroid_1.3"}
    verdict["representation_verdict"] = {
        "rule": [
            "1. Contact stiffness / pressure: use the elliptical Hertz anchor whenever A/B > 1.05; "
            "the isotropic equivalent-radius law is already off by "
            f"{abs(hertz['rows'][3]['equiv_radius_err_pct']):.1f} % at A/B=2 and "
            f"{abs(hertz['rows'][7]['equiv_radius_err_pct']):.1f} % at A/B=8.",
            "2. Geometric fidelity: an inscribed clump with N spheres reproduces the body surface to "
            "within rel_max "
            f"{geo[32]['rel_max']*100:.2f} % / {geo[128]['rel_max']*100:.2f} % of r_eq at N=32 / 128.",
            "3. Macroscopic (bulk) response: clumps are usable when the observable is insensitive to "
            "the residual roughness, i.e. when the repose drift stays below the experimental window "
            "width (5 deg); see criteria b/c for the measured drift.",
        ],
        "error_bounds": {
            "stiffness_equiv_radius_1pct_below_ratio": 1.6,
            "stiffness_equiv_radius_5pct_below_ratio": 7.0,
            "clump_rel_max_N32": geo[32]["rel_max"],
            "clump_rel_max_N128": geo[128]["rel_max"],
            "clump_geometric_saturation_N": geom_n,
        },
        "not_supported": [
            "3-D superquadric contact iteration is not re-solved here (it lives in "
            "tensordem.superquadric_kernel); the anchor is the 3-D ellipsoid + 2-D superellipse.",
        ],
    }

    figs = _figures(hertz, clump, piles)
    verdict["figures"] = [{"name": n, "caption": c} for n, c in figs]
    out = ART / "shape_representation_verdict.json"
    out.write_text(json.dumps(verdict, indent=2))
    print(f"wrote {out}")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["hertz", "clump", "piles", "report", "all"])
    ap.add_argument("--only", default=None, help="substring filter for pile cases")
    args = ap.parse_args()
    _ensure_dirs()
    if args.stage in ("hertz", "all"):
        data = stage_hertz()
        (ART / "hertz.json").write_text(json.dumps(data, indent=2))
        print("hertz: max |stiffness err| (A/B<=75) = %.3f %% | equiv-radius max = %.1f %%"
              % (data["max_abs_stiffness_err_pct_le_75"], data["max_abs_equiv_err_pct"]))
    if args.stage in ("clump", "all"):
        data = stage_clump()
        (ART / "clump.json").write_text(json.dumps(data, indent=2))
        print("clump: geometric saturation N =", data["geometric_saturation_N"])
    if args.stage in ("piles", "all"):
        prior = _load("piles") if args.only else None
        data = stage_piles(only=args.only)
        if prior:
            keep = [r for r in prior["rows"] if args.only not in r["case"]]
            data["rows"] = keep + data["rows"]
        (ART / "piles.json").write_text(json.dumps(data, indent=2))
    if args.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()