"""Campaign runner for node p223w2-n03 (ice floe rafting/ridging + Brazil nut).

Produces ``artifacts/p223w2n03/acceptance.json``, the four required figures and
the raw run records.  Everything is deterministic: fixed seeds, no wall-clock
dependence.  Heavy configurations can be sharded by the ``--face`` argument so
the same script runs as a cluster job.

Usage
-----
    python scripts/run_p223w2n03_campaign.py            # both faces
    python scripts/run_p223w2n03_campaign.py --face rafting
    python scripts/run_p223w2n03_campaign.py --face convection --quick
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "p223w2n03"

from tensordem.ice_rafting import (  # noqa: E402
    RAFTING_ANCHORS,
    RIDGE_GEOMETRY_ANCHORS,
    IceProperties,
    bracket_verdict,
    compare,
    rafting_resistance,
    run_rafting_ramp,
    run_ridge,
)
from tensordem.granular_convection import (  # noqa: E402
    BNE_ANCHORS,
    RBNE_BOUNDARY_ANCHOR,
    ONSET_ANCHOR,
    BedConfig,
    ShakeCase,
    anchor_errors,
    interpolated_boundary,
    run_phase_scan,
    run_shake,
)
from tensordem.granular_convection.anchors import point_error  # noqa: E402


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


# ---------------------------------------------------------------------------
# Face A: rafting / ridging
# ---------------------------------------------------------------------------

def run_rafting_face(quick: bool = False) -> dict:
    ice = IceProperties()
    cases = []
    plan = [(45.0, 0.3), (30.0, 0.3), (45.0, 0.1)]
    for angle, mu in plan:
        res = run_rafting_ramp(h=0.5, length=6.0, ice=ice, ramp_angle_deg=angle,
                               mu=mu, piston_velocity=-0.2,
                               max_steps=6000 if quick else 12000,
                               sample_every=200, settle_steps=2500)
        zs = 0.5 * 0.5
        analytic = rafting_resistance(zs, 0.5, 6.0, ice, bevel=0.5, mu=mu)
        tau_dem = res["tau_mid_climb"]
        reached = tau_dem is not None
        if not reached and res["history"]:
            # quick mode may stop before half climb: report the last load and flag it
            tau_dem = res["history"][-1]["load"] / 0.5
        climb_max = max((h["climb_ratio"] for h in res["history"]), default=0.0)
        cases.append({
            "ramp_angle_deg": angle,
            "mu": mu,
            "climb_max": float(climb_max),
            "half_climb_reached": bool(reached),
            "tau_raft_dem": tau_dem,
            "tau_raft_analytic": analytic["stress"],
                "ratio_dem_over_analytic": (None if tau_dem is None else
                                        float(tau_dem / analytic["stress"])),
            "peak_stress": res["tau_peak"],
            "history": res["history"],
            "dt": res["dt"],
            "n_steps": len(res["history"]) * 200,
        })
    base = cases[0]
    ridge = run_ridge(h=0.5, n_floes=6 if quick else 8, ice=ice,
                      max_steps=4000 if quick else 15000, sample_every=500,
                      convergence=0.12, piston_velocity=-0.3)
    ridge.pop("polygons", None)
    return {"ice": ice.__dict__, "ramp_cases": cases, "ridge": ridge,
            "ridge_stable": bool(ridge["sail_height"] < 2.0)}


def rafting_acceptance(face: dict) -> dict:
    """AC1 (rafting threshold vs field observation) and AC2 (ridge geometry)."""
    ice = IceProperties()
    base = face["ramp_cases"][0]
    tau = base["tau_raft_dem"]
    acc1 = compare(tau, RAFTING_ANCHORS["rafting_onset_h050"])
    acc1["hard_criterion_15pct"] = bool(acc1["abs_rel_error"] <= 0.15)
    acc1["half_climb_reached"] = bool(base.get("half_climb_reached", False))
    acc1["climb_at_measurement"] = base.get("climb_max")
    acc1["verdict"] = "PASS" if (acc1["within_range"]
                                 and acc1["hard_criterion_15pct"]) else "FAIL"
    # two-limit bracket: rigid-floe limit (DEM + closed form) and the flexural
    # limit (floe bends over its elastic-foundation length)
    flex = rafting_resistance(0.5 * 0.5, 0.5, min(6.0, 8.69), ice,
                              bevel=0.5, mu=0.3)["stress"]
    acc1["bracket"] = bracket_verdict(min(tau, flex), max(tau, flex),
                                      RAFTING_ANCHORS["rafting_onset_h050"])
    acc1["flexural_limit_stress"] = float(flex)
    ridge = face["ridge"]
    acc2 = compare(ridge["sail_keel_ratio"], RIDGE_GEOMETRY_ANCHORS["sail_keel_ratio"])
    acc2["sail_aspect"] = compare(ridge["sail_aspect"],
                                  RIDGE_GEOMETRY_ANCHORS["sail_aspect"])
    acc2["stable"] = face["ridge_stable"]
    acc2["verdict"] = ("PASS" if (acc2["within_range"] and acc2["stable"]
                                  and acc2["abs_rel_error"] <= 0.10) else "FAIL")
    return {"AC1": acc1, "AC2": acc2}


# ---------------------------------------------------------------------------
# Face B: Brazil nut / reverse Brazil nut
# ---------------------------------------------------------------------------

def run_convection_face(quick: bool = False) -> dict:
    config = BedConfig()
    cycles = 20 if quick else 40
    runs = {}
    for label, gamma in (("bne", 6.0), ("control_novibration", 0.0), ("subcritical", 1.0)):
        t0 = time.time()
        res = run_shake(ShakeCase(gamma=max(gamma, 1e-9), frequency=20.0,
                                  n_cycles=cycles, sample_every=600,
                                  label=label),
                        config if gamma > 0 else _no_shake_config(config),
                        settle_steps=3000)
        res.pop("final", None)
        res["is_large"] = res["is_large"].tolist()
        res["wall_seconds"] = time.time() - t0
        res["gamma_label"] = gamma
        runs[label] = res
    gammas = [2.0, 4.0, 6.0, 8.0]
    ratios = [1.0, 2.0, 3.0]
    points = run_phase_scan(gammas, ratios, config=config,
                            case=ShakeCase(gamma=6.0, frequency=20.0,
                                           n_cycles=20 if quick else 25),
                            settle_steps=3000)
    boundary = interpolated_boundary(points)
    return {
        "config": {k: v for k, v in config.__dict__.items()},
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                 for k, v in runs.items()},
        "histories": {k: v["history"] for k, v in runs.items()},
        "phase_points": [p.as_dict() for p in points],
        "boundary": boundary,
    }


def _no_shake_config(config: BedConfig) -> BedConfig:
    """The negative control: same bed, no shaking (Gamma = 0)."""
    return config


def convection_acceptance(face: dict) -> dict:
    bne = face["runs"]["bne"]
    ctrl = face["runs"]["control_novibration"]
    sub = face["runs"]["subcritical"]
    # AC3: segregation timescale vs experimental anchor
    ac3 = point_error(bne["tau_cycles"], BNE_ANCHORS["segregation_timescale_cycles"])
    ac3["hard_criterion_10pct"] = bool(ac3["abs_rel_error"] <= 0.10)
    ac3["verdict"] = "PASS" if (ac3["within_range"] and ac3["hard_criterion_10pct"]) else "FAIL"
    # AC4: reverse Brazil-nut phase boundary vs experiment
    b0 = [b for b in face["boundary"] if abs(b["gamma"] - 6.0) < 1e-9]
    ac4 = point_error(b0[0]["rho_boundary"] if b0 and b0[0]["rho_boundary"] else
                      float("nan"), RBNE_BOUNDARY_ANCHOR)
    ac4["boundaries"] = face["boundary"]
    ac4["verdict"] = ("PASS" if (ac4["within_range"] and math.isfinite(
        ac4["abs_rel_error"])) else "FAIL")
    # AC5: negative control (no vibration -> no segregation)
    dz_ctrl = abs(ctrl["dz_from_initial"])
    dz_bne = abs(bne["dz_from_initial"])
    ac5 = {"dz_control_mm": 1e3 * dz_ctrl, "dz_bne_mm": 1e3 * dz_bne,
           "ratio": (dz_ctrl / dz_bne) if dz_bne > 0 else float("inf"),
           "verdict": "PASS" if (dz_ctrl < 0.25 * dz_bne) else "FAIL"}
    ac6 = {"onset_gamma": sub["gamma_label"], "dz_subcritical_mm": 1e3 * sub["dz_from_initial"],
           "dz_supercritical_mm": 1e3 * dz_bne,
           "verdict": "PASS" if abs(sub["dz_from_initial"]) < 0.5 * dz_bne else "FAIL"}
    return {"AC3": ac3, "AC4": ac4, "AC5": ac5, "onset_check": ac6}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(rafting_face: dict, acc_raft: dict, conv_face: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    made = []

    # 1) rafting threshold: load vs climb, DEM vs analytic, field anchor band
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {45.0: "C0", 30.0: "C1"}
    for case in rafting_face["ramp_cases"]:
        climb = np.array([h["climb"] for h in case["history"]])
        load = np.array([h["load"] for h in case["history"]]) / 1e3
        keep = climb > 0
        ax.plot(climb[keep], load[keep],
                color=colors.get(case["ramp_angle_deg"], "C2"),
                label=f"DEM ramp {case['ramp_angle_deg']:.0f} deg, mu={case['mu']}")
        zs = np.linspace(1e-3, 0.5, 60)
        ana = [rafting_resistance(z, 0.5, 6.0, IceProperties(), bevel=0.5,
                                  mu=case["mu"])["line_load"] / 1e3 for z in zs]
        ax.plot(zs, ana, "--", color=colors.get(case["ramp_angle_deg"], "C2"),
                lw=1.0, label=f"analytic ramp law, mu={case['mu']}")
    lo = RAFTING_ANCHORS["rafting_onset_h050"]["range"][0] * 0.5 / 1e3
    hi = RAFTING_ANCHORS["rafting_onset_h050"]["range"][1] * 0.5 / 1e3
    ax.axhspan(lo, hi, color="0.85", zorder=0,
               label="field onset band (Keinonen+1996), h=0.5 m")
    ax.set_xlabel("leading-edge climb z [m]")
    ax.set_ylabel("piston line load [kN/m]")
    ax.set_title("Ice-floe rafting resistance vs climb (h = 0.5 m, L = 6 m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "rafting_threshold.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    # 2) ridge geometry: measured (unstable) vs observed anchors
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ridge = rafting_face["ridge"]
    labels = ["sail/keel\nH_s/H_k", "sail aspect\nH_s/W_s"]
    meas = [ridge["sail_keel_ratio"], ridge["sail_aspect"]]
    obs = [RIDGE_GEOMETRY_ANCHORS["sail_keel_ratio"]["value"],
           RIDGE_GEOMETRY_ANCHORS["sail_aspect"]["value"]]
    obs_lo = [RIDGE_GEOMETRY_ANCHORS["sail_keel_ratio"]["range"][0],
              RIDGE_GEOMETRY_ANCHORS["sail_aspect"]["range"][0]]
    obs_hi = [RIDGE_GEOMETRY_ANCHORS["sail_keel_ratio"]["range"][1],
              RIDGE_GEOMETRY_ANCHORS["sail_aspect"]["range"][1]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, meas, width=0.34, label="rigid-floe chain DEM", color="C3")
    ax.bar(x + 0.18, obs, width=0.34, label="observed (Timco & Burden 1994)",
           color="C0")
    ax.errorbar(x + 0.18, obs, yerr=[np.array(obs) - np.array(obs_lo),
                                     np.array(obs_hi) - np.array(obs)],
                fmt="none", ecolor="k", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("aspect ratio [-]")
    ax.set_title("Ridge geometry: DEM chain (unstable, honest FAIL) vs observation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUT / "ice_ridge_geometry.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    # 3) Brazil-nut segregation histories
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, style in (("bne", "-"), ("subcritical", "--"),
                         ("control_novibration", ":")):
        hist = conv_face["histories"][label]
        cyc = [h["cycles"] for h in hist]
        dz = [(h["dz"] - hist[0]["dz"]) * 1e3 for h in hist]
        ax.plot(cyc, dz, style, label=f"{label} (Gamma={conv_face['runs'][label]['gamma_label']})")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("vibration cycles [-]")
    ax.set_ylabel("dz = z_large - z_small change [mm]")
    ax.set_title("Brazil-nut segregation: shaken bed vs no-shake control")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "brazil_nut_segregation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    # 4) reverse Brazil-nut phase diagram
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pts = conv_face["phase_points"]
    gammas = sorted({p["gamma"] for p in pts})
    ratios = sorted({p["density_ratio"] for p in pts})
    grid = np.zeros((len(ratios), len(gammas)))
    for p in pts:
        grid[ratios.index(p["density_ratio"]), gammas.index(p["gamma"])] = p["dz"] * 1e3
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm",
                   extent=[gammas[0] - 0.5, gammas[-1] + 0.5,
                           ratios[0] - 0.25, ratios[-1] + 0.25])
    for p in pts:
        ax.text(p["gamma"], p["density_ratio"], f"{p['dz']*1e3:+.1f}",
                ha="center", va="center", fontsize=8)
    for b in conv_face["boundary"]:
        if b["rho_boundary"] is not None:
            ax.plot(b["gamma"], b["rho_boundary"], "ko", ms=5)
    lo, hi = RBNE_BOUNDARY_ANCHOR["range"]
    ax.axhspan(lo, hi, color="none", edgecolor="k", hatch="///", alpha=0.35,
               label="observed RBNE band (Hong+2001)")
    fig.colorbar(im, ax=ax, label="dz change [mm]  (+ Brazil nut, - reverse)")
    ax.set_xlabel("shaking intensity Gamma [-]")
    ax.set_ylabel("density ratio rho_large / rho_small [-]")
    ax.set_title("Reverse Brazil-nut phase map (size ratio 2)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = OUT / "reverse_brazil_phase.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))
    return made


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", choices=["rafting", "convection", "both"], default="both")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    record = {"node": "p223w2-n03", "created": stamp, "faces": {}}

    rafting_face = conv_face = None
    if args.face in ("rafting", "both"):
        rafting_face = run_rafting_face(quick=args.quick)
        record["faces"]["rafting"] = rafting_face
        record["acceptance_rafting"] = rafting_acceptance(rafting_face)
        (OUT / "rafting_runs.json").write_text(
            json.dumps(rafting_face, default=_json_default, indent=1))
    if args.face in ("convection", "both"):
        conv_face = run_convection_face(quick=args.quick)
        record["faces"]["convection"] = conv_face
        record["acceptance_convection"] = convection_acceptance(conv_face)
        (OUT / "convection_runs.json").write_text(
            json.dumps(conv_face, default=_json_default, indent=1))

    figures = []
    if rafting_face is not None and conv_face is not None:
        figures = make_figures(rafting_face, record["acceptance_rafting"], conv_face)
    record["figures"] = figures
    out = OUT / "acceptance.json"
    out.write_text(json.dumps(record, default=_json_default, indent=1))
    print(json.dumps({k: v for k, v in record.items()
                      if k.startswith("acceptance")}, default=_json_default, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())