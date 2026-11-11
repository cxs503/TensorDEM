"""GAP-DEM-CLUMP round-2 shape-effect scan (mass matched, quiescence gated).

Runs the repose-angle protocol of ``repose_protocol_v2`` on the *same* solid
volume for every aspect ratio (so shape and particle count are no longer
entangled) and compares three rolling-resistance configurations:

``viscous/member``
    round-1 legacy law (rate proportional, body-isotropic) - the control.
``coulomb/member``
    static-capable branch, legacy moment arm - isolates defect (c) alone.
``coulomb/com``
    static branch + shape-aware moment arm ``|x_contact - x_com|`` - the
    round-2 candidate mechanism (adopts LIGGGHTS/YADE-style Coulomb rolling
    limit, but with the *geometry* of the clump providing the anisotropy
    instead of a body-level scalar ``mu_roll``).

Usage (single case, shardable)::

    python benchmarks/gap_dem_clump/run_round2_scan.py --case ar=3.0,seed=0,config=coulomb/com --outdir artifacts/GAP-DEM-CLUMP-R2

Usage (aggregate + figure + criteria)::

    python benchmarks/gap_dem_clump/run_round2_scan.py --aggregate artifacts/GAP-DEM-CLUMP-R2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "gap_dem_clump"))

from run_gap_dem_clump import AR_LIST, BASE_RADIUS, DENSITY, LIT, make_shape  # noqa: E402
from repose_protocol_v2 import (  # noqa: E402
    BOX,
    PROTOCOL_V1,
    PROTOCOL_V2,
    lift_steps_for,
    matched_fill_height,
    run_repose,
)
from tensordem.contact.clump_contact import ContactParams  # noqa: E402

#: mass-matching target: the round-1 sphere fill's solid volume [m^3]
TARGET_VOLUME = 4.690e-6
SEEDS = (0, 1)
CONFIGS = ("viscous/member", "coulomb/member", "coulomb/com")
MU_ROLL = 0.3


def params_for(config: str) -> ContactParams:
    model, lever = config.split("/")
    return ContactParams(mu_roll=MU_ROLL, rolling_model=model, rolling_lever=lever)


def fill_heights() -> dict:
    out = {}
    for ar in AR_LIST:
        m = matched_fill_height(make_shape(ar)(), TARGET_VOLUME)
        out[f"{ar:.1f}"] = m
    return out


def run_case(ar: float, seed: int, config: str, fills: dict) -> dict:
    h = fills[f"{ar:.1f}"]["fill_height"]
    protocol = dict(PROTOCOL_V2, fill_height_by_ar={k: v["fill_height"] for k, v in fills.items()})
    lift_steps = lift_steps_for(h, protocol)
    t0 = time.time()
    from tensordem.particle.clump_dem import lifted_box_repose

    kw = dict(protocol)
    for k in ("fill_height", "fill_height_by_ar", "lift_steps"):
        kw.pop(k, None)
    out = lifted_box_repose(
        make_shape(ar), seed=seed, params=params_for(config), fill_height=h,
        lift_steps=lift_steps, **kw, **BOX,
    )
    m = out["measurement"]
    return {
        "aspect_ratio_param": ar,
        "config": config,
        "seed": seed,
        "mu_roll": MU_ROLL,
        "fill_height": h,
        "n_particles": out["n_particles"],
        "solid_volume": out["solid_volume"],
        "mass": out["mass"],
        "angle_profile_deg": m["angle_profile_deg"],
        "angle_hull_deg": m["angle_hull_deg"],
        "steps": out["steps"],
        "tail_steps_used": out["tail_steps_used"],
        "quiesced": out["quiesced"],
        "tail_mean_speed": out["tail_spread"]["mean_speed"],
        "tail_saturation_median": out["rolling_saturation"]["median"],
        "rolling_saturation_model": out["rolling_saturation"].get("model"),
        "wall_seconds": round(time.time() - t0, 1),
    }


def criterion(trend: dict, sphere_in_band_ok: bool = True) -> dict:
    """C3-style acceptance on a per-configuration trend."""
    means = {k: v for k, v in trend["means"].items() if np.isfinite(v)}
    if not means:
        return {"monotone_increasing": False, "max_delta_vs_sphere_deg": 0.0,
                "sphere_mean_deg": float("nan"), "sphere_in_literature_band": False,
                "delta_criterion_deg": 3.0, "pass": False, "reason": "no valid case"}
    ars = sorted(means, key=float)
    mono = all(means[ars[i + 1]] >= means[ars[i]] - 1e-9 for i in range(len(ars) - 1))
    delta = (max(means[a] - means[ars[0]] for a in ars[1:]) if len(ars) > 1
             else float("nan"))
    if len(ars) < 2:
        return {"monotone_increasing": False, "max_delta_vs_sphere_deg": float("nan"),
                "sphere_mean_deg": float(means[ars[0]]),
                "sphere_in_literature_band": bool(
                    LIT["sphere_repose_deg"][0] <= means[ars[0]] <= LIT["sphere_repose_deg"][1]),
                "delta_criterion_deg": 3.0, "pass": False,
                "reason": "fewer than two valid aspect ratios (trend not estimable)"}
    sphere = means[ars[0]]
    in_band = LIT["sphere_repose_deg"][0] <= sphere <= LIT["sphere_repose_deg"][1]
    return {
        "monotone_increasing": bool(mono),
        "max_delta_vs_sphere_deg": float(delta),
        "sphere_mean_deg": float(sphere),
        "sphere_in_literature_band": bool(in_band),
        "delta_criterion_deg": 3.0,
        "pass": bool(mono and delta >= 3.0),
    }


def aggregate(outdir: Path, target: float = TARGET_VOLUME) -> dict:
    cases = []
    for f in sorted(outdir.glob("case_*.json")):
        cases.append(json.loads(f.read_text()))
    if not cases:
        raise SystemExit(f"no case_*.json in {outdir}")

    summary = {}
    for config in sorted({c["config"] for c in cases}):
        all_rows = [c for c in cases if c["config"] == config]
        # a case that did not reach quiescence is still flowing: its silhouette
        # angle is not a repose angle (round-1 artefact) -> excluded, and the
        # exclusion is reported, never silently averaged in.
        rows = [c for c in all_rows if c.get("quiesced")]
        invalid = [c for c in all_rows if not c.get("quiesced")]
        means, stds, counts = {}, {}, {}
        for ar in sorted({r["aspect_ratio_param"] for r in rows}):
            vals = [r["angle_profile_deg"] for r in rows
                    if r["aspect_ratio_param"] == ar and np.isfinite(r["angle_profile_deg"])]
            means[f"{ar:.1f}"] = float(np.mean(vals)) if vals else float("nan")
            stds[f"{ar:.1f}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            counts[f"{ar:.1f}"] = int(np.mean([r["n_particles"] for r in rows
                                              if r["aspect_ratio_param"] == ar]
                                             ) if vals else 0)
        trend = {"means": means, "stds": stds, "n_particles": counts}
        summary[config] = {"trend": trend, "criterion": criterion(trend),
                           "n_cases": len(rows), "n_invalid_not_quiesced": len(invalid),
                           "invalid": [{"ar": c["aspect_ratio_param"], "seed": c["seed"],
                                        "angle_deg": c["angle_profile_deg"],
                                        "fill_height": c["fill_height"]} for c in invalid],
                           "all_quiesced": bool(all(r["quiesced"] for r in rows)),
                           "n_particles_ratio": (max(counts.values()) / min(counts.values()))
                           if counts else float("nan")}

    # A/B deltas between configurations, per aspect ratio
    ab = {}
    base = "viscous/member"
    for config in CONFIGS[1:]:
        if config in summary and base in summary:
            ab[config] = {
                a: summary[config]["trend"]["means"][a] - summary[base]["trend"]["means"][a]
                for a in summary[base]["trend"]["means"]
            }

    results = {
        "mission": "m15028",
        "anchor": "GAP-DEM-CLUMP",
        "round": 2,
        "protocol": {
            "fill": "mass matched (bisection of fill height, target solid volume "
                    f"{target:.4e} m^3 = round-1 sphere fill)",
            "tail": "quiescence gate (mean body speed <= 0.01 m/s, probe every 300 steps)",
            "mu_roll": MU_ROLL,
            "seeds": list(SEEDS),
            "box": BOX,
        },
        "per_config": summary,
        "ab_vs_viscous_member_deg": ab,
        "cases": cases,
        "literature": LIT,
    }
    (outdir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    crit = {c: summary[c]["criterion"] for c in summary}
    (outdir / "criteria.json").write_text(json.dumps(crit, indent=2, sort_keys=True))
    return results


def make_figure(results: dict, outdir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=130)
    colors = {"viscous/member": "tab:gray", "coulomb/member": "tab:blue",
              "coulomb/com": "tab:red"}
    for config, s in results["per_config"].items():
        tr = s["trend"]
        ars = sorted(tr["means"], key=float)
        x = [float(a) for a in ars]
        y = [tr["means"][a] for a in ars]
        e = [tr["stds"][a] for a in ars]
        ax[0].errorbar(x, y, yerr=e, marker="o", capsize=3, label=config,
                       color=colors.get(config))
    ax[0].axhspan(*LIT["sphere_repose_deg"], color="tab:green", alpha=0.12,
                  label="sphere band 22-25 deg")
    ax[0].axhspan(35.0, 45.0, color="tab:orange", alpha=0.12,
                  label="ellipsoidal 35-45 deg")
    ax[0].set_xlabel("aspect ratio a/b [-]")
    ax[0].set_ylabel("repose angle [deg]")
    ax[0].set_title("angle of repose vs shape (mass matched)")
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.3)

    # right: shape effect = angle minus sphere baseline, per config
    width = 0.26
    ars = sorted(results["per_config"][CONFIGS[0]]["trend"]["means"], key=float)[1:]
    for i, config in enumerate(CONFIGS):
        if config not in results["per_config"]:
            continue
        tr = results["per_config"][config]["trend"]
        base = tr["means"][sorted(tr["means"], key=float)[0]]
        d = [tr["means"][a] - base for a in ars]
        ax[1].bar(np.arange(len(ars)) + (i - 1) * width, d, width, label=config,
                  color=colors.get(config))
    ax[1].axhline(3.0, ls="--", color="k", lw=1, label="3 deg acceptance")
    ax[1].axhline(0.0, color="k", lw=0.6)
    ax[1].set_xticks(np.arange(len(ars)))
    ax[1].set_xticklabels([f"{a}" for a in ars])
    ax[1].set_xlabel("aspect ratio a/b [-]")
    ax[1].set_ylabel("repose - sphere [deg]")
    ax[1].set_title("shape effect per rolling-law configuration")
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = outdir / "gap_dem_clump_r2.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="ar=3.0,seed=0,config=coulomb/com")
    ap.add_argument("--outdir", default=str(ROOT / "artifacts" / "GAP-DEM-CLUMP-R2"))
    ap.add_argument("--aggregate", nargs="?", const="__default__")
    ap.add_argument("--fills", help="json file with matched fill heights")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.case:
        kv = dict(part.split("=", 1) for part in args.case.split(","))
        ar = float(kv["ar"])
        seed = int(kv.get("seed", 0))
        config = kv.get("config", "coulomb/com")
        if args.fills and Path(args.fills).exists():
            fills = json.loads(Path(args.fills).read_text())
        else:
            fills = fill_heights()
            (outdir / "fills.json").write_text(json.dumps(fills, indent=2))
        row = run_case(ar, seed, config, fills)
        tag = f"case_ar{ar:.1f}_seed{seed}_{config.replace('/', '-')}.json"
        (outdir / tag).write_text(json.dumps(row, indent=2))
        print(json.dumps({k: row[k] for k in ("aspect_ratio_param", "config", "seed",
                                              "angle_profile_deg", "n_particles",
                                              "quiesced", "wall_seconds")}))
        return 0

    if args.aggregate is not None:
        d = outdir if args.aggregate == "__default__" else Path(args.aggregate)
        results = aggregate(d)
        fig = make_figure(results, d)
        print(json.dumps({"figure": str(fig),
                          "criteria": {c: s["criterion"]["pass"]
                                       for c, s in results["per_config"].items()}},
                         indent=2))
        ok = any(s["criterion"]["pass"] for s in results["per_config"].values())
        return 0 if ok else 1

    ap.error("need --case or --aggregate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())