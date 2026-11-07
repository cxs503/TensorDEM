"""Parameter sweep for the 3-D ride-up model (campaign node IN32-N03).

Runs the ``(Fr, h, alpha)`` grid with several seeds and writes one JSON per
shard; ``--aggregate`` then merges the shards, evaluates the machine-checkable
acceptance criteria and produces the figures.

Usage
-----
    python benchmarks/ice_rideup_sweep.py --shard 0 --of 32 --out artifacts/ice_rideup
    python benchmarks/ice_rideup_sweep.py --aggregate artifacts/ice_rideup

The sweep is sharded because it is the expensive part (300 runs of a 3-D contact
integration); it is designed to be dispatched over the 702 compute cluster.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from tensordem.ice_rideup import (
    BowSlope, ContactParams, RHO_ICE, RHO_WATER, RideUpSimulation,
    archimedes_buoyancy, monotonicity_slope, run_case,
)

FR_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)
H_VALUES = (0.3, 0.5, 0.9, 1.2, 1.5)
ANGLE_VALUES = (20.0, 30.0, 40.0, 45.0)
SEEDS = (0, 1, 2)


def grid() -> List[tuple]:
    points = []
    for fr in FR_VALUES:
        for h in H_VALUES:
            for ang in ANGLE_VALUES:
                for seed in SEEDS:
                    points.append((fr, h, ang, seed))
    return points


def run_shard(index: int, n_shards: int, out_dir: str, *, steps: int, warmup: int,
              n_floes: int, n_samples: int, dt: float) -> str:
    torch.set_num_threads(1)
    points = grid()
    mine = [p for k, p in enumerate(points) if k % n_shards == index]
    rows = []
    for fr, h, ang, seed in mine:
        res = run_case(fr=fr, thickness=h, angle_deg=ang, seed=seed, steps=steps,
                       dt=dt, n_floes=n_floes, warmup_steps=warmup,
                       n_samples=n_samples)
        rows.append(res.as_dict())
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard_{index:03d}.json")
    with open(path, "w") as fh:
        json.dump({"shard": index, "of": n_shards, "steps": steps,
                   "warmup_steps": warmup, "n_floes": n_floes,
                   "n_samples": n_samples, "dt": dt, "rows": rows}, fh, indent=1)
    return path


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    return 0.5 * (s[(n - 1) // 2] + s[n // 2])


def aggregate(in_dir: str) -> Dict:
    torch.set_num_threads(1)
    files = sorted(glob.glob(os.path.join(in_dir, "shard_*.json")))
    if not files:
        raise SystemExit(f"no shard files in {in_dir}")
    rows: List[Dict] = []
    meta = {}
    for path in files:
        with open(path) as fh:
            payload = json.load(fh)
        meta = payload
        rows.extend(payload["rows"])
    rows.sort(key=lambda r: (r["Fr"], r["thickness_m"], r["angle_deg"], r["seed"]))

    # ---- criteria -------------------------------------------------------
    # (a) contact kernel vs analytic solutions (independent of the sweep)
    contact_checks = _contact_analytic_checks()

    # (b) ride-up force exists for Fr > 0.2 and grows with Fr (reference point)
    ref = [r for r in rows if abs(r["thickness_m"] - 0.5) < 1e-9
           and abs(r["angle_deg"] - 30.0) < 1e-9]
    by_fr: Dict[float, List[float]] = {}
    for r in ref:
        by_fr.setdefault(r["Fr"], []).append(r["R_up_N"])
    fr_sorted = sorted(by_fr)
    r_up_median = {fr: _median(by_fr[fr]) for fr in fr_sorted}
    fr_gt = [fr for fr in fr_sorted if fr > 0.2 + 1e-9]
    slope_up = monotonicity_slope(fr_gt, [r_up_median[fr] for fr in fr_gt])
    b_exists = bool(fr_gt) and all(r_up_median[fr] > 0.0 for fr in fr_gt)
    b_monotone = slope_up == slope_up and slope_up > 0.0
    # every single seed, not just the median
    b_all_positive = bool([r for r in ref if r["Fr"] > 0.2 + 1e-9]) and all(
        r["R_up_N"] > 0.0 for r in ref if r["Fr"] > 0.2 + 1e-9)

    # (c) closure of the resistance split
    resid = [r["closure_residual"] for r in rows if r["closure_residual"] == r["closure_residual"]]
    c_max = max(resid) if resid else float("nan")

    # (d) R_up / R_total at Fr = 0.3, h = 0.5 m -- primary reading is the model's
    # default bow (alpha = 30 deg, deadrise beta = 30 deg); the spread over all
    # slope angles and the brief's own (sin a) convention are reported alongside
    d_rows = [r for r in rows if abs(r["Fr"] - 0.3) < 1e-9 and abs(r["thickness_m"] - 0.5) < 1e-9]
    ratio_all = [r["R_up_over_R_total"] for r in d_rows]
    ratio_med = _median(ratio_all)
    ref_rows = [r for r in d_rows if abs(r["angle_deg"] - 30.0) < 1e-9]
    ratio_ref = [r["R_up_over_R_total"] for r in ref_rows]
    ratio_ref_channel = [r["R_up_channel_over_R_total"] for r in ref_rows]
    d_ok = bool(ratio_ref) and 0.1 <= _median(ratio_ref) <= 0.3
    d_beta = _deadrise_sensitivity()

    # (e) static buoyancy/weight balance
    e_residual = _static_buoyancy_residual()

    summary = {
        "case_count": len(rows),
        "grid": {"Fr": list(FR_VALUES), "h": list(H_VALUES), "alpha": list(ANGLE_VALUES),
                 "seeds": list(SEEDS)},
        "steps": meta.get("steps"), "warmup_steps": meta.get("warmup_steps"),
        "n_floes": meta.get("n_floes"), "n_samples": meta.get("n_samples"),
        "dt_s": meta.get("dt"),
        "criteria": {
            "a_contact_vs_analytic": contact_checks,
            "b_ride_up_exists_and_monotone_in_Fr": {
                "R_up_median_by_Fr_at_h0.5_alpha30": {str(k): v for k, v in r_up_median.items()},
                "all_positive_for_Fr_gt_0.2": bool(b_all_positive),
                "slope_N_per_unit_Fr": slope_up,
                "monotone_increasing": bool(b_monotone),
                "pass": bool(b_exists and b_monotone and b_all_positive),
            },
            "c_decomposition_closure": {
                "max_relative_residual": c_max,
                "threshold": 0.02,
                "pass": bool(c_max < 0.02),
            },
            "d_R_up_share_at_Fr0.3_h0.5": {
                "reference_alpha_30_by_seed": ratio_ref,
                "reference_median": _median(ratio_ref),
                "reference_channel_convention_by_seed": ratio_ref_channel,
                "reference_channel_median": _median(ratio_ref_channel),
                "median_all_alpha": ratio_med,
                "fraction_all_cases_in_window": (
                    sum(1 for v in ratio_all if 0.1 <= v <= 0.3) / len(ratio_all)
                    if ratio_all else float("nan")),
                "window": [0.1, 0.3],
                "deadrise_sensitivity": d_beta,
                "pass": bool(d_ok),
            },
            "e_static_buoyancy_balance": {
                "max_abs_vertical_residual_N": e_residual,
                "threshold_N": 1e-10,
                "pass": bool(e_residual < 1e-10),
            },
        },
        "rows": rows,
    }

    # per-alpha and per-h tables (report material)
    tables: Dict[str, List[Dict]] = {"by_alpha": [], "by_thickness": [], "by_fr": []}
    for ang in ANGLE_VALUES:
        sel = [r for r in rows if abs(r["angle_deg"] - ang) < 1e-9]
        tables["by_alpha"].append({"alpha_deg": ang,
                                   "R_up_over_R_total_median": _median([r["R_up_over_R_total"] for r in sel]),
                                   "R_total_median_N": _median([r["R_total_N"] for r in sel])})
    for h in H_VALUES:
        sel = [r for r in rows if abs(r["thickness_m"] - h) < 1e-9]
        tables["by_thickness"].append({"h_m": h,
                                       "R_up_over_R_total_median": _median([r["R_up_over_R_total"] for r in sel]),
                                       "R_total_median_N": _median([r["R_total_N"] for r in sel]),
                                       "R_up_median_N": _median([r["R_up_N"] for r in sel])})
    for fr in FR_VALUES:
        sel = [r for r in rows if abs(r["Fr"] - fr) < 1e-9]
        tables["by_fr"].append({"Fr": fr,
                                "R_up_over_R_total_median": _median([r["R_up_over_R_total"] for r in sel]),
                                "R_up_median_N": _median([r["R_up_N"] for r in sel]),
                                "R_clear_median_N": _median([r["R_clear_N"] for r in sel]),
                                "R_break_median_N": _median([r["R_break_N"] for r in sel]),
                                "R_total_median_N": _median([r["R_total_N"] for r in sel]),
                                "n_climbing_median": _median([r["n_climbing"] for r in sel])})
    summary["tables"] = tables

    os.makedirs(in_dir, exist_ok=True)
    path = os.path.join(in_dir, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def _contact_analytic_checks() -> Dict[str, float]:
    """(a) contact geometry and force vs analytic solutions (relative errors)."""
    from tensordem.ice_rideup import sphere_plane_contact, sphere_plane_force, disk_patch_contact

    slope = BowSlope(angle_deg=30.0)
    plane = slope.slope_patch
    n = plane.normal_t

    # sphere-plane (horizontal facet): penetration = r - height exactly
    from tensordem.ice_rideup import RectPatch
    flat = RectPatch(origin=(0.0, 0.0, 0.0), e1=(1.0, 0.0, 0.0), length1=10.0,
                     normal=(0.0, 0.0, 1.0))
    err_sphere = 0.0
    for height in (1.3, 0.7, 0.25):
        c = torch.tensor([0.0, 0.0, height], dtype=torch.float64)
        got = float(sphere_plane_contact(c, 1.0, flat)["penetration"])
        want = max(0.0, 1.0 - height)
        err_sphere = max(err_sphere, abs(got - want) / max(want, 1e-12))

    # sphere on the incline: N = m g cos(a), |f_t| = mu N
    m, k, mu = 500.0, 5.0e6, 0.2
    n_want = m * 9.80665 * math.cos(slope.alpha)
    pen = n_want / k
    center = plane.origin_t + plane.e1_t * (0.5 * plane.length1) + n * (1.0 - pen)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    slip = -(up - n * float((up * n).sum()))
    slip = slip / slip.norm()
    out = sphere_plane_force(center, slip * 0.5, 1.0, plane, stiffness=k, damping=0.0,
                             friction=mu, tangential_damping=1e12)
    err_n = abs(float(out["normal_force"]) - n_want) / n_want
    err_t = abs(float(out["tangential_force"].norm()) - mu * n_want) / (mu * n_want)

    # disk (cylinder) vs incline: penetration vs the closed form
    r, t = 2.0, 0.5
    centers = torch.tensor([[0.0, 0.0, -0.3]], dtype=torch.float64)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    got = float(disk_patch_contact(centers, quats, torch.tensor([r]), torch.tensor([t]),
                                   (plane,), n_samples=1024)["penetration"])
    n_perp = math.sqrt(max(0.0, 1.0 - float(n[2]) ** 2))
    low_rim = centers[0] - torch.tensor([0.0, 0.0, 0.5 * t], dtype=torch.float64)
    want = -(float(((low_rim - plane.origin_t) * n).sum()) - r * n_perp)
    err_disk = abs(got - want) / abs(want)

    return {"sphere_plane_penetration_rel_err": err_sphere,
            "incline_normal_force_rel_err": err_n,
            "incline_friction_force_rel_err": err_t,
            "disk_incline_penetration_rel_err": err_disk,
            "max_rel_err": max(err_sphere, err_n, err_t, err_disk),
            "threshold": 0.03,
            "pass": bool(max(err_sphere, err_n, err_t, err_disk) < 0.03)}


def _deadrise_sensitivity(*, seeds=(0, 1, 2)) -> Dict:
    """R_up share vs bow deadrise ``beta`` at the reference point (geometry lever)."""
    out = {}
    for beta in (20.0, 25.0, 30.0, 35.0, 40.0):
        ratios = []
        for seed in seeds:
            sim = RideUpSimulation(fr=0.3, thickness=0.5,
                                   slope=BowSlope(angle_deg=30.0, plan_half_angle_deg=beta),
                                   seed=seed, steps=500, warmup_steps=300, dt=0.01,
                                   n_floes=40, n_samples=24, radius_mean=2.0)
            res = sim.run()
            ratios.append(res.ratio_up)
        out[str(beta)] = {"median": _median(ratios), "values": ratios}
    return out


def _static_buoyancy_residual() -> float:
    r = torch.tensor([0.5, 1.5, 4.0], dtype=torch.float64)
    t = torch.tensor([0.3, 0.6, 1.2], dtype=torch.float64)
    z = 0.5 * t - (RHO_ICE / RHO_WATER) * t
    return float(archimedes_buoyancy(z, r, t)["residual"].abs().max())


def make_figures(summary: Dict, out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["rows"]
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # Fig 1: R_up / R_total vs Fr, one curve per ice thickness (alpha = 30)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for h in H_VALUES:
        sel = [r for r in rows if abs(r["thickness_m"] - h) < 1e-9
               and abs(r["angle_deg"] - 30.0) < 1e-9]
        by_fr = {}
        for r in sel:
            by_fr.setdefault(r["Fr"], []).append(r["R_up_over_R_total"])
        xs = sorted(by_fr)
        ys = [_median(by_fr[x]) for x in xs]
        ax.plot(xs, ys, "o-", label=f"h = {h} m")
    ax.axhspan(0.1, 0.3, color="0.85", zorder=0, label="expected band 0.1-0.3")
    ax.set_xlabel("Fr = V/sqrt(g 10 m)")
    ax.set_ylabel("R_up / R_total")
    ax.set_title("Ride-up share vs Froude number (alpha = 30 deg)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "rideup_share_vs_fr.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Fig 2: resistance split vs Fr (stacked)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sel = [r for r in rows if abs(r["angle_deg"] - 30.0) < 1e-9 and abs(r["thickness_m"] - 0.5) < 1e-9]
    by_fr = {}
    for r in sel:
        by_fr.setdefault(r["Fr"], []).append(r)
    xs = sorted(by_fr)
    brk = [_median([r["R_break_N"] for r in by_fr[x]]) / 1e3 for x in xs]
    clr = [_median([r["R_clear_N"] for r in by_fr[x]]) / 1e3 for x in xs]
    up = [_median([r["R_up_N"] for r in by_fr[x]]) / 1e3 for x in xs]
    ax.bar([str(x) for x in xs], brk, label="R_break")
    ax.bar([str(x) for x in xs], clr, bottom=brk, label="R_clear")
    ax.bar([str(x) for x in xs], up, bottom=[b + c for b, c in zip(brk, clr)], label="R_up")
    ax.set_xlabel("Fr")
    ax.set_ylabel("resistance / kN")
    ax.set_title("Resistance split (h = 0.5 m, alpha = 30 deg, median of 3 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    p = os.path.join(out_dir, "resistance_split_vs_fr.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Fig 3: R_up / R_total vs slope angle and thickness (heat map over the sweep)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    mat = []
    for h in H_VALUES:
        row = []
        for ang in ANGLE_VALUES:
            sel = [r["R_up_over_R_total"] for r in rows
                   if abs(r["thickness_m"] - h) < 1e-9 and abs(r["angle_deg"] - ang) < 1e-9]
            row.append(_median(sel))
        mat.append(row)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ANGLE_VALUES)), [f"{a:g}" for a in ANGLE_VALUES])
    ax.set_yticks(range(len(H_VALUES)), [f"{h:g}" for h in H_VALUES])
    ax.set_xlabel("slope angle alpha / deg")
    ax.set_ylabel("ice thickness h / m")
    ax.set_title("median R_up / R_total over Fr and seeds")
    fig.colorbar(im, ax=ax, label="R_up / R_total")
    p = os.path.join(out_dir, "rideup_share_map.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Fig 4: time series of the reference case (force channels)
    torch.set_num_threads(1)
    sim = RideUpSimulation(fr=0.3, thickness=0.5, slope=BowSlope(angle_deg=30.0), seed=0,
                           steps=500, warmup_steps=300, dt=0.01, n_floes=40,
                           n_samples=24, radius_mean=2.0)
    sim.run()
    s = sim.series
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(s["t"], [v / 1e3 for v in s["R_total"]], lw=1.0, label="R_total")
    ax.plot(s["t"], [v / 1e3 for v in s["R_up"]], lw=1.0, label="R_up")
    ax.plot(s["t"], [v / 1e3 for v in s["R_clear"]], lw=1.0, label="R_clear")
    ax.plot(s["t"], [v / 1e3 for v in s["R_break"]], lw=1.0, label="R_break")
    ax.set_xlabel("time / s")
    ax.set_ylabel("instantaneous resistance / kN")
    ax.set_title("Reference case Fr=0.3, h=0.5 m, alpha=30 deg (seed 0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "reference_case_timeseries.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", default="artifacts/ice_rideup")
    ap.add_argument("--aggregate", default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--n-floes", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    if args.aggregate:
        summary = aggregate(args.aggregate)
        print(json.dumps(summary["criteria"], indent=1))
        if args.figures:
            paths = make_figures(summary, os.path.join(args.aggregate, "figures"))
            print("figures:", *paths, sep="\n  ")
        return 0

    if args.shard is None:
        ap.error("either --shard or --aggregate is required")
    path = run_shard(args.shard, args.of, args.out, steps=args.steps, warmup=args.warmup,
                     n_floes=args.n_floes, n_samples=args.n_samples, dt=args.dt)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())