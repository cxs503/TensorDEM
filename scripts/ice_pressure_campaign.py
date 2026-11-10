"""DEM campaign for the spatial statistics of the ice contact pressure field.

Node P82W2-N07: correlation length ``L_c`` of the contact pressure field and the
ISO 19906 pressure-area exponent ``n``, on three ice-slab / structure scenes
(flat wall, vertical cylinder, cone).

Each task is one ``(scene, seed)`` pair:

* run the DEM (:func:`tensordem.ice_pressure_stats.run_ice_plate_scenario`),
* re-coarse-grain the stored raw force windows at several scales
  (``G*F / G*A`` normalised convolution) and compute, for each scale,
  the pooled ACF, the correlation length (1/e crossing) and the design-pressure
  curve ``p_d(A)`` in both the typical-patch and the worst-patch reduction,
* check the 1/2/4-shard reproducibility of every reported number,
* write a compact JSON (no raw arrays) so the run can be sharded over a cluster.

Usage
-----
``python scripts/ice_pressure_campaign.py --shard 0/12``   # cluster shard
``python scripts/ice_pressure_campaign.py --task wall --seed 11``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tensordem.ice_pressure_stats as ips  # noqa: E402

OUT = ROOT / "artifacts" / "ice_pressure_stats"

#: production configuration (physical calibration: see the report)
CONFIG = dict(
    grid=(0.05, 0.05),
    width=2.0,
    length_factor=1.4,
    thickness=1.0,
    stroke=0.35,
    dt=3.0e-5,
    solid_fraction=0.60,
    n_max=420,
    sigma_c=5.0e6,        # ice crushing strength [Pa] (literature 3-10 MPa)
    k_n=3.0e8,            # E_eff * r_eq with E_eff ~ 3 GPa for a confined rubble
    mass_scale=20.0,      # quasi-static DEM mass scaling (keeps dt = 3e-5 s)
    struct_radius=0.60,
    cone_half_angle_deg=20.0,
    settle_steps=400,
    save_every=58,        # 201 snapshots over the 11666-step stroke
)

SCENES = ("wall", "cylinder", "cone")
SEEDS = (11, 12, 13, 14)
COARSE = (0.10, 0.15, 0.20, 0.30)
PRIMARY_COARSE = 0.15
AREAS = np.logspace(np.log10(0.015), np.log10(2.0), 14)


def run_task(scene: str, seed: int) -> dict:
    t0 = time.time()
    rec = ips.run_ice_plate_scenario(scene, seed=seed, coarse_length=PRIMARY_COARSE, **CONFIG)
    out: dict = {
        "scene": scene,
        "seed": int(seed),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()},
        "primary_coarse_length": PRIMARY_COARSE,
        "meta": rec.meta,
        "runtime_s": None,
        "shape": [int(v) for v in rec.fields.shape],
        "dx": rec.dx,
        "dz": rec.dz,
        "area_range": [float(v) for v in rec.area_bounds()],
        "p_max": float(rec.fields.max()),
        "p_mean": float(rec.fields.mean()),
        "p_cell_active_frac": float((rec.fields > 0.01 * rec.fields.max()).mean()),
        "coarse": {},
    }

    # The pressure statistics are taken on the steady crushing window: the first
    # 40 % of the stroke is the compaction transient (the layer is still being
    # densified and the transmitted load grows), during which the pressure field
    # is not stationary.  ``L_c`` and ``p_d`` of the full window are reported as
    # a sensitivity in ``full_window`` below.
    T = rec.n_samples
    rec_steady = rec.slice(int(0.4 * T), T)
    out["steady_window"] = [int(0.4 * T), int(T)]

    # ---- per coarse-graining scale -----------------------------------
    for cl in COARSE:
        rr = rec_steady if cl == PRIMARY_COARSE else rec_steady.regrain(cl)
        lc = rr.correlation_length()
        r, R = rr.radial_acf()
        entry = {
            "L_c": None if math.isinf(lc) else float(lc),
            "acf_r": [float(v) for v in r],
            "acf_R": [float(v) for v in R],
            "branch_mean": {},
            "branch_max": {},
        }
        for red in ("mean", "max"):
            A_eff, p_d, n_win = rr.pressure_area_curve(AREAS, reduce=red)
            ok = np.isfinite(A_eff) & np.isfinite(p_d) & (p_d > 0)
            f_all = ips.fit_pressure_area_law(A_eff[ok], p_d[ok])
            # continuum-valid range: a patch must contain >= 2 correlation
            # areas for the area average to be a continuum average
            cut = 2.0 * lc * lc if np.isfinite(lc) else 0.0
            ok2 = ok & (A_eff >= cut)
            f_cut = (
                ips.fit_pressure_area_law(A_eff[ok2], p_d[ok2])
                if int(ok2.sum()) >= 3
                else None
            )
            entry[f"branch_{red}"] = {
                "A_eff": [float(v) if np.isfinite(v) else None for v in A_eff],
                "p_d": [float(v) if np.isfinite(v) else None for v in p_d],
                "n_windows": [int(v) for v in n_win],
                "fit_all": {
                    "n": f_all.n, "p0": f_all.p0, "r2": f_all.r2,
                    "n_points": f_all.n_points,
                },
                "fit_cut": None
                if f_cut is None
                else {"n": f_cut.n, "p0": f_cut.p0, "r2": f_cut.r2,
                      "n_points": f_cut.n_points, "A_cut": float(cut)},
            }
        out["coarse"][f"{cl:.2f}"] = entry

    # ---- convergence in the number of snapshots (primary scale) ------
    n_st = rec_steady.n_samples
    out["convergence"] = rec_steady.correlation_length_convergence(
        [10, 25, 50, 100, n_st]
    )
    out["convergence_full"] = rec.correlation_length_convergence(
        [10, 25, 50, 100, T]
    )
    out["L_c_full_window"] = rec.correlation_length()
    out["directional_acf"] = {
        "horizontal": [list(map(float, v)) for v in rec_steady.directional_acf(axis=1)],
        "vertical": [list(map(float, v)) for v in rec_steady.directional_acf(axis=0)],
    }

    # ---- sharding reproducibility ------------------------------------
    A_probe = np.array([0.05, 0.25, 1.0])
    ref = None
    shard_checks = []
    for k in (1, 2, 4):
        r_acf = rec_steady.pooled_acf(n_shards=k)
        lc_k = rec_steady.correlation_length(n_shards=k)
        A_eff, p_mean, _ = rec_steady.pressure_area_curve(A_probe, reduce="mean", n_shards=k)
        A_eff, p_max, _ = rec_steady.pressure_area_curve(A_probe, reduce="max", n_shards=k)
        cur = {
            "n_shards": k,
            "acf": r_acf,
            "L_c": None if math.isinf(lc_k) else float(lc_k),
            "p_mean": p_mean,
            "p_max": p_max,
        }
        if ref is None:
            ref = cur
        else:
            d_acf = float(np.max(np.abs(cur["acf"] - ref["acf"])))
            d_pm = float(np.nanmax(np.abs(cur["p_mean"] - ref["p_mean"]) / ref["p_mean"]))
            d_px = float(np.nanmax(np.abs(cur["p_max"] - ref["p_max"]) / ref["p_max"]))
            lc_ref = ref["L_c"]
            d_lc = (
                float(abs(cur["L_c"] - lc_ref) / lc_ref)
                if (lc_ref and cur["L_c"])
                else 0.0
            )
            shard_checks.append(
                {"n_shards": k, "d_acf": d_acf, "d_L_c": d_lc,
                 "d_p_mean": d_pm, "d_p_max": d_px}
            )
    out["sharding"] = shard_checks

    # ---- fields for the report figures -------------------------------
    rr = rec_steady  # primary scale
    out["mean_field"] = [[float(v) for v in row] for row in rr.mean_field()]
    out["peak_field"] = [[float(v) for v in row] for row in rr.max_field()]
    snap = rr.fields[int(np.argmax(rr.fields.max(axis=(1, 2))))]
    out["peak_snapshot"] = [[float(v) for v in row] for row in snap]
    out["runtime_s"] = time.time() - t0
    return out


def task_list() -> list[tuple[str, int]]:
    return [(s, seed) for s in SCENES for seed in SEEDS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="i/n shard selector")
    ap.add_argument("--task", default=None, help="single scene")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    tasks = task_list()
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        tasks = [t for k, t in enumerate(tasks) if k % n == i]
    if args.task:
        tasks = [t for t in tasks if t[0] == args.task]
    if args.seed is not None:
        tasks = [t for t in tasks if t[1] == args.seed]
    OUT.mkdir(parents=True, exist_ok=True)
    for scene, seed in tasks:
        res = run_task(scene, seed)
        path = OUT / f"run_{scene}_{seed}.json"
        path.write_text(json.dumps(res, indent=1))
        lc = res["coarse"][f"{PRIMARY_COARSE:.2f}"]["L_c"]
        fm = res["coarse"][f"{PRIMARY_COARSE:.2f}"]["branch_mean"]["fit_cut"]
        print(
            f"[{scene} seed={seed}] t={res['runtime_s']:.1f}s "
            f"L_c={lc:.3f} m  n(mean,cut)={fm['n']:.3f} R2={fm['r2']:.3f} "
            f"p_d(1m2)={fm['p0']/1e6:.2f} MPa -> {path.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())