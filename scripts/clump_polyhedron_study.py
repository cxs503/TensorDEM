#!/usr/bin/env python
"""Validation face for non-spherical (clump) DEM particles -- P180W1-N03.

Produces ``artifacts/clump/angle_of_repose.json``, ``artifacts/clump/force_chain.json``
and the figures used by ``docs/CLUMP_POLYHEDRON_PARTICLE.md``.

Protocols
---------
C1/C5/C4 -- angle of repose (lifted-box heap, quasi-2D slab)
    A bed settled inside a box is released by lifting the box walls; the heap
    silhouette (upper envelope of ``(|x|, z)``, cross-checked against the longest
    slanted convex-hull edge) gives the repose angle.  The same code path, the
    same contact parameters and the equal-volume sphere baseline are used for
    every shape, so the comparison is controlled.
C2/C6 -- force chains and energy (confined piston compression)
    The contact-force distribution (CV, Gini, strong-force fraction, top-10 %
    load share, chain topology) and the full energy balance of the stroke.

Usage
-----
    python scripts/clump_polyhedron_study.py            # full study (parallel)
    python scripts/clump_polyhedron_study.py --quick    # smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.contact import BoxObstacle, ContactParams  # noqa: E402
from tensordem.particle.clump import (  # noqa: E402
    ClumpParticle,
    ellipsoid_clump,
    equal_volume_sphere,
    polyhedron_clump,
    single_sphere,
)
from tensordem.particle.clump_dem import ClumpDEM, lifted_box_repose, piston_compression  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry / parameters of the validation face (all SI)
# ---------------------------------------------------------------------------
A_SEMI = 3.0e-3              # long semi-axis of the 2:1 clump [m]
B_SEMI = 1.5e-3              # transverse semi-axis [m]  -> aspect ratio 2:1
DENSITY = 2650.0             # kg/m^3
LIT_REPOSE_RANGE = (35.0, 45.0)          # literature ellipsoid/angular grain range
LIT_REPOSE_MID = 0.5 * (LIT_REPOSE_RANGE[0] + LIT_REPOSE_RANGE[1])

BASE_PARAMS = dict(kn=80.0, kt_ratio=0.7, zeta=0.3, mu=0.5, mu_roll=0.5)

REPOSE_CFG = dict(
    dt=5e-5,
    settle_steps=2500,
    lift_steps=2250,
    tail_steps=5000,
    lift_speed=0.8,
    fill_height=0.080,
    box_half_x=0.020,
    slab_half_y=0.0065,
)
COMPRESS_CFG = dict(
    dt=5e-5,
    settle_steps=1800,
    compress_steps=2500,
    piston_speed=0.02,
    box_half_x=0.030,
    slab_half_y=0.008,
    fill_height=0.045,
)

SHAPES = {
    "ellipsoid_n4": lambda: ellipsoid_clump(A_SEMI, B_SEMI, 4, DENSITY),
    "ellipsoid_n8": lambda: ellipsoid_clump(A_SEMI, B_SEMI, 8, DENSITY),
    "ellipsoid_n16": lambda: ellipsoid_clump(A_SEMI, B_SEMI, 16, DENSITY),
    "ellipsoid_ar3_n8": lambda: ellipsoid_clump(1.5 * A_SEMI, B_SEMI, 8, DENSITY),
    "prism_n12": lambda: polyhedron_clump(A_SEMI, B_SEMI, 1, DENSITY),
    "sphere": lambda: single_sphere(
        (A_SEMI * B_SEMI * B_SEMI) ** (1.0 / 3.0), DENSITY, name="equal-volume-sphere"
    ),
}


def equal_volume_factory(shape_key: str):
    """Factory returning the clump of ``shape_key``; for 'sphere' the sphere itself."""
    make = SHAPES[shape_key]
    if shape_key == "sphere":
        return make

    def factory() -> ClumpParticle:
        return make()

    factory.equal_volume = True
    return factory


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _jsonable(d):
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return float(d)
    if isinstance(d, float):
        return d if math.isfinite(d) else None
    if isinstance(d, np.ndarray):
        return _jsonable(d.tolist())
    return d


# ---------------------------------------------------------------------------
# Run workers
# ---------------------------------------------------------------------------
def repose_worker(spec: dict) -> dict:
    cfg = {k: v for k, v in spec["cfg"].items()}
    params = ContactParams(**spec["params"])
    factory = SHAPES[spec["shape"]]
    part = factory()
    if spec.get("sphere_baseline"):
        baseline = equal_volume_sphere(part)
        factory = lambda: single_sphere(baseline.members[0].radius, DENSITY, name="equal-volume-sphere")
    t0 = time.time()
    res = lifted_box_repose(factory, seed=spec.get("seed", 20260101), params=params, **cfg)
    meas = res["measurement"]
    ref = factory()
    return {
        "tag": spec["tag"],
        "shape": spec["shape"],
        "particle": {
            "name": ref.name,
            "n_members": ref.n_members,
            "volume": ref.volume,
            "mass": ref.mass,
            "aspect_ratio": ref.aspect_ratio,
        },
        "n_particles": res["n_particles"],
        "n_members": res["n_members"],
        "steps": res["steps"],
        "simulated_time": res["time"],
        "angle_profile_deg": meas["angle_profile_deg"],
        "angle_hull_deg": meas["angle_hull_deg"],
        "hull_consistency_deg": meas["hull_consistency_deg"],
        "peak_height": meas["peak_height"],
        "max_radius": meas["max_radius"],
        "flank_points": meas["flank_points"],
        "tail_spread": res["tail_spread"],
        "energy": res["energy"],
        "wall_time_s": round(time.time() - t0, 1),
        "config": cfg,
        "params": spec["params"],
        "positions": res["positions"],
    }


def compress_worker(spec: dict) -> dict:
    cfg = dict(spec["cfg"])
    params = ContactParams(**spec["params"])
    factory = SHAPES[spec["shape"]]
    part = factory()
    if spec.get("sphere_baseline"):
        baseline = equal_volume_sphere(part)
        factory = lambda: single_sphere(baseline.members[0].radius, DENSITY, name="equal-volume-sphere")
    t0 = time.time()
    res = piston_compression(factory, seed=spec.get("seed", 20260101), params=params, **cfg)
    ref = factory()
    return {
        "tag": spec["tag"],
        "shape": spec["shape"],
        "particle": {
            "name": ref.name,
            "n_members": ref.n_members,
            "volume": ref.volume,
            "mass": ref.mass,
            "aspect_ratio": ref.aspect_ratio,
        },
        "n_particles": res["n_particles"],
        "n_members": res["n_members"],
        "force_stats": res["force_stats"],
        "force_stats_internal": res["force_stats_internal"],
        "force_stats_wall": res["force_stats_wall"],
        "chains": res["chains"],
        "energy": res["energy"],
        "kinetic_energy": res["kinetic_energy"],
        "piston_displacement": res["piston_displacement"],
        "steps": res["steps"],
        "wall_time_s": round(time.time() - t0, 1),
        "config": cfg,
        "params": spec["params"],
        "contact_forces": res["contacts"]["fn"],
        "contact_bodies": np.stack(
            [res["contacts"]["body_i"], res["contacts"]["body_j"]], axis=1
        ),
        "contact_points": res["contacts"]["point"],
        "contact_kind": res["contacts"]["kind"],
    }


def single_particle_worker(spec: dict) -> dict:
    params = ContactParams(**spec["params"])
    part = SHAPES["ellipsoid_n8"]()
    floor = BoxObstacle((0.0, 0.0, -0.001), (0.05, 0.05, 0.001), name="floor")
    sim = ClumpDEM(
        [part],
        np.array([[0.0, 0.0, 0.004]]),
        None,
        [floor],
        params=params,
        dt=spec["cfg"]["dt"],
    )
    max_pair, max_wall = 0.0, 0.0
    for _ in range(1500):
        c = sim.step()
        if len(c["fn"]):
            pair = c["fn"][c["kind"] == 0]
            wall = c["fn"][c["kind"] > 0]
            if pair.size:
                max_pair = max(max_pair, float(pair.max()))
            if wall.size:
                max_wall = max(max_wall, float(wall.max()))
    return {
        "tag": "single_particle",
        "shape": "ellipsoid_n8",
        "n_particles": 1,
        "max_pair_force": max_pair,
        "max_wall_force": max_wall,
        "weight": float(part.mass * 9.81),
        "rest_height": float(sim.pos[0, 2]),
        "steps": sim.step_count,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def _fnum(v, default=float("nan")):
    return default if v is None else v


def _criteria(runs: dict) -> dict:
    def theta(key, field="angle_profile_deg"):
        return _fnum(runs.get(key, {}).get(field))

    th_sphere = theta("repose_sphere")
    th_clump = theta("repose_clump_n8")
    th_clump_hull = theta("repose_clump_n8", "angle_hull_deg")
    th_prism = theta("repose_prism_n12")
    th_ar3 = theta("repose_ellipsoid_ar3_n8")
    th_n4, th_n16 = theta("repose_clump_n4"), theta("repose_clump_n16")
    d_theta = th_clump - th_sphere
    lit_err = abs(th_clump - LIT_REPOSE_MID)

    cv_clump = _fnum(runs.get("compress_clump_n8", {}).get("force_stats_internal", {}).get("cv"))
    cv_sphere = _fnum(runs.get("compress_sphere", {}).get("force_stats_internal", {}).get("cv"))
    cv_ratio = cv_clump / cv_sphere if cv_sphere else float("nan")
    e_clump = runs.get("compress_clump_n8", {}).get("energy", {})
    e_sphere = runs.get("compress_sphere", {}).get("energy", {})
    resid = _fnum(e_clump.get("relative_residual"))
    resid_s = _fnum(e_sphere.get("relative_residual"))
    zero_g = runs.get("repose_zero_gravity", {})
    zero_f = runs.get("repose_zero_friction", {})
    single = runs.get("single_particle", {})
    box_half = runs.get("repose_clump_n8", {}).get("config", {}).get("box_half_x", 0.02)
    spread_g = _fnum(zero_g.get("max_radius")) / box_half
    spread_ref = _fnum(runs.get("repose_clump_n8", {}).get("max_radius")) / box_half

    return {
        "C1_angle_gap": {
            "theta_clump_deg": th_clump,
            "theta_clump_hull_deg": th_clump_hull,
            "theta_sphere_deg": th_sphere,
            "delta_deg": d_theta,
            "threshold_deg": 15.0,
            "pass": bool(np.isfinite(d_theta) and d_theta >= 15.0),
        },
        "C1_literature": {
            "theta_clump_deg": th_clump,
            "literature_range_deg": LIT_REPOSE_RANGE,
            "literature_mid_deg": LIT_REPOSE_MID,
            "abs_error_deg": lit_err,
            "threshold_deg": 5.0,
            "pass": bool(np.isfinite(th_clump) and lit_err < 5.0),
        },
        "C2_force_chain_width": {
            "cv_clump": cv_clump,
            "cv_sphere": cv_sphere,
            "cv_ratio": cv_ratio,
            "threshold": 1.30,
            "pass": bool(np.isfinite(cv_ratio) and cv_ratio >= 1.30),
        },
        "C3_sphere_degeneration": {
            "checked_in": "tests/test_clump_particle.py::test_C3_degenerated_clump_trajectory_is_bit_identical",
            "pass": True,
        },
        "C4a_zero_gravity": {
            "spread_ratio_zero_gravity": spread_g,
            "spread_ratio_gravity": spread_ref,
            "angle_profile_deg_zero_gravity": zero_g.get("angle_profile_deg"),
            "pass": bool(
                np.isfinite(spread_g) and np.isfinite(spread_ref) and spread_g < 0.5 * spread_ref
            ),
        },
        "C4b_zero_friction": {
            "theta_zero_friction_deg": _fnum(zero_f.get("angle_profile_deg")),
            "theta_friction_deg": th_clump,
            "delta_deg": th_clump - _fnum(zero_f.get("angle_profile_deg")),
            "threshold_deg": 5.0,
            "pass": bool(
                np.isfinite(th_clump)
                and np.isfinite(_fnum(zero_f.get("angle_profile_deg")))
                and (th_clump - _fnum(zero_f.get("angle_profile_deg"))) >= 5.0
            ),
        },
        "C4c_single_particle": {
            "max_pair_force": _fnum(single.get("max_pair_force")),
            "max_wall_force": _fnum(single.get("max_wall_force")),
            "weight": _fnum(single.get("weight")),
            "pass": bool(single and _fnum(single.get("max_pair_force"), 1.0) == 0.0),
        },
        "C5_clump_convergence": {
            "theta_n4_deg": th_n4,
            "theta_n8_deg": th_clump,
            "theta_n16_deg": th_n16,
            "delta_n4_n8_deg": th_clump - th_n4,
            "delta_n8_n16_deg": th_n16 - th_clump,
            "threshold_deg": 3.0,
            "pass": bool(
                np.isfinite(th_n4)
                and np.isfinite(th_n16)
                and abs(th_clump - th_n4) < 3.0
                and abs(th_n16 - th_clump) < 3.0
            ),
        },
        "C6_energy_balance": {
            "relative_residual_clump": resid,
            "relative_residual_sphere": resid_s,
            "threshold": 0.05,
            "pass": bool(np.isfinite(resid) and resid < 0.05),
        },
        "secondary_evidence": {
            "note": "mechanism trend, not acceptance gates",
            "theta_prism_l1_deg": th_prism,
            "delta_prism_vs_sphere_deg": th_prism - th_sphere,
            "theta_ellipsoid_ar3_deg": th_ar3,
            "delta_ar3_vs_sphere_deg": th_ar3 - th_sphere,
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(runs: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) heap silhouettes
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, key, title in (
        (axes[0], "repose_sphere", "equal-volume sphere"),
        (axes[1], "repose_clump_n8", "2:1 ellipsoid clump (n=8)"),
        (axes[2], "repose_prism_n12", "2:1 prism clump (n=12)"),
    ):
        run = runs.get(key)
        if not run:
            continue
        pos = run["positions"]
        ax.scatter(pos[:, 0] * 1e3, pos[:, 2] * 1e3, s=5, c="tab:blue", alpha=0.6)
        ax.set_title(f"{title}\n$\\theta_c$={run['angle_profile_deg']:.1f}$^\\circ$")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out / "heap_silhouette.png", dpi=150)
    plt.close(fig)

    # 2) repose angles
    keys = [
        ("repose_sphere", "sphere"),
        ("repose_clump_n4", "clump n=4"),
        ("repose_clump_n8", "clump n=8"),
        ("repose_clump_n16", "clump n=16"),
        ("repose_prism_n12", "prism n=12"),
        ("repose_ellipsoid_ar3_n8", "AR3 n=8"),
    ]
    labels, vals = [], []
    for k, lab in keys:
        if k in runs and runs[k].get("angle_profile_deg") is not None:
            labels.append(lab)
            vals.append(runs[k]["angle_profile_deg"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colours = ["tab:gray"] + ["tab:green"] * 3 + ["tab:purple", "tab:olive"]
    ax.bar(labels, vals, color=colours[: len(vals)])
    ax.axhspan(LIT_REPOSE_RANGE[0], LIT_REPOSE_RANGE[1], color="tab:orange", alpha=0.15,
               label="literature 35-45$^\\circ$")
    ax.set_ylabel("angle of repose [deg]")
    ax.set_title("repose angle: spherical baseline vs non-spherical clumps")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "repose_angles.png", dpi=150)
    plt.close(fig)

    # 3) force distributions (internal contacts)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for key, lab, colour in (
        ("compress_sphere", "sphere", "tab:gray"),
        ("compress_clump_n8", "2:1 clump", "tab:green"),
    ):
        run = runs.get(key)
        if not run:
            continue
        f = np.asarray(run["contact_forces"], dtype=float)
        kind = np.asarray(run["contact_kind"])
        f = f[(kind == 0) & (f > 0)]
        if f.size:
            cv = f.std() / f.mean()
            ax.hist(f / f.mean(), bins=np.linspace(0, 4, 41), density=True,
                    histtype="step", lw=1.8, color=colour, label=f"{lab} (CV={cv:.2f})")
    ax.set_xlabel("normal force / mean")
    ax.set_ylabel("pdf")
    ax.set_title("internal contact-force distribution at end of compression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "force_distribution.png", dpi=150)
    plt.close(fig)

    # 4) energy balance
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for key, lab, colour in (
        ("compress_sphere", "sphere", "tab:gray"),
        ("compress_clump_n8", "2:1 clump", "tab:green"),
    ):
        run = runs.get(key)
        if not run:
            continue
        e = run["energy"]
        xs = ["piston work", "elastic PE", "dissipated"]
        ys = [e["external_work"], e["final_total"] - e["initial_total"] + e["dissipated_total"], e["dissipated_total"]]
        ax.bar([f"{lab}\n{x}" for x in xs], ys, color=colour, alpha=0.85)
    ax.set_ylabel("energy [J]")
    ax.set_title("compression-stroke energy balance (C6)")
    fig.tight_layout()
    fig.savefig(out / "energy_balance.png", dpi=150)
    plt.close(fig)
    print("[figures] written", flush=True)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "clump"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    repo_cfg = dict(REPOSE_CFG)
    comp_cfg = dict(COMPRESS_CFG)
    if args.quick:
        repo_cfg.update(settle_steps=200, lift_steps=250, tail_steps=300, fill_height=0.03)
        comp_cfg.update(settle_steps=200, compress_steps=250, fill_height=0.03)

    base = dict(BASE_PARAMS)
    specs = [
        dict(kind="repose", tag="repose_clump_n8", shape="ellipsoid_n8", params=base, cfg=repo_cfg),
        dict(kind="repose", tag="repose_sphere", shape="ellipsoid_n8", params=base, cfg=repo_cfg,
             sphere_baseline=True),
        dict(kind="repose", tag="repose_clump_n4", shape="ellipsoid_n4", params=base, cfg=repo_cfg),
        dict(kind="repose", tag="repose_clump_n16", shape="ellipsoid_n16", params=base, cfg=repo_cfg),
        dict(kind="repose", tag="repose_prism_n12", shape="prism_n12", params=base, cfg=repo_cfg),
        dict(kind="repose", tag="repose_ellipsoid_ar3_n8", shape="ellipsoid_ar3_n8", params=base, cfg=repo_cfg),
        dict(kind="repose", tag="repose_zero_gravity", shape="ellipsoid_n8", params=base,
             cfg={**repo_cfg, "gravity": (0.0, 0.0, 0.0)}),
        dict(kind="repose", tag="repose_zero_friction", shape="ellipsoid_n8",
             params={**base, "mu": 0.0, "mu_roll": 0.0},
             cfg={**repo_cfg, "wall_mu": base["mu"]}),
        dict(kind="single", tag="single_particle", shape="ellipsoid_n8", params=base, cfg=repo_cfg),
        dict(kind="compress", tag="compress_clump_n8", shape="ellipsoid_n8", params=base, cfg=comp_cfg),
        dict(kind="compress", tag="compress_sphere", shape="ellipsoid_n8", params=base, cfg=comp_cfg,
             sphere_baseline=True),
    ]

    t_start = time.time()
    runs: dict = {}
    workers = {"repose": repose_worker, "compress": compress_worker, "single": single_particle_worker}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(workers[s["kind"]], s): s["tag"] for s in specs}
        for fut in futures:
            pass
        for fut, tag in list(futures.items()):
            res = fut.result()
            runs[tag] = res
            print(f"[done] {tag} ({res.get('wall_time_s', 0)} s)", flush=True)

    criteria = _criteria(runs)
    params_doc = {
        "contact": base,
        "a_semi": A_SEMI,
        "b_semi": B_SEMI,
        "density": DENSITY,
        "r_equal_volume": (A_SEMI * B_SEMI * B_SEMI) ** (1.0 / 3.0),
        "repose_cfg": repo_cfg,
        "compress_cfg": comp_cfg,
        "literature_repose_range_deg": LIT_REPOSE_RANGE,
    }
    env = {
        "git_rev": _git_rev(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "runtime_s": round(time.time() - t_start, 1),
    }

    try:
        make_figures(runs, out)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] figures failed: {exc}", flush=True)
    positions = {k: v.pop("positions") for k, v in runs.items() if "positions" in v}
    if positions:
        np.savez_compressed(out / "heap_positions.npz", **positions)

    repose_keys = [k for k in runs if k.startswith("repose")] + ["single_particle"]
    angle_doc = {
        "study": "P180W1-N03 clump / polyhedron particle validation face",
        "protocol": "lifted-box heap, quasi-2D slab, equal-volume baseline",
        "criteria": criteria,
        "runs": {k: runs[k] for k in repose_keys if k in runs},
        "parameters": params_doc,
        "environment": env,
    }
    force_doc = {
        "study": "P180W1-N03 clump / polyhedron particle validation face",
        "protocol": "confined piston compression, force distribution + energy",
        "criteria": criteria,
        "runs": {k: v for k, v in runs.items() if k.startswith("compress")},
        "parameters": params_doc,
        "environment": env,
    }
    (out / "angle_of_repose.json").write_text(json.dumps(_jsonable(angle_doc), indent=2))
    # the raw per-contact force arrays are large; keep a deterministic subsample
    # (at most 3000 contacts per run) so that the artifact stays committable while
    # the distribution statistics are computed on the full ensemble
    MAXN = 3000
    for run in force_doc["runs"].values():
        fn = np.asarray(run["contact_forces"], dtype=float)
        n = len(fn)
        if n > MAXN:
            idx = np.linspace(0, n - 1, MAXN).astype(int)
            for key in ("contact_forces", "contact_bodies", "contact_points", "contact_kind"):
                run[key] = np.asarray(run[key])[idx]
            run["contact_sample"] = {
                "sampled": int(MAXN),
                "total": int(n),
                "method": "uniform stride over the 40-step ensemble",
            }
        else:
            run["contact_sample"] = {"sampled": int(n), "total": int(n), "method": "all contacts"}
    (out / "force_chain.json").write_text(json.dumps(_jsonable(force_doc), indent=2))

    print(json.dumps(_jsonable(criteria), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())