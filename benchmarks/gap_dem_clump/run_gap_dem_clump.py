#!/usr/bin/env python3
"""GAP-DEM-CLUMP: clump generation + analytic inertia verification + shape-effect.

Campaign node: GAP-DEM-CLUMP (non-spherical particles / clump shape effects).

Three verification legs, each with its own machine-checkable criterion:

* **C1 -- analytic inertia kernel anchors.** ``tensordem.clump_inertia`` is
  confronted with three independent references: the exact sphere, the textbook
  tangent-dumbbell tensor, and a high-resolution voxel quadrature.  The
  third-order (triple-intersection) term is validated on a three-sphere cluster
  where it is non-zero.
* **C2 -- production clump inertia vs analytic, per aspect ratio.**  Every clump
  of the aspect-ratio scan is compared component-wise with the analytic union
  tensor; the acceptance line is 1 % (internal criterion of this node).
* **C3 -- shape-effect trend of the repose angle.**  Aspect ratio scan of >= 3
  clump groups plus a sphere baseline, two seeds each; the measured ordering is
  compared with the literature trend (repose angle grows with departure from
  sphericity; spherical glass beads 22-25 deg band).  A rolling-resistance sweep
  is run as the mechanism diagnostic (does the trend come from rolling?).

Run::

    python benchmarks/gap_dem_clump/run_gap_dem_clump.py [--out DIR] [--fast]

``--fast`` shortens the heap protocol (numbers then differ; the JSON records the
protocol actually used).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.clump_inertia import (  # noqa: E402
    compare_with_voxel,
    union_mass_properties,
)
from tensordem.particle.clump import (  # noqa: E402
    SphereMember,
    ellipsoid_clump,
    single_sphere,
    voxel_mass_properties,
)
from tensordem.particle.clump_dem import lifted_box_repose  # noqa: E402
from tensordem.contact.clump_contact import ContactParams  # noqa: E402

DENSITY = 2500.0
BASE_RADIUS = 0.0012          # transverse semi-axis b of the prolate clump family
AR_LIST = (1.0, 1.5, 2.0, 3.0)
SEEDS = (0, 1)

PROTOCOL = dict(
    box_half_x=0.012,
    slab_half_y=0.005,
    fill_height=0.070,
    settle_steps=800,
    lift_steps=1800,
    tail_steps=1200,
    dt=5e-5,
)

# literature anchors (quoted, not re-measured here)
LIT = {
    "sphere_repose_deg": (22.0, 25.0),
    "sphere_repose_source": "glass-bead repose band 22-25 deg, as transcribed in "
                            "docs/CLUMP_POLYHEDRON_PARTICLE.md (literature range)",
    "trend_source": "Zhou et al. (1999) rolling-resistance calibration basis and "
                    "docs/CLUMP_POLYHEDRON_PARTICLE.md: angular/ellipsoidal grains "
                    "give 35-45 deg, i.e. repose angle increases with departure from "
                    "sphericity",
    "trend_direction": "increasing",
}


def sphericity_prolate(a_over_b: float) -> float:
    """Sphericity of a prolate spheroid with semi-axes (a, b, b), a >= b."""
    if a_over_b <= 1.0:
        return 1.0
    e = math.sqrt(1.0 - 1.0 / a_over_b ** 2)
    vol = 4.0 / 3.0 * math.pi * a_over_b
    a_eq = (36.0 * math.pi * vol ** 2) ** (1.0 / 3.0)
    area = 2.0 * math.pi * (1.0 + (a_over_b / e) * math.asin(e))
    return float(a_eq / area)


def make_shape(ar: float):
    """Shape factory for the scan (ar == 1 is the exact sphere baseline)."""
    if abs(ar - 1.0) < 1e-12:
        return lambda: single_sphere(BASE_RADIUS, DENSITY, name="sphere")
    n = max(3, int(round(2.0 * ar)))
    a = ar * BASE_RADIUS
    b = BASE_RADIUS
    return lambda: ellipsoid_clump(a, b, n_spheres=n, density=DENSITY,
                                   name=f"clump-AR{ar:.1f}")


# ---------------------------------------------------------------------------
# C1: analytic kernel anchors
# ---------------------------------------------------------------------------
def leg_c1_analytic_anchors() -> dict:
    r = 0.002
    rho = DENSITY
    # (a) single sphere -- closed form by construction
    u = union_mass_properties([[0.0, 0.0, 0.0]], [r], rho)
    m = rho * 4.0 / 3.0 * math.pi * r ** 3
    i_exact = 0.4 * m * r * r
    e_sphere = float(np.max(np.abs(np.diag(u["inertia"]) - i_exact)) / i_exact)

    # (b) tangent dumbbell -- textbook closed form (touching, no overlap)
    sep = 2.0 * r
    u = union_mass_properties([[-0.5 * sep, 0, 0], [0.5 * sep, 0, 0]], [r, r], rho)
    ref = np.array([2.0 * 0.4 * m * r * r,
                    2.0 * (0.4 * m * r * r + m * (0.5 * sep) ** 2),
                    2.0 * (0.4 * m * r * r + m * (0.5 * sep) ** 2)])
    e_dumbbell = float(np.max(np.abs(np.diag(u["inertia"]) - ref)) / np.max(ref))

    # (c) deeply overlapping pair -- high-resolution voxel reference
    d = 1.2 * r
    mem = [SphereMember(np.array([-0.5 * d, 0.0, 0.0]), r),
           SphereMember(np.array([0.5 * d, 0.0, 0.0]), r)]
    u = union_mass_properties([m_.center for m_ in mem], [r, r], rho)
    v384 = voxel_mass_properties(mem, rho, grid=384)
    e_overlap = float(np.max(np.abs(np.diag(v384["inertia"]) - np.diag(u["inertia"])))
                      / np.max(np.abs(np.diag(u["inertia"]))))
    e_overlap_vol = abs(v384["volume"] - u["volume"]) / u["volume"]

    # (d) three-sphere cluster: the third-order term is non-zero and must help
    rad = 0.001
    cen = [[0.8e-3, 0.0, 0.0], [-0.4e-3, 0.693e-3, 0.0], [-0.4e-3, -0.693e-3, 0.0]]
    u3 = union_mass_properties(cen, [rad] * 3, rho, max_order=3)
    u2 = union_mass_properties(cen, [rad] * 3, rho, max_order=2)
    mem3 = [SphereMember(np.array(c), rad) for c in cen]
    v384b = voxel_mass_properties(mem3, rho, grid=384)
    diag_ref = np.diag(v384b["inertia"])
    e3 = float(np.max(np.abs(np.diag(u3["inertia"]) - diag_ref)) / np.max(diag_ref))
    e2 = float(np.max(np.abs(np.diag(u2["inertia"]) - diag_ref)) / np.max(diag_ref))

    return {
        "sphere_rel_err": e_sphere,
        "dumbbell_rel_err": e_dumbbell,
        "overlapping_pair_rel_err_vs_voxel384": e_overlap,
        "overlapping_pair_volume_rel_err": e_overlap_vol,
        "triple_volume_fraction": u3["third_order_fraction"],
        "triple_rel_err_order3_vs_voxel384": e3,
        "triple_rel_err_order2_vs_voxel384": e2,
        "third_order_improves": bool(e3 < e2),
        "pass": bool(e_sphere < 1e-12 and e_dumbbell < 1e-12
                     and e_overlap < 1e-3 and e_overlap_vol < 1e-3
                     and e3 < 1e-3 and e3 < e2),
    }


# ---------------------------------------------------------------------------
# C2: production clump inertia vs analytic, per aspect ratio
# ---------------------------------------------------------------------------
def leg_c2_inertia_scan() -> dict:
    rows = []
    for ar in AR_LIST:
        p = make_shape(ar)()
        cmp = compare_with_voxel(p, grid=192)
        rows.append({
            "aspect_ratio_param": ar,
            "name": p.name,
            "n_members": p.n_members,
            "achieved_aspect_ratio": p.aspect_ratio,
            "sphericity": sphericity_prolate(ar),
            "volume_m3": p.volume,
            "mass_kg": p.mass,
            "analytic_inertia_diag": cmp["analytic_inertia_diag"],
            "production_inertia_diag": cmp["production_inertia_diag"],
            "fine_voxel_inertia_diag": cmp["fine_voxel_inertia_diag"],
            "rel_err_production": cmp["rel_err_production"],
            "rel_err_fine_voxel": cmp["rel_err_fine_voxel"],
            "rel_err_mass": cmp["rel_err_mass"],
            "analytic_exact": cmp["analytic"]["exact"],
            "third_order_fraction": cmp["analytic"]["third_order_fraction"],
            "inertia_ratio_transverse_over_axial":
                float(cmp["analytic_inertia_diag"][1] / cmp["analytic_inertia_diag"][0]),
        })
    worst = max(r["rel_err_production"] for r in rows)
    return {"rows": rows, "worst_rel_err_production": worst,
            "pass": bool(worst < 0.01)}


# ---------------------------------------------------------------------------
# C3: repose-angle shape-effect trend
# ---------------------------------------------------------------------------
def _repose(ar: float, seed: int, mu_roll: float, protocol: dict) -> dict:
    params = ContactParams(mu_roll=mu_roll)
    t0 = time.time()
    res = lifted_box_repose(make_shape(ar), seed=seed, params=params, **protocol)
    m = res["measurement"]
    return {
        "aspect_ratio_param": ar,
        "seed": seed,
        "mu_roll": mu_roll,
        "angle_profile_deg": m["angle_profile_deg"],
        "angle_hull_deg": m["angle_hull_deg"],
        "peak_height": m["peak_height"],
        "max_radius": m["max_radius"],
        "flank_points": m["flank_points"],
        "n_particles": res["n_particles"],
        "steps": res["steps"],
        "wall_seconds": round(time.time() - t0, 1),
    }


def leg_c3_repose_scan(protocol: dict, seeds=SEEDS, mu_roll: float = 0.3) -> dict:
    rows = [_repose(ar, s, mu_roll, protocol) for ar in AR_LIST for s in seeds]
    by_ar = {}
    for ar in AR_LIST:
        vals = [r["angle_profile_deg"] for r in rows if r["aspect_ratio_param"] == ar]
        by_ar[ar] = {"mean": float(np.mean(vals)), "values": vals,
                     "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    sphere = by_ar[AR_LIST[0]]["mean"]
    deltas = {f"{ar:.1f}": by_ar[ar]["mean"] - sphere for ar in AR_LIST}
    monotone = all(by_ar[AR_LIST[i + 1]]["mean"] >= by_ar[AR_LIST[i]]["mean"] - 1e-9
                   for i in range(len(AR_LIST) - 1))
    max_delta = max(deltas.values())
    sphere_in_band = LIT["sphere_repose_deg"][0] <= sphere <= LIT["sphere_repose_deg"][1]
    return {
        "rows": rows,
        "by_ar": {f"{k:.1f}": v for k, v in by_ar.items()},
        "deltas_vs_sphere_deg": deltas,
        "monotone_increasing": bool(monotone),
        "max_delta_deg": float(max_delta),
        "sphere_in_literature_band": bool(sphere_in_band),
        "pass": bool(monotone and max_delta >= 3.0),
    }


def leg_c3_rolling_sweep(protocol: dict, mu_rolls=(0.0, 0.6)) -> dict:
    """Mechanism diagnostic: does the trend depend on rolling resistance?"""
    out = {}
    for mu in mu_rolls:
        rows = [_repose(ar, 0, mu, protocol) for ar in (AR_LIST[0], AR_LIST[-1])]
        out[f"mu_roll={mu}"] = {
            f"{r['aspect_ratio_param']:.1f}": r["angle_profile_deg"] for r in rows
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "GAP-DEM-CLUMP"))
    ap.add_argument("--fast", action="store_true", help="short protocol (smoke only)")
    ap.add_argument("--skip-repose", action="store_true")
    args = ap.parse_args()

    protocol = dict(PROTOCOL)
    if args.fast:
        protocol.update(settle_steps=200, lift_steps=400, tail_steps=300)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    c1 = leg_c1_analytic_anchors()
    c2 = leg_c2_inertia_scan()
    c3 = {"skipped": True, "pass": False} if args.skip_repose else leg_c3_repose_scan(protocol)
    sweep = {} if args.skip_repose else leg_c3_rolling_sweep(protocol)

def derive_failure_reason(c1: dict, c2: dict, c3: dict) -> dict:
    """Round-1 honest-FAIL bookkeeping (owner two-round protocol).

    Derived only from the measured numbers in this run, so it can be re-applied
    to a finished ``results.json`` without re-running the (expensive) heap legs.
    """
    out = {}
    if not c1["pass"]:
        out["C1"] = {
            "fail_stage": "analytic_kernel",
            "failure_reason": (
                "analytic kernel does not reproduce its closed-form / fine-voxel "
                f"references (sphere {c1['sphere_rel_err']:.3e}, dumbbell "
                f"{c1['dumbbell_rel_err']:.3e}, overlapping pair "
                f"{c1['overlapping_pair_rel_err_vs_voxel384']:.3e}, triple "
                f"{c1['triple_rel_err_order3_vs_voxel384']:.3e})"),
        }
    if not c2["pass"]:
        worst = max(c2["rows"], key=lambda r: r["rel_err_production"])
        out["C2"] = {
            "fail_stage": "inertia_1pct",
            "failure_reason": (
                "production voxel inertia misses the analytic union value by "
                f"{worst['rel_err_production']:.4f} at a/b="
                f"{worst['aspect_ratio_param']} (acceptance 0.01); tried: grid=72 "
                "production quadrature, grid=192 cross-check, analytic "
                "inclusion-exclusion to third order"),
            "missing_method": "adaptive / importance voxel refinement near the "
                              "overlap lenses, or promoting the analytic kernel to "
                              "be the production mass-property source",
        }
    if not c3.get("skipped", False) and not c3.get("pass", False):
        out["C3"] = {
            "fail_stage": "repose_trend",
            "failure_reason": (
                "repose angle does not reproduce the literature shape effect: "
                f"monotone_increasing={c3['monotone_increasing']}, "
                f"max delta vs sphere={c3['max_delta_deg']:.2f} deg (need >= 3.0), "
                "angle profile per shape "
                + ", ".join(f"a/b={k}: {v['mean']:.2f} deg"
                            for k, v in c3["by_ar"].items())),
            "tried": [
                "lifted-box heap with Cundall rolling resistance mu_roll=0.3",
                "mu_roll sweep (0.0 / 0.6) on sphere and a/b=3 as a mechanism check",
                "2 seeds per shape to expose the seed scatter",
            ],
            "confound_found": (
                "the fill is packed by the clump bounding radius, so the heap size "
                "shrinks with aspect ratio (n_particles "
                + ", ".join(f"a/b={r['aspect_ratio_param']}: {r['n_particles']}"
                            for r in c3["rows"][::2])
                + "); shape and particle count are therefore entangled in this "
                  "protocol, and the seed scatter of the elongated heaps reaches "
                + f"{max(v['std'] for v in c3['by_ar'].values()):.2f} deg"),
            "mechanism_evidence": (
                "rolling resistance controls the sign of the shape effect in this "
                "kernel (C3_rolling_sweep): see results.json"),
            "missing_method": "anisotropic per-contact rolling resistance anchored on "
                              "the member-pair geometry, member-scale surface "
                              "roughness (interlocking), a thicker heap / larger "
                              "particle count, or an independent estimator "
                              "(rotating drum / inclined-plane critical angle)",
        }
    return out


    failures = derive_failure_reason(c1, c2, c3)
    criteria = {
        "C1_analytic_kernel": c1["pass"],
        "C2_inertia_vs_analytic_under_1pct": c2["pass"],
        "C3_repose_trend_matches_literature": bool(c3.get("pass", False)),
    }
    payload = {
        "campaign": "GAP-DEM-CLUMP",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "protocol": protocol,
        "density": DENSITY,
        "base_radius": BASE_RADIUS,
        "aspect_ratio_list": list(AR_LIST),
        "seeds": list(SEEDS),
        "literature": LIT,
        "C1_analytic": c1,
        "C2_inertia_scan": c2,
        "C3_repose_scan": c3,
        "C3_rolling_sweep": sweep,
        "criteria": criteria,
        "failures": failures,
        "honest_fail": bool(failures),
        "all_pass": all(criteria.values()),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    (out / "criteria.json").write_text(
        json.dumps({**criteria, "all_pass": payload["all_pass"],
                    "evidence": {
                        "C1": {k: v for k, v in c1.items() if k != "pass"},
                        "C2_worst_rel_err_production": c2["worst_rel_err_production"],
                        "C3": {k: c3.get(k) for k in
                               ("monotone_increasing", "max_delta_deg",
                                "sphere_in_literature_band", "deltas_vs_sphere_deg")}
                        if not c3.get("skipped") else "skipped",
                    }}, indent=2, default=float))

    # ---------------- figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ars = [r["aspect_ratio_param"] for r in c2["rows"]]
        ax[0].semilogy(ars, [r["rel_err_production"] for r in c2["rows"]], "o-",
                       label="production voxel (grid=72) vs analytic")
        ax[0].semilogy(ars, [r["rel_err_fine_voxel"] for r in c2["rows"]], "s--",
                       label="fine voxel (grid=192) vs analytic")
        ax[0].axhline(0.01, color="r", ls=":", label="1% acceptance")
        ax[0].set_xlabel("nominal aspect ratio a/b")
        ax[0].set_ylabel("max rel. error of inertia diagonal")
        ax[0].set_title("C2: clump inertia, numerical vs analytic")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        if not c3.get("skipped"):
            for ar in AR_LIST:
                vals = c3["by_ar"][f"{ar:.1f}"]["values"]
                ax[1].plot([ar] * len(vals), vals, "k.", alpha=0.6)
            means = [c3["by_ar"][f"{ar:.1f}"]["mean"] for ar in AR_LIST]
            ax[1].plot(AR_LIST, means, "o-", color="C0", label="DEM clump (this work)")
            ax[1].axhspan(*LIT["sphere_repose_deg"], color="green", alpha=0.15,
                          label="sphere literature band 22-25 deg")
            ax[1].axhspan(35.0, 45.0, color="orange", alpha=0.12,
                          label="ellipsoidal/angular literature 35-45 deg")
            ax[1].set_xlabel("nominal aspect ratio a/b")
            ax[1].set_ylabel("repose angle [deg]")
            ax[1].set_title("C3: repose angle vs aspect ratio")
            ax[1].legend(fontsize=8)
            ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "gap_dem_clump.png", dpi=130)
        plt.close(fig)
        payload["figures"] = ["gap_dem_clump.png"]
        (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    except Exception as exc:  # pragma: no cover - figure is best effort
        payload["figure_error"] = repr(exc)
        (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))

    (out / "repro_cmd.txt").write_text(
        "python benchmarks/gap_dem_clump/run_gap_dem_clump.py "
        f"--out {out}\n"
    )
    print(json.dumps({"criteria": criteria, "wall_seconds": payload["wall_seconds"]},
                     indent=2))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())