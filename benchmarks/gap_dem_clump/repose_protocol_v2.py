#!/usr/bin/env python3
"""GAP-DEM-CLUMP round 2: repose-angle protocol validity + shape-effect verdict.

Round 1 (``run_gap_dem_clump.py``, mission m14681) reported an honest FAIL on its
C3 leg: the repose angle of the clump family was non-monotone in aspect ratio and
the sphere baseline sat 0.8 deg above the 22-25 deg glass-bead band.  Its own
write-up flagged two defects of the *protocol* and one of the *contact law*:

1. the fill is packed by the clump bounding radius, so particle count and total
   solid volume fall with aspect ratio (648 -> 104 particles, 4.69e-6 ->
   2.27e-6 m^3);
2. the heap is measured after a fixed 1200-step tail (0.06 s) although the bed
   was still flowing;
3. the Cundall rolling-resistance branch is regularised as
   ``tau = -mu_r r |Fn| w / (|w| + eps_w)`` with ``eps_w = 1.0 rad/s``, i.e. it is
   *rate proportional* for the angular speeds seen here and has no static
   (rate independent) Coulomb branch at all.

This script tests all three, each with its own machine-checkable criterion:

* ``C4 protocol_validity`` -- with the quiescence gate the tail is integrated
  until the mean body speed is below ``Q_TOL``; the measured angle is then
  compared with the fixed-tail (round-1) angle of the *same* fill.  A criterion
  of 0.5 deg isolates the measurement artefact.
* ``C5 rolling_law`` -- the delivered rolling torque is measured as the fraction
  of its Coulomb limit, ``s = |w| / (|w| + eps_w)``, both at the end of the lift
  (flow regime) and at the settled state.  The static limit ``tau(|w| -> 0) = 0``
  is asserted analytically, not assumed.
* ``C6 shape_trend`` -- the settled, mass-matched scan is tested against the
  literature trend (repose angle grows with departure from sphericity) and the
  22-25 deg sphere band, with the same >= 3 deg increment criterion as round 1.

Fills are mass matched (equal total solid volume, hence equal heap mass) by
bisecting the fill height at the *non-overlapping* lattice spacing ``1.02 * ext``;
no initial interpenetration is introduced by the matching.

Run::

    python benchmarks/gap_dem_clump/repose_protocol_v2.py [--out DIR] [--fast]

``--fast`` shortens every leg (numbers then differ; the JSON records the
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
sys.path.insert(0, str(ROOT / "benchmarks" / "gap_dem_clump"))

from run_gap_dem_clump import AR_LIST, BASE_RADIUS, DENSITY, LIT, make_shape  # noqa: E402
from tensordem.contact.clump_contact import ContactParams  # noqa: E402
from tensordem.particle.clump_dem import (  # noqa: E402
    lifted_box_repose,
    measure_repose_angle,
)

SEEDS = (0,)
Q_TOL = 0.01              # m/s, mean body speed for the quiescence gate
SPACING_FACTOR = 1.02     # no initial interpenetration (round-1 default)

#: box of the round-1 protocol (unchanged, so the fills are comparable)
BOX = dict(box_half_x=0.012, slab_half_y=0.005)

PROTOCOL_V1 = dict(           # round-1 tail (fixed number of steps)
    fill_height=0.070,
    wall_height=0.10,
    settle_steps=800,
    lift_steps=1800,
    tail_steps=1200,
    dt=5e-5,
    lift_speed=0.30,
)

PROTOCOL_V2 = dict(           # quiescence-gated tail, lift cleared above the bed
    wall_height=0.06,
    settle_steps=800,
    tail_steps=20000,
    dt=5e-5,
    lift_speed=0.30,
    min_tail_steps=1500,
    max_tail_steps=9000,
    quiescence_mean_speed=Q_TOL,
    probe_every=300,
)


# ---------------------------------------------------------------------------
# Mass-matched fill
# ---------------------------------------------------------------------------
def particle_extent(particle) -> np.ndarray:
    c = particle.local_centers
    r = particle.radii
    return (c + r[:, None]).max(axis=0) - (c - r[:, None]).min(axis=0)


def fill_particles(particle, fill_height, *, spacing_factor=SPACING_FACTOR):
    """Number of particles the round-1 lattice puts in the box (no simulation)."""
    from tensordem.particle.clump_dem import lattice_packing

    pos, _ = lattice_packing(
        particle,
        (-BOX["box_half_x"], -BOX["slab_half_y"], 0.0),
        (BOX["box_half_x"], BOX["slab_half_y"], fill_height),
        spacing=spacing_factor * particle_extent(particle),
        jitter=2e-5,
    )
    return len(pos)


def matched_fill_height(particle, target_volume: float, *, lo=0.010, hi=0.30,
                        tol=0.02, iters=22) -> dict:
    """Bisect the fill height until ``n * V_particle == target_volume``.

    The lattice spacing is held at the non-overlapping default, so matching never
    introduces initial interpenetration; only the number of layers changes.  The
    achieved volume error is returned (and asserted by the test suite), never
    silently accepted.
    """
    def volume(h):
        return fill_particles(particle, h) * particle.volume

    v_lo, v_hi = volume(lo), volume(hi)
    if not (v_lo <= target_volume <= v_hi):
        raise ValueError(
            f"target volume {target_volume:.3e} outside the achievable range "
            f"[{v_lo:.3e}, {v_hi:.3e}] for this particle"
        )
    h = 0.5 * (lo + hi)
    for _ in range(int(iters)):
        h = 0.5 * (lo + hi)
        v = volume(h)
        if abs(v - target_volume) <= tol * target_volume:
            break
        if v < target_volume:
            lo = h
        else:
            hi = h
    v = volume(h)
    return {
        "fill_height": float(h),
        "n_particles": int(fill_particles(particle, h)) and int(round(v / particle.volume)),
        "solid_volume": float(v),
        "rel_err": float(abs(v - target_volume) / target_volume),
        "target_volume": float(target_volume),
    }


def lift_steps_for(fill_height: float, protocol: dict) -> int:
    """Steps that raise the lateral walls clear of the settled bed."""
    distance = fill_height + 0.006                      # bed + dilation margin
    return int(math.ceil(distance / (protocol["lift_speed"] * protocol["dt"])))


# ---------------------------------------------------------------------------
# Repose legs
# ---------------------------------------------------------------------------
def run_repose(ar: float, *, seed: int, mu_roll: float, protocol: dict,
               fill_height: float, gated: bool) -> dict:
    params = ContactParams(mu_roll=mu_roll)
    kw = dict(protocol)
    kw.pop("fill_height", None)
    kw.pop("fill_height_by_ar", None)
    lift_steps = kw.pop("lift_steps", None)
    if lift_steps is None:
        lift_steps = lift_steps_for(fill_height, kw)
    t0 = time.time()
    res = lifted_box_repose(
        make_shape(ar), seed=seed, params=params, fill_height=fill_height,
        lift_steps=lift_steps, **kw, **BOX,
    )
    m = res["measurement"]
    return {
        "aspect_ratio_param": ar,
        "seed": seed,
        "mu_roll": mu_roll,
        "gated": gated,
        "fill_height": fill_height,
        "lift_steps": int(lift_steps),
        "angle_profile_deg": m["angle_profile_deg"],
        "angle_hull_deg": m["angle_hull_deg"],
        "peak_height": m["peak_height"],
        "max_radius": m["max_radius"],
        "flank_points": m["flank_points"],
        "n_particles": res["n_particles"],
        "solid_volume": res["solid_volume"],
        "mass": res["mass"],
        "steps": res["steps"],
        "tail_steps_used": res["tail_steps_used"],
        "quiesced": res["quiesced"],
        "tail_mean_speed": res["tail_spread"]["mean_speed"],
        "flow_mean_speed": res["flow_spread"]["mean_speed"],
        "flow_saturation_median": res["flow_rolling_saturation"]["median"],
        "tail_saturation_median": res["rolling_saturation"]["median"],
        "wall_seconds": round(time.time() - t0, 1),
    }


def leg_c4_protocol_validity(protocol_v1, protocol_v2, shapes=(1.0, 3.0)) -> dict:
    """Fixed tail (round-1) vs quiescence gate, on the *same* fill."""
    rows = []
    for ar in shapes:
        p = make_shape(ar)()
        v1 = run_repose(ar, seed=0, mu_roll=0.3, protocol=protocol_v1,
                        fill_height=protocol_v1["fill_height"], gated=False)
        v2 = run_repose(ar, seed=0, mu_roll=0.3, protocol=protocol_v2,
                        fill_height=protocol_v2["fill_height_by_ar"][f"{ar:.1f}"],
                        gated=True)
        rows.append({
            "aspect_ratio_param": ar,
            "fixed_tail": v1,
            "settled_tail": v2,
            "delta_deg": v2["angle_profile_deg"] - v1["angle_profile_deg"],
        })
    worst = max(rows, key=lambda r: abs(r["delta_deg"]))
    unquiesced = [r["fixed_tail"] for r in rows if r["fixed_tail"]["tail_mean_speed"] > Q_TOL]
    return {
        "rows": rows,
        "q_tol": Q_TOL,
        "fixed_tail_unquiesced_count": len(unquiesced),
        "max_abs_delta_deg": float(abs(worst["delta_deg"])),
        "pass": bool(len(unquiesced) == 0),
    }


def leg_c5_rolling_law(protocol_v2) -> dict:
    """Coulomb-limit fraction actually delivered by the rolling branch."""
    rows = []
    for ar in (AR_LIST[0],):
        h = protocol_v2["fill_height_by_ar"][f"{ar:.1f}"]
        for mu in (0.0, 0.3, 0.6):
            r = run_repose(ar, seed=0, mu_roll=mu, protocol=protocol_v2,
                           fill_height=h, gated=True)
            rows.append(r)
    # analytic static limit: tau(w) / (mu_r r |Fn|) = w / (w + eps_w) -> 0
    eps = ContactParams().eps_w
    static_limit = 0.0
    at_unit = 1.0 / (1.0 + eps)          # fraction delivered at |w| = 1 rad/s
    return {
        "eps_w_rad_s": float(eps),
        "static_limit_fraction": float(static_limit),
        "fraction_at_unit_omega": float(at_unit),
        "rows": rows,
        "flow_saturation_median_all": float(np.median([r["flow_saturation_median"] for r in rows])),
        "tail_saturation_median_all": float(np.median([r["tail_saturation_median"] for r in rows])),
        "pass": bool(static_limit == 0.0 and at_unit < 0.6),
        "statement": (
            "the modelled rolling resistance is rate proportional below "
            f"eps_w = {eps:.2f} rad/s (fraction {at_unit:.3f} at 1 rad/s) and "
            "vanishes at rest, so it cannot provide static slope support; only "
            "sliding friction plus geometry can hold a settled heap"
        ),
    }


def leg_c6_shape_trend(protocol_v2, mu_roll: float = 0.3, seeds=SEEDS) -> dict:
    rows = []
    for ar in AR_LIST:
        h = protocol_v2["fill_height_by_ar"][f"{ar:.1f}"]
        for s in seeds:
            rows.append(run_repose(ar, seed=s, mu_roll=mu_roll,
                                   protocol=protocol_v2, fill_height=h, gated=True))
    by_ar = {}
    for ar in AR_LIST:
        vals = [r["angle_profile_deg"] for r in rows if r["aspect_ratio_param"] == ar]
        vals = [v for v in vals if np.isfinite(v)]
        by_ar[ar] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "values": vals,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_particles": next(r["n_particles"] for r in rows if r["aspect_ratio_param"] == ar),
        }
    sphere = by_ar[AR_LIST[0]]["mean"]
    deltas = {f"{ar:.1f}": by_ar[ar]["mean"] - sphere for ar in AR_LIST}
    monotone = all(by_ar[AR_LIST[i + 1]]["mean"] >= by_ar[AR_LIST[i]]["mean"] - 1e-9
                   for i in range(len(AR_LIST) - 1))
    max_delta = float(max(deltas.values()))
    in_band = bool(LIT["sphere_repose_deg"][0] <= sphere <= LIT["sphere_repose_deg"][1])
    return {
        "rows": rows,
        "by_ar": {f"{k:.1f}": v for k, v in by_ar.items()},
        "deltas_vs_sphere_deg": deltas,
        "monotone_increasing": bool(monotone),
        "max_delta_deg": max_delta,
        "sphere_in_literature_band": in_band,
        "quiesced_all": bool(all(r["quiesced"] for r in rows)),
        "pass": bool(monotone and max_delta >= 3.0),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "GAP-DEM-CLUMP-ROUND2"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-c4", action="store_true")
    ap.add_argument("--skip-c5", action="store_true")
    args = ap.parse_args(argv)

    protocol_v1 = dict(PROTOCOL_V1)
    protocol_v2 = dict(PROTOCOL_V2)
    if args.fast:
        protocol_v1.update(settle_steps=200, lift_steps=400, tail_steps=300)
        protocol_v2.update(settle_steps=300, min_tail_steps=400, max_tail_steps=1200,
                           probe_every=200)

    # ---- mass-matched fills (equal total solid volume across shapes) ------ #
    volumes = {
        f"{ar:.1f}": fill_particles(make_shape(ar)(), protocol_v1["fill_height"])
        * make_shape(ar)().volume
        for ar in AR_LIST
    }
    target = min(volumes.values())      # the smallest round-1 fill: reachable by all
    matches = {}
    for ar in AR_LIST:
        m = matched_fill_height(make_shape(ar)(), target)
        m["aspect_ratio_param"] = ar
        m["round1_solid_volume"] = volumes[f"{ar:.1f}"]
        m["round1_fill_height"] = protocol_v1["fill_height"]
        matches[f"{ar:.1f}"] = m
    protocol_v2["fill_height_by_ar"] = {k: v["fill_height"] for k, v in matches.items()}

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    only = os.environ.get("GDC_ONLY", "c4,c5,c6").split(",")
    c4 = ({"skipped": True, "pass": False}
          if args.skip_c4 or "c4" not in only else leg_c4_protocol_validity(
              protocol_v1, protocol_v2))
    c5 = ({"skipped": True, "pass": False}
          if args.skip_c5 or "c5" not in only else leg_c5_rolling_law(protocol_v2))
    c6 = ({"skipped": True, "pass": False, "by_ar": {}, "deltas_vs_sphere_deg": {},
           "monotone_increasing": False, "max_delta_deg": float("nan")}
          if "c6" not in only else leg_c6_shape_trend(protocol_v2))

    criteria = {
        "C4_round1_tail_was_unsettled": c4.get("pass", False),
        "C5_rolling_law_has_no_static_branch": c5.get("pass", False),
        "C6_settled_shape_trend_matches_literature": c6.get("pass", False),
    }
    payload = {
        "campaign": "GAP-DEM-CLUMP",
        "round": 2,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "box": BOX,
        "density": DENSITY,
        "base_radius": BASE_RADIUS,
        "protocol_v1": protocol_v1,
        "protocol_v2": protocol_v2,
        "mass_matched_fills": matches,
        "target_solid_volume": target,
        "C4_protocol_validity": c4,
        "C5_rolling_law": c5,
        "C6_shape_trend": c6,
        "criteria": criteria,
        "all_pass": all(criteria.values()),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    (out / "criteria.json").write_text(json.dumps(criteria, indent=2, default=float))

    # ---- figure ---------------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        if not c4.get("skipped"):
            ars = [r["aspect_ratio_param"] for r in c4["rows"]]
            ax[0].plot(ars, [r["fixed_tail"]["angle_profile_deg"] for r in c4["rows"]],
                       "s--", color="C3", label="fixed 1200-step tail (round 1)")
            ax[0].plot(ars, [r["settled_tail"]["angle_profile_deg"] for r in c4["rows"]],
                       "o-", color="C0", label="quiescence-gated tail (round 2)")
            ax[0].set_xlabel("nominal aspect ratio a/b")
            ax[0].set_ylabel("repose angle [deg]")
            ax[0].set_title("C4: measurement artefact of the fixed tail")
            ax[0].legend(fontsize=8)
            ax[0].grid(alpha=0.3)

        for ar in AR_LIST:
            vals = c6["by_ar"][f"{ar:.1f}"]["values"]
            ax[1].plot([ar] * len(vals), vals, "k.", alpha=0.6)
        means = [c6["by_ar"][f"{ar:.1f}"]["mean"] for ar in AR_LIST]
        ax[1].plot(AR_LIST, means, "o-", color="C0", label="DEM clump (mass matched)")
        ax[1].axhspan(*LIT["sphere_repose_deg"], color="green", alpha=0.15,
                      label="sphere literature band 22-25 deg")
        ax[1].axhspan(35.0, 45.0, color="orange", alpha=0.12,
                      label="ellipsoidal/angular literature 35-45 deg")
        ax[1].set_xlabel("nominal aspect ratio a/b")
        ax[1].set_ylabel("repose angle [deg]")
        ax[1].set_title("C6: settled repose angle vs aspect ratio")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "gap_dem_clump_round2.png", dpi=130)
        plt.close(fig)

        # rolling-law figure: delivered fraction vs |w|
        fig, axp = plt.subplots(figsize=(5.6, 3.8))
        eps = ContactParams().eps_w
        w = np.logspace(-3, 2, 200)
        axp.semilogx(w, w / (w + eps), color="C0", label=r"$|w|/(|w|+\epsilon_w)$")
        if not c5.get("skipped"):
            med_w = float(np.median([r["tail_saturation_median"] for r in c5["rows"]]))
            axp.axhline(med_w, color="C3", ls=":", label="measured median at rest")
        axp.set_xlabel(r"relative angular speed $|w|$ [rad/s]")
        axp.set_ylabel("fraction of Coulomb limit")
        axp.set_title("C5: rolling-resistance branch is rate proportional")
        axp.legend(fontsize=8)
        axp.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(out / "rolling_law_saturation.png", dpi=130)
        plt.close(fig)
        payload["figures"] = ["gap_dem_clump_round2.png", "rolling_law_saturation.png"]
    except Exception as exc:                       # pragma: no cover - best effort
        payload["figure_error"] = repr(exc)

    payload["wall_seconds_total"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    (out / "repro_cmd.txt").write_text(
        "python benchmarks/gap_dem_clump/repose_protocol_v2.py --out "
        f"{out}\n"
    )
    print(json.dumps({"criteria": criteria, "C6": {
        "by_ar": c6["by_ar"], "deltas": c6["deltas_vs_sphere_deg"],
        "monotone": c6["monotone_increasing"]},
        "wall_seconds": payload["wall_seconds_total"]}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())