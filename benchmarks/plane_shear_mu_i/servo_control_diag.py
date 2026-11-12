"""CASE-DEM-PLANE-SHEAR-MUIDI round 4a: is the constant-pressure premise met?

Round 4 (m16633) diagnosed the mu(I)/phi(I) mismatch as "thin bed / boundary /
soft contact".  A cheaper and logically prior question was left unmeasured:
the reference laws are **constant-pressure** laws, but the in-repo servo cell
already reports ``pressure_control_error`` and those numbers were never judged.
On the round-3 baseline artefact the achieved normal stress falls to
0.72-0.84 of the 62.5 kPa target for I >= 0.03 -- i.e. the high-I points were
not measured at the reference condition at all.

This script quantifies that, then tests whether the deficit is a *control*
artefact (servo authority) or a *measurement* artefact (sampling window) or a
genuine material response:

  --stage trace      per-sample series of normal stress / mu / plate height for
                     a set of I, plus stationarity statistics of the sampling
                     window (mean, median, std, linear drift, fraction of
                     samples below 0.9 P_target).
  --stage authority  grid over the servo knobs (``servo_relax``,
                     ``servo_max_step_d``) at the worst points, to see whether
                     the target pressure can be held without touching the
                     contact physics (delta = servo authority only).
  --stage verdict    reads whichever sweep artefact is named and prints the
                     acceptance table (tracking error + mu/phi errors).

Nothing here changes the contact law, the material parameters or the reference
constants; the servo knobs are numerics only.

    PYTHONPATH=src python3 benchmarks/plane_shear_mu_i/servo_control_diag.py \
        --stage trace --points 0.01,0.1,0.3 --jobs 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_sweep as rs  # noqa: E402
from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402

OUT = HERE / "runs" / "servo_diag"
OUT.mkdir(parents=True, exist_ok=True)


def build_case(I: float, *, servo_relax: float | None = None,
               servo_max_step_d: float | None = None, seed: int | None = None,
               kn: float = rs.KN, settle_scale: float = 1.0,
               shear_strain: float | None = None) -> ServoShearCase:
    """Baseline device of run_sweep.run_point, with optional servo overrides."""
    kwargs = {}
    if servo_relax is not None:
        kwargs["servo_relax"] = float(servo_relax)
    if servo_max_step_d is not None:
        kwargs["servo_max_step_d"] = float(servo_max_step_d)
    scale = rs.stiffness_step_scale(kn) * settle_scale
    min_steps = (1 if shear_strain is not None
                 else int(round(rs.MIN_SHEAR_STEPS_BASE * scale)))
    return ServoShearCase(inertial_number=I, mu=rs.MU_CONTACT,
                          mu_rolling=rs.MU_ROLLING, restitution=rs.E, kn=kn,
                          normal_stress=rs.P,
                          shear_strain=rs.SHEAR_STRAIN if shear_strain is None
                          else float(shear_strain),
                          settle_steps=int(round(rs.SETTLE_STEPS_BASE * scale)),
                          min_shear_steps=min_steps,
                          packing_gap=0.02, jitter=0.01,
                          seed=rs.SEED if seed is None else int(seed),
                          radius=rs.R, density=rs.RHO, box_width=rs.W,
                          box_depth=rs.D, fill_height=rs.H, **kwargs)


def window_stats(series: list[dict], target_p: float) -> dict:
    """Stationarity of the tail window the solver averages over."""
    n_tail = max(1, int(0.6 * len(series)))
    tail = series[-n_tail:]
    p = [s["normal_stress_pa"] for s in tail]
    mu = [s["mu"] for s in tail]
    t = [s["strain"] for s in tail]
    mean_p = float(sum(p) / len(p))
    mean_t = float(sum(t) / len(t))
    var_t = sum((x - mean_t) ** 2 for x in t)
    slope = (sum((t[i] - mean_t) * (p[i] - mean_p) for i in range(len(p)))
             / var_t) if var_t > 0 else 0.0
    var_p = sum((x - mean_p) ** 2 for x in p) / len(p)
    return {
        "n_samples": len(series),
        "n_tail": n_tail,
        "strain_window": [tail[0]["strain"], tail[-1]["strain"]],
        "p_mean": mean_p,
        "p_median": float(sorted(p)[len(p) // 2]),
        "p_std": math.sqrt(var_p),
        "p_cv": math.sqrt(var_p) / mean_p if mean_p else float("nan"),
        "p_drift_per_strain": slope,
        "p_rel_drift_over_window": (slope * (tail[-1]["strain"] - tail[0]["strain"]) / mean_p)
        if mean_p else float("nan"),
        "frac_below_0.9p_target": sum(1 for x in p if x < 0.9 * target_p) / len(p),
        "mu_mean": float(sum(mu) / len(mu)),
        "plate_z_mean": float(sum(s["plate_z"] for s in tail) / len(tail)),
    }


def _trace(args):
    I, relax, clip = args[:3]
    seed = args[3] if len(args) > 3 else None
    strain = args[4] if len(args) > 4 else None
    case = build_case(I, servo_relax=relax, servo_max_step_d=clip, seed=seed,
                      shear_strain=strain)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case, return_series=True)
    wc = time.perf_counter() - t0
    series = res["series"]
    stats = window_stats(series, rs.P)
    return {
        "kind": "trace",
        "I_target": I,
        "seed": case.seed,
        "shear_strain": case.shear_strain,
        "I_measured": res["inertial_number"],
        "servo_relax": case.servo_relax,
        "servo_max_step_d": case.servo_max_step_d,
        "p_target": rs.P,
        "p_achieved": res["normal_stress_measured_pa"],
        "pressure_control_error": res["normal_stress_measured_pa"] / rs.P - 1.0,
        "mu": res["mu_effective"],
        "ref_mu": rs.ref_mu(res["inertial_number"]),
        # sweep-artefact compatible aliases so --stage verdict can read either file
        "normal_stress_pa": res["normal_stress_measured_pa"],
        "normal_stress_applied_pa": rs.P,
        "mu_measured": res["mu_effective"],
        "phi_bed_steady": res["phi_bed_steady"],
        "ref_phi": rs.ref_phi(res["inertial_number"]),
        "coordination": res["coordination_number"],
        "dilatancy": res["dilatancy"],
        "shear_steps": res["shear_steps"],
        "strain_achieved": res["series"][-1]["strain"] if res.get("series") else None,
        "settle_steps": res["settle_steps"],
        "wall_clock_s": wc,
        "window": stats,
        "series": {k: [s[k] for s in series]
                   for k in ("strain", "normal_stress_pa", "shear_stress_pa",
                             "mu", "plate_z")},
    }


def stage_trace(points, jobs, relax, clip):
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        recs = list(pool.map(_trace, [(I, relax, clip) for I in points]))
    recs.sort(key=lambda r: r["I_target"])
    for r in recs:
        w = r["window"]
        print(f"I={r['I_target']:<6g} P_err={r['pressure_control_error'] * 100:+7.2f}% "
              f"(cv={w['p_cv'] * 100:5.2f}%, drift={w['p_rel_drift_over_window'] * 100:+6.2f}%, "
              f"below90={w['frac_below_0.9p_target'] * 100:5.1f}%) "
              f"mu={r['mu']:.4f} phi_s={r['phi_bed_steady']:.4f} "
              f"[{r['wall_clock_s']:.0f} s]", flush=True)
    path = OUT / f"trace_relax{relax:g}_clip{clip:g}.json"
    path.write_text(json.dumps({"case": "CASE-DEM-PLANE-SHEAR-MUIDI",
                                "stage": "trace", "rows": recs}, indent=1))
    print("WROTE", path, flush=True)
    return recs


def stage_authority(points, jobs):
    grid = [(relax, clip) for relax in (0.2, 0.5, 1.0) for clip in (0.002, 0.01, 0.05)]
    tasks = [(I, relax, clip) for I in points for relax, clip in grid]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        recs = list(pool.map(_trace, tasks))
    recs.sort(key=lambda r: (r["I_target"], r["servo_relax"], r["servo_max_step_d"]))
    print(f"{'I':>6} {'relax':>6} {'clip/d':>7} {'P_err%':>8} {'mu':>7} {'ref_mu':>7} "
          f"{'phi_s':>7} {'ref_phi':>7} {'dilat':>8}")
    for r in recs:
        print(f"{r['I_target']:>6g} {r['servo_relax']:>6g} {r['servo_max_step_d']:>7g} "
              f"{r['pressure_control_error'] * 100:>+8.2f} {r['mu']:>7.4f} {r['ref_mu']:>7.4f} "
              f"{r['phi_bed_steady']:>7.4f} {r['ref_phi']:>7.4f} {r['dilatancy']:>+8.5f}",
              flush=True)
    path = OUT / "authority_grid.json"
    path.write_text(json.dumps({"case": "CASE-DEM-PLANE-SHEAR-MUIDI",
                                "stage": "authority", "rows": recs}, indent=1))
    print("WROTE", path, flush=True)
    return recs


def stage_ensemble(points, jobs, relax, clip, seeds):
    """Seed ensemble: run-to-run scatter of the judged quantities.

    A <10 % acceptance band cannot be adjudicated from a single seed if the
    seed-to-seed scatter is of the same order.
    """
    tasks = [(I, relax, clip, seed) for I in points for seed in seeds]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        recs = list(pool.map(_trace, tasks))
    recs.sort(key=lambda r: (r["I_target"], r["seed"]))
    table = []
    for I in points:
        sub = [r for r in recs if r["I_target"] == I]
        mu = [r["mu"] for r in sub]
        pe = [r["pressure_control_error"] for r in sub]
        ph = [r["phi_bed_steady"] for r in sub]
        mu_m = sum(mu) / len(mu)
        mu_sd = math.sqrt(sum((x - mu_m) ** 2 for x in mu) / max(1, len(mu) - 1))
        ph_m = sum(ph) / len(ph)
        ph_sd = math.sqrt(sum((x - ph_m) ** 2 for x in ph) / max(1, len(ph) - 1))
        ref_mu = sub[0]["ref_mu"]
        ref_phi = sub[0]["ref_phi"]
        table.append({
            "I_target": I, "n_seeds": len(sub), "seeds": [r["seed"] for r in sub],
            "mu_mean": mu_m, "mu_sd": mu_sd, "ref_mu": ref_mu,
            "mu_bias_pct": (mu_m / ref_mu - 1.0) * 100.0,
            "mu_scatter_pct": (mu_sd / mu_m) * 100.0 if mu_m else float("nan"),
            "phi_mean": ph_m, "phi_sd": ph_sd, "ref_phi": ref_phi,
            "phi_bias_pct": (ph_m / ref_phi - 1.0) * 100.0,
            "phi_scatter_pct": (ph_sd / ph_m) * 100.0 if ph_m else float("nan"),
            "p_err_mean_pct": 100.0 * sum(pe) / len(pe),
            "p_err_sd_pct": 100.0 * math.sqrt(sum((x - sum(pe) / len(pe)) ** 2 for x in pe)
                                              / max(1, len(pe) - 1)),
        })
    print(f"{'I':>6} {'n':>3} {'mu_mean':>8} {'mu_sd':>7} {'bias%':>7} {'scat%':>6} "
          f"{'phi_mean':>9} {'phi_sd':>7} {'bias%':>7} {'P_err%':>8}")
    for t in table:
        print(f"{t['I_target']:>6g} {t['n_seeds']:>3} {t['mu_mean']:>8.4f} {t['mu_sd']:>7.4f} "
              f"{t['mu_bias_pct']:>+7.2f} {t['mu_scatter_pct']:>6.2f} "
              f"{t['phi_mean']:>9.4f} {t['phi_sd']:>7.4f} {t['phi_bias_pct']:>+7.2f} "
              f"{t['p_err_mean_pct']:>+8.2f}", flush=True)
    path = OUT / f"ensemble_relax{relax:g}_clip{clip:g}.json"
    path.write_text(json.dumps({"case": "CASE-DEM-PLANE-SHEAR-MUIDI",
                                "stage": "ensemble", "servo_relax": relax,
                                "servo_max_step_d": clip, "summary": table,
                                "rows": recs}, indent=1))
    print("WROTE", path, flush=True)
    return table


def stage_strain(points, jobs, relax, clip, strains):
    """Is the reported average a steady state?  Strain (duration) ladder.

    The tail window is a fixed *fraction* of the shear phase, so a transient
    that has not decayed shows up as a systematic drift of the normal stress
    across the window; growing the total strain is the direct test.
    """
    tasks = [(I, relax, clip, None, strain) for I in points for strain in strains]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        recs = list(pool.map(_trace, tasks))
    recs.sort(key=lambda r: (r["I_target"], r["shear_strain"]))
    print(f"{'I':>6} {'strain':>7} {'steps':>7} {'P_err%':>8} {'cv%':>6} {'drift%':>8} "
          f"{'mu':>7} {'ref_mu':>7} {'phi_s':>7} {'ref_phi':>7} {'t(s)':>6}")
    for r in recs:
        w = r["window"]
        print(f"{r['I_target']:>6g} {r['shear_strain']:>7.2f} {r['shear_steps']:>7d} "
              f"{r['pressure_control_error'] * 100:>+8.2f} {w['p_cv'] * 100:>6.2f} "
              f"{w['p_rel_drift_over_window'] * 100:>+8.2f} {r['mu']:>7.4f} "
              f"{r['ref_mu']:>7.4f} {r['phi_bed_steady']:>7.4f} {r['ref_phi']:>7.4f} "
              f"{r['wall_clock_s']:>6.0f}", flush=True)
    path = OUT / "strain_ladder.json"
    path.write_text(json.dumps({"case": "CASE-DEM-PLANE-SHEAR-MUIDI",
                                "stage": "strain", "servo_relax": relax,
                                "servo_max_step_d": clip, "rows": recs}, indent=1))
    print("WROTE", path, flush=True)
    return recs


def stage_verdict(path, band_pct):
    """Premise gate over an existing sweep artefact (no new runs).

    Judges three things the earlier rounds left unread:
      * pressure control: |P_achieved/P_target - 1| <= band
      * steady window:    reported per point by ``window`` stats when available
      * mu/phi error:     |measured/reference - 1| <= band
    """
    payload = json.loads(Path(path).read_text())
    rows = payload["rows"]
    print(f"# premise gate for {path} (band {band_pct:.0f} %)")
    print(f"{'I_meas':>9} {'P_err%':>8} {'P_gate':>7} {'mu_err%':>8} {'mu_gate':>8} "
          f"{'phi_err%':>8} {'phi_gate':>9}")
    fails = {"pressure": 0, "mu": 0, "phi": 0}
    for r in rows:
        p_err = 100.0 * (r["normal_stress_pa"] / r.get("normal_stress_applied_pa", 62500.0) - 1.0)
        mu_err = 100.0 * (r["mu_measured"] / r["ref_mu"] - 1.0)
        if "phi_bed_steady" in r:
            phi_err = 100.0 * (r["phi_bed_steady"] / r["ref_phi"] - 1.0)
        else:
            phi_err = 100.0 * (r["phi_bed"] / r["ref_phi"] - 1.0)
        p_ok = abs(p_err) <= band_pct
        mu_ok = abs(mu_err) <= band_pct
        phi_ok = abs(phi_err) <= band_pct
        fails["pressure"] += (not p_ok)
        fails["mu"] += (not mu_ok)
        fails["phi"] += (not phi_ok)
        print(f"{r['I_measured']:>9.4g} {p_err:>+8.2f} {'PASS' if p_ok else 'FAIL':>7} "
              f"{mu_err:>+8.2f} {'PASS' if mu_ok else 'FAIL':>8} {phi_err:>+8.2f} "
              f"{'PASS' if phi_ok else 'FAIL':>9}")
    print(f"# fails: {fails} of {len(rows)} points")
    out = {k: {"fails": v, "points": len(rows)} for k, v in fails.items()}
    (OUT / "premise_gate.json").write_text(json.dumps(out, indent=1))
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("trace", "authority", "ensemble", "strain", "verdict"),
                    default="trace")
    ap.add_argument("--points", default="0.01,0.1,0.3")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--servo-relax", type=float, default=None)
    ap.add_argument("--servo-clip", type=float, default=None)
    ap.add_argument("--seeds", default="5279,5279,0", help="comma list of seeds")
    ap.add_argument("--sweep", default=str(HERE / "runs" / "sweep_baseline.json"))
    ap.add_argument("--band", type=float, default=10.0)
    ap.add_argument("--strains", default="0.2,0.6,1.5")
    a = ap.parse_args()
    points = [float(x) for x in a.points.split(",") if x.strip()]
    if a.stage == "trace":
        stage_trace(points, a.jobs, a.servo_relax if a.servo_relax is not None else 0.2,
                    a.servo_clip if a.servo_clip is not None else 0.002)
    elif a.stage == "authority":
        stage_authority(points, a.jobs)
    elif a.stage == "strain":
        strains = [float(x) for x in a.strains.split(",") if x.strip()]
        relax = a.servo_relax if a.servo_relax is not None else 0.5
        clip = a.servo_clip if a.servo_clip is not None else 0.05
        stage_strain(points, a.jobs, relax, clip, strains)
    elif a.stage == "ensemble":
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        # first entry is doubled so the list can be a seed + extra-variation list
        relax = a.servo_relax if a.servo_relax is not None else 0.5
        clip = a.servo_clip if a.servo_clip is not None else 0.05
        stage_ensemble(points, a.jobs, relax, clip, seeds)
    else:
        stage_verdict(a.sweep, a.band)


if __name__ == "__main__":
    main()