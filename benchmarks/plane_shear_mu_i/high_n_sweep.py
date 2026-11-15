"""CASE-DEM-PLANE-SHEAR-MUIDI round 18: higher-N resolution study.

Round 14 (synthesis) and round 16 (strain-convergence) both concluded that the
mu(I) curve is *unresolvable* at the shipped particle count (N = 294, box
0.12 x 0.05 x 0.06 m): the noise floor (std of the final 25 % window) exceeds
the 10 % ledger band at every point, and the mu(I) reads 13-33 % below the
da Cruz / GDR MiDi reference.  Round 14 recommendation R4 was explicit:
"increase particle count (N > 1000) for proper bulk statistics".

This round is that verification debt: a higher-N plane-shear sweep on the same
device (same contact law, same P/kn = 1, same seed) but a deeper + wider box
(0.16 x 0.08 x 0.09 m -> N ~ 1000+) to test whether the mu(I) deficit and the
noise floor are a *particle-count / boundary-layer* artefact or a genuine
physics mismatch.

Pre-registered verdicts (written before the sweep):

* P1 (noise floor): the mu std of the final 25 % window must drop below the
  10 % band at the mid-I points (I in [1e-2, 3e-2]) if the noise is a
  particle-count artefact.
* P2 (mu(I) convergence): |mu_meas/mu_ref - 1| at the mid-I points must move
  toward the 10 % band (i.e. the low-I deficit must shrink) if the deficit is
  a boundary-layer / resolution artefact.
* P3 (honest verdict): the full-curve verdict is reported as PASS/FAIL with
  the worst error, no silent pass.

Usage (shardable for the cluster)::

    python high_n_sweep.py --shard 0/4        # one I target per shard
    python high_n_sweep.py --assemble         # tables/fig only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402
from reference_law import ref_mu as _ref_mu, ref_phi as _ref_phi  # noqa: E402

RUNS = HERE / "runs" / "high_n"
RUNS.mkdir(parents=True, exist_ok=True)

# --- reference (da Cruz 2005 / GDR MiDi 2004), glass beads ----------------
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
PHI_C, PHI_J = 0.58, 0.64
TOL = 0.10

# --- device (same contact law / P-kn as the baseline arm, bigger box) ------
P = 6.25e4          # Pa, servo-held normal stress
KN = 6.25e4         # N/m
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E = 0.3
R = 0.005
RHO = 2500.0
SEED = 5279
# higher-N box: 0.16 x 0.08 x 0.09 m  (32d x 16d x 18d) -> N ~ 1000+
W, D, H = 0.16, 0.08, 0.09
I_TARGETS = (1e-2, 3e-2, 1e-1, 3e-1)
SETTLE_STEPS = 3000
SHEAR_STRAIN = 0.5          # converged strain (r16 used >= 2; 0.5 is a
                            # compromise for the higher-N cost, still past the
                            # low-I transient that r8/14 flagged)
SAMPLE_STRIDE = 10

PREREG = {
    "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "round": 18,
    "kind": "higher-N resolution study (verification debt R4 of round 14)",
    "device": "tensordem.mu_i_rheology.simulate_servo_shear",
    "box": [W, D, H],
    "n_particles_expected": ">1000",
    "reference": {
        "mu": {"mu_s": MU_S, "mu_2": MU_2, "I_0": I_0,
                "source": "da Cruz 2005 / GDR MiDi 2004"},
        "phi": {"phi_c": PHI_C, "phi_j": PHI_J,
                "status": "PROVENANCE-UNVERIFIED (hand-copied)"},
    },
    "hypotheses": {
        "P1_noise_floor": "mu std (final 25 % window) < 10 % at I in [1e-2, 3e-2]",
        "P2_mu_convergence": "|mu_meas/mu_ref - 1| < 10 % at I in [1e-2, 3e-2]",
        "P3_honest_verdict": "full-curve PASS/FAIL reported with worst error",
    },
}


def ref_mu(I: float) -> float:
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def ref_phi(I: float) -> float:
    return PHI_C + (PHI_J - PHI_C) / (I_0 / I + 1.0)


def run_point(I: float) -> dict:
    case = ServoShearCase(inertial_number=I, mu=MU_CONTACT,
                          mu_rolling=MU_ROLLING, restitution=E, kn=KN,
                          normal_stress=P, shear_strain=SHEAR_STRAIN,
                          settle_steps=SETTLE_STEPS,
                          min_shear_steps=1,          # strain target is exact
                          sample_stride=SAMPLE_STRIDE,
                          packing_gap=0.02, jitter=0.01, seed=SEED,
                          radius=R, density=RHO, box_width=W,
                          box_depth=D, fill_height=H)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case, return_series=True)
    wall = time.perf_counter() - t0
    series = res["series"]

    # --- noise floor: std of mu over the final 25 % window ----------------
    mu = [s["mu"] for s in series]
    n_tail = max(1, int(0.25 * len(mu)))
    tail = mu[-n_tail:]
    mu_std = float(_std(tail))
    mu_std_rel_pct = 100.0 * mu_std / (abs(_mean(tail)) + 1e-12)

    # --- steady-state mu / phi (final 25 % window) -------------------------
    mu_meas = float(_mean(tail))
    phi_steady = res.get("phi_bed_steady")
    phi_meas = float(phi_steady) if phi_steady is not None else float("nan")

    i_meas = res["inertial_number"]
    mu_ref = ref_mu(i_meas)
    phi_ref = ref_phi(i_meas)
    err_mu_pct = 100.0 * (mu_meas / mu_ref - 1.0)
    err_phi_pct = (100.0 * (phi_meas / phi_ref - 1.0)
                   if not math.isnan(phi_meas) else float("nan"))

    return {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 18,
        "I_target": I,
        "I_measured": i_meas,
        "n_particles": res["n_particles"],
        "mu_meas": mu_meas,
        "mu_ref": mu_ref,
        "err_mu_pct": err_mu_pct,
        "mu_std_rel_pct": mu_std_rel_pct,
        "phi_meas": phi_meas,
        "phi_ref": phi_ref,
        "err_phi_pct": err_phi_pct,
        "normal_stress_pa": res["normal_stress_measured_pa"],
        "pressure_control_error": (res["normal_stress_measured_pa"]
                                   / res["normal_stress_applied_pa"] - 1.0),
        "shear_stress_pa": res["shear_stress_pa"],
        "coordination": res["coordination_number"],
        "wall_clock_s": wall,
        "dt": res["dt"],
        "shear_steps": res["shear_steps"],
        "settle_steps": res["settle_steps"],
        "servo_valid": res.get("servo_valid"),
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if xs else float("nan")


def assemble() -> dict:
    rows = []
    for f in sorted(RUNS.glob("point_*.json")):
        rows.append(json.loads(f.read_text()))
    rows.sort(key=lambda r: r["I_target"])
    verdict = {
        "P1_noise_floor": all(r["mu_std_rel_pct"] < 10.0
                              for r in rows if 1e-2 <= r["I_target"] <= 3e-2),
        "P2_mu_convergence": all(abs(r["err_mu_pct"]) < TOL
                                 for r in rows if 1e-2 <= r["I_target"] <= 3e-2),
        "worst_mu_err_pct": max(abs(r["err_mu_pct"]) for r in rows),
        "mu_within_10pct_full": all(abs(r["err_mu_pct"]) < TOL for r in rows),
        "rows": rows,
    }
    (RUNS / "result.json").write_text(json.dumps(verdict, indent=1))
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n, i in [0,n)")
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.assemble:
        v = assemble()
        print(json.dumps(v, indent=1))
        return
    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit("shard index out of range")
    assigned = [I for k, I in enumerate(I_TARGETS) if k % n == i]
    print(f"shard {i}/{n}: I targets {assigned}", flush=True)
    for I in assigned:
        rec = run_point(I)
        (RUNS / f"point_{I:g}.json").write_text(json.dumps(rec, indent=1))
        print(f"I={I:g}: mu={rec['mu_meas']:.4f} err={rec['err_mu_pct']:+.1f}% "
              f"noise={rec['mu_std_rel_pct']:.1f}% phi={rec['phi_meas']:.4f} "
              f"N={rec['n_particles']} ({rec['wall_clock_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
