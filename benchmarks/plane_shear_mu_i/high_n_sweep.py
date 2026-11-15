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
# round 19 (m21228): the reference law is imported, never re-copied.  This file
# used to carry its own ``(mu_s, mu_2, I_0) = (0.36, 0.84, 0.30)`` trio *and*
# its own ``ref_phi`` -- and that local ``ref_phi`` had the dilatancy sign
# inverted (``phi_c + dphi/(I0/I+1)`` grows with I, whereas the physical
# reference must *decrease*).  The round-2 defect D2 was thereby reintroduced
# in the round-18 sweep, so the phi column of a round-18 assembly would have
# had the wrong sign; no round-18 row was ever assembled, so no archived number
# is affected.  ``reference_law.ref_phi`` is now the only definition.
from reference_law import (  # noqa: E402
    I_0, MU_2, MU_REF_STATUS, MU_S, PHI_MAX, PHI_MIN, PHI_REF_STATUS,
    ref_mu, ref_phi, reference as reference_law,
)

RUNS = HERE / "runs" / "high_n"
RUNS.mkdir(parents=True, exist_ok=True)

# --- reference: imported from reference_law (single source of truth) ------
# Round 19 (m21228) defect fix: this was ``TOL = 0.10`` while every residual in
# this file is a *percentage* (``err_mu_pct``), so the ledger band "error < 10 %"
# was silently enforced as "error < 0.1 %" -- any assembly would have reported a
# spurious FAIL.  The band is now stated in the unit it is compared in.
TOL_PCT = 10.0
TOL = TOL_PCT          # backward-compatible name, now correctly scaled

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
                "source": "da Cruz 2005 / GDR MiDi 2004",
                "status": MU_REF_STATUS},
        "phi": {"phi_max": PHI_MAX, "phi_min": PHI_MIN,
                "form": "phi_max - (phi_max - phi_min)/(I0/I + 1)",
                "status": PHI_REF_STATUS},
        "origin": "benchmarks/plane_shear_mu_i/reference_law.py (imported)",
    },
    "hypotheses": {
        "P1_noise_floor": "mu std (final 25 % window) < 10 % at I in [1e-2, 3e-2]",
        "P2_mu_convergence": "|mu_meas/mu_ref - 1| < 10 % at I in [1e-2, 3e-2]",
        "P3_honest_verdict": "full-curve PASS/FAIL reported with worst error",
    },
}


def summarise(res: dict, I_target: float) -> dict:
    """Steady-state summary of one servo run (pure: no simulation).

    Split out of ``run_point`` in round 19 (m21228) so the abscissa and the
    reference handling are unit-testable without a 5-minute DEM run.

    Two abscissae are reported, and the verdict is taken on the *measured*
    one:

    * ``I_target``     -- the value the servo was asked to hold;
    * ``I_imposed_rate`` -- ``inertial_number`` (from the platen rate);
    * ``I_measured_rate`` -- from the *measured* velocity-profile shear rate
      (final window).  Rounds 9/10 established this as the physical abscissa:
      with wall slip the bed shears slower than the platen drives it, so the
      imposed-rate abscissa labels a high-I point at a higher I than the
      material actually experiences and biases ``mu`` low.
    """
    series = res.get("series") or []
    mu = [s["mu"] for s in series]
    n_tail = max(1, int(0.25 * len(mu))) if mu else 0
    tail = mu[-n_tail:] if n_tail else []
    mu_std_rel_pct = (100.0 * float(_std(tail)) / (abs(_mean(tail)) + 1e-12)
                      if tail else float("nan"))
    mu_meas = float(_mean(tail)) if tail else float("nan")

    i_imposed = res.get("inertial_number")
    i_measured = res.get("inertial_number_measured_rate_final")
    if i_measured is None:
        i_measured = res.get("inertial_number_measured_rate")
    abscissa = i_measured if i_measured is not None else i_imposed

    # phi: the container-averaged steady value is the definition the reference
    # law uses; the two alternative estimators are carried so the spread is
    # visible rather than hidden behind one number (rounds 13-15 audits).
    phi_meas = res.get("phi_bed_steady")
    phi_meas = float(phi_meas) if phi_meas is not None else float("nan")
    phi_extent = res.get("phi_extent_final")
    phi_container = res.get("phi_container_final")

    mu_ref = ref_mu(abscissa) if abscissa else float("nan")
    phi_ref = ref_phi(abscissa) if abscissa else float("nan")
    mu_ref_imposed = ref_mu(i_imposed) if i_imposed else float("nan")
    phi_ref_imposed = ref_phi(i_imposed) if i_imposed else float("nan")

    def _err(v, r):
        if v is None or r is None or math.isnan(v) or math.isnan(r) or r == 0:
            return float("nan")
        return 100.0 * (v / r - 1.0)

    return {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 19,
        "I_target": I_target,
        "I_imposed_rate": i_imposed,
        "I_measured_rate": i_measured,
        "abscissa": "I_measured_rate",
        "I_abscissa": abscissa,
        "n_particles": res.get("n_particles"),
        "mu_meas": mu_meas,
        "mu_ref": mu_ref,
        "err_mu_pct": _err(mu_meas, mu_ref),
        "mu_std_rel_pct": mu_std_rel_pct,
        "mu_ref_at_imposed_I": mu_ref_imposed,
        "err_mu_pct_at_imposed_I": _err(mu_meas, mu_ref_imposed),
        "phi_meas": phi_meas,
        "phi_ref": phi_ref,
        "err_phi_pct": _err(phi_meas, phi_ref),
        "phi_ref_at_imposed_I": phi_ref_imposed,
        "phi_extent_final": phi_extent,
        "phi_container_final": phi_container,
        "normal_stress_pa": res.get("normal_stress_measured_pa"),
        "pressure_control_error": (
            res["normal_stress_measured_pa"] / res["normal_stress_applied_pa"] - 1.0
            if res.get("normal_stress_applied_pa") else float("nan")),
        "shear_stress_pa": res.get("shear_stress_pa"),
        "coordination": res.get("coordination_number"),
        "shear_rate_imposed": res.get("shear_rate_imposed"),
        "shear_rate_measured": res.get("shear_rate_measured"),
        "shear_rate_ratio": (
            res["shear_rate_measured"] / res["shear_rate_imposed"]
            if res.get("shear_rate_measured") and res.get("shear_rate_imposed")
            else float("nan")),
        "shear_strain_achieved": res.get("shear_strain_achieved"),
        "dt": res.get("dt"),
        "shear_steps": res.get("shear_steps"),
        "settle_steps": res.get("settle_steps"),
        "servo_valid": res.get("servo_valid"),
    }


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
    rec = summarise(res, I)
    rec["wall_clock_s"] = wall
    return rec


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if xs else float("nan")


def assemble(runs_dir=None) -> dict:
    """Collate the per-point rows into the pre-registered verdict (P1/P2/P3)."""
    runs = Path(runs_dir) if runs_dir is not None else RUNS
    rows = []
    for f in sorted(runs.glob("point_*.json")):
        rows.append(json.loads(f.read_text()))
    rows.sort(key=lambda r: r["I_target"])
    if not rows:
        raise SystemExit(f"no point_*.json in {runs}")

    def _mid(rs):
        return [r for r in rs if 1e-2 <= r["I_target"] <= 3e-2]

    def _num(r, key):
        v = r.get(key)
        return float("nan") if v is None else float(v)

    mid = _mid(rows)
    verdict = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 19,
        "abscissa": "I_measured_rate (fallback: imposed rate if unavailable)",
        "P1_noise_floor": all(_num(r, "mu_std_rel_pct") < 10.0 for r in mid),
        "P2_mu_convergence": all(abs(_num(r, "err_mu_pct")) < TOL_PCT
                                 for r in mid),
        "P3_honest_verdict": True,
        "tolerance_pct": TOL_PCT,
        "worst_mu_err_pct": max(abs(_num(r, "err_mu_pct")) for r in rows),
        "mu_within_10pct_full": all(abs(_num(r, "err_mu_pct")) < TOL_PCT
                                    for r in rows),
        "worst_mu_err_pct_at_imposed_I": max(
            abs(_num(r, "err_mu_pct_at_imposed_I")) for r in rows),
        "phi_within_10pct_full": all(
            abs(_num(r, "err_phi_pct")) < TOL_PCT for r in rows),
        "worst_phi_err_pct": max(abs(_num(r, "err_phi_pct")) for r in rows),
        "n_particles": [_num(r, "n_particles") for r in rows],
        "servo_valid_all": all(r.get("servo_valid") in (True, None) for r in rows),
        "rows": rows,
    }
    (runs / "result.json").write_text(json.dumps(verdict, indent=1))
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
