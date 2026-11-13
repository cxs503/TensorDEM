"""Round 11 of CASE-DEM-PLANE-SHEAR-MUIDI: smooth vs rough walls (wall slip).

The case has spent ten rounds on the same device and the same question -- why is
the measured ``mu`` 20-27 % below da Cruz et al. (2005)?  Round 10
(``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R10.md``) ruled out the two bookkeeping
excuses (abscissa, averaging window) and left *sample identification* as the
open cause.  The leading mechanism for that is **wall slip**: the cell drives
the sample with a flat, frictionally-limited platen, and round 8 measured

    gamma_dot_measured / gamma_dot_imposed = 0.38 ... 0.97   (wall slip)

da Cruz et al. drive their cell with **rough walls** precisely to remove this
degree of freedom.

This script runs the *same* device (``ServoShearCase``), the same protocol, the
same seeds and the same reference law, with only ``wall_mode`` switched:

* ``smooth`` -- the historical flat platen + frictionless floor (baseline),
* ``rough``  -- grafted monolayers of spheres on the platen and the floor,
  moving rigidly with the wall (``tensordem.mu_i_rheology.graft_lattice``).

Every point reports the wall-traction ``mu``, the volume-averaged ``mu``, the
bulk-slab ``mu``, the measured / imposed shear-rate ratio and the velocity
profile ``R^2``, so the effect of the wall can be read off directly.

Usage
-----
    python rough_wall_sweep.py --list
    python rough_wall_sweep.py --index 3           # one point of the plan
    python rough_wall_sweep.py --index 3 --total 8 # sharded dispatch (cluster)
"""
from __future__ import annotations

import argparse
import importlib.util
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

from reference_law import ref_mu  # noqa: E402

OUT = HERE / "runs" / "rough_wall"
OUT.mkdir(parents=True, exist_ok=True)

# --- device: identical to benchmarks/plane_shear_mu_i/run_sweep.py baseline --
P = 6.25e4
KN = 6.25e4
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E = 0.3
R = 0.005
RHO = 2500.0
W, D, H = 0.12, 0.05, 0.06
SHEAR_STRAIN = 0.2
SETTLE_STEPS = 2500
MIN_SHEAR_STEPS = 400

I_TARGETS = (1e-2, 3e-2, 1e-1, 3e-1)
MODES = ("smooth", "rough")
SEEDS = (5279,)

PLAN = [(mode, I, seed) for mode in MODES for I in I_TARGETS for seed in SEEDS]


def run_point(mode: str, I_target: float, seed: int, **over) -> dict:
    case = ServoShearCase(
        inertial_number=I_target, mu=MU_CONTACT, mu_rolling=MU_ROLLING,
        restitution=E, kn=KN, normal_stress=P, shear_strain=SHEAR_STRAIN,
        settle_steps=SETTLE_STEPS, min_shear_steps=MIN_SHEAR_STEPS,
        packing_gap=0.02, jitter=0.01, seed=seed, radius=R, density=RHO,
        box_width=W, box_depth=D, fill_height=H, wall_mode=mode, **over)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case)
    wall = time.perf_counter() - t0

    def g(key):
        v = res.get(key)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None

    rec = {
        "wall_mode": mode,
        "I_target": I_target,
        "seed": seed,
        "rough_spacing": case.rough_spacing if mode == "rough" else None,
        "mu_ref": ref_mu(I_target),
        "mu_wall": g("mu_effective"),
        "mu_full": g("full_mu"),
        "mu_full_zz": g("full_mu_zz"),
        "mu_bulk": g("bulk_mu"),
        "mu_bulk_mid": g("bulk_mid_mu"),
        "mu_bulk_final": g("bulk_mu_final"),
        "P_measured_pa": g("normal_stress_measured_pa"),
        "P_rel_err_final": g("pressure_control_error_final"),
        "mu_wall_final": g("mu_final"),
        "gamma_dot_imposed": g("shear_rate_imposed"),
        "gamma_dot_measured": g("shear_rate_measured"),
        "gamma_dot_ratio": g("shear_rate_ratio"),
        "profile_r2": g("profile_r2"),
        "I_imposed_rate": g("inertial_number_imposed_rate"),
        "I_measured_rate": g("inertial_number_measured_rate"),
        "phi_solid_fraction": g("solid_fraction"),
        "phi_bed_steady": g("phi_bed_steady"),
        "dilatancy": g("dilatancy"),
        "servo_valid": res.get("servo_valid"),
        "n_particles": res.get("n_particles"),
        "wall_clock_s": wall,
    }
    for key in ("mu_wall", "mu_full", "mu_full_zz", "mu_bulk"):
        if rec[key] is not None:
            rec[key + "_err_pct"] = 100.0 * (rec[key] / rec["mu_ref"] - 1.0)
    rec["case_fingerprint"] = {
        "P": P, "kn": KN, "mu_contact": MU_CONTACT, "mu_rolling": MU_ROLLING,
        "restitution": E, "radius": R, "density": RHO,
        "box": [W, D, H], "shear_strain": SHEAR_STRAIN,
        "settle_steps": SETTLE_STEPS, "min_shear_steps": MIN_SHEAR_STEPS,
    }
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--index", type=int, default=None, help="run plan[index]")
    ap.add_argument("--total", type=int, default=None,
                    help="number of shards (with --index: shard stride dispatch)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--servo-max-step", type=float, default=None,
                    help="override ServoShearCase.servo_max_step_d (settle budget)")
    ap.add_argument("--settle-steps", type=int, default=None,
                    help="unused placeholder: settle budget is fixed by the protocol")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    if args.list:
        for i, (mode, I, seed) in enumerate(PLAN):
            print(i, mode, I, seed)
        return 0

    if args.index is None:
        points = PLAN
        out = args.out or OUT / "sweep.jsonl"
    else:
        if args.total and args.total > 1:
            # modulo dispatch: tolerates both 0-based and 1-based shard ids
            points = [p for k, p in enumerate(PLAN)
                      if k % args.total == args.index % args.total]
        else:
            points = [PLAN[args.index % len(PLAN)]]
        out = args.out or OUT / f"shard_{args.index:03d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for mode, I, seed in points:
            over = {}
            if args.servo_max_step is not None:
                over["servo_max_step_d"] = args.servo_max_step
            try:
                rec = run_point(mode, I, seed, **over)
            except Exception as exc:  # honest FAIL is a result, not a crash
                rec = {"wall_mode": mode, "I_target": I, "seed": seed,
                       "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                       "overrides": over, "tag": args.tag}
            rec["tag"] = args.tag
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if rec.get("status") == "FAIL":
                print(f"{mode:6s} I={I:<6g} seed={seed} FAIL {rec['error']}", flush=True)
            else:
                print(f"{mode:6s} I={I:<6g} seed={seed} mu_wall={rec['mu_wall']} "
                      f"mu_full={rec['mu_full']} rate_ratio={rec['gamma_dot_ratio']} "
                      f"r2={rec['profile_r2']} ({rec['wall_clock_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())