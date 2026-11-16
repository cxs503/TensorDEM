"""Round 31 campaign: the R30 configuration, instrumented with the depth profile.

R30 (``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R30.md``) closed the dimensional mismatch
(2-D disks vs the 2-D law of da Cruz 2005) and rewrote the failure signature:
``nu_max`` agrees to +1.7 % but the rate sensitivity is 3-4x too small
(``b = 0.279 +- 0.031`` vs 1.0) and the intercept is 29 % low.  It left two
open items, both requiring a measurement that did not exist:

1. **the abscissa** - is the realised shear rate ``2 V / h_eff``?  R30's own
   numbers are internally inconsistent (KE implies ~0.1 m/s of granular motion,
   ``I = 0.19`` needs ``gammadot d ~ 0.39 m/s``).
2. **the rate-dependent stress** - only an upper bound ``p_kin <= KE/A``
   (0.132 % of ``p_bulk``) exists; the flow-stress term was never measured.

This driver runs the **bit-identical** rounds-24..30 case (458 free disks,
``P = 10 kPa``, ``mu_p = 0.4``, ``w = 0.5``, 8-strain window, same combos as
``cj4332``) with ``profile_r31.ProfileAccumulator`` attached, so every archived
number stays comparable and the new ordinates are measured on the same samples.

    python benchmarks/plane_shear_mu_i/run_profile_r31.py --workers 6
    python benchmarks/plane_shear_mu_i/run_profile_r31.py --smoke

Output: ``runs/plane_shear_2d/prof_r31_w{w}_I{I}_s{seed}.json`` + index.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
for _p in (str(REPO / "src"), str(BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plane_shear_2d as ps  # noqa: E402

RUNS = BENCH / "runs" / "plane_shear_2d"

# (polydispersity w, I_target, seed) - the archived cj4332 combo set, unchanged.
COMBOS = [
    (0.5, 0.02, 0),
    (0.5, 0.05, 0),
    (0.5, 0.05, 1),
    (0.5, 0.20, 0),
    (0.5, 0.20, 1),
    (0.0, 0.05, 0),      # monodisperse crystal control (not fitted)
]

STRAIN_MEASURE = 8.0


def case_for(w: float, inertial: float, seed: int, *, smoke: bool = False):
    if smoke:
        return ps.PlaneShear2DCase(
            I_target=inertial, n_particles=200, width=0.24, gap=0.12,
            settle_time_units=20.0, shear_settle_strain=0.3, strain_measure=0.6,
            init_mode="random", seed=seed, polydispersity=w)
    return ps.PlaneShear2DCase(
        I_target=inertial, n_particles=458, width=0.30, gap=0.15,
        settle_time_units=40.0, shear_settle_strain=0.5,
        strain_measure=STRAIN_MEASURE,
        init_mode="random", seed=seed, polydispersity=w)


def run_one(task: tuple[float, float, int, bool]) -> dict:
    w, inertial, seed, smoke = task
    case = case_for(w, inertial, seed, smoke=smoke)
    case.validate()
    t0 = time.time()
    out = ps.simulate(case)
    out["argv"] = {"w": w, "I_target": inertial, "seed": seed,
                   "strain_measure": case.strain_measure, "smoke": smoke}
    out["round"] = 31
    tag = f"prof_r31_w{w:g}_I{inertial:g}_s{seed}" + ("_smoke" if smoke else "")
    path = RUNS / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    der = out.get("profile_derived", {})
    prof = out.get("profile", {})
    print(("prof_r31 w=%.2f I=%.3f seed=%d -> mu_bulk %.4f phi %.4f | "
           "gdot ratio %.3f r2 %.3f slip %.3f | p_kin %.3e dur %.0fs -> %s")
          % (w, inertial, seed, out.get("mu_bulk", float("nan")),
             out.get("phi_gap", float("nan")),
             der.get("gamma_dot_ratio", float("nan")),
             prof.get("fit", {}).get("r2", float("nan")),
             der.get("slip_length_bottom_over_h", float("nan")),
             prof.get("p_kinetic_total", float("nan")),
             time.time() - t0, path.name), flush=True)
    return {"file": str(path.relative_to(REPO)), "w": w, "I_target": inertial,
            "seed": seed, "mu_bulk": out.get("mu_bulk"),
            "mu_wall": out.get("mu_wall"), "phi_gap": out.get("phi_gap"),
            "I_bulk": out.get("I_bulk"),
            "gamma_dot_ratio": der.get("gamma_dot_ratio"),
            "r2_linear": prof.get("fit", {}).get("r2"),
            "duration_s": out.get("duration_s")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", type=str, default=None,
                    help="single combo, e.g. 'w=0.5,I=0.05,seed=0'")
    args = ap.parse_args(argv)

    combos = COMBOS
    if args.only:
        parts = dict(p.split("=") for p in args.only.split(","))
        combos = [(float(parts["w"]), float(parts["I"]), int(parts["seed"]))]
    if args.smoke and not args.only:
        combos = [combos[1]]

    tasks = [(w, inertial, seed, args.smoke) for w, inertial, seed in combos]
    t0 = time.time()
    if args.workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
            index = pool.map(run_one, tasks)
    else:
        index = [run_one(t) for t in tasks]

    index_path = RUNS / ("prof_r31_index_smoke.json" if args.smoke
                         else "prof_r31_index.json")
    index_path.write_text(json.dumps(
        {"round": 31, "strain_measure": STRAIN_MEASURE, "combos": index}, indent=1))
    print(f"wrote {index_path} ({len(index)} runs, {time.time() - t0:.0f}s total)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())