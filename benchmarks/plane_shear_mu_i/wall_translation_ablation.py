"""Round 33: is the rough wall a *moving* boundary?  Ablation ``wall_translate``.

Why this exists
---------------
The 2-D device (``plane_shear_2d``, rounds 24-32) builds the rough wall as two
rows of **grain-glued** disks and gives those disks the wall velocity ``+-V``.
Round 31 then measured the flow profile and found the achieved strain rate to be
only ``0.07-0.13`` of the imposed one, with a slip length of ``2.8 h`` -- i.e.
the material almost plug-flows and the ``I`` axis of the whole calibration is not
the one the device claims to impose.

Reading ``RoughWallShearCell.advance`` (pre-round-33) locates a defect that
explains that signature without any new physics: ``_set_wall_kinematics``
re-imposes the rows' **y** positions and assigns their **velocity**, but nothing
ever advances their **x** positions.  The row is therefore a *frozen* corrugated
surface that happens to carry a velocity.  Its only remaining coupling to the
material is the tangential friction law, which saturates at ``mu_p N``; the
geometric interlocking that makes a rough wall rate-controlling -- a bump that
sweeps past and *pushes* the near-wall disks -- is absent.  The sibling 3-D path
in this repository does not have that defect: ``rough_wall_sweep`` grafts a
monolayer "moving rigidly with the wall".

The fix (``PlaneShear2DCase.wall_translate``, default ``True``) advects the wall
grains by ``+-V dt`` each step.  This module is the falsifiable test of that
diagnosis: the *same* case, the *same* protocol, the *same* reference ruler,
with only ``wall_translate`` switched.  If the diagnosis is right, the
measured/imposed shear-rate ratio must move towards 1 and the slip length must
collapse; if it is wrong (plug flow is genuine bulk localisation), the ratio
stays where round 31 found it.

Usage::

    python wall_translation_ablation.py --I 0.05 --mode fixed
    python wall_translation_ablation.py --I 0.05 --mode frozen --seed 1
    python wall_translation_ablation.py --I 0.05 --mode fixed --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parents[1]
for _p in (str(REPO / "src"), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plane_shear_2d import PlaneShear2DCase, simulate  # noqa: E402

RUNS = BENCH_DIR / "runs" / "wall_translation_r33"


def run_point(*, inertial_number: float, mode: str, seed: int, n_particles: int,
              smoke: bool, width: float | None = None,
              gap: float | None = None, settle: float | None = None,
              strain_settle: float | None = None,
              strain_measure: float | None = None) -> dict:
    """Run one ablation point and return the extractable numbers."""
    if mode not in ("fixed", "frozen"):
        raise ValueError("mode must be 'fixed' (wall grains advected) or 'frozen'")
    over = dict(
        I_target=float(inertial_number),
        seed=int(seed),
        n_particles=int(n_particles),
        width=float(width if width is not None else (0.24 if smoke else 0.40)),
        gap=float(gap if gap is not None else (0.14 if smoke else 0.30)),
        sample_every=40 if smoke else 80,
        # the smoke protocol is the round-31 instrument probe: same device, same
        # ruler, shorter settle -- a diagnostic, not anchor evidence
        settle_time_units=float(settle if settle is not None
                                else (20.0 if smoke else 60.0)),
        shear_settle_strain=float(strain_settle if strain_settle is not None
                                  else (0.4 if smoke else 0.5)),
        strain_measure=float(strain_measure if strain_measure is not None
                             else (0.6 if smoke else 1.0)),
        wall_translate=(mode == "fixed"),
    )
    case = PlaneShear2DCase(**over)
    out = simulate(case, progress=False)
    prof = out.get("profile", {})
    der = out.get("profile_derived", {})

    def g(key, default=float("nan")):
        v = out.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else float("nan")

    return {
        "anchor": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 33,
        "mode": mode,
        "wall_translate": bool(case.wall_translate),
        "smoke": bool(smoke),
        "I_target": case.I_target,
        "seed": case.seed,
        "n_free": out["n_free"],
        "n_wall_per_row": out["n_wall_per_row"],
        "gap_eff_final": g("gap_eff_final"),
        "phi_gap": g("phi_gap"),
        "mu_bulk": g("mu_bulk"),
        "mu_wall": g("mu_wall"),
        "p_wall_over_P": g("p_wall_over_P"),
        "sigma_yy_over_P": g("sigma_yy_over_P"),
        "I_bulk": g("I_bulk"),
        # -- the closure diagnostics (round-31 instrument) -------------------
        "gamma_dot_ratio": float(der.get("gamma_dot_ratio", float("nan"))),
        "gamma_dot_measured": float(der.get("gamma_dot_measured", float("nan"))),
        "gamma_dot_nominal": float(der.get("gamma_dot_nominal", float("nan"))),
        "I_measured": float(der.get("I_measured", float("nan"))),
        "slip_length_bottom_over_h": float(der.get("slip_length_bottom_over_h", float("nan"))),
        "slip_length_top_over_h": float(der.get("slip_length_top_over_h", float("nan"))),
        "profile_r2": float(prof.get("fit", {}).get("r2", float("nan"))),
        "psi6_mean": g("psi6_mean"),
        "case": asdict(case),
        "duration_s": g("duration_s"),
        "config_digest": out["config_digest"],
    }


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--I", dest="I", type=float, default=0.05)
    ap.add_argument("--mode", choices=("fixed", "frozen"), default="fixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", dest="n", type=int, default=240)
    ap.add_argument("--smoke", action="store_true",
                    help="round-31 short settle protocol (diagnostic only)")
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--settle", type=float, default=None,
                    help="compaction time in d/sqrt(P/rho) units")
    ap.add_argument("--strain-settle", dest="strain_settle", type=float, default=None)
    ap.add_argument("--strain-measure", dest="strain_measure", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = run_point(inertial_number=args.I, mode=args.mode, seed=args.seed,
                    n_particles=args.n, smoke=args.smoke,
                    width=args.width, gap=args.gap, settle=args.settle,
                    strain_settle=args.strain_settle,
                    strain_measure=args.strain_measure)
    RUNS.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out = Path(args.out) if args.out else (
        RUNS / f"{tag}_w{args.mode}_I{args.I:g}_s{args.seed}.json")
    out.write_text(json.dumps(res, indent=2, sort_keys=True))
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in res.items() if k != "case"}, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())