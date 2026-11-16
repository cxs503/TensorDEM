"""Round 33 campaign: ``wall_translate`` ablation on the archived device.

Sharded exactly like the other campaign runners in this directory::

    python run_wall_translation_campaign.py --index 0 --total 8
    python run_wall_translation_campaign.py --list

The plan is a 2 x 4 grid: the wall kinematics (``frozen`` = the archived rounds
24-32, ``fixed`` = wall grains advected) against the four inertial numbers of the
archived sweep.  Device, protocol, seeds and ruler are the archived ones
(``n_particles=1220``, ``W=0.40``, ``gap=0.30``, settle 60, transient 0.5 strain,
window 1.0 strain), so a ``frozen`` point is a direct reproduction of the archived
number and the paired ``fixed`` point differs by one boolean.

Agenda pre-registration (written before any campaign result was read, from the
smoke pair in ``runs/wall_translation_r33/smoke_filled_n310_*``):

* H1 (closure): ``gamma_dot_ratio`` rises from ``<0.15`` to ``>0.60`` when the
  walls translate; slip length ``/h`` drops below ``0.5``;
* H2 (confinement): the constant-P servo reaches ``p_wall/P > 0.8`` only on the
  ``fixed`` device -- a frozen wall leaks the confining load through slip;
* H3 (rheology): ``mu_bulk`` moves up towards the da Cruz 2-D law as a *result*
  of H1/H2, not by tuning anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parents[1]
for _p in (str(REPO / "src"), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wall_translation_ablation import RUNS, run_point  # noqa: E402

MODES = ("frozen", "fixed")
INERTIAL = (0.03, 0.05, 0.10, 0.20)
# archived full device (round 24 pre-registration)
FULL = dict(n_particles=1220, width=0.40, gap=0.30, smoke=False)


def plan() -> list[dict]:
    out = []
    for mode in MODES:
        for I in INERTIAL:
            out.append(dict(mode=mode, I=I, seed=0, **FULL))
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--total", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    todo = plan()
    if args.list:
        for i, p in enumerate(todo):
            print(i, json.dumps(p))
        return 0
    if args.index is None:
        raise SystemExit("--index required (see --list)")
    mine = [p for i, p in enumerate(todo) if i % args.total == args.index]
    RUNS.mkdir(parents=True, exist_ok=True)
    for p in mine:
        res = run_point(inertial_number=p["I"], mode=p["mode"], seed=args.seed,
                        n_particles=p["n_particles"], smoke=p["smoke"],
                        width=p["width"], gap=p["gap"])
        out = RUNS / f"full_w{p['mode']}_I{p['I']:g}_s{args.seed}.json"
        out.write_text(json.dumps(res, indent=2, sort_keys=True))
        print(f"{out.name}: ratio={res['gamma_dot_ratio']:.3f} "
              f"p/P={res['p_wall_over_P']:.3f} mu={res['mu_bulk']:.3f} "
              f"phi={res['phi_gap']:.4f} ({res['duration_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())