"""Entry point: run the anchor case ``CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE``.

Runs both halves of the case and writes the artefacts:

* ``artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/run.json``  -- reference vs
  recovered tables, acceptance arithmetic, environment echo, repro command;
* ``artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/shear_band_case.png`` -- (a)
  known-angle benchmark, (b) local shear-strain field of the late window with
  the fitted band line.

Usage
-----
``PYTHONPATH=src python benchmarks/biaxial_shear_band/run_anchor_case.py``
``--quick``       reduced specimen (smoke / CI; NOT a result)
``--no-dem``      known-angle benchmark only (seconds)
``--figures/--no-figures``
``TENSORDEM_THREADS=4``  torch thread count (default 4; 128-core oversubscription
is ~50x slower for this cell)

Exit status: 0 if every pre-declared criterion passes, 1 otherwise (an honest
FAIL is a valid outcome and still writes the artefacts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-dem", action="store_true")
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    parser.add_argument("--angles", default="30,45,60,70")
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = os.path.join(here, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    # This cell is small (a few thousand contacts): torch defaults to one thread
    # per core, and on a 128-core node the oversubscription is ~50x slower than
    # 4 threads.  Thread count is part of the reproducibility recipe.
    import torch

    torch.set_num_threads(int(os.environ.get("TENSORDEM_THREADS", "4")))

    from tensordem.biaxial_anchor_case import run_case

    angles = [float(a) for a in args.angles.split(",") if a.strip()]
    payload = run_case(output_dir=args.out_dir, quick=args.quick,
                       run_dem=not args.no_dem, angles_deg=angles,
                       figure=args.figures)

    print(f"case: {payload['case']}")
    print("known-angle benchmark (reference = injected angle):")
    print("  injected  recovered  abs_err  legacy_recovered  legacy_abs_err")
    for row in payload["known_angle"]["rows"]:
        legacy = row["legacy_recovered_deg"]
        print(f"  {row['injected_deg']:8.1f}  {row['recovered_deg']:9.1f}  "
              f"{row['abs_error_deg']:7.1f}  "
              f"{(legacy if legacy is not None else float('nan')):16.1f}  "
              f"{row['legacy_abs_error_deg']:14.1f}")
    if "dem" in payload:
        verdict = payload["dem"]["verdict"]
        print("DEM anchor run (24x48, p=100 kPa, wall_friction=0.55, "
              "4000 consolidation steps):")
        print(f"  ref {verdict['ref_deg']:.1f} deg  got {verdict['got_deg']:.2f} deg  "
              f"rel.err {100 * verdict['relative_error']:.1f} %  "
              f"tol {100 * verdict['tolerance']:.0f} %  "
              f"within={verdict['within_tolerance']}  localised={verdict['localised']}")
        print(f"  phi {payload['dem']['peak']['friction_angle_deg']:.2f} deg  "
              f"psi(global) {payload['dem']['peak']['dilation_angle_deg']:.2f} deg  "
              f"z {verdict['z_score']:.1f}  late snapshots {verdict['late_snapshots']}")
    print(json.dumps(payload["summary"], indent=1))
    print(f"[artifacts] {payload['artifacts']}")
    return 0 if payload["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())