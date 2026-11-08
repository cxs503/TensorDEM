"""Run the P35-N2 wet-ice cohesion campaign and write artifacts + figures.

Usage::

    PYTHONPATH=src python scripts/wet_ice_cohesion_study.py
    PYTHONPATH=src python -c "from tensordem.wet_ice_cohesion import figures; \
        print(figures.make_all())"
"""
from __future__ import annotations

import argparse
import json

from tensordem.wet_ice_cohesion import figures, study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts/wet_ice_cohesion")
    ap.add_argument("--skip-negative-control", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()
    result = study.run_all(args.outdir, with_negative_control=not args.skip_negative_control)
    print(json.dumps(result["summary"], indent=1))
    if not args.skip_figures:
        for path in figures.make_all(f"{args.outdir}/campaign.json"):
            print("figure:", path)


if __name__ == "__main__":
    main()
