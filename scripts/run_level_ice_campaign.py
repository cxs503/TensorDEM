#!/usr/bin/env python3
"""Run the level-ice resistance DEM campaign and write the JSON artifact.

Example::

    python scripts/run_level_ice_campaign.py --workers 48 \
        --output artifacts/level_ice/level_ice_campaign.json --figures

The artifact is the single source of every number quoted in
``docs/LEVEL_ICE_RESISTANCE_DEM.md`` and in the campaign report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.level_ice import campaign as C  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "level_ice"
                                                / "level_ice_campaign.json"))
    parser.add_argument("--figures", action="store_true",
                        help="also render the report figures")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--mass-scale", type=float, default=200.0)
    parser.add_argument("--advance", type=float, default=6.0)
    parser.add_argument("--length", type=float, default=24.0)
    args = parser.parse_args(argv)

    started = time.time()
    artifact = C.run_campaign(workers=args.workers, layers=args.layers,
                              mass_scale=args.mass_scale,
                              advance_m=args.advance, length_m=args.length)
    artifact["wall_clock_s"] = time.time() - started
    path = C.save_artifact(artifact, args.output)
    print(f"artifact: {path}")
    print(json.dumps({name: bool(block.get("pass"))
                      for name, block in artifact["verdicts"].items()}, indent=2))
    if args.figures:
        from plot_level_ice_campaign import main as plot_main  # type: ignore
        plot_main([sys.argv[0], str(path), str(Path(args.output).parent / "figures")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())