"""P316-W1-N09 campaign entry point: 3-D DEM contact -> macro resistance closure.

Runs the closure campaign (thickness sweep, particle refinement, mass-scaling
axis, no-load negative control) and writes ``results.json`` plus the case files
under ``artifacts/p316_n09/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tensordem.resistance_closure.campaign import CampaignSpec, run_campaign

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "p316_n09"


def main(argv: list[str] | None = None) -> int:
    out = Path(argv[1]) if argv and len(argv) > 1 else OUT
    report = run_campaign(out, CampaignSpec())
    print(json.dumps(report["criteria"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
