"""Run the P54-W2-N01 mixed FY/MY ice campaign and write every artifact.

Usage
-----
    python scripts/mixed_ice_campaign.py [--no-dem] [--workers N]

Outputs
-------
``artifacts/mixed_ice/mixed_ice_campaign.json``   the full artifact (verdicts,
                                                  tables, provenance hashes)
``artifacts/mixed_ice/figures/*.png``             figures of the report
``artifacts/mixed_ice/RESULTS.md``                generated results section
``docs/MIXED_ICE_RESISTANCE.md``                  the report itself (prose from
                                                  ``mixed_ice/report.py``, every
                                                  number injected)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.mixed_ice import campaign as cmp  # noqa: E402
from tensordem.mixed_ice import figures, report  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "mixed_ice"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-dem", action="store_true", help="skip the DEM factorial")
    ap.add_argument("--no-resolution", action="store_true",
                    help="skip the n_z = 6 resolution pair")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(ARTIFACTS))
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    artifact = cmp.run_campaign(workers=args.workers, with_dem=not args.no_dem,
                               dem_resolution_check=not args.no_resolution)
    path = cmp.save_artifact(artifact, outdir / "mixed_ice_campaign.json")

    figdir = outdir / "figures"
    made = figures.make_all(artifact, figdir)

    (outdir / "RESULTS.md").write_text(report.render_results_markdown(artifact))
    (ROOT / "docs" / "MIXED_ICE_RESISTANCE.md").write_text(
        report.render_report(artifact, figures=[p.name for p in made])
    )
    print(json.dumps({
        "artifact": str(path),
        "figures": [str(p) for p in made],
        "n_pass": artifact["n_pass"],
        "n_fail": artifact["n_fail"],
        "wall_clock_s": artifact["wall_clock_s"],
        "verdicts": {k: v["verdict"] for k, v in artifact["verdicts"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())