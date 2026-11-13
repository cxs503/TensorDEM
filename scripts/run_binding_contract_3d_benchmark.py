#!/usr/bin/env python3
"""Run the 3-D numeric benchmark of the frozen binding contract.

Anchor: XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT (round 2 / m17960).

What it does
------------
Calls :func:`tensordem.binding_contract.benchmark_physics_3d` (the rows are
analytic ref-vs-got comparisons for ``IceDEM3D``) and writes, next to each
other, a machine-readable artifact and the human table.  Every number in the
artifact is produced by this run: nothing is transcribed by hand.

Outputs (defaults)::

    artifacts/xval_library_binding_contract/benchmark_physics_3d.json
    artifacts/xval_library_binding_contract/benchmark_physics_3d_table.md

Repro (one line, inside the repo checkout; no GPU, ~15 s on one core)::

    python scripts/run_binding_contract_3d_benchmark.py

Exit code is 0 only when every row passes its own pre-registered tolerance.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem import binding_contract as bc  # noqa: E402

OWNER_TOLERANCE = 0.15


def _table_markdown(report: dict) -> str:
    lines = [
        "| row | ref | got | rel. error | tol | pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for r in report["rows"]:
        ref, got = r["ref"], r["got"]
        if isinstance(ref, list):
            ref_s = "[" + ", ".join(f"{v:.6e}" for v in ref) + "]"
            got_s = "[" + ", ".join(f"{v:.6e}" for v in got) + "]"
        elif isinstance(ref, float) and abs(ref) < 1e-6 and abs(got) < 1e-6:
            ref_s, got_s = f"{ref:.3e}", f"{got:.3e}"
        else:
            ref_s, got_s = f"{float(ref):.9g}", f"{float(got):.9g}"
        lines.append(
            f"| `{r['name']}` | {ref_s} | {got_s} | {r['rel_error']:.3e} | "
            f"{r['tolerance']:.0e} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append(f"- rows passed: **{report['rows_passed']}/{report['rows_total']}**")
    lines.append(f"- worst relative error: **{report['worst_rel_error']:.3e}** "
                 f"(owner acceptance line {OWNER_TOLERANCE:.0%})")
    lines.append(f"- observed free-flight convergence orders: "
                 f"{[round(o, 6) for o in report['free_flight_convergence_order']]}")
    lines.append(f"- verdict: **{report['verdict']}**")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "artifacts" / "xval_library_binding_contract",
        help="directory for the JSON artifact and the markdown table",
    )
    parser.add_argument("--energy-steps", type=int, default=1000,
                        help="steps for the conservative-lattice energy-drift row")
    args = parser.parse_args()
    if args.energy_steps < 1:
        parser.error("--energy-steps must be positive")

    import torch  # local import: report the version in the artifact

    t0 = time.perf_counter()
    report = bc.benchmark_physics_3d(n_steps_energy=args.energy_steps)
    wall = time.perf_counter() - t0

    payload = {
        "schema": "tensordem.xval.binding-contract-3d/1",
        "anchor": "XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT",
        "round": 2,
        "mission": "m17960",
        "entry_point": "tensordem.binding_contract.benchmark_physics_3d",
        "repro_cmd": "python scripts/run_binding_contract_3d_benchmark.py",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "contract_schema": bc.CONTRACT_SCHEMA,
            "contract_version": bc.CONTRACT_VERSION,
            "contract_fingerprint": bc.fingerprint(),
            "frozen_fingerprint_match": bc.fingerprint() == bc.FROZEN_FINGERPRINT,
        },
        "accept_rel_error": OWNER_TOLERANCE,
        "benchmark_physics_3d": report,
        "wall_s": wall,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / "benchmark_physics_3d.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                         encoding="utf-8")
    md_path = args.outdir / "benchmark_physics_3d_table.md"
    md_path.write_text(_table_markdown(report), encoding="utf-8")

    print(md_path.read_text(encoding="utf-8"))
    print(f"fingerprint match: {payload['environment']['frozen_fingerprint_match']}")
    print(f"wall: {wall:.2f} s -> {json_path}")
    return 0 if report["verdict"] == "MET" else 1


if __name__ == "__main__":
    raise SystemExit(main())