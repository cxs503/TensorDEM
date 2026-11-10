#!/usr/bin/env python
"""Run the p302w2-n03 FEM -> DEM handoff study and write its artefacts.

    PYTHONPATH=src python scripts/run_fem_dem_seeding.py [out_dir]
"""
from __future__ import annotations

import json
import os
import sys
import time

from tensordem.fem_dem_seeding import DEFAULT_OUT, run_study, write_figures


def main(argv: list[str]) -> int:
    out_dir = argv[1] if len(argv) > 1 else DEFAULT_OUT
    t0 = time.time()
    summary = run_study(out_dir)
    figs = write_figures(summary, out_dir)
    lines = ["", "acceptance criteria"]
    for name, rec in summary["criteria"].items():
        lines.append(f"  [{'PASS' if rec['pass'] else 'FAIL'}] {name}: {rec['value']} "
                     f"(target {rec['target']})")
    lines.append("")
    lines.append(f"figures: {len(figs)} -> {os.path.join(out_dir, 'figures')}")
    lines.append(f"elapsed {time.time() - t0:.1f} s")
    print("\n".join(lines))
    with open(os.path.join(out_dir, "figures.json"), "w", encoding="utf-8") as fh:
        json.dump([{"name": f["name"], "caption": f["caption"]} for f in figs], fh, indent=1)
    return 0 if all(r["pass"] for r in summary["criteria"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))