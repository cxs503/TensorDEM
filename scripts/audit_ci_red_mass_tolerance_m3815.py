#!/usr/bin/env python3
"""Measure the float64 resolution of the LBM mass-conservation assertion.

``tests/test_threefield_lbm_transport.py::test_streaming_is_mass_conserving``
asserted ``|sum(stream(f)) - sum(f)| < 1e-12`` in absolute units.  The streaming
step is an *exact* permutation of the 19 discrete velocities followed by a
periodic shift, so every population is preserved bit for bit and the only error
is the rounding of the two reductions ``sum()``.

Two things make an absolute bound fragile:

1. the reduced quantity grows with the grid (``nx*ny*nz*19`` populations of mean
   1/2), so the achievable absolute error grows with it as well; and
2. the reduction order of ``Tensor.sum()`` depends on the number of OpenMP
   threads, so the same assertion passes or fails on the same input depending on
   how busy the machine is.

This audit measures both effects and reports, per grid size, the measured
``|delta M|`` and the *unit in the last place* (ulp) of the total mass, which is
the smallest error a float64 reduction can possibly achieve.  A tolerance below
one ulp is unsatisfiable, and the report states where ``1e-12`` sits relative to
that floor.

Usage::

    python scripts/audit_ci_red_mass_tolerance_m3815.py           # write artifacts
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.threefield.lbm_transport import periodic_stream  # noqa: E402
from tensordem.threefield.wake_lbm import LBMConfig, WakeLBM  # noqa: E402

OUT = ROOT / "artifacts" / "ci_red_mass_tolerance_m3815"
GRIDS = (12, 16, 20, 24, 32)
ABS_BOUND = 1e-12          # the bound the old assertion used
REL_BOUND = 1e-13          # the bound the test now uses (relative to the mass)


def make(n: int) -> WakeLBM:
    return WakeLBM(
        LBMConfig(nx=n, ny=n, nz=n, tau0=0.6, smagorinsky_cs=0.1,
                  disk_radius=1.0, disk_x=n - 4, disk_thickness=1)
    )


def mass_defect(n: int, threads: int) -> tuple[float, float]:
    """Return ``(|delta M|, M)`` for one grid, with ``threads`` OMP threads."""
    torch.set_num_threads(threads)
    inst = make(n)
    torch.manual_seed(1)
    f = torch.rand((inst.nx, inst.ny, inst.nz, 19), dtype=torch.float64)
    total = float(f.sum())
    streamed = float(periodic_stream(inst, f).sum())
    return abs(streamed - total), total


def ulp(x: float) -> float:
    """Spacing of the float64 grid at ``|x|``: the irreducible reduction error."""
    return math.ulp(x) if hasattr(math, "ulp") else math.ulp(x)


def main() -> int:
    rows = []
    hardware_threads = max(1, os.cpu_count() or 1)
    for n in GRIDS:
        total = 0.0
        for threads in (1, hardware_threads):
            defect, total = mass_defect(n, threads)
            rows.append(
                {
                    "n": n,
                    "populations": n * n * n * 19,
                    "threads": threads,
                    "total_mass": total,
                    "abs_defect": defect,
                    "rel_defect": defect / total,
                    "ulp_total": ulp(total),
                    "abs_defect_in_ulp": defect / ulp(total),
                    "passes_abs_bound": defect < ABS_BOUND,
                    "passes_rel_bound": defect <= REL_BOUND * total,
                }
            )
            print(
                f"n={n:2d} threads={threads:3d} M={total:12.4f} "
                f"|dM|={defect:9.3e} ({defect / ulp(total):6.2f} ulp) "
                f"abs<1e-12:{defect < ABS_BOUND!s:5s} "
                f"rel<=1e-13:{defect <= REL_BOUND * total!s:5s}"
            )
    torch.set_num_threads(hardware_threads)

    broken = [r for r in rows if not r["passes_abs_bound"]]
    report = {
        "schema": "tensordem.ci-red-mass-tolerance/1",
        "generated_by": "scripts/audit_ci_red_mass_tolerance_m3815.py",
        "grids": list(GRIDS),
        "hardware_threads": hardware_threads,
        "abs_bound": ABS_BOUND,
        "rel_bound": REL_BOUND,
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "old_absolute_bound_violated": len(broken),
            "old_bound_violated_at_1_thread": sum(
                1 for r in rows if r["threads"] == 1 and not r["passes_abs_bound"]
            ),
            "new_relative_bound_violated": sum(
                1 for r in rows if not r["passes_rel_bound"]
            ),
            "worst_defect_in_ulp": max(r["abs_defect_in_ulp"] for r in rows),
            "abs_bound_in_ulp_of_largest_total": ABS_BOUND
            / max(r["ulp_total"] for r in rows),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mass_tolerance_audit.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=1))

    # --- figure: measured defect vs grid, with the two bounds and the ulp floor --
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for threads, style in ((1, "o-"), (hardware_threads, "s--")):
        sel = [r for r in rows if r["threads"] == threads]
        ax.plot([r["n"] for r in sel], [abs(r["abs_defect"]) or 1e-19 for r in sel],
                style, label=f"measured |ΔM|, {threads} OMP thread(s)")
    ulps = [r["ulp_total"] for r in rows if r["threads"] == 1]
    ax.plot(GRIDS, ulps, "k:", label="ulp(M): irreducible reduction error")
    ax.plot(GRIDS, [ABS_BOUND] * len(GRIDS), "r-",
            label="old assertion: |ΔM| < 1e-12 (absolute)")
    ax.plot(GRIDS, [REL_BOUND * r["total_mass"] for r in rows if r["threads"] == 1], "g-",
            label="new assertion: |ΔM| ≤ 1e-13·M")
    ax.set_yscale("log")
    ax.set_xlabel("grid edge n (n³ cells, 19 populations)")
    ax.set_ylabel("mass defect |sum(stream f) − sum f|  [kg]")
    ax.set_title("Why an absolute 1e-12 mass bound is unsatisfiable for a growing grid")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "mass_tolerance_vs_grid.png")
    print(f"wrote {OUT / 'mass_tolerance_vs_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())