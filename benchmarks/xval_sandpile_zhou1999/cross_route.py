#!/usr/bin/env python3
"""Cross-route leg of XVAL-DEM-SANDPILE-REPOSE-CALIBRATION-ANCHOR.

The repository has *two* independent drivers that produce a macroscopic angle of
repose from the same contact physics:

* route A -- ``tensordem.protocols.sandpile`` (``sandpile-v1``): nozzle pour onto
  a flat frictional base plate, free surface measured on annulus sectors.
* route B -- ``tensordem.repose_calibration``: a hexagonally packed cylinder is
  filled, the wall is lifted and the column collapses (a *lifted-box/lifted-
  cylinder* boundary condition -- the method family of the Zhou et al. (1999)
  anchor), measured on the settled heap.

Same ``(mu_s, mu_r, material)`` through both drivers with *only* the boundary
condition changed is the discriminating comparison for the XVAL verdict: if the
two routes disagree, the deviation from the literature band cannot be blamed on
the contact law alone.

Route-B case parameters follow the ones documented for the calibration surface
in ``docs/repose_angle_calibration.md`` (r = 8 mm, R_c = 35 mm, H = 140 mm,
k_n = 6e4 N/m, e = 0.05, jitter = 2e-3 r).  The documented table itself is
``mu_r = 0.20``; the run below repeats that node (as a reproduction check
against the documented angle, parsed from the markdown, not typed here) and adds
the ``mu_r = 0.05`` node, which is the frozen prior of route A.

Usage::

    python benchmarks/xval_sandpile_zhou1999/cross_route.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

OUT_DIR = os.path.join(REPO, "artifacts", "XVAL-SANDPILE-ZHOU1999")
CALIB_DOC = os.path.join(REPO, "docs", "repose_angle_calibration.md")


def documented_node(mu: float) -> dict[str, object] | None:
    """Parse the documented calibration table row for ``mu`` (zero hand-copy)."""
    with open(CALIB_DOC, encoding="utf-8") as fh:
        text = fh.read()
    pat = re.compile(r"^\|\s*" + re.escape(f"{mu:.2f}") + r"\s*\|\s*([0-9.]+)\s*\|")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return {"mu": mu, "theta_r_deg": float(cells[1]),
                    "raw": line.strip()}
    return None


def main() -> int:
    import torch
    torch.set_num_threads(1)
    from tensordem.repose_calibration import ReposeCase, simulate

    os.makedirs(OUT_DIR, exist_ok=True)
    out: dict[str, object] = {
        "doc": {"path": os.path.relpath(CALIB_DOC, REPO),
                "documented_mu_r": 0.20},
        "cases": {},
        "documented_node_mu_0.50_mu_r_0.20": documented_node(0.50),
    }
    base = dict(mu=0.50, radius=0.008, density=2500.0, cylinder_radius=0.035,
                fill_height=0.140, kn=6.0e4, kt_ratio=0.5, restitution=0.05,
                jitter=0.002, packing_gap=0.02, hold_time=0.05, lift_time=0.15,
                settle_time=0.30, seed=4970)
    for mu_r in (0.20, 0.05):
        t0 = time.time()
        case = ReposeCase(mu_rolling=mu_r, **base)
        res = simulate(case)
        name = f"mu_r_{mu_r:.2f}"
        out["cases"][name] = {
            "mu": 0.50, "mu_r": mu_r,
            "theta_r_deg": res["theta_r_deg"], "r2": res["r2"],
            "steps": res["steps"], "dt": res["dt"],
            "n_particles": res["n_particles"],
            "max_speed": res["max_speed"], "mean_speed": res["mean_speed"],
            "shoulder_radius": res["shoulder_radius"],
            "toe_radius": res["toe_radius"], "apex_z": res["apex_z"],
            "wall_time_s": time.time() - t0,
            "case": {k: v for k, v in base.items() if k != "mu"} | {"mu": 0.50},
        }
        print(json.dumps({k: out["cases"][name][k] for k in
                          ("theta_r_deg", "r2", "steps", "wall_time_s")}))
    path = os.path.join(OUT_DIR, "cross_route.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())