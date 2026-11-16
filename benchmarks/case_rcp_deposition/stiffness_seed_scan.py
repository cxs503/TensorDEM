#!/usr/bin/env python3
"""Seed scan of the RCP stiffness ladder (one case per shard index).

Why this script exists
----------------------
The hard-sphere (``k_n -> inf``) limit of the deposition packing fraction is
obtained from a *stiffness ladder*: ``phi(k_n) = phi_inf + A k_n^-p`` with
``p`` taken from the two **outermost** deltas of the ladder
(:func:`tensordem.deposition_packing.stiffness_limit`).  Until now only
``k_n = 4000`` (three seeds) and ``k_n = 64000`` (three seeds for ``mu = 0``,
one seed for ``mu = 0.1``) had seed replicates; the two stiff levels
``k_n = 16000`` and ``k_n = 256000`` -- the levels that actually set the slope
``p`` and therefore the intercept -- were **single-seed points**.

This driver runs exactly the missing replicates (``CASES`` below), so the ladder
can be re-adjudicated with a seed standard error at *every* level, and the
extrapolated limit can be given a confidence interval instead of a point value.

Every case is idempotent: an existing JSON with a finite ``phi_bulk`` is
skipped, so re-running a shard costs nothing.

Usage
-----
::

    PYTHONPATH=src python3 benchmarks/case_rcp_deposition/stiffness_seed_scan.py 0 10
    PYTHONPATH=src python3 benchmarks/case_rcp_deposition/stiffness_seed_scan.py --list
    PYTHONPATH=src python3 benchmarks/case_rcp_deposition/stiffness_seed_scan.py --dry-run 7
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")

N = 10000
E = 0.1
DT = 1.0 / 20.0

# (mu, kn, seed) -- the missing replicates of the stiffness ladder.
CASES = [
    (0.0, 16000.0, 2),
    (0.0, 16000.0, 3),
    (0.1, 16000.0, 2),
    (0.1, 16000.0, 3),
    (0.0, 256000.0, 2),
    (0.0, 256000.0, 3),
    (0.1, 256000.0, 2),
    (0.1, 256000.0, 3),
    (0.1, 64000.0, 2),
    (0.1, 64000.0, 3),
]


def tag(mu: float, kn: float, seed: int, dt_frac: float = DT) -> str:
    out = f"deposition_n{N}_mu{mu:g}_kn{kn:g}_e{E:g}_seed{seed}"
    if dt_frac != 1.0 / 20.0:
        out += f"_dtf{dt_frac:g}"
    return out


def done(path: str) -> bool:
    """True when the run already produced a numeric ``phi_bulk``."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            d = json.load(fh)
        phi = d["metrics"]["phi_bulk"]
    except Exception:
        return False
    return phi is not None and phi == phi


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--list":
        for i, c in enumerate(CASES):
            print(i, c)
        return 0
    dry = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    idx = int(argv[0]) if argv else 0
    idx %= len(CASES)
    mu, kn, seed = CASES[idx]
    path = os.path.join(RUNS, tag(mu, kn, seed) + ".json")
    print(f"RUN idx={idx} mu={mu} kn={kn} seed={seed} -> {path}", flush=True)
    if dry:
        return 0
    if done(path):
        print(f"SKIP (already complete): {path}", flush=True)
        return 0
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, HERE)
    import run_case
    return run_case.main(["--n", str(N), "--mu", str(mu), "--seed", str(seed),
                          "--kn", str(kn), "--restitution", str(E),
                          "--out", RUNS])


if __name__ == "__main__":
    raise SystemExit(main())