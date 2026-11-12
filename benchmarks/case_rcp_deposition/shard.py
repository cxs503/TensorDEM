"""Run one case of ``plan.json`` by index (for the cluster shard template).

Usage::

    python3 shard.py <index> [offset]

``index`` is taken modulo the plan length and ``offset`` (default 0) is added,
so a second cluster wave can address the tail of the plan with its own
``{i}``-template without touching the first wave's indices.  The bound is read
from the file: the plan grew from 12 screening cases to the N=10000 production
+ stiffness + timestep probe, so a hard-coded modulus would silently drop cases.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    plan = json.load(open(os.path.join(HERE, "plan.json")))["cases"]
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    idx = (int(sys.argv[1]) + offset) % len(plan)
    case = plan[idx]
    cmd = [sys.executable, os.path.join(HERE, "run_case.py"),
           "--n", str(case["n"]), "--mu", str(case["mu"]),
           "--seed", str(case["seed"]), "--kn", str(case["kn"]),
           "--out", os.path.join(HERE, "runs")]
    if "dt_frac" in case:
        cmd += ["--dt-frac", repr(case["dt_frac"])]
    print("RUN", idx, case, flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())