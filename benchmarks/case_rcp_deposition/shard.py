"""Run one case of ``plan.json`` by index (for the cluster shard template)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    idx = int(sys.argv[1]) % 12
    plan = json.load(open(os.path.join(HERE, "plan.json")))["cases"]
    case = plan[idx]
    cmd = [sys.executable, os.path.join(HERE, "run_case.py"),
           "--n", str(case["n"]), "--mu", str(case["mu"]),
           "--seed", str(case["seed"]), "--kn", str(case["kn"]),
           "--out", os.path.join(HERE, "runs")]
    print("RUN", idx, case, flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
