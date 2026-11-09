"""IN100-N07 acceptance run: ridge geometry, rubble strength, ship penetration.

Writes ``artifacts/ridge_mechanics/acceptance.json`` and the report figures.
Run with ``PYTHONPATH=src python3 scripts/ridge_mechanics_acceptance.py``.
"""
from __future__ import annotations

import json
import os
import time

from tensordem.ice_ridge_mechanics import AcceptanceConfig, run_acceptance

OUT = os.path.join("artifacts", "ridge_mechanics")


def main() -> dict:
    config = AcceptanceConfig()
    started = time.time()
    results = run_acceptance(config, verbose=True)
    results["runtime_seconds"] = time.time() - started
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "acceptance.json"), "w") as handle:
        json.dump(results, handle, indent=2, default=float)
    summary = results["summary"]
    print("passed:", summary["passed"])
    print("failed:", summary["failed"])
    for key in ("F1", "F2", "F3", "F4", "F5", "F6"):
        print(key, json.dumps(results[key], default=float)[:300])
    print("runtime %.1f s" % results["runtime_seconds"])
    return results


if __name__ == "__main__":
    main()
