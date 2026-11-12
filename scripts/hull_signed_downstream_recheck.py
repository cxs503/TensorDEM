"""m16872: re-run the merged hull-signed *downstream* metrics after the fix.

``docs/hull_mesh_contact_signed_distance.md`` §5 (dynamic floe impact) and
``metrics.json.escape_map`` were produced by ``scripts/hull_signed_audit.py``
before the deep-penetration ordering fix, so their numbers can move.  This
runner imports those two functions and writes a small, diffable artefact
instead of regenerating (and overwriting) the merged document.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hull_signed_audit",
                                              ROOT / "scripts/hull_signed_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main() -> int:
    out = {}
    out["dynamic_impact"] = audit.dynamic_impact()
    out["escape_map"] = audit.escape_runs(480.0, 0.5, v_values=(1.0, 2.0, 3.0, 5.0),
                                          k_values=(2e3, 1e4, 1e5, 1e6))
    lost = {(r["k"], r["v"], r["closed"]): r["lost"] for r in out["escape_map"]}
    out["escape_summary"] = {
        "n": len(lost),
        "lost_baseline": sum(1 for k, c in lost.items() if not k[2] and c),
        "lost_signed": sum(1 for k, c in lost.items() if k[2] and c),
    }
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "runs/m16872/downstream_after.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"dynamic_impact": out["dynamic_impact"],
                      "escape_summary": out["escape_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())