"""Decode log-transported case payloads back into ``runs/`` JSON files (m20873).

Companion to :mod:`scripts.emit_column_collapse_payload_m20873`.  The compute
gate returns job logs but not the node worktree (see
``docs/AUDIT_COLUMN_COLLAPSE_REFINEMENT_FETCH_m20559.md``), so the payload is
shipped inside the log and reconstructed here, with the sha256 recorded at emit
time verified against the reconstructed bytes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emit_column_collapse_payload_m20873 import decode_payload  # noqa: E402

OUT_DIR = os.path.join("runs", "case_column_collapse_time_m18261", "cases")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: decode_..._m20873.py <logfile> [dest_dir]", file=sys.stderr)
        return 2
    with open(argv[1], errors="replace") as fh:
        text = fh.read()
    dest = argv[2] if len(argv) > 2 else OUT_DIR
    os.makedirs(dest, exist_ok=True)
    payloads = decode_payload(text)
    if not payloads:
        print("no payload block found", file=sys.stderr)
        return 3
    manifest = {}
    for name, (sha, raw) in payloads.items():
        path = os.path.join(dest, name)
        with open(path, "wb") as fh:
            fh.write(raw)
        manifest[name] = {"sha256": sha, "bytes": len(raw)}
        case = json.loads(raw)
        print(f"{name:<16} N={case['n_particles']:<6} runout={case['runout_ratio_p995']:.4f} "
              f"stop={case['stop_reason']:<10} wall={case['wall_clock_s']:.0f}s sha={sha[:12]}")
    with open(os.path.join(dest, "payload_manifest_m20873.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))