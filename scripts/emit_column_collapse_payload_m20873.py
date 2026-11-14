"""Emit the two gap cases as a compact, log-transportable payload (m20873).

Why this exists
---------------
``docs/AUDIT_COLUMN_COLLAPSE_REFINEMENT_FETCH_m20559.md`` records that a cluster
job writing its results into ``runs/`` (or ``artifacts/``) cannot be retrieved:
``fetch`` returns the *committed* repo state, not the node worktree.  So the two
refinement-gap cases declared but never run by m18261 (``R_a1_g16``,
``R_a3_g12``) are re-run here and their results are **encoded into the job log**
(base64 of gzip of the case JSON), which the scheduler does return.  The
decoder is :func:`decode_payload` / ``scripts/decode_column_collapse_payload_m20873.py``.

Usage (on the compute node, repo checkout)::

    python3 scripts/emit_column_collapse_payload_m20873.py R_a1_g16 R_a3_g12
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import sys

CASE_DIR = os.path.join("runs", "case_column_collapse_time_m18261", "cases")
BEGIN = "===BEGIN-PAYLOAD"
END = "===END-PAYLOAD"


def encode(path: str) -> str:
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    return f"{BEGIN} {os.path.basename(path)} {sha} {len(raw)} {len(blob)}\n{blob}\n{END}"


def decode_payload(text: str) -> dict[str, tuple[str, bytes]]:
    """Parse every payload block in ``text`` -> {filename: (sha256, raw bytes)}."""
    out: dict[str, tuple[str, bytes]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith(BEGIN):
            _, name, sha, nbytes, _nblob = lines[i].split()
            raw = gzip.decompress(base64.b64decode(lines[i + 1]))
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError(f"sha256 mismatch for {name}")
            if int(nbytes) != len(raw):
                raise ValueError(f"length mismatch for {name}")
            out[name] = (sha, raw)
            i += 3
        else:
            i += 1
    return out


def main(argv: list[str]) -> int:
    ids = argv[1:] or ["R_a1_g16", "R_a3_g12"]
    missing = []
    for cid in ids:
        path = os.path.join(CASE_DIR, f"{cid}.json")
        if not os.path.exists(path):
            missing.append(cid)
            continue
        print(encode(path), flush=True)
    if missing:
        print(f"MISSING {missing} (case integration did not produce JSON)", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))