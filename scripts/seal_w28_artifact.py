#!/usr/bin/env python3
"""Seal the W28 artifact: record sha256 / git-blob hashes of every delivered file.

W28 delivery discipline (lesson from W27): the artifact must live on the main
work tree at the declared path and carry its own content hashes, plus the source
ref if it was recovered from an unmerged branch.  This script

  1. adds ``artifacts_manifest_path`` to the artifact (path only, so there is no
     hash circularity),
  2. writes ``manifest.json`` next to it with sha256 and ``git hash-object`` blob
     hashes of the artifact, the figures, the generator and the recovered W27
     sources.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

ART_DIR = os.path.join("papers", "artifacts", "w28", "ice-ridge-keel-grounding")
ART = os.path.join(ART_DIR, "dual_convergence.json")
MANIFEST = os.path.join(ART_DIR, "manifest.json")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob(path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "hash-object", path], text=True).strip()
    except Exception:  # pragma: no cover
        return None


def main() -> int:
    with open(ART) as fh:
        art = json.load(fh)
    art.setdefault("provenance", {})["artifacts_manifest_path"] = MANIFEST
    with open(ART, "w") as fh:
        json.dump(art, fh, indent=1)

    files = [ART]
    files += [os.path.join(ART_DIR, n) for n in sorted(os.listdir(ART_DIR))
              if n.endswith(".png")]
    files += ["scripts/run_w28_dual_convergence.py",
              "scripts/figures_w28_dual_convergence.py",
              "src/tensordem/keel_grounding_coupled.py",
              "src/tensordem/keel_foundation_dem.py",
              "tests/test_w28_dual_convergence.py",
              "docs/ice_ridge_keel_grounding_dual_convergence.md"]
    entries = []
    for f in files:
        if not os.path.exists(f):
            entries.append({"path": f, "exists": False})
            continue
        entries.append({"path": f, "exists": True, "sha256": sha256(f),
                        "git_blob_sha1": blob(f), "bytes": os.path.getsize(f)})
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                         text=True).strip()
    except Exception:  # pragma: no cover
        head, branch = None, None
    manifest = {
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": "TensorDEM", "mission_id": 19906, "branch": branch, "commit": head,
        "declared_artifact_path": ART,
        "delivered_files": entries,
        "recovered_from_ref": art["provenance"].get("recovered_from_ref"),
        "check": ("the delivered artifact is the one on the main work tree at the declared "
                  "path; the recovered W27 sources come from agent/m18963 (commit 594c334e) "
                  "because the W27 mission branch was never merged"),
    }
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps({"manifest": MANIFEST, "n_files": len(entries)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
