#!/usr/bin/env python3
"""W27 · ISO 19906:2019 A.8 anchor register for the ice-ridge keel grounding case.

What this script does (and refuses to do)
-----------------------------------------
It performs the *acquisition probe* for the verbatim standard text and writes
``benchmarks/anchors/ice-ridge-keel-grounding.json``.  It never writes a physical
constant taken from memory or from a secondary source: if the clause text is not
present in this environment the anchor is registered ``WITHHELD`` and only the
operator layer (dimensional closure, log-log slope, ratio/composition
identities, ``n -> 0`` limit, inversion round-trip) is machine checked -- those
identities hold for arbitrary ``(p0, A0, n)`` and therefore need no standard
value.

Secondary in-tree citations of ISO 19906 A.8 are registered by path + sha256 +
line position only; their numbers are *not* extracted, because a transcription
that cannot be checked against the clause text must not become an anchor.

Usage
-----
    PYTHONPATH=src python scripts/gen_w27_ice_ridge_keel_anchor.py [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tensordem.ridge_keel_grounding import pressure_area_operator_checks

OUT_DEFAULT = os.path.join("benchmarks", "anchors", "ice-ridge-keel-grounding.json")
STANDARD_ID = "ISO 19906:2019"
CLAUSE = "A.8"
SEARCH_ROOTS = ("/root", "/opt", "/srv", "/data", "/mnt", "/usr/share/doc", "/tmp")
PRUNE = (".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages",
         "dist-packages", ".cache", ".mypy_cache", ".pytest_cache", "evo1")
DOC_SUFFIX = {".pdf", ".html", ".htm", ".docx", ".doc", ".txt", ".md", ".epub",
              ".zip", ".djvu", ".xps", ".rtf", ".odt"}
SECONDARY_SOURCES = (
    "src/tensordem/hull_collision/indentation.py",
    "src/tensordem/hull_collision/scenario.py",
    "src/tensordem/ice_rubble_pileup_vertical.py",
    "src/tensordem/ice_crushing_vertical_structure.py",
    "src/tensordem/ice_pressure_stats.py",
    "src/tensordem/ice_ridge_keel_hull.py",
    "benchmarks/ice_crushing_iso19906/REPORT.md",
    "benchmarks/anchors/ice-ridge-pipeline-seabed-bearing.json",
)
NEEDLES = ("19906", "A.8", "pressure-area", "pressure–area", "pressure_area")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_documents() -> dict:
    """Filesystem / toolchain / network probe for the licensed standard text."""
    name_hits: list[str] = []
    text_hits: list[dict] = []
    scanned = 0
    t0 = time.time()
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if depth > 6:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in PRUNE and "evo1" not in d]
            for fn in filenames:
                low = fn.lower()
                if "19906" in low:
                    name_hits.append(os.path.join(dirpath, fn))
                suffix = os.path.splitext(fn)[1].lower()
                if suffix not in DOC_SUFFIX or scanned > 40000:
                    continue
                scanned += 1
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > 40e6:
                        continue
                    with open(p, "rb") as fh:
                        blob = fh.read()
                except OSError:
                    continue
                if b"19906" in blob:
                    text_hits.append({"path": p, "sha256": _sha256(p),
                                      "size_bytes": len(blob)})
            if time.time() - t0 > 120:
                break
        if time.time() - t0 > 120:
            break
    tools = {name: shutil.which(name) for name in
             ("pdftotext", "pdfinfo", "mutool", "antiword", "unzip")}
    network = {}
    for url in ("https://www.iso.org/standard/65760.html",
                "https://www.iso.org/obp/ui/#iso:std:iso:19906:ed-2:v1:en"):
        try:
            out = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
                                  "--max-time", "8", url],
                                 capture_output=True, text=True, timeout=15)
            network[url] = {"http_code": out.stdout.strip(), "returncode": out.returncode}
        except Exception as exc:  # pragma: no cover
            network[url] = {"error": str(exc)}
    licensed = [h for h in text_hits
                if os.path.splitext(h["path"])[1].lower() in {".pdf", ".docx", ".doc", ".epub"}]
    return {
        "search_roots": list(SEARCH_ROOTS),
        "pruned_dirs": list(PRUNE),
        "files_text_scanned": scanned,
        "name_hits": name_hits[:20],
        "content_hits": text_hits[:20],
        "licensed_container_hits": licensed,
        "extraction_tools": tools,
        "network": network,
        "elapsed_s": round(time.time() - t0, 1),
        "verdict": ("OBTAINED" if licensed else "WITHHELD"),
    }


def secondary_citations() -> list[dict]:
    """In-tree citations of ISO 19906 A.8: path + sha256 + line positions only."""
    out = []
    for rel in SECONDARY_SOURCES:
        if not os.path.exists(rel):
            continue
        hits = []
        with open(rel, "r", errors="replace") as fh:
            for lineno, line in enumerate(fh, start=1):
                if any(nd in line for nd in NEEDLES):
                    hits.append({"line": lineno,
                                 "text": line.strip()[:200],
                                 "counts_as_verbatim_anchor": False})
        out.append({"path": rel, "sha256": _sha256(rel), "n_hits": len(hits),
                    "hits": hits[:12],
                    "kind": "secondary_code_or_doc_citation",
                    "numbers_extracted": False,
                    "why_rejected": ("secondary transcription: it cannot be checked "
                                     "against the clause text, so it is registered but "
                                     "never used as an anchor value")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    probe = probe_documents()
    secondary = secondary_citations()
    op = pressure_area_operator_checks()
    status = probe["verdict"] if probe["verdict"] in ("OBTAINED",) else "WITHHELD"
    anchor = {
        "schema": "tensordem.anchor.v1",
        "anchor_id": "ISO-19906-2019-A8-PRESSURE-AREA",
        "mission": "paper-exp-ice-ridge-keel-grounding-1-rework1",
        "mission_id": 18963,
        "repo": "TensorDEM",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "generator": "scripts/gen_w27_ice_ridge_keel_anchor.py",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
        "standard": {"id": STANDARD_ID, "clause": CLAUSE,
                     "subject": "global ice action: pressure-area law p = p0 (A/A0)^(-n) "
                                "and the areal exponent n; design pressure values",
                     "title_from_secondary_only": "Arctic offshore structures"},
        "acquisition_status": status,
        "status": status,
        "verbatim_quotes": [],
        "document_sha256": None,
        "quote_position": None,
        "withheld_reason": (
            "The licensed clause text of ISO 19906:2019 A.8 is not present in this "
            "environment. The probe below searched every accessible document container by "
            "name and by content (bounded, with the EVO-1 blind-review paths pruned), "
            "recorded the available extraction toolchain and the publisher endpoints, and "
            "found no copy that could be quoted. Mission discipline forbids filling any "
            "number from memory or from a secondary transcription, so the anchor is "
            "registered WITHHELD and only the operator layer is machine checked."
        ),
        "acquisition_probe": probe,
        "operator_layer": {
            "module": "src/tensordem/ridge_keel_grounding.py",
            "module_sha256": _sha256("src/tensordem/ridge_keel_grounding.py"),
            "functions": ["pressure_area_pressure", "pressure_area_force",
                          "pressure_area_exponent_from_pair",
                          "pressure_area_operator_checks"],
            "checks": op,
            "note": ("the checks run on arbitrary (p0, A0, n) = (1, 1, 0.5) placeholders "
                     "declared as dimensionless test values, never as standard values; "
                     "every identity holds for all (p0, A0, n), so the same function "
                     "re-runs unchanged on a transcribed anchor"),
        },
        "secondary_sources": secondary,
        "gate": {
            "readable_by_next_anchor_gate": True,
            "required_status_values": ["TRANSCRIBED", "VERIFIED", "OBTAINED"],
            "current_status_meets_gate": status in ("TRANSCRIBED", "VERIFIED", "OBTAINED"),
            "blocking_items": ([] if status != "WITHHELD" else
                               ["verbatim clause text of ISO 19906:2019 A.8",
                                "page/table number of the quoted clause",
                                "sha256 of the source document"]),
            "operator_layer_verified": bool(op["all_pass"]),
        },
        "downstream_use": {
            "artifact": "papers/artifacts/w27/ice-ridge-keel-grounding/ridge_keel_anchor.json",
            "how_used": ("no computed quantity in the W27 artifact depends on an ISO 19906 "
                         "value; the A.8 law enters only through its operator layer and "
                         "through the applicability statement (continuous ice, not rubble)"),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(anchor, fh, indent=1)
    print(json.dumps({"out": args.out, "status": status,
                      "operator_all_pass": op["all_pass"],
                      "n_secondary_sources": len(secondary),
                      "name_hits": len(probe["name_hits"]),
                      "content_hits": len(probe["content_hits"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())