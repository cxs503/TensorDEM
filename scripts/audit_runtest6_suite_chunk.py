#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 (m14498): one chunk of the full pytest suite.

Why chunked: the control-plane shell in this sandbox reaps any command after
~120 s, so a single full-suite invocation is impossible here without hogging the
control plane (which the task forbids).  The suite is therefore split into
``--nchunks`` deterministic slices of the test-file list, each slice run in its
own pytest process with ``-n`` workers, and the per-slice JSON results are summed
by ``--summary``.

The two files that fail *collection* on this checkout (module/package name
shadowing, see docs/AUDIT_RUNTEST-6_run_evidence.md) are excluded from the slices
and reported separately: the documented command ``python -m pytest -q`` aborts on
them, which is itself an audit finding.

Usage:
    python3 scripts/audit_runtest6_suite_chunk.py --chunk 0 --nchunks 12
    python3 scripts/audit_runtest6_suite_chunk.py --summary
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "suite_chunks"
BROKEN = ("tests/test_freeze_bond.py", "tests/test_ice_structure_contact.py")

SUMMARY_RE = re.compile(
    r"(?:(\d+) failed)?[, ]*(?:(\d+) passed)?[, ]*(?:(\d+) skipped)?"
    r"[, ]*(?:(\d+) error[s]?)?[, ]*(?:(\d+) xfailed)?[, ]*(?:(\d+) xpassed)?"
)


def files() -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py")
                  if str(p.relative_to(ROOT)) not in BROKEN)


def parse(output: str) -> dict:
    tail = [ln for ln in output.strip().splitlines() if ln.strip()][-4:]
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "xfailed": 0, "xpassed": 0}
    for ln in tail:
        for key in counts:
            m = re.search(rf"(\d+) {key}(?:s)?\b", ln)
            if m:
                counts[key] = max(counts[key], int(m.group(1)))
        m = re.search(r"in ([\d.]+)s", ln)
        if m:
            counts["seconds"] = float(m.group(1))
    counts["tail"] = tail
    return counts


def run_chunk(index: int, nchunks: int, workers: int, timeout: int) -> dict:
    all_files = files()
    n = len(all_files)
    lo = index * n // nchunks
    hi = (index + 1) * n // nchunks
    subset = all_files[lo:hi]
    cmd = [sys.executable, "-m", "pytest", *subset, "-q", "-p", "no:cacheprovider",
           "--continue-on-collection-errors", "--tb=no", "-rf", f"-n{workers}"]
    try:
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out, rc, timed_out = res.stdout + res.stderr, res.returncode, False
    except subprocess.TimeoutExpired as exc:
        def _txt(v):
            if v is None:
                return ""
            return v.decode(errors="replace") if isinstance(v, bytes) else v
        out = _txt(exc.stdout) + _txt(exc.stderr)
        rc, timed_out = -1, True
    counts = parse(out)
    counts.update(chunk=index, nchunks=nchunks, n_files=len(subset), rc=rc,
                  timed_out=timed_out, first=subset[0] if subset else None,
                  last=subset[-1] if subset else None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"chunk_{index:02d}.json").write_text(json.dumps(counts, indent=1) + "\n")
    (OUT / f"chunk_{index:02d}.log").write_text(out)
    print(json.dumps({k: v for k, v in counts.items() if k != "tail"}, ensure_ascii=False))
    if counts["tail"]:
        print("tail:", counts["tail"][-1][:200])
    return counts


def summary() -> int:
    parts = sorted(OUT.glob("chunk_*.json"))
    if not parts:
        print("no chunks yet")
        return 1
    tot = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "xfailed": 0, "xpassed": 0,
           "seconds": 0.0, "n_files": 0}
    failed_files: list[str] = []
    for p in parts:
        d = json.loads(p.read_text())
        for k in ("passed", "failed", "skipped", "error", "xfailed", "xpassed"):
            tot[k] += d.get(k, 0)
        tot["seconds"] += d.get("seconds", 0.0)
        tot["n_files"] += d.get("n_files", 0)
        if d.get("timed_out"):
            failed_files.append(f"chunk_{d['chunk']} TIMEOUT ({d['first']}..{d['last']})")
        if d.get("rc", 0) != 0:
            log = (OUT / f"chunk_{d['chunk']:02d}.log").read_text()
            failed_files += [ln.strip() for ln in log.splitlines()
                             if ln.startswith(("FAILED", "ERROR"))]
    tot["chunks"] = len(parts)
    tot["nonpassing"] = failed_files
    tot["collection_blocked_files"] = list(BROKEN)
    tot["note"] = ("chunked execution: each chunk is an independent pytest process, so "
                   "suite-level fixtures/state are not shared across chunks")
    tot["level"] = 3 if tot["failed"] == 0 and tot["error"] == 0 else 2
    (OUT / "pytest_summary.json").write_text(json.dumps(tot, indent=1) + "\n")
    print(json.dumps(tot, indent=1, ensure_ascii=False)[:2000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--nchunks", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=112)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if a.chunk is None:
        ap.error("--chunk or --summary required")
    counts = run_chunk(a.chunk, a.nchunks, a.workers, a.timeout)
    return 0 if counts["rc"] == 0 else 0  # keep going; the driver reads the JSON


if __name__ == "__main__":
    raise SystemExit(main())