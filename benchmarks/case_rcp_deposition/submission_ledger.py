"""Gate a submission ledger against the artefacts that actually came back.

Why this module exists
----------------------
The RCP deposition campaign is driven from a per-mission ledger (``plan_m*.json``:
one row per cluster job, saying which ``(mu, k_n, seed, dt)`` case it is meant to
produce).  Twice now the *ledger* has disagreed with the *disk*:

* m22558 / m22824 shipped documents whose tables carried rows marked
  ``queued（未回件）`` -- a human read the queue and wrote prose, nothing checked
  the artefact;
* on 2026-11-16 the compute gate reported six jobs as ``succeeded`` (rc=0) while
  every one of them had died at the staging step (``git fetch`` to GitHub timed
  out) and produced **no file at all** (``cj4410``-``cj4415``).  A job that never
  started is indistinguishable from a job that finished if the only signal you
  look at is the queue state.

Both are the same failure: a *claim* ("this run exists") was never bound to the
*evidence* (the ``.json``/``.npz`` on disk).  This module performs that binding
mechanically.  It re-simulates nothing; it reads the ledger, reads the run JSONs,
and decides -- per row -- whether the artefact is there, whether its *spec*
matches what the ledger claims, and whether the run actually settled.

Verdict vocabulary (per row)
----------------------------
=================  ===========================================================
``PRESENT``        artefact exists, settled, spec matches the tag, digest known
``MISSING``        no artefact for that tag in any searched directory
``UNSETTLED``      artefact exists but ``settled`` is false (state not reached)
``MISMATCH``       artefact exists but its spec/seed/dt contradicts its own tag
``UNPARSED``       the row has no tag this module can read -- never "present"
=================  ===========================================================

``complete`` is true only when every row is ``PRESENT`` and there is at least one
row: an empty ledger is not a finished campaign.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")

#: ``deposition_n<N>_mu<MU>_kn<KN>_e<E>_seed<S>[_dtf<F>]``
TAG_RE = re.compile(
    r"^deposition_n(?P<n>\d+)_mu(?P<mu>-?[\d.]+)_kn(?P<kn>[\d.]+)"
    r"_e(?P<e>[\d.]+)_seed(?P<seed>\d+)"
    r"(?:_dtf(?P<dtf>[\d.]+))?$"
)

#: tolerance when comparing a tag's numbers with the run's recorded spec
TOL_REL = 1e-9
DEFAULT_DT_FRAC = 1.0 / 20.0

PRESENT, MISSING, UNSETTLED, MISMATCH, UNPARSED = (
    "PRESENT", "MISSING", "UNSETTLED", "MISMATCH", "UNPARSED")

__all__ = ["parse_tag", "load_ledger", "find_run", "check_row",
           "audit_ledger", "PRESENT", "MISSING", "UNSETTLED", "MISMATCH",
           "UNPARSED"]


def parse_tag(tag: str) -> dict | None:
    """Expected spec encoded by a run tag, or ``None`` if unreadable."""
    m = TAG_RE.match(tag or "")
    if not m:
        return None
    g = m.groupdict()
    return {
        "n_particles": int(g["n"]),
        "mu": float(g["mu"]),
        "kn": float(g["kn"]),
        "restitution": float(g["e"]),
        "seed": int(g["seed"]),
        "dt_frac": float(g["dtf"]) if g["dtf"] else DEFAULT_DT_FRAC,
    }


def load_ledger(path: str) -> list:
    """Rows of a ``plan_m*.json`` ledger (``{"jobs": [...]}``)."""
    with open(path) as fh:
        doc = json.load(fh)
    if isinstance(doc, list):
        return doc
    return doc.get("jobs", doc.get("rows", []))


def find_run(tag: str, runs_dirs) -> str | None:
    """First existing ``<tag>.json`` in ``runs_dirs`` (order given by caller)."""
    for d in runs_dirs:
        p = os.path.join(d, tag + ".json")
        if os.path.exists(p):
            return p
    return None


def _close(a, b) -> bool:
    return abs(float(a) - float(b)) <= TOL_REL * max(1.0, abs(float(a)),
                                                     abs(float(b)))


def check_row(row: dict, runs_dirs, require_settled: bool = True) -> dict:
    """Bind one ledger row to its artefact.  Never raises on bad ledger data."""
    tag = row.get("tag") or ""
    jid = row.get("jid") or ""
    out = {"jid": jid, "tag": tag, "why": row.get("why", ""),
           "path": None, "impl_sha256": None, "reasons": []}
    expected = parse_tag(tag)
    if expected is None:
        out["status"] = UNPARSED
        out["reasons"].append("ledger row has no parsable run tag")
        return out
    dirs = runs_dirs
    if row.get("dir"):
        # a row may pin the directory it must land in: a *reproduction* run
        # must never be satisfied by the archived run it is meant to reproduce
        pinned = [d for d in runs_dirs
                  if os.path.basename(os.path.normpath(d)) == row["dir"]]
        if not pinned:
            pinned = [os.path.join(HERE, row["dir"])]
        dirs = pinned
    path = find_run(tag, dirs)
    if path is None:
        out["status"] = MISSING
        out["reasons"].append("no artefact %s.json in %s"
                              % (tag, [os.path.basename(d) for d in dirs]))
        return out
    out["path"] = os.path.relpath(path, ROOT)
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:  # truncated file == no evidence
        out["status"] = MISSING
        out["reasons"].append("artefact unreadable: %s" % exc)
        return out

    spec = data.get("spec") or {}
    out["impl_sha256"] = data.get("impl_sha256")
    if require_settled and not data.get("settled", False):
        out["status"] = UNSETTLED
        out["reasons"].append("settled=false (steady state never reached)")
        return out
    for key in ("n_particles", "mu", "kn", "restitution", "seed"):
        if key not in spec:
            out["reasons"].append("spec lacks %r" % key)
        elif not _close(spec[key], expected[key]):
            out["reasons"].append(
                "tag says %s=%r, artefact says %r" % (key, expected[key],
                                                      spec[key]))
    got_dt = float(spec.get("dt_frac", DEFAULT_DT_FRAC))
    if not _close(got_dt, expected["dt_frac"]):
        out["reasons"].append("tag says dt_frac=%r, artefact says %r"
                              % (expected["dt_frac"], got_dt))
    out["status"] = MISMATCH if out["reasons"] else PRESENT
    return out


def audit_ledger(ledger, runs_dirs, require_settled: bool = True) -> dict:
    """Per-row verdicts plus the campaign-level ``complete`` flag."""
    rows = [check_row(r, runs_dirs, require_settled) for r in ledger]
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    digests = sorted({r["impl_sha256"] for r in rows if r["impl_sha256"]})
    return {
        "n_rows": len(rows),
        "rows": rows,
        "counts": counts,
        "digests": digests,
        "n_digests": len(digests),
        "complete": bool(rows) and counts.get(PRESENT, 0) == len(rows),
        "missing": [r["tag"] for r in rows if r["status"] == MISSING],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ledger", help="plan_m*.json")
    ap.add_argument("--runs", nargs="*", default=[RUNS])
    ap.add_argument("--allow-unsettled", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    res = audit_ledger(load_ledger(args.ledger), args.runs,
                       require_settled=not args.allow_unsettled)
    if args.json:
        print(json.dumps(res, indent=1, sort_keys=True))
    else:
        for r in res["rows"]:
            extra = "; ".join(r["reasons"])
            print(f"{r['jid']:>8}  {r['status']:<9} {r['tag']}"
                  + (f"  [{extra}]" if extra else ""))
        print(f"counts={res['counts']} complete={res['complete']}")
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())