#!/usr/bin/env python3
"""Doc numeric-claim self-consistency audit (content layer of the evidence chain).

Motivation
----------
`docs/dem3d_doc_evidence_pointer_audit.md` (m2985) made every *pointer* in the
documentation resolvable (doc -> file), and closed that at ORPHAN = 0.  It could
not say anything about the *numbers* sitting next to those pointers, and said so:
"指针≠正确性 ... 内容一致性是另一层".  This script is that layer, restricted to the
part of it that is machine-decidable without re-running any simulation:

  A  recompute    for an error-judgement row that prints basis, measured and a
                  relative error, the stated error must be reproducible from the
                  two printed numbers inside the interval implied by the printed
                  resolution of *measured* (the reference is a closed-form /
                  nominal value and is treated as exact).  Evaluated only on rows
                  where that interval is narrow enough to discriminate, so a
                  "pass" here is informative rather than vacuous.
  B  verdict      a row answering 是/yes to "<3 %?" must be explainable, under one
                  single unit reading of its error column (fraction or percent,
                  the same reading for the whole row), as "all errors < 3 %";
                  a row answering 否/no must be explainable as "at least one
                  error >= 3 %".

Rows whose error cell is prose ("0.00 deg (bracket 0.1 deg = 0.3 % of 11.3 deg)")
are classified UNCHECKABLE with the reason recorded, never silently passed.

Output: a ledger (`--ledger`, default docs/doc_numeric_claims_ledger.json) that
`--check` compares against the current tree, so the audit is a ratchet: a new doc
claim that fails A or B, or a previously-checkable row that degrades to
UNCHECKABLE, breaks CI.

Exit codes: 0 clean, 1 regression (only with --check), 2 usage error.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys

SCHEMA = "tensordem.doc-numeric-claims/1"
THRESH = 0.03          # the repo-wide "< 3 %" acceptance threshold
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
YES_RE = re.compile(r"^\s*(是|yes|true|达标|pass(?:ed)?|ok)\b", re.I)
NO_RE = re.compile(r"^\s*(否|no|false|不达标|fail(?:ed)?)\b", re.I)
#: characters allowed to remain in an error cell once its numbers are removed;
#: anything else means the cell is prose and the row is UNCHECKABLE.
CELL_FILLER = set(" \t<>~≈±=,;/:;·%()（）、，。[]|")

VERDICT_HEADERS = ("< 3", "<3", "3 %", "3%", "达标", "是否通过")


# --------------------------------------------------------------------------- #
# markdown table parsing
# --------------------------------------------------------------------------- #
def iter_tables(lines):
    """Yield (header_cells, body_rows) for every markdown table in `lines`."""
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip().startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            if len(block) >= 2:
                rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
                sep = bool(rows[1]) and all(
                    re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in rows[1] if c
                )
                yield rows[0], (rows[2:] if sep else rows[1:])
        else:
            i += 1


def find_col(header, keys, exclude=()):
    for j, h in enumerate(header):
        hl = h.lower()
        if any(k in hl for k in keys) and not any(x in hl for x in exclude):
            return j
    return None


def parse_num(tok):
    """(value, half_ulp) of a printed numeric literal."""
    val = float(tok)
    low = tok.lower()
    mant, _, exp_s = low.partition("e")
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    exponent = int(exp_s) if exp_s else 0
    return val, 0.5 * 10.0 ** (exponent - decimals)


def numbers(cell):
    return [parse_num(m.group(0)) for m in NUM_RE.finditer(cell)]


def numbers_units(cell):
    """(value, half_ulp, is_percent) -- "%" pins the unit of a literal, so an
    error printed as '0.4 %' can never be re-read as a fraction (40 %)."""
    out = []
    for m in NUM_RE.finditer(cell):
        val, ulp = parse_num(m.group(0))
        tail = cell[m.end():m.end() + 2].lstrip()
        out.append((val, ulp, tail.startswith("%")))
    return out


def is_numeric_cell(cell):
    """True when the cell is a number list ('1.2e-3 %', '<1e-15', '0, 12.5')
    rather than prose that happens to contain numbers."""
    rest = NUM_RE.sub("", cell)
    return all(ch in CELL_FILLER for ch in rest)


# --------------------------------------------------------------------------- #
# the two checks
# --------------------------------------------------------------------------- #
def verdict_satisfied(verdict, errors):
    """Explain the printed error list under ONE unit reading; return its name.

    `errors` is a list of (value, half_ulp, is_percent).  If every literal carries
    an explicit '%' the percent reading is pinned and the fraction reading is not
    offered; otherwise both readings are tried and either may explain the row."""
    pinned = [p for _, _, p in errors]
    scales = ((0.01, "percent"),) if all(pinned) and errors else ((1.0, "fraction"), (0.01, "percent"))
    for scale, name in scales:
        fr = [(v * scale, u * scale) for v, u, _ in errors]
        if verdict and all(v + h < THRESH for v, h in fr):
            return name
        if (not verdict) and any(v - h >= THRESH for v, h in fr):
            return name
    return None


def rel_bounds(reference, measured, measured_ulp):
    """Interval of |m - r| / r over m within its printed resolution."""
    if reference == 0:
        return None
    lows_highs = []
    for m in (abs(measured) - measured_ulp, abs(measured) + measured_ulp):
        if m >= 0:
            lows_highs.append(abs(m - abs(reference)) / abs(reference))
    if not lows_highs:
        return None
    return min(lows_highs), max(lows_highs)


def check_row(row, cols):
    """Classify one table row -> dict."""
    ie, ib, im, iv = cols
    verdict_cell = row[iv]
    if YES_RE.match(verdict_cell):
        verdict = True
    elif NO_RE.match(verdict_cell):
        verdict = False
    else:
        return None
    rec = {"verdict": verdict, "check_a": None, "check_b": None,
           "status": None, "reason": None, "errors": []}

    err_cell = row[ie] if ie is not None and ie < len(row) and ie != iv else None
    if err_cell is None or err_cell == "":
        # a verdict with no error value next to it cannot be cross-checked; count it
        # as UNCHECKABLE rather than as a vacuous pass (evidence honesty).
        rec["status"] = "UNCHECKABLE"
        rec["reason"] = "verdict column without an error column"
        return rec
    elif not is_numeric_cell(err_cell):
        rec["status"] = "UNCHECKABLE"
        rec["reason"] = "error cell is prose, not a number list"
        rec["reason_cell"] = err_cell[:120]
        return rec
    else:
        errs = numbers_units(err_cell)
        rec["errors"] = [v for v, _, _ in errs]
        scale = verdict_satisfied(verdict, errs)
        rec["check_b"] = scale is not None
        rec["unit_reading"] = scale

    if not rec["errors"]:
        rec["why_not_recomputed"] = "no error values on the row"
    elif ib is None or im is None:
        rec["why_not_recomputed"] = "table has no reference/measured columns"
    elif ib == im:
        # one composite header ("measured vs reference") served as both columns:
        # there is no independent pair to recompute from
        rec["why_not_recomputed"] = "reference and measured share one composite column"
    elif max(ib, im) >= len(row):
        rec["why_not_recomputed"] = "row shorter than the table header"
    else:
        bn, mn = numbers(row[ib]), numbers(row[im])
        if len(bn) != 1 or len(mn) != 1:
            rec["why_not_recomputed"] = "reference/measured cells are not single numbers"
        elif bn[0][0] == 0:
            rec["why_not_recomputed"] = "reference is zero"
        else:
            bounds = rel_bounds(bn[0][0], mn[0][0], mn[0][1])
            if not bounds:
                rec["why_not_recomputed"] = "no admissible error interval"
            elif bounds[1] <= 0 or (bounds[1] - bounds[0]) > 0.5 * bounds[1]:
                # the printed resolution of `measured` is coarser than the error
                # being claimed, so any judgement here would be noise
                rec["why_not_recomputed"] = "printed resolution too coarse to discriminate"
            else:
                ok = False
                for ev, eh, is_pct in numbers_units(err_cell):
                    for cand in ((ev / 100.0,) if is_pct else (ev, ev / 100.0)):
                        if bounds[0] - eh - 1e-15 <= cand <= bounds[1] + eh + 1e-15:
                            ok = True
                rec["check_a"] = ok
                rec["basis"] = bn[0][0]
                rec["measured"] = mn[0][0]
                rec["bounds"] = [bounds[0], bounds[1]]

    if rec["status"] is None:
        if rec["check_a"] is False or rec["check_b"] is False:
            rec["status"] = "FAIL"
        elif rec["check_a"] is True or rec["check_b"] is True:
            rec["status"] = "PASS"
        else:
            rec["status"] = "UNCHECKABLE"
            rec["reason"] = rec["reason"] or "no discriminating reference/measured pair"
    return rec


def audit(root="."):
    records = []
    total_rows = 0
    for path in sorted(glob.glob(f"{root}/docs/**/*.md", recursive=True)):
        try:
            lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        doc = path[len(root) + 1:] if path.startswith(root) else path
        for header, rows in iter_tables(lines):
            total_rows += len(rows)
            iv = find_col(header, VERDICT_HEADERS)
            if iv is None:
                continue
            ie = find_col(header, ("相对误差", "relative error", "误差", "error"),
                          exclude=VERDICT_HEADERS)
            ib = find_col(header, ("基准", "reference"))
            im = find_col(header, ("实测", "measured", "修正后"))
            cols = (ie, ib, im, iv)
            for row in rows:
                if iv >= len(row):
                    continue
                rec = check_row(row, cols)
                if rec:
                    rec["doc"] = doc
                    rec["label"] = row[0] if row else ""
                    rec["row"] = " | ".join(row)[:160]
                    records.append(rec)
    audit.total_table_rows = total_rows
    return records


def _count(values):
    out = {}
    for v in values:
        if v:
            out[v] = out.get(v, 0) + 1
    return out


def key_of(rec):
    return f"{rec['doc']}::{rec.get('label', '')}"


def summarise(records):
    by_status = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "schema": SCHEMA,
        "threshold": THRESH,
        "rows_with_verdict": len(records),
        "check_a_evaluated": sum(1 for r in records if r["check_a"] is not None),
        "check_b_evaluated": sum(1 for r in records if r["check_b"] is not None),
        "by_status": by_status,
        "docs_touched": len({r["doc"] for r in records}),
        "markdown_table_rows": getattr(audit, "total_table_rows", 0),
        "recomputed_rows": sum(1 for r in records if r["check_a"] is not None),
        "not_recomputed": _count(r.get("why_not_recomputed") for r in records),
        "records": records,
    }


def markdown_tables(ledger):
    """Emit the result tables used by the audit document (no hand transcription)."""
    out = []
    out.append("| metric | value |\n|---|---|")
    out.append(f"| markdown table rows in `docs/**/*.md` | {ledger['markdown_table_rows']} |")
    out.append(f"| rows carrying an error verdict | {ledger['rows_with_verdict']} |")
    out.append(f"| documents contributing verdict rows | {ledger['docs_touched']} |")
    out.append(f"| check B (verdict logic) evaluated | {ledger['check_b_evaluated']} |")
    out.append(f"| check A (recompute) evaluated | {ledger['check_a_evaluated']} |")
    for status in ("PASS", "FAIL", "UNCHECKABLE"):
        out.append(f"| status {status} | {ledger['by_status'].get(status, 0)} |")
    out.append("")
    out.append("| why a row could not be recomputed | rows |\n|---|---|")
    for reason, n in sorted((ledger.get("not_recomputed") or {}).items(),
                            key=lambda kv: -kv[1]):
        out.append(f"| {reason} | {n} |")
    out.append(f"| (recomputed) | {ledger['recomputed_rows']} |")
    out.append("")
    out.append("| document | rows | PASS | FAIL | UNCHECKABLE |\n|---|---|---|---|---|")
    per = {}
    for r in ledger["records"]:
        per.setdefault(r["doc"], {}).setdefault(r["status"], 0)
        per[r["doc"]][r["status"]] += 1
    for doc in sorted(per, key=lambda d: (-sum(per[d].values()), d)):
        c = per[doc]
        out.append(f"| `{doc}` | {sum(c.values())} | {c.get('PASS', 0)} | "
                   f"{c.get('FAIL', 0)} | {c.get('UNCHECKABLE', 0)} |")
    out.append("")
    out.append("| document | reference | measured | stated error | admissible interval | recomputed? |\n|---|---|---|---|---|---|")
    for r in ledger["records"]:
        if r["check_a"] is None:
            continue
        lo, hi = r["bounds"]
        out.append(f"| `{r['doc']}` | {r['basis']!r} | {r['measured']!r} | "
                   f"{r['errors'][0]!r}{' %' if r.get('unit_reading') == 'percent' else ''} | "
                   f"[{lo:.3g}, {hi:.3g}] | {'yes' if r['check_a'] else 'NO'} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--ledger", default="docs/doc_numeric_claims_ledger.json")
    ap.add_argument("--write", action="store_true", help="rewrite the ledger")
    ap.add_argument("--check", action="store_true", help="fail on regression vs ledger")
    ap.add_argument("--emit-tables", action="store_true",
                    help="print the markdown result tables and exit")
    a = ap.parse_args(argv)

    ledger = summarise(audit(a.root))
    fails = [r for r in ledger["records"] if r["status"] == "FAIL"]
    for r in fails:
        print(f"FAIL {r['doc']}: verdict={'yes' if r['verdict'] else 'no'} "
              f"errors={r['errors']} A={r['check_a']} B={r['check_b']} :: {r['row']}")
    print(f"rows_with_verdict={ledger['rows_with_verdict']} "
          f"checkA={ledger['check_a_evaluated']} checkB={ledger['check_b_evaluated']} "
          f"status={ledger['by_status']}")

    import os
    if a.emit_tables:
        print(markdown_tables(ledger))
        return 0

    ledger_path = os.path.join(a.root, a.ledger) if a.root != "." else a.ledger
    if a.write:
        with open(ledger_path, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"ledger written: {ledger_path}")
    if a.check:
        try:
            old = json.load(open(ledger_path, encoding="utf-8"))
        except OSError:
            print("no ledger to check against")
            return 1
        def index(recs):
            out = {}
            for r in recs:
                k = key_of(r)
                out.setdefault(k, []).append(r["status"])
            return out

        old_idx, new_idx = index(old["records"]), index(ledger["records"])
        added, worsened, degraded = [], [], []
        for k, statuses in new_idx.items():
            prev = old_idx.get(k)
            if prev is None:
                if "FAIL" in statuses:
                    added.append(k)
                continue
            if "FAIL" in statuses and "FAIL" not in prev:
                worsened.append(k)
            if "UNCHECKABLE" in statuses and all(st != "UNCHECKABLE" for st in prev):
                degraded.append(k)
        for k in sorted(added):
            print(f"REGRESSION new failing claim in {k}")
        for k in sorted(worsened):
            print(f"REGRESSION claim now fails in {k}")
        for k in sorted(degraded):
            print(f"REGRESSION claim became uncheckable in {k}")
        for k in sorted(set(old_idx) - set(new_idx)):
            print(f"note: claim no longer present in {k}")
        return 1 if (added or worsened or degraded) else 0
    return 0 if not fails else 0   # findings are reported, not a CI failure, unless --check


if __name__ == "__main__":
    sys.exit(main())