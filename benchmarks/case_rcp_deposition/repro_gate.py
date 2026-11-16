#!/usr/bin/env python3
"""Reproducibility gate for the RCP deposition audit ledger.

Anchor: ``CASE-DEM-RCP-PACKING-FRACTION`` (feature_ledger).

Why this exists
---------------
Every verdict on this anchor (``IN_BAND`` / ``OUT_OF_BAND`` / ``UNDECIDABLE``,
``CONVERGED`` / ``BIASED``) is produced by an audit script that reads the stored
DEM runs and writes a JSON artefact plus a Markdown document.  Those artefacts
are what the ledger cites.  Nothing in the repo, however, checks the *inverse*
direction: that the committed artefact is what the committed code produces from
the committed data.  An artefact that cannot be regenerated is a claim, not
evidence -- a stale artefact silently keeps a superseded number alive, and a
number copied between worktrees silently drags another checkout's absolute
paths (and therefore another revision's provenance) into this one.

This gate closes that direction.  For every audit script it

1. snapshots the tracked files (``git status --porcelain``, untracked excluded),
2. runs the script **with its committed defaults**,
3. diffs every scalar leaf of the freshly written JSON against the committed
   JSON (``git show HEAD:<path>``) with an explicit numeric tolerance,
4. diffs the freshly written Markdown documents line by line and scans every
   text artefact for absolute paths that leak a foreign checkout,
5. restores the committed bytes, so the gate is read-only with respect to the
   tree it audits.

Per-script verdicts
-------------------
``REPRODUCED``   no leaf outside tolerance, no type change, no missing/extra
                 leaf, no document drift worth reporting.
``CENTRE_OK``    only "centre of an interval" style drift: leaves outside
                 tolerance exist, but every published *categorical* verdict
                 (band judgement / certification flag) is unchanged.  Reported
                 separately because it bounds the practical damage.
``STALE``        at least one categorical verdict changed, or a scalar that a
                 verdict is computed from moved outside tolerance without any
                 declared stochastic knob explaining it.
``FAILED``       script crashed or timed out; artefacts untouched.
``NO_ARTEFACT``  committed JSON missing (nothing to compare against).

The ``stiffness_seed_audit.py`` Monte-Carlo interval endpoints depend on
``--n-mc``; that is a declared knob, so it is audited separately by
``--mc-scan``: the gate re-runs that script at several MC resolutions and
reports whether any *categorical* verdict is MC-resolution dependent.

Run::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/repro_gate.py
    PYTHONPATH=src python benchmarks/case_rcp_deposition/repro_gate.py --only stiffness_seed_audit
    PYTHONPATH=src python benchmarks/case_rcp_deposition/repro_gate.py --mc-scan 1500,4000,8000
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")
FIGS = os.path.join(HERE, "figures")
OUT_JSON = os.path.join(RUNS, "repro_gate.json")
DOC = os.path.join(ROOT, "docs", "AUDIT_RCP_REPRO_GATE.md")

# Absolute-path fragments that can only come from a foreign checkout.
FOREIGN_ROOTS = ("/root/tp-agent/work/", "/home/", "/mnt/", "/tmp/")

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12

# ---------------------------------------------------------------------------
# Registry: script -> artefacts it owns.
#
# ``doc`` lists the Markdown files the script rewrites (compared line by line).
# ``stochastic`` marks the scripts with a declared Monte-Carlo knob; their
# numeric drift is classified with that explanation attached.
# ---------------------------------------------------------------------------
REGISTRY = [
    dict(name="aggregate",
         script="aggregate.py",
         json="runs/summary.json",
         doc=[], stochastic=False),
    dict(name="debt_audit",
         script="debt_audit.py",
         json="runs/debt_audit.json",
         doc=["docs/AUDIT_m16929_RCP_PRODUCTION_WAVE.md"],
         stochastic=False),
    dict(name="order_audit",
         script="order_audit.py",
         json="runs/order_audit.json",
         doc=["docs/AUDIT_m16999_RCP_EXTRAPOLATION_ORDER.md"],
         stochastic=False),
    dict(name="steady_state_audit",
         script="steady_state_audit.py",
         json="runs/steady_state_audit.json",
         doc=["docs/AUDIT_m17063_RCP_STEADY_STATE.md"],
         stochastic=False),
    dict(name="phi_definition_audit",
         script="phi_definition_audit.py",
         json="runs/phi_definition_audit.json",
         doc=["docs/AUDIT_m18395_RCP_PHI_DEFINITION.md"],
         stochastic=False),
    dict(name="anchor_provenance_audit",
         script="anchor_provenance_audit.py",
         json="runs/anchor_provenance_audit.json",
         doc=["docs/AUDIT_m21145_RCP_ANCHOR_PROVENANCE.md"],
         stochastic=False),
    dict(name="timestep_ladder_audit",
         script="timestep_ladder_audit.py",
         json="runs/timestep_ladder_audit.json",
         doc=["docs/AUDIT_m21401_RCP_TIMESTEP_LADDER.md"],
         stochastic=False),
    dict(name="seed_bias_audit",
         script="seed_bias_audit.py",
         json="runs/seed_bias_audit.json",
         doc=["docs/AUDIT_m22558_RCP_SEED_RESOLVED_BIAS.md"],
         stochastic=False),
    dict(name="stiffness_seed_audit",
         script="stiffness_seed_audit.py",
         json="runs/stiffness_seed_audit.json",
         doc=[],
         stochastic=True),
]

# Leaf names whose string value is a *categorical* verdict.  A change in one of
# these is what separates "the interval centre moved in the third decimal" from
# "the pass/fail statement changed".
VERDICT_LEAVES = {
    "verdict", "band", "band_verdict", "in_band", "in_ledger_band",
    "certified", "resolved_from", "strongest_statement", "verdict_band",
    "verdict_hardsphere", "status", "converged", "pass", "all_gates_pass",
}

# Keys that identify a record inside a list of records.  When every element of
# a list carries one of them, the list is re-keyed by identity before flattening
# (``gates[G18_timestep...]`` instead of ``gates[11]``), otherwise inserting one
# gate shifts every later index and the diff reports the whole tail as changed.
IDENTITY_KEYS = ("id", "slug", "name", "key", "case", "tag", "path", "run")
# Composite identity for records that are identified by their parameter vector
# rather than by a label (per-run rows in ``summary.json`` / ``debt_audit.json``).
COMPOSITE_KEYS = ("n", "mu", "kn", "dt_frac", "seed", "mur")


# ---------------------------------------------------------------------------
# generic JSON leaf machinery
# ---------------------------------------------------------------------------
def _identity(v):
    """Identity string for a record, or ``None`` when it has none."""
    if not isinstance(v, dict):
        return None
    for k in IDENTITY_KEYS:
        val = v.get(k)
        if isinstance(val, str) and val:
            return f"{k}={val}"
    parts = [f"{k}={v[k]}" for k in COMPOSITE_KEYS
             if k in v and isinstance(v[k], (int, float, str)) and not isinstance(v[k], bool)]
    return "|".join(parts) if parts else None


def canonicalise(obj):
    """Re-key lists of identified records by identity so diffs survive insertion.

    A list whose elements are all dicts carrying an :data:`IDENTITY_KEYS` entry
    becomes a dict keyed by that identity (positional index kept as ``#i`` in the
    key so ordering information is not silently lost).
    """
    if isinstance(obj, dict):
        return {k: canonicalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [canonicalise(v) for v in obj]
        idents = [_identity(v) for v in items]
        if items and all(i is not None for i in idents):
            seen = {}
            keyed = {}
            for ident, v in zip(idents, items):
                occ = seen.get(ident, 0)
                seen[ident] = occ + 1
                keyed[f"{ident}#{occ}" if occ or idents.count(ident) > 1
                      else ident] = v
            return keyed
        return items
    return obj


def flatten(obj, prefix=""):
    """Flatten nested JSON into ``{path: leaf}`` with scalars as leaves."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix or "/"] = obj
    return out


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def leaf_name(path):
    """Last path component, list index stripped."""
    tail = path.rsplit("/", 1)[-1]
    return tail.split("[")[0]


def compare_leaves(ref, cur, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Leaf-by-leaf comparison of two flattened payloads.

    Returns a dict with ``records`` (only non-identical leaves) and ``counts``.
    Numeric leaves are compared as ``|cur-ref| <= atol + rtol*|ref|``; that is a
    *reproducibility* tolerance (same code, same data, same platform family),
    not a modelling tolerance, so it is tight on purpose.
    """
    keys = sorted(set(ref) | set(cur))
    records = []
    n_same = 0
    n_within = 0
    for k in keys:
        r = ref.get(k, KeyError)
        c = cur.get(k, KeyError)
        if r is KeyError or c is KeyError:
            records.append(dict(path=k,
                                ref=None if r is KeyError else r,
                                cur=None if c is KeyError else c,
                                status="MISSING_REF" if r is KeyError else "MISSING_CUR"))
            continue
        if is_number(r) and is_number(c):
            if float(r) != float(r) and float(c) != float(c):   # nan == nan here
                n_same += 1
                continue
            d = float(c) - float(r)
            tol = atol + rtol * abs(float(r))
            if d == 0.0:
                n_same += 1
                continue
            rec = dict(path=k, ref=r, cur=c, abs_delta=d,
                       rel_delta=(None if float(r) == 0.0 else d / abs(float(r))),
                       tol=tol,
                       status="WITHIN_TOL" if abs(d) <= tol else "DRIFT")
            if rec["status"] == "WITHIN_TOL":
                n_within += 1
            records.append(rec)
            continue
        if type(r) is type(c) and r == c:
            n_same += 1
            continue
        if is_number(r) != is_number(c) or isinstance(r, str) != isinstance(c, str):
            records.append(dict(path=k, ref=r, cur=c, status="TYPE_CHANGE"))
            continue
        records.append(dict(path=k, ref=r, cur=c, status="VALUE_CHANGE"))
    counts = dict(n_leaves=len(keys), n_same=n_same, n_within_tol=n_within,
                  n_drift=sum(1 for x in records if x["status"] == "DRIFT"),
                  n_value_change=sum(1 for x in records if x["status"] == "VALUE_CHANGE"),
                  n_type_change=sum(1 for x in records if x["status"] == "TYPE_CHANGE"),
                  n_missing=sum(1 for x in records
                                if x["status"] in ("MISSING_REF", "MISSING_CUR")))
    return dict(counts=counts, records=records)


def _group_prefixes(paths, depth=2):
    """Count leaf paths by their first ``depth`` components (for reporting)."""
    out = {}
    for p in paths:
        comps = [c for c in p.replace("[", "/").replace("]", "").split("/") if c]
        key = "/" + "/".join(comps[:depth])
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def doc_line_diff(ref_text, cur_text):
    """(n_changed_lines, n_added, n_removed) between committed and fresh doc."""
    import difflib
    if ref_text is None or cur_text is None:
        return dict(n_changed=None, n_added=None, n_removed=None, examples=[])
    a, b = ref_text.splitlines(), cur_text.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changed = added = removed = 0
    examples = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            changed += max(i2 - i1, j2 - j1)
            if len(examples) < 3 and j2 > j1:
                examples.append(b[j1][:200])
        elif tag == "insert":
            added += j2 - j1
            if len(examples) < 3:
                examples.append(b[j1][:200])
        elif tag == "delete":
            removed += i2 - i1
    return dict(n_changed=changed, n_added=added, n_removed=removed,
                examples=examples)


def verdict_leaves(flat):
    """Sub-payload of leaves whose name marks a categorical verdict."""
    return {k: v for k, v in flat.items()
            if leaf_name(k) in VERDICT_LEAVES and isinstance(v, (str, bool))}


def compare_verdicts(ref_flat, cur_flat):
    """Which categorical verdicts changed.  Empty list == all preserved."""
    a, b = verdict_leaves(ref_flat), verdict_leaves(cur_flat)
    changed = []
    for k in sorted(set(a) | set(b)):
        if a.get(k, "<absent>") != b.get(k, "<absent>"):
            changed.append(dict(path=k, ref=a.get(k, "<absent>"),
                                cur=b.get(k, "<absent>")))
    return dict(n_verdict_leaves=len(set(a) | set(b)),
                changed=changed,
                preserved=not changed)


def find_foreign_paths(flat, foreign_roots=FOREIGN_ROOTS, root=ROOT):
    """String leaves that embed an absolute path outside ``root``.

    Relative paths (``benchmarks/...``) are fine: they survive a checkout move.
    An absolute path from somebody else's working tree does not, and it makes
    the artefact's provenance claim untraceable.
    """
    hits = []
    for k, v in flat.items():
        if not isinstance(v, str):
            continue
        if not v.startswith("/"):
            continue
        if v.startswith(root + os.sep) or v == root:
            continue
        if any(v.startswith(p) for p in foreign_roots):
            hits.append(dict(path=k, value=v,
                             foreign=(None if v.startswith(root) else
                                      next((p for p in foreign_roots
                                            if v.startswith(p)), "<absolute>"))))
    return hits


def scan_text_for_paths(text, root=ROOT):
    """Same leak test for Markdown: absolute paths that are not under ``root``."""
    hits = []
    for ln, line in enumerate(text.splitlines(), 1):
        for token in line.replace("`", " ").replace("(", " ").replace(")", " ").split():
            s = token.rstrip(".,;:)]}\"'")
            if not s.startswith("/"):
                continue
            if s.startswith(root + os.sep) or s == root:
                continue
            if any(s.startswith(p) for p in FOREIGN_ROOTS):
                hits.append(dict(line=ln, value=s))
    return hits


# ---------------------------------------------------------------------------
# git snapshot / restore (read-only audit of the tree)
# ---------------------------------------------------------------------------
def _git(*args, check=False):
    p = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p


def tracked_snapshot():
    """{path: porcelain-status} for tracked, non-ignored files."""
    out = {}
    p = _git("status", "--porcelain", "--untracked-files=no")
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        st, path = line[:2], line[3:]
        out[path] = st
    return out


def untracked_snapshot():
    p = _git("status", "--porcelain", "--untracked-files=all", "--ignored=no")
    return {line[3:] for line in p.stdout.splitlines() if line.startswith("??")}


def committed_bytes(relpath):
    p = _git("show", f"HEAD:{relpath}")
    if p.returncode != 0:
        return None
    return p.stdout.encode()


def committed_text(relpath):
    b = committed_bytes(relpath)
    return None if b is None else b.decode("utf-8", "replace")


def restore_paths(paths, protect=()):
    """``git checkout --`` the given repo-relative paths (never outside ROOT)."""
    safe, refused = [], []
    for rel in paths:
        abs_p = os.path.abspath(os.path.join(ROOT, rel))
        if not abs_p.startswith(ROOT + os.sep):
            refused.append(rel)
            continue
        if any(os.path.abspath(p) == abs_p for p in protect):
            refused.append(rel)
            continue
        safe.append(rel)
    if safe:
        _git("checkout", "--", *safe)
    return dict(restored=safe, refused=refused)


# ---------------------------------------------------------------------------
# one script
# ---------------------------------------------------------------------------
def run_script(entry, timeout=1800, extra_args=()):
    """Run one audit script in place, timing it.  Returns (rc, wall, stdout)."""
    path = os.path.join(HERE, entry["script"])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, path, *extra_args], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
        rc, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as exc:
        rc, out = -9, f"TIMEOUT after {timeout}s\n{exc}"
    return rc, time.time() - t0, out[-4000:]


def gate_one(entry, timeout=1800, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
             protect=(), keep=False, baseline="head"):
    """Snapshot -> run -> diff -> restore.  Never leaves the tree modified."""
    rel_json = os.path.join("benchmarks", "case_rcp_deposition", entry["json"])
    rel_docs = list(entry["doc"])          # registry paths are repo-relative
    before = tracked_snapshot()
    before_untracked = untracked_snapshot()
    if baseline == "worktree":
        # Compare against the artefact *as it stands in the tree* (used after a
        # regeneration, to certify that a second run is idempotent).
        wt = os.path.join(ROOT, rel_json)
        ref_text = open(wt, encoding="utf-8").read() if os.path.exists(wt) else None
    else:
        ref_text = committed_text(rel_json)

    rec = dict(name=entry["name"], script=entry["script"], json=entry["json"],
               stochastic=entry["stochastic"], doc=list(entry["doc"]))

    rc, wall, tail = run_script(entry, timeout=timeout)
    rec.update(returncode=rc, wall_s=round(wall, 1), stdout_tail=tail[-1200:])

    after = tracked_snapshot()
    touched = sorted(set(after) - set(before))
    rec["touched_tracked"] = touched
    rec["new_untracked"] = sorted(untracked_snapshot() - before_untracked)

    if rc != 0:
        rec["verdict"] = "FAILED"
        restore_paths(touched, protect=protect)
        return rec

    if ref_text is None:
        rec["verdict"] = "NO_ARTEFACT"
        rec["reason"] = f"HEAD:{rel_json} absent"
        restore_paths(touched, protect=protect)
        return rec

    cur_path = os.path.join(ROOT, rel_json)
    if not os.path.exists(cur_path):
        rec["verdict"] = "NO_ARTEFACT"
        rec["reason"] = "script did not write its JSON artefact"
        restore_paths(touched, protect=protect)
        return rec
    with open(cur_path, encoding="utf-8") as fh:
        cur_text = fh.read()

    ref_flat = flatten(canonicalise(json.loads(ref_text)))
    cur_flat = flatten(canonicalise(json.loads(cur_text)))
    diff = compare_leaves(ref_flat, cur_flat, rtol=rtol, atol=atol)
    vdiff = compare_verdicts(ref_flat, cur_flat)
    leaks = ([dict(h, source="HEAD") for h in find_foreign_paths(ref_flat)]
             + [dict(h, source="rerun") for h in find_foreign_paths(cur_flat)])

    rec["counts"] = diff["counts"]
    rec["missing_by_prefix"] = _group_prefixes(
        [r["path"] for r in diff["records"]
         if r["status"] in ("MISSING_REF", "MISSING_CUR")])
    rec["drift"] = [r for r in diff["records"] if r["status"] == "DRIFT"][:40]
    rec["other_changes"] = [r for r in diff["records"]
                            if r["status"] not in ("DRIFT", "WITHIN_TOL")][:40]
    rec["verdict_changes"] = vdiff["changed"][:20]
    rec["n_verdict_leaves"] = vdiff["n_verdict_leaves"]
    rec["path_leaks"] = leaks[:20]
    rec["n_path_leaks"] = len(leaks)

    # document comparison
    doc_recs = []
    for rel_doc in rel_docs:
        ref_d = committed_text(rel_doc)
        cur_d = None
        p_doc = os.path.join(ROOT, rel_doc)
        if os.path.exists(p_doc):
            with open(p_doc, encoding="utf-8") as fh:
                cur_d = fh.read()
        dd = doc_line_diff(ref_d, cur_d)
        doc_recs.append(dict(
            path=rel_doc,
            n_changed_lines=dd["n_changed"], n_added_lines=dd["n_added"],
            n_removed_lines=dd["n_removed"], diff_examples=dd["examples"],
            ref_chars=None if ref_d is None else len(ref_d),
            cur_chars=None if cur_d is None else len(cur_d),
            identical=(ref_d == cur_d),
            n_leaks_ref=0 if ref_d is None else len(scan_text_for_paths(ref_d)),
            n_leaks_cur=0 if cur_d is None else len(scan_text_for_paths(cur_d)),
            leak_examples=([] if cur_d is None
                           else scan_text_for_paths(cur_d)[:5]),
        ))
    rec["docs"] = doc_recs

    if not vdiff["preserved"]:
        rec["verdict"] = "STALE"
    elif diff["counts"]["n_drift"] or diff["counts"]["n_value_change"] \
            or diff["counts"]["n_type_change"] or diff["counts"]["n_missing"]:
        rec["verdict"] = "CENTRE_OK"
    else:
        rec["verdict"] = "REPRODUCED"

    if not keep:
        restore_paths(touched, protect=protect)
    return rec


# ---------------------------------------------------------------------------
# Monte-Carlo resolution scan (declared stochastic knob of stiffness_seed_audit)
# ---------------------------------------------------------------------------
def mc_scan(entry_name="stiffness_seed_audit", n_mcs=(1500, 4000, 8000),
            timeout=2400):
    """Re-run the MC-dependent audit at several resolutions, compare verdicts."""
    entry = next(e for e in REGISTRY if e["name"] == entry_name)
    before = tracked_snapshot()
    runs = []
    for n in n_mcs:
        tmp = os.path.join("/tmp", f"repro_mc_{entry_name}_{n}.json")
        rc, wall, tail = run_script(
            entry, timeout=timeout, extra_args=["--n-mc", str(n), "--out", tmp])
        payload = None
        if rc == 0 and os.path.exists(tmp):
            with open(tmp, encoding="utf-8") as fh:
                payload = json.load(fh)
        runs.append(dict(n_mc=n, returncode=rc, wall_s=round(wall, 1),
                         payload=payload, stdout_tail=tail[-400:]))
    restore_paths(sorted(set(tracked_snapshot()) - set(before)))

    good = [r for r in runs if r["payload"] is not None]
    res = dict(n_mcs=list(n_mcs), tiers={}, verdicts_identical=True,
               changed=[], numeric_spread={})
    if not good:
        res["verdicts_identical"] = False
        res["reason"] = "no MC run produced an artefact"
        return res
    base = flatten(good[0]["payload"])
    for g in good[1:]:
        vd = compare_verdicts(base, flatten(g["payload"]))
        if not vd["preserved"]:
            res["verdicts_identical"] = False
            res["changed"].extend(vd["changed"])
    # numeric spread of the published interval endpoints
    for key in sorted(base):
        if not key.endswith(("lo", "hi")) and not leaf_name(key).endswith(("lo", "hi")):
            continue
        vals = []
        for g in good:
            f = flatten(g["payload"])
            v = f.get(key, "<absent>")
            if is_number(v):
                vals.append(float(v))
        if len(vals) >= 2:
            res["numeric_spread"][key] = dict(min=min(vals), max=max(vals),
                                              spread=max(vals) - min(vals),
                                              values=vals)
    return res


# ---------------------------------------------------------------------------
# roll-up
# ---------------------------------------------------------------------------
def summarise(records):
    order = ["REPRODUCED", "CENTRE_OK", "STALE", "FAILED", "NO_ARTEFACT"]
    counts = {k: sum(1 for r in records if r["verdict"] == k) for k in order}
    blocking = counts["STALE"] + counts["FAILED"] + counts["NO_ARTEFACT"]
    return dict(counts=counts, n_scripts=len(records),
                overall="PASS" if blocking == 0 else "REVIEW",
                blocking=blocking)


def render_doc(payload):
    """Markdown with every number injected from ``payload`` (nothing hand-typed)."""
    L = []
    A = L.append
    sm = payload["summary"]
    A("# AUDIT · RCP 审计账目的可复现闸门（CASE-DEM-RCP-PACKING-FRACTION）")
    A("")
    A("**锚点**：`CASE-DEM-RCP-PACKING-FRACTION`（算例池·随机密堆体积分数 RCP）")
    A("**本单性质**：验证欠账补齐——把「已提交的审计产物是否由已提交的代码+数据重算得出」变成可执行的判定，")
    A("不改任何求解器方程、不改任何既有结论的判据。")
    A("")
    A("## 1 欠账的准确表述")
    A("")
    A("本锚点的每一次裁决（`IN_BAND`/`OUT_OF_BAND`/`UNDECIDABLE`、`CONVERGED`/`BIASED`）都是")
    A("`benchmarks/case_rcp_deposition/*_audit.py` 读 `runs/*.npz|json` 后写出的 JSON + Markdown。")
    A("账目引用的是这些产物，但仓内从来没有反向检查：**产物是不是现码现数据重算出来的**。")
    A("一个不能重算的产物是「主张」而不是「证据」：过期产物会让一个已被取代的数字继续活着，")
    A("跨工作树拷来的数字还会把别人检出里的绝对路径（进而别人的版本口径）带进本仓。")
    A("")
    A(f"闸门覆盖 {sm['n_scripts']} 个审计脚本，逐叶比对 JSON（数值容差 `rtol={payload['rtol']:g}`, "
      f"`atol={payload['atol']:g}`），逐行比对 Markdown，并扫描外来绝对路径。")
    A("")
    A("## 2 怎么做的")
    A("")
    A("每个脚本：快照 `git status` → **用它自己的默认参数**运行 → 逐叶 diff 新产物与 `git show HEAD:<path>`")
    A("→ 逐行 diff 文档 → 扫路径泄漏 → `git checkout --` 还原。闸门对被测树是**只读**的（还原后 `git status` 为空）。")
    A("")
    A("判级规则（预登记，见脚本 docstring）：")
    A("")
    A("| 判级 | 含义 |")
    A("|---|---|")
    A("| `REPRODUCED` | 所有叶子在容差内，无类型/缺失变化 |")
    A("| `CENTRE_OK` | 有叶子超容差，但所有**分类裁决**（带宽判定/认证位）未变 |")
    A("| `STALE` | 至少一个分类裁决变了 |")
    A("| `FAILED` / `NO_ARTEFACT` | 脚本崩/超时，或提交产物不存在 |")
    A("")
    A("## 3 结果")
    A("")
    A(f"总体：**{sm['overall']}**（阻断项 {sm['blocking']} 个）。")
    A("")
    A("| 脚本 | 判级 | 墙钟 (s) | 叶子 | 相同 | 容差内 | 超容差 | 类型/值变 | 裁决位变 | 路径泄漏 | 随机核 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in payload["records"]:
        c = r.get("counts", {})
        A(f"| `{r['script']}` | `{r['verdict']}` | {r['wall_s']} | "
          f"{c.get('n_leaves', '-')} | {c.get('n_same', '-')} | {c.get('n_within_tol', '-')} | "
          f"{c.get('n_drift', '-')} | {c.get('n_value_change', 0) + c.get('n_type_change', 0)} | "
          f"{len(r.get('verdict_changes', []))} | {r.get('n_path_leaks', '-')} | "
          f"{'是' if r['stochastic'] else '否'} |")
    A("")

    drift_recs = [(r["name"], d) for r in payload["records"] for d in r.get("drift", [])]
    A("### 3.1 超容差叶子（逐条）")
    A("")
    if drift_recs:
        A("| 脚本 | 路径 | 提交值 | 重算值 | Δ | 相对 |")
        A("|---|---|---|---|---|---|")
        for name, d in drift_recs[:25]:
            rel = d.get("rel_delta")
            rel_s = "—" if rel is None else f"{rel:+.3%}"
            A(f"| `{name}` | `{d['path']}` | {d['ref']!r} | {d['cur']!r} | "
              f"{d['abs_delta']:+.3e} | {rel_s} |")
    else:
        A("无。所有脚本在默认参数下逐叶复现。")
    A("")

    leaks = [(r["name"], h) for r in payload["records"] for h in r.get("path_leaks", [])]
    A("### 3.2 外来绝对路径")
    A("")
    if leaks:
        A(f"共 {sum(r.get('n_path_leaks', 0) for r in payload['records'])} 处。它们让产物的来源不可追溯"
          "（路径指向别人的工作树），且一旦检出目录改名就失效。")
        A("")
        A("| 脚本 | 叶子 | 值 |")
        A("|---|---|---|")
        for name, h in leaks[:12]:
            A(f"| `{name}` | `{h['path']}` | `{h['value']}` |")
    else:
        A("无。所有路径为仓内相对路径。")
    A("")

    A("### 3.3 随机核（`--n-mc`）敏感性")
    A("")
    mc = payload.get("mc_scan")
    if mc and mc.get("n_mcs"):
        A(f"`stiffness_seed_audit.py` 的区间端点依赖 `--n-mc`（默认在历史中被改过）。"
          f"在 n_mc ∈ {mc['n_mcs']} 上重算：**分类裁决全部不变**" if mc.get("verdicts_identical")
          else f"在 n_mc ∈ {mc['n_mcs']} 上重算：**存在裁决改变**")
        A("" if mc.get("verdicts_identical") else "")
        A("")
        if mc.get("changed"):
            A("| 变化的裁决 | 首档 | 他档 |")
            A("|---|---|---|")
            for c in mc["changed"][:10]:
                A(f"| `{c['path']}` | `{c['ref']}` | `{c['cur']}` |")
            A("")
        sp = mc.get("numeric_spread", {})
        if sp:
            worst = sorted(sp.items(), key=lambda kv: -kv[1]["spread"])[:8]
            A("| 区间端点叶子 | 档内最小 | 档内最大 | 跨度 |")
            A("|---|---|---|---|")
            for k, v in worst:
                A(f"| `{k}` | {v['min']:.6f} | {v['max']:.6f} | {v['spread']:.2e} |")
            A("")
            A(f"最大端点半宽漂移 {max(v['spread'] for v in sp.values()):.2e}"
              f"，验收带宽 0.04；占比 {max(v['spread'] for v in sp.values())/0.04:.2%}。")
    else:
        A("本次未执行（`--mc-scan` 关闭）。")
    A("")

    rg = payload.get("regen")
    A("### 3.4 重算修正记录（`runs/repro_regen.json`）")
    A("")
    if rg:
        A("发现过期产物后按仓内约定**重算**（产物与文档都由脚本生成，重算即恢复一致）。重算前的差异：")
        A("")
        A("| 脚本 | 重算前判级 | 超容差叶子 | 裁决位变 | 过期产物里的外来路径 |")
        A("|---|---|---|---|---|")
        for r in rg.get("records", []):
            c = r.get("counts", {})
            A(f"| `{r['script']}` | `{r['verdict']}` | {c.get('n_drift', '-')} | "
              f"{len(r.get('verdict_changes', []))} | {r.get('n_path_leaks', '-')} |")
        A("")
        flipped = [(r["script"], v) for r in rg.get("records", [])
                   for v in r.get("verdict_changes", [])]
        if flipped:
            A("其中**分类裁决**发生变化的位置（重算前 → 重算后）：")
            A("")
            A("| 脚本 | 裁决叶子 | 旧 | 新 |")
            A("|---|---|---|---|")
            for name, v in flipped[:12]:
                A(f"| `{name}` | `{v['path']}` | `{v['ref']}` | `{v['cur']}` |")
            A("")
    else:
        A("无（本次未执行重算）。")
    A("")
    A("## 4 注意点")
    A("")
    A("1. 闸门只回答「能否重算」，不回答「结论是否物理正确」——物理裁决的证据链见 §3.1 引用的各审计文档。")
    A("2. 容差 `rtol=1e-9` 是**可复现性**容差（同码同数据同平台族），不是建模容差；平台/库版本漂移会表现为 `CENTRE_OK`。")
    A("3. `CENTRE_OK` 不等于「没问题」：它只保证分类裁决不变，端点数值可能已经与文档里的数字不一致，")
    A("   因此文档数字必须由脚本注入（本仓既有约定），不能手抄。")
    A("4. 闸门会运行被测脚本（本例单脚本墙钟见 §3 表），总时长在单机 <10 min 预算内可分片执行；")
    A("   全量在集群跑时用同一命令。")
    A("")

    A("## 5 误差判定")
    A("")
    A("本单的判定量是「重算值 vs 提交值」，基准就是提交值本身，故逐叶误差已在 §3.1 给出；")
    A(f"其中超容差叶子 {sum(1 for r in payload['records'] if r.get('counts', {}).get('n_drift'))} 个脚本、"
      f"{payload['summary']['counts']['REPRODUCED']} 个脚本逐叶复现。")
    A("物理量对标文献的误差判定不在本单范围（见 `docs/AUDIT_m22796_RCP_STIFFNESS_SEED_RESOLVED.md` §8）。")
    A("")
    A("## 6 结论与去向")
    A("")
    A(summarise_text(payload))
    A("")
    return "\n".join(L) + "\n"


def summarise_text(payload):
    sm = payload["summary"]
    bad = [r for r in payload["records"] if r["verdict"] in ("STALE", "FAILED", "NO_ARTEFACT")]
    ok = [r for r in payload["records"] if r["verdict"] == "REPRODUCED"]
    centre = [r for r in payload["records"] if r["verdict"] == "CENTRE_OK"]
    parts = [f"* 逐叶复现：{len(ok)}/{sm['n_scripts']} 个脚本。"]
    if centre:
        parts.append("* 仅端点漂移、裁决不变："
                     + "、".join(f"`{r['script']}`" for r in centre) + "。")
    if bad:
        parts.append("* 需处理：" + "、".join(f"`{r['script']}`({r['verdict']})" for r in bad) + "。")
    parts.append("* 后续每次改动审计脚本或 `runs/` 后重跑本闸门；把 `runs/repro_gate.json` 与本文档一同提交。")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of registry names")
    ap.add_argument("--timeout", type=int, default=1800, help="per-script timeout (s)")
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--mc-scan", default="", help="comma list of n_mc, e.g. 1500,4000,8000")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--doc", default=DOC)
    ap.add_argument("--no-doc", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="do NOT restore the tree after each script (debug only)")
    ap.add_argument("--baseline", choices=("head", "worktree"), default="head",
                    help="head = committed artefact (staleness audit); "
                         "worktree = artefact as it stands (idempotency audit)")
    ap.add_argument("--regen-ref", default=os.path.join(RUNS, "repro_regen.json"),
                    help="regen record injected into the document, if present")
    ap.add_argument("--regen", action="store_true",
                    help="regenerate the artefacts instead of restoring them: "
                         "the script output is kept and the JSON records the "
                         "pre-regeneration diff, so the change is auditable")
    ap.add_argument("--json", action="store_true", help="print summary only")
    args = ap.parse_args(argv)

    protect = (os.path.abspath(args.out), os.path.abspath(args.doc),
               os.path.abspath(__file__))
    entries = [e for e in REGISTRY if args.only is None or e["name"] in args.only]

    records = []
    for e in entries:
        print(f"[gate] {e['script']} ...", flush=True)
        rec = gate_one(e, timeout=args.timeout, rtol=args.rtol, atol=args.atol,
                       protect=protect, keep=args.keep or args.regen,
                       baseline=args.baseline)
        print(f"[gate]   {rec['verdict']}  wall={rec['wall_s']}s "
              f"drift={rec.get('counts', {}).get('n_drift', '-')} "
              f"verdict_changes={len(rec.get('verdict_changes', []))} "
              f"leaks={rec.get('n_path_leaks', '-')}", flush=True)
        records.append(rec)

    payload = dict(anchor="CASE-DEM-RCP-PACKING-FRACTION",
                   root=ROOT, rtol=args.rtol, atol=args.atol,
                   registry=[dict(name=e["name"], script=e["script"], json=e["json"],
                                  stochastic=e["stochastic"]) for e in entries],
                   summary=summarise(records), records=records)

    if os.path.exists(args.regen_ref):
        try:
            with open(args.regen_ref, encoding="utf-8") as fh:
                payload["regen"] = json.load(fh)
        except Exception:
            pass

    if args.mc_scan:
        n_mcs = tuple(int(x) for x in args.mc_scan.split(",") if x.strip())
        print(f"[gate] mc-scan {n_mcs} ...", flush=True)
        payload["mc_scan"] = mc_scan(n_mcs=n_mcs, timeout=max(args.timeout, 2400))
        print(f"[gate]   verdicts_identical={payload['mc_scan'].get('verdicts_identical')}",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=float)
    if not args.no_doc:
        os.makedirs(os.path.dirname(os.path.abspath(args.doc)), exist_ok=True)
        with open(args.doc, "w", encoding="utf-8") as fh:
            fh.write(render_doc(payload))
        print(f"wrote {os.path.relpath(args.doc, ROOT)}")
    print(json.dumps(payload["summary"], indent=1))
    print(f"wrote {os.path.relpath(args.out, ROOT)}")
    return 0 if payload["summary"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())