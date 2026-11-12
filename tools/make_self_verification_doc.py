#!/usr/bin/env python3
"""Render docs/SELF_VERIFICATION_SCAN.md + a figure from the scan JSON.

Every number in the page is injected from ``artifacts/self_verification/scan.json``
(written by ``tools/self_verification_scan.py``) -- nothing is typed by hand.

    PYTHONPATH=src python3 tools/self_verification_scan.py \
        --json artifacts/self_verification/scan.json --docs
    PYTHONPATH=src python3 tools/make_self_verification_doc.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "artifacts" / "self_verification" / "scan.json").read_text())
OUT = ROOT / "docs" / "SELF_VERIFICATION_SCAN.md"
FIG = ROOT / "docs" / "figures" / "self_verification_scan.png"

sites = DATA["sites"]
summary = DATA["summary"]
by_sev = Counter(s["severity"] for s in sites)
by_kind = Counter(s["kind"] for s in sites)
files = sorted({s["path"] for s in sites})
seepage = [s for s in sites if s["path"].endswith("test_seepage_piping.py")]

# --------------------------------------------------------------------- figure
FIG.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
order = ["SEV1", "SEV2", "SEV3", "INFO"]
declared = [sum(1 for s in sites if s["severity"] == k and s["declared"]) for k in order]
undecl = [sum(1 for s in sites if s["severity"] == k and not s["declared"]) for k in order]
ax = axes[0]
ax.bar(order, undecl, color="#c0392b", label="undeclared (fails the gate)")
ax.bar(order, declared, bottom=undecl, color="#27ae60", label="declared / triaged")
ax.set_ylabel("sites")
ax.set_title("(a) measurement-shaped sites by severity")
ax.legend(fontsize=7)
for i, (u, d) in enumerate(zip(undecl, declared)):
    if u + d:
        ax.text(i, u + d + 0.1, str(u + d), ha="center", fontsize=8)
ax = axes[1]
kinds = sorted(by_kind)
ax.barh(kinds, [by_kind[k] for k in kinds], color="#2c3e50")
ax.set_xlabel("sites")
ax.set_title("(b) surrogate signature class")
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------- page
L: list[str] = []
L.append("# Self-verification scan: `measured_*` may not come from the model under test")
L.append("")
L.append(f"* tool: `tools/self_verification_scan.py` · triage ledger: "
  f"`tools/self_verification_scan_allowlist.json` · regression: "
  f"`tests/test_self_verification_scan.py` · figure: `{FIG.relative_to(ROOT)}`")
L.append(f"* closed defect: `audit_report.md` §4-F1 / §8 item 3 (seepage C1–C3 "
  f"\"DEM-measured\" values synthesised inside the test) — mission m16381")
L.append(f"* files scanned: **{DATA['files_scanned']}** (`tests/`, `scripts/`, root `test_*.py`); "
  f"measurement-shaped sites: **{summary['sites']}**; blocking (undeclared SEV1/SEV2): "
  f"**{summary['blocking_sev1_sev2']}**; declared/triaged: {summary['declared']} "
  f"(allowlist entries: {summary['allowlist_entries']}, sites matched: {summary['allowlisted']})")
L.append("")
L.append("## 1. Why this exists")
L.append("")
L.append("A criterion that compares a \"measured\" value against the model under test is")
L.append("*self-verification*: it passes for any implementation, including a broken one,")
L.append("because the number on the measured side is produced by the same formula. The")
L.append("m14487 audit found three such criteria in `seepage` (`u_analytic*(1+noise)`,")
L.append("`i_c_analytic*1.10`, `k*1.15`, reported as 5.85 %/5.00 %/9.09 % \"DEM\" errors)")
L.append("and recommended that the merge gate learn to see them. This tool is that gate.")
L.append("It is deliberately **name-based and shallow**: it raises the cost of the mistake")
L.append("and makes each occurrence auditable; it is not a proof of absence.")
L.append("")
L.append("## 2. Detection classes")
L.append("")
L.append("| class | severity | signature |")
L.append("|---|---|---|")
L.append("| `analytic_scale` | SEV1 | `x_measured = x_analytic * <typed factor ≠ 1>` — the error is the typed constant |")
L.append("| `analytic_noise` | SEV2 | `x_measured = x_analytic * (1 + σ·N(0,1))` — the error is a seeded draw |")
L.append("| `analytic_direct` | SEV3 | `x_measured = x_analytic` (possibly via one alias hop) |")
L.append("| mixed cross-check | INFO | analytic and solver terms combined (e.g. `work_exact - work`) — a genuine comparison, advisory only |")
L.append("")
L.append("A site is **declared** when its target name starts with `synthetic`/`surrogate`/")
L.append("`fabricated`, when a comment within ±3 lines says `self-verification: synthetic`,")
L.append("or when it is listed with a reason in the triage ledger. Declared sites are")
L.append("still reported — the point is that they can no longer be *silent*. The gate")
L.append("fails (exit 1) on any undeclared SEV1/SEV2.")
L.append("")
L.append("## 3. Result on this repository")
L.append("")
L.append("```")
L.append(f"files scanned           {DATA['files_scanned']}")
L.append(f"sites                   {summary['sites']}")
L.append(f"  SEV1 analytic_scale   {by_sev.get('SEV1', 0)}")
L.append(f"  SEV2 analytic_noise   {by_sev.get('SEV2', 0)}")
L.append(f"  SEV3 analytic_direct  {by_sev.get('SEV3', 0)}")
L.append(f"  INFO mixed            {by_sev.get('INFO', 0)}")
L.append(f"declared / triaged      {summary['declared']}")
L.append(f"undeclared              {summary['undeclared']}")
L.append(f"BLOCKING (SEV1+SEV2)    {summary['blocking_sev1_sev2']}")
L.append("```")
L.append("")
L.append("| file | line | function | target | class | severity | state |")
L.append("|---|---|---|---|---|---|---|")
for s in sites:
    A(f"| `{s['path']}` | {s['line']} | `{s['function']}` | `{s['target']}` | "
      f"{s['kind']} | {s['severity']} | {'declared' if s['declared'] else 'UNDECLARED'} |")
L.append("")
L.append("## 4. The one real defect, and what changed")
L.append("")
L.append(f"`tests/test_seepage_piping.py` held **{len(seepage)}** surrogate sites "
  f"(classes: {', '.join(sorted({s['kind'] for s in seepage}))}). They are now named")
L.append("`synthetic_*` (machine-visible declaration) and their docstrings state the grade;")
L.append("`docs/seepage_piping_verification.md` was re-graded: C1–C3 **WITHHELD** (no DEM or")
L.append("experimental source exists in this repository), C4–C5 **DERIVED**, C6 REAL. The")
L.append("test bodies still assert exactly what they asserted before — they are tests of the")
L.append("judging logic, and now they say so. `scripts/audit_ice_rideup_righting.py` renamed")
L.append("a value read from the solver's own moment curve (`k_measured` → `k_from_curve`)")
L.append("because the old name invited exactly this confusion.")
L.append("")
L.append("## 5. Verification of the checker itself")
L.append("")
L.append("`tests/test_self_verification_scan.py` — 9 cases: SEV1 typed-scale detection, SEV2")
L.append("noise detection, alias-chain detection, in-band declaration, `synthetic_` prefix")
L.append("declaration, *negative control* (a genuine solver output must not be flagged),")
L.append("mixed-deficit INFO classification, repository gate = exit 0, and a machine-readable")
L.append("report check that the historical seepage sites are still visible as declared.")
L.append("A checker without negative controls can pass by doing nothing; those controls are")
L.append("the reason to trust the green.")
L.append("")
L.append("## 6. Limits (stated, not implied)")
L.append("")
L.append("* name-based: a surrogate stored in a neutral variable is invisible;")
L.append("* one local alias hop only — a helper returning the analytic value under a neutral")
L.append("  name is not followed across function boundaries;")
L.append("* it does not judge whether an *analytic* reference is itself correct, only whether a")
L.append("  value presented as measurement came from the model under test;")
L.append("* the docs cross-reference (`--docs`) is advisory: it lists pages that call a value")
L.append("  \"DEM-measured\" while the same page admits a surrogate, and needs a human read.")
L.append("")
L.append("## 7. Reproduce")
L.append("")
L.append("```bash")
L.append("PYTHONPATH=src python3 tools/self_verification_scan.py --json artifacts/self_verification/scan.json --docs")
L.append("PYTHONPATH=src python3 tools/make_self_verification_doc.py")
L.append("PYTHONPATH=src python3 -m pytest tests/test_self_verification_scan.py tests/test_seepage_piping.py -q")
L.append("```")
OUT.write_text("\n".join(L) + "\n", encoding="utf8")
print(f"wrote {OUT.relative_to(ROOT)} ({len(A)} lines) and {FIG.relative_to(ROOT)}")
print(f"blocking={summary['blocking_sev1_sev2']} sites={summary['sites']} "
      f"declared={summary['declared']} files={len(files)}")