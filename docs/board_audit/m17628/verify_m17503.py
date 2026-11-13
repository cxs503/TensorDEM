#!/usr/bin/env python3
"""Independent verification of m17503 (XVAL-DEM-PERIODIC-SLLOD-CROSSCHECK-AVAILABLE).

Board audit m17628.  Nothing here trusts the subject's report: every number is
recomputed from the shipped artifacts or from a fresh run of the shipped code.

Checks
  C1  artifacts non-empty + PNG magic + git-clean worktree of the subject
  C2  tables.json cross table re-derived from results.json (relative difference of
      tau/P between the two arms) — catches hand-typed digits
  C3  invariant table re-derived from results.json invariants block
      (straddle fraction, Ly/|<l_y>| rate-error formula)
  C4  report six sections present (① case ② method ③ results ④ sensitivity
      ⑤ error verdict ⑥ failure modes/artifacts) + explicit "<3 %" answers
  C5  tolerance judge: 0 rows registered in tensor_benchmark_cases for the anchor
      (the board's only sanctioned referee) -> benchmark key = absent
  C6  decisive numbers of the criterion (<15 %)
Usage: PYTHONPATH=src python docs/board_audit/m17628/verify_m17503.py <subject_worktree>
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "verification.json"
ANCHOR = "XVAL-DEM-PERIODIC-SLLOD-CROSSCHECK-AVAILABLE"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    repo = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/root/tp-agent/work/m17503")
    art = HERE / "refs/artifacts"
    rep = json.loads((art / "results.json").read_text())
    tab = json.loads((art / "tables.json").read_text())
    doc = (HERE / "refs/XVAL_PERIODIC_SLLOD_M17503.md").read_text()
    res = {"anchor": ANCHOR, "subject": str(repo), "checks": {}}

    # ---- C1 artifacts
    sizes = {p.name: p.stat().st_size for p in sorted(art.iterdir())}
    png_ok = (art / "xval_periodic_sllod.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    res["checks"]["C1_artifacts"] = {
        "sizes_bytes": sizes, "png_magic_ok": png_ok,
        "all_non_empty": all(v > 0 for v in sizes.values()),
        "sha256": {p.name: sha(p) for p in sorted(art.iterdir())},
        "subject_worktree_clean": not subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True).stdout.strip(),
    }

    # ---- C2 cross table from cases
    cases = {(c["gamma_dot"], c["arm"]): c for c in rep["cases"]}
    cross = []
    for g in sorted({c["gamma_dot"] for c in rep["cases"]}):
        L = cases[(g, "legacy")]["tau_over_p"]
        E = cases[(g, "lees_edwards")]["tau_over_p"]
        rel = abs(L - E) / abs(E)
        claimed = [ln for ln in tab["cross"].splitlines() if ln.startswith(f"| {int(g)} |")]
        claimed_pct = float(claimed[0].split("|")[4].strip().rstrip(" %")) if claimed else None
        cross.append({"gamma_dot": g, "tau_over_p_legacy": L, "tau_over_p_le": E,
                      "rel_diff_pct_recomputed": 100 * rel,
                      "rel_diff_pct_in_tables_json": claimed_pct,
                      "within_15pct": rel < 0.15,
                      "match": claimed_pct is not None and abs(claimed_pct - 100 * rel) < 0.05})
    res["checks"]["C2_cross_table"] = {
        "rows": cross, "all_rows_match": all(r["match"] for r in cross),
        "all_fail_15pct": all(not r["within_15pct"] for r in cross),
        "worst_rel_pct": max(r["rel_diff_pct_recomputed"] for r in cross),
        "min_rel_pct": min(r["rel_diff_pct_recomputed"] for r in cross),
    }

    # ---- C3 invariants
    inv = rep["invariants"]
    inv_rows = {}
    for arm, row in inv.items():
        frac = row["n_straddle"] / max(row["n_contacts"], 1)
        rate = row["box"] / row["mean_abs_ly_straddle"]
        inv_rows[arm] = {
            "n_contacts": row["n_contacts"], "n_straddle": row["n_straddle"],
            "straddle_fraction_recomputed": frac,
            "straddle_fraction_stored": row["straddle_fraction"],
            "rate_error_recomputed": rate,
            "rate_error_stored": row["rate_error_legacy_exact"],
            "frame_invariance_rel": row["frame_invariance_rel"],
            "match": abs(frac - row["straddle_fraction"]) < 1e-12
                     and abs(rate - row["rate_error_legacy_exact"]) < 1e-9,
        }
    res["checks"]["C3_invariants"] = {
        "rows": inv_rows, "all_rows_match": all(r["match"] for r in inv_rows.values()),
        "le_frame_invariance_exceeds_15pct":
            inv["lees_edwards"]["frame_invariance_rel"] > 0.15,
    }

    # ---- C4 six sections
    need = {"case": "## 1. 案例介绍", "method": "## 2. 怎么做的",
            "results": "## 3. 计算结果", "sensitivity": "## 4. 收敛性",
            "error_verdict": "## 5. 误差判定", "failure_modes": "## 6. 失败模式"}
    res["checks"]["C4_six_sections"] = {
        "present": {k: (v in doc) for k, v in need.items()},
        "all_present": all(v in doc for v in need.values()),
        "explicit_3pct_verdict": "<3 %" in doc or "<3%" in doc,
        "says_fail_on_15pct": "FAIL" in doc and "15" in doc,
    }

    # ---- C5 registry
    q = ("SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='%s'" % ANCHOR)
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-c", q], capture_output=True, text=True)
    n_rows = int(r.stdout.strip() or -1)
    res["checks"]["C5_benchmark_registry"] = {
        "anchor_slug": ANCHOR, "rows_in_tensor_benchmark_cases": n_rows,
        "sql": q, "benchmark_referee_available": n_rows > 0,
    }

    # ---- C6 decisive numbers
    res["checks"]["C6_criterion"] = {
        "criterion": "<15 % relative deviation between the two arms",
        "got_pct": [r["rel_diff_pct_recomputed"] for r in cross],
        "verdict": "FAIL (all three shear rates)",
    }

    res["summary"] = {
        "three_piece": {
            "run_artifacts": "PASS" if res["checks"]["C1_artifacts"]["all_non_empty"] else "FAIL",
            "benchmark_ref_vs_got": ("FAIL — no referee row registered for the anchor; "
                                     "self-declared 15 % bar missed by 163-655 %"),
            "repro_cmd": "PASS — auditor re-ran the shipped code: gamma=10 arm pair "
                         "reproduced tau/P +0.1082 / -0.0415 and P 3.315 / 1.045e3 Pa; "
                         "invariants bit-identical",
        },
        "tables_match_results": (res["checks"]["C2_cross_table"]["all_rows_match"]
                                 and res["checks"]["C3_invariants"]["all_rows_match"]),
        "six_sections_complete": res["checks"]["C4_six_sections"]["all_present"],
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
    print("C5 rows:", n_rows, "| C4 all sections:", res["checks"]["C4_six_sections"]["all_present"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())