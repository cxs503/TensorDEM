#!/usr/bin/env python3
"""m17588 board audit — machine verification of m17500's three-piece evidence.

Checks (all programmatic, zero hand-copy):
  A. branch/merge state of the m17500 deliverables (git, both worktrees)
  B. run_artifacts existence + sha256 (artifacts/xval_mu_i_contact_proxy_m17500/)
  C. internal arithmetic of results.json: recompute i_mapping_error / mu_error /
     shortfall from the primitives, compare with the stored columns
  D. prereg criteria applied to the stored rows (DEFAULT_TOL from module)
  E. report #16473 numerals vs results.json (markdown table parse)
  F. registered referee: tensor_benchmark_cases rows for the anchor
Outputs JSON to stdout.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, "/root/tp-agent/work/m17500/src")
from tensordem.mu_i_contact_proxy import DEFAULT_TOL, jop_bulk_mu  # noqa: E402

MU_C = 0.5   # ServoShearCase.mu of the DEM contact law
W = "/root/tp-agent/work/m17500"
MAIN = "/root/tp-agent/work/m17588"
ANCHOR = "XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE"
ART = "artifacts/xval_mu_i_contact_proxy_m17500"
ENV = "/root/tensor-ecosystem/TensorPlatform/.env"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dsn():
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace(
                "postgresql+psycopg://", "postgresql://")
    raise SystemExit("no DATABASE_URL")


def git(cwd, *args):
    try:
        return subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return "ERR:" + str(e)[:80]


def git_rc(cwd, *args):
    """(returncode, stdout) -- for predicates such as merge-base --is-ancestor."""
    try:
        p = subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                           text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()[:200]
    except Exception as e:  # noqa: BLE001
        return -1, str(e)[:200]


def main():
    out = {"anchor": ANCHOR}
    # ---------- A. deliverable location ----------
    out["git"] = {
        "mission_branch": git(W, "branch", "--show-current"),
        "mission_head": git(W, "log", "-1", "--format=%H %s"),
        "mission_commit_files": git(W, "show", "--name-only", "--format=", "HEAD"
                                    ).split(),
        "worktree_clean": git(W, "status", "--porcelain") == "",
        "main_head": git(MAIN, "log", "-1", "--format=%H %s"),
        "in_main": {f: os.path.exists(os.path.join(MAIN, f)) for f in (
            "src/tensordem/mu_i_contact_proxy.py",
            "tests/test_xval_mu_i_contact_proxy.py",
            "scripts/xval_mu_i_contact_proxy_m17500.py",
            "docs/xval_mu_i_contact_proxy_m17500.md",
            "prereg/xval_mu_i_contact_proxy_m17500.md",
            ART + "/results.json")},
        "merged_into_main_rc": git_rc(MAIN, "merge-base", "--is-ancestor",
                                      "agent/m17500", "HEAD")[0],
        "merged_into_main": git_rc(MAIN, "merge-base", "--is-ancestor",
                                   "agent/m17500", "HEAD")[0] == 0,
        "pushed_to_origin": git(W, "ls-remote", "--heads", "origin",
                                "agent/m17500"),
        "brach_in_main_log": git(MAIN, "log", "--all", "--oneline",
                                 "--grep=m17500").splitlines()[:3],
    }

    # ---------- B. artifacts ----------
    art_dir = os.path.join(W, ART)
    out["artifacts"] = {}
    for name in sorted(os.listdir(art_dir)) if os.path.isdir(art_dir) else []:
        p = os.path.join(art_dir, name)
        out["artifacts"][name] = {"bytes": os.path.getsize(p), "sha256": sha(p)}

    res = json.load(open(os.path.join(art_dir, "results.json")))
    out["results_meta"] = {k: res[k] for k in
                           ("anchor", "mission", "tolerance", "wall_clock_total_s")}
    out["results_meta"]["ref_source"] = res["ref_source"]

    # ---------- C. internal arithmetic ----------
    rows = res["rows"]
    arith = []
    for r in rows:
        ib, mb = r["i_bulk_measured"], r["mu_bulk_measured"]
        ic, mp = r["i_proxy_fn_weighted"], r["mu_proxy_fn_weighted"]
        rec_i = abs(ic / ib - 1.0)
        rec_mu = abs(mp / mb - 1.0)
        rec_sf = (MU_C - jop_bulk_mu(ic)) / MU_C   # mu_c = 0.5 contact law
        arith.append({
            "label": r["label"],
            "i_mapping_error_stored": r["i_mapping_error"],
            "i_mapping_error_recomputed": rec_i,
            "i_match_1e-9": abs(rec_i - r["i_mapping_error"]) < 1e-9,
            "mu_error_stored": r["mu_error"],
            "mu_error_recomputed": rec_mu,
            "mu_match_1e-9": abs(rec_mu - r["mu_error"]) < 1e-9,
            "shortfall_stored": r["shortfall"],
            "shortfall_recomputed": rec_sf,
            "shortfall_match_1e-9": abs(rec_sf - r["shortfall"]) < 1e-9
            if not math.isnan(r["shortfall"]) else None,
            "servo_valid": r["servo_valid"],
            "pressure_control_error_final": r["pressure_control_error_final"],
            "bulk_vs_jop_rel_error": r["bulk_vs_jop_rel_error"],
            "n_contacts": r["n_contacts"],
            "n_proxy_samples": r["n_proxy_samples"],
        })
    out["arithmetic"] = arith

    # ---------- D. prereg criteria ----------
    tol = res["tolerance"]
    assert abs(DEFAULT_TOL - tol) < 1e-12, (DEFAULT_TOL, tol)
    worst_map = max(r["i_mapping_error"] for r in rows)
    worst_mu = max(r["mu_error"] for r in rows)
    worst_ref = max(r["bulk_vs_jop_rel_error"] for r in rows)
    out["prereg_check"] = {
        "tol": tol,
        "worst_i_mapping_error": worst_map,
        "worst_mu_error": worst_mu,
        "worst_bulk_vs_jop_rel_error": worst_ref,
        "criterion_mapping_all_under_tol": worst_map < tol,
        "criterion_friction_all_under_tol": worst_mu < tol,
        "reference_gate_pass": worst_ref < tol,
        "reference_gate_voids_absolute_mu": worst_ref >= tol,
        "rows_servo_invalid": sum(1 for r in rows if not r["servo_valid"]),
        "n_rows": len(rows),
    }

    # ---------- E. report numerals vs results.json ----------
    import psycopg
    with psycopg.connect(dsn()) as c, c.cursor() as cc:
        cc.execute("SELECT markdown, title, mission_id FROM tensor_reports WHERE id=16473")
        md, title, mid = cc.fetchone()
        cc.execute(
            "SELECT id, anchor_slug, case_name, metric, tolerance, status "
            "FROM tensor_benchmark_cases WHERE anchor_slug=%s", (ANCHOR,))
        out["benchmark_cases_for_anchor"] = cc.fetchall()
        cc.execute("SELECT to_regclass('tensor_benchmark_cases')")
        out["referee_table_present"] = cc.fetchone()[0] is not None
        cc.execute("SELECT count(*) FROM tensor_benchmark_cases WHERE repo='TensorDEM'")
        out["tensordem_registered_cases"] = cc.fetchone()[0]

    out["report"] = {"id": 16473, "title": title, "mission_id": mid,
                     "sections": {k: (k in md) for k in
                                  ("① 案例介绍", "② 怎么", "③ 计算结果", "④ 结果图",
                                   "⑤ 注意点", "⑥ 误差判定")},
                     "chars": len(md)}

    # parse the ③ table: | label | I_bulk | mu_bulk | I_c | mu_proxy | map% | mu% | sf% |
    tbl = {}
    for line in md.splitlines():
        if line.startswith("| base_") or line.startswith("| sens_"):
            cells = [x.strip() for x in line.strip("|").split("|")]
            tbl[cells[0]] = cells
    cmp_rows = []
    for r in rows:
        cells = tbl.get(r["label"])
        if not cells:
            cmp_rows.append({"label": r["label"], "in_report_table": False})
            continue
        def pct(x):
            return abs(float(cells[x].replace("%", "").replace("+", "").replace("−", "-"))
                       / 100.0)
        cmp_rows.append({
            "label": r["label"], "in_report_table": True,
            "report_i_bulk": float(cells[1]), "json_i_bulk": r["i_bulk_measured"],
            "d_i_bulk": abs(float(cells[1]) / r["i_bulk_measured"] - 1),
            "report_map_pct": cells[5], "json_map_pct": round(100 * r["i_mapping_error"], 1),
            "map_pct_agrees_0p1": abs(pct(5) - r["i_mapping_error"]) < 1e-3,
            "report_mu_pct": cells[6], "json_mu_pct": round(100 * r["mu_error"], 1),
            "mu_pct_agrees_0p1": abs(pct(6) - r["mu_error"]) < 1e-3,
        })
    out["report_vs_json"] = cmp_rows

    # ---------- F. thresholds the report itself quotes ----------
    out["verdict_claims_in_json"] = res["summary_all_runs"]
    out["json_has_nan_literals"] = "NaN" in open(
        os.path.join(art_dir, "results.json")).read()

    json.dump(out, open("/tmp/m17588_verify.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()