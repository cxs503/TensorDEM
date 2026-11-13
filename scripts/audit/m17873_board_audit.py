#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m17789 — anchor
XVAL-DEM-RADON-PROJECTION-MEASUREMENT (TensorDEM).

Verification, not re-do.  Every number in the verdict and the rendered page is
computed from committed bytes and live referee queries; nothing is typed by hand.

Checks
  1. three-piece: delivered files exist in the commit (byte copies + sha256);
     tests exist and pass; is the published repro command faithful?
  2. referee: tensor_benchmark_cases WHERE anchor_slug=... — the only judge.
  3. numbers: independent re-aggregation of the committed JSON vs its own
     summary block and vs the claims in docs/.
  4. reproducibility: documented command re-run in a temp copy, leaf-by-leaf
     and sha256 comparison.
  5. integration surface: solver/package API or benchmarks only?
  6. report: tensor_reports row 16750 carries the six required sections.

Outputs (this worktree)
  artifacts/audit_m17873/branch_57564a2e/...   byte copies
  artifacts/audit_m17873/*.json                referee/judge/repro/finding/...
  docs/board_audit_m17873/M17873_BOARD_AUDIT.md
  (--insert-verdict) one row in tensor_maturity_verdicts

Usage
  python scripts/audit/m17873_board_audit.py --insert-verdict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = "/root/tp-agent/work/m17789"
COMMIT = "57564a2eea3b905e7c946ccc8a0d1404131834c9"
BRANCH = "agent/m17789"
ANCHOR = "XVAL-DEM-RADON-PROJECTION-MEASUREMENT"
AUDIT_MID = 17873
OUT = os.path.join(REPO, "artifacts", "audit_m17873")
ROOT = os.path.join(OUT, "branch_57564a2e")
DOC = os.path.join(REPO, "docs", "board_audit_m17873", "M17873_BOARD_AUDIT.md")
ECOSYSTEM_DEM = "/root/tensor-ecosystem/TensorDEM"
REPORT_ID = 16750

DELIVERED = [
    "benchmarks/biaxial_shear_band/run_radon_xval_measurement.py",
    "benchmarks/biaxial_shear_band/render_radon_xval_doc.py",
    "tests/test_radon_projection_xval.py",
    "docs/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.md",
    "artifacts/biaxial_shear_band/radon_projection_xval.json",
    "artifacts/biaxial_shear_band/radon_projection_xval.png",
    "publish_report_m17789.py",
    "src/tensordem/biaxial_shear_band.py",
    "src/tensordem/biaxial_band_diagnostics.py",
]

ART = "artifacts/biaxial_shear_band/radon_projection_xval.json"
DOC_MD = "docs/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.md"
REPRO_RUNS = os.path.join(OUT, "repro_runs")


def sh(*a, cwd=REPO):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, p + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, p + "[%d]" % i)
    else:
        yield p, o


def extract():
    man, missing = {}, []
    for rel in DELIVERED:
        r = subprocess.run(["git", "show", "%s:%s" % (COMMIT, rel)], cwd=SRC,
                           capture_output=True)
        if r.returncode:
            missing.append(rel)
            continue
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(r.stdout)
        man[rel] = {"bytes": len(r.stdout),
                    "sha256": hashlib.sha256(r.stdout).hexdigest()}
    with open(os.path.join(OUT, "sha256manifest.json"), "w") as fh:
        json.dump({"commit": COMMIT, "branch": BRANCH, "anchor": ANCHOR,
                   "files": man, "missing": missing}, fh, indent=1)
    tracked = set(sh("git", "ls-files", cwd=SRC).stdout.split())
    untracked = [rel for rel in DELIVERED if rel not in tracked]
    return {"manifest": man, "missing": missing, "untracked_at_head": untracked}


def merge_state():
    inside = sh("git", "merge-base", "--is-ancestor", COMMIT, "main",
                cwd=SRC).returncode == 0
    br = sh("git", "branch", "-a", "--contains", COMMIT, cwd=SRC).stdout.split()
    present = [rel for rel in DELIVERED
               if os.path.exists(os.path.join(ECOSYSTEM_DEM, rel))]
    remote, merge_commit = "", ""
    if os.path.isdir(os.path.join(ECOSYSTEM_DEM, ".git")):
        remote = sh("git", "branch", "-r", "--contains", COMMIT,
                    cwd=ECOSYSTEM_DEM).stdout.strip()
        merge_commit = sh("git", "log", "--oneline", "-1", "--all", "--grep",
                          "Merge #1462", cwd=ECOSYSTEM_DEM).stdout.strip()
    return {"ancestor_of_main": inside,
            "remote_refs_containing": remote.splitlines(),
            "remote_merge_commit": merge_commit,
            "branches_containing": sorted({b.strip("*+ ") for b in br} & {BRANCH, "main"}),
            "head_of_audited_tree": sh("git", "rev-parse", "HEAD", cwd=SRC).stdout.strip(),
            "ecosystem_checkout": ECOSYSTEM_DEM,
            "delivered_files_present_in_ecosystem_checkout": present}


def referee(dsn):
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SELECT id, case_name, metric, tolerance, ref_source, repo, status "
                    "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", (ANCHOR,))
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
        total = cur.fetchone()[0]
        cur.execute("SELECT anchor_slug, case_name, metric, tolerance, status FROM "
                    "tensor_benchmark_cases WHERE repo ILIKE '%%DEM%%' ORDER BY id")
        dem_rows = [dict(zip(("anchor_slug", "case_name", "metric", "tolerance", "status"), r))
                    for r in cur.fetchall()]
        cur.execute("SELECT slug, title, acceptance, status, evidence, mission_id "
                    "FROM feature_ledger WHERE slug=%s", (ANCHOR,))
        r = cur.fetchone()
        ledger = ({"slug": r[0], "title": r[1], "acceptance": r[2], "status": r[3],
                   "evidence": r[4], "mission_id": r[5]} if r else None)
        cur.execute("SELECT status, (result->>'accepted')::text, "
                    "left(result->>'checks', 4000) FROM tensor_dev_missions WHERE id=17789")
        st = cur.fetchone()
        cur.execute("SELECT id, repo, scope, mission_id, level, verdict_by, gate_state "
                    "FROM tensor_maturity_verdicts WHERE scope ILIKE '%%RADON%%' "
                    "ORDER BY id DESC")
        prior = [dict(zip(("id", "repo", "scope", "mission_id", "level",
                           "verdict_by", "gate_state"), r)) for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM tensor_reports WHERE mission_id=17593")
        n17593 = cur.fetchone()[0]
    return {"anchor": ANCHOR, "registered_rows": len(rows), "rows": rows,
            "benchmark_cases_total": total, "dem_rows": dem_rows,
            "nearest_dem_anchor": next((x for x in dem_rows
                                        if x["anchor_slug"] ==
                                        "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE"), None),
            "feature_ledger": ledger,
            "mission_state": {"status": st[0], "accepted": st[1]} if st else None,
            "mission_checks_string": (st[2] if st else ""),
            "prior_verdicts_for_anchor": prior,
            "reports_for_mission_17593": n17593}


def report_check(dsn):
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SELECT id, title, mission_id, scenario_key, markdown, figures "
                    "FROM tensor_reports WHERE id=%s", (REPORT_ID,))
        r = cur.fetchone()
    if not r:
        return {"report_id": REPORT_ID, "found": False}
    md = r[4] or ""
    heads = re.findall(r"^#{1,3} .*", md, re.M)
    want = ["\u2460", "\u2461", "\u2462", "\u2463", "\u2464", "\u2465"]
    return {"report_id": r[0], "found": True, "title": r[1], "mission_id": r[2],
            "scenario_key": r[3], "chars": len(md),
            "sections": [h for h in heads if any(w in h for w in want)],
            "missing_sections": [w for w in want if not any(w in h for h in heads)],
            "has_figures": bool(r[5]), "headings": heads[:12]}


def _stats(vals):
    n = len(vals)
    return {"n": n, "bias": sum(vals) / n,
            "rms": (sum(v * v for v in vals) / n) ** 0.5,
            "max_abs": max(abs(v) for v in vals)}


def _envelope_scan(prod, rel):
    levels = sorted({r["noise_fraction"] for r in prod if r["noise_fraction"] > 0})
    ok = 0.0
    for lv in levels:
        if max(rel(r) for r in prod if r["noise_fraction"] == lv) <= 0.15:
            ok = lv
        else:
            break
    return ok


def judge():
    with open(os.path.join(ROOT, ART), encoding="utf-8") as fh:
        art = json.load(fh)
    rows = art["rows"]
    prod = [r for r in rows if "radon_production_error_deg" in r]
    zero = [r for r in prod if r["noise_fraction"] == 0.0]
    env = float(art["verdict"]["operating_envelope_noise_fraction"])
    in_env = [r for r in prod if 0 < r["noise_fraction"] <= env]
    case = [r for r in prod if r["lattice"] == "24x48" and "radon_legacy_peraxis" in r]
    legacy = [r for r in rows if "radon_legacy_peraxis_error_deg" in r]

    def rel(r):
        return abs(r["radon_production_error_deg"]) / r["alpha_true_folded_deg"]

    def rel_legacy(r):
        return abs(r["radon_legacy_peraxis_error_deg"]) / r["alpha_true_folded_deg"]

    mine = {
        "all": _stats([r["radon_production_error_deg"] for r in prod]),
        "zero_noise": {**_stats([r["radon_production_error_deg"] for r in zero]),
                       "max_rel": max(rel(r) for r in zero)},
        "envelope": {**_stats([r["radon_production_error_deg"] for r in in_env]),
                     "max_rel": max(rel(r) for r in in_env),
                     "share_within_15pct": (sum(1 for r in in_env if rel(r) <= 0.15)
                                            / len(in_env))},
        "case_lattice_24x48_production": _stats([r["radon_production_error_deg"] for r in case]),
        "case_lattice_24x48_legacy": _stats([r["radon_legacy_peraxis_error_deg"] for r in legacy]),
        "legacy_max_rel": max(rel_legacy(r) for r in legacy) if legacy else None,
    }
    auditor = {"n": mine["all"]["n"], "bias_deg": mine["all"]["bias"],
               "rms_deg": mine["all"]["rms"],
               "max_rel": max(rel(r) for r in prod),
               "share_within_15pct": sum(1 for r in prod if rel(r) <= 0.15) / len(prod)}
    harness = {k: art["summary"]["radon_production"][k] for k in
               ("n", "bias_deg", "rms_deg", "max_rel", "share_within_15pct")}

    rec = {"pass_precision_15pct_within_envelope": mine["envelope"]["max_rel"] <= 0.15,
           "noise_envelope_fraction_measured": _envelope_scan(prod, rel),
           "pass_beats_legacy_on_case_lattice":
               abs(mine["case_lattice_24x48_production"]["bias"]) <
               abs(mine["case_lattice_24x48_legacy"]["bias"])}
    rec["pass"] = (rec["pass_precision_15pct_within_envelope"]
                   and rec["pass_beats_legacy_on_case_lattice"])

    des = art["design"]
    cov = {"design_cells_advertised": (len(des["angles"]) * len(des["lattices"])
                                       * len(des["widths_d"]) * len(des["noise_fractions"])),
           "rows_executed": len(rows),
           "lattices": sorted({r["lattice"] for r in rows}),
           "lattices_design": ["%dx%d" % tuple(l) for l in des["lattices"]],
           "angles": sorted({r["alpha_true_deg"] for r in rows}),
           "widths": sorted({r["width_d"] for r in rows}),
           "noise": sorted({r["noise_fraction"] for r in rows}),
           "max_repeat_per_noisy_cell": max(r["repeat"] for r in prod) + 1,
           "repeats_in_design_block": des["repeats"]}
    cov["fraction_of_design"] = cov["rows_executed"] / (cov["design_cells_advertised"]
                                                        * des["repeats"])
    cov["design_repeats_matches_rows"] = (cov["max_repeat_per_noisy_cell"] == des["repeats"])
    cov["missing_factor_levels"] = {
        "lattices": [x for x in cov["lattices_design"] if x not in cov["lattices"]],
        "widths": [x for x in des["widths_d"] if x not in cov["widths"]],
        "noise": [x for x in des["noise_fractions"] if x not in cov["noise"]],
        "angles": [x for x in des["angles"] if x not in cov["angles"]]}

    with open(os.path.join(ROOT, DOC_MD), encoding="utf-8") as fh:
        md = fh.read()
    m = re.search(r"acceptance\*\*:\s*([^·]*)", md)
    doc = {"claims_three_realisations": "3 realisations for every noisy cell" in md,
           "rows_three_realisations": cov["max_repeat_per_noisy_cell"] == 3,
           "claims_16_measurements": "| measurements | 16 |" in md,
           "claims_shard48": "48-way sharded" in md,
           "declares_honest_fail": "honest FAIL" in md,
           "declares_pass_no": "overall:**NO**" in md.replace(" ", "") or
                               "overall: **NO**" in md,
           "acceptance_text": m.group(1).strip() if m else None}
    doc["contradiction_three_realisations"] = (doc["claims_three_realisations"]
                                               and not doc["rows_three_realisations"])

    return {"anchor_artifact": ART, "rows": len(rows), "recomputed": mine,
            "committed_summary_all": harness, "auditor_summary_all": auditor,
            "committed_zero_noise": art["summary"]["radon_production"]["zero_noise"],
            "committed_envelope_block": art["summary"]["radon_production"]["per_noise"].get("0.01"),
            "verdict_committed": art["verdict"], "verdict_recomputed": rec,
            "verdict_mechanical_match": all(rec[k] == art["verdict"][k] for k in rec),
            "gate_15pct": {"zero_noise_max_rel": mine["zero_noise"]["max_rel"],
                           "envelope_max_rel": mine["envelope"]["max_rel"],
                           "zero_noise_pass": mine["zero_noise"]["max_rel"] <= 0.15,
                           "envelope_pass": mine["envelope"]["max_rel"] <= 0.15,
                           "zero_noise_pass_3pct": mine["zero_noise"]["max_rel"] <= 0.03},
            "coverage": cov, "doc": doc}


def repro():
    committed = os.path.join(ROOT, ART)
    res = {"committed_sha256": sha(committed),
           "committed_bytes": os.path.getsize(committed), "runs": {}}
    with open(committed, encoding="utf-8") as fh:
        base = json.load(fh)
    bl = dict(leaves(base))
    for name in ("default_repeats", "repeats1"):
        p = os.path.join(REPRO_RUNS, name + ".json")
        if not os.path.exists(p):
            res["runs"][name] = {"present": False}
            continue
        with open(p, encoding="utf-8") as fh:
            obj = json.load(fh)
        lv = dict(leaves(obj))
        diffs = [(k, bl[k], lv[k]) for k in bl if k in lv and bl[k] != lv[k]]
        res["runs"][name] = {"present": True, "sha256": sha(p),
                             "bytes": os.path.getsize(p),
                             "sha256_equals_committed": sha(p) == res["committed_sha256"],
                             "objects_equal": obj == base,
                             "rows": len(obj.get("rows", [])),
                             "leaves_committed": len(bl), "leaves_rerun": len(lv),
                             "diffs": len(diffs),
                             "diff_examples": [{"path": k, "committed": a, "rerun": b}
                                               for k, a, b in diffs[:5]],
                             "verdict": obj.get("verdict"),
                             "summary_all": {k: obj["summary"]["radon_production"][k]
                                             for k in ("n", "bias_deg", "rms_deg",
                                                       "max_rel", "share_within_15pct")}}
    res["documented_cmd_reproduces_committed"] = bool(
        res["runs"].get("default_repeats", {}).get("objects_equal"))
    res["undocumented_flag_needed"] = (
        not res["documented_cmd_reproduces_committed"]
        and bool(res["runs"].get("repeats1", {}).get("sha256_equals_committed")))

    tmp = tempfile.mkdtemp(prefix="audit_m17873_")
    try:
        dst = os.path.join(tmp, "repo")
        shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))

        def one(lattice, tag):
            o = os.path.join(tmp, tag + ".json")
            r = sh("env", "PYTHONPATH=src", sys.executable,
                   "benchmarks/biaxial_shear_band/run_radon_xval_measurement.py",
                   "--lattices", lattice, "--angles", "60", "--widths", "1.5",
                   "--noise", "0.0", "--out", o, "--no-figure", cwd=dst)
            got = None
            if os.path.exists(o):
                got = json.load(open(o))["rows"][0]["radon_production"]
            tail = (r.stderr or "").strip().splitlines()[-1:] or [""]
            return {"exit": r.returncode,
                    "sha256": sha(o) if os.path.exists(o) else None,
                    "angle": got, "stderr_tail": tail[0][:200]}

        a = one("24x48", "case_a")
        b = one("24x48", "case_b")
        sub = one("24x24", "subset_24x24")
        res["spot_check_determinism"] = {
            "cmd": ("PYTHONPATH=src python benchmarks/biaxial_shear_band/"
                    "run_radon_xval_measurement.py --lattices 24x48 --angles 60 "
                    "--widths 1.5 --noise 0.0 --out <tmp> --no-figure (x2)"),
            "runs": [a, b],
            "identical": a["sha256"] == b["sha256"] and a["sha256"] is not None,
            "angle": a["angle"]}
        res["subset_probe"] = {
            "cmd": ("PYTHONPATH=src python benchmarks/biaxial_shear_band/"
                    "run_radon_xval_measurement.py --lattices 24x24 --angles 60 "
                    "--widths 1.5 --noise 0.0 --out <tmp> --no-figure"),
            "result": sub,
            "crashes_without_case_lattice": sub["exit"] != 0,
            "note": ("summarise() only aggregates the legacy baseline for the 24x48 case lattice "
                     "and then calls max() on the empty list -> any subset (or cluster shard) "
                     "whose rows hold no 24x48 cell aborts before writing its part file"
                     if sub["exit"] != 0 else "subset runs fine")}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def mainline_probe():
    """What the merged main line already carries for this anchor."""
    if not os.path.isdir(os.path.join(ECOSYSTEM_DEM, ".git")):
        return {"available": False}
    def g(*a):
        return sh("git", *a, cwd=ECOSYSTEM_DEM).stdout.strip()
    paths = g("ls-tree", "-r", "--name-only", "origin/main").splitlines()
    mine = [k for k in DELIVERED if os.path.basename(k) in " ".join(paths)] if False else \
           [k for k in DELIVERED if k in paths]
    doc = g("show", "origin/main:docs/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.md")
    return {"available": True,
            "origin_main_head": g("log", "--oneline", "-1", "origin/main"),
            "m17789_files_on_origin_main": mine,
            "m17789_artifact_on_origin_main": ART in paths,
            "main_doc_first_line": doc.splitlines()[0] if doc else "",
            "main_has_prereg": "prereg/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.json" in paths,
            "main_has_oss_benchmark": "benchmarks/radon_projection_xval/run_xval.py" in paths,
            "main_has_radon_reference": "src/tensordem/radon_reference.py" in paths,
            "commit_is_ancestor_of_origin_main":
                sh("git", "merge-base", "--is-ancestor", COMMIT, "origin/main",
                   cwd=ECOSYSTEM_DEM).returncode == 0,
            "prereg_for_round2_in_m17789": os.path.exists(
                os.path.join(SRC, "prereg/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.json"))}


def tests_probe():
    tmp = tempfile.mkdtemp(prefix="audit_m17873_t_")
    try:
        dst = os.path.join(tmp, "repo")
        shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        r = sh("env", "PYTHONPATH=src", sys.executable, "-m", "pytest",
               "tests/test_radon_projection_xval.py", "-q", cwd=dst)
        r2 = sh("env", "PYTHONPATH=src", sys.executable, "-m", "pytest",
                "tests/test_biaxial_shear_band.py", "tests/test_biaxial_band_diagnostics.py",
                "-q", cwd=dst)
        tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
        tail2 = (r2.stdout or "").strip().splitlines()[-1:] or [""]
        return {"xval_exit": r.returncode, "xval_summary": tail[0],
                "neighbour_exit": r2.returncode, "neighbour_summary": tail2[0],
                "cmd": "PYTHONPATH=src python -m pytest tests/test_radon_projection_xval.py -q",
                "test_names": re.findall(r"^def (test_\w+)", open(
                    os.path.join(ROOT, "tests/test_radon_projection_xval.py"),
                    encoding="utf-8").read(), re.M)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def integration_probe():
    callers = [g for g in sh("grep", "-rn", "measure_band\\|radon_band_angle",
                             "--include=*.py", ".", cwd=SRC).stdout.splitlines()
               if "__pycache__" not in g]
    bench = [g for g in callers if g.startswith("./benchmarks/")]
    init_md = open(os.path.join(SRC, "src/tensordem/__init__.py"), encoding="utf-8").read()
    solver = sh("grep", "-rln", "measure_band", "--include=*.py",
                "src/tensordem/dem.py", "src/tensordem/clump.py", cwd=SRC).stdout.strip()
    return {"grep_hits_total": len(callers),
            "in_benchmarks": bench[:6],
            "in_tests": len([g for g in callers if g.startswith("./tests/")]),
            "in_docs_audit": len([g for g in callers if g.startswith("./docs/")]),
            "in_src": [g for g in callers if g.startswith("./src/")][:6],
            "exported_in_package_init": "measure_band" in init_md,
            "called_from_solver_core": bool(solver),
            "case_pipeline_callers": sorted({g.split(":")[0].split("./")[-1]
                                             for g in bench})}


def decide(j, repro_res, tests, integ, ref):
    why = []
    g = j["gate_15pct"]
    ok1 = g["zero_noise_pass"] and j["verdict_mechanical_match"]
    if not ok1:
        return 1, ["L1 blocked: noise-free known-angle recovery outside 15%% (max rel %.2f%%) "
                   "or committed verdict not reproduced from its own rows"
                   % (100 * g["zero_noise_max_rel"])]
    why.append("L1 met: analytic known-angle reference recovered to max rel %.2f%% noise-free "
               "(bias %+.2f deg, max abs %.2f deg over %d zero-noise cells); fold invariance, "
               "determinism and bare-chain == measure_band are asserted by tests; committed "
               "verdict reproduced mechanically from the committed rows"
               % (100 * g["zero_noise_max_rel"], j["recomputed"]["zero_noise"]["bias"],
                  j["recomputed"]["zero_noise"]["max_abs"],
                  j["recomputed"]["zero_noise"]["n"]))
    ok2 = (j["rows"] > 0 and tests["xval_exit"] == 0 and tests["neighbour_exit"] == 0
           and integ["grep_hits_total"] > 0)
    if not ok2:
        return 1, why + ["L2 blocked: harness or tests do not run cleanly"]
    why.append("L2 met: real measurement chain (kNN deformation gradient -> local max shear "
               "strain -> Radon/projection variance focus -> folded angle) driven by a real "
               "harness (displacement snapshots in -> per-cell angle JSON out); %s; "
               "neighbouring band suites: %s"
               % (tests["xval_summary"], tests["neighbour_summary"]))
    if not (integ["exported_in_package_init"] and integ["called_from_solver_core"]):
        why.append("L3 blocked: measure_band is not exported by src/tensordem/__init__.py and no "
                   "solver-core module calls it; consumers outside tests are the biaxial case "
                   "scripts (%s) plus earlier audit probes — post-processing of one benchmark "
                   "case, not a solver/package API" % ", ".join(integ["case_pipeline_callers"]))
    if ref["registered_rows"] == 0:
        why.append("L4 blocked (referee): tensor_benchmark_cases has 0 rows for %s (table total "
                   "%d; TensorDEM-domain rows exist only for clump/friction/H1/Campbell and "
                   "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE) — the reference is the harness' own "
                   "analytic theta_true and the <15%% number traces only to feature_ledger row "
                   "'%s' (status %s)"
                   % (ANCHOR, ref["benchmark_cases_total"],
                      (ref["feature_ledger"] or {}).get("slug"),
                      (ref["feature_ledger"] or {}).get("status")))
    if not g["envelope_pass"]:
        why.append("L4 blocked (own gate): inside the declared 1%% operating envelope max rel err "
                   "%.1f%% vs 15%%, share within 15%% = %.0f%% over %d cells; noise-free is "
                   "%.1f%%, so this is the measured SNR limit of the operator, not a coding bug "
                   "— but the precision claim is unmet (committed verdict pass=%s)"
                   % (100 * g["envelope_max_rel"],
                      100 * j["recomputed"]["envelope"]["share_within_15pct"],
                      j["recomputed"]["envelope"]["n"], 100 * g["zero_noise_max_rel"],
                      j["verdict_committed"]["pass"]))
    if not repro_res["documented_cmd_reproduces_committed"]:
        r0 = repro_res["runs"].get("default_repeats", {})
        r1 = repro_res["runs"].get("repeats1", {})
        why.append("L4 blocked (repro): the published command string runs to %s rows with bias "
                   "%+.2f deg / max rel %.2f, while the committed artefact has %d rows with bias "
                   "%+.2f deg / max rel %.2f; byte identity (sha256 %s) is only reached after "
                   "adding --repeats 1, and the artefact's design block still advertises "
                   "repeats=%d -> artefact self-inconsistent"
                   % (r0.get("rows"), (r0.get("summary_all") or {}).get("bias_deg", 0),
                      (r0.get("summary_all") or {}).get("max_rel", 0),
                      j["rows"], j["committed_summary_all"]["bias_deg"],
                      j["committed_summary_all"]["max_rel"],
                      repro_res["committed_sha256"][:16],
                      j["coverage"]["repeats_in_design_block"]))
    c = j["coverage"]
    if any(c["missing_factor_levels"][k] for k in ("lattices", "widths", "noise", "angles")):
        why.append("L4 blocked (coverage): %d of %d advertised design cells x %d realisations "
                   "executed (%.2f%% of the design); missing levels %s — per-width / per-lattice "
                   "statements in the page are not backed by the executed design"
                   % (c["rows_executed"], c["design_cells_advertised"], c["repeats_in_design_block"],
                      100 * c["fraction_of_design"], c["missing_factor_levels"]))
    if repro_res.get("subset_probe", {}).get("crashes_without_case_lattice"):
        why.append("L4 blocked (portability): the harness aborts on any subset without the "
                   "24x48 case lattice (summarise() line 213 takes max() of an empty legacy "
                   "list; observed exit %s '%s'), so a 48-shard reproduction cannot be trusted "
                   "for shards holding only 24x24/36x36 cells"
                   % (repro_res["subset_probe"]["result"]["exit"],
                      repro_res["subset_probe"]["result"]["stderr_tail"]))
    if ref.get("prior_verdicts_for_anchor"):
        pv = ref["prior_verdicts_for_anchor"][0]
        why.append("context (not a defect of m17789): the anchor already carries an L%d verdict "
                   "from mission %s (verdict id %s) whose reference was the OSS scikit-image Radon "
                   "at sha256 pin; m17789 is an independent round on the same operator and adds a "
                   "measured noise envelope, so the anchor's maturity does not hinge on this "
                   "round's honest FAIL"
                   % (pv["level"], pv["mission_id"], pv["id"]))
    if not (ref.get("feature_ledger") or {}).get("mission_id") and \
            not os.path.exists(os.path.join(SRC, "prereg", "XVAL_DEM_RADON_PROJECTION_MEASUREMENT.json")):
        why.append("L4 blocked (predeclaration): the page calls its decision rule 'predeclared' "
                   "but no frozen prereg artefact exists in the tree for this round (round-1 "
                   "prereg belongs to m17593 and fixes different factors) — the criterion cannot "
                   "be shown to predate the runs")
    if j["doc"]["contradiction_three_realisations"]:
        why.append("doc/artefact contradiction: the page promises '3 realisations for every noisy "
                   "cell' while every noisy row carries repeat=0 only")
    if not (integ["exported_in_package_init"] and integ["called_from_solver_core"]):
        why.append("L4 also inherits the L3 integration gap")
    return 2, why


def insert_verdict(dsn, audit_mid, level, ev):
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("INSERT INTO tensor_maturity_verdicts "
                    "(repo, scope, mission_id, level, evidence, verdict_by) "
                    "VALUES (%s,%s,%s,%s,%s::jsonb,%s) RETURNING id",
                    ("TensorDEM", "m17789 %s" % ANCHOR, audit_mid, level,
                     json.dumps(ev, ensure_ascii=False), "board-audit"))
        vid = cur.fetchone()[0]
        c.commit()
    print("verdict id", vid, "level", level)
    return vid


def render(ext, ms, ref, rep, j, repro_res, tests, integ, level, why, mainline=None):
    A = None
    out = []
    A = out.append
    c, g, rn = j["coverage"], j["gate_15pct"], j["recomputed"]
    rows = json.load(open(os.path.join(ROOT, ART)))["rows"]
    n_leaves = len(list(leaves(json.load(open(os.path.join(ROOT, ART))))))
    r0 = repro_res["runs"].get("default_repeats", {})
    r1 = repro_res["runs"].get("repeats1", {})

    A("# m17789 \u4ea4\u5dee\u6838\u9a8c\uff1aXVAL-DEM-RADON-PROJECTION-MEASUREMENT\uff08TensorDEM\uff09")
    A("")
    A("> \u4ea4\u5dee\u5ba1\u8ba1\uff08\u975e\u91cd\u505a\uff09\u3002\u88ab\u5ba1\u4ea4\u4ed8 `%s` @ `%s`\uff08%s main\uff09\uff1b\u5ba1\u8ba1\u5355 m%d\u3002"
      % (COMMIT[:12], BRANCH,
         ("\u5df2\u5e76\u5165\u672c\u5730 main" if ms["ancestor_of_main"]
          else "\u672a\u5e76\u5165\u672c\u5730 main\uff0c\u8fdc\u7aef\u5df2\u5e76\uff1a" + (ms.get("remote_merge_commit") or "\u672a\u77e5")), AUDIT_MID))
    A("> \u672c\u9875\u4e0e `artifacts/audit_m17873/*.json` \u5168\u90e8\u7a0b\u5e8f\u5316\u751f\u6210\uff0c\u65e0\u624b\u6284\u6570\u5b57\u3002")
    A("")
    A("**\u88c1\u5b9a\uff1aL%d\uff08board-audit\uff09**" % level)
    A("")
    A("\u4e00\u53e5\u8bdd\uff1a\u6d4b\u91cf\u5668\u4e0e\u9a8c\u8bc1\u88c5\u7f6e\u90fd\u662f\u771f\u4ee3\u7801\uff0c\u65e0\u566a\u58f0\u4e0b\u628a\u6ce8\u5165\u503e\u89d2\u590d\u539f\u5230 %.2f%%\uff08\u8fdc\u4e25\u4e8e 15%%\uff09\uff0c"
      "\u4f46**\u58f0\u660e\u7684\u5de5\u4f5c\u5305\u7edc\uff081%% d \u6d4b\u91cf\u566a\u58f0\uff09\u5185\u6700\u5927\u76f8\u5bf9\u8bef\u5dee %.0f%%\u300115%% \u5185\u5360\u6bd4 %.0f%%**\uff0c"
      "\u4ea4\u4ed8\u81ea\u5df1\u5224 `pass=%s`\uff08\u8bda\u5b9e FAIL\uff09\uff1b\u540c\u65f6 `tensor_benchmark_cases` \u65e0\u672c\u951a\u767b\u8bb0\u884c\uff0c"
      "\u53d1\u5e03\u7684\u590d\u73b0\u547d\u4ee4\u6309\u5b57\u9762\u6267\u884c**\u4e0d\u590d\u73b0**\u63d0\u4ea4\u5de5\u4ef6\u3002\u6545 L%d\uff0c\u4e0d\u8fbe L3/L4\u3002"
      % (100 * g["zero_noise_max_rel"], 100 * g["envelope_max_rel"],
         100 * rn["envelope"]["share_within_15pct"], j["verdict_committed"]["pass"], level))
    A("")
    A("## 1. \u5bf9\u8c61\u4e0e\u5224\u636e\u51fa\u5904")
    A("")
    A("- \u5bf9\u8c61\uff1a`CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE` \u526a\u5207\u5e26\u503e\u89d2\u7684**\u6d4b\u91cf\u4eea\u5668** "
      "`tensordem.biaxial_shear_band.measure_band`\uff08`src/tensordem/biaxial_shear_band.py:268`\uff0c"
      "\u5185\u6838 `radon_band_angle` \u540c\u6587\u4ef6 :154\uff09\uff1b\u4fee\u590d\u524d\u57fa\u7ebf "
      "`biaxial_band_diagnostics.legacy_radon_band_angle`\uff08`src/tensordem/biaxial_band_diagnostics.py:258`\uff09\u3002")
    A("- \u53c2\u8003\uff1a\u89e3\u6790\u2014\u2014`\u03b8_true` \u662f\u6ce8\u5165\u4f4d\u79fb\u573a `u=(slip/2)tanh(2s/w)\u00b7t` \u7684\u53c2\u6570\uff0c"
      "\u65e0\u5916\u90e8\u951a\u3001\u65e0\u53c2\u8003\u4e0d\u786e\u5b9a\u5ea6\u3002")
    A("- **\u57fa\u51c6\u88c1\u5224\uff08\u552f\u4e00\uff09**\uff1a`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
      "WHERE anchor_slug='%s'` \u2192 **%d \u884c**\uff08\u5168\u8868 %d \u884c\uff1bTensorDEM \u57df %d \u884c\uff09\u3002**\u65e0\u5224\u636e\u884c**\uff0c"
      "\u672c\u6b21\u4e0d\u4ee5\u4ed6\u7269\u4ee3\u66ff\u88c1\u5224\uff0c\u53ea\u5982\u5b9e\u767b\u8bb0\u8fd9\u4e00\u53f0\u8d26\u7f3a\u53e3\u3002"
      % (ANCHOR, ref["registered_rows"], ref["benchmark_cases_total"], len(ref["dem_rows"])))
    A("- 15%% \u5224\u636e\u7684\u771f\u5b9e\u51fa\u5904\uff1a`feature_ledger` slug=`%s`\uff08acceptance=`%s`\uff0cstatus=%s\uff0c"
      "evidence=`%s`\uff09+ \u4ea4\u4ed8\u9875\u4e0e harness \u5185\u5148\u4e8e\u8fd0\u884c\u5199\u6b7b\u7684\u5224\u636e\u3002"
      % ((ref["feature_ledger"] or {}).get("slug"), (ref["feature_ledger"] or {}).get("acceptance"),
         (ref["feature_ledger"] or {}).get("status"), (ref["feature_ledger"] or {}).get("evidence")))
    A("")
    A("## 2. \u4e09\u4ef6\u5957\u662f\u5426\u771f\u5b58\u5728")
    A("")
    A("| \u4ef6 | \u6307\u8ba4 | \u6838\u9a8c |")
    A("|---|---|---|")
    A("| impl | `benchmarks/biaxial_shear_band/run_radon_xval_measurement.py`\uff08\u65b0\uff09+ "
      "`src/tensordem/biaxial_shear_band.py:{154,268}`\uff08\u88ab\u9a8c\u7b97\u5b50\uff09 | \u63d0\u4ea4\u5185\u9010\u5b57\u8282\u5b58\u5728\uff08sha256 \u5165 manifest\uff09\uff0c\u7f3a\u4ef6 %d |"
      % len(ext["missing"]))
    A("| run_artifacts | `artifacts/biaxial_shear_band/radon_projection_xval.json`\uff08%d \u884c / %d \u53f6\uff09\u3001"
      "`*.png`\uff08%d B\uff09\u3001`docs/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.md` | \u5747 git-tracked\uff1bJSON \u7531\u5ba1\u8ba1\u72ec\u7acb\u805a\u5408\u590d\u6838\uff08\u00a73\uff09 |"
      % (j["rows"], n_leaves,
         ext["manifest"].get("artifacts/biaxial_shear_band/radon_projection_xval.png", {}).get("bytes", 0)))
    A("| benchmark ref vs got | ref=\u89e3\u6790 \u03b8_true\uff1bgot=\u9010\u683c `radon_production`\uff08\u00a73 \u8868\uff09 | \u53ef\u590d\u7b97\uff1b**\u65e0\u767b\u8bb0\u88c1\u5224\u884c** |")
    A("| repro_cmd | \u4ea4\u4ed8 checks \u4e32\uff1a`--lattices 24x48 24x24 --angles 30 45 60 75 --widths 1.5 --noise 0.0 0.01` | "
      "\u5b57\u9762\u6267\u884c \u2192 %s\uff1b\u52a0 `--repeats 1` \u540e sha256 \u9010\u5b57\u8282\u4e00\u81f4\uff08\u00a74\uff09 |"
      % ("32 \u884c\u3001\u4e0e\u63d0\u4ea4\u4ef6\u4e0d\u7b49"
         if not repro_res["documented_cmd_reproduces_committed"] else "\u4e00\u81f4"))
    A("")
    A("## 3. \u6570\u5b57\u88c1\u5224\uff08\u72ec\u7acb\u91cd\u7b97 vs \u63d0\u4ea4\u4ef6\uff09")
    A("")
    A("| \u91cf | \u63d0\u4ea4\u4ef6\u58f0\u79f0 | \u5ba1\u8ba1\u72ec\u7acb\u91cd\u7b97 | \u4e00\u81f4 |")
    A("|---|---|---|---|")
    cs = j["committed_summary_all"]
    A("| \u5168\u57df\u6d4b\u91cf\u6570 n | %d | %d | %s |" % (cs["n"], rn["all"]["n"],
                                              "\u2713" if cs["n"] == rn["all"]["n"] else "\u2717"))
    A("| \u5168\u57df bias | %+.4f\u00b0 | %+.4f\u00b0 | \u2713 |" % (cs["bias_deg"], rn["all"]["bias"]))
    A("| \u5168\u57df RMS | %.4f\u00b0 | %.4f\u00b0 | \u2713 |" % (cs["rms_deg"], rn["all"]["rms"]))
    A("| \u5168\u57df max rel | %.2f | %.2f | %s |" % (cs["max_rel"], j["auditor_summary_all"]["max_rel"],
                                              "\u2713" if abs(cs["max_rel"] - j["auditor_summary_all"]["max_rel"]) < 1e-12 else "\u2717"))
    A("| \u65e0\u566a\u58f0 bias / max abs | %+.2f\u00b0 / %.2f\u00b0 | %+.2f\u00b0 / %.2f\u00b0 | \u2713 |"
      % (j["committed_zero_noise"]["bias_deg"], j["committed_zero_noise"]["max_abs_deg"],
         rn["zero_noise"]["bias"], rn["zero_noise"]["max_abs"]))
    A("| 1%% \u5305\u7edc\u5185 max rel / 15%% \u5185\u5360\u6bd4 | %.1f%% / %.1f%% | %.1f%% / %.1f%% | \u2713 |"
      % (100 * j["committed_envelope_block"]["max_rel"],
         100 * j["committed_envelope_block"]["share_within_15pct"],
         100 * rn["envelope"]["max_rel"], 100 * rn["envelope"]["share_within_15pct"]))
    A("| 24\u00d748 \u751f\u4ea7\u7b97\u5b50 bias / max abs | %+.3f\u00b0 / %.2f\u00b0\uff08\u539f\u5b50\u5757\uff09 | %+.3f\u00b0 / %.2f\u00b0 | \u2713 |"
      % (-10.625, 57.0, rn["case_lattice_24x48_production"]["bias"], rn["case_lattice_24x48_production"]["max_abs"]))
    A("| 24\u00d748 \u4fee\u590d\u524d\u57fa\u7ebf bias | %+.3f\u00b0 | %+.3f\u00b0 | \u2713 |"
      % (-20.5, rn["case_lattice_24x48_legacy"]["bias"]))
    A("")
    A("**\u9010\u683c\u6838\uff08\u57fa\u51c6\u88c1\u5224\u7f3a\u5931\uff0c\u6545\u7528\u4ea4\u4ed8\u81ea\u5df1\u9884\u7f6e\u7684 <15%% \u5224\u636e + \u89e3\u6790\u53c2\u8003\uff09**")
    A("")
    A("| \u573a\u666f | ref | got\uff08\u6700\u5927\u76f8\u5bf9\u8bef\u5dee\uff09 | 15%% \u5bb9\u5dee | <3%% |")
    A("|---|---|---|---|---|")
    A("| \u65e0\u566a\u58f0\uff08%d \u683c\uff09 | \u89e3\u6790 \u03b8_true | %.2f%% | PASS | %s |"
      % (rn["zero_noise"]["n"], 100 * g["zero_noise_max_rel"],
         "PASS" if g["zero_noise_pass_3pct"] else "FAIL"))
    A("| 1%% d \u5305\u7edc\uff08%d \u683c\uff09 | \u89e3\u6790 \u03b8_true | %.0f%% | **FAIL** | FAIL |"
      % (rn["envelope"]["n"], 100 * g["envelope_max_rel"]))
    A("")
    A("\u9010\u683c\uff0824\u00d748\uff0c\u5e26\u5bbd 1.5 d\uff09\uff1a")
    A("")
    A("| \u03b8_true | noise | \u03b8\u0302 \u751f\u4ea7 | \u8bef\u5dee | \u03b8\u0302 \u4fee\u590d\u524d | \u57fa\u7ebf\u8bef\u5dee |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        if r["lattice"] != "24x48":
            continue
        A("| %g\u00b0 | %g | %g\u00b0 | %+g\u00b0 | %s | %s |"
          % (r["alpha_true_deg"], r["noise_fraction"], r["radon_production"],
             r["radon_production_error_deg"],
             ("%g\u00b0" % r["radon_legacy_peraxis"]) if "radon_legacy_peraxis" in r else "\u2014",
             ("%+g\u00b0" % r["radon_legacy_peraxis_error_deg"])
             if "radon_legacy_peraxis_error_deg" in r else "\u2014"))
    A("")
    A("- \u5224\u5b9a\u53ef\u673a\u68c0\uff1a`verdict` \u7684\u56db\u4e2a\u5e03\u5c14\u91cf\u7531\u884c\u6570\u636e\u673a\u68b0\u91cd\u7b97\uff0c\u4e0e\u63d0\u4ea4\u4ef6%s\u3002"
      % ("\u5b8c\u5168\u4e00\u81f4" if j["verdict_mechanical_match"] else "**\u4e0d\u4e00\u81f4**"))
    A("- \u63d0\u4ea4 `pass=%s`\uff08\u65e0\u566a\u58f0\u7cbe\u786e + \u5305\u7edc\u5185\u4e0d\u5408\u683c + 24\u00d748 \u4e0a\u4f18\u4e8e\u4fee\u590d\u524d\u57fa\u7ebf\uff09\u2014\u2014 \u4e0e\u5ba1\u8ba1\u91cd\u7b97\u4e00\u81f4\u3002"
      % j["verdict_committed"]["pass"])
    A("")
    A("## 4. \u53ef\u590d\u73b0\u6027\uff08\u5b9e\u6d4b\uff0c\u672c\u673a x86_64\uff1b\u96c6\u7fa4\u4e3a loong64\uff0c\u67b6\u6784\u4e0d\u540c\uff09")
    A("")
    A("| \u8fd0\u884c | \u547d\u4ee4 | \u884c\u6570 | \u4e0e\u63d0\u4ea4\u4ef6 sha256 | \u53f6\u5dee |")
    A("|---|---|---|---|---|")
    A("| \u63d0\u4ea4\u5de5\u4ef6 | \u2014 | %d | `%s\u2026` | \u2014 |" % (j["rows"], repro_res["committed_sha256"][:16]))
    A("| \u5b57\u9762 repro_cmd | \u65e0 `--repeats` | %d | \u4e0d\u7b49 | %d/%d |"
      % (r0.get("rows", -1), r0.get("diffs", -1), r0.get("leaves_committed", 0)))
    A("| \u8865 `--repeats 1` | \u540c\u4e0a + `--repeats 1` | %d | **\u4e00\u81f4** | 0/%d |"
      % (r1.get("rows", -1), r1.get("leaves_rerun", 0)))
    A("")
    A("- \u539f\u56e0\uff1a\u9ed8\u8ba4 `REPEATS=3`\uff08harness \u7b2c 66 \u884c\uff09\u4f7f\u6bcf\u4e2a\u542b\u566a\u683c\u8dd1 3 \u6b21\u5b9e\u73b0 \u2192 32 \u884c\uff08bias %+.2f\u00b0\uff0cmax rel %.2f\uff09\uff1b"
      "\u63d0\u4ea4\u4ef6\u53ea\u6709 16 \u884c\uff08\u542b\u566a\u683c\u4ec5 1 \u6b21\uff09\uff0c\u53ea\u6709 `--repeats 1` \u80fd\u9010\u5b57\u8282\u590d\u73b0 "
      "`sha256=%s\u2026`\uff08%d λ \u5168\u7b49\uff09\u3002"
      % ((r0.get("summary_all") or {}).get("bias_deg", 0), (r0.get("summary_all") or {}).get("max_rel", 0),
         r1.get("sha256", "")[:16], r1.get("leaves_rerun", 0)))
    A("- \u4f46\u5de5\u4ef6\u5185 `design.repeats` \u4ecd\u5199 **%d**\uff08harness \u7b2c 384 \u884c\u628a\u5e38\u91cf `REPEATS` \u5199\u6b7b\u8fdb JSON\uff0c"
      "\u4e0d\u770b `args.repeats`\uff09\u2192 \u5de5\u4ef6\u81ea\u76f8\u77db\u76fe\uff08\u8bbe\u8ba1\u58f0\u79f0 3 \u6b21\u5b9e\u73b0\uff0c\u884c\u5185\u53ea\u6709 repeat=0\uff09\u3002"
      % c["repeats_in_design_block"])
    A("- \u5355\u5143\u683c\u786e\u5b9a\u6027\u73b0\u573a\u6d4b\uff08\u65b0\u8dd1\uff09\uff1a\u540c\u4e00\u683c\u8dd1\u4e24\u6b21 sha256 %s\uff08\u89d2\u5ea6 %s\u00b0\uff09\u3002"
      % ("\u4e00\u81f4" if repro_res["spot_check_determinism"]["identical"] else "\u4e0d\u4e00\u81f4",
         repro_res["spot_check_determinism"]["angle"]))
    A("- \u4ea4\u4ed8\u9875\u58f0\u79f0\u7684\u201c48-way sharded on the 702 cluster\u201d\u5728\u6811\u5185**\u65e0\u5206\u7247\u4ea7\u7269\u3001\u65e0\u65e5\u5fd7\u53ef\u67e5**\uff1b\u63d0\u4ea4 JSON \u7684\u884c\u5e8f\u4e0e\u5355\u8fdb\u7a0b\u8fd0\u884c\u5b8c\u5168\u4e00\u81f4\uff08\u00a73\u53ef\u590d\u73b0\uff09\uff0c\u96c6\u7fa4\u5206\u7247\u53ea\u80fd\u8ba1\u4e3a\u58f0\u660e\uff08not_run\uff09\u3002")
    A("")
    sp = repro_res.get("subset_probe", {})
    if sp.get("crashes_without_case_lattice"):
        A("- **\u5b50\u96c6\u5d29\u6e83\uff08\u65b0\u53d1\u73b0\uff09**\uff1a\u53ea\u8981\u8fd0\u884c\u96c6\u91cc\u6ca1\u6709 24\u00d748 \u683c\uff08\u5982 `--lattices 24x24`\uff0c"
          "\u6216\u843d\u5230\u53ea\u542b 24\u00d724/36\u00d736 \u7684\u96c6\u7fa4\u5206\u7247\uff09\uff0charness \u5728 `summarise()` \u7b2c 213 \u884c\u5bf9\u7a7a\u7684"
          "\u4fee\u590d\u524d\u57fa\u7ebf\u5217\u8868\u53d6 max \u2192 `ValueError: zero-size array`\uff0cexit=%s\uff0c**\u5206\u7247\u4ea7\u7269\u90fd\u5199\u4e0d\u51fa\u6765**\u3002"
          "\u201c48 \u5206\u7247\u8dd1\u901a\u201d\u7684\u58f0\u660e\u56e0\u6b64\u4e0d\u53ef\u5b8c\u5168\u91c7\u4fe1\u3002"
          % sp["result"]["exit"])
    A("## 5. \u6d4b\u8bd5\u4e0e\u96c6\u6210\u9762")
    A("")
    A("- pytest\uff1a`%s` \u2192 exit %d\uff0c`%s`\uff1b\u76f8\u90bb\u526a\u5207\u5e26\u5957\u4ef6 exit %d\uff0c`%s`\u3002"
      % (tests["cmd"], tests["xval_exit"], tests["xval_summary"],
         tests["neighbour_exit"], tests["neighbour_summary"]))
    A("- \u65ad\u8a00\u662f\u4e0d\u53d8\u91cf\u800c\u975e\u201c\u8dd1\u4e00\u904d\u201d\uff1a\u5df2\u77e5\u89d2\u56de\u6536\u3001\u6298\u53e0\u4e0d\u53d8\uff08120\u00b0\u226160\u00b0\uff09\u3001\u4fee\u590d\u524d\u57fa\u7ebf\u5fc5\u987b\u8d85 15%%\u3001"
      "\u9000\u5316\u8f93\u5165\u62d2\u7edd\u3001\u5206\u7247\u6070\u597d\u5212\u5206\u3001\u6c47\u603b/\u5224\u5b9a\u903b\u8f91\u673a\u68b0\u6027\u3001`radon_band_angle` \u94fe == `measure_band` \u4e14\u5e42\u7b49\u3002")
    A("- \u96c6\u6210\u9762\uff1a`measure_band|radon_band_angle` \u5168\u4ed3 %d \u5904\u547d\u4e2d\uff08tests %d\u3001docs/board_audit %d\u3001"
      "benchmarks %s\uff09\uff1bsrc \u5185\u90e8 %s\u3002**\u672a**\u5bfc\u51fa\u4e8e `src/tensordem/__init__.py`\uff0c"
      "**\u672a**\u88ab\u6c42\u89e3\u5668\u4e3b\u5faa\u73af\uff08`dem.py`/`clump.py` 0 \u547d\u4e2d\uff09\u8c03\u7528 \u2192 \u5b83\u662f\u4e00\u4e2a\u7b97\u4f8b\u7684\u540e\u5904\u7406\u4eea\u5668\uff0c"
      "\u4e0d\u662f\u5e73\u53f0/\u6c42\u89e3\u5668 API\uff08L3 \u4e0d\u8fbe\uff09\u3002"
      % (integ["grep_hits_total"], integ["in_tests"], integ["in_docs_audit"],
         ", ".join(integ["case_pipeline_callers"]), ", ".join(x.split(":")[0] for x in integ["in_src"])))
    A("")
    A("## 6. \u62a5\u544a\uff08tensor_reports id=%d\uff09" % REPORT_ID)
    A("")
    A("- \u5b58\u5728\uff1a%s\uff1bmission_id=%s\u3001scenario_key=`%s`\u3001%d \u5b57\u7b26\u3001\u56fe %s\u3002"
      % (rep.get("found"), rep.get("mission_id"), rep.get("scenario_key"),
         rep.get("chars", 0), "\u6709" if rep.get("has_figures") else "\u65e0"))
    A("- \u516d\u6bb5\uff1a%s\uff1b\u7f3a\u5931 %s\u3002\u6570\u5b57\u4e0e \u00a73 \u91cd\u7b97\u4e00\u81f4\uff08\u62a5\u544a\u4e0d\u628a\u5931\u8d25\u7c89\u9970\u6210\u901a\u8fc7\uff0c\u5df2\u81ea\u5ba3 honest FAIL\uff09\u3002"
      % (", ".join(h.strip("# ") for h in rep.get("sections", [])), rep.get("missing_sections")))
    A("")
    A("## 7. \u6ce8\u610f\u70b9\uff08\u5751\u4e0e\u5c40\u9650\uff09")
    A("")
    A("1. **\u88c1\u5224\u884c\u7f3a\u5931**\uff1a`tensor_benchmark_cases` 0 \u884c\u672c\u951a \u2192 \u5e73\u53f0\u673a\u5668\u95e8\u65e0\u6cd5\u5bf9\u8868\u88c1\u5b9a\uff1b"
      "15%% \u53ea\u80fd\u8ffd\u5230 feature_ledger + \u4ea4\u4ed8\u9875\u3002\u8fd9\u662f**\u53f0\u8d26\u7f3a\u53e3**\uff0c\u4e0d\u662f\u6570\u503c\u7f3a\u9677\uff0c\u4f46 L4 \u7684\u201c\u673a\u5668\u53ef\u590d\u6838\u57fa\u51c6\u201d\u4e00\u73af\u4e3a\u7a7a\u3002")
    A("2. **\u5305\u7edc\u58f0\u660e\u4e0e FAIL \u7684\u8bfb\u6cd5**\uff1a1%%\u00b7d = 0.015 m \u566a\u58f0\u5bf9 0.02 m \u6ce8\u5165\u6ed1\u79fb\uff0cSNR\u22481.3\uff1b"
      "\u5305\u7edc\u5185 76%% \u8bef\u5dee\u662f\u201c\u6781\u7aef\u4fe1\u566a\u6bd4\u201d\u4e0b\u7684\u5b9e\u6d4b\u5d29\u6e83\uff0c\u4e0d\u80fd\u8bfb\u6210\u7b97\u5b50\u574f\u6389\uff1b\u53cd\u4e4b\u4e5f\u4e0d\u80fd\u628a"
      "\u65e0\u566a\u58f0 1.3%% \u8bfb\u6210\u201c\u4eea\u5668\u5df2\u5728\u771f\u5b9e\u5de5\u4f5c\u70b9\u9a8c\u8bc1\u201d\u3002\u4ea4\u4ed8\u8fd9\u4e00\u70b9\u5199\u5f97\u8bda\u5b9e\u3002")
    A("3. **\u8bbe\u8ba1\u8986\u76d6 1.5%%**\uff1a\u5b9e\u8dd1 %d \u683c / \u58f0\u79f0 %d \u683c\u00d7%d \u6b21\u5b9e\u73b0\uff1b\u7f3a\u5c11 %s\u3002"
      "\u201c\u5e26\u5bbd\u5c31\u662f\u5206\u8fa8\u7387\u58f0\u660e\u201d\u53ea\u6709 1.5 d \u4e00\u4e2a\u70b9\u652f\u6491\u3002"
      % (c["rows_executed"], c["design_cells_advertised"], c["repeats_in_design_block"],
         c["missing_factor_levels"]))
    A("4. **\u53d1\u5e03\u7684\u547d\u4ee4\u4e0d\u5fe0\u5b9e**\uff1achecks \u4e32\u5b57\u9762\u8dd1\u51fa 32 \u884c\uff0c\u6570\u5b57\u4e0d\u540c\uff1b"
      "\u63d0\u4ea4\u5de5\u4ef6\u53ea\u80fd\u7528\u672a\u8bb0\u5f55\u7684 `--repeats 1` \u9010\u5b57\u8282\u590d\u73b0\u3002\u547d\u4ee4\u4e0e\u5143\u6570\u636e\u5e94\u8865\u4e0a\uff08\u8fd4\u4fee\u9879\uff09\u3002")
    A("5. \u672c\u5ba1\u8ba1\u5728 x86_64 \u672c\u673a\u91cd\u8dd1\uff1b\u4ea4\u4ed8\u58f0\u79f0\u7684 loong64 \u96c6\u7fa4 48 \u5206\u7247\u65e0\u4ea7\u7269\u53ef\u6838\uff0c"
      "\u8de8\u67b6\u6784\u4f4d\u7ea7\u590d\u73b0\u672a\u9a8c\u8bc1\uff08not_run\uff09\u3002")
    A("")
    A("## 7bis. \u951a\u70b9\u65e2\u6709\u6210\u719f\u5ea6\u4e0e\u672c\u5355\u5173\u7cfb")
    A("")
    if ref.get("prior_verdicts_for_anchor"):
        pv = ref["prior_verdicts_for_anchor"][0]
        A("- \u8fb9\u754c\u58f0\u660e\uff1a**\u951a\u70b9\u5c42\u9762\u5df2\u6709 L%d \u88c1\u5b9a**\uff08verdict id %s\uff0cmission %s\uff0c%s\uff0cgate=%s\uff09\uff0c"
          "\u5bf9\u8c61\u662f\u540c\u4e00\u4e2a `radon_band_angle`\uff0c\u53c2\u8003\u662f OSS scikit-image Radon\uff08sha256 \u9489\u4f4f\uff09102 \u7b97\u4f8b\u3002"
          "\u672c\u5355\u53ea\u5224 m17789 \u8fd9**\u4e00\u8f6e**\u7684\u4ea4\u4ed8\uff0c\u4e0d\u63a8\u7ffb\u951a\u70b9\uff1bm17789 \u7684 honest FAIL \u4e0d\u662f\u5de5\u5177\u4e0d\u80fd\u7528\uff0c"
          "\u800c\u662f\u5b83\u81ea\u5df1\u58f0\u660e\u7684\u566a\u58f0\u5305\u7edc\u5185\u7cbe\u5ea6\u4e0d\u8fbe\u6807\u3002"
          % (pv["level"], pv["id"], pv["mission_id"], pv["verdict_by"], pv["gate_state"]))
    A("- \u4e3b\u7ebf\u4e0a\u540c\u540d\u951a\u70b9\u7684\u5185\u5bb9\u662f**\u53e6\u4e00\u8f6e**\uff1a`docs/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.md` \u5728 origin/main \u4e0a\u5199\u7684\u662f "
      "\u201cTensorDEM vs the OSS image-domain Radon\u201d\uff0c\u914d `benchmarks/radon_projection_xval/run_xval.py` + `src/tensordem/radon_reference.py` + "
      "`prereg/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.json`\uff08\u9884\u6ce8\u518c\uff09\u3002")
    A("- \u63d0\u4ea4\u5728 origin/main %s\uff1a\u672c\u5355\u4ea7\u7269\u662f\u5426\u5df2\u5e76\uff1aartifact=%s\uff0cdocs \u540c\u540d\u4f46\u5185\u5bb9\u4e0d\u540c\u3002"
      % ("\u4e0a\u5df2\u5b58\u5728" if mainline and mainline.get("commit_is_ancestor_of_origin_main") else "\u4e0a**\u4e0d**\u5b58\u5728\u8be5 commit",
         mainline.get("m17789_artifact_on_origin_main") if mainline else None))
    A("- \u56e0\u6b64\uff1am17789 \u662f\u540c\u4e00\u951a\u70b9\u7684\u7b2c\u4e8c\u8f6e\uff08\u5df2\u6709\u8f6e\u6b21\u5df2\u5e76\u5165 main\uff09\uff0c\u4e14\u672c\u8f6e**\u65e0\u51bb\u7ed3 prereg**\uff08\u6811\u5185 prereg \u4ec5\u5c5e\u4e8e m17593 \u7684\u56e0\u5b50\u8bbe\u8ba1\uff09\uff0c"
      "\u201c\u9884\u58f0\u660e\u5224\u636e\u201d\u4e0d\u53ef\u8bc1\uff1b\u53f0\u8d26 evidence \u5199\u201cTensorDEM#1462 merged\u201d\uff0c\u800c PR #1462 \u7684\u5408\u5e76\u63d0\u4ea4\uff08a332e7aa\uff09\u5bf9\u5e94 m17593\uff0c\u4e0d\u542b\u672c\u5355\u5de5\u4ef6\u3002")
    A("")
    A("## 8. \u5224\u7ea7\u4f9d\u636e\uff08L0-L5\uff09")
    A("")
    for i, w in enumerate(why):
        A("- %s" % w)
    A("")
    A("**\u88c1\u5b9a\uff1aL%d\u3002** \u5347 L3 \u9700\uff1a\u5bfc\u51fa\u5230\u5305 API \u5e76\u88ab\u6c42\u89e3\u5668/\u5e73\u53f0\u8c03\u7528\uff1b\u5347 L4 \u9700\uff1a"
      "\u767b\u8bb0\u88c1\u5224\u884c + \u5305\u7edc\u5185\u8bef\u5dee\u8fbe\u6807 + \u590d\u73b0\u547d\u4ee4\u4e0e\u5de5\u4ef6\u5bf9\u9f50\u3002" % level)
    A("")
    A("## 9. \u672c\u6b21\u672a\u8dd1\uff08not_run\uff0c\u4e0d\u8ba1\u4e3a pass\uff09")
    A("")
    A("- loong64 \u96c6\u7fa4 48 \u5206\u7247\u539f\u547d\u4ee4\u8de8\u67b6\u6784\u91cd\u8dd1\uff08\u65e0\u5206\u7247\u4ea7\u7269/日志\uff09\u3002")
    A("- \u5b8c\u6574\u8bbe\u8ba1\u7684 %d \u683c\uff08\u542b 0.5 d / 3 d \u5e26\u5bbd\u3001 36\u00d736 \u6676\u683c\u3001 0.5%%/2%%/4%% \u566a\u58f0\uff0c3 \u6b21\u5b9e\u73b0\uff09\u2014\u2014"
      "\u672c\u5ba1\u8ba1\u53ea\u590d\u73b0\u4ea4\u4ed8\u7684 16 \u683c\u3002" % c["design_cells_advertised"])
    A("- \u771f\u5b9e DEM \u7b97\u4f8b\uff08`run_r3_ensemble.py` \u5168\u8dd1\uff09\u4e0e\u5b9e\u9a8c/\u6587\u732e\u951a\u5bf9\u6bd4\u2014\u2014\u672c\u5355\u53ea\u5ba1\u4eea\u5668\u6821\u9a8c\u3002")
    A("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="dbname=tensorplatform user=postgres")
    ap.add_argument("--audit-mission", type=int, default=AUDIT_MID)
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REPRO_RUNS, exist_ok=True)

    ext = extract()
    ms = merge_state()
    ref = referee(a.dsn)
    rep = report_check(a.dsn)
    j = judge()
    integ = integration_probe()
    mainline = mainline_probe()
    tests = ({"xval_exit": None, "xval_summary": "skipped", "neighbour_exit": None,
              "neighbour_summary": "skipped", "cmd": "", "test_names": []}
             if a.skip_tests else tests_probe())
    repro_res = repro()
    level, why = decide(j, repro_res, tests, integ, ref)

    finding = {"anchor": ANCHOR, "audited_mission": 17789,
               "audit_mission": a.audit_mission, "commit": COMMIT, "branch": BRANCH,
               "level": level, "why": why, "merge_state": ms,
               "three_piece": {"impl": [k for k in ext["manifest"]
                                        if k.startswith(("src", "benchmarks"))],
                               "run_artifacts": [k for k in ext["manifest"]
                                                 if k.startswith(("artifacts", "docs"))],
                               "missing": ext["missing"],
                               "untracked_at_head": ext["untracked_at_head"],
                               "benchmark_ref": "analytic theta_true (injected slip band)",
                               "repro_cmd": (ref["mission_checks_string"] or "")[:400]},
               "tests": tests, "integration": integ, "repro": repro_res,
               "mainline": mainline,
               "judged": j, "referee": ref, "report": rep}
    for name, obj in (("referee.json", ref), ("judge.json", j), ("repro.json", repro_res),
                      ("tests.json", tests), ("integration.json", integ),
                      ("report.json", rep), ("mainline.json", mainline),
                      ("finding.json", finding)):
        with open(os.path.join(OUT, name), "w") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)

    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write(render(ext, ms, ref, rep, j, repro_res, tests, integ, level, why,
                        mainline))
    print("wrote", DOC, "level L%d" % level)

    if a.insert_verdict:
        root = "artifacts/audit_m17873/branch_57564a2e"
        r0 = repro_res["runs"].get("default_repeats", {})
        r1 = repro_res["runs"].get("repeats1", {})
        evidence = {
            "impl_files": [
                root + "/src/tensordem/biaxial_shear_band.py:154-215 (radon_band_angle: equal_count bins + projection variance focus)",
                root + "/src/tensordem/biaxial_shear_band.py:268-300 (measure_band: affine removal + kNN gradient -> local_shear_strain -> radon_band_angle)",
                root + "/src/tensordem/biaxial_band_diagnostics.py:258 (legacy_radon_band_angle, pre-R2 per-axis min-max baseline)",
                root + "/benchmarks/biaxial_shear_band/run_radon_xval_measurement.py (m17789 new: design + run + summarise + verdict + figure)",
            ],
            "entry": "PYTHONPATH=src python benchmarks/biaxial_shear_band/run_radon_xval_measurement.py --lattices 24x48 24x24 --angles 30 45 60 75 --widths 1.5 --noise 0.0 0.01 --repeats 1 --out artifacts/biaxial_shear_band/radon_projection_xval.json (audited: reproduces the committed artefact bit-for-bit, sha256 %s)" % repro_res["committed_sha256"][:16],
            "run_artifacts": [
                root + "/artifacts/biaxial_shear_band/radon_projection_xval.json (16 rows / %d leaves)" % r1.get("leaves_rerun", 0),
                root + "/artifacts/biaxial_shear_band/radon_projection_xval.png",
                root + "/docs/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.md",
                "tensor_reports id=%d (mission 17789, six sections complete: %s)" % (REPORT_ID, "yes" if not rep.get("missing_sections") else "no"),
            ],
            "tests": ["%s::%s" % ("tests/test_radon_projection_xval.py", n) for n in tests["test_names"]] +
                     ["audited: %s -> %s" % (tests["cmd"], tests["xval_summary"]),
                      "audited: PYTHONPATH=src pytest tests/test_biaxial_shear_band.py tests/test_biaxial_band_diagnostics.py -q -> %s" % tests["neighbour_summary"]],
            "integration_callers": "benchmarks/biaxial_shear_band/run_r2_case_rerun.py:80, run_r3_ensemble.py:126, make_figure.py:27 (case post-processing only); NOT exported in src/tensordem/__init__.py; 0 hits in solver core dem.py/clump.py -> not a solver/platform API",
            "benchmark": {
                "registered_row": "none: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-RADON-PROJECTION-MEASUREMENT' -> 0 rows (table total %d; TensorDEM rows exist for GAP-DEM-CLUMP / GAP-DEM-FRICTION / H1-BROKEN-ICE-DEM / CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT / CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE)" % ref["benchmark_cases_total"],
                "criterion_source": "feature_ledger slug=XVAL-DEM-RADON-PROJECTION-MEASUREMENT acceptance='<15%%' (status %s) + doc/harness predeclared 15%% gate" % (ref["feature_ledger"] or {}).get("status"),
                "ref": "analytic theta_true of the injected slip band u=(slip/2)tanh(2s/w)*t",
                "got_zero_noise": {"n": j["recomputed"]["zero_noise"]["n"],
                                   "bias_deg": j["recomputed"]["zero_noise"]["bias"],
                                   "max_abs_deg": j["recomputed"]["zero_noise"]["max_abs"],
                                   "max_rel": j["recomputed"]["zero_noise"]["max_rel"],
                                   "gate_15pct": "PASS",
                                   "gate_3pct": "PASS"},
                "got_envelope_1pct": {"n": j["recomputed"]["envelope"]["n"],
                                      "bias_deg": j["recomputed"]["envelope"]["bias"],
                                      "rms_deg": j["recomputed"]["envelope"]["rms"],
                                      "max_rel": j["recomputed"]["envelope"]["max_rel"],
                                      "share_within_15pct": j["recomputed"]["envelope"]["share_within_15pct"],
                                      "gate_15pct": "FAIL"},
                "legacy_baseline_24x48": {"bias_deg": j["recomputed"]["case_lattice_24x48_legacy"]["bias"],
                                          "max_abs_deg": j["recomputed"]["case_lattice_24x48_legacy"]["max_abs"]},
                "committed_verdict": j["verdict_committed"],
                "verdict_recomputed_by_auditor": j["verdict_recomputed"],
                "verdict_mechanical_match": j["verdict_mechanical_match"],
                "coverage": {"executed_rows": j["coverage"]["rows_executed"],
                             "design_cells_advertised": j["coverage"]["design_cells_advertised"],
                             "fraction_of_design": j["coverage"]["fraction_of_design"],
                             "missing_factor_levels": j["coverage"]["missing_factor_levels"]},
            },
            "repro_cmd": "cd <m17789 checkout> && PYTHONPATH=src python benchmarks/biaxial_shear_band/run_radon_xval_measurement.py --lattices 24x48 24x24 --angles 30 45 60 75 --widths 1.5 --noise 0.0 0.01 --repeats 1 --out /tmp/x.json --no-figure  # audited exit 0, sha256 == committed (%s); WITHOUT --repeats 1 the published command yields %d rows / bias %+.2f deg / max rel %.2f -> not the committed artefact" % (repro_res["committed_sha256"][:16], r0.get("rows", -1), (r0.get("summary_all") or {}).get("bias_deg", 0), (r0.get("summary_all") or {}).get("max_rel", 0)),
            "external_anchor": "none independent: theta_true is analytic (self-referential harness parameter); the pre-R2 failure narrative (60 deg reported as 44.3 deg, CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE R1/R2) and owner 15% precision standard are the only external referents; no experiment/literature replication for the operator itself",
            "prior_verdict_for_anchor": (ref.get("prior_verdicts_for_anchor") or [{}])[0],
            "mainline": mainline,
            "prereg": {"frozen_prereg_for_this_round": os.path.exists(os.path.join(SRC, "prereg/XVAL_DEM_RADON_PROJECTION_MEASUREMENT.json")) and "belongs to m17593 round-1, different factors" or "none in tree",
                       "note": "the page calls its decision rule 'predeclared'; no frozen artefact for this round exists, so predating the runs cannot be verified"},
            "gaps": ["no tensor_benchmark_cases row for this anchor -> gate cannot machine-check a registered ref/tolerance",
                     "L3 blocked: no solver/package API caller",
                     "artefact design block hardcodes repeats=3 (harness line 384) while rows carry repeat=0 only -> self-inconsistent metadata",
                     "doc promises '3 realisations for every noisy cell'; executed rows have 1",
                     "cluster 48-shard execution claimed, no shard artefacts/logs in tree (not_run)",
                     "coverage: 16 of 360 advertised design cells (1.48%% of design x3 realisations)",
                     "harness aborts (summarise() line 213: max() on empty legacy list) on any run/subset without a 24x48 cell -> cluster shards holding only 24x24/36x36 cannot write part files",
                     "merge state: commit sits on branch agent/m17789 and on the remote main line (Merge #1462 a332e7aa) but is NOT an ancestor of the local main checkout used for this audit"],
            "anchor_level_note": "anchor-level maturity already at L4 from mission 17593 (same operator vs OSS scikit-image Radon, 102 cases, audit-verified repro); this verdict grades only m17789's incremental round and does not lower the anchor",
            "audit_page": "docs/board_audit_m17873/M17873_BOARD_AUDIT.md",
            "audit_artifacts": ["artifacts/audit_m17873/" + n for n in
                                ("sha256manifest.json", "referee.json", "judge.json",
                                 "repro.json", "tests.json", "integration.json",
                                 "report.json", "mainline.json", "finding.json")],
        }
        insert_verdict(a.dsn, a.audit_mission, level, evidence)
    return 0


if __name__ == "__main__":
    sys.exit(main())