#!/usr/bin/env python3
"""m15708: post the audit report to the platform and insert the maturity verdict.

Run once.  Writes docs/board_audit/m15708/report_payload.json (without the
base64 figure bodies) for provenance.
"""
from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
HERE = pathlib.Path(__file__).resolve().parent
WT = HERE.parents[2]                     # /root/tp-agent/work/m15708
REPRO = WT / "runs/m15708/repro/artifacts/case_rotating_drum_r2/figures"
PRIS = HERE / "reviewed/artifacts/case_rotating_drum_r2/figures"
ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and not l.startswith("#") and "=" in l)

AUD = pathlib.Path("/root/tp-agent/work/m15649")   # audited worktree (read only)
VERD = json.loads((HERE / "verdict.json").read_text())
C4, C5, C6 = VERD["C4_applicability"], VERD["C5_dt_sensitivity"], VERD["C6_rev20_plateau"]
C7, C9 = VERD["C7_report_sections"], VERD["C9_tests"]

EVIDENCE = {
    "anchor": "CASE-ROTATING-DRUM",
    "audited_mission": "m15649",
    "audited_commit": "074148fc (branch agent/m15649, TensorDEM)",
    "impl_files": [
        "/root/tp-agent/work/m15649/src/tensordem/drum_segregation/kernel2d.py",
        "/root/tp-agent/work/m15649/src/tensordem/drum_flow/regimes.py",
    ],
    "entry": "/root/tp-agent/work/m15649/benchmarks/rotating_drum_froude_r2.py",
    "entry_locator": "benchmarks/rotating_drum_froude_r2.py:233 main(); "
                     "benchmarks/rotating_drum_froude_r2_report.py:39 main()",
    "run_artifacts": [
        "/root/tp-agent/work/m15649/artifacts/case_rotating_drum_r2/summary.json",
        "/root/tp-agent/work/m15649/artifacts/case_rotating_drum_r2/merged.json",
        "/root/tp-agent/work/m15649/artifacts/case_rotating_drum_r2/table_main.md",
        "/root/tp-agent/work/m15649/artifacts/case_rotating_drum_r2/cases/case_main_fr0.7.json",
        "/root/tp-agent/work/m15649/artifacts/case_rotating_drum_r2/figures/fig1_release_angle.png",
        "/root/tp-agent/work/m15708/runs/m15708/repro/artifacts/case_rotating_drum_r2/figures/fig3_S_rev.png",
        "/root/tp-agent/work/m15708/docs/board_audit/m15708/verdict.json",
    ],
    "tests": [
        "/root/tp-agent/work/m15649/tests/test_drum_froude_r2.py",
        "test_detachment_angle_is_arccos_minus_froude PASS",
        "test_literature_bands_are_ordered_and_cover_the_axis PASS",
        "test_measured_classifier_uses_frozen_rule_order PASS",
        "test_bed_preparation_reaches_the_rest_criterion PASS",
        "test_contact_counting_and_kinematic_observables_are_produced PASS",
        "test_centrifuging_at_froude_above_one_is_measured_not_asserted FAIL "
        "(r_p05_over_R=0.3181 vs >0.75, reproduced twice)",
    ],
    "integration_callers": [
        "benchmarks/rotating_drum_mixing_resting.py",
        "scripts/run_drum_seg_verification.py",
    ],
    "integration_callers_note": "repo-internal benchmark/script call sites only; no library or "
                                "platform-side caller of kernel2d -> L3 not met",
    "benchmark": {
        "judge_rows_in_tensor_benchmark_cases": 0,
        "judge_query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
                       "WHERE anchor_slug='CASE-ROTATING-DRUM' -> 0 rows",
        "ref": "theta_d = arccos(-Fr) [deg] (exact single-particle radial force balance; "
               "180 deg saturated for Fr>=1)",
        "cases": C4["main_cases"],
        "cases_with_finite_release_angle": C4["finite_release_cases"],
        "pass_under_20pct_among_valid": C4["pass_among_finite"],
        "max_err_among_valid": C4["max_err_among_finite"],
        "err_fr1.0_saturated_ref": 0.1853,
        "err_fr1.1_saturated_ref": 0.1497,
        "table_vs_json_mismatches": VERD["C3_table_vs_json"]["mismatches"],
        "dt_sensitivity": {"err_dt_1e-4": C5.get("err_dt_base"),
                           "err_dt_5e-5": C5.get("err_dt_half"),
                           "ratio": C5.get("err_ratio")},
        "p5_plateau": {"delivered_summary_plateau": C6["plateau_summary"],
                       "rev20_in_merged": C6["rev20_in_merged"],
                       "regenerated_S_rev5_rev10_rev20": [0.5109, 0.2927, 0.5007]},
        "centrifuging_onset": {"measured": C6["centrifuging_onset_measured"],
                               "rel_err": "NaN", "measured_regime_fr1.0": "cataracting"},
        "spearman_P1_P2_P3": [0.6333, 0.9833, 0.8],
        "report_sections_present": C7["sections_present"],
        "report_sections_missing": C7["missing"],
    },
    "repro_cmd": "cd /root/tp-agent/work/m15708/runs/m15708/repro && "
                 "PYTHONPATH=src python benchmarks/rotating_drum_froude_r2.py "
                 "--fr 0.7 --revs 10 --out fresh_cases && "
                 "PYTHONPATH=src python benchmarks/rotating_drum_froude_r2_report.py && "
                 "PYTHONPATH=src python -m pytest -q tests/test_drum_froude_r2.py",
    "external_anchor": [
        "/root/tp-agent/work/m15649/prereg/CASE_ROTATING_DRUM_R2.json",
        "/root/tp-agent/work/m15649/docs/OSS_DRUM_R2_RESEARCH.md",
    ],
    "audit_artifacts": {
        "report": "/root/tp-agent/work/m15708/docs/board_audit/m15708/AUDIT_M15708.md",
        "verdict_json": "/root/tp-agent/work/m15708/docs/board_audit/m15708/verdict.json",
        "verify_script": "/root/tp-agent/work/m15708/docs/board_audit/m15708/verify_r2.py",
        "reviewed_snapshot": "/root/tp-agent/work/m15708/docs/board_audit/m15708/reviewed",
        "repro_tree": "/root/tp-agent/work/m15708/runs/m15708/repro",
    },
    "findings": [
        "B1 blocking: tensor_benchmark_cases has 0 rows for CASE-ROTATING-DRUM -> no machine judge",
        "B2 blocking: delivered test suite 1/6 FAIL "
        "(test_centrifuging_at_froude_above_one_is_measured_not_asserted, r_p05=0.3181<0.75, x2)",
        "B3 major: '9/9 <20%' includes 2 rows whose reference is the saturated 180 deg "
        "(Fr=1.0 err 18.53%, Fr=1.1 15.0%); valid quantitative result is 7/7, max 11.32%",
        "B4 major: dt=1e-4 err 9.09% vs dt=5e-5 err 1.12% at Fr=0.3 -> production dt not converged",
        "B5 major: rev20 computed (S 0.511/0.293/0.501, non-monotone) but not merged; "
        "summary.plateau={} -> P5 plateau not demonstrated, data contradict it",
        "B6 major: centrifuging onset acceptance undecided (null/NaN) while measured regime at "
        "Fr=1.0/1.1 is cataracting -> should be booked FAIL, not blank",
        "B7 major: report has 4 of 6 required sections (missing convergence/sensitivity and "
        "failure-modes/artifacts)",
        "B8 minor: post-run prereg amendment d_s 0.006->0.010 m (disclosed) leaves rev10coarse unrun",
        "B9 minor: oss_solutions row for feature_slug=CASE-ROTATING-DRUM not inserted",
        "B10 minor (honest): quantitative anchor is a single-particle closed form, no measured "
        "bed-angle experiment",
    ],
    "verified_positive": [
        "ref/got/err re-derived from the closed form: 12/12 case JSONs consistent",
        "table_main.md matches the per-case JSON on 9/9 rows (no hand-edited table)",
        "fresh 10-rev run at Fr=0.7 reproduces delivered values exactly "
        "(136.9 deg, 1.81%, r_p05 0.35, cataracting, resting_ok=True, 93.9 s)",
        "resting_ok 9/9 True, settle_rms_max 3.92e-3 < 5e-3 (round-1 defect fixed, measured)",
        "P1/P2/P3 Spearman 0.633/0.983/0.800 signs as predicted",
    ],
    "not_run": [
        "3-D analogue (kernel is a 2-D slice)",
        "comparison against measured drum experiments (no data source in repo)",
        "cluster (loong64) rerun of the full 11-case set: audit reran one case + tests on the control node",
    ],
}

REPORT_MD = (HERE / "AUDIT_M15708.md").read_text()
MD = REPORT_MD + """

---

## 6. 审计者口径声明

* 判级 = **L2**（声明级也是 L2，无虚标上调；下调项见 B1/B2/B4/B7）。
* 基准裁判（`tensor_benchmark_cases`）对该锚点**无行**，故"基准通过"一词在本报告里一律写
  **WITHHELD**，只在复算/复现意义上写"已核验"。
* 所有数字由 `verify_r2.py` 从交付工件程序化产生；单算例与测试由审计者在独立 worktree
  `m15708` 的副本 `runs/m15708/repro` 上重跑（控制面 99，非集群）。
"""


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def fig(path, caption):
    return {"name": path.name, "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode()}


status, tok = api("POST", "/auth/login",
                  {"username": ENV["TENSOR_AGENT_USER"], "password": ENV["TENSOR_AGENT_PASS"]})
token = tok.get("access_token")
print("login", status, bool(token))

payload = {
    "title": "交差审计 m15708 ⊣ m15649（TensorDEM / CASE-ROTATING-DRUM 第2轮）：L2 确认，基准行缺席+测试1/6失败",
    "summary": ("独立复算+重跑审计 m15649：数字全部可从 arccos(-Fr) 复算且表与 JSON 零偏差、"
                "单算例重跑逐位一致（非伪造）；但机读基准行 0 行、交付测试 1/6 失败、"
                "9/9 PASS 中 2 行为饱和引用（有效 7/7，max 11.32%）、dt 未收敛（9.09%→1.12%）、"
                "P5 平台未交付且数据反证、报告缺 ⑤⑥。判 L2，请求返修（基准/收敛面）。"),
    "scenario_key": "board_audit",
    "mission_id": 15708,
    "tags": ["board-audit", "TensorDEM", "CASE-ROTATING-DRUM", "m15649", "L2",
             "honest-FAIL-partial"],
    "markdown": MD,
    "figures": [
        fig(PRIS / "fig1_release_angle.png",
            "交付图 1（审计快照）：释放角 vs Fr，got（DEM 实测）与闭式 arccos(-Fr) 对比；"
            "可见 Fr>=1 两行引用为饱和 180°（B3）"),
        fig(REPRO / "fig3_S_rev.png",
            "审计用交付工件重跑报告脚本补出的 fig3：S(rev) 在 5/10/20 圈为 0.511/0.293/0.501，"
            "非单调 → 无稳态平台（B5，prereg P5 应记 failed）"),
    ],
}
payload_copy = {k: v for k, v in payload.items() if k != "figures"}
payload_copy["figures"] = [{"name": f["name"], "caption": f["caption"],
                            "bytes": len(base64.b64decode(f["png_base64"]))}
                           for f in payload["figures"]]
(HERE / "report_payload.json").write_text(json.dumps(payload_copy, indent=1, ensure_ascii=False))

st, res = api("POST", "/tensor/reports", payload, token)
rid = res.get("id") or res.get("report_id")
print("report post", st, {"id": rid, "keys": list(res)[:8]})

EVIDENCE["audit_report_id"] = rid
ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
scope = "m15649 CASE-ROTATING-DRUM"
check = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-c",
                        f"SELECT id FROM tensor_maturity_verdicts WHERE mission_id=15708"],
                       capture_output=True, text=True)
print("existing", check.stdout.strip())
if not check.stdout.strip():
    sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
           f"VALUES ('TensorDEM', '{scope}', 15708, 2, '{ev}', 'board-audit') RETURNING id")
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-c", sql], capture_output=True, text=True)
    print("insert", r.returncode, r.stdout.strip(), r.stderr.strip()[:300])
r2 = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-c",
                     "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state "
                     "FROM tensor_maturity_verdicts WHERE mission_id=15708"],
                    capture_output=True, text=True)
print(r2.stdout, r2.stderr[:200])