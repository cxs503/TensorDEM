#!/usr/bin/env python3
"""Board cross-audit verdict for m16065 (CASE-DEM-MINDLIN-TANGENTIAL) -> tensor_maturity_verdicts.

Run: python3 scripts/audit/m16127_verdict.py [--write]
Without --write it only renders/validates the row (dry run).
"""
from __future__ import annotations

import json
import pathlib
import sys

import psycopg

MISSION = 16127
REPO = "TensorDEM"
SCOPE = "m16065 CASE-DEM-MINDLIN-TANGENTIAL"
LEVEL = 4
DSN = "dbname=tensorplatform user=postgres"
OUT = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "audit_m16065"

EVIDENCE = {
    "impl_files": [
        "src/tensordem/contact_tangential.py:525 mindlin_backbone() 包络闭式 "
        "F=mu*F_n*[1-(1-|d|/d*)^1.5], d*=3*mu*F_n/(2*k_t0)",
        "src/tensordem/contact_tangential.py:535 mindlin_tangential_step() 有状态切向增量："
        "锚点回弹 F_r/d_r、unloading_ratio、滑移掩码、分离接触清零",
        "benchmarks/case_mindlin_tangential/run.py:120 ref_virgin (R1), :131 mdr_tangential_curve (R2), "
        ":190 run_law_curve (G1), :242/:288 动力学与反演驱动, :337 curve_errors (误差表), :151/:541 反演参考 R3",
        "benchmarks/case_mindlin_tangential/validate.py:21 TOL=0.05 预登记门（只读 results.json，不可回删重跑）",
    ],
    "entry": "PYTHONPATH=src python3 benchmarks/case_mindlin_tangential/run.py "
             "&& PYTHONPATH=src python3 benchmarks/case_mindlin_tangential/validate.py",
    "run_artifacts": {
        "git_tracked_at_14470b0e": [
            "benchmarks/case_mindlin_tangential/results.json (9113 B)",
            "benchmarks/case_mindlin_tangential/curves.npz (49518 B)",
            "benchmarks/case_mindlin_tangential/figures/fig1_ft_delta_curve.png (92069 B)",
            "benchmarks/case_mindlin_tangential/figures/fig2_microslip_error.png (61072 B)",
            "benchmarks/case_mindlin_tangential/figures/fig3_dt_convergence.png (45354 B)",
            "benchmarks/case_mindlin_tangential/figures/fig4_traction_field.png (94418 B)",
            "benchmarks/case_mindlin_tangential/figures/fig5_reversal.png (76020 B)",
            "benchmarks/case_mindlin_tangential/REPORT.md (17632 B)",
        ],
        "platform_report": "tensor_reports id=15188 (mission_id=16065, scenario_key=case-dem-mindlin-tangential, ①-⑥ 齐全)",
        "figures_on_platform": "backend/uploads/tensor_reports/15188/*.png sha256 与仓内逐张相同 "
                               "(53d2ddf6/44d2e6d3/7ca85947/646dabd6/09cba1a8)",
    },
    "tests": {
        "pytest": "PYTHONPATH=src python3 -m pytest tests/test_case_mindlin_tangential.py -q -> 7 passed in 8.49 s "
                  "(断言值而非仅执行: R1-vs-R2<1e-6, 微滑移/平台<5%, 线性律必须>5%, 反演初斜率<1%, "
                  "MDR 完全反向=2*delta*, delta*∝mu, 曲线确定性)",
        "gate": "benchmarks/case_mindlin_tangential/validate.py -> exit 0, VERDICT: PASS (8/8 checks)",
    },
    "integration_callers": [
        "src/tensordem/protocols/sandpile.py:94-96 from ..contact_tangential import mindlin_tangential_step",
        "src/tensordem/protocols/sandpile.py:833 / :1109 / :1169 三处真实 DEM 运行调用 "
        "(默认 tangential_model='mindlin_history', backbone='mindlin')",
        "src/tensordem/__init__.py:65 公共导出",
        "tests/test_dem_macro_anchor_closure.py:1 'Mindlin history contact, single rolling law, sandpile protocol' 覆盖该路径",
        "缺口: src/tensordem/dem3d.py 颗粒-颗粒切向仍为 Cundall-Strack（:123-128），Mindlin backbone 未接主引擎",
    ],
    "benchmark": {
        "referee_row": "tensor_benchmark_cases WHERE anchor_slug='CASE-DEM-MINDLIN-TANGENTIAL' -> 0 行 (2026-11-12 审计时, 全表 25 行)",
        "criterion_used": "使命 goal 文本 '<5%' + run.py/validate.py 预登记 TOLERANCE=0.05（两者一致）",
        "ref_vs_got": {
            "micro_slip_max_dev_over_muP": {"ref": "Mindlin 1953 闭式 1-(1-s)^1.5", "got": 1.503794540610078e-05,
                                             "auditor_recompute": 1.503794540610078e-05,
                                             "true_relative_error": 1.573089282761432e-05, "tol": 0.05, "pass": True},
            "gross_slip_plateau_N": {"ref": 3.0, "got": 3.0, "rel_err": 0.0, "tol": 0.05, "pass": True},
            "transition_delta_0.99muP": {"rel_err": 0.0025334821612249077, "tol": 0.05, "pass": True},
            "dynamics_G2_micro": {"got": 6.63449836385462e-05, "tol": 0.05, "pass": True},
            "reference_selfconsistency_R1_R2": 8.975190960806382e-12,
            "discrimination_linear_Cundall_Strack": {"micro": 0.1911167563965419, "expected": ">0.05 (算例有判别力)"},
            "auditor_independent_quadrature_vs_closed_form_max_dev_over_muP": 5.341862354679705e-07,
            "reversal_branch_F1": {"dev_over_muP": 1.171324037147707, "claimed": 1.171326287112455,
                                   "full_reversal_amplitude_over_delta_star": 0.6065, "analytic": 2.0,
                                   "verdict": "not reproduced -> capability boundary, honestly declared"},
        },
    },
    "repro_cmd": {
        "cmd": "PYTHONPATH=src python3 benchmarks/case_mindlin_tangential/run.py && "
               "PYTHONPATH=src python3 benchmarks/case_mindlin_tangential/validate.py",
        "audited": "干净 /tmp 副本（去 .git）：run.py exit 0, 56.3 s, overall_pass=True；validate.py exit 0, 8/8；"
                   "results.json 与交付件 0 个叶子差（浮点 1e-12 相对判等，跳过 wall_clock 类键）",
    },
    "external_anchor": {
        "literature": "Mindlin & Deresiewicz 1953, J. Appl. Mech. 20:327-344, DOI 10.1115/1.4010702 "
                      "(部分滑移构型 p.331)",
        "access": "原文本环境 HTTP 403 不可达 -> 未抄表；参考由闭式重构 + 独立 1-D 降维积分交叉验证（9.0e-12）",
        "auditor_crosscheck": "审计侧不 import TensorDEM，独立 Hertz 闭式重建 (E*,G*,a,p0,k_t0,d* 逐项 rel_diff=0)"
                              " 与独立 scipy 自适应求积（max dev 5.3e-7*muP），反演支独立重建得 2*delta* 与 1.171*muP 偏差",
        "goal_gap": "使命 goal 要求'目标数值从 ref 原文/手册查得、引用原表原页'未满足（已如实申报）",
    },
    "level_note": "L4=完整算例+逐项误差分析+Δt 一阶收敛+μ/P 阶梯+判别力(线性律 19.11%>5%)+干净副本逐比特复现+"
                  "接入 protocols.sandpile 默认路径。不给 L5：无新物理。"
                  "口径限制：G1 微滑移与 R1 为同一闭式的代数一致性检查（交付方 REPORT ⑤.2 自认），"
                  "非循环定量门为 G2 动力学(6.6e-5)与过渡点反解(0.2533%)。",
    "defects": {
        "F1": "反向支不可复现（dev 1.171*muP, 完全反向 0.6065δ* vs 2δ*）——交付方主动申报为能力边界；"
              "变法向力分支同缺，返修子项（不影响本算例消账）",
        "F2": "结果字段 *_rel_err 实为 |ΔF|/(muP) 归一化偏差（run.py:324,:356），不是相对误差；"
              "commit message '0.55 δ*' 与 results.json 0.6065δ* 不一致（取 results.json）",
        "F3": "tensor_benchmark_cases 该 anchor 0 行：机器裁判未登记（池侧补登，本审计不代填）",
        "F4": "遗留占位报告 id=15187 'probe test title' 仍 published（交付方已如实披露，无删除权限）",
        "F5": "平台图端点 404：图字节在 backend/uploads/tensor_reports/15188/，服务 cwd 解析到另一路径（部署级）",
    },
    "disposition": "approve with notes -> L4 消账; 返修子项 F1(反向支/变法向力)、F2(误差口径与 commit 数字一致)、"
                   "F3(平台裁判登记); 平台侧观察 F4/F5",
    "audit_page": "docs/board_audit_m16065/M16127_BOARD_AUDIT.md",
    "audit_artifacts": ["artifacts/audit_m16065/independent_check.json",
                        "scripts/audit/m16065_independent_check.py（不 import TensorDEM 的独立重算入口）"],
}

REQUIRED = ["impl_files", "entry", "run_artifacts", "tests", "integration_callers",
            "benchmark", "repro_cmd", "external_anchor"]


def main() -> None:
    missing = [k for k in REQUIRED if not EVIDENCE.get(k)]
    if missing:
        raise SystemExit(f"evidence missing keys: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdict.json").write_text(json.dumps(
        {"repo": REPO, "scope": SCOPE, "mission_id": MISSION, "level": LEVEL,
         "verdict_by": "board-audit", "evidence": EVIDENCE}, ensure_ascii=False, indent=1))
    blob = json.dumps(EVIDENCE, ensure_ascii=False)
    assert len(blob) < 200_000, len(blob)
    print(f"evidence keys OK, {len(blob)} chars")

    if "--write" not in sys.argv:
        print("dry run (pass --write to INSERT)")
        return
    with psycopg.connect(DSN) as c, c.cursor() as cur:
        cur.execute("""INSERT INTO tensor_maturity_verdicts
                       (repo, scope, mission_id, level, evidence, verdict_by)
                       VALUES (%s,%s,%s,%s,CAST(%s AS jsonb),%s)
                       RETURNING id, created_at""",
                    (REPO, SCOPE, MISSION, LEVEL, blob, "board-audit"))
        row = cur.fetchone()
        c.commit()
        print("inserted verdict id=%s at %s" % row)
        cur.execute("""SELECT repo, scope, mission_id, level, verdict_by, jsonb_typeof(evidence),
                              length(evidence::text)
                       FROM tensor_maturity_verdicts WHERE id=%s""", (row[0],))
        print("readback:", cur.fetchone())


if __name__ == "__main__":
    main()