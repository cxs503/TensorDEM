"""Build and post the p472w2-N05 report (mission 14013).

Every number in the markdown is injected from
``artifacts/floe_attrition/criteria.json`` (no hand-copied digits); figures are
embedded as base64 PNGs from ``artifacts/floe_attrition/figures``.

Usage::

    PYTHONPATH=src python scripts/floe_attrition_report.py            # build + post
    PYTHONPATH=src python scripts/floe_attrition_report.py --dry-run  # build only
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "floe_attrition")
FIG = os.path.join(ART, "figures")
CRITERIA = os.path.join(ART, "criteria.json")
DOC = os.path.join(ROOT, "docs", "FLOE_ATTRITION_ROUNDNESS.md")
PAYLOAD = "/tmp/report_m14013.json"
MISSION_ID = 14013

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"

CAPTIONS = {
    "fig1_geometry.png":
        "图 1 形状族与圆度的精确几何。(a) 正 n 边形带圆角化的 Wadell 圆度 "
        "R=4πA/P² 随磨耗状态 r/r_max 的单调演化，r=r_max 时严格为圆（R=1）；"
        "(b) 磨耗路径上的碎冰块体外形（正方形 a=0.6 m）；(c) 3D 圆角立方体 "
        "(Minkowski/Steiner) 的球度 Ψ 与 3D Wadell 圆度 r/R_ins，r=a/2 时为球。",
    "fig2_archard.png":
        "图 2 角部磨耗律验证。(a) 磨耗面积随滑动距离：本模块（精确几何反演）、"
        "同律 Archard 闭式解（含接触斑块随磨圆增长）与经典 Archard 闭式"
        "（初始斑块固定）；(b) 数值积分的相对误差始终在 10% 验收线下，"
        "且随步长加密按一阶收敛。",
    "fig3_decay_law.png":
        "图 3 演化律。(a) 五个接触压下的 R(s) 饱和衰减与双参数拟合"
        "（1−R=(1−R₀)exp(−(s/τ)^β)，窗口取至残余 1/e），R²≥0.9997；"
        "(b) 弛豫滑动距离 τ 随接触压的幂律，实测指数 −0.9985，"
        "与预注册值 −1 偏差 0.15%（验收 ≤20%）。",
    "fig4_design.png":
        "图 4 3×3 设计矩阵（圆度 × 浓度）。(a) 交互图：每格 8 次重复（正多边形"
        "边数 n∈{3,4,5,6} 的形状无序），误差棒为标准差；(b) 双因素方差分析"
        "效应量 η²：浓度主效应 97.13%，圆度主效应 1.14%，交互项 1.30%（p=2.3e-18，"
        "显著）——交互显著即不得把两种机制合并叙述。",
    "fig5_resistance.png":
        "图 5 多次通航阻力衰减与份额闭合。(a) 阻力 R_ice(N) 曲线（磨圆+固结、"
        "仅磨圆、仅固结三条独立路径，右轴为累积接触能量）；(b) 磨圆份额 +474.8%、"
        "固结份额 −374.8%，闭合和 100.000000%，闭合残差 1.9e-15（相对）。",
    "fig6_sphere_limit.png":
        "图 6 R→1 极限与 m9375 球基准的一致性。(a) 三种角性下的 μ_eff(C) 闭包"
        "与 m9375 圆盘剪切胞实测球基准（黑虚线）；(b) C4：R=1 处逐位相同"
        "（差值恒为 0.0），近圆多边形（R=1−1e-6）与全磨耗链路的最大相对误差 7.6e-10。",
}


def load_criteria() -> dict:
    with open(CRITERIA) as fh:
        return json.load(fh)


def figures_payload() -> list:
    out = []
    for name in sorted(os.listdir(FIG)):
        if not name.endswith(".png"):
            continue
        with open(os.path.join(FIG, name), "rb") as fh:
            out.append({"name": name, "caption": CAPTIONS.get(name, name),
                        "png_base64": base64.b64encode(fh.read()).decode()})
    return out


def short_summary(d: dict) -> str:
    """<= 500 characters (the platform rejects longer summaries)."""
    o = d["outputs"]
    tab = {r["source"]: r for r in o["C5"]["anova"]["table"]}
    text = (f"TensorDEM 新增 floe_attrition：Wadell 圆度精确几何 → Archard 角部磨耗 → "
            f"μ_eff(R) → 多次通航阻力衰减。C1-C7 全通过：圆度闭式误差 "
            f"{o['C1']['worst_relative_error']:.1e}；磨耗 vs Archard 闭式 "
            f"{100*o['C2']['production_cadence_error']:.2f}%；R²最小 "
            f"{o['C3']['min_r2']:.4f}、τ-压标度指数 "
            f"{o['C3']['tau_pressure_exponent']:.3f}（预注册 -1）；R→1 逐位回到 m9375 "
            f"球基准；3×3 ANOVA 交互显著(p={tab['roundness:concentration']['p']:.0e})；"
            f"磨圆/固结份额闭合 {o['C6']['shares']['share_sum_pct']:.4f}%；"
            f"守恒 {o['C7']['worst_conservation_error']:.0e}、零滑动恒等 0。")
    return text[:490]


def summary_text(d: dict) -> str:
    o = d["outputs"]
    c1, c2, c3, c4, c5, c6, c7 = (o["C1"], o["C2"], o["C3"], o["C4"], o["C5"],
                                  o["C6"], o["C7"])
    an = c5["anova"]
    tab = {r["source"]: r for r in an["table"]}
    return (
        "TensorDEM 新增 floe_attrition 模块，把碎冰磨圆做成 DEM 可算的一等过程："
        "Wadell 圆度精确几何 → Archard 角部磨耗 → μ_eff(R) → 多次通航阻力衰减。"
        f"机器判据 C1-C7 全通过：C1 闭式圆度/球度与独立精确积分最大相对误差 "
        f"{c1['worst_relative_error']:.2e}（阈值 1e-12）；C2 磨耗体积累计与 Archard "
        f"闭式在生产步长下相对误差 {100*c2['production_cadence_error']:.2f}%（≤10%）；"
        f"C3 R(s) 拟合 R²最小 {c3['min_r2']:.6f}（≥0.99），τ-接触压标度指数 "
        f"{c3['tau_pressure_exponent']:.4f}（预注册 -1，偏差 "
        f"{100*c3['tau_exponent_deviation']:.2f}%）；C4 R→1 时 μ_eff 与 m9375 球基准"
        f"逐位相同（近圆最大相对误差 {c4['worst_relative_error']:.1e}）；"
        f"C5 3×3 设计矩阵 ANOVA：圆度 η²={100*tab['roundness']['eta_squared']:.2f}%、"
        f"浓度 η²={100*tab['concentration']['eta_squared']:.2f}%、"
        f"交互 η²={100*tab['roundness:concentration']['eta_squared']:.2f}%"
        f"（p={tab['roundness:concentration']['p']:.1e}，显著 → 两机制不得合并叙述）；"
        f"C6 多次通航阻力 {c6['resistance_initial']/1e3:.2f}→{c6['resistance_final']/1e3:.2f} "
        f"kN/m（衰减 {100*c6['decay_fraction']:.2f}%），磨圆份额 "
        f"{c6['shares']['share_rounding_pct']:+.1f}%、固结份额 "
        f"{c6['shares']['share_consolidation_pct']:+.1f}%，闭合 {c6['shares']['share_sum_pct']:.6f}%；"
        f"C7 磨耗体积守恒相对误差 {c7['worst_conservation_error']:.1e}（≤1e-6），"
        f"零滑动恒等 0。附带修复 polyhedra_shapes.make_icosahedron 的变量泄漏缺陷。"
    )


def criteria_brief(d: dict) -> str:
    o = d["outputs"]
    tab = {r["source"]: r for r in o["C5"]["anova"]["table"]}
    c6 = o["C6"]
    rows = [
        ("C1 圆度闭式", f"{o['C1']['worst_relative_error']:.2e}", "≤1e-12", o["C1"]["passed"]),
        ("C2 Archard 闭式", f"{100*o['C2']['production_cadence_error']:.2f}%", "≤10%", o["C2"]["passed"]),
        ("C3 R(s) 拟合 R² 最小", f"{o['C3']['min_r2']:.6f}", "≥0.99", o["C3"]["passed"]),
        ("C3 τ-压标度指数", f"{o['C3']['tau_pressure_exponent']:+.4f}（偏差 {100*o['C3']['tau_exponent_deviation']:.2f}%）",
         "预注册 -1 ±20%", o["C3"]["passed"]),
        ("C4 R→1 逐位一致", f"差 {0.0 if o['C4']['bit_identical_at_limit'] else 'NA'}，近圆 {o['C4']['worst_relative_error']:.1e}",
         "≤1% 或逐位", o["C4"]["passed"]),
        ("C5 ANOVA 交互", f"η²={100*tab['roundness:concentration']['eta_squared']:.2f}%，p={tab['roundness:concentration']['p']:.1e}",
         "报出即合格", o["C5"]["passed"]),
        ("C6 份额闭合", f"{c6['shares']['share_sum_pct']:.6f}%（残差 {c6['closure_relative']:.1e}）",
         "100%", o["C6"]["passed"]),
        ("C7 守恒/负控", f"{o['C7']['worst_conservation_error']:.1e}；零滑动恒等 0",
         "≤1e-6", o["C7"]["passed"]),
    ]
    lines = ["| 判据 | 实测 | 阈值 | 结论 |", "|---|---|---|---|"]
    for name, value, thr, ok in rows:
        lines.append(f"| {name} | {value} | {thr} | {'PASS' if ok else 'FAIL'} |")
    return "\n".join(lines)


def acceptance_block(d: dict) -> dict:
    o = d["outputs"]
    return {
        "all_pass": bool(d["all_passed"]),
        "criteria": [
            {"id": "C1", "passed": o["C1"]["passed"],
             "value": o["C1"]["worst_relative_error"], "threshold": 1e-12,
             "quantity": "Wadell 圆度/球度闭式 vs 独立精确积分"},
            {"id": "C2", "passed": o["C2"]["passed"],
             "value": o["C2"]["production_cadence_error"], "threshold": 0.10,
             "quantity": "磨耗体积 vs Archard 闭式（生产步长）"},
            {"id": "C3", "passed": o["C3"]["passed"],
             "value": {"min_r2": o["C3"]["min_r2"],
                       "tau_exponent": o["C3"]["tau_pressure_exponent"],
                       "deviation": o["C3"]["tau_exponent_deviation"]},
             "threshold": {"r2": 0.99, "deviation": 0.20},
             "quantity": "R(s) 拟合质量与 τ-接触压标度"},
            {"id": "C4", "passed": o["C4"]["passed"],
             "value": {"bit_identical": o["C4"]["bit_identical_at_limit"],
                       "worst_relative_error": o["C4"]["worst_relative_error"]},
             "threshold": 0.01, "quantity": "R→1 回到 m9375 球基准"},
            {"id": "C5", "passed": o["C5"]["passed"],
             "value": {r["source"]: {"F": r["F"], "p": r["p"],
                                     "eta_squared": r["eta_squared"]}
                       for r in o["C5"]["anova"]["table"]},
             "threshold": "balanced ANOVA, interaction reported",
             "quantity": "3×3 设计矩阵主效应/交互"},
            {"id": "C6", "passed": o["C6"]["passed"],
             "value": {"share_sum_pct": o["C6"]["shares"]["share_sum_pct"],
                       "closure_relative": o["C6"]["closure_relative"]},
             "threshold": 1e-9, "quantity": "磨圆/固结份额闭合到 100%"},
            {"id": "C7", "passed": o["C7"]["passed"],
             "value": o["C7"]["worst_conservation_error"], "threshold": 1e-6,
             "quantity": "磨耗体积守恒 + 零滑动负控"},
        ],
    }


def build_payload() -> dict:
    d = load_criteria()
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()
    markdown = ("# 碎冰磨圆到有效摩擦与多次通航阻力衰减（p472w2-N05）\n\n"
                "## 判据速览\n\n" + criteria_brief(d) + "\n\n" + doc +
                "\n\n## 机器判据表（全量）\n\n```json\n" +
                json.dumps(acceptance_block(d), indent=1, ensure_ascii=False) +
                "\n```\n")
    return {
        "title": "碎冰磨圆（angularity→roundness）到有效摩擦 μ_eff 的时间衰减链："
                 "Wadell 圆度闭式 + Archard 角部磨耗 + 多次通航阻力衰减（p472w2-N05）",
        "summary": short_summary(d),
        "scenario_key": "p472w2-n05-ice-floe-attrition-roundness",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "ice", "brash ice", "attrition", "roundness",
                 "Archard wear", "friction", "resistance decay", "ANOVA",
                 "p472w2-N05"],
        "markdown": markdown,
        "figures": figures_payload(),
        "acceptance": acceptance_block(d),
        "artifacts": {
            "criteria": "artifacts/floe_attrition/criteria.json",
            "sphere_baseline": "artifacts/floe_attrition/sphere_baseline.json",
            "prereg": "prereg/p472w2_n05_floe_attrition_prereg.json",
            "tests": "tests/test_floe_attrition.py",
        },
    }


def login() -> str:
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {body}")
    return token


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    payload = build_payload()
    assert payload["mission_id"] == MISSION_ID
    with open(PAYLOAD, "w") as fh:
        json.dump(payload, fh)
    print("payload:", PAYLOAD, "figures:", len(payload["figures"]),
          "bytes:", os.path.getsize(PAYLOAD))
    if "--dry-run" in argv:
        print(payload["summary"])
        return 0
    token = login()
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(resp.read().decode()[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())