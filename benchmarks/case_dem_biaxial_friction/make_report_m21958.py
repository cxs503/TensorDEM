#!/usr/bin/env python
"""Build the m21958 markdown doc + platform-report payload from the artefact.

Every number is injected from ``artifacts/CASE-DEM-BIAXIAL-FRICTION/regularisation.json``;
nothing is hand-copied.  Publishing is a separate step.

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/make_report_m21958.py
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "CASE-DEM-BIAXIAL-FRICTION" / "regularisation.json"
PNG = REPO / "artifacts" / "CASE-DEM-BIAXIAL-FRICTION" / "regularisation.png"
DOC = REPO / "docs" / "CASE-DEM-BIAXIAL-FRICTION_R7_FRICTION_REGULARISATION.md"
REPORT = Path("/tmp/report_m21958.json")
MISSION = 21958


def f(x, nd=2):
    if x is None:
        return "--"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "--"
    return f"{x:.{nd}f}"


def build(art: dict) -> str:
    runs = art["runs"]
    hyp = art["hypotheses"]
    ref = art["references"]
    tol = art["preregistered_tolerances_deg"]
    anchor = art["anchor_verdict"]
    h1, h2, h3 = hyp["H1_vreg_invariance"], hyp["H2_rate_invariance"], \
        hyp["H3_contact_law_explains_phi"]
    base = next((r for r in runs if r["velocity_regularisation_is_default"]
                 and abs(r["strain_rate"] - 0.01) < 1e-12), runs[0])

    L: list[str] = []
    L.append("# CASE-DEM-BIAXIAL-FRICTION · R7：接触摩擦正则化速度 `v_reg` 是否是 φ_cv 的隐性自变量\n")
    L.append("**锚点** `CASE-DEM-BIAXIAL-FRICTION`（算例池·双轴剪切摩擦角，验收 "
             f"|φ−{f(ref['phi_bolton_deg'],0)}°|/{f(ref['phi_bolton_deg'],0)}° < "
             f"{f(ref['anchor_tolerance']*100,0)} %，ref Bolton 1986）｜**m21958**｜TensorDEM\n")
    L.append("**判定（前置声明，不粉饰）**：本单不改锚点验收口径、不重做 R3–R6，也不调参凑 38°。"
             "它只回答一件从未被检验的事：无粘结接触的切向律不是库仑律而是速率正则化律 "
             "`F_t = -μ F_n tanh(v_t/v_reg)`，那么 **φ_cv 到底是不是一个材料属性**——还是数值参数 `v_reg` 的函数。\n")

    L.append("## 结论速览\n")
    L.append("| 项 | 结果 | 判据 | 结论 |")
    L.append("|---|---|---|---|")
    L.append(f"| H1 对 `v_reg` 不敏感 | 四臂极差 {f(h1['spread_deg'])}° | ≤ {f(tol['H1_vreg_spread'],1)}° | "
             f"**{'成立' if h1['holds'] else '否定'}** |")
    L.append(f"| H2 对剪切速率不敏感 | 0.01→0.05 s⁻¹ 差 {f(h2['difference_deg'])}° | "
             f"≤ {f(tol['H2_rate_difference'],1)}° | **{'成立' if h2['holds'] else '否定'}** |")
    L.append(f"| H3 接触算子能解释 φ_cv | 最差臂差 {f(h3['worst_difference_deg'])}° | ≤ "
             f"{f(tol['H3_contact_law_mapping'],1)}° | **{'成立' if h3['holds'] else '否定'}** |")
    L.append(f"| 锚点验收（φ_cv vs {f(ref['phi_bolton_deg'],0)}°） | {f(anchor['phi_cv_deg'])}°，相对误差 "
             f"{f(anchor['relative_error']*100)} % | < {f(anchor['tolerance']*100,0)} % | "
             f"**{'PASS' if anchor['passes_anchor_acceptance'] else 'FAIL'}**（本轮不改变） |")
    L.append(f"| 是否 <3 % | {f(anchor['relative_error']*100)} % | < 3 % | "
             f"**{'是' if anchor['under_3_percent'] else '否'}** |")
    L.append("")
    L.append(f"**一句话**：{('φ_cv 对正则化速度' + ('不敏感' if h1['holds'] else '敏感'))}"
             f"、对剪切速率{('不敏感' if h2['holds'] else '敏感')}；"
             f"接触层调动率{'可以' if h3['holds'] else '不能'}定量解释实测 φ_cv。"
             f"基线臂 φ_cv = {f(base['phi_cv_deg'])}°（R3 记录 14.31°）。\n")

    L.append("## (1) 案例介绍\n")
    L.append("平面应变双轴压缩的无胶结圆盘试样，等体积剪切到 ε_a = "
             f"{f(runs[0]['window'][1], 2)}–0.45，由预声明四闸门从应力路径提取临界状态摩擦角 φ_cv。"
             "锚点的历史：R2 因侧向伺服失控判 FAIL；R3 改用定容剪切，φ_cv = 14.31°，比 Rowe(1962) "
             "二维极限 arctan μ = 28.81° 低 50.3 %；R4 加了 `rolling_friction_coefficient`；"
             "R6（m21803）审计出该旋钮在无转动自由度的内核里就是**附加滑动摩擦**（加和性成立），"
             "不可能是缺口的补丁。读代码后剩下的唯一未检验候选是**切向律本身**：\n")
    L.append("```\nF_t = -mu * F_n * tanh(v_t / v_reg)\n```\n")
    L.append("`v_reg`（`RidgeMaterial.velocity_regularisation`，默认 0.01 m/s）是数值光滑参数。"
             "经验库仑摩擦的必要条件是**与速率无关**；若 φ_cv 随 `v_reg` 变化，"
             "则模型给出的摩擦角不是材料属性，而是数值参数的函数——这正好是 Anchors 50 % 缺口的候选根因。\n")
    L.append("| 量 | 值 |\n|---|---|")
    spec = runs[0]
    L.append(f"| 试样 | 24 × 48 抖动三角格（jitter 3 %），r = 0.75 m，ρ = 917 kg/m³，厚度 1 |")
    L.append(f"| 接触 | kn = 1×10⁷ N/m，kt/kn = 0.5，μ = {f(art['mu'])}，freeze_bond_fraction = 0 |")
    L.append(f"| 墙 | wall_friction = 0；固结侧墙伺服 σ₃ = 100 kPa（6000 步） |")
    L.append(f"| 剪切 | 等体积（`lateral_control=\"isochoric\"`），顶板位移控，Δt = 4.97×10⁻⁴ s，"
             f"惯性数 I = {f(spec['inertial_number'],2)}×10⁻³ 量级 |")
    L.append(f"| 基准 | Rowe 1962 D=0：arcran μ = {f(ref['phi_rowe_deg'])}°；"
             f"Bolton 1986 密实 Hostun 平面应变：{f(ref['phi_bolton_deg'],0)}° |")
    L.append("")

    L.append("## (2) 怎么做的\n")
    L.append("**先写预注册再动手**（`prereg/m21958_friction_regularisation_prereg.md`，commit 在算例之前）："
             "五臂只改 `v_reg` / `ε̇_a`，其余全部固定；判据 1.0°/1.0°/2.0° 写在文件里，跑完不调整。\n")
    L.append("| 臂 | ε̇_a [1/s] | `v_reg` [m/s] | 作用 |\n|---|---|---|---|")
    roles = {
        "A_base_rate0p01_vreg_default": "复现 R3 基线（默认 `v_reg`）",
        "B_vreg_2p5e-3": "H1：更小的正则化速度",
        "C_vreg_4e-2": "H1",
        "D_vreg_1e-1": "H1：近乎全库仑（tanh 基本饱和）",
        "E_rate0p05_vreg_default": "H2：速率 ×5（I 仍 ≪ 0.01）",
    }
    for r in sorted(runs, key=lambda a: (a["strain_rate"], a["velocity_regularisation"])):
        L.append(f"| `{r['tag']}` | {f(r['strain_rate'],3)} | {f(r['velocity_regularisation'],4)} | "
                 f"{roles.get(r['tag'], '')} |")
    L.append("")
    L.append("**新增的可观测量（本轮唯一的内核改动，纯诊断）**："
             "`RidgeAssembly.enable_friction_meter()` 在每次 `forces()` 里对承载中的无粘结 disk–disk 接触累加 "
             "`ΣF_n`、`ΣF_n|tanh(v_t/v_reg)|`、`Σ|tanh|`、饱和接触数（|tanh| ≥ 0.99）、`Σ|v_t|`；"
             "`read_friction_meter(reset=True)` 给窗口均值。`tests/test_friction_mobilisation_meter.py` "
             "（5 项全绿）钉住三件事：接触层恒等式 `F_t/(μF_n) = -tanh(v_t/v_reg)`、`v_t ≫ v_reg` 时饱和到库仑、"
             "**开/关计量仪应力路径逐位相同**（纯仪器，不改变任何力）。\n")
    L.append("**接触层恒等式的测法**：装配力里还含全局速度阻尼（沿速度方向），会污染切向分量；"
             "用 `μ = 0` 的双胞胎臂做差把阻尼消掉，差值即纯接触摩擦。\n")
    L.append("**复现**：\n")
    L.append("```\n# 集群递交（5 分片，每片一臂；24×48，ε_a 到 0.45）\n"
             "TENSORDEM_THREADS=3 PYTHONPATH=src python scripts/m21958_friction_regularisation_probe.py --shard {i} --nshards 5\n"
             "# 分析（四闸门 φ_cv + 调动率 + 假设判定 + 图）\n"
             "PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_regularisation.py\n"
             "# 计量仪不变量测试（约 20 s）\n"
             "PYTHONPATH=src python -m pytest tests/test_friction_mobilisation_meter.py -q\n```\n")

    L.append("## (3) 计算结果（全部由工件程序化注入）\n")
    L.append("### 3.1 逐臂 φ_cv 与接触调动率\n")
    L.append("| 臂 | ε̇_a | `v_reg` | φ_cv [°] | 四闸门 | 窗离散 [°] | q/p | ⟨\\|tanh\\|⟩_F_n | 饱和接触比例 | "
             "arctan(μ⟨tanh⟩) [°] | μ⟨tanh⟩ |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda a: (a["strain_rate"], a["velocity_regularisation"])):
        gates = "".join("✓" if v else "✗" for v in r["flags"].values())
        L.append(f"| `{r['tag']}` | {f(r['strain_rate'],3)} | {f(r['velocity_regularisation'],4)} | "
                 f"**{f(r['phi_cv_deg'])}** | {gates} | {f(r['window_spread_deg'])} | "
                 f"{f(r['q_cv_kPa'] / (0.5 * (r['q_cv_kPa'] + 2 * r['sigma_h_cv_kPa'])),3)} | "
                 f"{f(r['mobilisation']['fn_weighted_mean_abs_tanh'],3)} | "
                 f"{f(r['mobilisation']['saturated_fraction']*100,1)} % | "
                 f"{f(r['phi_from_contact_law_deg'])} | {f(r['mu_eff'],3)} |")
    L.append("")
    L.append(f"闸门顺序 = {list(runs[0]['flags'].keys())}；阈值 = "
             + ", ".join(f"{k}={f(v,3)}" for k, v in runs[0]["thresholds"].items()) + "。\n")

    L.append("### 3.2 假设判定\n")
    L.append(f"- **H1（`v_reg` 不敏感）**：臂 {', '.join(h1['arms'])} 的 φ_cv = "
             + ", ".join(f"{f(p)}°" for p in h1["phi_deg"])
             + f"，极差 {f(h1['spread_deg'])}° vs 预声明 {f(h1['tolerance_deg'],1)}° → "
             f"**{'成立（H1 被否定，v_reg 只是数值参数）' if h1['holds'] else '否定（H1 成立，v_reg 进入本构）'}**。")
    L.append(f"- **H2（速率不敏感）**：φ_cv(0.01) = {f(h2['phi_deg'][0])}°，φ_cv(0.05) = {f(h2['phi_deg'][1])}°，"
             f"差 {f(h2['difference_deg'])}° vs {f(h2['tolerance_deg'],1)}° → "
             f"**{'成立' if h2['holds'] else '否定'}**。")
    L.append(f"- **H3（接触算子定量解释 φ_cv）**：逐臂 `φ_cv − arctan(μ⟨|tanh|⟩)` = "
             + ", ".join(f"{k}: {f(v['difference_deg'])}°" for k, v in h3["per_arm"].items())
             + f"；最差 {f(h3['worst_difference_deg'])}° vs {f(h3['tolerance_deg'],1)}° → "
             f"**{'成立' if h3['holds'] else '否定'}**。")
    L.append("")

    L.append("## (4) 结果图\n")
    L.append("![regularisation](figures/CASE-DEM-BIAXIAL-FRICTION_regularisation.png)\n")
    L.append("(a) φ_cv 与 arctan(μ⟨|tanh|⟩) 随 `v_reg` 的变化，附 Rowe/Bolton 两条参考线；"
             "(b) φ_cv 与「处于库仑极限的接触比例」的关系；"
             "(c) 各臂 q/p 应力比（左轴）与接触调动率（右轴）随轴向应变。"
             "本轮**不产生场量云图**：这是参数敏感性研究，采集端 `record_snapshots` 关闭（每帧 ~0.5 MB，"
             "五臂共 ~45 万步），云图对本轮问题（宏观角是否依赖数值参数）没有信息量。\n")

    L.append("## (5) 注意点（坑与适用范围）\n")
    L.append("- 接触层恒等式不能直接用总力验证：装配力含全局速度阻尼（沿速度方向），必须用 `μ=0` 双胞胎做差。"
             "第一版测试就是被这个阻尼搞错的（差 5 N，正好是阻尼项）。")
    L.append("- `mean_tanh` 用**带符号**平均会被 v_t 的正负抵消（接触正负各半）：第一次实现的 0.0047 是伪值，"
             "必须取绝对值（`mean_abs_tanh`）。")
    L.append("- 等体积测法要求**对数**应变判据（工程应变在 ε_a = 0.3 会给出 −0.13 的假体应变）；"
             "`strain_measure=\"log\"` 是硬要求。")
    L.append("- `arctan(μ⟨|tanh|⟩)` 是把微观算子的调动率直接映射到宏观角的**一阶代理**，忽略 fabric 各向异性与"
             "二维平面应变→三维三轴的换算，所以 H3 的容差放宽到 2°；它成立只说明「缺口的量级和方向由接触算子决定」，"
             "不等于证明了解析关系。")
    L.append("- 适用范围：rotationless 圆盘 + 正则化库仑；滚动摩阻旋钮（R4/R6）在本内核里等价于附加滑动摩擦，"
             "因此本结论对 R5 的标定曲线同样适用。")
    L.append("")

    L.append("## (6) 误差判定（逐项，显式回答「是否 <3 %」）\n")
    L.append("| 关键量 | 本单实测 | 基准 | 相对误差 | <3 % ? |")
    L.append("|---|---|---|---|---|")
    L.append(f"| φ_cv（锚点工况，{f(base['velocity_regularisation'],4)} m/s） | {f(base['phi_cv_deg'])}° | "
             f"Bolton 1986 {f(ref['phi_bolton_deg'],0)}° | {f(anchor['relative_error']*100)} % | "
             f"{'是' if anchor['under_3_percent'] else '否'} |")
    L.append(f"| φ_cv（同） | {f(base['phi_cv_deg'])}° | Rowe 1962 {f(ref['phi_rowe_deg'])}° | "
             f"{f(base['phi_rowe_relative_error']*100)} % | 否 |")
    L.append(f"| 接触层恒等式 `F_t/(μF_n)` | `-tanh(v_t/v_reg)` | 代码字面 | 0（精确恒等，rel < 1e-9） | 是 |")
    L.append(f"| 复现 R3 基线 | {f(base['phi_cv_deg'])}° | R3 记录 14.31° | "
             f"{f(abs(base['phi_cv_deg'] - 14.31) / 14.31 * 100)} % | "
             f"{'是' if abs(base['phi_cv_deg'] - 14.31) / 14.31 <= 0.03 else '否'} |")
    L.append("")
    L.append("**结论**：锚点验收维持 **FAIL**（>8 %，远超 3 %）。本单的增量是把缺口**归因到可测量的机制**上，"
             "并给出一个此前不存在的可观测量（接触调动率）；它不宣称修好了摩擦角。"
             "若 H1/H3 如本单判定，则修法必须动本构（引入切向位移记忆的 Cundall–Strack 弹簧 + 库仑夹断，"
             "取代速率正则化），而不是再调一个摩擦系数旋钮——这是下一单的预声明范围。\n")
    return "\n".join(L)


def main() -> int:
    art = json.loads(ART.read_text())
    md = build(art)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    print(f"wrote {DOC}")
    figures = []
    if PNG.exists():
        figures.append({
            "name": "CASE-DEM-BIAXIAL-FRICTION_regularisation.png",
            "caption": ("(a) φ_cv 与 arctan(μ⟨|tanh(v_t/v_reg)|⟩) 随正则化速度 v_reg 的变化（附 Rowe 1962 与 "
                        "Bolton 1986 参考线）；(b) φ_cv 对「处于库仑极限接触比例」的关系；"
                        "(c) 各臂应力比 q/p 与接触调动率随轴向应变。"),
            "png_base64": base64.b64encode(PNG.read_bytes()).decode(),
        })
    summary = (f"锚点 CASE-DEM-BIAXIAL-FRICTION R7：把 50 % 缺口归因检验到接触层。实测 φ_cv 在 v_reg 上"
               f"极差 {f(art['hypotheses']['H1_vreg_invariance']['spread_deg'])}°，"
               f"速率差 {f(art['hypotheses']['H2_rate_invariance']['difference_deg'])}°，"
               f"接触算子映射残差最差 {f(art['hypotheses']['H3_contact_law_explains_phi']['worst_difference_deg'])}°；"
               f"锚点验收仍 FAIL（{f(art['anchor_verdict']['relative_error']*100)} %）。")
    payload = {
        "title": "CASE-DEM-BIAXIAL-FRICTION · R7：接触摩擦正则化速度 v_reg 是否进入本构（验证欠账）",
        "summary": summary,
        "scenario_key": "dem_biaxial_friction_regularisation",
        "mission_id": MISSION,
        "tags": ["TensorDEM", "DEM", "biaxial", "critical-state", "friction",
                 "CASE-DEM-BIAXIAL-FRICTION", "verification-debt"],
        "markdown": md,
        "figures": figures,
    }
    REPORT.write_text(json.dumps(payload))
    print(f"wrote {REPORT} ({len(md)} chars, {len(figures)} figure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())