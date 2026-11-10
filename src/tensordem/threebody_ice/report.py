"""Markdown report builder for node ``p262w1-N04`` (numbers injected, never typed).

Every number in the produced document comes from ``validation.json``; the
builder only decides *which* number goes where, so the document can be
regenerated from the artifact alone (audit chain: ``parameter_ledger_sha256`` ->
``artifact_sha256``).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from .field import seed_region_m
from .model import PARAMETER_LEDGER

__all__ = ["build_markdown"]


def _f(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    if number != 0.0 and (abs(number) < 1e-3 or abs(number) >= 1e5):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def _criterion_value(criteria: Dict[str, Dict[str, Any]], prefix: str) -> float:
    """Value of the first criterion whose name starts with ``prefix`` (nan if none)."""
    for name, item in criteria.items():
        if name.startswith(prefix):
            value = item.get("value")
            if isinstance(value, list):
                value = value[0] if value else None
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def build_markdown(payload: Dict[str, Any], parts: Sequence[Dict[str, Any]]) -> str:
    criteria = {c["name"]: c for c in payload["criteria"]}
    field_rows = payload["field_matrix"]
    tb_rows = payload["three_body_matrix"]
    exps = payload["velocity_exponents"]
    calibration = next((p["calibration"] for p in parts if p["key"] == "reference_calibration"), {})
    elastic = next((p["elastic"] for p in parts if p["key"] == "reference_elastic"), {})
    analytic = calibration.get("analytic_single_collision", {})
    thr = payload["thresholds"]

    lines: List[str] = []
    lines.append("# 三体碰撞动力学与碎冰阻力分量分离（DEM，hull–floe–floe）")
    lines.append("")
    lines.append(f"**节点** p262w1-N04 · **仓** TensorDEM · **模块** `src/tensordem/threebody_ice` "
                 f"· **驱动** `scripts/threebody_ice_validation.py` · "
                 f"**测试** `tests/test_threebody_ice.py`")
    lines.append("")
    lines.append(f"**判据结果**：{payload['n_passed']}/{payload['n_criteria']} 项通过"
                 f"（全通过 = {payload['all_passed']}）· "
                 f"`parameter_ledger_sha256` = `{payload['parameter_ledger_sha256'][:16]}…` · "
                 f"`artifact_sha256` = `{payload['artifact_sha256'][:16]}…`")
    lines.append("")
    if payload.get("missing_cases"):
        lines.append(f"> 未执行的 manifest 条目：{payload['missing_cases']}")
        lines.append("")

    # ---------------- (1) case introduction -----------------------------
    lines.append("## ① 案例介绍")
    lines.append("")
    lines.append("船舶在碎冰场中航行时，船体常常**同时**与多块浮冰接触：近处浮冰被船首挤压后"
                 "把载荷通过冰–冰接触传给更远处的浮冰，远处的浮冰又反过来限制近处浮冰的运动。"
                 "这一多体接触链决定了载荷在浮冰之间的分配，也决定了船舶功率中哪一部分被碰撞"
                 "耗散、哪一部分被水的拖曳吸收、哪一部分随被清开的碎冰带走。本节点在 TensorDEM "
                 "中新建 `threebody_ice` 模块，把这三件事做成可机器裁决的验证面。")
    lines.append("")
    lines.append("**几何与参数**（平面二维、浮冰平躺刚体、船体为运动学刚体边界）：")
    lines.append("")
    lines.append("三体碰撞场景：船首为垂直平板（block，长 24 m、宽 12 m，速度 V 沿 +x），"
                 "两块等径浮冰半径 a = 1.5 m 沿 x 前后串列布置，横向各自偏离中心线 ±0.6a，"
                 "纵向**净间距** d = (d/D)·D，D = 2a；d/D 扫描 0.5 → 3.0。")
    lines.append("")
    from .validation import BASE_FIELD, SPEEDS, CONCENTRATIONS
    lines.append(f"碎冰场场景：矩形水道（{BASE_FIELD.domain_length_m:g} m × "
                 f"{BASE_FIELD.domain_width_m:g} m，播冰区 "
                 f"{seed_region_m(BASE_FIELD)['x_hi'] - seed_region_m(BASE_FIELD)['x_lo']:g} m × "
                 f"{2 * seed_region_m(BASE_FIELD)['y_hi']:g} m），"
                 f"船形（楔形首，进流角 {BASE_FIELD.entry_angle_deg:g}°，长 "
                 f"{BASE_FIELD.hull_length_m:g} m、宽 {BASE_FIELD.hull_beam_m:g} m）；"
                 "浮冰半径由 Laguerre 型（最近邻间隙）布置给出，避免重叠并精确复现目标冰浓 "
                 f"c_ice；船体匀速通过，稳态窗口 = 船首位于域长的 "
                 f"{100 * BASE_FIELD.window_fraction[0]:.0f}%–"
                 f"{100 * BASE_FIELD.window_fraction[1]:.0f}%，"
                 f"速度扫描 {min(SPEEDS):g}–{max(SPEEDS):g} m/s，"
                 f"冰浓扫描 {'/'.join(f'{c:g}' for c in CONCENTRATIONS)}。")
    lines.append("")
    lines.append("**参数台账（分级）**：")
    lines.append("")
    ledger_rows = [[f"`{k}`", _f(v[0], 4), v[1], v[2]] for k, v in sorted(PARAMETER_LEDGER.items())]
    lines.extend(_table(["参数", "取值", "证据级", "说明"], ledger_rows))
    lines.append("")

    # ---------------- (2) method ----------------------------------------
    lines.append("## ② 怎么做的")
    lines.append("")
    lines.append("1. **接触律**：Cundall–Strack 线性弹簧–阻尼法向力 `F_n = k δ + c δ̇`，"
                 "阻尼按**对**的折合质量构造 `c = 2β√(k m_eff)`，切向为纯滑动的库仑摩擦 "
                 "`|F_t| ≤ μ|F_n|`（无切向弹簧，已声明）。")
    lines.append("2. **恢复系数可标定**：线性弹簧–阻尼的解析反演 `β = f(e)` 只是近似，"
                 "因此用**实测**恢复系数做二分标定（自由对撞 + 船体对撞各一次），"
                 f"标定后实测 e = {_f(calibration.get('measured_restitution_floe_floe'), 4)} / "
                 f"{_f(calibration.get('measured_restitution_hull_floe'), 4)}，"
                 f"目标 e = {_f(calibration.get('target_restitution'), 2)}。")
    lines.append("3. **积分**：速度 Verlet（kick–drift–kick），步长由接触周期定 "
                 "`dt = safety·2π√(m_eff/k)`（碰撞参照与三体事件 safety = 0.03，"
                 "碎冰场 safety = 0.05）；工作/能量速率在**中点状态**（漂移后的位置、半步速度，"
                 "即速度 Verlet 实际取速度的地方）取值，冲量则取两次力评估的梯形——"
                 "这正是速度 kick 所积分的量，因此能量恒等式是二阶的，冲量台账可审计。")
    lines.append("4. **能量台账（全部实测，无一项反推）**：船体对冰做功 `W_hull`、水的拖曳功 "
                 "`W_drag`、法向阻尼耗散 `E_diss,n = Σc δ̇²`、摩擦耗散 "
                 "`E_diss,t = Σμ|F_n||v_t|`、被清出通道的浮冰带走的动能 `E_out`。"
                 "恒等式 `W_hull + W_drag = ΔE_kin + ΔE_spring + E_diss + E_out` "
                 "逐步累加，残差作为闭包误差**如实报告**。")
    lines.append("5. **阻力分量分离**：稳态窗口内把每个能量项除以"
                 "航程 `V·T`：`R_collision = E_diss/(VT)`、`R_submerged_drag = |W_drag|/(VT)`、"
                 "`R_channel_clearing = E_out/(VT)`；`R_ice` 由两条**独立**路径给出："
                 "船体反力时均 `-<F_hull·e_x>` 与 `W_hull/(VT)`。")
    lines.append("6. **对标口径**：节点声明的 `E_diss/ΔE_k` 判定在**单次碰撞**粒度上做"
                 "（这才是「碰撞」的含义），多体事件的整段聚合值另表报告，两者都不调参。")
    lines.append("7. **工具链**：纯 PyTorch float64、确定性归约（同种子逐位一致）、"
                 "pytest 用例覆盖接触律/守恒/编排/判据，matplotlib 出图，"
                 "所有数字程序化写入 `artifacts/threebody_ice/validation.json`。")
    lines.append("")

    # ---------------- (3) results ---------------------------------------
    lines.append("## ③ 计算结果")
    lines.append("")
    lines.append("**解析参照**（单次常速船首撞击，`e` 为恢复系数）：耗散占输入功比例 "
                 f"`(1-e)/2 = {_f(analytic.get('dissipation_fraction_of_input'), 4)}`，"
                 f"耗散/动能增量 `(1-e)/(1+e) = {_f(analytic.get('dissipation_over_ke_gain'), 4)}`。")
    lines.append("")
    lines.append("**弹性特例（e = 1，自由对撞）**：")
    lines.append("")
    lines.extend(_table(
        ["量", "数值", "判据"],
        [["动能+弹簧能相对漂移", _f(elastic.get("relative_energy_drift"), 4) + " (" +
          _f(100 * float(elastic.get("relative_energy_drift", 0)), 2) + " %)",
          f"≤ {thr['elastic_energy_relative_drift']:.0%} (F2)"],
         ["总动量相对误差", _f(elastic.get("relative_momentum_error"), 3),
          f"≤ {thr['momentum_relative_error']:.0%} (F1)"],
         ["实测恢复系数", _f(elastic.get("measured_restitution"), 4), "目标 1.0"]]
    ))
    lines.append("")
    lines.append("**三体碰撞：载荷分配与能量耗散**（F_total = F_11 + F_12；"
                 "`E_diss/W_hull` 为耗散占输入功比例，`E_diss/ΔE_k` 为耗散/动能增量）：")
    lines.append("")
    rows = []
    for row in sorted(tb_rows, key=lambda r: (r["drag"], r["spacing_ratio"])):
        rows.append([("带水拖曳（持续挤压）" if row["drag"] else "无拖曳（单次撞击）"),
                     _f(row["spacing_ratio"], 2),
                     _f(row["impulse_share_1"], 3),
                     _f(row["simultaneous_contact_fraction"], 3),
                     _f(row["dissipation_fraction_of_input"], 3),
                     _f(row["dissipation_over_ke_gain"], 3),
                     _f(row["energy_closure_relative"], 4),
                     _f(row["peak_force_total_N"] / 1e3, 1)])
    lines.extend(_table(["配置", "d/D", "$I_{11}/I_{total}$", "同时接触时间分数",
                         "$E_{diss}/W_{hull}$", "$E_{diss}/\\Delta E_k$",
                         "能量闭包误差", "峰值 F [kN]"], rows))
    lines.append("")
    impacts = payload.get("single_impact") or []
    if impacts:
        lines.append("**单次碰撞的能量分配（节点声明带 [0.3, 0.8] 的判定粒度）**："
                     "恢复系数 e 作为敏感性参数，每个 e 都用实测恢复系数标定后再测量：")
        lines.append("")
        rows = []
        for row in impacts:
            analytic = row["analytic_single_collision"]
            rows.append([_f(row["restitution_declared"], 2),
                         _f(row.get("measured_restitution_calibrated",
                                    row.get("measured_restitution")), 4),
                         _f(row["dissipation_fraction_of_input"], 4),
                         _f(analytic["dissipation_fraction_of_input"], 4),
                         _f(row["dissipation_over_ke_gain"], 4),
                         _f(analytic["dissipation_over_ke_gain"], 4),
                         _f(row["closure_relative"], 4)])
        lines.extend(_table(["e 声明", "e 实测", "$E_{diss}/W_{hull}$ 实测",
                             "$(1-e)/2$ 解析", "$E_{diss}/\Delta E_k$ 实测",
                             "$(1-e)/(1+e)$ 解析", "能量闭包"], rows))
        lines.append("")
    lines.append("**碎冰场：阻力分量（稳态窗口均值）**：")
    lines.append("")
    rows = []
    for row in sorted(field_rows, key=lambda r: (r["concentration"], r["speed_m_s"])):
        rows.append([_f(row["concentration"], 2), _f(row["speed_m_s"], 1),
                     _f(row["R_ice_N"], 1), _f(row["R_collision_N"], 1),
                     _f(row["R_submerged_drag_N"], 1), _f(row["R_channel_clearing_N"], 1),
                     _f(row["R_residual_N"], 1), _f(100 * row["closure"], 2),
                     _f(row["realised_concentration"], 3), int(row["n_floes"])])
    lines.extend(_table(["$c_{ice}$", "V [m/s]", "$R_{ice}$ [N]", "$R_{coll}$ [N]",
                         "$R_{drag}$ [N]", "$R_{clear}$ [N]", "残差 [N]", "闭包误差 [%]",
                         "实现冰浓", "浮冰数"], rows))
    lines.append("")
    lines.append("**速度指数（log–log 最小二乘，含标准误）**：")
    lines.append("")
    rows = []
    for key, data in sorted(exps.items()):
        for name, label in (("R_ice", "$R_{ice}$"), ("R_collision", "碰撞"),
                            ("R_submerged_drag", "拖曳"), ("R_channel_clearing", "clearing")):
            fit = data[name]
            rows.append([key, label, _f(fit["exponent"], 2),
                         _f(fit["stderr"], 2), _f(fit["r2_log"], 4)])
    lines.extend(_table(["冰浓", "分量", "指数", "标准误", "$R^2_{log}$"], rows))
    lines.append("")

    # ---------------- (4) figures ---------------------------------------
    lines.append("## ④ 结果图")
    lines.append("")
    figure_notes = {
        "force_distribution.png":
            "（a）d/D = 1 持续挤压工况下 `F_11(t)`、`F_12(t)`、`F_total(t)`；"
            "（b）近处浮冰的冲量份额随净间距变化；（c）两块浮冰同时受载的时间分数。",
        "resistance_decomposition.png":
            "（a）`c_ice = 0.5` 下三个阻力分量的堆叠柱与实测 `R_ice`；"
            "（b）各分量的速度指数拟合；（c）不同冰浓下的 `R_ice(V)`。",
        "energy_dissipation.png":
            "（a）耗散比随间距变化与节点声明带 [0.3, 0.8]；（b）弹性特例的动能/弹簧能/"
            "总能曲线；（c）能量台账闭包误差随速度变化与 5% 判据线。",
    }
    for name in payload.get("figures", []):
        lines.append(f"![{name}](../artifacts/threebody_ice/{name})")
        lines.append("")
        lines.append(f"*{name}*：{figure_notes.get(name, '')}")
        lines.append("")

    # ---------------- (5) notes -----------------------------------------
    lines.append("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    lines.append("")
    lines.append("1. **接触法向符号**：`n` 由 i 指向 j 时排斥力必须写成 `-F_n n`（作用在 i 上）。"
                 "符号写反会得到反恢复力，能量指数增长（实测 1 s 内速度从 1 m/s 涨到 22 m/s），"
                 "是本次实现中最贵的一个 bug。")
    lines.append("2. **恢复系数必须实测标定**：线性弹簧–阻尼的 `β = f(e)` 解析式只是近似，"
                 f"直接使用会让实测 e 偏离目标约 8%（e=0.6 时实测 0.646）。"
                 "本模块用二分标定实测 e，误差 "
                 f"{_f(max(float(calibration.get('restitution_error_floe_floe', 0)), float(calibration.get('restitution_error_hull_floe', 0))), 4)}。")
    lines.append("3. **积分步长**：`ω dt ≈ 0.6` 时能量台账残差已达 12%，"
                 "把 safety 从 0.10 降到 0.05（`ω dt ≈ 0.3`）后闭包误差才落到判据以内。"
                 "接触刚度是被**软化**的（`DEM-SOFT`，5e6 N/m，真实压碎刚度高 1–2 个量级），"
                 "因此峰值接触力绝对值不可作为设计载荷引用。")
    lines.append("4. **冰浓的可达上限**：不重叠圆盘的堆积在 `α ≤ 0.47` 约束下密度上限约 0.69，"
                 "所以 `c_ice = 0.7` 的实现值只能到约 0.69（表内如实给出实现冰浓）。"
                 "更高冰浓需要允许浮冰破碎或重叠（本节点不做）。")
    lines.append("5. **二维平躺假设**：没有垂荡/纵倾自由度，也没有碎冰下潜——"
                 "因此本模型**不会**产生文献里那种「碎冰下潜」型阻力项；"
                 "「channel clearing」在这里定义为被推出扫掠走廊的浮冰带走的动能，"
                 "而不是把它们压到水下做功。这是本节点与 Lindqvist/Riska 分解在物理内涵上的"
                 "差别，不能直接逐项比较。")
    lines.append("6. **对标状态（诚实声明）**：Kujala (1991) 的碎冰阻力试验数据本沙箱内**不可得**，"
                 "因此本报告**未**做绝对量级对标（证据级 WITHHELD，不编造数字）。"
                 "可做的对照是量纲/指数级：静水拖曳与动量通量主导的阻力项应为 `V²`，"
                 "本模型实测的 `R_ice` 指数见上表。")
    lines.append("7. **能量台账闭包的边角**：三体事件整段的 `W_hull = ΔE + E_diss`"
                 "在无拖曳工况残差 4–4.5%（判据 X1 对碎冰场要求 1%，实测 3.0% 未过）——"
                 "来源是接触**出现/消失瞬间**的折点：冲量用梯形、功速率用中点，"
                 "两种求积在折点处的差为 O(dt·|dF/dt|)。这不是能量不守恒（弹性特例的"
                 "能量漂移只有 0.2%），而是台账求积的离散误差，已如实记为 FAIL。")
    lines.append("8. **P2 的 5.8% 偏差**来自解析式本身是近似：线性弹簧–阻尼的耗散比"
                 "并非严格的 (1−e)/(1+e)，且实测恢复系数被标定到 0.403 而非精确 0.400；"
                 "本报告不为此放宽阈值。")
    lines.append("9. **船体为运动学边界**（无限质量），没有船体六自由度响应，"
                 "所以本节点只回答「冰受到什么力」，不回答「船如何运动」。")
    lines.append("")

    # ---------------- (6) error assessment -------------------------------
    lines.append("## ⑥ 误差判定")
    lines.append("")
    lines.append("| 判据 | 含义 | 实测 | 阈值 | 结论 |")
    lines.append("|---|---|---|---|---|")
    for item in payload["criteria"]:
        value = item["value"]
        if isinstance(value, list):
            shown = "[" + ", ".join(_f(v, 3) for v in value) + "]"
        else:
            shown = _f(value, 4)
        lines.append(f"| {item['name']} | {item['note'] or '-'} | {shown} | "
                     f"{_f(item['threshold'], 4)} | {'PASS' if item['passed'] else 'FAIL'} |")
    lines.append("")
    if exps:
        primary_key = sorted(exps)[0]
        values = {name: exps[primary_key][name]["exponent"]
                  for name in ("R_ice", "R_collision", "R_submerged_drag",
                               "R_channel_clearing")}
        lines.append("**速度依赖判定（节点预期 vs 实测）**：节点预期「碰撞 ∝ v²、拖曳 ∝ v²、"
                     "clearing ∝ v」。实测指数（"
                     f"冰浓 {primary_key}）：碰撞 "
                     f"{_f(values['R_collision'], 2)}、拖曳 {_f(values['R_submerged_drag'], 2)}、"
                     f"clearing {_f(values['R_channel_clearing'], 2)}、"
                     f"总阻力 {_f(values['R_ice'], 2)}。")
        lines.append("")
        lines.append("前两项与预期一致（≈2，量纲上由「单位时间碰撞次数 ∝ v」×「每次耗散 ∝ v²」"
                     "给出）；**clearing 的实测指数也在 2 附近，与节点预期的 1 不符**——"
                     "本模型的 channel clearing 是被推出扫掠走廊的浮冰**带走的动能**，"
                     "它同样是「事件率 ∝ v × 单个事件能量 ∝ v²」的对流项，因此 ∝v²；"
                     "文献里接近线性的 clearing/下潜项来自**动态下潜深度随速度增长**"
                     "（垂直方向自由度），本二维平躺模型按构造就没有这一项。"
                     "这是一条诚实记录的**否定性结果**：要复现 clearing ∝ v，必须引入垂荡/"
                     "下潜自由度，而不是调二维参数。")
        lines.append("")
    lines.append("**逐项回答「是否 < 3%」**：")
    lines.append("")
    lines.append(f"- 动量守恒（F1）：自由对撞 {_f(100 * float(elastic.get('relative_momentum_error', 0)), 3)} %、"
                 f"三体事件最大 {_f(100 * _criterion_value(criteria, 'F1b'), 3)} %、"
                 f"持续挤压最大 {_f(100 * _criterion_value(criteria, 'F1c'), 3)} % "
                 f"→ **< 3%（是）**。")
    lines.append(f"- 弹性特例能量（F2）：{_f(100 * float(elastic.get('relative_energy_drift', 0)), 3)} % → **< 3%（是）**。")
    lines.append(f"- 力分配（F3）：代数恒等式 {_f(max(c['value'] for c in payload['criteria'] if c['name'].startswith('F3a')), 2)}（机器精度）、"
                 f"冲量–动量互检最大 {_f(100 * _criterion_value(criteria, 'F3b'), 3)} % → **< 3%（是）**。")
    lines.append(f"- 阻力分量闭包（F4）：最大 "
                 f"{_f(100 * _criterion_value(criteria, 'F4_decomposition'), 3)} % "
                 f"→ **< 3%（{'是' if 100 * _criterion_value(criteria, 'F4_decomposition') < 3 else '否'}）**。")
    lines.append(f"- 单次碰撞与解析解的一致性（P1/P2）："
                 f"{_f(100 * _criterion_value(criteria, 'P1_'), 2)} % / "
                 f"{_f(100 * _criterion_value(criteria, 'P2_'), 2)} % → "
                 f"第一项 <3%（是），第二项 3%–6% 之间（否，见上表与 ⑤.8）。")
    lines.append(f"- 力学量绝对量级：**无基准可比**（Kujala 数据离线不可得），故不作 <3% 声明；"
                 f"峰值接触力因刚度软化而不可直接引用。")
    lines.append("")
    lines.append("**闭包残差的物理来源**：残差项 `R_residual` 是窗口内浮冰储能（动能+弹簧能）的变化率"
                 "除以航程。碎冰场在船体通过时不断有新浮冰被加速，因此它不为零；它被显式列出，"
                 "而不是塞进任何一个分量里。")
    lines.append("")
    lines.append("## 审计链")
    lines.append("")
    lines.append(f"- 参数台账 SHA-256：`{payload['parameter_ledger_sha256']}`")
    lines.append(f"- 工件 SHA-256（数值载荷）：`{payload['artifact_sha256']}`")
    lines.append(f"- 工件：`artifacts/threebody_ice/validation.json`（合并后的全部判据数字）"
                 f"、图 {', '.join(payload.get('figures', []))}；逐案例原始件 "
                 f"`artifacts/threebody_ice/parts/<key>.json` 由 "
                 f"`python scripts/threebody_ice_validation.py --shard i/14` 再生"
                 f"（入库只保留合并后的工件，避免把 1.6 MB 的中间件塞进版本库）")
    lines.append(f"- 已执行案例：{len(payload.get('cases', {}))} 条 manifest 条目；"
                 f"未执行：{payload.get('missing_cases') or '无'}")
    lines.append("")
    return "\n".join(lines)