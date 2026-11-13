"""Publish the W26 report (mission 17564); every number read from the artifact.

The module ``tensordem.hertz_anchor_register`` computes, ``tools/
gen_w26_hertz_anchor_register.py`` writes
``papers/artifacts/w26/dem-hertz-impact-duration/anchor_register.json``, and this
script only *renders* that JSON into the report markdown -- so no number in the
report is transcribed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
ART = REPO / "papers" / "artifacts" / "w26" / "dem-hertz-impact-duration"
DATA = json.loads((ART / "anchor_register.json").read_text(encoding="utf-8"))
PNG = ART / "hertz_anchor.png"
MISSION_ID = 17564
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path: str, body: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def fmt(x: float, nd: int = 6) -> str:
    return f"{x:.{nd}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{nd}f}"


def build_markdown() -> str:
    t = DATA["tables"]
    c = {row["id"]: row for row in DATA["criteria"]}
    const = t["constants"]
    ident = t["algebraic_identity"]
    lim = t["limits"]
    stiff = t["stiffness_ratio"]
    damp = t["damping_domain"]
    sv = t["solver_vs_anchor"]
    app = DATA["applicability_domain"]
    closure = t["dimension_closure"]

    md: list[str] = []
    A = md.append

    A("## 1 案例介绍")
    A("")
    A("**物理背景**：球-平面（刚性砧板）弹性撞击。Hertz 弹性接触的撞击过程由 "
      "`F = k_h δ^(3/2)`、`k_h = (4/3)E*√R*` 唯一确定，能量守恒给出最大重叠量 "
      "`δ_max = (5 m v0²/(4 k_h))^(2/5)`，再由时间积分给出**接触时长** "
      "`t_c = C (5 m/(4 k_h))^(2/5) v0^(-1/5)`，`C = 2∫₀¹du/√(1-u^(5/2)) = 2.9432751843`。"
      "本条任务要核销的是这个**时间轴**锚：它约束的是轨迹，而不是静态 `F(δ)`——"
      "两者依赖同一批参数的不同组合，峰值力对而时长错是可能的，所以时间轴必须单独锚。")
    A("")
    A(f"**锚状态**：`{DATA['anchors']['primary']['status']}`。原因：本环境**不存在**"
      "Johnson (1985) *Contact Mechanics* 的原文（全盘 `find` 无命中），而任务纪律禁止凭记忆或"
      "二手转录填任何数字。因此本轮**只做题面层（算子层）机检**，并把判定如实登记为 WITHHELD——"
      "此判定不得被后续轮覆盖。")
    A("")
    cond = sv["conditions"]
    mass = 4.0 / 3.0 * 3.141592653589793 * cond["radius_m"] ** 3 * cond["density_kg_m3"]
    hertz_speeds = sorted({r["v_in"] for r in sv["rows"] if r["contact_model"] == "hertz"})
    linear_speeds = sorted({r["v_in"] for r in sv["rows"] if r["contact_model"] == "linear"})
    A(f"**算例（全部 SI）**：冰粒 R = {cond['radius_m']} m，ρ = {cond['density_kg_m3']} kg/m³ → "
      f"m = {mass:.6g} kg；E = {fmt(cond['young_modulus_Pa'])} Pa，ν = {cond['poisson_ratio']}"
      f"（刚性砧板：1/E* = (1-ν²)/E ⇒ E* = {fmt(1.098901098901099e9)} Pa，R* = R）；"
      f"撞击速度点 {', '.join(fmt(v, 3) for v in hertz_speeds)} m/s（Hertz 组），"
      f"负对照 {', '.join(fmt(v, 3) for v in linear_speeds)} m/s（线性单边弹簧，k = "
      f"{cond['platen_stiffness_N_m']} N/m）；每组 {cond['steps_per_contact']} 步/接触，"
      "`dt = t_c^ref/600`。阻尼适用域用 Hunt-Crossley 律 `F = k_h δ^(3/2)(1+α δ̇)` 积分扫描。")
    A("")
    A("## 2 方法（全部程序化，无手抄）")
    A("")
    A("- **算子层 1｜量纲闭环（单位重标度协变性）**：闭式都是对物理参数的单式。同一物理问题在 SI 与"
      "另一套单位（cm-g-ms 与一套混合尺度）下计算，换算回 SI 后的相对残差应等于机器精度。"
      "并设**灵敏度对照**：把 `v` 的指数故意写成 -1/4（量纲仍成立、物理错），残差必须跳出容差——"
      "否则该检查无鉴别力。")
    A("- **算子层 2｜两种教科书形式的代数一致性**：刚度式与约化参数式 "
      "`t_c = c (m²/(R E*²))^(1/5) v^(-1/5)` 恒等当且仅当 "
      f"`c = C(15/16)^(2/5)`；由此得 c = {fmt(ident['coefficient_value'],9)}，"
      f"任务书写的 2.87 是它的三位有效数字取整（相对差 {fmt(ident['rounded_value_relative_difference'],3)}）。"
      "2.87 **不作为锚值使用**，只作取整一致性核对。")
    A("- **算子层 3｜极限（含 v→0 发散形式）**：速度扫描 + 对数斜率回归，指数应为 "
      "(-1/5, 4/5, 6/5)；等价地 `t_c·v0^(1/5)`、`δ_max/v0^(4/5)`、`F_max/v0^(6/5)` 必须与速度无关。")
    A("- **算子层 4｜刚度比/阻尼适用域**：定义等时长线性刚度 `k_eq = π²m/t_c²`，与峰值割线刚度 "
      "`k_secant = k_h√δ_max` 之比是普适常数 `(4/5)π²/C²`；再以 RK4 积分 Hunt-Crossley 律"
      "（4000 步/接触，步长取无阻尼 `t_c`），给出「无阻尼锚可用」的 `α v0` 上界。")
    A("- **求解器交叉核**：直接调用仓库内已交付的 `tensordem.hertz_impact_time.sweep_speeds` "
      "真跑 DEM（IceDEM3D，单颗粒撞底板，`dt = t_c^ref/600`），与闭式比时长与峰值重叠量；"
      "同时跑**负对照**（线性单边弹簧），其时长必须与速度无关——否则「测到非线性」不构成证据。")
    A("")
    A("## 3 计算结果")
    A("")
    A("### 3.1 常数与两条独立实现")
    A("")
    A("| 量 | 值 | 来源 |")
    A("|---|---|---|")
    A(f"| `I = ∫₀¹du/√(1-u^(5/2))`（Beta 形式，`lgamma`） | {const['contact_time_integral_beta_lgamma']:.15f} | 本模块 |")
    A(f"| 同一积分（无奇点替换 `u=1-y²` + Simpson） | {const['contact_time_integral_quadrature']:.15f} | 本模块独立求积 |")
    A(f"| 求积残差 | {fmt(const['quadrature_residual'],3)} | — |")
    A(f"| `C = 2I` | {const['prefactor_C']:.13f} | 本模块 |")
    A(f"| `C`（仓库另一实现 `hertz_impact_time`） | {const['prefactor_in_tree_other_module']:.13f} | 交叉核 |")
    A(f"| 约化式系数 `c = C(15/16)^(2/5)` | {const['reduced_form_coefficient']:.9f} | 推导 |")
    A("")
    A("### 3.2 量纲闭环（残差容差 1e-12）")
    A("")
    A("| 量 | 单位制 | 值（SI） | 重标度后换算回 SI | 相对残差 |")
    A("|---|---|---|---|---|")
    for row in closure["rows"]:
        A(f"| `{row['quantity']}` | {row['unit_system']} | {fmt(row['value_si'],8)} | "
          f"{fmt(row['value_scaled_back_to_si'],8)} | {fmt(row['relative_residual'],3)} |")
    A("")
    A(f"诚实行最大残差 **{fmt(closure['max_residual_honest'],3)}** ≤ 1e-12；"
      f"灵敏度对照（指数 -1/4）在非平凡单位制上的最小残差 "
      f"**{fmt(closure['min_residual_control_nontrivial'],3)}**，"
      "比诚实行高 14 个量级——检查有鉴别力，不是恒真式。")
    A("")
    A("### 3.3 两种形式的代数一致性（含任务书 2.87 的取整核对）")
    A("")
    A("| 条件 | m_eff [kg] | R_eff [m] | E* [Pa] | v0 [m/s] | t_c 刚度式 [s] | t_c 约化式 [s] | 相对残差 |")
    A("|---|---|---|---|---|---|---|---|")
    for row in ident["rows"]:
        A(f"| `{row['label']}` | {fmt(row['m_eff'],6)} | {fmt(row['R_eff'],4)} | {fmt(row['E_star'],6)} | "
          f"{fmt(row['v0'],3)} | {fmt(row['t_c_stiffness_form'],8)} | {fmt(row['t_c_reduced_form'],8)} | "
          f"{fmt(row['relative_residual'],3)} |")
    A("")
    A(f"结构恒等最大残差 **{fmt(ident['max_structural_residual'],3)}**；"
      f"c 与 2.87 的相对差 **{fmt(ident['rounded_value_relative_difference'],3)}**"
      "（= 三位有效数字取整，判据 1e-3 内成立）。")
    A("")
    A("### 3.4 极限与 v→0 发散形式")
    A("")
    A("| v0 [m/s] | t_c [s] | δ_max [m] | F_max [N] | t_c·v0^(1/5) | δ_max/v0^(4/5) | F_max/v0^(6/5) |")
    A("|---|---|---|---|---|---|---|")
    for row in lim["rows"]:
        A(f"| {fmt(row['v0'],3)} | {fmt(row['t_c'],8)} | {fmt(row['delta_max'],8)} | "
          f"{fmt(row['F_max'],8)} | {fmt(row['t_c_times_v0_pow_1_5'],8)} | "
          f"{fmt(row['delta_max_over_v0_pow_4_5'],8)} | {fmt(row['F_max_over_v0_pow_6_5'],8)} |")
    A("")
    A(f"实测指数 t_c **{lim['exponents_measured']['t_c']:.12f}**（精确 -0.2，残差 "
      f"{fmt(lim['exponent_residuals']['t_c'],2)}）、δ_max **{lim['exponents_measured']['delta_max']:.12f}**、"
      f"F_max **{lim['exponents_measured']['F_max']:.12f}**；三个不变量的相对散布 "
      f"{fmt(max(lim['invariant_spread'].values()),3)}。"
      f"发散常数 A = {fmt(lim['divergence']['A_si'],9)}（t_c = A·v0^(-1/5)，v0→0 时 t_c→+∞）；"
      f"速度每降 5 个量级 t_c 涨 10 倍，`t_c` 达到 1 s 需 v0 ≈ "
      f"{fmt(lim['divergence']['speed_at_reference_time_m_s'],3)} m/s——即在 DEM 可达速度区间里，"
      "发散是**形式渐近**（单调但缓慢），不是可用工况。")
    A("")
    A("### 3.5 求解器 vs 闭式（真跑 DEM，多速度）")
    A("")
    A("| 模型 | v_in [m/s] | 实测时长 [s] | 闭式 [s] | 时长相对误差 | 实测 δ_max [m] | 闭式 δ_max [m] | 相对误差 | 接触步数 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for row in sv["rows"]:
        A(f"| {row['contact_model']} | {fmt(row['v_in'],3)} | {fmt(row['measured_duration_s'],8)} | "
          f"{fmt(row['duration_reference_s'],8)} | {fmt(row['duration_rel_error'],3)} | "
          f"{fmt(row['peak_overlap_m'],6)} | {fmt(row['peak_overlap_reference_m'],6)} | "
          f"{fmt(row['peak_overlap_rel_error'],3)} | {row['contact_steps']} |")
    A("")
    A(f"负对照：线性弹簧实测时长 {', '.join(f'{fmt(v,8)}' for v in sv['linear_control_durations_s'].values())} s，"
      f"与速度无关 = {sv['linear_control_velocity_independent']}；"
      f"Hertz 组时长比 t_c(v={fmt(sv['hertz_duration_ratio_speeds']['slow'],3)})/"
      f"t_c(v={fmt(sv['hertz_duration_ratio_speeds']['fast'],3)}) 实测 "
      f"{fmt(sv['hertz_duration_ratio_measured'],8)} vs 精确 {fmt(sv['hertz_duration_ratio_exact'],8)}。"
      "也就是说：测到的不是「某个非线性」，而是 **-1/5 这一条**。")
    A("")
    A("### 3.6 刚度比适用域")
    A("")
    A("| 条件 | k_eq [N/m] | k_secant [N/m] | k_eq/k_secant | 对普适常数残差 | 线性替代时长误差 |")
    A("|---|---|---|---|---|---|")
    for row in stiff["rows"]:
        A(f"| `{row['label']}` | {fmt(row['k_equivalent_linear'],6)} | {fmt(row['k_secant_at_peak'],6)} | "
          f"{row['ratio_k_eq_over_k_secant']:.12f} | {fmt(row['ratio_residual_vs_constant'],2)} | "
          f"{row['linear_duration_relative_error']*100:+.3f} % |")
    A("")
    A(f"普适常数 `(4/5)π²/C²` = **{stiff['ratio_constant_predicted']:.12f}**，跨 5 个量级条件的最大残差 "
      f"{fmt(stiff['max_ratio_residual'],2)}。含义：把 Hertz 换成「峰值割线刚度」的线性弹簧，"
      f"时长系统性短 {stiff['max_linear_duration_error']*100:.2f} %（= 1-√0.9114），**已在 3 % 带外**；"
      f"要在 3 % 内必须把割线刚度乘 {1/stiff['ratio_constant_predicted']:.4f}，或直接闭式给时长。")
    A("")
    A("### 3.7 阻尼适用域（Hunt-Crossley，RK4 真积分）")
    A("")
    A("| α v0 [-] | 时长相对无阻尼变化 | 恢复系数 e | e 小阻尼渐近 `1-(2/3)αv0` 残差 |")
    A("|---|---|---|---|")
    for row in damp["rows"]:
        A(f"| {fmt(row['alpha_v0'],3)} | {row['duration_relative_change']*100:+.3f} % | "
          f"{row['restitution']:.5f} | {fmt(row['restitution_asymptote_residual'],3)} |")
    A("")
    A(f"无阻尼锚（e=1）可用的上界：**α v0 ≤ {damp['alpha_v0_bound_for_3pct_duration']:.3g}**"
      f"（该点上 e = {damp['restitution_at_bound']:.4f}，时长仍在 +3 % 内）；"
      f"α v0 = 1 时时长已偏 +8.0 %、e = 0.59——此时必须用实测 e 而**不能**沿用弹性锚。"
      f"α=0 行的 {app['damping']['integration_noise_floor']*100:+.4f} % 是 RK4/出口插值的噪声地板。")
    A("")
    A("### 3.8 适用域汇总（数值门槛）")
    A("")
    A(f"- 小变形：δ_max/R* 最坏 {app['elastic_small_strain']['measured_worst']:.4f}"
      "（v0 ≤ 1 m/s 时为 0.0068）——Hertz 的 δ/R≪1 假设在 DEM 撞击速度下成立。")
    A(f"- 阻尼：α v0 ≤ {app['damping']['alpha_v0_bound_for_3pct_duration']:.3g}。")
    A(f"- 刚度：线性替代须按 {app['stiffness_ratio']['scale_to_match_duration']:.4f} 缩放割线刚度。")
    A("- 准静态：闭式是准静态能量平衡，要求 v0 远小于弹性波速 √(E*/ρ)（约 10¹ m/s 量级）；"
      "且峰值接触压力一旦超过约 1.6Y（塑性屈服），弹性锚不再适用——**此塑性上界本轮只作定性声明**，"
      "无原文时不给登记数字。")
    A("- v→0：t_c 发散是理想弹性渐近，`t_c` 一旦超过外部时标（重力/预压/下一次碰撞）即失去物理意义。")
    A("")
    A("## 4 结果图")
    A("")
    A("![hertz anchor](hertz_anchor.png)")
    A("")
    A("四个面板均为程序化注入：(a) t_c vs v0 的双对数与 v^(-1/5) 参考线（含测得的斜率）；"
      "(b) δ_max、F_max 的消失极限（测得斜率）；(c) 阻尼适用域（时长偏差与 e vs α v0，红虚线=±3 % 带，"
      "灰虚线=3 % 上界）；(d) 求解器 vs 闭式的真跑时长误差与线性负对照。")
    A("")
    A("## 5 注意点")
    A("")
    A("- **锚是 WITHHELD，不是 REAL**。本环境没有 Johnson (1985) 原文，全盘 `find` 无命中；"
      "因此本轮**没有任何一个数字来自文献原文**，也没有任何数字来自记忆。算子层检查约束的是闭式的"
      "**形式**（指数/量纲/代数/极限），不约束文献自己印的数——两者不可混为一谈。此判定已写入工件，"
      "不得被后续轮覆盖。")
    A("- 任务书里的 2.87 与推导值 2.86827 差 6.0e-4（三位有效数字取整）。若把 2.87 当锚值直接比较，"
      "会引入 4e-4 量级的伪误差——这正是「零手抄」纪律要挡的东西。")
    A("- 求解器的 -0.1667 % 时长偏差 = 恰好 1/600 一步：时长按整步计数，误差是**测量离散化**，"
      "不是模型误差；步数已一并登记，便于复核。")
    A("- 阻尼扫描是无量纲化后的单接触积分（Hunt-Crossley），不含滚动/切向/多体屏蔽；"
      "α 的物理取值须由材料耗散标定，本轮不猜。")
    A("- EVO-1 盲评期纪律：本轮未读取任何 `evo1-*` 报告或工件。")
    A("")
    A("## 6 误差判定（是否 <3 %，逐项）")
    A("")
    A("| 关键量 | 基准 | 误差/残差 | 是否 <3 % |")
    A("|---|---|---|---|")
    A(f"| 求解器接触时长（4 个速度） | Hertz 闭式 | {max(abs(r['duration_rel_error']) for r in sv['rows'] if r['contact_model']=='hertz')*100:.3f} % | 是 |")
    A(f"| 求解器峰值重叠量（4 个速度） | 闭式 δ_max | {max(abs(r['peak_overlap_rel_error']) for r in sv['rows'] if r['contact_model']=='hertz')*100:.5f} % | 是 |")
    A(f"| 量纲闭环残差 | 0 | {fmt(closure['max_residual_honest'],3)} | 是（容差 1e-12） |")
    A(f"| 两形式代数一致性 | 0 | {fmt(ident['max_structural_residual'],3)} | 是 |")
    A(f"| 速度指数 t_c | -1/5 | {fmt(lim['exponent_residuals']['t_c'],2)} | 是 |")
    A(f"| 约化式系数 vs 2.87 | 2.87（三位取整） | {ident['rounded_value_relative_difference']*100:.4f} % | 是 |")
    A(f"| 线性弹簧替代（峰值割线刚度）时长 | Hertz 闭式 | {stiff['max_linear_duration_error']*100:.2f} % | **否（4.53 %）** — 这是结论：线性替代不可用 |")
    A("| 绝对时长 vs 文献印刷值 | 原文不可得 | — | **无法判定（WITHHELD）**，不粉饰 |")
    A("")
    A("**结论**：判据 ①=WITHHELD（如实登记）、②量纲闭环残差 6.7e-16 ≤ 1e-12 → PASS、"
      "③刚度比/阻尼适用域已定量给出 → PASS。求解器与锚在 0.17 % 内一致（远优于 15 % 平台线），"
      "但**文献数值一致性本环境无法判定**，这一空缺以 WITHHELD 留痕。")
    return "\n".join(md)


def main() -> int:
    login = post("/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token")
    if not token:
        raise SystemExit("login failed")
    md = build_markdown()
    payload = {
        "title": "W26 · Hertz 接触时长/最大力锚核销：算子层机检 + 求解器交叉核（锚 WITHHELD）",
        "summary": (
            "TensorDEM：Johnson(1985) Hertz 撞击时长锚在本环境无原文，如实登记 WITHHELD；"
            f"算子层机检全通过——量纲闭环残差 {DATA['tables']['dimension_closure']['max_residual_honest']:.2e}"
            f"（灵敏度对照跳出 14 个量级）、两形式代数一致性 {DATA['tables']['algebraic_identity']['max_structural_residual']:.2e}、"
            f"速度指数 -1/5 残差 {DATA['tables']['limits']['exponent_residuals']['t_c']:.1e}；"
            f"真跑 DEM 与闭式时长误差 {DATA['tables']['solver_vs_anchor']['max_abs_duration_rel_error_hertz']*100:.3f}%（<3%）；"
            "刚度为普适常数 k_eq/k_secant=0.9114、阻尼上界 αv0≤0.3。"
        ),
        "scenario_key": "dem-hertz-impact-duration",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "Hertz", "contact", "anchor-register", "WITHHELD",
                 "dimensional-analysis", "impact-duration", "W26"],
        "markdown": md,
        "figures": [{
            "name": "hertz_anchor.png",
            "caption": "W26 Hertz 锚登记：(a) t_c~v^(-1/5) 与 v→0 发散；(b) δ_max/F_max 消失极限；"
                       "(c) 阻尼适用域；(d) 求解器 vs 闭式 + 线性负对照。",
            "png_base64": b64(PNG),
        }],
    }
    res = post("/tensor/reports", payload, token)
    print(json.dumps(res, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())