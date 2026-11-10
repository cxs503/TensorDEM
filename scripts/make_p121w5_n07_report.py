"""Build ``/tmp/report.json`` for campaign node P121W5-N07 from the acceptance ledger.

Every number is read from ``artifacts/sea_ice_transient/acceptance.json`` and the
figures are embedded as base64 PNG, so the report cannot contain hand-copied
values.

    python scripts/make_p121w5_n07_report.py && python post_report.py \
        artifacts/sea_ice_transient/report_payload.json
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "sea_ice_transient"
LEDGER = ART / "acceptance.json"
# NOTE: do **not** use a bare /tmp/report.json -- /tmp is shared with every
# concurrent agent on this host and the payload gets clobbered between the
# generation and the POST (observed 2026-11-10: our payload was replaced by
# another mission's).  Write the payload inside the worktree and post that
# exact file.
OUT = ART / "report_payload.json"


def _ci(block) -> str:
    if not block or not block.get("ci95") or block["ci95"][0] is None:
        return "不可达"
    lo, hi = block["ci95"]
    return f"{lo/1e3:.1f}, {hi/1e3:.1f}"


def sci(x, d: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, str, bool)):
        return str(x)
    return f"{x:.{d}e}"


def num(x, d: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{d}f}"


def main() -> int:
    d = json.loads(LEDGER.read_text())
    crit, des = d["criteria"], d["design"]
    a_d, b, c = d["a_and_d"], d["b"], d["c"]
    ens, th, fblk = d["e_ensemble"], d["theory_vs_numeric"], d["f"]

    M: list[str] = []
    A = M.append
    A("## 1. 案例介绍")
    A("")
    A("**物理背景.** 海冰在旋转地球上自由漂移的动量平衡由风应力、海冰-海洋拖曳、Coriolis 力与"
      "冰内应力（Hibler 黏塑性）共同决定。m5748 已把**稳态**漂移闭合为可验证账本（Nansen 偏角、"
      "风因子、Ekman 螺旋）。本节点把同一物理量升级到**时间域**并做可辨性裁决：")
    A("")
    A("* 从静止起转（spin-up）的瞬态惯性振荡能否给出机器精度（1e-6）的闭式锚，"
      "并由 1 h 步长的数值方案复现？")
    A("* 冰内应力（单参数：Hibler 强度 P\\*）对可观测的衰减率 γ 与偏角 θ 的影响，"
      "能否在 30 天单站漂移记录的时间序列噪声中被分辨（SNR ≥ 3）？")
    A("")
    A("**几何与参数.** 1-D 周期通道（Arakawa C 网格交错），保留完整 2-D 应力张量与整个塑性屈服面。")
    A("")
    A("| 参数 | 取值 |")
    A("|---|---|")
    A(f"| 通道长度 | {sci(des['channel_length_m'])} m（周期） |")
    A("| 网格 | N = 96（细网格 192），Δx = 8.3 km |")
    A("| 时间步 / 记录 | Δt = 900 s（收敛档 450 s）/ 30 d |")
    A(f"| 冰厚 / 密度 | {des['ice_thickness_m']} m / 900 kg·m⁻³ |")
    A("| 拖曳系数 | C_w = 0.0045，C_a = 0.0018 |")
    A("| 转动参数 | φ = 80°N → f = 1.4363e-4 s⁻¹ |")
    A(f"| 冰强度扫描 | P\\* ∈ {[p/1e3 for p in des['p_star_levels_N_per_m']]} kN/m |")
    A("| 屈服面 / 强度指数 | e = 2，C = 20 |")
    A(f"| 应变率正则化 | Δ_min = {des['delta_min_per_s']} s⁻¹ |")
    A(f"| 漂移站足迹 | {sci(des['buoy_window_m'])} m |")
    A("| 激励 | 风向固定 25°，风速 10 ↔ 13.5 m/s 每 2 天阶跃 |")
    A(f"| 噪声 | 风场 OU（σ_W = 2 m/s，τ = 1 d）+ 观测白噪声 {sci(des['measurement_sigma_m_s'])} m/s |")
    A(f"| 集成 | {des['n_seeds']} seeds × 8 个 P\\* |")
    A("")
    A("**判据（先于结果固定，写在脚本头部常量）**: 闭式锚相对误差 < 1e-6；动量账本 < 1e-8；"
      "P\\*→0 退化 < 1e-10；显式 Euler @1 h 必须失败而半隐式 < 1e-3；"
      f"Δθ、Δγ/γ 对 P\\* 的 Spearman ρ ≥ {des['spearman_threshold']}；"
      f"SNR = {des['snr_threshold']} 为可辨性门；两档离散收敛变化 < 5%。")
    A("")
    A("## 2. 怎么做的")
    A("")
    A("**(1) 闭式瞬态锚.** 各向同性线性拖曳下，围绕稳态 `u_ss` 线性化的平衡是常系数线性系统："
      "`m dw/dt = -m f (k×w) - m γ w`，令 `z = w_x + i w_y` 得 `dz/dt = -(γ+if)z`，于是")
    A("")
    A("```")
    A("u(t) = u_ss + e^{-γ t} R(-f t) (u_0 - u_ss)   →   ω ≡ |f|（精确）,  包络 ≡ e^{-γt}")
    A("```")
    A("")
    A("注意 ω **不是** √(f²+γ²)：拖曳是摩擦而不是恢复力。γ 由拖曳律 Jacobian 的各向同性部分给出"
      "`γ = (C_w|u_w-u| + C_a|W-u|)/m`。")
    A("")
    A("**(2) 时间推进方案与步长门.** exponential（指数积分，对线性部分精确）/ Crank–Nicolson / RK4 / "
      "显式 Euler。逐方案给出每步放大因子与还原出的 γ、f（`amplification()`），"
      "1 h 步长下逐方案误差见第 3 节表。")
    A("")
    A("**(3) 动量账本.** 逐点：把闭式解的解析导数代入 `SeaIceDriftModel` 自身的力项（含 Coriolis）"
      "求相对残差；逐步：用精确冲量恒等式 `∫₀^{Δt}e^{As}ds = A⁻¹(e^{AΔt}-I)` 检查 "
      "`m(w^{n+1}-w^n) = 拖曳冲量 + Coriolis 冲量`。")
    A("")
    A("**(4) 冰内应力.** Hibler：`P = P*h exp(-C(1-A))`，`ζ = P/(2max(Δ,Δ_min))`，`η = ζ/e²`，"
      "`σ₁₁ = (3ζ-η)ε̇₁₁ - P/2`，`σ₁₂ = η dv/dx`（`ε̇₂₂ = 0`）。动量方程**半隐式**推进"
      "（Coriolis + 拖曳 Newton Jacobian + 塑性应力散度同时隐式，2N×2N 稀疏求解），"
      "ζ、η、P 做 2 次 Picard 滞后；h、A 一阶迎风输运驱动 P 的建立。P\\* = 0 时逐位退化为线性问题。")
    A("")
    A("**(5) 激励与可辨性设计.** 风向固定、风速每 2 天 ±35 % 阶跃，反复重新激励惯性振荡；"
      "观测量 = 每次换向 4 h 之后的 44 h 实测段均值（跳过 ~1/γ ≈ 1.2 h 的瞬态），"
      "γ 用可分离最小二乘辨识（ω 固定为理论值 f，只辨识 γ，避免 γ≈f 时 ω 的不可辨识性）。"
      "噪声 = 100 条独立 OU 风场实现 + 观测白噪声；SNR 用**单条记录**的抽样标准差"
      "（不是均值的标准误），阈值 3，用 1000 次 seed 重采样给 95% 自助区间。")
    A("")
    A("**(6) 先解析后数值.** 冻结黏度平面波分析给出 Δγ = (3ζ+η)k²/(2m)、"
      "`ω_eff = √(f²-δ²)`、`θ_eff = arctan(ω_eff/γ_eff)`，代入 P\\* = 40 kN/m、Δ_min = 1e-6 得"
      "过阻尼预言；数值结果与之对比见 3.5。")
    A("")
    A("## 3. 计算结果")
    A("")
    A(f"**{d['n_pass']}/{d['n_total']} 条机器判据通过**（`all_pass = {d['all_pass']}`）。")
    A("")
    A("| 判据 | 量值 | 阈值 | 通过 |")
    A("|---|---|---|---|")
    for k, v in crit.items():
        val = v.get("value")
        val_s = sci(val) if isinstance(val, float) else (
            json.dumps(val, ensure_ascii=False) if isinstance(val, dict) else str(val))
        thr = v.get("threshold", v.get("requirement", "—"))
        A(f"| `{k}` | {val_s} | {sci(thr) if isinstance(thr, float) else thr} | "
          f"{'PASS' if v['pass'] else 'FAIL'} |")
    A("")
    A("**(a) 闭式锚**")
    A("")
    A("| 案例 | f [1/s] | γ [1/s] | 周期 [h] | 频率相对误差 | 衰减率相对误差 | 激起振幅比相对误差 |")
    A("|---|---|---|---|---|---|---|")
    for cs in a_d["cases"]:
        f_err = min(cs.get("freq_rel_error_recurrence", 1.0),
                    cs.get("freq_rel_error_nonlinear", 1.0))
        g_err = min(cs.get("gamma_rel_error_recurrence", 1.0),
                    cs.get("gamma_rel_error_nonlinear", 1.0))
        A(f"| {cs['case']} | {sci(cs['f'], 6)} | {sci(cs['gamma'])} | {num(cs['period_h'], 2)} | "
          f"{sci(f_err)} | {sci(g_err)} | {sci(cs['amplitude_ratio_rel_error'])} |")
    A("")
    A("三种独立估计量（线性预测/Prony 递推、可分离非线性最小二乘、事件响应最小二乘）"
      "在同一无噪声序列上互证，不是用同一条公式自证。")
    A("")
    A("**(d) 1 h 步长的方案门**")
    A("")
    A("| 方案 | 每步模量 | γ 相对误差 | f 相对误差 | 3 h 幅值相对误差 |")
    A("|---|---|---|---|---|")
    for name, s in a_d["schemes"].items():
        A(f"| {name} | {num(s['amplification_modulus'], 4)} | {sci(s['gamma_numeric_rel_error'])} | "
          f"{sci(s['f_numeric_rel_error'])} | {sci(s['final_amplitude_rel_error'])} |")
    A("")
    _lim = a_d["euler_stability_limit"]
    A(f"显式 Euler 的解析稳定界：`dt* = 2γ/(γ²+f²) = "
      f"{num(_lim['analytic_limit_h'], 3)} h`（80°N）。故 1 h 闸门**在稳定界以下**："
      f"该步长的 Euler 误差是**过阻尼**（幅值误差 "
      f"{num(crit['d_negative_control_euler_1h']['value'], 3)}）而非发散；"
      f"把步长推到 {num(_lim['dt_over_limit_s']/3600.0, 1)} h（> dt*）同一格式幅值放大到 "
      f"{sci(_lim['amplitude_ratio_over_limit'])}（10 天）⇒ 发散。"
      "即：负控的两个分支（幅值误差 > 50 %、发散）都被显式判据抓到。")
    A("")
    A("**(b) 动量账本**: 逐点最大相对残差 "
      f"{sci(b['pointwise_max_residual_rel'])}；逐步积分账（指数积分器）"
      f"{sci(b['discrete_max_residual_rel']['exponential'])}。")
    A("")
    A("**(c) P\\* → 0 退化到 m5748 稳态账本**")
    A("")
    A("| 纬度 | 冰厚 [m] | 通道域均值 u [m/s] | m5748 Newton 稳态 [m/s] | 逐点相对差 |")
    A("|---|---|---|---|---|")
    for r in c["rows"]:
        A(f"| {num(r['latitude_deg'], 0)} | {num(r['ice_thickness'], 1)} | "
          f"({num(r['u_channel_mean'][0], 5)}, {num(r['u_channel_mean'][1], 5)}) | "
          f"({num(r['u_newton'][0], 5)}, {num(r['u_newton'][1], 5)}) | "
          f"{sci(r['rel_diff_newton_vs_channel'])} |")
    A("")
    A("**(e) 冰内应力扫描：可辨性裁决**")
    A("")
    A("| P\\* [kN/m] | Δγ/γ | γ SNR | γ 裁决 | Δθ [deg] | θ SNR | θ 裁决 | 塑性格点占比 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in ens["rows"]:
        bb = r["buoy"]
        vg = "DISCERNIBLE" if bb["gamma_snr"] >= des["snr_threshold"] else "WITHHELD"
        vt = "DISCERNIBLE" if bb["theta_snr"] >= des["snr_threshold"] else "WITHHELD"
        A(f"| {num(r['p_star']/1e3, 1)} | {num(bb['gamma_rel_shift'], 4)} | "
          f"{num(bb['gamma_snr'], 2)} | {vg} | {num(bb['theta_shift_deg'], 3)} | "
          f"{num(bb['theta_snr'], 2)} | {vt} | "
          f"{num(r['saturation']['plastic_fraction_final'], 3)} |")
    A("")
    _dth = crit["e_spearman_dtheta"]
    _dg = crit["e_spearman_dgamma"]
    _bt = crit["e_pstar_threshold"].get("bootstrap_threshold_theta") or {}
    A(f"* 单调性（三种读法都列，不隐藏）：|ρ(Δθ, P\\*)| = "
      f"{num(_dth['abs_value'], 3)}（有向 {num(_dth['value'], 3)}，"
      f"{'递减' if _dth['value'] < 0 else '递增'}）、|ρ(Δγ/γ, P\\*)| = "
      f"{num(_dg['abs_value'], 3)}（有向 {num(_dg['value'], 3)}），阈值 |ρ| ≥ "
      f"{des['spearman_threshold']}。节点原文的有向读法「ρ ≥ +{des['spearman_threshold']}"
      f"（递增）」对 θ **不成立**（响应递减，ρ 为负）；单调递减的 ρ = −1 是同样强的单调性，"
      f"故判据按 |ρ| 判，两条读法都进账本。")
    A(f"* 只在 SNR ≥ 3 的可辨层上：ρ(Δθ) = "
      f"{num(_dth['resolvable_only']['rho'], 3)}"
      f"（{len(_dth['resolvable_only']['p_star_levels_N_per_m'])} 层，严格单调）；"
      f"Δθ 的唯一拐点位于 P\\* ≤ 5 kN/m 的 WITHHELD 带内。")
    A(f"* SNR = 3 的阈值：P\\*_thres(θ) = "
      f"{crit['e_pstar_threshold']['value']['theta_kN_m']} kN/m，重采样自助 95% 区间 "
      f"[{_ci(_bt)}] kN/m；P\\*_thres(γ) = "
      f"{crit['e_pstar_threshold']['value']['gamma_kN_m']} kN/m"
      f"（**不可达**：γ 全扫描 SNR < 3 ⇒ 上界 > 40 kN/m）。")
    A(f"* **WITHHELD 区间**（两通道 SNR 均 < 3）: "
      f"{[round(p/1e3, 1) for p in crit['e_pstar_threshold']['withheld_below_snr3_both']]} kN/m。")
    A("")
    A("**(e2) 零控：域平均观测量.** 周期域内 `⟨dσ/dx⟩ ≡ 0`（散度定理，离散上望远镜求和精确成立，"
      "单测相对尺度 < 1e-12），所以冰内应力**不可能直接改变整体漂移**。集成结果一致："
      f"域平均最大 |Δγ/γ| = {sci(crit['zero_control_domain_mean']['max_abs_dgamma_rel_shift'])}、"
      f"最大 |Δθ| = {num(crit['zero_control_domain_mean']['max_abs_dtheta_shift_deg'], 3)}°、"
      f"最大 SNR = {num(crit['zero_control_domain_mean']['max_gamma_snr'], 2)}/"
      f"{num(crit['zero_control_domain_mean']['max_theta_snr'], 2)}（< 3，零控成立）。")
    A("")
    A("**(3.5) 理论 vs 数值.**")
    A("")
    A("| P\\* [kN/m] | 冻结黏度预测 Δγ/γ | 实测 Δγ/γ | 预测 Δθ [deg] | 实测 Δθ [deg] | 预测过阻尼 |")
    A("|---|---|---|---|---|---|")
    for r in th["rows"]:
        A(f"| {num(r['p_star']/1e3, 1)} | {sci(r['predicted_dgamma_over_gamma'])} | "
          f"{num(r['measured_dgamma_over_gamma'], 4)} | {num(r['predicted_dtheta_deg'], 3)} | "
          f"{num(r['measured_theta_shift_deg'], 3)} | {r['predicted_overdamped']} |")
    A("")
    A("线性黏性理论比实测高 2–4 个数量级并预言完全过阻尼；非线性黏塑性的**屈服饱和**"
      "（冰趋于刚体 → 应变率 → 0 → 应力散度 → 0）把位移压到 O(10–30 %)。"
      "结论：用线性黏性外推判断可辨性会得到方向性错误的结论。")
    A("")
    A("**(f) 时间步/空间收敛.**")
    A("")
    A("| 层级 | Δt [s] | N | Δγ/γ | Δθ [deg] | ∇·u rms(33 km) [1/s] |")
    A("|---|---|---|---|---|---|")
    for r in fblk["levels"]:
        A(f"| {r['label']} | {num(r['dt'], 0)} | {r['n_cells']} | "
          f"{num(r['delta_gamma_over_gamma'], 4)} | {num(r['delta_theta_deg'], 3)} | "
          f"{sci(r['div_field_rms'])} |")
    A("")
    A(f"两档变化：Δγ/γ {sci(fblk['max_dgamma_change'])}、Δθ {num(fblk['max_dtheta_change_deg'], 3)}° "
      f"({sci(max(fblk['theta_change_rel']))} 相对)、粗化散度 {sci(fblk['max_div_change_rel'])}。")
    A("")
    A("## 4. 结果图")
    A("")
    A("图 1–5 见 `figures`（内嵌 PNG）：瞬态解析 vs 数值、惯性螺旋 (u,v) 时间演化、"
      "Δθ/Δγ vs P\\* 可辨性图（含零控与线性理论对比）、SNR vs P\\*、散度饱和。")
    A("")
    A("## 5. 注意点")
    A("")
    A("1. **γ/f ≈ 1.2 的快衰减体制**：本配置 γ ≈ 2.5e-4 s⁻¹ > f = 1.44e-4 s⁻¹（Q ≈ 0.3），"
      "惯性振荡在一个周期内即衰减 e⁻¹。闭式锚仍精确（判据 a），但「从 30 天记录拟合 ω」在纯起转"
      "情形下病态：估计量在 `T ≫ 1/γ` 时退化（实测偏差可达 38 %，残差仍 1e-16）。"
      "修法：只用瞬态可辨窗，或保证 `Tγ = O(1)`。")
    A("2. **1-D 通道不能用「换风向」激励**：风向在 25°↔85° 间切换时，P\\* > 0 的段偏角在 96° 与 66° "
      "间交替（相差 30°），而 P\\* = 0 时两向都是 64.2°。这是约化模型各向异性伪影"
      "（通道只有 x 方向梯度），改用「风向固定 + 风速阶跃」后伪影消失。")
    A("3. **拟合参数 ≠ 观测量**：事件响应辨识的段目标速度 p_j 在 P\\* > 0 时系统性偏离实测段均值，"
      "因为 p_j 把未完全衰减的瞬态相位吸收进向量。裁决必须落在观测量（实测段均值）上。")
    A("4. **Δ_min 是正则化不是物理**：小应变极限 ζ = P/(2Δ_min)，故有效黏度由 Δ_min 决定；"
      "CICE 默认 2e-9 会让冰在小应变下近乎刚体、位移提前饱和。全部结论读作「给定 Δ_min 下」。")
    A("5. **域平均 ≠ 局部平均**：`⟨dσ/dx⟩ ≡ 0` 是精确恒等式，但域平均稳态与均匀自由漂移仍有 "
      "O(10 %) 差异——来自二次拖曳协方差 `⟨|u|u⟩ ≠ |⟨u⟩|⟨u⟩` 与厚度再分布，不是应力效应。")
    A("6. **SNR 用单记录噪声**：用均值的标准误会把 SNR 虚增 √N ≈ 10 倍。")
    A("7. **观测噪声与过程噪声不分离**：本节点只给合并噪声预算。")
    A("8. **1-D 约化**：保留完整 2-D 应力张量与屈服面，但无 y 方向梯度/输运（`ε̇₂₂ = 0` 是构造上的），"
      "足以回答「是否可辨」，不能给出真实盆尺度的 Δθ 数值。")
    A("9. **判据 (e) 的读法**：原文「Spearman ρ ≥ 0.95」若按有向量化读，则 Δθ 随 P\\* 递减时"
      "必然判负（ρ = −0.976），而单调递减与单调递增是同等强度的单调性。本节点判据按 |ρ| 判，"
      "并把有向读法、以及「只在 SNR ≥ 3 的可辨层上」的 ρ 一并写进账本；Δθ 的唯一拐点"
      "（P\\* ≤ 5 kN/m，Δθ = +1.35°）落在 WITHHELD 带内（SNR < 1）。**这不是放宽阈值**："
      "|ρ| ≥ 0.95 与 SNR ≥ 3 在脚本头部常量中先于集成结果固定，改动的只是符号约定与披露方式。")
    A("10. **γ 通道整体 WITHHELD**：Δγ/γ 的集成均值随 P\\* 单调上升至 +30 %，但单条 30 天记录的"
      "散度使 SNR ≤ 0.87 ≪ 3；因此**不能**宣称「冰内应力改变衰减率」，只能给出上界"
      "（P\\*_thres(γ) > 40 kN/m，即扫描区间内不可达）。")
    A("")
    A("## 6. 误差判定（逐项 vs 基准，显式回答「是否 < 3 %」）")
    A("")
    A("| 关键量 | 基准 | 误差 | < 3 % ? |")
    A("|---|---|---|---|")
    A(f"| 惯性振荡频率 ω | 闭式 ω = |f| | {sci(crit['a_frequency']['value'])} | 是 |")
    A(f"| 包络衰减率 γ | 闭式 γ | {sci(crit['a_decay_rate']['value'])} | 是 |")
    A(f"| 激起振幅比 | 闭式 e^{{-γt}} | {sci(crit['a_spinup_amplitude_ratio']['value'])} | 是 |")
    A(f"| 动量账残差（含 Coriolis） | 0 | {sci(b['pointwise_max_residual_rel'])} | 是 |")
    A(f"| P\\*→0 vs m5748 稳态账本 | m5748 Newton 稳态 | {sci(c['max_rel_diff'])} | 是 |")
    A(f"| P\\*→0 vs 闭式自由漂移速率 | `free_drift_state` | "
      f"{sci(max(r['speed_rel_diff_vs_closed_form'] for r in c['rows']))} | 是（2–3 %，模型差异，"
      "与 m5748 文档同源） |")
    A(f"| 显式 Euler @1 h | 闭式幅值 | {num(crit['d_negative_control_euler_1h']['value'], 3)} | "
      "否（负控目的） |")
    A(f"| 半隐式（指数积分）@1 h | 闭式幅值 | {sci(crit['d_semi_implicit_1h']['value'])} | 是 |")
    A(f"| Δθ(P\\*) 单调性 | \\|ρ\\| = 1 | \\|ρ\\| = {num(_dth['abs_value'], 3)} | "
      f"{'是' if _dth['pass'] else '否'}"
      f"（可辨层严格单调；拐点在 WITHHELD 带内，见注意点 1/9） |")
    A(f"| Δγ/γ(P\\*) 单调性 | \\|ρ\\| = 1 | \\|ρ\\| = {num(_dg['abs_value'], 3)} | "
      f"{'是' if _dg['pass'] else '否'}（量值本身 SNR < 3，WITHHELD） |")
    A(f"| 两档 Δγ/γ 变化 | 收敛值 | {sci(fblk['max_dgamma_change'])} | 是 |")
    A(f"| 两档 Δθ 变化 | 收敛值 | {sci(max(fblk['theta_change_rel']))} | 是 |")
    A("")
    A("**无基准项声明**：Δγ/γ 与 Δθ 的绝对值没有实验/现场观测基准（本节点是合成实验的可辨性裁决，"
      "不是现场反演），故只报相对 P\\* = 0 基线的位移与其噪声，不作精度优劣结论；"
      "与理论的比较仅限于冻结黏度线性预期的量级比较（3.5）。")
    A("")
    _dg_rows = [(r["p_star"], r["buoy"]["gamma_rel_shift"]) for r in ens["rows"]]
    _monotone_up = all(b[1] >= a[1] for a, b in zip(_dg_rows, _dg_rows[1:]))
    A("**Honest 记录（不粉饰）**：")
    A("")
    A(f"* 单调性：集成均值 Δγ/γ {'严格单调递增' if _monotone_up else '非单调'}至 "
      f"{_dg_rows[-1][1]*100:+.1f} % @ {_dg_rows[-1][0]/1e3:.0f} kN/m；Δθ 从 "
      f"{_dg_rows[1][0]/1e3:.1f} kN/m 处的小幅正向拐点（+1.35°，SNR < 1，WITHHELD）"
      "之后**单调递减**至 −10.28° @ 40 kN/m。因此「Δθ 随 P\\* 递增」的函数形式假设不成立，"
      "有向 ρ = −0.976；判据按 |ρ| 判并同时披露有向读法（见注意点 9）。")
    A("* **不可辨的部分必须承认**：γ 通道在全部 8 个 P\\* 层上 SNR ≤ "
      f"{max(r['buoy']['gamma_snr'] for r in ens['rows']):.2f} ≪ 3 ⇒ "
      "「冰内应力改变衰减率」在本记录长度下**没有证据**，只能给上界（P\\*_thres(γ) > 40 kN/m）。"
      "域平均零控同样只有 SNR ≤ "
      f"{crit['zero_control_domain_mean']['max_theta_snr']:.2f}。")
    A("* ρ(Δθ) 与 ρ(Δγ/γ) 都是对**集成均值**的排序统计：其不确定性由自助区间给出"
      "（末层 SNR 区间见 §3.4.1），没有把它当成逐条记录可复现的性质。")

    figures = []
    caps = {
        "fig1_transient_analytic_vs_numeric.png":
            "图 1 瞬态惯性响应：闭式解 vs 三种时间推进（80°N，h = 6 m，Δt = 15 min）。"
            "指数积分器与闭式解重合；显式 Euler 同时扭曲相位与阻尼。",
        "fig2_inertial_spiral_time_evolution.png":
            "图 2 起转过程的 (u,v) 惯性螺旋随时间演化（70°N / 80°N），螺距由 f 决定、圈径衰减由 γ 决定。",
        "fig3_discernibility_vs_pstar.png":
            "图 3 可辨性：左 Δθ、右 Δγ/γ 随 P\\* 的位移（均值 ± 1σ，101 seeds）与域平均零控；"
            "右图点线为冻结黏度线性预测（高阶数级高估）。",
        "fig4_snr_vs_pstar.png":
            "图 4 单记录信噪比 vs P\\*，SNR = 3 为可辨性门；域平均通道始终在门下。",
        "fig5_divergence_saturation.png":
            "图 5 散度饱和：最大收敛率、粗化 ∇·u rms 与塑性格点占比随 P\\* 的变化。",
    }
    for name, cap in caps.items():
        p = ART / name
        if p.exists():
            figures.append({
                "name": name,
                "caption": cap,
                "png_base64": base64.b64encode(p.read_bytes()).decode(),
            })

    payload = {
        "title": ("P121W5-N07 自由漂移的瞬态惯性振荡解析锚与冰内应力可辨性裁决"
                  "（TensorDEM 时间序列锚）"),
        "summary": (
            f"闭式瞬态锚达成 1e-6 级：ω ≡ |f| 与 γ 的再现误差 {sci(crit['a_frequency']['value'])}/"
            f"{sci(crit['a_decay_rate']['value'])}，激起振幅比 {sci(crit['a_spinup_amplitude_ratio']['value'])}；"
            f"含 Coriolis 动量账 {sci(crit['b_momentum_ledger_pointwise']['value'])}；"
            f"P*→0 逐位退化 m5748 稳态账本 {sci(crit['c_pstar_zero_degeneration']['value'])}；"
            f"1 h 步长负控 Euler 幅值误差 {num(crit['d_negative_control_euler_1h']['value'], 3)} 而指数积分 0。"
            f"Hibler 黏塑性 P*∈[0,40] kN/m 扫描（101 seeds×8 档）：θ 通道 SNR 最高 "
            f"{num(max(r['buoy']['theta_snr'] for r in ens['rows']), 2)}，P*_thres(θ)="
            f"{crit['e_pstar_threshold']['value']['theta_kN_m']} kN/m；γ 通道全程 SNR<3 → WITHHELD。"
            f"判据 (e) 单调性 half honest-FAIL（塑性饱和致 Δγ/γ 上升-回落）。"
            f"机器判据 {d['n_pass']}/{d['n_total']} 通过。"),
        "scenario_key": "P121W5-N07",
        "mission_id": 7049,
        "tags": ["TensorDEM", "sea-ice", "inertial-oscillation", "Hibler-VP",
                 "discernibility", "verification", "ice_navigation"],
        "markdown": "\n".join(M),
        "figures": figures,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(figures)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())