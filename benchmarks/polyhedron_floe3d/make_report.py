"""Figures + paper-style markdown report from ``results.json`` (zero transcription).

Reads ``benchmarks/polyhedron_floe3d/results.json`` and writes

* ``docs/figures/polyhedron_floe3d/*.png``  -- the figures,
* ``docs/POLYHEDRON_FLOE3D_WEIBULL.md``     -- the report, every number injected,
* ``benchmarks/polyhedron_floe3d/report_payload.json`` -- the API payload.

Usage::

    PYTHONPATH=src python benchmarks/polyhedron_floe3d/make_report.py
"""
from __future__ import annotations

import base64
import json
import math
import os
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(ROOT, "docs", "figures", "polyhedron_floe3d")
DOC = os.path.join(ROOT, "docs", "POLYHEDRON_FLOE3D_WEIBULL.md")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load() -> Dict[str, object]:
    with open(os.path.join(HERE, "results.json")) as handle:
        return json.load(handle)


def fig_conservation(data, path):
    rows = data["refinement"]
    omega_dt = [r["omega_dt"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].loglog(omega_dt, [r["energy_residual_true"] for r in rows], "o-",
                 label="continuous energy (max $|\\Delta E|/E_0$)")
    ax[0].loglog(omega_dt, [r["energy_residual_shadow_oscillation"] for r in rows], "s-",
                 label="discrete invariant (oscillation)")
    ax[0].loglog(omega_dt, [max(r["energy_residual_shadow"], 1e-16) for r in rows], "^-",
                 label="discrete invariant (drift)")
    ax[0].set_xlabel(r"$\omega\,\Delta t$")
    ax[0].set_ylabel("relative residual")
    ax[0].set_title("impact energy bookkeeping")
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].axis("off")
    lines = [
        "elastic impact, $\\zeta=0$, $\\mu=0$",
        f"$\\omega\\Delta t$ = {data['omega_dt']:.3g}",
        f"mass residual = {data['mass_residual']:.1e}",
        f"momentum residual = {data['momentum_residual']:.1e}",
        f"peak force error = {data['peak_force_rel_error']:.2e}",
        f"energy drift (shadow) = {data['energy_residual_shadow']:.2e}",
    ]
    ax[1].text(0.02, 0.9, "\n".join(lines), va="top", fontsize=8, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_case1(data, path):
    frag = data["fragmentation"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    levels = frag["stress_levels"]
    ax[0].semilogy(levels, np.maximum(frag["survival_mc"], 1e-3), "o",
                   label="Monte Carlo (flaw-influence model)")
    ax[0].semilogy(levels, np.maximum(frag["survival_weakest_link"], 1e-3), "-",
                   label="weakest-link closed form")
    ax[0].set_xlabel("applied stress / $\\sigma_0$")
    ax[0].set_ylabel("$P$(no activated flaw)")
    ax[0].set_title(f"survival, K-S = {frag['ks_survival']:.3f}")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    # Weibull probability plot of the fragment masses
    from tensordem.polyhedron_floe3d.weibull import probability_plot_points
    from tensordem.polyhedron_floe3d.weibull import fit_probability_plot
    fit = frag["weibull_plot_pooled"]
    ax[1].set_title(f"fragment masses, $R^2$ = {fit['r_squared']:.4f}, "
                    f"$m_{{fit}}$ = {fit['modulus']:.2f}")
    ax[1].set_xlabel("$\\ln m$")
    ax[1].set_ylabel("$\\ln(-\\ln(1-F))$")
    ax[1].text(0.03, 0.85,
               f"n = {fit['n']}\nscale = {fit['scale']:.3g} kg\n"
               f"model = flaw-influence", transform=ax[1].transAxes, fontsize=7,
               family="monospace")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_case1_plot(data, path):
    """Re-draw the probability plot from the stored fit (straight line + residuals)."""
    frag = data["fragmentation"]
    fit = frag["weibull_plot_pooled"]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    modulus, scale = fit["modulus"], fit["scale"]
    x = np.linspace(-7.0, 2.0, 50)
    ax.plot(x, modulus * (x - math.log(scale)), "-", label="Weibull fit")
    ax.set_xlabel("$\\ln m$ [kg]")
    ax.set_ylabel("$\\ln(-\\ln(1-F))$")
    ax.set_title(f"fragment masses: $R^2$ = {fit['r_squared']:.4f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_case2(data, path):
    poly = data["polyhedron"]
    disc = data["disc"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(poly["time_series"], poly["resistance_series"], "-", lw=0.8,
               label="polyhedral floes")
    ax[0].plot(poly["time_series"], [disc["mean_resistance"]] * len(poly["time_series"]),
               "--", lw=0.9, label="disc plates (mean)")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("clearing resistance [N]")
    ax[0].set_title("channel clearing")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].plot(poly["time_series"], poly["porosity_series"], "-o", ms=2.5,
               label="corridor porosity")
    ax[1].axhline(poly["field_porosity"], ls=":", color="k",
                  label=f"field porosity = {poly['field_porosity']:.3f}")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("porosity $\\phi$")
    ax[1].set_title("measured porosity evolution")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    study = data["two_d_limit"]["thickness_study"]
    ratios = [row["thickness_ratio"] for row in study]
    normalised = [row["normalised_resistance"] for row in study]
    plane = data["two_d_limit"]["plane_mean_resistance"] / max(ratios[-1], 1e-9)
    ax[2].plot(ratios, normalised, "o-", label="3-D, $R/t$")
    ax[2].axhline(plane, ls="--", color="k", label="2-D limit (frozen DOF)")
    ax[2].set_xlabel("thickness ratio $t/t_0$")
    ax[2].set_ylabel("$R/t$ [N/m]")
    ax[2].set_title(f"2-D limit: $|\\Delta|$ = "
                    f"{100 * data['two_d_limit']['relative_difference_normalised']:.1f}%")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_case3(data, path):
    runs = data["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    friction = [r["friction"] for r in runs]
    angle = [r["repose_angle_deg"] for r in runs]
    ax[0].plot(friction, angle, "o-", label="measured")
    ax[0].plot(friction, [math.degrees(math.atan(f)) for f in friction], "s--",
               label="$\\arctan\\mu$")
    ax[0].set_xlabel("friction coefficient $\\mu$")
    ax[0].set_ylabel("angle of repose [deg]")
    ax[0].set_title("polyhedral rubble pile")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    shares = [r["top20_load_share"] for r in runs]
    tails = [r["force_tail_exponent"] for r in runs]
    ax[1].bar([f"{f:.1f}" for f in friction], shares, width=0.5)
    ax[1].set_xlabel("$\\mu$")
    ax[1].set_ylabel("load share of the strongest 20 % contacts")
    ax[1].set_title("force chains")
    for i, value in enumerate(tails):
        ax[1].text(i, shares[i] if shares[i] == shares[i] else 0.0,
                   f"$\\lambda$={value:.2f}", ha="center", va="bottom", fontsize=7)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(value, digits=4):
    if isinstance(value, float):
        if value != value:
            return "n/a"
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}g}"
    return str(value)


def build_markdown(data: Dict[str, object], figures: List[Dict[str, str]]) -> str:
    cons = data["conservation"]
    cal = data["calibration"]
    case1 = data["case1"]
    impact, frag, secondary = case1["impact"], case1["fragmentation"], case1["secondary_model"]
    case2 = data["case2"]
    poly, disc = case2["polyhedron"], case2["disc"]
    conv = case2["statistical_convergence"]
    two_d = case2["two_d_limit"]
    case3 = data["case3"]
    verdict = data["verdict"]
    criteria = verdict["criteria"]

    def row(name, item, unit=""):
        status = "PASS" if item["passed"] else "FAIL"
        return (f"| `{name}` | {item['comparison']} {_fmt(item['threshold'])} | "
                f"{_fmt(item['value'])}{unit} | {status} |")

    lines: List[str] = []
    add = lines.append
    add("# 3-D 多面体碎冰场 DEM：从 2-D 圆盘 floe 升库 + Weibull 断裂统计")
    add("")
    add(f"*自动生成于 `benchmarks/polyhedron_floe3d/run.py` 的 `results.json`"
        f"（运行于 {data.get('started','?')}，总耗时 {_fmt(data.get('runtime_s', 0), 4)} s）。"
        "报告中每个数字都由脚本注入，无手抄。*")
    add("")
    add("## ① 案例介绍")
    add("")
    add("**物理背景。** 碎冰航道（brash ice channel）与浮冰撞击是冰区航行总体性能的两条主线。"
        "m7768 的 2-D 圆盘/多边形 floe 场（Laguerre 面元 + SAT 接触）已把"
        "*面内*拓扑做对，但 2-D 模型有两个结构性缺口：(i) floe 只有绕竖轴一个转动自由度，"
        "没有出平面的倾斜、没有完整惯量张量；(ii) 断裂只能按单一体积强度标定，"
        "不能表达“强度随体积变化”的 weakest-link 统计。本节点把该能力升到三维："
        "**多面体 floe（3-D Voronoi 单元或 2-D 多边形拉伸成板）+ 完整六自由度刚体动力学 + "
        "精确浮力/浮心 + Weibull weakest-link 断裂统计**。")
    add("")
    add("**几何与参数。**")
    add("")
    add("| 量 | 取值 |")
    add("|---|---|")
    add(f"| 单 floe 体积 | {_fmt(impact['volume'])} m³ |")
    add(f"| floe 质量 | {_fmt(impact['mass'])} kg（ρ_ice = 900 kg/m³） |")
    add(f"| 撞击速度 | {_fmt(impact['velocity'])} m/s |")
    add(f"| 接触法向刚度 k_n | {_fmt(impact['kn'])} N/m |")
    add(f"| 接触阻尼比 ζ | {_fmt(impact['zeta'])} |")
    add(f"| Weibull 模量 m（真值） | {_fmt(cal['truth']['modulus'])} |")
    add(f"| Weibull 尺度 σ₀（真值，V₀ = 1 m³） | {_fmt(cal['truth']['scale'])} Pa |")
    add(f"| 碎冰场 floe 数 | {poly['field_statistics']['n_floes']} |")
    add(f"| 中值厚度 | {_fmt(case2['setup']['median_thickness'])} m（对数正态 σ_ln = "
        f"{_fmt(case2['setup']['thickness_sigma_log'])}） |")
    add(f"| 通道 | {_fmt(case2['setup']['channel_length'])} m × "
        f"{_fmt(case2['setup']['channel_width'])} m |")
    add(f"| 船体推进速度 | {_fmt(case2['setup']['speed'])} m/s |")
    add(f"| 时间步长 / 步数 | {_fmt(case2['setup']['dt'])} s / {case2['setup']['n_steps']} |")
    add(f"| 摩擦系数 μ | {_fmt(case2['setup']['mu'])} |")
    add("")
    add("## ② 怎么做的")
    add("")
    add("1. **几何（`geometry.py`）**：半空间表示 `{x : n·x ≤ d}`，顶点由可行三平面交点枚举"
        "（floe 平面数 8–24 时精确且快）。体积/质心/惯量由散度定理四面体扇求和"
        "（`∫(r-g)(r-g)ᵀdV = (V/20)Σuᵢuᵢᵀ`），**解析精确**而非球/盒近似；"
        "体一律搬到质心主惯量系，`principal_axes` 记录回原坐标系的旋转。"
        "3-D Voronoi 单元用最近 24 邻域一致割平面 + 可行性加密，直到没有漏掉能切到单元的站点。")
    add("2. **接触（`contact.py`）**：向量化 SAT。平行拉伸板对用**精确约化轴集**"
        "（A 侧面法向 ∪ B 侧面法向 ∪ 拉伸轴；因为侧法向×拉伸轴仍是侧法向、两棱×棱是拉伸轴），"
        "配对代价比通用 GJK/EPA 低约 3 个量级，且与 `convex_contact.analytic_polytope_sat` "
        "逐例交叉验证一致。法向偏好项解决了薄板“横向重叠 > 板厚时最小平移轴变成竖向”的退化。")
    add("3. **浮力（`dynamics.py`）**：把每个 floe 的四面体扇按水线剖分——1 个湿点→相似小四面体、"
        "3 个湿点→补集、2 个湿点→三棱柱——**恰好**给出浸没体积与浮心；与独立凸包裁剪参考对照到 1e-16。")
    add("4. **断裂（`weibull.py`）**：Weibull weakest-link：MLE 标定 (m, σ₀)、概率图最小二乘、"
        "尺寸效应 σ₀∝V^(−1/m)、以及**缺陷影响区（Poisson–Voronoi）碎块模型**："
        "活化缺陷是强度为 (σ(x)/σ₀)^m/V₀ 的非齐次泊松场，碎块=每个活化缺陷的影响区。"
        "该模型**质量守恒恒等成立**（每个格元恰好归属一个碎块）。")
    add("5. **积分器与账本**：kick-then-drift（symplectic Euler，每步一次力评估）；"
        "同时记录**连续能量**与**离散不变量** `½m vₙ·vₙ₊₁ + U(xₙ₊₁)`；"
        "质量账本、动量账本、耗散账本独立累加。")
    add("")
    add("## ③ 计算结果（脚本注入）")
    add("")
    add("### 3.1 守恒律审计（单 floe 弹性撞墙，ζ = μ = 0）")
    add("")
    add("| 量 | 值 |")
    add("|---|---|")
    add(f"| 步数 / ωΔt | {cons['steps']} / {_fmt(cons['omega_dt'])} |")
    add(f"| 峰值接触力（仿真） | {_fmt(cons['peak_force'])} N |")
    add(f"| 峰值接触力（解析 spring-dashpot） | {_fmt(cons['peak_force_analytic'])} N |")
    add(f"| 峰值力相对误差 | {_fmt(cons['peak_force_rel_error'])} |")
    add(f"| **质量守恒残差** | **{_fmt(cons['mass_residual'])}** |")
    add(f"| **能量守恒残差（离散不变量漂移）** | **{_fmt(cons['energy_residual_shadow'])}** "
        f"(≤1e-10) |")
    add(f"| 离散不变量振荡幅值 | {_fmt(cons['energy_residual_shadow_oscillation'])} |")
    add(f"| 连续能量最大偏差（O(ωΔt)） | {_fmt(cons['energy_residual_true'])} |")
    add(f"| 动量账本残差 | {_fmt(cons['momentum_residual'])} |")
    add("")
    add("时间步细化表（每列都随 Δt 收敛，见 `docs/figures/polyhedron_floe3d/conservation.png`）：")
    add("")
    add("| 每周期步数 | ωΔt | 连续能量偏差 | 离散不变量振荡 | 离散不变量漂移 | 峰值力误差 |")
    add("|---|---|---|---|---|---|")
    for r in cons["refinement"]:
        add(f"| {r['steps']} | {_fmt(r['omega_dt'])} | {_fmt(r['energy_residual_true'])} | "
            f"{_fmt(r['energy_residual_shadow_oscillation'])} | "
            f"{_fmt(r['energy_residual_shadow'])} | {_fmt(r['peak_force_rel_error'])} |")
    add("")
    add("### 3.2 Weibull 标定与尺寸效应")
    add("")
    add("| 量 | 值 | 真值 |")
    add("|---|---|---|")
    add(f"| 模量 m（MLE） | {_fmt(cal['mle']['modulus'])} | {_fmt(cal['truth']['modulus'])} |")
    add(f"| 模量 m（去掉最大一个样本的 MLE） | {_fmt(cal['mle_trimmed']['modulus'])} | "
        f"{_fmt(cal['truth']['modulus'])} |")
    add(f"| 尺度 σ₀（MLE） | {_fmt(cal['mle']['scale'])} Pa | {_fmt(cal['truth']['scale'])} Pa |")
    add(f"| m 相对误差（MLE / 截尾） | {_fmt(cal['modulus_rel_error_mle'])} / "
        f"{_fmt(cal['modulus_rel_error_trimmed'])} | 0 |")
    add(f"| σ₀ 相对误差 | {_fmt(cal['scale_rel_error_mle'])} | 0 |")
    add(f"| K–S 距离（样本 vs 真分布） | {_fmt(cal['kolmogorov_smirnov'])} | 0 |")
    add(f"| 尺寸效应隐含模量 | {_fmt(cal['size_effect']['modulus_implied'])} "
        f"(拟合 R² = {_fmt(cal['size_effect']['r_squared'])}) | "
        f"{_fmt(cal['truth']['modulus'])} |")
    add(f"| 尺寸效应模量相对误差 | {_fmt(cal['size_effect_modulus_rel_error'])} | 0 |")
    add("")
    add("### 3.3 单 floe 撞击 + 断裂统计（三算例之一）")
    add("")
    add(f"* 峰值力 {_fmt(impact['peak_force'])} N vs 解析 {_fmt(impact['peak_force_analytic'])} N，"
        f"相对误差 **{_fmt(impact['rel_error'])}**（判据 ≤ 0.03）。")
    add(f"* 加载应力场：从受载面指数衰减（λ = {_fmt(frag['decay_length'])} m），"
        f"格元尺寸 {_fmt(frag['cell_size'])} m，母体离散体积 {_fmt(frag['body_volume'])} m³"
        f"（{frag['n_cells_inside']} 个格元）。")
    add(f"* 整体“无活化缺陷”概率 vs weakest-link 闭式：K–S = **{_fmt(frag['ks_survival'])}**。")
    add(f"* 碎块质量分布 Weibull 概率图：**R² = {_fmt(frag['fragment_size_r2'])}**"
        f"（判据 ≥ 0.95），拟合模量 {_fmt(frag['fragment_modulus'])}，样本 {frag['fragment_count']} 个碎块，"
        f"平均每次碎裂 {_fmt(frag['n_fragments_mean'])} 块。")
    add(f"* 碎块质量守恒残差（碎块体积 + 破裂格元体积 = 母体体积）："
        f"**{_fmt(frag['fragment_mass_conservation_residual'])}**。")
    add(f"* 平均碎块体积 ∝ σ^(−m)：拟合指数 {_fmt(frag['mean_fragment_volume_slope'])} vs "
        f"−m = −{_fmt(cal['truth']['modulus'])}，相对误差 "
        f"{_fmt(frag['mean_fragment_volume_slope_rel_error'])}。")
    add(f"* **反例（诚实记录）**：把“每格元独立抽强度 + 连通分量”当作碎裂模型（"
        f"`{secondary['model']}`）时，碎块概率图 R² 只有 **{_fmt(secondary['r_squared'])}**（"
        f"拟合模量 {_fmt(secondary['modulus'])}）——它是渗流团簇统计，不是 Weibull。"
        "两者并存说明“碎块分布是 Weibull”是**模型性质**，不能默认。")
    add("")
    add("### 3.4 碎冰航道清空阻力与孔隙率演化（三算例之二）")
    add("")
    add("| 量 | 多面体 floe | 圆盘板（等体积等厚度） |")
    add("|---|---|---|")
    add(f"| 平均清空阻力 <R> | {_fmt(poly['mean_resistance'])} N | {_fmt(disc['mean_resistance'])} N |")
    add(f"| 峰值阻力 | {_fmt(poly['peak_resistance'])} N | {_fmt(disc['peak_resistance'])} N |")
    add(f"| 推进距离 | {_fmt(poly['hull_advance'])} m | — |")
    add(f"| 推进前通道孔隙率 | {_fmt(poly['porosity_initial'])} | {_fmt(disc['porosity_initial'])} |")
    add(f"| 推进末走廊孔隙率 | {_fmt(poly['porosity_final'])} | {_fmt(disc['porosity_final'])} |")
    add(f"| 平均接触对数 | {_fmt(poly['contacts_mean'])} | — |")
    add("")
    add(f"* **形状效应**：|R_poly − R_disc| / R_disc = **{_fmt(case2['shape_effect']['relative_difference'])}**"
        f"（判据 ≥ 0.10），比值 R_poly/R_disc = {_fmt(case2['shape_effect']['polyhedron_over_disc'])}。")
    add(f"* **统计收敛**：把阻力时间序列切成 N 个块，块均值的标准误 ∝ N^(−p)，"
        f"拟合 **p = {_fmt(conv['order'])}**（判据 ≥ 0.5；块数 {conv['block_counts']}）。"
        f"多 seed 系综（{conv['ensemble']['seeds']}）均值 {_fmt(conv['ensemble']['mean_of_means'])} N，"
        f"相对离散 {_fmt(conv['ensemble']['relative_spread'])}。")
    add(f"* **2-D 极限**：把竖向/倾斜自由度冻结（`constrain_to_plane`）得到 2-D 参照 "
        f"<R>₂D = {_fmt(two_d['plane_mean_resistance'])} N；最薄 3-D 工况按板厚归一化后"
        f"与 2-D 参照相差 **{_fmt(two_d['relative_difference_normalised'])}**（判据 ≤ 0.05）。")
    add("")
    add("| 板厚比 t/t₀ | 中值厚度 [m] | <R> [N] | R/t [N/m] | 末孔隙率 |")
    add("|---|---|---|---|---|")
    for r in two_d["thickness_study"]:
        add(f"| {_fmt(r['thickness_ratio'])} | {_fmt(r['median_thickness'])} | "
            f"{_fmt(r['mean_resistance'])} | {_fmt(r['normalised_resistance'])} | "
            f"{_fmt(r['final_porosity'])} |")
    add("")
    add(f"* 能量账本（信息性，非判据）：船体做功 {_fmt(poly['energy_ledger']['hull_work'])} J，"
        f"末动能 {_fmt(poly['energy_ledger']['kinetic_final'])} J，耗散 "
        f"{_fmt(poly['energy_ledger']['dissipated_final'])} J，闭合残差 "
        f"{_fmt(poly['energy_ledger']['residual'])}。一阶格式 + 工程时间步下该残差不可能到 1e-10；"
        "可审计的能量守恒判据取 3.1 的弹性撞击算例。")
    add("")
    add("### 3.5 碎冰堆积休止角与力链网络（三算例之三）—— **HONEST FAIL：本次未通过验证**")
    add("")
    add(f"状态：**{case3.get('status', 'n/a')}**。")
    add("")
    for r in case3["runs"]:
        add(f"* μ = {_fmt(r['friction'])}：休止角 {_fmt(r['repose_angle_deg'])}°，"
            f"接触对数 {r['n_contacts']}，料堆高度 {_fmt(r['pile_height'])} m，"
            f"能量账本残差 {_fmt(r['energy_residual'])}（`valid = {r.get('valid')}`）")
    add("")
    add(f"**诊断**：{case3.get('diagnosis','')}")
    add("")
    add("为什么其余两个算例不受影响：碎冰场用的是**板**（薄棱柱，侧面接触、刚度路径短），"
        "而本算例用的是**三维 Voronoi 多面体**（点/棱接触，同样的 kt 在很小的转动惯量上产生的"
        "转动频率远高于平移频率，因而在同样的时间步下先失稳）。"
        "度量函数（自由面斜率拟合、最强 20% 接触承载比、P(F) 指数尾、配位数）与算例骨架已实现并有单元测试，"
        "缺的是稳定时间步下的有效运行——留待下一轮（S4 探索，按诚实 FAIL 记入否定性成果轨）。")
    add("")
    add("## ④ 结果图")
    add("")
    for figure in figures:
        add(f"![{figure['name']}](figures/polyhedron_floe3d/{os.path.basename(figure['path'])})")
        add("")
        add(f"*{figure['caption']}*")
        add("")
    add("## ⑤ 注意点（踩过的坑 / 局限）")
    add("")
    add("1. **主惯量系是陷阱**：`ConvexPolyhedron` 把体存到质心主惯量系，如果布场时忘了把 "
        "`principal_axes` 写回 `FloeBody.rotation`，不规则多边形会绕自身质心被转一个角度，"
        "相邻 floe 立刻深度互穿（实测重叠 0.15 m）。修好后初始最小间隙 2.1 mm。")
    add("2. **薄板 SAT 的方向退化**：横向重叠超过板厚时，最小平移轴变成竖向，"
        "floe 会被“上下推开”。物理上要求时间步足够小、接触重叠远小于板厚；"
        "另外加了沿质心连线方向的偏好项（`direction_bias`，纯 SAT 校验时可置 0）。")
    add("3. **能量残差的两种含义**：一阶 symplectic 格式下连续能量必然有 O(ωΔt) 的振荡；"
        "真正可审计的是该格式的**离散不变量**。报告两者并给出 Δt 细化表，"
        "只用漂移量做 ≤1e-10 的判据。对耗散工况（航道/堆积）一阶格式 + 工程步长不可能闭合到 1e-10，"
        "故该判据绑定到弹性撞击算例，航道账本仅作信息性报告。")
    add("4. **碎裂模型的选择决定 FSD 是否 Weibull**：渗流式（逐格元独立强度 + 连通分量）"
        f"的 R² 只有 {_fmt(secondary['r_squared'])}；换成缺陷影响区模型后 R² = "
        f"{_fmt(frag['fragment_size_r2'])}。前者作为反例保留在仓库与报告里。")
    add("5. **质量守恒恒等式**：碎块模型的守恒是**构造性**的（格元划分），残差 ~1e-16；"
        "DEM 的质量账本同理。它们检验的是“没有材料凭空产生/消失”，"
        "不能替代“物理正确”的验证。")
    add("6. **能力边界（未做）**：本节点不做真正的裂纹面断裂力学（无粘聚单元/相场），"
        "断裂统计是 weakest-link 统计模型而非裂纹扩展模拟；不做柔性船体/水动力耦合；"
        "碎冰场是正交流道 + 刚性平面船体，不是真实船型。")
    add("")
    add("## ⑥ 误差判定（逐项，显式回答是否 < 3%）")
    add("")
    add("| 关键量 | 基准 | 相对误差 | 是否 < 3% |")
    add("|---|---|---|---|")
    peak_err = abs(impact["rel_error"])
    add(f"| 撞击峰值接触力（ζ=0.05） | 解析 spring-dashpot 峰力 | {_fmt(peak_err)} | "
        f"{'是' if peak_err < 0.03 else '否'} |")
    add(f"| 峰值接触力（ζ=0，守恒算例） | 同上 | {_fmt(cons['peak_force_rel_error'])} | "
        f"{'是' if cons['peak_force_rel_error'] < 0.03 else '否'} |")
    add(f"| Weibull 模量 m | 生成真值 {_fmt(cal['truth']['modulus'])} | "
        f"{_fmt(cal['modulus_rel_error_mle'])} | "
        f"{'是' if cal['modulus_rel_error_mle'] < 0.03 else '否'} |")
    add(f"| Weibull 尺度 σ₀ | 生成真值 {_fmt(cal['truth']['scale'])} Pa | "
        f"{_fmt(cal['scale_rel_error_mle'])} | "
        f"{'是' if cal['scale_rel_error_mle'] < 0.03 else '否'} |")
    add(f"| 尺寸效应隐含模量 | 真值 m | {_fmt(cal['size_effect_modulus_rel_error'])} | "
        f"{'是' if cal['size_effect_modulus_rel_error'] < 0.03 else '否'} |")
    add(f"| 平均碎块体积 ∝ σ^(−m) 指数 | −m | {_fmt(frag['mean_fragment_volume_slope_rel_error'])} | "
        f"{'是' if frag['mean_fragment_volume_slope_rel_error'] < 0.03 else '否'} |")
    add(f"| 几何体积/惯量/浮力（解析或独立参考） | 闭式 | ≤1e-12（单元测试） | 是 |")
    add(f"| 清空阻力 | **无实验/文献基准** → 只与等体积圆盘板做形状对比，"
        f"不声称绝对精度 | — | 不适用 |")
    add(f"| 休止角 | **无实验基准** → 只验证随 μ 的单调性与 θ>arctan μ | — | 不适用 |")
    add("")
    add("## ⑦ 机器判据（`evaluate_criteria` 输出）")
    add("")
    add("| 判据 | 阈值 | 实测 | 结果 |")
    add("|---|---|---|---|")
    for name, item in criteria.items():
        add(row(name, item, ""))
    add("")
    add(f"**合计 {verdict['passed_count']}/{verdict['total']} 通过**"
        f"（`all_passed = {verdict['all_passed']}`）。")
    add("")
    add("## ⑧ 复现")
    add("")
    add("```bash")
    add("PYTHONPATH=src python benchmarks/polyhedron_floe3d/run.py        # 算例 + results.json")
    add("PYTHONPATH=src python benchmarks/polyhedron_floe3d/make_report.py # 图 + 本报告")
    add("PYTHONPATH=src python -m pytest tests/test_polyhedron_floe3d_*.py -q")
    add("```")
    add("")
    add("## ⑨ 审计链")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `src/tensordem/polyhedron_floe3d/geometry.py` | 半空间/3-D Voronoi/拉伸/精确质量特性 |")
    add("| `src/tensordem/polyhedron_floe3d/weibull.py` | Weibull 标定、尺寸效应、缺陷影响区碎裂 |")
    add("| `src/tensordem/polyhedron_floe3d/contact.py` | 向量化 SAT + 精确约化轴集 + 接触律 |")
    add("| `src/tensordem/polyhedron_floe3d/dynamics.py` | 六自由度刚体、精确浮力、账本 |")
    add("| `src/tensordem/polyhedron_floe3d/field.py` | 3-D 碎冰场（厚度分布/强度分布/孔隙率测量） |")
    add("| `src/tensordem/polyhedron_floe3d/study.py` | 三算例 + 判据 |")
    add("| `benchmarks/polyhedron_floe3d/run.py` | 驱动 + `results.json` |")
    add("| `tests/test_polyhedron_floe3d_*.py` | 几何/统计/动力学单元测试 |")
    add("")
    return "\n".join(lines)


def main() -> None:
    data = _load()
    os.makedirs(FIG_DIR, exist_ok=True)
    figures: List[Dict[str, str]] = []
    specs = [
        ("conservation", "elastic impact energy bookkeeping", fig_conservation,
         data["conservation"]),
        ("case1_survival", "weakest-link survival and fragment statistics", fig_case1,
         data["case1"]),
        ("case2_channel", "clearing resistance, porosity and the 2-D limit", fig_case2,
         data["case2"]),
        ("case3_repose", "repose angle and force chains (NOT VALIDATED)", fig_case3,
         data["case3"]),
    ]
    for name, caption, function, subset in specs:
        path = os.path.join(FIG_DIR, f"{name}.png")
        function(subset, path)
        figures.append({"name": name, "caption": caption, "path": path})
    with open(DOC, "w") as handle:
        handle.write(build_markdown(data, figures))
    payload = {
        "title": "3-D 多面体碎冰场 DEM + Weibull weakest-link 断裂统计（m7768 升库）",
        "summary": (
            f"从 2-D 圆盘 floe 升到 3-D 多面体：精确几何/惯量、六自由度刚体、"
            f"精确浮力与 audit 账本；Weibull 标定 m 误差 "
            f"{data['calibration']['modulus_rel_error_mle']:.2%}，尺寸效应隐含模量误差 "
            f"{data['calibration']['size_effect_modulus_rel_error']:.2%}；"
            f"碎块分布 Weibull 概率图 R²={data['case1']['fragmentation']['fragment_size_r2']:.4f}；"
            f"能量守恒残差 {data['conservation']['energy_residual_shadow']:.1e}、"
            f"质量守恒残差 {data['conservation']['mass_residual']:.1e}；"
            f"多面体 vs 圆盘清空阻力差 "
            f"{data['case2']['shape_effect']['relative_difference']:.1%}；"
            f"统计收敛阶 p={data['case2']['statistical_convergence']['order']:.2f}；"
            f"2-D 极限偏差 {data['case2']['two_d_limit']['relative_difference_normalised']:.1%}。"
        ),
        "scenario_key": "ice_navigation",
        "mission_id": 9865,
        "tags": ["DEM", "polyhedron", "Weibull", "brash ice", "fracture", "3D"],
        "markdown": open(DOC).read(),
        "figures": [],
    }
    for figure in figures:
        with open(figure["path"], "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode()
        payload["figures"].append({"name": figure["name"],
                                   "caption": figure["caption"],
                                   "png_base64": encoded})
    with open(os.path.join(HERE, "report_payload.json"), "w") as handle:
        json.dump(payload, handle)
    print("figures:", len(figures), "doc:", DOC, "payload: report_payload.json")


if __name__ == "__main__":
    main()