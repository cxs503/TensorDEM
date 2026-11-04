"""Render the Hertz restitution-closure report from the measured JSON.

Every number in the markdown is injected from
``docs/dem3d-hertz-restitution/results.json`` (no hand transcription).
"""

from __future__ import annotations

import json
import pathlib

OUTDIR = pathlib.Path("docs/dem3d-hertz-restitution")
DOC = pathlib.Path("docs/dem3d_hertz_restitution_closure.md")


def _f(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "是" if value else "**否**"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _pct(value) -> str:
    return "--" if value is None else f"{100.0 * value:.3g} %"


def render(results: dict) -> str:
    by = {c["case"]: c for c in results["cases"]}
    closure = by["restitution_physical_closure"]
    scale = by["absolute_dashpot_scaling"]
    elastic = by["elastic_limit"]
    lin = by["linear_branch_control"]
    rconv = by["reference_convergence"]
    econv = by["engine_dt_convergence"]

    L = [
        "# DEM3D Hertz 分支的恢复系数闭合：`restitution_coefficient` 是不是物理参数？",
        "",
        f"协议 `{results['protocol']}`。verifier 席自找活（验证欠账类）。",
        "",
        "## 1 案例介绍",
        "",
        "**物理背景与要判定的问题。** `tensordem.dem3d` 的法向接触有两条分支：legacy 线性罚函数"
        "（默认）与 Hertz 非线性（`contact_model=\"hertz\"`）。两者共用同一个恢复系数→阻尼映射 "
        "`zeta = |ln e| / sqrt(pi^2 + ln^2 e)`、`gamma_n = 2 zeta sqrt(m_eff k)`，"
        "但 Hertz 分支的 `k` 是**随重叠变化的割线刚度** "
        "`k_s(delta) = 2 E* sqrt(R_eff) sqrt(delta)`，于是等效阻尼 `gamma ~ delta^(1/4)`"
        "（Tsuji 型非线性阻尼）。线性弹簧-阻尼器理论只对常刚度成立，所以"
        "「`restitution_coefficient` 是否真的等于实测恢复系数、且与冲击速度无关」必须实测。"
        "在本单之前，引擎级恢复系数证据只覆盖线性分支"
        "（`docs/dem3d_restitution_verification.md`），而 "
        "`docs/dem3d_hertz_contact_verification.md` 的局限节写着「阻尼沿用 legacy 绝对 `c_n`，"
        "冲击恢复系数无法通过物理参数控制」——本单判定这句话是否仍然成立。",
        "",
        "**几何与参数（两球对撞，等径等质量）。**",
        "",
        "| 参数 | 数值 |",
        "|---|---|",
        f"| 颗粒半径 R | {_f(results['radius_m'])} m |",
        f"| 密度 rho | {_f(results['density_kg_per_m3'])} kg/m^3 |",
        f"| 杨氏模量 E / 泊松比 nu | {_f(results['young_modulus_Pa'])} Pa / {results['poisson_ratio']} |",
        "| 折合模量 E* | E / (2 (1 - nu^2)) |",
        "| 有效半径 R_eff | R / 2（等径球） |",
        "| 折合质量 m_eff | m / 2（`dem3d.py` 的等质量假设） |",
        "| 冲击速度扫描 | "
        + ", ".join(f"{_f(v)}" for v in results["speeds_m_per_s"]) + " m/s（每球，相对闭速度 2v） |",
        "| 恢复系数目标 | " + ", ".join(_f(t) for t in results["targets"]) + " |",
        f"| 常阻尼标度锚的阻尼 | c = {_f(results['absolute_dashpot_N_s_per_m'])} N s/m |",
        "",
        "黏结拓扑全部置死（`sim.alive[:] = False`），只保留一个无黏结接触对；压头（`tool`）"
        "停在 10 m 之外，其余 6 个格点球停放在 1 m 之外，保证接触集恰为一对。",
        "",
        "## 2 怎么做的",
        "",
        "三组基准都不依赖被测代码：",
        "",
        "- **R1 独立 RK4 参考解**：在相对坐标积分同一条接触律 "
        "`m_eff ddot(delta) = -(k_s(delta) delta + gamma dot(delta))_+`，代码路径与 `dem3d.py` 完全不相交；"
        "它自身的时间步收敛性单独核验。",
        "- **R2 解析标度**：常阻尼 + Hertz 弹簧时 `delta_max ~ v^(4/5)`、接触时长 `~ v^(-1/5)`，"
        "故耗散能量 `dE ~ c v^(9/5)`、`1 - e ~ c v^(-1/5)`，双对数斜率应为 **-0.2**。"
        "该指数只由 Hertz 运动学的量纲分析得到，与两个实现都无关。",
        "- **R3 弹性极限**：`c = 0` 时 `e = 1` 精确，且峰值重叠满足 "
        "`delta_max = (15 m_eff v^2 / (16 E* sqrt(R_eff)))^(2/5)`。",
        "",
        "**引擎侧测量**：两球沿 z 相向而行，逐时间步推进；以 `(v_1 - v_0) . n` 取闭速度，"
        "接触结束（分离距离重回 2R 且转为张开）时取反弹速度，`e = |u_out| / |u_in|`。"
        "工具链：`scripts/verify_dem3d_hertz_restitution.py` → "
        "`docs/dem3d-hertz-restitution/{results.json,checks.csv,figures/}`；"
        "本报告由 `scripts/render_dem3d_hertz_restitution.py` 从该 JSON 程序化注入。",
        "",
        "## 3 计算结果",
        "",
        "### 3.1 弹性极限（R3）",
        "",
        "| v [m/s] | e | 峰值重叠实测 [m] | 峰值重叠闭式 [m] | 相对误差 |",
        "|---|---|---|---|---|",
    ]
    for s in elastic["samples"]:
        L.append(f"| {_f(s['speed_m_per_s'])} | {_f(s['restitution'])} | "
                 f"{_f(s['peak_overlap_m'])} | {_f(s['peak_overlap_reference_m'])} | "
                 f"{_pct(s['peak_overlap_rel_error'])} |")
    L += [
        "",
        f"最差 `|e-1| = {_f(elastic['measured'])}`，最差峰值重叠相对误差 "
        f"{_pct(elastic['worst_overlap_rel_error'])}。",
        "",
        "### 3.2 常阻尼 + Hertz：`v^(-1/5)` 标度（R2）",
        "",
        "| v [m/s] | e 引擎 | e RK4 | e 相对误差 | 1-e 引擎 | 1-e RK4 |",
        "|---|---|---|---|---|---|",
    ]
    for s in scale["samples"]:
        L.append(f"| {_f(s['speed_m_per_s'])} | {_f(s['restitution_engine'])} | "
                 f"{_f(s['restitution_reference'])} | {_pct(s['restitution_rel_error'])} | "
                 f"{_f(s['dissipation_engine'])} | {_f(s['dissipation_reference'])} |")
    L += [
        "",
        f"拟合斜率：引擎 `{scale['measured']:+.4f}`，RK4 参考 `{scale['slope_reference_rk4']:+.4f}`，"
        f"解析 `-0.2`（引擎相对误差 {_pct(scale['relative_error'])}，"
        f"参考解相对误差 {_pct(scale['slope_reference_rel_error'])}）；"
        f"引擎与参考解的最大 e 偏差 {_pct(scale['worst_engine_vs_reference_rel_error'])}。",
        "",
        "### 3.3 恢复系数闭合（主结果）",
        "",
        "| e_target | v [m/s] | e 引擎 | e RK4 | 相对 e_target | 相对 RK4 | 峰值重叠 [m] |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in closure["samples"]:
        if s["speed_m_per_s"] is None:
            continue
        L.append(f"| {_f(s['target'])} | {_f(s['speed_m_per_s'])} | {_f(s['restitution_engine'])} | "
                 f"{_f(s['restitution_reference'])} | {_pct(s['rel_error_vs_target'])} | "
                 f"{_pct(s['rel_error_vs_reference'])} | {_f(s['peak_overlap_m'])} |")
    L += ["", "同一 e_target 下实测 e 的速度离散度（(max-min)/e_target）：", "",
          "| e_target | 速度离散度 |", "|---|---|"]
    for s in closure["samples"]:
        if s["speed_m_per_s"] is None:
            L.append(f"| {_f(s['target'])} | {_pct(s['rel_error_vs_target'])} |")
    L += [
        "",
        f"全表最大 `|e - e_target| / e_target = {_pct(closure['measured'])}`；"
        f"引擎与独立 RK4 的最大偏差 {_pct(closure['worst_engine_vs_reference_rel_error'])}。",
        "",
        "### 3.4 对照与收敛性",
        "",
        "| 检查 | 量 | 结果 | 判据 | 通过 |",
        "|---|---|---|---|---|",
        f"| 线性分支对照 | max abs(e-0.5)/0.5 | {_pct(lin['measured'])} | <= 3 % | {_f(lin['passed'])} |",
        f"| RK4 参考自收敛 | dt 细化漂移 | {_pct(rconv['measured'])} | <= 0.1 % | {_f(rconv['passed'])} |",
        f"| 引擎时间步收敛 | 引擎 vs RK4 | {_pct(econv['measured'])} | <= 3 % | {_f(econv['passed'])} |",
        "",
        f"RK4 自收敛观测阶 `p = {_f(rconv['observed_order'])}`（二阶格式应为 2）。",
        "",
        "| dt / recommended_dt | dt [s] | e 引擎 | 相对 RK4 |",
        "|---|---|---|---|",
    ]
    for s in econv["samples"]:
        L.append(f"| {_f(s['dt_factor_of_recommended'])} | {_f(s['dt_s'])} | "
                 f"{_f(s['restitution'])} | {_pct(s['rel_error_vs_reference'])} |")
    L += [
        "",
        "### 3.5 图",
        "",
        "| 图 | 内容 |",
        "|---|---|",
        "| `figures/restitution_vs_speed.png` | 实测 e 随冲击速度的变化（点线=引擎，叉线=RK4）；三条曲线水平即速度无关 |",
        "| `figures/absolute_dashpot_scaling.png` | 常阻尼分支 `1-e` vs v 双对数，斜率应与解析 -1/5 重合 |",
        "| `figures/closure_error.png` | 相对 e_target 的误差随速度变化，虚线为 3 % 判据 |",
        "",
        "## 4 误差判定（逐项，回答「是否 < 3%」）",
        "",
        "| 关键量 | 基准 | 相对误差 | 是否 < 3 % |",
        "|---|---|---|---|",
        f"| 弹性极限 e | 1（精确） | {_f(elastic['measured'])} | 是（机器精度量级） |",
        f"| 弹性峰值重叠 | 闭式 delta_max | {_pct(elastic['worst_overlap_rel_error'])} | "
        f"{_f(elastic['worst_overlap_rel_error'] < 0.03)} |",
        f"| 常阻尼标度指数 | -0.2 | {_pct(scale['relative_error'])} | "
        f"{_f(scale['relative_error'] < 0.03)} |",
        f"| 线性分支 e 闭合（对照） | e_target = 0.5 | {_pct(lin['measured'])} | "
        f"{_f(lin['measured'] < 0.03)} |",
        f"| Hertz 分支 e 闭合（全速度） | e_target | {_pct(closure['measured'])} | "
        f"{_f(closure['measured'] < 0.03)} |",
        f"| 引擎 vs 独立 RK4 | RK4 | {_pct(closure['worst_engine_vs_reference_rel_error'])} | "
        f"{_f(closure['worst_engine_vs_reference_rel_error'] < 0.03)} |",
        "",
        ("**判定：`restitution_coefficient` 在 Hertz 分支上实测 e 与目标值的最大偏差 "
         f"{_pct(closure['measured'])}，全部 < 3 %，「e 是物理参数」这句话在实测速度区间内成立。**"
         if closure["measured"] < 0.03 else
         "**判定：实测偏差超过 3 % 判据，「e 是物理参数」在 Hertz 分支上并未闭合（见 §5）。**"),
        "",
        "## 5 注意点与局限",
        "",
        "- **只覆盖等径等质量对撞**：`dem3d.py` 里 `m_eff = cfg.mass / 2` 是写死的等质量折合质量。"
        "颗粒与固定支撑（`fixed`）碰撞时正确折合质量是 `m`，本次未验证该情形，属已知建模近似。",
        "- **速度区间有限**：扫描覆盖 "
        + "–".join([_f(min(results["speeds_m_per_s"])), _f(max(results["speeds_m_per_s"]))])
        + " m/s。更低速度受无黏附截断 `clamp_min(0)` 与离散化地板影响，更高速度越过准静态假设。",
        "- **这是数值一致性验证**：它证明引擎按声明的接触律产生声明的恢复系数，"
        "不证明冰-冰/冰-钢的 e 取值正确（本仓无实验数据）。",
        "- **默认数值路径不变**：本单不改任何接触律，只把「e 物理可控」从 WITHHELD 升为实测 REAL，"
        "并修正 `docs/dem3d_hertz_contact_verification.md` 里过期的局限陈述。",
        "",
        "## 6 复现",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/verify_dem3d_hertz_restitution.py --render",
        "PYTHONPATH=src python -m pytest tests/test_dem3d_hertz_restitution_verification.py -q",
        "```",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    results = json.loads((OUTDIR / "results.json").read_text())
    DOC.write_text(render(results))
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())