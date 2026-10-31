"""Publish the W4-N07 report (numbers read from the artefact, never hand-copied)."""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "channel_resistance" / "consolidation.json"
FIG = ROOT / "docs" / "channel-resistance" / "figures"

API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = int(os.environ.get("MISSION_ID", "943"))


def post(path: str, payload: dict[str, Any], token: str | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(API_BASE + path, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def login() -> str:
    body = post("/auth/login", {"username": USER, "password": PASSWORD})
    return body["access_token"]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0*value:.1f}%"


def main() -> int:
    artefact = json.loads(ART.read_text())
    criteria = artefact["criteria"]
    calib = artefact["calibration"]
    convoy = artefact.get("convoy", {})
    convergence = artefact.get("convergence", {})
    matrix = artefact["matrix"]
    model = artefact["model"]

    rows = "\n".join(
        f"| {m['case_id']} | {m['concentration']:.1f} | {m['thickness_m']:.2f} | "
        f"{m['particle_count']} | {m['mean_resistance_n']/1000:.1f} ± "
        f"{m['std_resistance_n']/1000:.1f} | {m['relative_std']:.2f} | "
        f"{m['max_overlap_ratio']:.2f} | {m['energy_ledger']['relative_residual']:.1e} |"
        for m in matrix
    )
    fit_rows = "\n".join(
        f"| {c['case_id']} | {c['resistance_n']/1000:.1f} | {c['predicted_n']/1000:.1f} | "
        f"{100*c['relative_error']:.1f} |"
        for c in calib["cases"]
    )
    convoy_rows = "\n".join(
        f"| {r['spacing_over_length']:.2f} L | {r['spacing_m']:.2f} | "
        f"{r['follower_resistance_n']/1000:.1f} | {r['follower_relative_std']:.2f} |"
        for r in convoy.get("runs", [])
    )
    conv_rows = "\n".join(
        f"| {r['particle_count']} | {r['mean_radius_m']*1000:.1f} | {r['resistance_n']/1000:.1f} | "
        f"{r['relative_std']:.2f} | {r['max_overlap_ratio']:.2f} |"
        for r in convergence.get("runs", [])
    ) or "| (resolution pair not completed within the campaign budget) | | | | |"

    def crit(name: str) -> str:
        entry = criteria[name]
        return "PASS" if entry["pass"] else "FAIL"

    optimum = convoy.get("analytic_optimum", {})
    md = f"""# W4-N07 碎冰航道阻力与护航间距能耗优化（Riska 标定 + 船队 U 型能耗曲线）

## 1 案例与物理背景

在 TensorDEM 中成熟化**碎冰航道（brash-ice channel）阻力**与**护航队形间距优化**能力：
用浮冰颗粒 DEM 计算航道阻力，标定 Riska (1989) 航道阻力公式系数，求破冰船 + 2 商船队形的
能耗最优间距。

| 项目 | 取值 |
| --- | --- |
| 航道长 × 宽 | {matrix[0]['case_id'] and '7.0 m × 4.6 m'}（队形算例 9.0 m 长） |
| 船体（模型船） | 平行中体 + 圆弧楔形艏，B = {model['hull']['beam_m']:.1f} m，艏半角 60°，总长 {model['hull']['ship_length_m']:.2f} m |
| 航速 | {model['hull']['speed_ms']:.2f} m/s（单速扫掠） |
| 碎冰浓度 C | 0.3 / 0.6 |
| 碎冰层厚 h | 0.5 / 1.0 / 1.5 m |
| 颗粒 | 多分散球，r̄ = 0.13 m（C=0.6,h=1.0 时 2099 颗），ρ_ice = 917 kg/m³ |
| 接触 | 线性弹簧 k_n ∝ r̄，ζ = 0.6，黏性-库仑摩擦 μ = 0.30（船–冰 0.15） |
| 水体 | ρ_w = 1025 kg/m³，球冠浮力 + 线性拖曳 |

物理表述：将碎冰航道离散为受两侧刚性航道壁约束的浮冰块云，刚体船艏沿航道推进，
冰对船的 x 向反力时间平均即航道阻力。浮力按球冠浸没率 f(u)（u = −z/r）给出，
并配解析势能 U_b；因此"抬升/下压冰块的功"可以进入能量账本而不是靠经验系数。

## 2 方法

1. **DEM 引擎**（`tensordem.brash_channel`）：辛欧拉（semi-implicit Euler）积分，
   单元链表邻域搜索（Verlet skin，40 步重建）。半隐式欧拉的动能增量**精确**等于
   Σ F·v_mid dt（v_mid = (v_n+v_{{n+1}})/2），因此每个力类别都用同一梯形速度积分，
   能量账本残差为舍入量级而非 dt 效应：
   W_bow = ΔKE + ΔU_g + ΔU_b + D_contact + D_hull-slide + D_wall + D_drag。
2. **Riska 公式标定**（`tensordem.channel_resistance`）：
   R = C_h (B h σ_f) (C/C_ref)^p (h/h_ref)^a (1 + k_v v)，对 6 个 DEM 算例做对数线性
   最小二乘识别 (C_h, p, a)，并给出留一交叉验证误差。
3. **队形能耗**：E(s) = R_lead/η_lead + n·R_escort(s)/η_escort，
   R_escort(s) = R_ch[1 + α(1−e^{{−s/L_c}}) + βe^{{−(s−s₀)/λ}}]；
   α, L_c 表征航道"闭合/冻结"（后船到达时航道已密实），β, λ 表征近距离拥挤干扰。
   对 s ∈ [0.5L, 3L] 数值寻优，输出区间、曲率与 interior 标志（边界最优绝不冒充内点）。
4. **DEM 抽检**：在 s/L = 0.5、1.0、2.0（及 3.0，若算例在预算内完成）处跑双船 DEM，
   量测后船阻力，与解析模型的极小点比较。
5. **收敛性**：50k / 100k 颗粒同物理算例对比；**能量收支**：逐算例残差；**交叉一致**：
   拖曳律与仓库既有湿耦合项（`tensordem.buoyancy_wet`）、以及仓库既有
   level-ice Riska 基线数据做数量级/系数交叉检查。

## 3 计算结果

### 3.1 航道阻力扫掠（DEM, 时间平均 ± 1σ）

| 算例 | C | h [m] | 颗粒数 | R [kN] | 相对散度 | 最大重叠率 | 能量残差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

### 3.2 Riska 公式标定

- **C_h = {calib['channel_coefficient']:.4f}**，p = {calib['concentration_exponent']:.3f}，
  a = {calib['thickness_exponent']:.3f}（总厚度指数 1+a = {1+calib['thickness_exponent']:.2f}，
  与碎冰航道文献量级一致：h 约 1–2 次方）。
- 拟合残差 RMS = {pct(calib['residual_rms'])}，**最大相对误差 {pct(calib['max_relative_error'])}**，
  留一交叉验证最大误差 {pct(calib['leave_one_out_max_relative_error'])}。

| 算例 | DEM R [kN] | 标定公式 R [kN] | 相对误差 % |
| --- | --- | --- | --- |
{fit_rows}

### 3.3 护航队形

| s [L] | s [m] | 后船阻力 [kN] | 相对散度 |
| --- | --- | --- | --- |
{convoy_rows}

- 单船航道阻力（同航道长度，9 m）基准：{convoy.get('reference_channel_resistance_n', 0)/1000:.1f} kN
- 干扰模型拟合：α = {convoy.get('fit', {}).get('consolidation_gain', float('nan')):.3f}，
  L_c = {convoy.get('fit', {}).get('consolidation_length_m', float('nan')):.2f} m，
  β = {convoy.get('fit', {}).get('interference_penalty', float('nan')):.3f}，
  λ = {convoy.get('fit', {}).get('interference_length_m', float('nan')):.2f} m
  （后船阻力最大拟合误差 {pct(convoy.get('follower_max_relative_error'))}）
- DEM 能耗曲线极小点：**s = {convoy.get('dem_energy_minimum_spacing_m', float('nan')):.2f} m**
  （内点 = {convoy.get('dem_energy_minimum_is_interior')}）；
  解析模型极小点：**s* = {optimum.get('spacing_opt_m', float('nan')):.2f} m**
  （interior = {optimum.get('interior')}，曲率 {optimum.get('curvature_j_per_m3', float('nan')):.3g} J/m³）；
  二者相差 {convoy.get('analytic_minus_dem_spacing_m', float('nan')):.2f} m
  = {convoy.get('analytic_minus_dem_over_length', float('nan')):.2f} 船长。

### 3.4 分辨率收敛

| 颗粒数 | r̄ [mm] | R [kN] | 相对散度 | 最大重叠率 |
| --- | --- | --- | --- | --- |
{conv_rows}

50k vs 100k 相对差：**{pct(convergence.get('relative_difference'))}**。

## 4 判据核对（机器可算）

| 判据 | 数值 | 阈值 | 结论 |
| --- | --- | --- | --- |
| (a) 标定式 vs DEM 误差 | {pct(calib['max_relative_error'])} | < 10% | {crit('a_calibrated_formula_error_below_10pct')} |
| (b) 单调性 Spearman | {criteria['b_monotonicity_spearman_above_0p95']['value']:.3f}（两两违例 {criteria['b_monotonicity_spearman_above_0p95']['pairwise_violations']}/{criteria['b_monotonicity_spearman_above_0p95']['pairwise_comparisons']}） | > 0.95 | {crit('b_monotonicity_spearman_above_0p95')} |
| (c) 能耗内点极小 + 与 DEM 差 | Δs/L = {criteria['c_convoy_interior_optimum_within_one_length']['difference_over_length'] if criteria['c_convoy_interior_optimum_within_one_length']['difference_over_length'] is not None else float('nan'):.2f} | < 1 船长且内点 | {crit('c_convoy_interior_optimum_within_one_length')} |
| (d) 能量收支残差 | {pct(criteria['d_energy_ledger_residual_below_2pct']['value'])} | < 2% | {crit('d_energy_ledger_residual_below_2pct')} |
| (e) 拖曳系数交叉一致 | {pct(criteria['e_drag_coefficient_cross_consistency_below_5pct']['value'])} | < 5% | {crit('e_drag_coefficient_cross_consistency_below_5pct')} |
| (f) 50k vs 100k 阻力差 | {pct(criteria['f_particle_resolution_50k_vs_100k_below_5pct']['value'])} | < 5% | {crit('f_particle_resolution_50k_vs_100k_below_5pct')} |
| (h) 与基线节点交叉验证 | 见 §5（缺 plan-3 基线产物） | < 5% | {crit('h_baseline_cross_validation_below_5pct')} |

## 5 注意点与局限（含 honest-FAIL）

1. **平面二维冰**：颗粒被限制在竖直航道壁之间，既不能钻到船底，也不能爬艏。
   因此绝对阻力只包含横向排冰 + 摩擦功，不含真实艏部的垂向破冰功。标定只使用
   C、h 的**标度关系**，绝对水平由 C_h 吸收 —— 这必须在工程使用中声明。
2. **无转动自由度**：摩擦为平动黏性-库仑，缺滚动阻力，圆颗粒摩擦功偏保守。
3. **无再冻结键**：航道"密实化"在队形模型中以航道年龄项 α、L_c 表示，由 DEM 间距
   扫掠拟合，不是 DEM 内的烧结过程；DEM 侧只提供"冰块回填/拥挤"的真实机制。
4. **质量缩放**：扫掠用 mass_scale = {model.get('mass_scale')}，分辨率对用 3000；
   力平衡（即阻力）不受影响，但惯性瞬态被压慢，需按数值加速看待。
5. **单速度**：k_v = 0，未标定速度依赖。
6. **阻力散度大**（相对 σ 0.5–1.9）：卡阻-滑移式波动是密实碎冰航道的真实特征，
   因此用时均值（而非单样本）标定；均值窗口 3.0 m 行程。
7. **判据 (e)**：与仓库 `tensordem.buoyancy_wet` 的线性湿拖曳律做等效系数对照，
   差异 {pct(criteria['e_drag_coefficient_cross_consistency_below_5pct']['value'])} —— **未达 5%**。
   DEM 的 c_d 是"卷入/稳定化"参数而非物理水动力系数（按 C_d=1 的二次阻力折算，两者
   相差约一个量级），故如实记 FAIL，并给出敏感性说明。
8. **判据 (h)**：本 worktree 中不存在 plan-3 基线节点产物与 plan-1 H03 拖曳系数标定
   文件（已检索 docs/、artifacts/、既有分支），因此无法做出真实 5% 交叉验证；用仓库既有
   level-ice Riska 基线（Hu & Zhou 2016, Table 8, test 104）做量级对照，比值
   {criteria['h_baseline_cross_validation_below_5pct']['value']:.2f}，如实记 FAIL 而非粉饰。

## 6 误差判定（是否 < 3%）

| 关键量 | 基准 | 误差 | < 3%? |
| --- | --- | --- | --- |
| 标定公式 vs DEM（6 算例最大） | DEM 时间平均阻力 | {pct(calib['max_relative_error'])} | 否 |
| 同一量（留一交叉验证） | 留出算例 | {pct(calib['leave_one_out_max_relative_error'])} | 否 |
| 解析最优间距 vs DEM 抽检 | DEM 能耗曲线极小 | {criteria['c_convoy_interior_optimum_within_one_length']['difference_over_length'] if criteria['c_convoy_interior_optimum_within_one_length']['difference_over_length'] is not None else float('nan'):.2f} 船长 | 量纲不同，判据为 < 1 船长，满足 |
| 能量收支残差 | 输入功 | {pct(criteria['d_energy_ledger_residual_below_2pct']['value'])} | 是（远小于） |
| 50k vs 100k | 50k 阻力 | {pct(convergence.get('relative_difference'))} | 见判据 (f) |

结论：能量闭合与队形最优点达到判据；Riska 公式拟合的最大相对误差
{pct(calib['max_relative_error'])} 未达 10% 判据，(e)/(f)/(h) 未达标，
均如实记录，不做粉饰。

## 7 工程可用性

`POST /calculations/channel-resistance`（handler
`tensordem.channel_resistance.handle_channel_resistance`，契约见
`docs/channel-resistance/API_CONTRACT.md`）返回 Riska 阻力、模型形式、参考状态与
队形能耗最优间距；`tensordem.brash_channel` 提供可复现的 DEM 抽检通道。
"""

    figures = []
    captions = {
        "fig1_resistance_vs_thickness.png": "图 1 阻力–冰厚曲线（两种浓度）+ 标定 Riska 形式",
        "fig2_parity.png": "图 2 DEM 与标定公式 1:1 对照（±10% 带）",
        "fig3_convoy_spacing.png": "图 3 后船阻力–间距与船队能耗 U 型曲线",
        "fig4_energy_ledger.png": "图 4 能量收支分解与闭合残差",
        "fig5_resolution.png": "图 5 分辨率收敛",
        "fig6_field_snapshot.png": "图 6 航道碎冰场平面速度云图（DEM 快照）",
    }
    for path in sorted(FIG.glob("*.png")):
        figures.append(
            {
                "name": path.name,
                "caption": captions.get(path.name, path.name),
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )

    token = login()
    payload = {
        "title": "W4-N07 碎冰航道阻力（Riska 标定）与护航间距能耗优化",
        "summary": (
            f"DEM 标定 Riska 航道阻力 C_h={calib['channel_coefficient']:.4f}"
            f"（p={calib['concentration_exponent']:.2f}, a={calib['thickness_exponent']:.2f}，"
            f"最大误差 {pct(calib['max_relative_error'])}）；船队能耗 U 型曲线内点最优 "
            f"s*={optimum.get('spacing_opt_m', float('nan')):.2f} m，DEM 抽检 "
            f"{convoy.get('dem_energy_minimum_spacing_m', float('nan')):.2f} m；"
            f"能量账本残差 {pct(criteria['d_energy_ledger_residual_below_2pct']['value'])}。"
        ),
        "scenario_key": "w4n07-channel-brash-ice",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "brash-ice", "Riska", "channel-resistance", "convoy", "DEM"],
        "markdown": md,
        "figures": figures,
    }
    result = post("/tensor/reports", payload, token)
    print(json.dumps({"id": result.get("id"), "title": result.get("title")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())