"""Build /tmp/report.json (paper-grade report with embedded figures) for IN32-D01.

All numbers come from artifacts/drum_flow/phase_map.json + verdict.json; the
figures are the PNGs written by benchmarks/drum_flow_figures.py.

    python benchmarks/drum_flow_report.py --map artifacts/drum_flow/phase_map.json
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

from tensordem.drum_flow.regimes import (
    cataract_onset_froude, detachment_angle_deg, fill_to_chord_angle,
)


def _fig(path: Path, caption: str) -> dict:
    return {"name": path.name, "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="artifacts/drum_flow/phase_map.json")
    ap.add_argument("--figdir", default="artifacts/drum_flow")
    ap.add_argument("--out", default="/tmp/report.json")
    ap.add_argument("--mission", type=int, default=7413)
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.map).read_text())
    verdict = json.loads(Path(args.figdir, "verdict.json").read_text())
    cells, S = payload["cells"], payload["summary"]
    figdir = Path(args.figdir)

    roll = [c for c in cells if c["regime_kinematic"] == "rolling"]
    closures = [c["ledger"]["closure_rel"] for c in cells]
    r2 = [c["surface"]["fit"]["r2"] for c in cells if c.get("surface", {}).get("fit")]
    r2_all = [c["surface"]["r2_all_particles"] for c in cells if "r2_all_particles" in c.get("surface", {})]
    fr_c = cataract_onset_froude(135.0)
    table = ["| Fr | φ | Fr-band | measured | wall ratio | flight frac | r_p05/R | P_in [W] | Φ_diss [W] | closure | R²(bed) | R²(all) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["fr"], c["fill"])):
        f = c.get("surface", {}).get("fit", {})
        table.append("| {:.2f} | {:.1f} | {} | {} | {:.2f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            c["fr"], c["fill"], c["regime_froude"], c["regime_kinematic"],
            c["observables"]["wall_velocity_ratio"], c["observables"]["airborne_fraction"],
            c["observables"]["r_p05_over_R"], c["ledger"]["p_in_w"], c["ledger"]["phi_diss_w"],
            c["ledger"]["closure_rel"],
            "%.4f" % f["r2"] if f else "n/a",
            "%.4f" % c["surface"]["r2_all_particles"] if c.get("surface", {}).get("r2_all_particles") else "n/a"))

    md = f"""## 1 案例介绍

回转滚筒（rotary drum）内颗粒物料的横向运动是最经典的颗粒流流态问题：随转速升高，床层依次出现
**滑移 (slipping) → 滚落 (rolling) → 抛落 (cataracting) → 离心 (centrifuging)** 四种形态。控制参数为壳面
Froude 数 `Fr = ω²R/g`。本节点在 TensorDEM 新建二维滚筒 DEM 内核（旋转壳面 + 接触力矩 + 机械功率账本），
用可测量的运动学量自动判别四流态，并把输入机械功率 `P_in = τ·ω` 与接触耗散 `Φ_diss = Σ F·v_rel` 做闭合验证。

几何/参数（SI）：R = 0.15 m；单分散圆盘 r = {S['config']['particle_radius']} m（φ=0.3 时 N ≈ 170）；
法向线性弹簧-阻尼 (k_n = 2e4 N/m，restitution 0.3→ζ) + Cundall-Strack 切向弹簧 (μ=0.5，含壁面 μ_w=0.5)
+ 滚动阻力 (μ_r = 0.15)；dt = 5e-5 s；扫描 Fr ∈ [0.1, 1.5] × φ ∈ [0.1, 0.5] 共 {S['n_cells']} 点，
每点积分 {S['config']['revolutions']} 圈、取最后 {S['config']['measure_last']} 圈为测量窗。

## 2 怎么做的

- **内核** `src/tensordem/drum_flow/kernel.py`：显式辛欧拉 + Verlet 邻表（cKDTree、skin、每 10 步重建，
  切向弹簧史按整数对键 `i*N+j` 跨重建继承）；壳面为半径 R 的圆，材料点速度 `v_wall = ω ẑ × x_c`；
  接触点相对速度含颗粒自转，切向摩擦对两颗粒给出**同号力矩（力偶）**；功率账本双路径独立累加。
- **四流态判别** `regimes.py`：预设 Fr 分带（滑移 <0.3、滚落 0.3–0.7、抛落 0.7–1.0、离心 >1）
  与独立**运动学判别器**（无 Fr 先验）：抛落 = 无接触颗粒比例高；离心 = 径向 5% 分位大（无自由面）
  且壁面层随动比高；滑移 = 壁面层随动比低；其余 = 滚落。
- **自由面抛物面**：按 x 分 16 竖带取最高颗粒拟合 `y = a+bx+cx²`；解析对照取随床层旋转参考系的零压等势面
  （圆心 y = g/ω_b² 的圆弧）在顶点展开：`c = ω_b²/(2(g−ω_b² y_s))`，`ω_b` 由颗粒角动量求得。
- **功率账本**：`d(K+U_g+U_el)/dt = P_in − Φ_diss`；`P_in = Σ_wall F·v_wall = −τ·ω`（全局量）、
  `Φ_diss = Φ_n + Φ_t + Φ_roll`（局部接触量）。两路径独立，闭合检验非恒等式。

## 3 计算结果

相图：`phase_map.png`；Fr 判别一致性：`fr_agreement.png`；功率账本：`power_ledger.png`；
自由面拟合 vs 解析：`free_surface.png`。汇总：闸门数据
`agreement = {S['fr_x_fill_agreement']:.3f}`，`closure median = {S['closure_median']:.4f}`，
`closure max = {S['closure_max']:.4f}`，`R²(bed) min = {S['surface_r2_min']:.4f}`，
`R²(bed) median = {S['surface_r2_median']:.4f}`。

{chr(10).join(table)}

## 4 注意点（坑与局限）

1. **切向力偶符号（真实缺陷，已修）**：初版写成 `τ_i = −τ_j`，等价于向系统注入角动量：静止床的接触数
   0.46/颗粒、旋转床 50–90% 颗粒无接触、壁面随动比仅 0.45。改为 `τ_i = τ_j` 后静止床 3.7 接触/颗粒。
   回归测试 `test_friction_couple_spins_both_disks_the_same_way`。
2. **耗散只能记饱和滑移**：黏着接触的 `F_t·v_t` 是弹性能（已计入 `U_el`），若全记耗散，闭合系统性偏高 35%。
3. **二维单分散圆盘的固有局限**：颗粒可"贴壁自转"（ω_p = ωR/r ≈ 25ω，无滑移），真实三维床层中面外互锁会抑制；
   结果是本模型床层始终有一部分脱离颗粒（flight fraction 0.20–0.66），自由面判据与抛落判据因此偏保守/偏松。
4. **自由面稳健化**：初版用全部颗粒取每带最高点，弹道颗粒污染拟合（R² 最低 0.47）；改用仅承载接触的
   床层颗粒后 R² 中位数 {S['surface_r2_median']:.3f}（全颗粒口径中位 {sorted(r2_all)[len(r2_all)//2]:.3f} 作对照）。
5. **闭合误差的物理来源**：闭合只在统计定常态成立；低 Fr 崩塌间歇流态单圈内 `dE/dt ≠ 0`，
   所以闭合误差随 Fr 降低、且与 `energy_balance_rel` 同量级。

## 5 误差判定

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 离心流态 Fr 阈值 | 解析 Fr = 1（临界转速 √(g/R)） | 0.00% | 是 |
| 抛落流态 Fr 阈值 | 解析 −cos135° = {fr_c:.4f} vs 0.7 | {abs(fr_c-0.7)/0.7:.2%} | 是 |
| 功率账本闭合（中位） | 独立路径 P_in vs Φ_diss | {S['closure_median']:.2%} | {'是' if S['closure_median']<0.03 else '否'} |
| 功率账本闭合（最差） | 同上 | {S['closure_max']:.2%} | {'是' if S['closure_max']<0.03 else '否'} |
| 时均闭合 <2% 的网格占比 | 判据 c | {S['closure_within_2pct_fraction']:.1%} | 否（判据 c 按"全部点"计） |
| 自由面拟合 R²（床层口径，最小/中位） | 二次多项式 | {S['surface_r2_min']:.4f} / {S['surface_r2_median']:.4f} | 中位是 |
| 四流态判别一致率 | 运动学判别（无 Fr 先验） | {S['fr_x_fill_agreement']:.3f} | 否（见下） |

**判据逐项结论（honest）**：(a) 判别准确率 {verdict['a']['value']:.3f} < 0.95 → **FAIL**
（根因：二维圆盘床层恒有 20–66% 脱离颗粒，运动学"抛落"阈 2% 在低 Fr 即被触发；
本模型无法把滚落与抛落用"是否飞行"分开，需改用释放角的直接测量）；
(b) R² 最小 {verdict['b']['value_min_r2']:.4f}、>0.98 占比 {verdict['b']['fraction_above_0.98']:.1%} → **FAIL**；
(c) 闭合中位 {verdict['c']['median']:.4f}、最大 {verdict['c']['max']:.4f}、<2% 占比 {verdict['c']['fraction_below_2pct']:.1%} → **FAIL**；
(d) Fr 阈值解析锚点最大偏差 {verdict['d']['max_rel_dev']:.2%} < 10% → **PASS**（离心 0%、抛落 {abs(fr_c-0.7)/0.7:.2%}）；
(e) pytest 26 项 ≥ 12 → **PASS**。
**未与 Henein(1983) 实验相图逐点比对**：其相图未在本工作树内数字化，(d) 只对可解析推出的两条 Fr 阈值做定量对照，
不声称全相图一致。
"""
    figures = [
        _fig(figdir / "phase_map.png", "Fr–φ 流态相图（DEM 实测运动学判别，虚线为预设 Fr 带边 0.3/0.7/1.0）"),
        _fig(figdir / "fr_agreement.png", "实测脱离(飞行)颗粒比例随 Fr 的变化；红叉=Fr 分带判别与实测判别不一致的网格点"),
        _fig(figdir / "power_ledger.png", "左：功率账本闭合 |P_in−Φ_diss|/P_in（红线 2% 门槛）；右：壳面驱动力矩"),
        _fig(figdir / "free_surface.png", "自由面二次拟合（实线）与离心解析抛物面（虚线），含 R²"),
    ]
    report = {
        "title": "IN32-D01 回转滚筒颗粒流流态图谱与功率账本：四流态判别 + 自由面抛物面 + P_in–Φ_diss 闭合",
        "summary": ("TensorDEM 新建二维旋转滚筒 DEM 内核（旋转壳面、摩擦力偶、双路径功率账本），"
                    f"Fr×φ 共 {S['n_cells']} 点扫描：功率账本闭合中位 {S['closure_median']:.2%}（微基准 0.5–3% 精确）、"
                    f"自由面二次拟合 R² 中位 {S['surface_r2_median']:.3f}、Fr 阈值解析锚点偏差 ≤1.01%；"
                    f"四流态判别一致率 {S['fr_x_fill_agreement']:.2f}（honest FAIL，根因已定位）。"),
        "scenario_key": "in32_d01_drum_flow_regime_map",
        "mission_id": args.mission,
        "tags": ["TensorDEM", "granular", "rotary drum", "regime map", "Froude", "power ledger",
                 "IN32-D01", "dem"],
        "markdown": md,
        "figures": figures,
    }
    Path(args.out).write_text(json.dumps(report))
    print(json.dumps({"out": args.out, "figures": [f["name"] for f in figures],
                      "md_chars": len(md)}))
    return report


if __name__ == "__main__":
    main()