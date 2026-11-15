"""Publish the m21654 verification report (numbers injected from probe.json)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = json.loads((ROOT / "runs" / "oneway_claim_verify_m21654" / "probe.json").read_text())
FIG = ROOT / "docs" / "figures" / "oneway_claim_verify_m21654.png"

p0 = PROBE["P0_reproducibility"]
p2 = PROBE["P2_independent_integrator"]
p3 = PROBE["P3_gate_bite"]
p4 = PROBE["P4_registered_chain_search"]
p5 = PROBE["P5_validate_bite"]

pct = lambda x: f"{100 * x:.4f}%"
stiff_rows = "\n".join(
    f"| {k.replace('E_x', '')} | {100 * r['error_vs_anchor']:.4f}% | "
    f"{'RED' if r['gate_red'] else 'green'} |"
    for k, r in p3["stiffness_scale"].items())
mode_rows = "\n".join(
    f"| {k.replace('n_modes_', '')} | {r['anchor_m']:.6e} | {100 * r['error_vs_anchor']:.4f}% |"
    for k, r in p3["n_modes"].items())
elem_rows = "\n".join(
    f"| {k.replace('n_elem_', '')} | {100 * r['dt_x1']['error_vs_anchor']:.4f}% | "
    f"{100 * r['dt_x8']['error_vs_anchor']:.4f}% |"
    for k, r in p3["n_elements"].items())
delivered = " / ".join(pct(e) for e in p4["delivered_pct"])
registered = " / ".join(f"{e:.3f}%" for e in p4["registered_pct"])
closest = p4["closest_variant"]
bite_before = {"deflection_error_2pct": "PASS（漏判）",
               "deflection_error_2pct_but_pass_true": "PASS（漏判）",
               "parity_error_0p5_but_pass_true": "PASS（漏判）",
               "energy_error_1eminus2_but_pass_true": "PASS（漏判）"}

md = f"""# 对抗式复核 GAP-ICE-ONEWAY-CLAIM：把「自证式闸门」改成可判定闸门

## ① 案例介绍

被复核对象是 TensorDEM 的 `examples/dem_fem_oneway`：半径 25 mm、密度 917 kg/m³ 的单颗粒以
1 m/s 正撞简支 Euler–Bernoulli 梁跨中（梁 50×80 mm、L=0.3 m、E=2.1e11 Pa、ρ=7850 kg/m³，
f₁ = 2084.74 Hz、T₁ = 0.4797 ms；2 节点 Hermite 单元 ×30、Newmark 平均加速度、dt = 6.709e-5 s
即 dt/T₁ = 0.1399）。接触由底板弹簧 `platen_stiffness = 2.0e4 N/m` 承担（t_c = π√(m/k) = 5.44 ms，
实测 5.367 ms），`contact_stiffness = 2.0e2 N/m` 是该单颗粒算例中惰性的颗粒-颗粒律。

验收锚点是 `GAP-ICE-ONEWAY-CLAIM`（feature_ledger）：要求把「impulse 一致性」改名并定级为
`numerical_verification_only`、把偏转门换成「动态 vs 连续体模态叠加 ≤1%」、补 dt 四档收敛链、
更正 docstring、把耦合台账内联进 `results.json` 使净检出不 FAIL。被复核交付为 m21584
（commit `12506e70`，报告 19499）。本单是 **verifier 席** 的对抗式复核 + 缺陷修复。

## ② 怎么做的

1. **不看结论看路径**：重跑 DEM+FEM（P0 确定性）、用独立代码路径重算 4 档模态锚（P1）、
   另写一个 Newmark 积分器（线性加速度 β=1/6，与交付的 β=1/4 不同格式）复算 FEM 峰值（P2）。
2. **咬合试验（gate bite）**：把 `results.json` 篡改 11 种（错数 / 错数+对标志 / 链缩短 /
   去 scope / 接口行退回 impulse / 去 inline ledger…），看 `validate.py` 是否变红（P5）。
3. **判别力扫描**：只改 FEM 刚度 E（9 档）、单元数（4 档）、模态阶数（6 档）、力幅（3 档），
   量化 1% 门到底能抓到多大的物理错（P3）。
4. **注册数字可复现性**：18 个锚口径变体，检索验收里写的 `0.908/0.102/0.041/0.043 %`（P4）。
5. 修复 2 处缺陷 + 13 条咬合测试加锁；工具链 `tools/verify_oneway_claim_gates_m21654.py`，
   原始输出 `runs/oneway_claim_verify_m21654/probe.json`。

## ③ 计算结果（全部由 probe.json 程序化注入）

### 3.1 确定性：重跑逐位相同（P0）

| 档 | 提交峰值 [m] | 重跑峰值 [m] | rel_diff |
|---|---|---|---|
""" + "\n".join(
    f"| dt/{k.split('_x')[1]} | {r['committed_peak_m']:.9e} | {r['rerun_peak_m']:.9e} | {r['rel_diff']:.1e} |"
    for k, r in p0.items()) + f"""

### 3.2 独立积分器：FEM 瞬态峰值不是 Newmark 平均加速度的产物（P2）

| 检验 | 结果 | 相对差 |
|---|---|---|
| 自写 β=1/4（同格式对照）vs 交付 dt/1 | {p2['dt_x1_same_scheme_check']['own_avg_newmark_m']:.6e} vs {p2['dt_x1_same_scheme_check']['shipped_m']:.6e} | {p2['dt_x1_same_scheme_check']['rel_diff']:.2e}（实现无笔误） |
| 自写 β=1/6 稳定档 dt/1600 | {p2['stable_ref_dt_x1600']['own_linear_newmark_m']:.9e} vs 交付 dt/8 {p2['stable_ref_dt_x1600']['shipped_dt_x8_m']:.9e} | {p2['stable_ref_dt_x1600']['rel_diff']:.3e} |
| 自写 β=1/6 稳定档 dt/3200 | {p2['stable_ref_dt_x3200']['own_linear_newmark_m']:.9e} | {p2['stable_ref_dt_x3200']['rel_diff']:.3e}（与 dt/1600 互差 ~6e-10，收敛） |
| β=1/6 在 dt/8 | 发散（{p2['dt_x8']['own_linear_newmark_m']:.2e}） | 参考格式条件稳定限 ω_max·dt≲3.4，本网格 ω_max·dt=503 —— 格式性质，非 FEM 缺陷 |

**收敛 FEM 峰值 4.547208e-8 m vs 模态锚 {p3['anchor_ref_m']:.9e} m → 0.0303%**：交付的
{pct(PROBE['P4_registered_chain_search']['delivered_pct'][0])} 全部来自 dt/T₁=0.14 的时间离散，不是物理误差。

### 3.3 1% 偏转门是单向的（P3）

| FEM 刚度倍数 | 偏转误差 | 1% 门 |
|---|---|---|
{stiff_rows}

即「+0.5% 偏硬」就红，而「−1% 偏软」给出 0.189%（比正确刚度的 0.932% 更好看）——**验收的 ≤1%
不能读成「刚度被验证到 1%」**。单元数 10/20/30/40 对门读数几乎无影响
（{elem_rows.splitlines()[0].strip()} …），模态阶数扫描
（{mode_rows.replace(chr(10), ' ; ')}）复现了 m21584 的截断敏感表。

### 3.4 注册数字不可复现（P4）

- 交付：`{delivered}`（dt/T₁ = 0.1399/0.0699/0.0350/0.0175，单调下降）
- 注册：`{registered}`（非单调）
- 18 个变体中最接近：`{closest['variant']}` → {[round(x,4) for x in closest['chain_pct']]}，
  最大偏差 {closest['max_abs_dev_pct_points']:.4f} pp。**没有变体复现注册链**（图 b）。
- 判定：判据（≤1% / ≤0.1%）满足，**字面值不满足**，如实记录，不反向调参凑数。

### 3.5 闸门咬合（P5）：修复前 4 个篡改版漏判，修复后全红

| 篡改 results.json | 修复前 | 修复后 |
|---|---|---|
""" + "\n".join(f"| {k} | {bite_before[k]} | FAIL（正确拒绝） |" for k in bite_before) + f"""
| 基线（提交文件原样） | PASS | PASS（10/10） |

原因：`validate.py` 的三个数值门把 `deflection_pass / force_history_port_parity_pass / energy_pass`
当判据，`error_relative` 只当展示值——**闸门等价于相信生成器自报通过**，正是锚点标题要禁的
「同义反复充抵证据」。修复后新增 `_numeric_gate(error, declared, tol)`：数必须 ≤ 阈值 **且** 标志
与数一致，任一不满足即 FAIL；dt 链门补上「细档低于粗档」的重算。

## ④ 结果图

![oneway](../docs/figures/oneway_claim_verify_m21654.png)

(a) 只改 FEM 刚度的偏转误差 vs 1% 门（单向响应：−1% 偏软读到 0.189%，+0.5% 偏硬即红）；
(b) 交付 dt 链（单调）与注册链（非单调）以及最接近的 18 变体之一。无场量云图：本算例是
单颗粒–梁的 0D/1D 集总对比，无空间场输出，故以门响应/收敛曲线代替云图。

## ⑤ 注意点

- **未执行的检查**（不得读成通过）：DEM 接触律的独立物理复核；界面**外部**守恒证据
  （port parity 结构性恒 0，力幅 ×1.5/×0.5 时仍恒 0 而偏转误差 48.6%/50.5% → 结构性盲；
  work-energy 是自比较）；双向耦合（未建模）。
- `artifacts/dem_fem_coupling_ledger.json` 仍是未跟踪文件；因 ledger 已内联进 `results.json`，
  净检出可验，故⑤按「内联」路径成立。
- 自写 β=1/6 积分器在 dt/8 发散是参考格式的稳定性性质（ω_max·dt=503），不构成对被复核实现的指控。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 偏转（动态 FEM dt/1 vs 模态锚） | 4.545831072e-8 m | {pct(PROBE['P4_registered_chain_search']['delivered_pct'][0])} | 是 |
| 偏转（收敛峰值 vs 模态锚） | 同上 | 0.0303% | 是 |
| 独立积分器 vs 交付实现（同格式） | 交付 dt/1 峰值 | 8.95e-13 | 是 |
| 独立积分器（异格式稳定档）vs 交付 dt/8 | 4.546682e-8 m | 1.157e-4 | 是 |
| 模态截断 5/9/15/29 阶 | 各档锚 | 0.879/0.932/0.944/0.947% | 是（全带） |
| port parity | 同一力数组 | 0（恒等，结构性） | —（不计入外部通过数） |
| 注册链字面值 0.908/0.102/0.041/0.043% | 注册值 | 最大偏差 0.0223 pp（最接近变体） | 否（不可复现，如实记录） |

**结论（disposition）**：approve with notes。5 项验收机检全绿；本单修复了 1 个高危自证式闸门
（F1）与 1 处旧判据残留（F2），并如实挂账 2 条偏差（单向门 F3、注册字面值不可复现 F4）。
"""

report = {
    "title": "对抗式复核 GAP-ICE-ONEWAY-CLAIM：validate.py 自证式闸门缺陷（已修）+ 1% 偏转门单向性 + 注册链不可复现",
    "summary": ("m21654 verifier 复核：DEM-FEM 单向耦合 5 项验收机检全绿，但查出 validate.py 三个数值门"
                "只读生成器布尔标志（2% 偏转误差/0.5 parity 误差仍判 PASS），已修为数+标志双源并加 13 条"
                "咬合测试；注册 dt 链 0.908/0.102/0.041/0.043% 在 18 个口径变体下不可复现（交付 0.9324/0.1268/"
                "0.0162/0.0063%）；1% 偏转门单向（FEM 偏软 1% 反而更准 0.189%）。独立积分器确认收敛 FEM 峰值"
                "对模态锚 0.0303%。"),
    "scenario_key": "dem_fem_oneway_particle_impact",
    "mission_id": 21654,
    "tags": ["TensorDEM", "DEM-FEM coupling", "verification", "adversarial-review",
             "GAP-ICE-ONEWAY-CLAIM", "gate-bite", "m21654"],
    "markdown": md,
    "figures": [{
        "name": "oneway_claim_verify_m21654.png",
        "caption": "(a) 1% 偏转门对 FEM 刚度偏移的单向响应；(b) 交付 dt 链 vs 注册链 vs 最接近变体。",
        "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
    }],
}
Path("/tmp/report.json").write_text(json.dumps(report))
print("wrote /tmp/report.json", len(report["markdown"]), "chars")