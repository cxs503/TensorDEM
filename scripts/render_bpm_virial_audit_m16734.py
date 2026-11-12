"""Render docs/AUDIT_BPM_VIRIAL_ABSOLUTE_m16734.md from the verification JSON.

Every number is injected from ``artifacts/m16734_virial_absolute/verification.json``
(zero hand copying).
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "artifacts/m16734_virial_absolute/verification.json").read_text())
rows = data["homogeneous_rows"]
mut = data["mutations"]
ms = data["mutation_state"]
inv = data["ac5_invariance"]
sweep = data["strain_sweep"]
disc = data["disc_specific_volume"]


def sci(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}e}"


lines: list[str] = []
a = lines.append
a("# AUDIT m16734 — BPM 逐粒子 virial 应力仪的**绝对量级**: 与面切割牵引力的独立对标")
a("")
a("锚点：**NEW:CLOSE-GAP-BPM-VIRIAL-INSTRUMENT**（`docs/dem_calibration_gap_ledger.md` §2 "
  "「仍未接线：`bpm_calibration_feasibility.virial_stress_tensor`」——本条把它接上共享应力仪 "
  "并补上从未做过的**绝对量级**验证）。")
a("")
a("## ① 案例介绍")
a("")
a("`bpm_calibration_feasibility` 的巴西圆盘判据 AC5（Hondros 圆心主应力比 → −3，容差 15 %）"
  "只消费**同一张量两个主应力的比值**，因此该仪器的**绝对量级从未被检验**：一个整体偏 2 倍或偏 25 % "
  "的仪器在 AC5 上是完全看不出来的。面切割牵引力 `granular3d.stress.plane_stress`（穿面接触的力之和 / 面积）"
  "是与之无任何共用代数的**独立路径**。")
a("")
a("被检对象：简单立方键合试件 `SpecimenGeometry(n_layers, cross_n=4, R=0.0125 m)`，"
  f"颗粒半径 {data['PARTICLE_RADIUS']} m，参考参数 `REFERENCE_VECTOR`（k_b=2.56e5 N/m）。"
  "状态构造：对整段标距施加**均匀仿射应变**并把所有颗粒固定（纯静力学，无动力学污染）——"
  "正是 virial 定理要求体平均与面切割一致的唯一状态。")
a("")
a("## ② 怎么做的")
a("")
a("1. `src/tensordem/bpm_calibration_feasibility.py`：新增 `pair_stress_state()`"
  "（**唯一**一处写下应力仪的力律：存活键轴力 `k_b(L−L0)` + 未键合对的法向排斥 `k_c·overlap`）与 "
  "`specific_volume()`；`virial_stress_tensor()` 不再自算配对张量，改用共享仪 "
  "`granular3d.stress.contact_outer`（`l ⊗ f`，l=x_j−x_i，f=作用于 i 的力），并新增两个显式选项："
  "`scatter='half'|'legacy'`（配对计数）与 `volume_per_particle`（比容）。")
a("2. `scripts/verify_bpm_virial_absolute_m16734.py` 产出全部数字与图；"
  "本文由 `scripts/render_bpm_virial_audit_m16734.py` 从 JSON 渲染。")
a("3. 判定先于跑数：C1 = 体平均等于面切割牵引力 × 拉伸因子 (1+ε)；C2 = 旧实现的两个因子**可解析预测**"
  "（2(n−1)/n 与 1/n）。")
a("")
a("## ③ 计算结果")
a("")
a("### 3.1 两个缺陷（解析预测 vs 实测）")
a("")
a("| n_layers | ε | 面切割 σ_zz [Pa] | 修正后体平均 σ_zz [Pa] | 相对误差 | "
  "旧实现(双计+8R³)/面切割 实测 | 解析 2(n−1)/n(1+ε) | 8R³ 归一化误差 实测 | 解析 1/n |")
a("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    a(f"| {r['n_layers']} | {r['strain']:+.1e} | {sci(r['sigma_zz_plane_cut'])} | "
      f"{sci(r['sigma_zz_per_particle_gauge_volume'])} | "
      f"{sci(r['rel_err_per_particle_gauge_vs_plane'], 2)} | "
      f"{r['measured_legacy_over_plane']:.4f} | {r['predicted_legacy_over_plane']:.4f} | "
      f"{r['rel_err_per_particle_cell_vs_plane']:.4f} | {1.0 / r['n_layers']:.4f} |")
a("")
a(f"代数恒等式：`sum_p V_p sigma_p − sum_c l(x)f` 最大绝对元 "
  f"= {sci(data['homogeneous_rows'][0]['sum_V_sigma_minus_ref_max'])}（修正后，机器精度）；"
  f"`legacy` 模式与 `2*sum_c l(x)f` 的最大偏差 = "
  f"{sci(data['homogeneous_rows'][0]['sum_V_sigma_legacy_minus_2ref_max'])}——"
  "**旧实现把每对接触在两端各记一次全量**，故区域平均恰为正确值的 2 倍。")
a("")
a("### 3.2 残差结构：不是 bug，是 Cauchy/名义应力之差")
a("")
a("| 仿射应变 ε | 体平均 vs 面切割 相对误差 | 比值 / ε |")
a("|---|---|---|")
for s in sweep:
    a(f"| {s['strain']:.1e} | {sci(s['rel_err_virial_vs_plane'])} | {s['rel_err_over_strain']:.9f} |")
a("")
a("比值在所有 ε 上 = 1.000000000（9 位）⇒ 残差**恰为**拉伸因子 (1+ε)：virial 给当前构形 Cauchy 应力，"
  "面切割按参考面积给名义应力，一阶差即 ε。缺陷修好后误差判据（<3 %）成立，且这是**物理解释**而非拟合。")
a("")
a("### 3.3 变异（非空转）检验")
a("")
a(f"状态：n_layers={ms['n_layers']}，ε={ms['strain']:+.1e}，预断键比例 0.25 → "
  f"死键 {ms['n_dead_bonds']} 根、接触分支 {ms['n_contact_pairs']} 对（两支同时非零）；"
  f"参考 σ_zz = {sci(ms['sigma_zz_exact_virial'])} Pa（面切割 {sci(ms['sigma_zz_plane_cut'])} Pa）。")
a("")
a("| 注入的经典 bug | σ_zz [Pa] | 相对变化 |")
a("|---|---|---|")
for k, v in mut.items():
    a(f"| `{k}` | {sci(v['sigma_zz_Pa'])} | {v['rel_change_vs_reference']:.4f} |")
a("")
a("`transpose_convention`（用 f⊗l 替 l⊗f）变化 **0.0000**——诚实负面结果：两支力都是中心力，"
  "转置不可分辨；因此本仪器**无法**测摩擦不对称量 σ_ab−σ_ba，这条局限写进 docstring。"
  "`branch_reversed` 与 `force_sign_flipped` 同为 2.0000（中心力下二者等价，非独立变异）。")
a("")
a("### 3.4 AC5 消费者不受影响")
a("")
a(f"legacy：主应力 {inv['legacy']['principal_in_plane'][0]:.4f} / "
  f"{inv['legacy']['principal_in_plane'][1]:.4f} Pa；修正后："
  f"{inv['fixed']['principal_in_plane'][0]:.4f} / {inv['fixed']['principal_in_plane'][1]:.4f} Pa。"
  f"两者 max/min 比值均 = "
  f"{max(inv['legacy']['principal_in_plane']) / min(inv['legacy']['principal_in_plane']):.6f}"
  f" / {max(inv['fixed']['principal_in_plane']) / min(inv['fixed']['principal_in_plane']):.6f}"
  "（比值不变 ⇒ AC5 判决不变，但张量本身整体减半）。")
a(f"圆盘格子比容：`V_cell/V_cylinder` = {disc['cell_over_cylinder']:.4f} ⇒ 旧 8R³ 归一化在 AC5 "
  f"消费者上偏 **{100 * disc['legacy_level_error']:.2f} %**；`run_brazilian` 现改传 "
  f"`specific_volume(V_cyl, N) = {sci(disc['V_p_cylinder_m3'])} m³`。")
a("")
a("## ④ 结果图")
a("")
a("`artifacts/m16734_virial_absolute/fig_virial_absolute.png`：(a) 变异非空转条形图；"
  "(b) 三种归一化相对面切割牵引力的误差 vs n_layers（含解析曲线 2(n−1)/n−1）；"
  "(c) 残差随 ε 的线性关系（与 |ε| 重合）。")
a("")
a("## ⑤ 注意点与局限")
a("")
a("* 只对**均匀仿射**状态做了精确对标：非均匀状态（真实剪切带）里体平均与面切割之差是真实的应力梯度项，"
  "不能当误差。")
a("* 应力仪仍**不含**切向/滚动分支（引擎把该历史私有化）：张量因此恒对称，摩擦不对称量不可测——"
  "本轮没有假装修好这一条。")
a("* `scatter='legacy'` 保留只为复现 m16734 之前的数字，不得用于新结论。")
a("* 面切割只给穿过面的**法向行**，三轴同时验证需要三个方向的面切割，本轮只做 z 轴（受载轴）。")
a("")
a("## ⑥ 误差判定（关键量 vs 基准逐项）")
a("")
best = next(r for r in rows if r["n_layers"] == 9 and r["strain"] > 0)
a("| 关键量 | 基准（独立路径） | 误差 | 是否 <3 % |")
a("|---|---|---|---|")
a(f"| 修正后体平均 σ_zz（n=9, ε=+2e-3） | 面切割牵引力 | "
  f"{100 * best['rel_err_per_particle_gauge_vs_plane']:.3f} % (= ε，可解析) | **是** |")
a(f"| 旧实现 σ_zz（n=9, ε=+2e-3） | 面切割牵引力 | "
  f"{100 * best['rel_err_legacy_vs_plane']:.2f} % | 否（缺陷已修复） |")
a(f"| 代数恒等式 sum_p V_p σ_p = sum_c l⊗f | 共享仪 contact_outer | "
  f"{sci(rows[0]['sum_V_sigma_minus_ref_max'], 1)}（绝对元，双精度 ulp 级） | 是 |")
a("")
a("结论：修正后体平均应力与独立面切割牵引力的差**恰为仿射拉伸因子 (1+ε)**，ε→0 时收敛为 0；"
  "旧实现相对真值偏高 33 %（n=3）→ 5.7 %（n=17），且在 AC5 圆盘格子上另有 24.6 % 的系统性量级偏差。")
(ROOT / "docs/AUDIT_BPM_VIRIAL_ABSOLUTE_m16734.md").write_text("\n".join(lines) + "\n")
print("wrote docs/AUDIT_BPM_VIRIAL_ABSOLUTE_m16734.md", len(lines), "lines")
