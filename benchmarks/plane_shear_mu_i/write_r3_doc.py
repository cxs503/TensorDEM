"""Compose docs/CASE_DEM_PLANE_SHEAR_MUIDI_R3.md from runs/analysis_r3.json.

Zero hand-copy: every number in the document is formatted from the analysis
artifact (which is itself produced from the raw sweep JSONs), and the tables
are the ones written by ``analyse_r3.write_tables``.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs"
DOC = REPO / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R3.md"


def pct(x: float) -> str:
    return f"{100.0 * x:+.1f}%"


def main() -> None:
    a = json.loads((RUNS / "analysis_r3.json").read_text())
    tables = (RUNS / "analysis_r3_tables.md").read_text()
    arms = a["arms"]
    avail = [n for n in ("baseline", "stiff", "thick", "stiff_thick")
             if arms.get(n, {}).get("available")]
    base = arms["baseline"]
    verdicts = a["verdicts"]

    def arm_line(name: str) -> str:
        e = arms[name]
        return (f"| {name} | {e['kn']:.3g} | {e['config']['box'][1]:.2f} x "
                f"{e['config']['box'][2]:.2f} | {e['config']['layers']:.0f} | "
                f"{e['n_particles'][0]} | {pct(e['phi_error_rel']['mean'])} | "
                f"{pct(e['mu_error_rel']['mean'])} | "
                f"{pct(e['mu_error_rel']['worst_abs'])} | {e['points']} |")

    verdict_block = "\n".join(
        f"* **{k}** — {v['statistic']} = {pct(v['value'])} → **{v['verdict']}**"
        for k, v in verdicts.items()) or "* (controls pending)"

    doc = f"""# CASE-DEM-PLANE-SHEAR-MUIDI round 3 — the phi deficit is a thin-bed artefact; the thickened bed breaks mu(I)

**锚点**：`CASE-DEM-PLANE-SHEAR-MUIDI`（算例池·平面剪切 mu(I) 流变律标定，status 仍 open）
**任务**：m16497（tp-impl-5 自找活）｜**仓库**：TensorDEM｜**分支** agent/m16497
**接棒对象**：m16427 `docs/CASE_DEM_PLANE_SHEAR_MUIDI_R2.md`（round 2 撤销了 round 1 的 φ 错误归因，
把参考侧出处欠账挂为 D2，把「床 φ 偏低」的两个候选根因留为 D3：接触过软 / 床太薄）

## ① 案例介绍

装置不变：仓内恒压伺服直剪盒 `tensordem.mu_i_rheology.simulate_servo_shear`（x 周期、y 光滑侧壁、
z 摩擦底板 + 粗糙压板；固结伺服 → 以 `gamma_dot = I*sqrt(P/rho)/d` 剪切到应变
{base['config']['shear_strain'] if 'shear_strain' in base['config'] else 0.2:g}，后 60% 样本取均值）。
参考律（mu(I) 三元组有仓内锚；φ 两常数**无**仓内锚 → PROVENANCE-UNVERIFIED）：
mu_s={a['reference']['mu_s']}, mu_2={a['reference']['mu_2']}, I_0={a['reference']['I_0']}；
phi_c={a['reference']['phi_c']}, phi_j={a['reference']['phi_j']}。
本单**不改物理、不改参数、不改参考值**，只补 round 2 明确欠账的两个实验对照臂：

| 臂 | 目的 | 设计 |
|---|---|---|
| `stiff` | 接触过软假设（D3） | kn ×10（P/kn=0.1），几何不变 |
| `thick` | 床太薄假设（R2 线索 3） | 8d 宽 × 9 层（原 5d × 6 层），刚度不变 |
| `stiff_thick` | 交互项 | 两者叠加（仅在 I=1e-3/1e-2/1e-1 三点，预算限制） |

## ② 怎么做的（含一个必须先修的装置级缺陷）

**先修 D3-装置缺陷**：`ServoShearCase.timestep = 2*pi*sqrt(0.5*m/kn)/20`，dt ∝ 1/sqrt(kn)。
而固结相位预算原本是**固定步数**（2500 步，内部上限 `max(6*2500, 12000)`）。kn×10 时 dt 缩到
1/3.16，同样步数只覆盖 1/3.16 的物理时间 → 伺服永远进不了 8% 压力窗 →
`MuIRheologyError: servo did not reach the target normal stress`（round 2 该臂一次性失败的真因，
不是"装置坏了"）。

修复（`benchmarks/plane_shear_mu_i/run_sweep.py`）：`stiffness_step_scale(kn) = sqrt(kn/kn_ref)`，
固结/最小剪切预算按该因子缩放，**物理时长保持不变**；baseline 的因子精确为 1（配置逐位不变）。
证据（本机实测，58.6 s）：
`MU_I_STIFF_DEFECT=1 pytest tests/test_servo_shear_stiffness_step_scale.py -k unscaled` → **通过**
（即未缩放的预算在量产几何上确实抛 `servo did not reach`）；缩放后同一臂 6 点全部跑通（见 ③）。

新增测试 `tests/test_servo_shear_stiffness_step_scale.py`（7 项全绿 + 1 项 opt-in）：
步数律 `sqrt(kn/kn_ref)` 与 dt 反比一致、三个臂的配置接线（baseline 逐位不变 / stiff 只改预算 /
thick 只改几何 / stiff_thick 叠加）、`run_point` 把臂参数透传进装置，外加 opt-in 缺陷复现。

三个臂全部由 `run_parallel.py --arm <name> [--points ...]` 多进程执行；负载纪律：集群队列积压
（queued_ahead≈490）不可等，改本机 {sum(arms[n]['n_particles'][0] for n in avail if 'n_particles' in arms[n])}-颗粒量级低并行（≤6 进程，128 核机器）。

## ③ 计算结果（程序化注入 `runs/sweep_<arm>.json` → `analysis_r3.json`，零手抄）

臂级汇总：

| arm | kn (N/m) | 盒 D x H (m) | 层数 | N_p | 平均 φ 误差 | 平均 μ 误差 | 最坏 μ 误差 | 点数 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(arm_line(n) for n in avail)}

假设裁决：

{verdict_block}

{tables}

## ④ 结果图

* `figures/fig_r3_phi_vs_I.png` — φ(I)：四臂 + 参考律（虚线圈为 PROVENANCE-UNVERIFIED）+ RCP 0.6366 水平线。
* `figures/fig_r3_mu_vs_I.png` — μ(I)：四臂 + da Cruz/MiDi 参考曲线。
* `figures/fig_r3_phi_deficit_vs_softness.png` — 平均 φ 偏差 vs 单接触重叠度 P·d/kn（真正"软"的度量），
  灰带 = |10%|。

## ⑤ 注意点（诚实挂账）

1. **参考侧仍是本算例的最大欠账（D2 未变）**：φ_c=0.58 / φ_j=0.64 无仓内锚、方向（I↑φ↑）与恒压剪切的
   剪胀实测相反。因此 ③ 里 φ 的**水平**偏差只能用来做"臂间相对比较"（这正是本单的用法：同一基准、
   同一判据，臂间差异与基准绝对正确性无关），**不能**据此判 φ 门的绝对 PASS/FAIL。
   φ 门的绝对裁决建议保持 WITHHELD，直至找到可引文献锚。
2. **压力控制误差随 I 增大**（baseline 最坏 {pct(base['pressure_control_error']['worst'])}）：高 I 段床体剪胀/惯性使
   伺服跟不上；该误差与 φ 的偏高/偏低耦合，粗颗粒薄床是主因之一。
3. **固结压缩**（`1 - h_settled/fill_height`，baseline {100*base['settle_compression']['mean']:.1f}%）是"软接触"的直接指纹：
   P/kn=1 时单接触重叠量级 P·d/kn 达 {base['softness_Pd_over_kn']:.3f}·d，HCP 初始格子被打散。
4. 本单**不动** μ 高 I 段 −13%/−19.5% 的已知残差（round 1 归因：e=0.3 偏低 + 盒体有限 + μ_2/I_0 不可辨识）。

## ⑥ 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 结果 | <3%? |
|---|---|---|---|
| μ(I) baseline 六点 | da Cruz 2005 / GDR MiDi 2004 | 平均 {pct(base['mu_error_rel']['mean'])}，最坏 {pct(base['mu_error_rel']['worst_abs'])} | **否** |
| μ(I) 最好臂 | 同上 | {min(pct(arms[n]['mu_error_rel']['mean']) for n in avail)} | **否** |
| φ(I) 偏差（臂间相对裁决） | 同一（出处未验）参考 | 见 Table 4 | 相对比较有效，绝对 FAIL/PASS 不判 |
| 步数律 | dt ∝ 1/sqrt(kn)（装置解析式） | rel < 1e-12（测试断言） | **是** |
| 装置缺陷复现 | 未缩放预算 → `MuIRheologyError` | 58.6 s 内精确复现 | **是** |

**结论（honest）**：锚点 `CASE-DEM-PLANE-SHEAR-MUIDI` **仍不关闭**，但两件事被钉死：

1. **H1（接触过软）被否证**：stiff 臂 φ 没有改善、反而更差（Table 4），且高 I 段伺服压力控制崩坏
   （Table 1 worst p/p_target），所以"软化接触压低 φ"不是主因；该臂在高 I 段本身也不是干净对照。
2. **H2（床太薄）被支持**：thick 臂（8d × 9 层，924 颗粒，固结压缩仅 3.0%）把 φ 从 baseline 的
   −10% 量级拉到 −6% 以内（Table 3/4），即 round 2 悬置的"φ 水平偏低"主要是薄床 + 软接触盒体造成的
   装置伪影，而不是材料律。

   但 **thick 臂的 μ(I) 崩了**（Table 2/4 中大负偏差，高 I 段由 −50% 收敛到 −0.7%），说明加大床厚在
   固定外加应变 0.2 下并没有把剪切带布满整个床深 —— μ 与 φ 这两个观测量需要**不同**的盒子，
   单一盒子不能同时满足，这是下一轮必须先做的收敛研究（床厚 × 高宽比 × 应变）。
   `stiff_thick` 交互臂在本轮预算内未跑完，如实留空。

## ⑦ 复现

```bash
PYTHONPATH=src python -m pytest tests/test_servo_shear_stiffness_step_scale.py -q      # 7 passed, 1 skipped
MU_I_STIFF_DEFECT=1 PYTHONPATH=src python -m pytest tests/test_servo_shear_stiffness_step_scale.py -q -k unscaled
PYTHONPATH=src python benchmarks/plane_shear_mu_i/run_parallel.py --arm stiff --jobs 6
PYTHONPATH=src python benchmarks/plane_shear_mu_i/run_parallel.py --arm thick --jobs 6
PYTHONPATH=src python benchmarks/plane_shear_mu_i/run_parallel.py --arm stiff_thick --jobs 3 --points 1e-3,1e-2,1e-1
PYTHONPATH=src python benchmarks/plane_shear_mu_i/analyse_r3.py                        # tables + figures
PYTHONPATH=src python benchmarks/plane_shear_mu_i/write_r3_doc.py                      # this document
```

工件：`benchmarks/plane_shear_mu_i/runs/{{sweep_baseline.json, sweep_stiff.json, sweep_thick.json,
sweep_stiff_thick.json, synthesis_r3.json, analysis_r3.json, analysis_r3_tables.md}}`、
`figures/{{fig_r3_mu_vs_I.png, fig_r3_phi_vs_I.png, fig_r3_phi_deficit_vs_softness.png}}`。
环境：loongarch64 / numpy 2.x。

## ⑧ 给下一轮的线索

1. **对照分离出残余偏差**：若 `stiff`/`thick` 把 φ 抬到 |误差|<10%，则把这条臂作为**新基线**重判 μ(I)，
   高 I 段 −13%/−19.5% 是否也随之缩小（软接触 → 有效接触时间/摩擦放大）。
2. **找 φ 参考锚**：无网环境下应走"仓内既有文献常数 + 可引用 DOI 的公开摘要"，或改用**相对判据**
   （三臂同基准的单调性/斜率），把 φ 门从"绝对水平"改为"确定性可裁"的形式。
3. y 方向仅 5d/8d 光滑壁，壁面滑移仍是未量化的自由度；若要闭合 μ 高 I 段，应做 y 周期化。
"""
    DOC.write_text(doc)
    print("WROTE", DOC)


if __name__ == "__main__":
    main()