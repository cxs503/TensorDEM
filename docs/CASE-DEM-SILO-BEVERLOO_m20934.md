# CASE-DEM-SILO-BEVERLOO · 料仓卸料流率 Beverloo 律 — kn 敏感性验证（m20934）

**锚点**：`CASE-DEM-SILO-BEVERLOO`（总账状态：open；验收：|相对误差| < 10%，
ref: Beverloo et al. 1961, Chem. Eng. Sci. 15(3-4), 260-269, DOI 10.1016/0009-2509(61)80059-4）
**本单**：m20934（tp-gen-3 自找活，generalist 席）
**日期**：2026-11-14

## ① 案例介绍（物理背景 / 几何 / 参数）

平底圆柱料仓（flat-bottom silo）中心圆孔重力卸料。稳态质量流率遵循 Beverloo 律：

    W = C · ρ_b · √g · (D − k·d)^(5/2)

文献系数（球状颗粒/玻璃珠，光滑壁）：C = 0.58（0.55–0.60），k = 1.4（1.2–1.6），
取自原文 Eq.(4) p.265；机器可读参考 `benchmarks/m20055_beverloo/ref_beverloo_1961.json`
（零手抄，算例脚本只从该文件读系数）。

**前史（防重复声明）**：

| 单 | 几何 | kn [N/m] | D/d | 误差 | 状态 |
|---|---|---|---|---|---|
| m20055 | 平底料仓 `silo_discharge` | 1e5 | 4 | **+25.0%** | honest FAIL（仅 D4 一算例） |
| m15259 | hopper 内核 `hopper_discharge` | 1e6 | 5 | **−0.4%** | PASS |
| m20585 | 诊断单（集群作业 SSH 超时，零计算产出） | — | — | — | 假设「kn=1e5 软接触是系统误差主因」从未被计算验证（见 `verdict_m20585.json`） |

**本单问题**：在锚点几何（平底料仓）上，把接触刚度从 1e5 提到 1e6 是否能把
Beverloo 符合度拉进 <10%？并补齐 m20055 从未跑过的 D5/D6 算例。

**几何与材料（与 m20055 逐字节一致，唯一变量是 kn；dt/2 臂另见 §②）**：

| 量 | 值 | 无量纲 |
|---|---|---|
| 颗粒直径 d | 0.020 m | — |
| 颗粒密度 ρ_p | 2500 kg/m³ | — |
| 料仓半径 | 0.130 m（仓径 0.26 m） | 仓径/D = 2.6–3.25 |
| 填充高度 | 0.40 m | H/D = 3.3–5.0 |
| 孔径 D | 0.080 / 0.100 / 0.120 m | D/d = 4 / 5 / 6 |
| 底板厚 | 0.010 m（0.5 d） | — |
| 颗粒摩擦 μ | 0.30 | — |
| 滚阻 μ_r | 0.02 | — |
| 恢复系数 e | 0.30 | — |
| 壁面摩擦 μ_w | 0.0（光滑壁，对应 Beverloo 玻璃珠实验） | — |
| 床层 | 无序压实床（loose gap 0.30r，jitter 0.05r，seed 20055，压实期堵孔 0.6 s） | — |
| 稳态窗口 | 卸料流质量分位 [0.2, 0.8]，5 子窗 | — |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **算例矩阵**（预声明，`benchmarks/m20934_beverloo_kn/run_kn_sweep.py`）：
   D/d ∈ {4, 5, 6} × kn ∈ {1e5, 3e5, 1e6}，共 9 臂，外加 D5/kn=1e6 的
   **dt/2 臂**（时间步伪影守卫，共 10 臂）。除 kn（及 dt/2 臂的 dt）外，
   几何/材料/摩擦/压实协议/种子与 m20055 完全相同。
2. **求解器**：`tensordem.silo_discharge`（REAL 3-D DEM：线性弹簧-阻尼法向 +
   Cundall-Strack 切向 + 滚阻，`granular3d_core` 内核；孔口=带圆孔平板，
   顶面/底面/边缘柱/双锐缘圆五特征最近接触）。时间步 = 最硬弹簧周期/20。
3. **稳态测量**：开孔后按卸料累计质量分位 [0.2, 0.8] 取中段，切 5 子窗，
   均值±1σ；CV 作稳态性检查。ρ_b 取**开孔瞬间实测床值**（Beverloo 的 ρ_b
   是床密度，不是初始装填密度）。
4. **集群执行**：10 分片并行（run_compute cj3073，stage_dir=本工作树，
   每片 1 核，loong64 节点由调度器指派）。
5. **分析**：`analyse.py` 汇总 `summary.json` + 3 图；`render_doc.py` 把
   表格程序化注入本文（零手抄）。
6. **回归测试**：`tests/test_silo_beverloo_kn_sweep.py`（6 项，含 kn=1e6
   小算例冒烟：有限性 + 实际卸料 + 壁内约束）。

## ③ 计算结果（程序化注入）

<!-- TABLES:BEGIN -->
（待集群结果回填）
<!-- TABLES:END -->

## ④ 结果图

- `runs/m20934_beverloo_kn/fig_w_vs_dd.png` — W vs D/d，各 kn 的 DEM 点 vs Beverloo 曲线
- `runs/m20934_beverloo_kn/fig_err_vs_kn.png` — 相对误差 vs kn（log 轴，±10% 验收带）
- `runs/m20934_beverloo_kn/fig_cumulative_D5.png` — D5 卸料动力学，kn=1e5 vs 1e6

（DEM 为颗粒级模拟，无连续场量云图；本算例的关键量是标量流率 W，
比较对象为 Beverloo 解析关联，故以比较曲线代替云图。）

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **kn 是 DEM 的数值参数，不是材料参数**：真实玻璃珠的弹性模量 ~70 GPa，
  对应 kn 远大于 1e6 N/m；kn 扫描回答的是「接触刚度对有效孔口的影响是否
  主导系统误差」，不是「哪个 kn 最物理」。
- Beverloo 律适用范围 D/d ≳ 4、仓径/D ≥ 2.5、床高/D ≥ 1.5；本矩阵全部在域内。
- 1961 原文 PDF 不在仓库、环境无外网：C/k 为引用值（cited, not re-digitized），
  与 m20055 相同的残余风险。
- 小体系泊松噪声：窗口内计数量有限，CV 含物理起伏与计数噪声。
- 单种子（seed 20055）：统计不确定度未做多种子重复（m20055 同协议）。

## ⑥ 误差判定（逐项 vs 基准）

<!-- VERDICT:BEGIN -->
（待集群结果回填）
<!-- VERDICT:END -->

**总判定**：<!-- VERDICT:SUMMARY -->（待回填）

## 工件与复现

- 算例脚本：`benchmarks/m20934_beverloo_kn/run_kn_sweep.py`（`--list` 看矩阵）
- 分析/绘图：`benchmarks/m20934_beverloo_kn/analyse.py`
- 文档渲染：`benchmarks/m20934_beverloo_kn/render_doc.py`
- 原始数据：`runs/m20934_beverloo_kn/*.json`（含 git sha、host、dt、窗口序列、卸料事件序列）
- 汇总：`runs/m20934_beverloo_kn/summary.json`
- 一键复现（单臂）：
  `PYTHONPATH=src python benchmarks/m20934_beverloo_kn/run_kn_sweep.py --arm D5_kn1e6`
- 回归测试：`PYTHONPATH=src python -m pytest tests/test_silo_beverloo_kn_sweep.py -q`
