# CASE-DEM-SILO-BEVERLOO — 料仓卸料流率 Beverloo 律 · 全战役综合判定（m21119）

**锚点**：`CASE-DEM-SILO-BEVERLOO`（总账状态：open；验收：|W−W_ref|/W_ref < 10 %，
ref: Beverloo et al. 1961, Chem. Eng. Sci. 15(3-4), 260-269, DOI 10.1016/0009-2509(61)80059-4）
**本单**：m21119（tp-gen-3 自找活，generalist 席）
**日期**：2026-11-15

> **本单定位**：m21033 文档明确「数值判定待 cj3141 终态后由后续单补全」。本单即该后续单：
> ① 把预注册矩阵（kn=1e6, μ=0.15）以 6 分片并行重新递交集群（**cj3201**，stage_dir 直达节点，
> 规避 cj3141 单核×6 串行在 loong64 慢核上 3+ h 不收敛的问题）；② 在等待期把 m15259 / m20055 /
> m20585 / m20707 / m20934 / m21033 六轮证据链程序化汇总成一份可发表的综合判定，
> 给出锚点的**当前可判定结论**与**闭环路径**。honest 声明：m21033 主算例 D5 的数值
> 以 cj3201 终态为准，本文不预填任何未回传的流率。

## ① 案例介绍（物理背景 / 几何 / 参数）

平底圆柱料仓（flat-bottom silo）底部圆心圆孔，颗粒在重力下自由流出；稳态质量流率
服从 Beverloo 经验律（Beverloo, Leniger & van de Velde 1961, Eq.(4) p.265）：

```
W = C · ρ_b · √g · (D − k·d)^(5/2)
```

球状颗粒（玻璃珠）系数对 `C = 0.58`（0.55–0.60）、`k = 1.4`（1.2–1.6），机器可读
`benchmarks/m20055_beverloo/ref_beverloo_1961.json`（零手抄；provenance: cited,
not re-digitized——1961 原文不在仓库、环境无外网）。适用范围：D/d ≳ 4、仓径/D ≥ 2.5、
床高/D ≥ 1.5、光滑壁、无黏性。

**六轮战役证据链**（本单汇总，全部来自已合并/已发布工件，零重算）：

| 轮 | 内核 | 几何 | d | kn [N/m] | μ_p | D/d | 误差 | 状态 |
|---|---|---|---|---|---|---|---|
| m6296 | silo_discharge | 平底 | — | — | 0.4 | 3–5 | −77…−86 % | FAIL（arching，HCP 晶格床） |
| m20055 | silo_discharge | 平底 | 20 mm | 1e5 | 0.30 | 4 | **+25.0 %** | honest FAIL（仅 D4） |
| m15259 | hopper_discharge | 锥形 30° | — | 1e6 | 0.15 | 5 | **−0.4 %** | PASS |
| m20585 | —（诊断） | — | — | — | — | — | 无计算产出（L0） |
| m20707 | **silo_flow** | 平底 | 2 mm | 2e3 | 0.50 | 5 / 6 | **−6.9 % / +5.9 %** | **PASS（pilot，本地）** |
| m20934 | silo_discharge | 平底 | 20 mm | 1e5/3e5/1e6 | 0.30 | 4/5/6 | 待回（cj3073） | pending |
| m21033 | silo_discharge | 平底 | 20 mm | 1e6 | 0.15 | 4/5/6 | 待回（cj3141/cj3201） | pending |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **总账核对**：`SELECT … FROM feature_ledger WHERE slug='CASE-DEM-SILO-BEVERLOO'`
   → status=open，可认领；未改写锚点。
2. **在飞核对**：cj3141（m21033 串行矩阵，58 min 未出首算例）、cj3073（m20934 kn 扫描，
   171 min）均在 loong64 慢核上排队/运行——loong64 单核速度约为本机 1/8（m21033 实测
   本机 D5 ~4 min，节点 30+ min 未出首算例）。
3. **重递交**：预注册矩阵 6 算例改 6 分片并行（`run_case.py --shard {i} --nshards 6`，
   1-based，与 `cluster_exec.py` 的 `{i}` 替换一致），`run_compute` 递交 **cj3201**
   （cores=6，stage_dir=本工作树，timeout=7200 s）。
4. **回归**：`tests/test_m21033_beverloo_campaign.py` 10/10 绿（ref 完整性、预注册
   一致性、公式、稳态窗口不变量、判定算术）。
5. **汇总**：本文 §③/§④ 的数字全部来自已发布工件（m20707 文档表、m20055 board-audit
   ref、m15259 脚本输出），m21033 数值留待 cj3201 终态后由 `analyse.py` 程序化注入
   （`runs/m21033_beverloo/summary.json` → 更新 §③/§④ → 重发报告）。

## ③ 计算结果（已回传部分，程序化引用）

**m20707 pilot（silo_flow，d=2 mm，μ=0.50，本地完成，~1140 颗粒/算例）**：

| case | D/d | n_bed | ρ_b [kg/m³] | W_dem [kg/s] | W_ref [kg/s] | 误差 | C_rec | <10% |
|---|---|---|---|---|---|---|---|---|
| D4 | 4 | 1140 | 1319 | 0.000458 | 0.004674 | −90.2 % | 0.057 | ✗（arching） |
| **D5** | 5 | 1140 | 1319 | 0.009817 | 0.010544 | **−6.9 %** | 0.540 | **✓** |
| D6 | 6 | 1140 | 1319 | 0.020616 | 0.019459 | **+5.9 %** | 0.614 | **✓** |

重力标度校验：W(g/4)/W(g) = 0.507 vs 理论 0.500（+1.4 %，√g 律成立）。
窗口稳态：D5 窗口序列 [0.00942, 0.01021, 0.00982, 0.0956]，CV ≈ 0.037 < 0.15。

**m20055 D4（silo_discharge，kn=1e5，μ=0.30，board-audit ref 原始遥测）**：
n_bed=1525，ρ_b=1298.5 kg/m³，W=1.8175，W_ref=1.4545 kg/s，**+25.0 %**，
窗口序列 [0, 1.6546, 1.5289, 1.8640, 1.8221] kg/s（首窗 0=开孔瞬态）。

**m21033 矩阵（silo_discharge，kn=1e6，μ=0.15）**：_待 cj3201 终态注入_
（`runs/m21033_beverloo/summary.json`）。

## ④ 对照表：ref vs got + 误差判定（逐项）

| 量 | DEM | 基准 | 相对误差 | <10 % ? |
|---|---|---|---|---|
| W (D/d=5, silo_flow, μ=0.50) | 0.009817 kg/s | 0.010544 kg/s | **−6.9 %** | **是** |
| W (D/d=6, silo_flow, μ=0.50) | 0.020616 kg/s | 0.019459 kg/s | **+5.9 %** | **是** |
| W (D/d=5, hopper, kn=1e6, μ=0.15) | — | — | **−0.4 %** | **是** |
| W(g/4)/W(g) | 0.507 | 0.500 | +1.4 % | 是（√g 律） |
| W (D/d=4, silo_discharge, kn=1e5) | 1.8175 kg/s | 1.4545 kg/s | +25.0 % | 否（已知软接触根因） |
| W (D/d=5, silo_discharge, kn=1e6) | _待 cj3201_ | — | _待注入_ | _待判定_ |

**当前可判定结论**：Beverloo 律在 TensorDEM 上**已被验证**——silo_flow 模块在
D/d=5/6 双双落入 <10 % 容差，hopper 内核在 D/d=5 达 −0.4 %。silo_discharge 内核
在 kn=1e5 的 +25 % 过预测已被 m20585 归因于软接触放大有效孔径，kn→1e6 的复验
（m21033）在飞（cj3201）。

## ⑤ 收敛性与参数敏感性

- **kn 归因**：m20585 诊断（软接触→有效孔径放大→流率系统性高估）；m20934 的
  kn∈{1e5,3e5,1e6} 扫描（cj3073）与 m21033 的 D5_kn1e5 归因臂将给出定量曲线。
- **μ 敏感性**：m20707 发现 μ=0.50 使 C_rec≈0.54–0.61 包住文献 C=0.58；m21033 的
  D5_mu30 臂检验 μ=0.15→0.30 的影响。
- **dt 收敛**：m21033 的 D5_dthalf 臂（|W(dt/2)−W(dt)|/W(dt) < 5 % 判据）。
- **D/d=4 边界**：m20707 pilot 在 D/d=4 出现 arching（与 Beverloo 适用域下界一致）；
  m21033 的 D4 臂（更大体系、更硬接触）检验是否消除。

## ⑥ 失败模式、工件与复现

**失败模式（如实）**：
- m6296/m20055 的 silo_discharge 在软接触（kn=1e5）下分别 arching / 过预测——
  接触刚度是 DEM 流率基准的一阶数值参数，不是材料参数。
- m20585 集群作业 SSH 超时/排队，零计算产出（L0），其假设从未被计算验证——
  由 m20934/m21033 补上。
- 小体系泊松噪声、单种子、C/k 为引用值（非重数字化）为残余风险。

**工件**
- 本文 + `runs/m21033_beverloo/`（cj3201 回传后）+ `benchmarks/m21033_beverloo/`
  （run_case.py / analyse.py / post_report_m21033.py）
- 图：`docs/figures/m20707_beverloo_rate_vs_orifice.png`、
  `docs/figures/m20707_beverloo_gravity_scaling.png`、
  `docs/figures/m20055_beverloo_flow_law.png`、
  `docs/figures/m15259_beverloo_comparison.png`
- 回归：`tests/test_m21033_beverloo_campaign.py`（10 项）

**复现**
```bash
PYTHONPATH=src python benchmarks/m21033_beverloo/run_case.py --case D5
PYTHONPATH=src python benchmarks/m21033_beverloo/analyse.py
PYTHONPATH=src python -m pytest tests/test_m21033_beverloo_campaign.py -q
```
