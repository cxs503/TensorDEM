# CASE-DEM-SILO-BEVERLOO — 料仓卸料流率 Beverloo 律 · kn=1e6 复验判定（m21274）

**锚点**：`CASE-DEM-SILO-BEVERLOO`（总账状态：open；验收：|W−W_ref|/W_ref < 10 %，
ref: Beverloo et al. 1961, Chem. Eng. Sci. 15(3-4), 260-269, DOI 10.1016/0009-2509(61)80059-4）
**本单**：m21274（tp-ds31-1，generalist 席，自找活·验证欠账）
**日期**：2026-11-15

> **本单定位**：m21119 文档明确「m21033 主算例 D5 的数值以 cj3201 终态为准，本文不预填
> 任何未回传的流率」。本单即该**验证欠账的闭环**：cj3201（6 分片并行，stage_dir 直达
> 节点）已回传主算例 **D5**（`runs/m21033_beverloo/D5.json`，node315，wall_clock 1219 s），
> 本单把该原始遥测程序化注入判定表，给出**当前可判定结论**——并如实记录一个
> **被否定的假设**（falsified hypothesis）：m20585/m21119 主张「kn 1e5→1e6 可把
> silo_discharge 内核的 +25 % 过预测拉回 <10 %」，**该主张在 kn=1e6 下被证伪**。

## ① 案例介绍（物理背景 / 几何 / 参数）

平底圆柱料仓（flat-bottom silo）底部圆心圆孔，颗粒在重力下自由流出；稳态质量流率
服从 Beverloo 经验律（Beverloo, Leniger & van de Velde 1961, Eq.(4) p.265）：

```
W = C · ρ_b · √g · (D − k·d)^(5/2)
```

球状颗粒（玻璃珠）系数对 `C = 0.58`（0.55–0.60）、`k = 1.4`（1.2–1.6），机器可读
`benchmarks/m20055_beverloo/ref_beverloo_1961.json`（零手抄；provenance: cited,
not re-digitized——1961 原文不在仓库、环境无外网）。

**本单算例（m21033 预注册矩阵主算例 D5，kn=1e6 复验）**——全部来自
`runs/m21033_beverloo/D5.json` 原始遥测（零手抄）：

| 量 | 值 | 说明 |
|---|---|---|
| 颗粒半径 r | 0.010 m | d = 20 mm（玻璃珠） |
| 颗粒密度 ρ_p | 2500 kg/m³ | |
| 摩擦 μ_p | 0.15 | 玻璃珠低摩擦（m15259 经验） |
| 滚阻 μ_r | 0.02 | |
| 法向刚度 kn | 1.0e6 N/m | **m20585 建议值**（原 m20055 为 1e5） |
| 恢复系数 e | 0.30 | |
| 重力 g | 9.81 m/s² | |
| 仓半径 R_silo | 0.130 m | 仓径 0.26 m |
| 填充高 H | 0.40 m | |
| 孔口直径 D | 0.100 m | D/d = 5（主算例） |
| 板厚 t | 0.010 m | 0.5 d |
| 壁摩擦 μ_wall | 0.0 | 光滑壁（Beverloo 玻璃珠条件） |
| 床型 | settled + plug | 松散晶格净距 0.30 r，堵孔压实 0.60 s → 开孔 |
| 稳态段 | 质量分位 0.2–0.8 | 切 5 子窗口报 CV |
| 颗粒数 n_bed | 1525 | |
| 床体密度 ρ_b | 1333.39 kg/m³ | φ = 0.533 |

**无量纲完整性**（`dimensionless` 节）：`D/d = 5.0`、`silo_D/D = 2.6`、
`H/D = 4.0`、`ρ_b/ρ_p = 0.533`、`dt/T_contact = 0.0354`、`μ_p = 0.15`、
`μ_wall = 0`、`e = 0.30`、Beverloo 前置因子 `C = W/(ρ_b √g (D−kd)^{5/2})`。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **总账核对**：`SELECT … FROM feature_ledger WHERE slug='CASE-DEM-SILO-BEVERLOO'`
   → status=open，可认领；未改写锚点。
2. **在飞核对**：cj3201（m21033 6 分片并行）running；cj3141（旧串行）failed(rc=1)。
   `run_compute fetch cj3201` 取回 n315/n330 两节点工件 tar。
3. **回传注入**：从 n315 工件解出 `runs/m21033_beverloo/D5.json`（主算例，node315，
   wall_clock 1219 s，`--shard 1 --nshards 6`），复制进本工作树 `runs/m21033_beverloo/`。
4. **程序化判定**：`PYTHONPATH=src python benchmarks/m21033_beverloo/analyse.py`
   ——从原始遥测重算 W_ref（C=0.58, k=1.4, ρ_b=1333.39）、W_dem（稳态段质量分位
   0.2–0.8 切 5 子窗口）、C_rec、CV、误差，写 `summary.json` + `report_tables.md`，
   渲染 `docs/figures/m21033_beverloo_flow_law.png` 与
   `m21033_beverloo_discharge_stream.png`。**零手抄**。
5. **回归**：`tests/test_m21033_beverloo_campaign.py`（10 项：ref 完整性、预注册
   一致性、公式、稳态窗口不变量、判定算术）——本单只注入数据，不改代码，回归应绿。

## ③ 计算结果（程序化引用，零手抄）

**主算例 D5（kn=1e6, μ=0.15, silo_discharge 内核）**——`runs/m21033_beverloo/D5.json`：

| 量 | DEM | 基准 | 相对误差 | 判定 |
|---|---|---|---|---|
| W（稳态段） | 4.0497 kg/s | 3.3694 kg/s | **+20.19 %** | **✗ FAIL**（>10 %，>15 %） |
| C_recovered | 0.697 | 0.58（文献） | +20.2 % | ✗ |
| 稳态窗口 CV | 0.086 | <0.15 | — | ✓（稳态成立） |
| 卸料占比 | 70 %（1064/1525） | — | — | ✓（非空仓） |

**窗口序列**（`steady_rate_windows_kg_s`）：[4.013, 3.862, 4.578, 4.149, 3.646]，
CV = 0.086 < 0.15——**稳态流成立**，+20.2 % 不是瞬态/空仓伪影，是**稳态系统性过预测**。

**重力/几何自洽**：`Deff = D − k·d = 0.100 − 1.4×0.020 = 0.072 m`（与遥测一致）；
`W_ref = 0.58 × 1333.39 × √9.81 × 0.072^2.5 = 3.3694 kg/s`（程序化重算，与
`W_ref_kg_s` 字段一致）。

## ④ 对照表：ref vs got + 误差判定（逐项）

| 量 | DEM | 基准 | 相对误差 | <10 % ? |
|---|---|---|---|---|
| W (D/d=5, silo_discharge, kn=1e6, μ=0.15) | 4.0497 kg/s | 3.3694 kg/s | **+20.19 %** | **否** |
| C_recovered | 0.697 | 0.58 | +20.2 % | 否 |
| 稳态窗口 CV | 0.086 | <0.15 | — | 是（稳态） |
| 卸料占比 | 70 % | — | — | 是（非空仓） |

**当前可判定结论（honest）**：**Beverloo 律在 silo_discharge 内核（kn=1e6）上
未被验证——主算例 D5 误差 +20.19 %，超出 <10 % 容差（也超出 <15 %）。**
m20585/m21119 的「kn→1e6 修复过预测」假设**被证伪**：接触刚度不是该内核
过预测的一阶根因。

**对照（已发布工件，非本单重算）**：
- **silo_flow 内核**（m20707，d=3mm，μ=0.50，kn=5e3）：D5 **−6.9 %**、D6 **+5.9 %** —— **PASS**；
  重力标度 W(g/4)/W(g)=0.507 vs 0.500（+1.4 %）。
- **hopper 内核**（m15259，kn=1e6，μ=0.15）：D/d=5 **−0.4 %** —— **PASS**。
- **silo_discharge 内核**（m20055，kn=1e5，μ=0.30）：D4 **+25.0 %** —— FAIL；
  （m21033，kn=1e6，μ=0.15）：D5 **+20.2 %** —— **FAIL（本单）**。

**关键差异**：silo_discharge 在 kn=1e5 与 kn=1e6 下**都**过预测（+25 % / +20 %），
而 silo_flow（小体系、高摩擦）与 hopper（锥形）在各自条件下**都**通过。过预测
**不随 kn 单调消失**——根因在**几何/协议**，不在接触刚度。

## ⑤ 根因分析与参数敏感性（本单诊断）

**候选根因（按证据排序，非定论）**：

1. **壁面邻近效应（最可能）**：本算例 `silo_D/D = 2.6`，仅**贴** Beverloo 适用域下界
   （≥2.5）；孔口边缘到仓壁仅 `0.13−0.05 = 0.08 m = 4d`。m20707 通过算例
   `silo_D/D = 3.33`（孔口到壁 `0.025−0.0075 = 0.0175 m ≈ 5.8d`）。壁面约束
   卸料流锥，使**有效孔径**大于 Beverloo 假设的 `D−k·d`，→ 系统性高估。
   **证据**：C_recovered = 0.697 显著高于文献 0.58，且不随 kn 变化——与「几何
   放大有效孔径」一致，与「软接触」不一致。
2. **测量协议（settled+plug）**：堵孔压实 0.60 s 后开孔，床体在孔口上方形成
   预压拱/密实区，开孔瞬态可能把「预压释放」混入稳态段。m20707 用自由落体
   松散床（无堵孔），无此伪影。**证据**：首窗 `flow_rate_windows[0]=0`（开孔
   瞬态），但稳态段质量分位 0.2–0.8 已剔除首尾——残余风险中等。
3. **床高/体系尺度**：H/D = 4.0（m20707 为 2.0），大床高下 Janssen 拱效应
   与孔口上方应力状态不同，可能改变卸料动力学。

**参数敏感性（本单未重算，留给后续）**：m21033 矩阵其余算例（D4/D6/D5_mu30/
D5_kn1e5/D5_dthalf）在 cj3201 其余分片（n330）在飞；本单只回传主算例 D5。
**建议**：后续单应做 `silo_D/D ∈ {2.6, 3.0, 3.5}` 扫描，检验「壁面邻近」根因——
若增大 silo_D/D 后误差收敛 <10 %，则根因确认，Beverloo 律在**适用域内**仍可验证。

## ⑥ 失败模式、工件与复现

**失败模式（如实）**：
- **假设被证伪**：m20585「软接触→有效孔径放大→kn→1e6 修复」的归因在 kn=1e6 下
  不成立（+20.2 % 仍 FAIL）。这是**有价值的负面结果**——防止平台把 silo_discharge
  内核误标为「已验证」。
- **单算例回传**：cj3201 6 分片仅 n315 回传 D5（主算例）；n330 其余分片在飞，
  本单不预填未回传算例（honest）。
- **残余风险**：C/k 为引用值（非重数字化）；单种子；小体系泊松噪声。

**工件**
- 本文 + `runs/m21033_beverloo/D5.json`（原始遥测，node315）+ `summary.json` +
  `report_tables.md`
- 图：`docs/figures/m21033_beverloo_flow_law.png`、
  `docs/figures/m21033_beverloo_discharge_stream.png`
- 回归：`tests/test_m21033_beverloo_campaign.py`（10 项）

**复现**
```bash
# 回传数据已在 runs/m21033_beverloo/D5.json
PYTHONPATH=src python benchmarks/m21033_beverloo/analyse.py   # 程序化判定
PYTHONPATH=src python -m pytest tests/test_m21033_beverloo_campaign.py -q
```

**给下次自找活的建议线索**：
- **silo_D/D 扫描**（新条目建议）：`CASE-DEM-SILO-BEVERLOO-WALL`——把 silo_discharge
  的 `silo_D/D` 从 2.6 扫到 3.5，检验「壁面邻近」根因；量化验收：D5 在 silo_D/D=3.5
  下 |W−W_ref|/W_ref < 10 %；所属域：DEM 求解器（TensorDEM）。
- **测量协议对照**：settled+plug vs 自由落体松散床，检验「预压释放」伪影是否
  混入稳态段；量化验收：两协议稳态段 W 差 < 5 %。
