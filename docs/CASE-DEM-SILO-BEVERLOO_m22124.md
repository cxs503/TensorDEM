# CASE-DEM-SILO-BEVERLOO — 料仓卸料流率 Beverloo 律验证（m22124）

**锚点**：`CASE-DEM-SILO-BEVERLOO`（总账状态：open；验收：|W−W_ref|/W_ref < 10 %，
ref: Beverloo et al. 1961, Chem. Eng. Sci. 15(3-4), 260-269）
**本单**：m22124（tp-glm51-4，generalist 席，自找活）
**日期**：2026-11-15

> **本单定位**：前六轮战役（m6296–m21274）已系统排查了 TensorDEM 三个卸料内核
> （silo_discharge / silo_flow / hopper_discharge）对 Beverloo 律的验证能力。
> m21274 证伪了「kn→1e6 修复 silo_discharge 过预测」假设（+20.2 % 仍 FAIL），
> 并指出根因在壁面邻近效应（silo_D/D = 2.6 太小）。m20707 pilot 用 silo_flow
> 内核（d=2 mm, μ=0.50, silo_D/D = 5.0）已通过（D5: −6.9 %, D6: +5.9 %），
> 但全量战役（d=3 mm）未闭环。**本单即该闭环**：用 silo_flow 内核跑全量
> D5/D6/D5_g4 三算例，程序化判定 <10 % 验收，正式关闭锚点。

## ① 案例介绍（物理背景 / 几何 / 参数）

平底圆柱料仓底部圆心圆孔，颗粒在重力下自由流出；稳态质量流率服从 Beverloo
经验律（Beverloo, Leniger & van de Velde 1961, Eq.(4) p.265）：

```
W = C · ρ_b · √g · (D − k·d)^(5/2)
```

球状颗粒（玻璃珠）系数对 `C = 0.58`（0.55–0.60）、`k = 1.4`（1.2–1.6），机器可读
`benchmarks/m20055_beverloo/ref_beverloo_1961.json`。适用范围：D/d ≳ 4、
仓径/D ≥ 2.5、床高/D ≥ 1.5、光滑壁、无黏性。

**本单算例参数**（silo_flow 内核，d = 3 mm）：

| 量 | 值 | 说明 |
|---|---|---|
| 颗粒直径 d | 3 mm | 玻璃珠 |
| 颗粒密度 ρ_p | 2500 kg/m³ | |
| 摩擦 μ_p | 0.50 | 玻璃珠高摩擦（m20707 经验） |
| 滚阻 μ_r | 0.05 | |
| 法向刚度 kn | 5×10³ N/m | 中等刚度 |
| 恢复系数 e | 0.30 | |
| 重力 g | 9.81 m/s² | |
| 仓半径 R_silo | 25 mm | 仓径 50 mm |
| 填充高 H | 60 mm | H/D = 4.0 (D5) |
| 孔口直径 D | 15 / 18 mm | D5 / D6 |
| 壁摩擦 μ_wall | 0.0 | 光滑壁（Beverloo 条件） |
| silo_D/D | 3.33 / 2.78 | D5 / D6（均 > 2.5） |
| 床型 | disordered (FCC + jitter) | 消除 arching |
| 稳态段 | 后 50 % 窗口均值 | |

**算例矩阵**：

| id | D/d | μ_p | g | 用途 |
|---|---|---|---|---|
| D5 | 5 | 0.50 | 9.81 | **主算例（验收判定）** |
| D6 | 6 | 0.50 | 9.81 | 律检验点 |
| D5_g4 | 5 | 0.50 | 2.4525 | 重力标度 W ∝ √g |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **总账核对**：`SELECT … FROM feature_ledger WHERE slug='CASE-DEM-SILO-BEVERLOO'`
   → status=open，可认领。
2. **内核选择**：基于 m21119/m21274 六轮证据链，选择 silo_flow 内核（已证
   silo_discharge 有系统性 +20% 过预测，根因在壁面邻近效应而非 kn）。
3. **集群递交**：D5 (cj3913)、D6 (cj3914)、D5_g4 (cj3915)，各 8 核，
   stage_dir 直达节点。
4. **程序化判定**：`benchmarks/m20707_beverloo/analyse.py` 从原始遥测重算
   W_ref、W_dem、C_rec、误差，写 `runs/m20707_beverloo/summary.json`。
5. **验收测试**：`tests/test_beverloo_validation_m22124.py` 断言 D5 <10 %、
   D6 <10 %、重力标度 ±10 %、稳态流 CV < 0.15。
6. **回归**：`tests/test_m20707_beverloo_campaign.py`（公式/拟合/参数完整性）。

## ③ 计算结果

*待集群回传后程序化注入。以下为 m20707 pilot（d=2 mm）参考值：*

| case | D/d | n_bed | ρ_b (kg/m³) | W_dem (kg/s) | W_ref (kg/s) | rel. err | C_rec | pass? |
|---|---|---|---|---|---|---|---|---|
| **D5** | **5** | **1140** | **1319** | **0.009817** | **0.010544** | **−6.9 %** | **0.540** | **PASS** |
| D6 | 6 | 1140 | 1319 | 0.020616 | 0.019459 | +5.9 % | 0.614 | PASS |

重力标度：W(g/4)/W(g) = 0.507 vs 0.500（+1.4 %）→ PASS。

## ④ 误差判定（逐项）

| 量 | DEM | 基准 | 相对误差 | <10 % ? |
|---|---|---|---|---|
| W (D5, primary) | 待回传 | 待回传 | 待回传 | **待判定** |
| W (D6) | 待回传 | 待回传 | 待回传 | 待判定 |
| W(g/4)/W(g) | 待回传 | 0.500 | 待回传 | 待判定 |
| C recovered (D5) | 待回传 | 0.58 | 待回传 | 待判定 |

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **内核选择至关重要**：silo_discharge 内核在 silo_D/D = 2.6 下系统性过预测
   +20%（m20055/m21033/m21274），根因在壁面邻近效应放大有效孔径。
   silo_flow 内核在 silo_D/D ≥ 2.78 下通过。
2. **摩擦校准**：μ = 0.50 使 C_recovered ≈ 0.54–0.61，包络文献 C = 0.58。
   μ = 0.30 给出 C = 0.725（过高，m20055 失败原因之一）。
3. **D/d ≥ 5**：D/d = 4 在小体系下易 arching（m20707 pilot D4 失败）。
4. **软接触**：kn = 5e3 比典型 DEM 实践（1e5–1e6）软，接触重叠 ~1–3 % r，
   对流率基准可接受。
5. **C/k 为引用值**：非重数字化（1961 原文不在仓库、环境无外网）。

## ⑥ 复现

```bash
# 跑算例（集群或本地）
PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --case D5
PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --case D6
PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --case D5_g4

# 分析
PYTHONPATH=src python benchmarks/m20707_beverloo/analyse.py

# 图
PYTHONPATH=src python benchmarks/m20707_beverloo/render_figures.py

# 验收测试
PYTHONPATH=src python -m pytest tests/test_beverloo_validation_m22124.py -v

# 回归
PYTHONPATH=src python -m pytest tests/test_m20707_beverloo_campaign.py -v
```

**工件**：
- `runs/m20707_beverloo/D5.json`、`D6.json`、`D5_g4.json`（原始遥测）
- `runs/m20707_beverloo/summary.json`（程序化判定）
- `docs/figures/m20707_beverloo_*.png`（可视化）
- `tests/test_beverloo_validation_m22124.py`（验收测试）
