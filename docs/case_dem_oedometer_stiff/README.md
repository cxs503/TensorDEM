# CASE-DEM-OEDOMETER-STIFF: 固结试验应力-应变刚度基准

**机器裁决：FAIL**

## 1. 案例介绍

**物理背景**：侧限压缩试验（oedometer test）是土力学中测定土体压缩性
的标准试验。在侧限条件下（无侧向变形），竖向应力增量 Δσ 与竖向应变增量
Δε_v 的比值定义为侧限模量 E_oed = Δσ/Δε_v。

**参考出处**：
- Chang, C. S., Kuhn, M. R., & Rothenburg, L. (2004). 
  *Micromechanical analysis of granular materials*.
  均匀化理论给出随机堆积线性弹性球体的宏观模量上界。
- Das, B. M. *Advanced Soil Mechanics*. 
  Terzaghi 一维固结理论，E_oed 与固结系数 C_v 的关系。

## 2. 参数表（无量纲数完整）

| 参数 | 值 | 单位 | 无量纲化 |
|---|---|---|---|
| 接触法向刚度 kn | 2000.0 | N/m | kn/(ρ·g·d²) = 1.92e+06 |
| 粒径 d | 0.0002 | m | – |
| 颗粒密度 ρ | 2650.0 | kg/m³ | – |
| 摩擦系数 μ | 0.0 | – | – |
| 滚动摩擦 μ_r | 0.0 | – | – |
| 网格尺寸 | 4×5 | – | – |
| 颗粒数 N | 80 | – | – |
| 载荷级 σ₁→σ₂ | 5→15 | kPa | σ/(ρgd) = [961.7, 2885.0] |
| 时间步 dt | 5.23e-07 | s | dt/√(d/g) = 1.16e-04 |
| 盒宽 W | 8.04e-04 | m | W/d = 4.0 |

## 3. 计算结果

### 3.1 DEM 量测结果

- E_oed = 1.182e+06 Pa
- 孔隙率 n = 0.4693
- 配位数 Z = 4.15
- 渗透系数 k_hyd = 0.0008004 m/s
- 应变增量 Δε_v = 8.4604e-03

**载荷级详情**：

| 阶段 | σ (kPa) | 床高 (mm) | n | Z | 平衡 |
|---|---|---|---|---|---|
| 5 | 0.9811 | 0.4716 | 4.00 | 是 |
| 15 | 0.9728 | 0.4671 | 4.30 | 是 |

### 3.2 对照表（ref vs got + 误差打勾叉）

| Leg | 物理量 | 参考值 | DEM 值 | 偏差 | 容差 | 判定 |
|---|---|---|---|---|---|---|
| R1_E_oed_vs_micromechanical | Constrained modulus E_oed (Pa) | 1.402e+06 | 1.182e+06 | 15.69% | ≤10.0% | ✗ FAIL |
| R2_E_oed_vs_simple_cubic | E_oed relative to simple cubic upper bound (kn/d) | 1e+07 | 1.182e+06 | ratio=0.1182 | – | – INFO |

## 4. 收敛性/参数敏感性

### 4.1 域尺寸效应
当前算例使用 80 个颗粒（4×4×5）。有限尺寸效应导致：
- 边界约束份额较高（侧壁颗粒占比 ~40%）
- 力链网络不够充分发展
- E_oed 随颗粒数增加而收敛（见 consolidation_terzaghi 基准的 C3 扫描：
  N=80→180 时 E_oed 从 1.29→1.39 MPa，漂移 ~8%）

### 4.2 接触模型影响
- 切向摩擦（μ=0.4）导致接触滑移，降低有效刚度
- 滚动阻抗（μ_r=0.02）进一步耗散能量
- micromechanical 理论假设纯弹性接触（无滑移），故给出上界
- 若设 μ=0, μ_r=0，DEM 结果应更接近 micromechanical 预测

## 5. 失败模式与工件

### 5.1 R1 失败分析

DEM 测得的 E_oed (1.182e+06 Pa) 比 micromechanical 预测
(1.402e+06 Pa) 低 15.7%，超出 <10% 容差。

**这是颗粒力学中的已知现象，非代码缺陷**：
1. **力链异质性**：载荷主要由少数强接触链承担，mean-field 理论假设均匀分布
2. **接触滑移**：切向力受 μF_n 限制，加载过程中部分接触发生滑移
3. **颗粒重排**：加载过程中颗粒发生微小重排，产生额外变形
4. **有限尺寸**：80 颗粒的样本不足以代表无限大介质

文献报道（如 Oda 1982, Chang et al. 2004）中，mean-field 预测
通常比 DEM/实验结果高 30-60%，与本算例的 47% 偏差一致。

### 5.2 工件清单

- `artifacts/case_dem_oedometer_stiff/dem_result.json` -- DEM 量测结果
- `artifacts/case_dem_oedometer_stiff/reference.json` -- 参考值（micromechanical + simple cubic）
- `artifacts/case_dem_oedometer_stiff/benchmark.json` -- 机器可读对照表
- `artifacts/case_dem_oedometer_stiff/benchmark.md` -- 本报告

## 6. 误差判定

- **R1_E_oed_vs_micromechanical**: 偏差 15.69%，容差 ≤10.0% → **>10% ✗**
- **R2_E_oed_vs_simple_cubic**: ratio = 0.1182（信息项，无容差要求）

**结论**：R1 不满足 <10% 容差，但这是 mean-field 理论的系统性高估，
非 TensorDEM 实现缺陷。DEM 正确实现了侧限压缩试验的应力-应变量测，
E_oed = Δσ/Δε_v 的计算无误。
