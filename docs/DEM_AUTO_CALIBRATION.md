# DEM 接触参数自动标定反演（TensorDEM，节点 `p284w3b-n05-dem-auto-calibration`）

> 本文件由 `scripts/make_calibration_report.py` 从 `artifacts/dem_auto_calibration/*.json` 程序化生成，零手抄；重跑标定即刷新。

## 1. 案例介绍

颗粒 DEM 的宏观响应由微观接触参数控制：滑动摩擦 `mu_pp`、恢复系数 `e`、法向/切向刚度比 `k_n/k_s`。工程上无法直接测量，只能以宏观试验反演。本节点实现**休止角锚定的自动标定闭环**：正演 DEM 堆积试验 → 宏观响应 (休止角 theta_r、堆积固体分数 phi) → GP 代理 + 期望改进贝叶斯优化 → 参数后验 → 独立工况泛化验证。

| material | r [mm] | rho [kg/m3] | mu_r | truth mu_pp | truth e | truth k_n/k_s |
|---|---|---|---|---|---|---|
| glass_beads | 4.0 | 2500 | 0.05 | 0.34 | 0.62 | 1.40 |
| sand | 3.0 | 2650 | 0.10 | 0.52 | 0.38 | 2.20 |
| gravel | 5.0 | 2650 | 0.15 | 0.68 | 0.18 | 2.80 |

正演协议（雨淋堆积法 pluviation）：

| parameter | value |
|---|---|
| shape | pluviation |
| n_grains | 240 |
| layer_size | 12 |
| settle_per_layer | 200 |
| inject_radius_factor | 7.0 |
| drop_height_factor | 1.5 |
| kn | 800.0 |
| packing_gap | 0.03 |
| jitter | 0.02 |
| mu_wall | 0.34 |
| dt | 0.00020332659009681875 |
| seed | 31001 |

单次正演约 22.6 s（单核，240 颗粒，每步两次核函数调用）。

## 2. 怎么做的

1. **正演** `heap.run_pluviation`：240 球按 12 个/层从堆顶上方 1.5r 雨淋注入，每层松弛 200 步；接触律复用已验证的 `granular_column.kernel`（线性弹簧-阻尼法向 / Cundall-Strack 切向饱和 mu*Fn / 各向同性滚动弹簧饱和 mu_r*r*Fn），velocity-Verlet 积分（弹簧每步推进一次），注入前 cKDTree 校验 2r 间距并整层抬升，避免重叠注入弹性能。
2. **宏观响应**：休止角用环扇形自由面拟合（环宽 1 粒径、8 扇区最高点、shoulder-toe 15–90% 窗口、单侧 2σ 剔除），并加**径向质量 0.95 分位 toe 截断**以剔除溅出散粒；同时输出堆积固体分数。
3. **反演**：`J = Σ w_i |sim_i-exp_i|/|exp_i|`（w=0.8/0.2），参数盒 mu∈[0.1,0.8]、e∈[0.1,0.9]、k_n/k_s∈[1,3]；拉丁超立方初始设计 + Matern-5/2 GP（各向异性长度尺度、边缘似然搜索）+ 期望改进 EI（512 候选，LCB 决胜）；每点 1 次正演。
4. **不确定性**：已访问响应面上随机游走 Metropolis，`logL=-1/2 Σ (m_i-d_i)²/(σ_i²+s_i(θ)²)`，代理误差入方差，盒内均匀先验，报 95% 可信区间。
5. **验证**：标定参数在**换种子**的同一工况重跑（验收数≠优化器最小化的数）；再迁移到**独立工况**（不同粒径/容器/填高/种子）与同工况孪生目标比较。
6. **可复现**：固定种子 → 末态位置 SHA-256 逐位一致；整条流程重跑两次同结果。

## 3. 计算结果

### 3.1 标定与验收

| material | θ target [deg] | θ verified [deg] | err [deg] | <5° | best J | J<0.05 | evals | ≤50 | gen rel err | <10% | CI covers | repro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glass_beads | 4.09 | 3.45 | 0.64 | PASS | 0.0258 | PASS | 18 | PASS | 0.6347 | FAIL | PASS | PASS |
| sand | 30.41 | 20.66 | 9.75 | FAIL | 0.0595 | FAIL | 18 | PASS | 0.0077 | PASS | PASS | PASS |
| gravel | 18.11 | 18.69 | 0.58 | PASS | 0.1003 | FAIL | 18 | PASS | 0.3222 | FAIL | PASS | PASS |

### 3.2 后验 95% 可信区间 vs 真值

| material | μ truth | μ median [2.5,97.5] | e truth | e median [2.5,97.5] | k_n/k_s truth | k_n/k_s median [2.5,97.5] | accept rate |
|---|---|---|---|---|---|---|---|
| glass_beads | 0.34 | 0.313 [0.100, 0.800] | 0.62 | 0.434 [0.100, 0.858] | 1.40 | 2.306 [1.000, 3.000] | 3.46 |
| sand | 0.52 | 0.435 [0.100, 0.800] | 0.38 | 0.397 [0.100, 0.900] | 2.20 | 2.229 [1.000, 3.000] | 4.00 |
| gravel | 0.68 | 0.386 [0.100, 0.800] | 0.18 | 0.365 [0.100, 0.870] | 2.80 | 2.188 [1.000, 3.000] | 3.34 |

### 3.3 独立工况泛化（不重标定）

| material | validation geometry | θ target [deg] | θ predicted [deg] | rel err |
|---|---|---|---|---|
| glass_beads | r=3.5 mm, wall=24.5 mm, seed 31002 | 7.99 | 2.92 | 0.6347 |
| sand | r=3.5 mm, wall=24.5 mm, seed 32002 | 17.21 | 17.08 | 0.0077 |
| gravel | r=4.0 mm, wall=28.0 mm, seed 33002 | 28.07 | 19.02 | 0.3222 |

### 3.4 战役判据

| criterion | requirement | result |
|---|---|---|
| (a) accuracy | 3 类材料 θ 误差 < 5° | FAIL |
| (b) convergence | ≤50 次正演且 J < 0.05 | FAIL |
| (c) generalisation | 独立工况 < 10% | FAIL |
| (d) uncertainty | 95% 区间覆盖真值 | PASS |
| (e) reproducibility | 逐位一致 | PASS |

**总判定：FAIL (honest)**

## 4. 图

fig1_response_surface.png（响应曲面与目标函数）、fig2_convergence.png（收敛）、fig3_posterior.png（区间 vs 真值）、fig4_generalisation.png（1:1 泛化）；均由 `scripts/plot_dem_calibration.py` 从同一批 JSON 生成。

## 5. 注意点

1. **释放类协议在本系统规模下不可用（最贵的发现）**：提升圆筒/退壁协议（`shape='column'`）给出 θ=0.4–5° 且随 μ 非单调（沙：μ=0.30→7.1°、0.45→3.3°、0.60→2.9°）——壁只从内侧施力时，下移的壁顶自顶部逐层解约束，料层摊平而非成锥；过陡初始锥松弛协议（`shape='cone'`，55°）在 μ=0.10 全崩（2.3°）、μ≥0.30 几乎不松弛（46–52°）——密排 hcp 锥是互锁结构，响应是 jam/unjam 双态而非连续映射。只有雨淋堆积在本算量下给出物理可用的休止角响应。
2. **估计器 max 统计偏差**：环扇区取最高点使斜率系统性偏平约 5%（解析锥 20°/30°/40° → 18.6°/28.1°/37.8°，斜率比 0.90–1.00）；该偏差对目标与预测同源，在标定中抵消，作为回归守卫入测试。
3. **溅出散粒**：同一堆在有/无外溅颗粒时给出 0.9° vs 约 39°，故 toe 取径向质量 0.95 分位。
4. **注入重叠=能量爆炸**：雨淋层与既有颗粒重叠（或层内互叠）会注入弹性能（实测归一化残动能 1e27）；必须整层抬升而非逐颗粒抬升。
5. **生长法能量归一化**：堆体是获得势能，`released=pe0-pe` 无意义，残动能须按堆体自身势能量级归一。
6. **主机陷阱**：共享磁盘 100% 占满时 scipy/LAPACK `cho_factor` 会 SIGSEGV/Bus error 崩解释器（输入并非 NaN）；GP 线性代数改为纯 NumPy Cholesky，消除崩溃面并保证可复现。
8. **已知缺陷（本批工件）**：`acceptance_rate` 分母少乘 thinning 因子，故工件里的接受率被放大 ×4（后验分位数与区间不受影响，因其直接取自保留样本）；代码已修正，重跑标定即得真值。
7. **适用范围**：类内粒度/密度/滚动阻力 `mu_r` 固定（未参与标定）；若 mu_r 也未知需 4 维反演，可识别性进一步下降。孪生试验（twin experiment）验证的是**反演机制**，不等价于证明正演模型复现真实材料，后者需实验数据。

## 6. 误差判定

| quantity | reference | measured | threshold | verdict |
|---|---|---|---|---|
| θ 误差 glass_beads | 孪生 DEM 目标 | 0.64 deg | < 5 deg | PASS |
| θ 误差 sand | 孪生 DEM 目标 | 9.75 deg | < 5 deg | FAIL |
| θ 误差 gravel | 孪生 DEM 目标 | 0.58 deg | < 5 deg | PASS |
| 泛化 glass_beads | 独立工况目标 | 63.47 % | < 10 % | FAIL |
| 泛化 sand | 独立工况目标 | 0.77 % | < 10 % | PASS |
| 泛化 gravel | 独立工况目标 | 32.22 % | < 10 % | FAIL |
| 估计器 vs 解析锥 20/30/40° | tan(θ) 精确值 | 斜率比 0.90–1.00（偏平≈5%） | 无阈值 | 系统偏差，标定中抵消 |

**是否 < 3%？** 逐项：估计器 vs 解析锥 **不是**（斜率偏平最高约 10%，角度 1.4–2.2°），但同源抵消；标定后 θ 误差与泛化误差见 §3.1/§3.3（判据线 5°/10%）；参数后验中位数 vs 真值**不设 3% 判据**（`e`、`k_n/k_s` 弱可识别，见区间宽度），本节点只要求区间覆盖。

## 7. 审计链

- 模块 `src/tensordem/dem_calibration/{heap,surrogate,inverse,pipeline,materials}.py`；驱动 `scripts/{calibrate_dem_materials,plot_dem_calibration,make_calibration_report,repose_response_scan,probe_heap_scan}.py`；工件 `artifacts/dem_auto_calibration/{glass_beads,sand,gravel}.json`；测试 `tests/test_dem_calibration.py`。
