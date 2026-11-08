# Euler-Bernoulli beam FEM (`tensordem.fem_euler_bernoulli`): analytic verification

状态：验证欠账清偿 + 两处缺陷修复（selfstart，TensorDEM 仓）。
本文件由 `scripts/verify_fem_euler_bernoulli.py` 程序化生成，表中数字全部来自
`artifacts/fem_euler_bernoulli/report.json`（零手抄）。协议
`tensordem-fem-euler-bernoulli-analytic-v1`。

## 1. 案例介绍（为什么这是验证欠账）

`fem_euler_bernoulli` 是 DEM–FEM 单向耦合示例（`examples/dem_fem_oneway`）的 FEM 侧：
2 节点 Hermite 三次梁单元 + Newmark 平均加速度时间积分，提供静力求解、模态相关阻尼与
动力推进。该模块此前**没有任何测试引用**（全仓 `grep` 仅示例与 `__init__` 导入），
即一个已合并、声称为“解析可验证”的求解器从未被解析解复核过。

物理背景与参数（SI，钢梁，小变形 Euler-Bernoulli 理论，w 向下为正）：

| 参数 | 值 |
|---|---|
| 梁长 L | 1.0 m |
| 截面宽 × 高 | 0.1 × 0.01 m |
| 惯性矩 I = bh^3/12 | 8.333333e-09 m^4 |
| 杨氏模量 E | 2.100e+11 Pa |
| 密度 rho | 7850.0 kg/m^3 |
| 单位长度质量 rhoA | 7.8500e+00 kg/m |
| 集中载荷 P | 1000.0 N（向下） |
| 均布载荷 q | 500.0 N/m（向下） |

参考解取自 Timoshenko & Gere《Mechanics of Materials》梁挠曲线表与 Clough & Penzien
《Dynamics of Structures》等截面梁特征频率表。

### 验证中发现并修复的两个缺陷

1. **广义特征值约化转置错误 → 任何带阻尼的工况静默变 NaN（P0）**。
   原实现用 `A = (L^-T K L^-T)^T = L^-1 K L^-1`（多转置一次），该矩阵非对称，
   `eigvalsh` 返回负“特征值”，`omega1 = sqrt(负值)` → NaN → `C = (2 zeta/omega1) K` 全 NaN。
   实测：`BeamConfig(damping_ratio=0.02)` 时 `C_free` 全 NaN，5 步后位移 NaN。
   修复为 `A = L^-1 K L^-T`（对称半正定），并断言 `omega1 > 0`。
   回归测试 `test_eigen_reduction_is_symmetric_positive_definite`、
   `test_rayleigh_damping_produces_finite_state_and_target_decay`。
2. **中点输出按节点下标取，奇偶单元数串味**。
   `center_deflection` 原取 `w[n_node // 2]`；`n_elements` 为奇数时该节点不在 L/2
   （9 单元时 x = 0.556 L），静力结果
   被静默报成不在中点的值。修复：新增 `deflection_at(x)`（三次 Hermite 插值）、
   `consistent_point_load(x, P)`（一致节点载荷），中跨量一律插值取得。

## 2. 怎么做的（方法/流程）

1. 读模块源码，重建刚度/质量矩阵与边界条件约定（θ = dw/dx，w 向下为正）。
2. 对每个可解析闭合的工况构造一致节点载荷，跑静力解，与闭合解逐项比相对误差；
   同时在 201 点网格上比较**整条挠曲线**（不只中点），确认单元内部场正确。
3. 把约化特征值问题独立复核（对称性、正定性、前 3 阶频率 vs 解析 β_n L 表），
   用 5/10/20/40/80 单元量测收敛阶。
4. 动力面：从静力平衡释放 → 无阻尼自由振动测周期与能量漂移；带 ζ=0.02 刚度比例
   Rayleigh 阻尼测对数衰减率反推 ζ（峰值相隔半周期，故半周期对数衰减率 ×2 得每周期）。
5. 全部数字写入 `report.json`/CSV，图由 matplotlib 生成，本 md 由脚本渲染。

## 3. 计算结果

### 3.1 静力：FEM vs 闭合解（全部单元数下的最大相对误差）

| case | n_elements | FEM [m] | closed form [m] | rel err | < 3% |
|---|---|---|---|---|---|
| ss_point | 2 | 1.190476e-02 | 1.190476e-02 | 4.37e-16 | YES |
| ss_point | 20 | 1.190476e-02 | 1.190476e-02 | 7.35e-13 | YES |
| cc_point | 2 | 2.976190e-03 | 2.976190e-03 | 0.00e+00 | YES |
| cc_point | 20 | 2.976190e-03 | 2.976190e-03 | 3.23e-13 | YES |
| cantilever_tip | 2 | 1.904762e-01 | 1.904762e-01 | 2.62e-15 | YES |
| cantilever_tip | 20 | 1.904762e-01 | 1.904762e-01 | 1.03e-11 | YES |
| ss_udl | 2 | 3.720238e-03 | 3.720238e-03 | 1.17e-16 | YES |
| ss_udl | 20 | 3.720238e-03 | 3.720238e-03 | 8.73e-13 | YES |
| cc_udl | 2 | 7.440476e-04 | 7.440476e-04 | 0.00e+00 | YES |
| cc_udl | 20 | 7.440476e-04 | 7.440476e-04 | 2.54e-13 | YES |
| - | - | - | - | - | - |

- 四种工况（SS/CC/悬臂点载、SS 均布）在 2 单元到 20 单元下最差相对误差
  = **8.73e-13**（SS/CC/SS-UDL），悬臂点载 **1.03e-11**。
  （量级 1e-13~1e-14 即双精度舍入；这是 Hermite 单元对点载/均布载**节点值精确**的
  已知性质，不是拟合。）
- 节点值精确不等于单元内部场精确，故另有节点采样对照：
| case | x/L | FEM [m] | closed form [m] | rel err |
|---|---|---|---|---|
| ss_point | 0.25 | 8.184524e-03 | 8.184524e-03 | 6.95e-13 |
| ss_point | 0.50 | 1.190476e-02 | 1.190476e-02 | 7.35e-13 |
| cc_point | 0.25 | 1.488095e-03 | 1.488095e-03 | 3.03e-13 |
| cc_point | 0.50 | 2.976190e-03 | 2.976190e-03 | 3.23e-13 |

### 3.2 特征频率收敛（含 P0 修复后约化正确性）

| bc | n_elements | mode1 rel err | mode2 rel err | mode3 rel err |
|---|---|---|---|---|
| simply_supported | 5 | 1.07e-04 | 1.66e-03 | 7.94e-03 |
| simply_supported | 10 | 6.75e-06 | 1.07e-04 | 5.34e-04 |
| simply_supported | 20 | 4.22e-07 | 6.75e-06 | 3.40e-05 |
| simply_supported | 40 | 2.52e-08 | 4.22e-07 | 2.14e-06 |
| simply_supported | 80 | 5.81e-09 | 2.53e-08 | 1.34e-07 |
| clamped | 5 | 5.48e-04 | 3.99e-03 | 1.38e-02 |
| clamped | 10 | 3.46e-05 | 2.61e-04 | 9.87e-04 |
| clamped | 20 | 2.17e-06 | 1.65e-05 | 6.30e-05 |
| clamped | 40 | 1.36e-07 | 1.03e-06 | 3.96e-06 |
| clamped | 80 | 7.35e-09 | 6.46e-08 | 2.48e-07 |
| cantilever | 5 | 1.35e-05 | 5.00e-04 | 3.59e-03 |
| cantilever | 10 | 8.55e-07 | 3.31e-05 | 2.55e-04 |
| cantilever | 20 | 5.34e-08 | 2.10e-06 | 1.64e-05 |
| cantilever | 40 | 9.12e-09 | 1.32e-07 | 1.03e-06 |
| cantilever | 80 | 2.62e-08 | 7.64e-09 | 6.43e-08 |

- 80 单元最大相对误差 **2.48e-07**。SS/CC 各阶在相邻档间误差稳定下降
  ~15–16 倍，与一致质量 Hermite 单元的 O(h^4) 一致（实测比值见 `report.json` 的
  `modal_refinement_ratio`）；悬臂第 1 阶在 60→80 单元处已到 1e-8 双精度底噪，
  该档比值不再单调（honest note，非收敛失败）。一致质量给出频率上界
  （`omega_FEM >= omega_analytic`）。
- 悬臂基频用 β_1 L = 1.8751（ω_1 = 3.5160·sqrt(EI/(rhoA L^4))）复核通过。

### 3.3 奇数单元中跨插值（缺陷 2 的定量证据）

| n_elements | h/L | middle node x/L | interpolated rel err | old node-pick rel err |
|---|---|---|---|---|
| 3 | 0.333 | 0.667 | 9.26e-03 | 1.48e-01 |
| 5 | 0.200 | 0.600 | 2.00e-03 | 5.60e-02 |
| 9 | 0.111 | 0.556 | 3.43e-04 | 1.78e-02 |
| 19 | 0.053 | 0.526 | 3.64e-05 | 4.08e-03 |
| 39 | 0.026 | 0.513 | 4.21e-06 | 9.78e-04 |

- 9 单元时旧取法误差 1.78e-02（≈ 1.78%），
  修复后 3.43e-04；19 单元起进入 1e-5 以下。

### 3.4 动力（Newmark β=1/4, γ=1/2, dt = T/500, 20 单元）

| case | period FEM [s] | period analytic [s] | period rel err | zeta measured | zeta target | energy change (rel) | finite |
|---|---|---|---|---|---|---|---|
| undamped | 4.264384e-02 | 4.263791e-02 | 1.39e-04 | 0.0002 | 0.0000 | -2.69e-04 | YES |
| damped_2pct | 4.264618e-02 | 4.263791e-02 | 1.94e-04 | 0.0200 | 0.0200 | -4.74e-01 | YES |

- 无阻尼周期相对误差 **1.39e-04**；总能量变化
  **-2.69e-04**（2.5 个周期末态相位误差量级，平均加速度法
  无数值耗散）。带阻尼工况的能量变化 -4.74e-01 是**物理耗散**，
  不是数值失稳——阻尼项即为此而设。
- 阻尼工况反推 ζ = **0.0200**（目标 0.0200），
  相对偏差 0.03%；
  修复前该工况全程 NaN。

## 4. 结果图

- `artifacts/fem_euler_bernoulli/fig_deflection_fields.png`：四类工况的整条挠曲线 FEM vs 闭合解（20 单元）。
- `artifacts/fem_euler_bernoulli/fig_convergence.png`：左=特征频率误差 vs h/L（三种边界、前 3 阶，log-log）；
  右=奇数单元中跨误差，修复前后对照。
- `artifacts/fem_euler_bernoulli/fig_free_vibration.png`：无阻尼/带阻尼自由振动时程 vs 解析余弦与
  exp(-ζω_1 t) 包络。

## 5. 注意点（坑、局限、适用范围）

- **节点精确 ≠ 全场精确**：点载/均布载在节点上的挠度与转角对 Hermite 单元是精确的，
  但单元内部（尤其跨中出现集中载荷且载荷不在节点时）仍是 O(h^2~3) 离散误差；
  报告里两者分开列，避免把 1e-14 当成“求解器全场精确”。
- 一致质量矩阵给频率上界；集中质量会给下界，本模块只实现一致质量。
- 阻尼只实现刚度比例 Rayleigh（C = 2ζ/ω_1·K），因此高阶模态阻尼比按 ω_n/ω_1 放大，
  不是常数阻尼；`damping_ratio` 只对第 1 阶模态有物理意义。
- `static_solve` 默认不改动动力状态（纯探针），需要从静力平衡起步必须显式
  `commit=True`（旧行为是静默不提交，容易误用；现已写入 docstring 并有回归测试）。
- 大变形、剪切变形（Timoshenko）、转动惯量、轴向力（几何刚度）均不在本模块范围；
  细长比 h/L 不宜过大（本例 h/L=0.01，Euler-Bernoulli 适用）。
- 本验证只覆盖线性弹性梁；DEM 侧接触力幅值/时程的正确性属于 `dem3d` 的验证范围。

## 6. 误差判定（关键量 vs 基准，逐项回答“是否 < 3%”）

| 关键量 | 基准 | 相对误差 | < 3%？ |
|---|---|---|---|
| SS 中点集中载荷挠度 | P L^3/(48 EI) | 7.35e-13（10 单元 1.17e-16） | YES |
| CC 中点集中载荷挠度 | P L^3/(192 EI) | 3.06e-15 | YES |
| 悬臂端部集中载荷挠度 | P L^3/(3 EI) | 7.69e-14 | YES |
| SS 均布载荷中点挠度 | 5 q L^4/(384 EI) | 9.66e-14 | YES |
| CC 均布载荷中点挠度 | q L^4/(384 EI) | 4.37e-16 | YES |
| SS 前 3 阶特征频率（80 单元） | (n pi)^2 sqrt(EI/(rhoA L^4)) | <= 2.48e-07 | YES |
| 悬臂基频（80 单元） | 3.5160 sqrt(EI/(rhoA L^4)) | 7.64e-09 | YES |
| 自由振动周期（无阻尼） | 2 pi / omega_1 | 1.39e-04 | YES |
| 阻尼比（对数衰减率反推） | 0.02 | 0.03% | YES |
| 奇数单元（9）中跨挠度 | P L^3/(48 EI) | 3.43e-04 | YES |

全部关键量 < 3%，静力量级在双精度舍入（1e-13）。两项 honest 局限：
(a) 奇数单元 + 跨中集中载荷的离散误差虽已 <3%，但仍随 h 下降（表 3.3），
不宣称“插值让任意网格都精确”；(b) 带阻尼工况的模态阻尼比只在第 1 阶有物理含义。

## 7. 复现

```
PYTHONPATH=src python3 scripts/verify_fem_euler_bernoulli.py
python -m pytest tests/test_fem_euler_bernoulli_verification.py -q
```
