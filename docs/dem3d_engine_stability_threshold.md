# 显式时间步稳定性阈值：求解器级动态验证

状态：把 `docs/dem3d_timestep_stability.md` 中登记为
`Engine-level threshold crossing verified dynamically  【未验证】` 的
那一行债务**结清**。这是关于积分器的数值稳定性证据，**不是**材料标定，
也不是对海冰力学行为的物理验证。

关联文档：`docs/dem3d_timestep_stability.md`（闭式放大因子与 Gershgorin 界）、
`src/tensordem/stability.py`（被判据的判据本身）、
`src/tensordem/stability_verification.py`（本文的工具）、
`benchmarks/engine_stability_threshold_audit.py`（产出本文全部数字与图）、
`tests/test_dem3d_engine_stability_threshold.py`（12 条可失败断言）。

## 1. 案例与物理背景

`tensordem.dem3d.IceDEM3D` 用半隐式（辛）Euler 显式积分：

```
v_{n+1} = v_n + dt f(x_n, v_n)/m ,    x_{n+1} = x_n + dt v_{n+1}
```

`src/tensordem/stability.py` 已给出该格式对单自由度问题 `f = -k x - c v` 的
**闭式**放大因子 `rho(dt, omega, gamma)`、临界步长 `dt_crit = min(2/omega, 2m/c)`
以及多体系统的 Gershgorin 频率上界。但是：

* `DEM3DConfig.recommended_dt` 把生产步长压在审计限值的约 2%；
* `IceDEM3D.__init__` 直接**拒绝**任何超过 `recommended_dt` 的步长。

结果是阈值的**失效侧**（`dt > dt_crit` 扰动增长）在求解器里从未被观测到，
判据的证据只停留在闭式推导与独立 SDOF 积分器上。本文补上这一段。

### 验证对象（几何与参数）

| 项 | 值 |
|---|---|
| 模型 | `IceDEM3D`，3x3x3 = 27 球，158 根黏结 |
| 半径 / 密度 | 0.025 m / 917 kg/m^3（冰） |
| 单粒子质量 | 0.060017510 kg（`DEM3DConfig.mass`）|
| 黏结法向刚度 k_b | 2000 N/m |
| 阻尼 | 拖曳 0、配对阻尼 0、摩擦 0、滚动阻力 0 |
| 约束 | 全自由（`fix_x_edges=False`、`fix_bottom=False`），工具头移出接触范围 |
| 断裂阈值 | `breaking_strain = 1.0`、`shear_breaking_strain = 1.0`（探针应变 ~2e-5，永不触发）|

于是只有黏结弹簧做功，模型是**精确的线性振子网络**：`dt_crit` 是格式的性质，
与任何事件无关。探针幅值 1e-6 m（应变 ~2e-5），保证在线性区。

## 2. 方法

1. **独立参考（不调求解器）**：`assemble_stiffness` 由黏结表和中心力定律重新组装
   平移刚度阵 `K`（`K_ii += k nn^T`、`K_ij -= k nn^T`），再用 `torch.linalg.eigh`
   求 `omega_max = sqrt(lambda_max/m)`。这条路完全绕开 `forces()`。
2. **引擎侧同算子**：`stiffness_apply` 把位移场喂给 `IceDEM3D.forces`
   （`update_fracture=False, update_history=False`）得到 `K u`，再做幂迭代 +
   Rayleigh 商，得到引擎自己看到的 `omega_max` 与模式残差。两者一致
   才说明力律确是该能量的梯度。
3. **引擎每步放大因子**：在最高模态子空间用两次 `IceDEM3D.step` 探测
   `(x=1,y=0)` 与 `(x=0,y=1)` 两个基态，取回四个矩阵元，按 `trace/det`
   重构 `rho_measured`，与闭式 `rho` 逐点比对。
4. **阈值独立定位**：只对 `sign(rho_measured - (1+1e-6))` 做二分，先不引入任何
   解析数，最后才与 `2/omega_max` 比对。
5. **长时程**：1.05 `dt_crit` 用闭式主特征方向起步，测每步增长与 `rho`；
   0.95 `dt_crit` 跑 2000 步，用格式的**精确二次不变量** `A^T P A = P`
   （`invariant_form`）度量有界性——因为在 `dt -> dt_crit` 时物理能量本身会以 O(1)
   幅度振荡，不是守恒量。
6. **随机扰动**（非特征向量）再做一次轨迹级能量有界/发散对照。

`dt > recommended_dt` 的步长通过新增的 `DEM3DConfig.allow_dt_above_recommended`
（默认 `False`，仅验证用逃生舱；构造时把 `dt_guard_bypassed` 记进模型）进入，
生产路径的守卫**未被削弱**，有回归测试守着。

## 3. 结果（由 `benchmarks/engine_stability_threshold_audit.py` 程序化生成）

### 1. Highest eigenfrequency: three routes to the same number

| route | omega_max (rad/s) | relative deviation |
|---|---:|---:|
| bond-lattice assembly + `eigh` (no solver call) | 604.241630 | reference |
| engine `forces()` power iteration (Rayleigh) | 604.241630 | 9.08e-12 |
| Gershgorin upper bound (`stability.stability_report`) | 1316.369115 | +1.18e+00 |

Engine mode residual `||K u - lambda u|| / (lambda ||u||)` = 1.25e-11; particles = 27, bonds = 158, m = 0.060017510 kg, k_b = 2000 N/m, dt_crit = 2/omega_max = 3.309934141e-03 s, engine `recommended_dt` = 7.138505650e-05 s (= 0.022 dt_crit), top-eigenvalue degeneracy gap = 0.0e+00.

### 2. Measured per-step amplification versus the closed form

`grows` means rho > 1 + 1e-06 (the undamped scheme is marginally stable below the threshold, so rho = 1 exactly).

| dt/dt_crit | dt (s) | rho engine | rho closed form | rel. error | engine verdict | guard would refuse |
|---:|---:|---:|---:|---:|:--|:--|
| 0.100 | 3.309934e-04 | 1.000000000 | 1.000000000 | 1.05e-10 | bounded | yes |
| 0.500 | 1.654967e-03 | 0.999999999 | 1.000000000 | 7.35e-10 | bounded | yes |
| 0.900 | 2.978941e-03 | 0.999999999 | 1.000000000 | 1.44e-09 | bounded | yes |
| 0.990 | 3.276835e-03 | 1.000000001 | 1.000000000 | 8.75e-10 | bounded | yes |
| 0.999 | 3.306624e-03 | 1.000000002 | 1.000000000 | 1.80e-09 | bounded | yes |
| 1.001 | 3.313244e-03 | 1.093556540 | 1.093556542 | 2.13e-09 | grows | yes |
| 1.010 | 3.343033e-03 | 1.326584437 | 1.326584427 | 7.29e-09 | grows | yes |
| 1.050 | 3.475431e-03 | 1.877328047 | 1.877328045 | 1.09e-09 | grows | yes |
| 1.200 | 3.971921e-03 | 3.471979900 | 3.471979899 | 9.55e-11 | grows | yes |
| 1.500 | 4.964901e-03 | 6.854101967 | 6.854101966 | 6.62e-11 | grows | yes |

### 3. Threshold located from engine measurements alone

Bisection on `sign(rho_measured - 1)` over 30 engine evaluations gives dt* = 3.309934143e-03 s against the closed-form 2/omega_max = 3.309934141e-03 s (relative error 4.66e-10, bracket width 3.08e-12 s).

### 4. Long runs on both sides of the threshold

| case | dt/dt_crit | steps | rho closed form | measured per-step growth | rel. error | statement |
|---|---:|---:|---:|---:|---:|:--|
| 0.95 dt_crit | 0.950 | 2000 | 1.000000000 | n/a (marginal) | — | no growth: exact quadratic-invariant drift 5.14e-07 over 2000 steps |
| 1.05 dt_crit | 1.050 | 14 | 1.877328045 | 1.877328044 | 6.62e-10 | amplitude 1.000e+00 -> 1.5315e+01 (closed-form prediction 1.5315e+01) |

The 1.05 run stops itself once the mode amplitude leaves the linear regime (peak projection > 10 probe units, i.e. bond strain > 2e-4); an unbounded exponential excursion is a *linear-algebra* statement and is not claimed beyond that window.

### 5. Generic random perturbation (not the eigenvector)

Mechanical-energy ratio to its own initial value, `0.5 (u.Ku/m + |v|^2)`:

| case | dt/dt_crit | steps | energy ratio end | energy ratio peak | energy ratio min | bounded |
|---|---:|---:|---:|---:|---:|:--|
| 0.95 dt_crit | 0.950 | 800 | 9.266748e-01 | 1.336753e+00 | 3.317938e-01 | yes |
| 1.05 dt_crit | 1.050 | 200 | 2.044108e+10 | 6.678913e+10 | 1.000000e+00 | no |

On the stable side the plain energy is *not* a conserved quantity of the scheme (the modified energy is); it oscillates, which is why boundedness is asserted with the quadratic invariant of section 4, not with this table.

## 4. 注意点与局限

* **单模态**：本文证明的是"引擎实际所取的步长，对其自身晶格的最高模态服从闭式阈值"，
  这正是 Gershgorin 界所声称的命题；不声称任意非线性轨迹在 `dt_crit` 之上必然发散。
* **1.05 `dt_crit` 的长跑会自我截断**：模态幅值一旦离开线性区（探针单位 > 10，
  即黏结应变 > 2e-4）就停止，因为指数发散是线性代数命题，超出该窗口不主张。
  实测每步增长与 `rho` 的偏差在这 14 步内是 6.6e-10。
* **0.95 `dt_crit` 侧是临界稳定（`rho = 1` 精确）**，不是衰减；判"有界"必须用
  格式不变量，不能用物理能量。2000 步不变量漂移 5.14e-7 来自探针幅值的
  二次非线性（每步 ~4e-10 累积），不是格式误差。
* **逃生舱的纪律**：`allow_dt_above_recommended=True` 会让 `IceDEM3D` 接受
  远超 `recommended_dt` 的步长。它只用于验证；任何物理算例必须保持 `False`。
  `tests/test_dem3d_engine_stability_threshold.py` 锁住了默认行为。
* 几何为立方晶格、单材料、无重力；`dt_crit` 随 `k_b`、`m` 缩放，本文只测一组
  参数（端口比 `k_b/m` 决定），不同刚度组合需要重跑本审计。

## 5. 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 参照 | 相对误差 | 是否 < 3% |
|---|---|---|---|---|
| `omega_max`（引擎 `forces()` 幂迭代 + Rayleigh） | 独立刚度阵 `eigh` | 604.241630 rad/s | 9.08e-12 | ✅ |
| 每步放大因子 `rho`（引擎 2x2 探针，10 个 `dt/dt_crit`） | 辛 Euler 闭式 | 见上表 | ≤ 7.3e-9 | ✅ |
| 阈值 `dt*`（引擎二分，30 次评估） | `2/omega_max` | 见上表 | 4.66e-10 | ✅ |
| 1.05 `dt_crit` 每步增长（14 步，线性区） | `rho` 闭式 | 1.877328045 | 6.62e-10 | ✅ |
| 1.05 `dt_crit` 幅值（14 步） | `rho^n` 闭式 | 1.5315e+01 | < 1e-9 | ✅ |
| 0.95 `dt_crit` 有界性（2000 步） | 二次不变量守恒 | 1 | 漂移 5.14e-07 | ✅（不变量，非物理能量）|
| 有界侧物理能量 | 非守恒量 | — | 振荡 0.33–1.34 | 不适用（说明于 §4）|

Gershgorin 上界比精确 `omega_max` 高 118%，这是**上界**的应有行为（保守），
不是误差：`recommended_dt` 相当于 0.022 `dt_crit`，与文档记载一致。

## 6. 复现

```bash
PYTHONPATH=src python3 benchmarks/engine_stability_threshold_audit.py \
    --output artifacts/engine_stability_threshold
PYTHONPATH=src python3 -m pytest tests/test_dem3d_engine_stability_threshold.py -q
```

工件：`artifacts/engine_stability_threshold/{audit.json,summary.md,engine_threshold.png}`
（`summary.md` 即本节 §3 的表格，由脚本原样写入；`audit.json` 为机器可读明细）。
