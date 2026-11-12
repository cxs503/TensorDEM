# H1-BROKEN-ICE-DEM · 浮冰垂荡瞬态的解析验证（verifier 席 m16270）

**对象**：`H1-BROKEN-ICE-DEM`（功能点锚点，status=in_progress）中的浮沉判据 C1，
即 `src/tensordem/polyhedron_floe3d/h1_cases.py::buoyancy_equilibrium_case`
（m14948 交付、m15029 审计裁定 request changes / L2）。

**本单定位：验证欠账**。已合并的 H1 测试只断言 C1 的**终点**（
`tests/test_h1_broken_ice_dem.py::test_buoyancy_equilibrium_matches_archimedes`
比对 `rho_ice/rho_water`）。终点相等是静态陈述：**恢复刚度错、阻尼系数错、
能量台账错，三者都能仍然收敛到正确吃水**。本单补上瞬态的闭式对照。

## 1. 物理背景与可积算例

竖直侧面棱柱浮冰（水平截面积恒定 `A`），自水线处静止释放，垂荡满足

```
m z'' = -rho_w g A (z - z_eq) - c z'         （模态选择：线性/Stokes 型水动阻力）
omega_0 = sqrt(rho_w g A / m),  gamma = c/(2m),  omega_d = omega_0 sqrt(1 - zeta^2)
z_eq    = h/2 - (rho_i/rho_w) h              （Archimedes 吃水）
```

**关键**：`buoyancy_equilibrium_case` 在求解前就按
`c = f_d · 2 m omega_0`（`f_d = 0.7`）设定阻力，因此参考解的**每一个系数都由算例
设置固定**，不是事后拟合出来的。几何/物性表：

| 量 | 值 | 出处 |
|---|---|---|
| 水平截面积 A | 0.36 m² | 算例设置 |
| 厚度 h | 0.12 m | 算例设置 |
| 棱柱边数 n_sides | 16 | 算例设置 |
| rho_ice / rho_water | 917 / 1025 kg/m³ | 算例设置 |
| 质量 m | 39.61 kg | `A·h·rho_ice` |
| 水线刚度 k = rho_w g A | 3619.0 N/m | 解析 |
| 阻尼 c | 530.4 N·s/m（ζ=0.7） | 算例设置 |
| z_eq 解析 | -0.04743 m | Archimedes |
| dt（基准） | 5e-4 s，4000 步（2 s） | 本单 |

## 2. 怎么做的

`scripts/h1_heave_analytic_verification.py`（协议
`tensordem-h1-heave-analytic-verification-v1`）：

1. 用主 DEM 求解器 `DEMSystem3D` + `equivalent_disc_prism` 复算同一浮冰，逐时间步记录
   `z(t)`、`v(t)`、`dissipated`；
2. 对 `z(t) - z_eq` 做最小二乘拟合 `A e^{-gamma t} cos(omega_d t + phi) + off`（`scipy.optimize.curve_fit`）；
3. 由拟合量**反推**刚度与阻尼：`k_fit = m(omega_d² + gamma²)`，`c_fit = 2 m gamma`，与解析值比较；
4. 台账交叉核对：把记录到的 `v(t)` 做梯形积分 `∫ c v² dt`，与求解器自己的 `dissipated`
   计数器、以及无量纲 `energy_residual` 比较；
5. 时间步收敛：dt / dt/2 / dt/4（步数同步加倍）看误差阶；
6. 负对照：拟合器对合成阻尼振子（已知 `gamma=3.1, omega=7.3`）的还原精度。

## 3. 计算结果（`artifacts/h1_broken_ice_dem/heave_analytic_verification.json`，程序化注入）

| 观测量 | 解析/参考值 | 拟合/实测 | 相对误差 | 判定 |
|---|---|---|---|---|
| 垂荡固有频率 omega_0 | 9.5607 rad/s | 9.5447 rad/s | **0.168 %** | PASS <3 % |
| 恢复刚度 k | 3619.0 N/m | 3606.8 N/m | **0.336 %** | PASS <3 % |
| 衰减率 gamma | 6.6925 1/s | 6.6700 1/s | **0.336 %** | PASS <3 % |
| 阻尼系数 c | 530.4 N·s/m | 528.6 N·s/m | **0.336 %** | PASS <3 % |
| 平衡 z_eq | -0.047432 m | -0.047432 m | **0.00022 %** | PASS <1 % |
| 阻力做功 ∫c v²dt | 4.059006 J | 台账 4.058983 J | **0.00057 %** | PASS <2 % |
| 能量残差（无量纲） | — | 5.96e-10 | — | PASS <1e-2 |

时间步收敛（同样的观测量）：

| dt [s] | omega_0 误差 | gamma 误差 | z_eq 误差 |
|---|---|---|---|
| 5e-4 | 0.168 % | 0.336 % | 0.00022 % |
| 2.5e-4 | 0.0838 % | 0.168 % | 0.00021 % |
| 1.25e-4 | 0.0419 % | 0.0837 % | 0.00021 % |

误差随 dt 减半而减半（观测阶 ≈1.0，`tests/test_h1_heave_analytic.py::
test_frequency_error_halves_with_time_step` 断言比值落在 1.6–2.6）——说明那 0.17 %/0.34 %
**全部**是时间积分的离散误差，不是物理模型误差；平衡值是线性解，任何 dt 下都精确到舍入。
**换句话说：把刚度或阻尼改错 1 %，本套门槛就会红。**

图：`artifacts/h1_broken_ice_dem/fig_heave_analytic.png`（左=实测 z(t) vs 闭式解；
右=包络与拟合指数衰减）。

## 4. 误差判定（显式回答）

- 关键量（恢复刚度 / 阻尼 / 平衡吃水）vs 闭式解：**0.336 % / 0.336 % / 0.00022 %，
  均 <3 %，也 <5 %**，判据满足。
- 台账 vs 解析积分：0.00057 % <2 %。
- 负对照（拟合器本身）：合成信号的 gamma/omega 还原相对误差 <1e-5（测试断言 `rel=1e-5`）。
- 无基准项：C2 休止角的**文献带**（25–40°）本仓无冰区实测数据集，仍为 WITHHELD
  （见 §5），未用任何实验值。

## 5. 注意点 / 局限（诚实披露）

1. **线性阻力是模态选择，不是材料常数**（`F = -c v`）。本单验证的是"求解器在给定 c 下
   是否精确复现该闭式解"，即刚度/阻尼/台账的**实现正确性**；它**不**证明 Stokes 型线性
   阻力是冰-水阻力的正确物理。非线性阻力（`v²` 型）没有闭式解，本单不涉及。
2. **小振幅范围**：释放位移 0.0474 m，潜没深度在 0.0126–0.1074 m 内，全程保持
   `0 < 潜没 < h`，水线面积恒为 A，所以闭式解严格成立；一旦出水/全淹（水面面积跳变），
   瞬态分段非线性，本线性参照失效。
3. **第一阶收敛**是这一速度相关力 + kick-drift 格式的预期阶；未来若换格式（如隐式或
   RK4），预期阶应随之改变，断言需同步更新（断言只约束"误差随 dt 单调缩小且比值在
   1.6–2.6"，不绑定实现）。
4. **C2 休止角（本锚点唯一未达项）**：audit m15029 的裁定是 honest FAIL
   （0–16.27°，未静止）。本单把审计建议的加时长扫描（45 体 × 3 个 mu × 12000 步 × dt=2e-4）
   **递交集群**（`cj1294`，TensorDEM@main，3 分片）。**截至本报告提交时该作业仍在排队
   （12:11 递交，仓内 460 单在前），未取回结果**——因此 C2 在本单**记为 not_run/待定**，
   不填任何数字、不覆盖 m15029 的 honest FAIL。
5. 审计遗留项「`artifacts/h1_broken_ice_dem/` 未提交」：本单已把可复现生成的
   `results.json` + 图与本单的 `heave_analytic_verification.json` 一并入库，证据不再只在
   按需产出里。

## 6. 复现

```bash
PYTHONPATH=src python scripts/h1_heave_analytic_verification.py      # 写 json + png，退出码=门槛
PYTHONPATH=src python -m pytest tests/test_h1_heave_analytic.py -q     # 6 passed
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py --quick
# 集群（C2 加时长扫描，结果待取回）
# run_cluster: repo=TensorDEM ref=main shards=3
#   PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py --cases repose \
#     --repose-shard {i} --repose-bodies 45 --repose-steps 12000 --repose-dt 2e-4 \
#     --out repose_shard_{i}.json
```