# p256w1-N04 - 冰-船碰撞应变硬化与冰压驱动损伤几何

**节点** `p256w1-n04-ice-collision-hardening` - 上层战役 DAG 节点，物理域=冰区结构/碰撞力学（吸能与损伤几何）。

**本轮判定：`PASS_WITH_WITHHELD`**（failed = []，withheld = ['criterion_d_minorsky_recalibration']）。

上游基线：`IN63-N01`（`prereg/IN63_N01_hull_collision_energy.json`）的刚塑性铰 + 双层壳串联 + Minorsky 交叉检验面。本节点把该面升级为**双线性应变硬化铰 + ISO 19906 压力-面积冰载驱动压痕 + 冰区重标定碰撞系数**，并保持 N01 为模型的严格极限。

## 1. 案例介绍（物理背景 / 几何 / 参数表）

被撞舷侧为双壳油船/散货船冰区加强结构（PC5 级），撞击体为浮冰/冰岩。N01 把冰当作刚体、把铰当作理想塑性、把损伤足迹当作等效矩形 `L_c + 2*span`；本节点引入三处物理：

1. **铰本构**：双线性硬化 `sigma = sigma_y + E_h*(eps - eps_y)`（`E_h/E` 为硬化比），另提供幂硬化 `sigma = sigma_y + K*eps_p^(1/n)` 作敏感性；
2. **冰侧载荷**：ISO 19906 压力-面积幂律 `p(A) = p0*(A/A_ref)^(-alpha)`，`alpha` 取 0.40-0.60 家族，接触面积随压痕增长 `A = H*d`；
3. **损伤长度**：由压力积分给出的屈服条长度 `L_y = N_ice/(kappa1*(1+dM_h/M_p) + 2*kappa2*d)` 取代 N01 的椭圆/等效矩形近似。

| 参数 | 值 | 出处 |
|---|---|---|
| 外壳板厚 t_out | 0.0220 m | 工况数据（scenario.py）|
| 内壳板厚 t_in | 0.0180 m | 同上 |
| 肋距 span | 2.40 m | 同上 |
| 屈服强度 sigma_y | 355.0 MPa | AH36 |
| 硬化比 E_h/E | 0.010 | 双线性拟合 0.005-0.02 中值 |
| 转动容量 eps_u | 0.20 | 断裂总应变 |
| 塑性铰长度下限 | 0.0220 m | L_h = max(t, (t/2)theta/eps_u) |
| 冰压参考 p0 (A_ref=1 m^2) | 3.00 MPa | ISO 19906 一类冰全球压力量级 |
| 压力-面积指数 alpha | 0.50 | ISO 19906/Masterson 家族 |
| 接触高度 H / 长度 L_c | 3.00 / 3.00 m | 工况 |
| 有效质量 m_eff | 2200000 kg | 撞击质量 + 附连水 10% |
| 冰-钢摩擦 mu | 0.15 | N01 同值 |
| 外壳铰/膜系数 k1, k2 | 1.181e+07 N, 9.762e+06 N/m | 由母材与骨材剖面算出 |
| 单位长度铰/膜系数 kappa1, kappa2 | 3.938e+06, 3.254e+06 | k1/W_eff, k2/W_eff |
| 外壳/内壳吸能容量（含硬化）| 81.85 / 20.18 MJ | 本节点 |

## 2. 怎么做的（方法 / 公式 / 工具链）

### 2.1 双线性硬化铰（`src/tensordem/hull_collision/strain_hardening.py`）

矩形板条剖面（半深 `c=t/2`，单位宽），曲率 `kappa`、弹性核半深 `a = eps_y/kappa`。精确剖面积分给出硬化部分弯矩（无级数展开）：

```
dM_h(kappa) = M_bilinear(kappa) - M_perfectly_plastic(kappa)
            = E_h*[(2c^3/3)kappa - eps_y*c^2 + eps_y^3/(3 kappa^2)]
```

对转动积分的闭式：

```
dE_h(theta) = L_h*E_h*[(c^3/3)kappa^2 - eps_y*c^2*kappa - eps_y^3/(3 kappa) + eps_y^2*c]
```

**退化性**：`E_h = 0` 时 `dM_h` 与 `dE_h` 均返回字面 `0.0`，层能量/抗力逐位回到 N01。塑性铰长度按转动容量封顶：`L_h(theta) = max(k_h*t, (t/2)*theta/eps_u)`，即峰值纤维应变不超过 `eps_u`，超出后铰扩展而不再继续硬化（不再上升的硬化弯矩）。

层级闭合（文档化假设）：板条剖面硬化比 `Phi(theta) = dE_h/(M_p*theta)` 乘以该层总铰能，即 `E_layer = k1*u + k1*(span/8)*(dE_h/M_p) + k2*u^2`；其导数恰为 `k1*(1 + dM_h/M_p) + 2*k2*u`。纵向骨材剖面更深、相对硬化储备更大，故该闭合给出**下界**。

### 2.2 冰侧：ISO 19906 压力-面积（`indentation.py`）

```
p(A) = p0*(A/A_ref)^(-alpha),  A(d) = H*d
N_ice(d) = p0*A_ref^alpha*H^(1-alpha)*d^(1-alpha)
E_ice(d) = p0*A_ref^alpha*H^(1-alpha)*d^(2-alpha)/(2-alpha)
```

### 2.3 能量账本与求解

```
E_k = E_steel(d) + E_friction(d) + E_ice(d) + residual,   E_friction = mu*E_steel
L_y  = N_ice / (kappa1*(1+dM_h/M_p) + 2*kappa2*d),   L_damage = max(L_c, L_y)
```

对 `d` 在 `[0, d_max]` 上做夹逼二分（与 N01 同一算法，tol 1e-12 m），残差如实上报，0 速度严格返回 `d = 0`。力-时程由能量参数化给出：`F(d)=R_struct(d)+N_ice(d)`，`v(d)=sqrt(2(E_k-E_abs(d))/m_eff)`，`t(d)=int dd/v`（准静态，不是动力仿真）。

### 2.4 工具链与复现命令

```
python -m pytest tests/test_ice_collision_hardening.py -q
python scripts/benchmark_ice_collision_hardening.py --output runs/ice_collision_hardening
python scripts/p256w1_n04_doc.py --report runs/ice_collision_hardening/report.json \
       --output docs/P256W1_N04_ICE_COLLISION_HARDENING.md
```

零领域硬编码进平台仓：本节点所有模块在 TensorDEM 域内，P0/alpha/硬化比均为算例数据。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 验收判据

| 判据 | 结果 | 关键数字 | 阈值 |
|---|---|---|---|
| (a) 硬化->0 退化到 N01 | PASS | 最大相对差 0.000e+00 | 1.000e-03 |
| (b) 硬化增量 vs 闭式 | PASS | 能量相对差 2.395e-05，弯矩 1.268e-08 | 2.000e-02 |
| (b') 上界紧致性 | PASS | 占刚硬化上界 0.9855，充分塑性时松弛 0.0163 | <= 0.10 |
| (c) 能量守恒残差 | PASS | 最大相对残差 1.544e-12 | 1.000e-06 |
| (d) Minorsky 重标定残差 | WITHHELD | 船-船 n=0，冰区 n=0 | 中位相对残差 <= 0.15 |
| (e) 压力-面积指数扫描 | PASS | 冰能闭式 vs Simpson 1.494e-07；标度指数相对差 9.560e-16 | 1e-06 / 2e-02 |
| (f) 负控（复现 N01）| PASS | 深度/能量/几何最大相对差 0.000e+00 | 1.000e-09 |

**(d) 的状态是 WITHHELD，理由（如实申报）**：collision case table not present in the repository and no network access in the build environment; no numbers are invented。本仓没有 Minorsky(1959) 原表、也没有冰区碰撞记录表，构建环境无外网；**没有数据就不编数字**。取而代之，下面给出模型侧的系数修正形式与不确定度包络，并把拟合代码（`recalibration.recalibrate_coefficient`）用合成算例单测证明可用（`test_recalibration_recovers_a_known_coefficient_on_synthetic_cases`）。

### 3.2 速度扫描（硬化 1% + 冰压 alpha=0.50）

| V [m/s] | E_k [MJ] | d [m] | d(N01) [m] | N_ice [MN] | L_y [m] | L_damage [m] | V_damage [m^3] | V_N01 [m^3] | 钢/摩/冰 份额 | 峰值纤维应变 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 4.40 | 0.1821 | 0.2656 | 2.22 | 0.271 | 3.00 | 0.2977 | 0.7741 | 0.816/0.122/0.061 | 0.200 |
| 4.0 | 17.60 | 0.5535 | 0.7855 | 3.87 | 0.364 | 3.00 | 0.2977 | 0.7741 | 0.799/0.120/0.081 | 0.200 |
| 6.0 | 39.60 | 1.0266 | 1.3681 | 5.26 | 0.385 | 3.00 | 0.2977 | 0.7741 | 0.790/0.119/0.091 | 0.200 |
| 8.0 | 70.40 | 1.5489 | 1.9711 | 6.47 | 0.378 | 3.00 | 0.2977 | 0.7741 | 0.787/0.118/0.095 | 0.200 |

**读法**：硬化 + 冰侧吸能一并把压痕深度压低（相对 N01 刚体压头），而等效箱式损伤体积在 N01 模型里与深度无关（Minorsky 的 R_T 是材料体积，不是压痕体积），因此系数修正来自两处独立来源：①能量份额重新分配（钢份额被冰/摩擦分走）；②压力积分把损伤长度从 `L_c+2*span` 换成 `max(L_c, L_y)`。

### 3.3 硬化分配（内外壳按铰长与硬化）

| 壳 | 局部压痕 [m] | 含硬化吸能 [MJ] | 理想塑性吸能 [MJ] | 硬化增量 [MJ] | 硬化占比 |
|---|---|---|---|---|---|
| inner_shell | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 |
| outer_shell | 1.5489 | 55.411 | 41.720 | 13.691 | 0.247 |

（上表为 V = 8 m/s 算例；内壳该速度下未进入，故为 0，与 N01 的内壳完整判据一致。）

### 3.4 压力-面积指数扫描与标度指数

| alpha | 冰能闭式 [MJ] | Simpson [MJ] | d~E^beta 数值 | ISO 1/(2-alpha) | L_y~E^gamma 数值（铰控）| ISO -alpha/(2-alpha) |
|---|---|---|---|---|---|---|
| 0.40 | 0.394 | 0.394 | 0.62500 | 0.62500 | -0.25000 | -0.25000 |
| 0.45 | 0.413 | 0.413 | 0.64516 | 0.64516 | -0.29032 | -0.29032 |
| 0.50 | 0.433 | 0.433 | 0.66667 | 0.66667 | -0.33333 | -0.33333 |
| 0.55 | 0.454 | 0.454 | 0.68966 | 0.68966 | -0.37931 | -0.37931 |
| 0.60 | 0.477 | 0.477 | 0.71429 | 0.71429 | -0.42857 | -0.42857 |

ISO 家族区间（由 alpha 家族导出）：beta in [0.625, 0.7142857142857143]；数值指数全部落在区间内：True。

**耦合模型（钢 + 冰同解）的指数另行如实报告**（不塞进区间）：

| alpha | d~E^beta（耦合）| L_y~E^gamma（耦合）| 冰份额范围 |
|---|---|---|---|
| 0.40 | 0.7691 | 0.1962 | 0.054-0.103 |
| 0.45 | 0.7713 | 0.1581 | 0.058-0.099 |
| 0.50 | 0.7735 | 0.1198 | 0.061-0.095 |
| 0.55 | 0.7757 | 0.0814 | 0.065-0.091 |
| 0.60 | 0.7779 | 0.0427 | 0.069-0.088 |

耦合指数高于 ISO 冰单独渐近值（约 0.63-0.71 -> 约 0.77），原因是设计算例中外壳铰项（对 d 线性）与膜项共同吸能，冰只占 6-10%；这说明**冰压指数只有在冰主导工况下才是压痕深度的控制参数**，不能把它当作设计算例的直接标度律。

### 3.5 碰撞系数 C(xi, 冰面)

定义：`V_damage = C * M * V^2 / sigma_y`，`C_model = V_damage*sigma_y/(M*V^2)`（逐算例反演）。

- 绝对包络（xi x p0 x alpha 共 80 例）：min 2.0044，中位 3.5021，max 4.9998（相对半宽 0.428）；
- 相对 N01 系数的比值（修正式）：中位 0.3846，区间 [0.3846, 0.3846]；
- 逐 xi 的绝对值（= C 对板厚的近似线性依赖）：
  - xi = 0.00458: C = 2.0044，比值 0.3846
  - xi = 0.00917: C = 3.0029，比值 0.3846
  - xi = 0.01375: C = 4.0013，比值 0.3846
  - xi = 0.01833: C = 4.9998，比值 0.3846

**物理解释（必须与数字一起读）**：系数比值 0.385 = `L_c/(L_c+2*span)`，即压力积分判定「接触区之外不再有整跨屈服」，而 N01 的 `+2 span` 是运动学猜测、不是算出来的量。该比值对冰面参数不敏感，是因为等效箱体积与压痕深度无关；冰面参数改变的是**深度**（见 3.2），不是 Minorsky 型材料体积。因此 C 的「重标定」在没有实测数据的条件下只能给出**结构化修正式 + 包络**，不能给出残差中位数——这就是判据 (d) 记 WITHHELD 的原因。

## 4. 结果图（matplotlib 生成，sha256 见审计链）

**`figures/force_history.png`** (sha256 `c9afefce8ea7a240...`) - Left: contact force vs penetration. Right: quasi-static force-time history (V = 4 / 8 m/s, hardening ratio 1 %, alpha = 0.5).

**`figures/hardening_law.png`** (sha256 `3afa84d4a4adf6c3...`) - Bilinear hinge: hardening moment ratio dM_h/M_p and hardening energy increment per unit width, closed form vs independent fibre/Simpson route vs the rigid-hardening bound.

**`figures/depth_volume_comparison.png`** (sha256 `b855a0e5524ec830...`) - Penetration depth and damaged steel volume vs impact velocity: N01 rigid-indenter surface, ice side only, and full hardening + ice model.

**`figures/exponent_and_coefficient.png`** (sha256 `d6158b387e5fa53d...`) - Left: penetration-depth scaling exponent vs pressure-area exponent (numerical ISO asymptote, closed form, coupled model, ISO family interval). Right: recalibrated collision coefficient C over the xi x ice-face box.

## 5. 注意点（踩坑 / 局限 / 适用范围）

- Bilinear hardening applied with the *plating* section ratio on the whole hinge term of a layer; the deeper longitudinal sections carry a larger relative hardening reserve, so the layer hardening increment is a lower bound (documented closure).
- Hinge length is capped by the rotation capacity (peak fibre strain <= 0.20); beyond that the hinge spreads instead of hardening further. Fracture/tearing is NOT modelled, so rotations past the capacity are a modelling extrapolation.
- No strain-rate sensitivity on steel or ice, no ice fracture/rubble, no elastic contact compliance, quasi-static monotone indentation.
- The pressure-area law is applied to the nominal contact area of the growing indentation (global relation); local ice-feature pressure spikes are not resolved.
- The empirical Minorsky recalibration residual is WITHHELD: no collision case table in the repository and no network access; C(xi, ice face) and its envelope are reported instead of a fitted residual.
- Quasi-static force-time history is energy-parameterised, not a dynamic impact simulation (no added-mass coupling in the deceleration).
- 数值实现层面踩过的两个坑（已由单测固定）：①硬化能到层尺度的换算必须带 `span/8`（`dθ/du`），否则层抗力与能量导数不一致（`test_hardened_layer_resistance_is_energy_derivative`）；②压力积分给出的屈服条长度是**单位长度抗力**（导数形式 `kappa1+2*kappa2*d`），写成 `kappa1*d+kappa2*d^2` 会量纲错位（`test_structural_coefficients_reproduce_n01_layer_resistance`）。
- 压痕深度在本算例达到 0.18-1.55 m，峰值纤维应变被转动容量封顶在 0.20；超出后真实结构会撕裂/断裂，本模型不含断裂能，故大变形段应视为外推。

## 6. 误差判定（是否 < 3%？逐项回答）

| 关键量 | 基准 | 相对误差 | 是否 < 3% |
|---|---|---|---|
| 硬化=0 时的深度/能量抗力量 | N01 `solve_penetration` | 0.000e+00 | 是（精确 0）|
| 硬化能量增量（闭式）| 纤维积分 + Simpson 独立路线 | 2.395e-05 | 是 |
| 硬化弯矩增量（闭式）| 剖面纤维积分 | 1.268e-08 | 是 |
| 能量守恒残差 | 自身账本（E_k）| 1.544e-12 | 是 |
| 冰能闭式 | Simpson 数值积分 | 1.494e-07 | 是 |
| 标度指数 | ISO 19906 闭式指数 | 9.560e-16 | 是 |
| 负控几何/能量 | N01 等效箱 | 0.000e+00 | 是（精确 0）|
| Minorsky 重标定残差中位数 | 实测碰撞表 | 无基准（n=0）| **无法判定**：数据缺失，如实记 WITHHELD，不粉饰 |

**结论**：凡有独立基准的量，误差均在 3% 以内（见表）；唯一没有基准的是 (d) 的经验残差，缺数据即缺判据，未用任何替代数字充数。

## 7. 审计链

- 报告内容 sha256：`88c0594d3ffc2f7b58c886ce409fa717329ef718a298180db912708d9f982a18`
- 源码 sha256：
  - `scripts/benchmark_ice_collision_hardening.py`: `f4540c2f161bda8dc8d5bea1cd8795e2...`
  - `src/tensordem/hull_collision/indentation.py`: `0a42130c815623f20f8243f94ce9ec53...`
  - `src/tensordem/hull_collision/minorsky.py`: `02b20c9cd40c4cb29adeaaa946e6d4bd...`
  - `src/tensordem/hull_collision/plastic_hinge.py`: `0415cf2bb9fc49ab846da6fad30ecb66...`
  - `src/tensordem/hull_collision/recalibration.py`: `5332bdb91d95e66b674addec3efe3a6f...`
  - `src/tensordem/hull_collision/resistance.py`: `17db5c7948b0e00c9b1fa2db73410d68...`
  - `src/tensordem/hull_collision/scenario.py`: `d96e36f8d471d4586966c778813e7056...`
  - `src/tensordem/hull_collision/strain_hardening.py`: `f10356bf569323aa5ff37f908f663fdb...`
  - `tests/test_ice_collision_hardening.py`: `64e0d88bb1d3d510f7885a6c08aecab3...`
- 证据包：`/root/tp-agent/work/m9796/runs/ice_collision_hardening/report.json`（含 sweep.csv / coefficient_grid.csv / force_history.csv / figures/*.png）

