# 冰-结构接触界面相变热-力耦合：Stefan 条件与摩擦热熔融的 DEM-FVM 双向耦合

模块 `src/tensordem/interface_thermal/`（子模块 `conduction` / `stefan` /
`friction_heat` / `lubrication` / `coupling`）。驱动脚本
`benchmarks/interface_thermal_coupling.py`，数据 `artifacts/interface_thermal_coupling.json`，
图 `docs/interface-thermal/figures/`，摘要 `runs/interface_thermal/summary.txt`，
用例 `tests/test_interface_thermal_{conduction,stefan,friction,coupling}.py`（59 项）。
**本文件中所有数字由 `scripts/interface_thermal_report.py` 从 JSON 注入，无手工抄写。**

## 1 案例介绍（物理背景 / 几何 / 参数表）

冰区航行中，船体与浮冰的滑动接触把切向功变成界面热。当界面温度逼近熔点时冰开始融化，
熔融水膜又反过来降低摩擦系数与法向刚度——这是从"纯力学接触"到"热-力耦合接触"的关键跨越。
本模块把该闭环拆成五个可独立验证的物理环节，再合成一个双向耦合求解器：

1. **导热**：颗粒/冰体内的傅里叶导热（FVM 瞬态导体），颗粒间接触热阻（Cooper-Mikic-Yovanovich）；
2. **摩擦热**：`q'' = μ p v`，按 Blok 分配系数分给冰/钢两侧；
3. **Stefan 条件**：界面达到熔点时相变潜热吸收、熔层厚度演化；
4. **接触修正**：熔膜作润滑膜，改变法向刚度与摩擦系数；
5. **双向耦合**：DEM 接触力与滑移速度 → 热流 → FVM 温度场 → 界面温度 → μ(T)、k(h) → 接触力。

物性（SI，默认值见 `IceSteelContact` / `EnthalpyStefan1D`）：

| 参数 | 符号 | 取值 | 说明 |
|---|---|---|---|
| 冰导热 / 密度 / 比热 | k_i, ρ_i, c_i | 2.2 W/(m·K), 917 kg/m³, 2100 J/(kg·K) | 淡水冰 |
| 冰潜热 / 熔点 | L, T_m | 3.34e5 J/kg, 0 °C | |
| 冰热扩散率 | α_i | 1.1424e-06 m²/s | k/(ρc) |
| 钢导热 / 密度 / 比热 | k_s, ρ_s, c_s | 45 W/(m·K), 7850 kg/m³, 470 J/(kg·K) | 船用低碳钢 |
| 熔点附近润滑起始 | T_onset | −3 °C | μ 开始下降 |
| 干摩擦系数 | μ_base | 0.08 | 冰-钢滑动 |
| 湿摩擦残值 | μ_wet | 0 / 0.03 | 见 §5 局限 |
| 接触斑面积 / 法向力 | A, F_n | 1.0e-4 m², 500 N | 等效 5 MPa 平均压力 |
| 初始温度 | T_0 | −10 °C | |

Stefan 验证用水的物性：k=0.56 W/(m·K)、ρ=1000 kg/m³、c_l=4200、c_s=2100、
L=3.34e5 J/kg，过热 ΔT=5 K ⇒ Ste = 0.06287，
Neumann 常数 λ = 0.175491（解析渐近 √(Ste/2) =
0.177305）。

Blok 分配（冰/钢）：冰侧 effusivity 2058，
钢侧 12885 ⇒ 冰侧份额 γ_i = 0.1377、
钢侧 γ_s = 0.8623（§3.3 给出守恒与对称性的负控）。

## 2 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**工具链**：纯 numpy/scipy（`scipy.linalg.solve_banded`、`scipy.optimize.brentq`、
`scipy.integrate.quad`、`scipy.sparse.linalg.spsolve`、`matplotlib`），不引入新依赖；
pytest 运行、`benchmarks/` 脚本一键重算。

**流程**（每一环都配一个不共用代码路径的参考解）：

1. `Conduction1D`：节点中心有限体积，θ-格式隐式（默认 θ=1 后向 Euler），
   表面可为热流/温度边界，远端绝热或 Dirichlet，可几何加密。
   参考解：半无限体阶跃 `T = T_i + (T_s−T_i) erfc(x/2√(αt))`、
   常热流 `T_s = T_i + 2q√(αt/π)/k`。
2. `ParticleThermalNetwork`：DEM 颗粒附加温度自由度（集总热容 ρcV），
   接触导热 = 两侧 Maxwell 收缩热阻 + 粗糙峰膜热阻（CMY），隐式 Euler；
   参考解：解析指数松弛 + 能量精确守恒。
3. `EnthalpyStefan1D`：固定网格焓法，潜热摊在一侧熔化带 `[T_m, T_m+2ΔT_melt]`，
   **对焓残差做 Newton 迭代**（未知量取 H，`dT/dH = 1/c_eff` 在带内很小，
   从而不会一步跨过熔化带）；熔层厚度取液相比 `X = ∫f_l dx`。
   参考解：Neumann 相似解 `X = 2λ√(αt)`，`λe^{λ²}erf λ = Ste/√π`。
4. `friction_heat`：`q''=μpv`、Blok 分配、Jaeger 移动带状热源
   `T(ξ) = (qa/πk)∫ e^{-V(ξ−u)}K0(V|ξ−u|)du`（`V = va/2α`），
   用对数奇异性替换 `s=e^t` 做 Gauss-Legendre 求积。
   两个**无自由参数**的解析极限作为独立判据：高 Peclet 渡越时间极限
   `T_max → (2q/k)√(α·2a/v/π)`、低 Peclet 对数发散
   `T(0) → (qa/πk)[−2ln(V/2)−2γ+2]`；另用 2D 有限体积直接解 Rosenthal 方程
   （上风对流 + 隐式稀疏直解）作第三路交叉验证。
5. `lubrication`：`s(T) = clip((T−T_onset)/(T_m−T_onset),0,1)`，
   `μ(T) = μ_wet + (μ_base−μ_wet)(1−s)^p` ⇒ μ(T_m)=μ_wet（默认 0，判据 e），
   熔膜刚度 `k_film = E_w A/h`，与接触刚度串联 `k_eff`；
   Stefan 条件熔层增长率 `dh/dt = q_net/(ρL)`。
6. `coupling`：每个时间步把 FVM 的**表面热流响应取仿射化**
   （两次试探步：`T_s(q) = T_a + Z q`），界面温度由
   `r(T_s) = T_a + Z q(T_s) − T_s = 0` 的 Newton 迭代求解（导数用解析
   `dμ/dT`），达到熔点时界面温度被约束钉住、富余热流交给 Stefan 条件。
   这是真正的双向耦合（不是交错显式），残差/迭代数逐步记录。

**关键设置**：耦合算例 dt=0.005 s、4 s、冰侧 1D 域 20 mm / 201 节点（壁面加密 1.05）；
Stefan 验证 10 mm / 1201 节点、dt=0.01 s、ΔT_melt=0.01 K（并做 0.02/0.01/0.005 K 加密）；
Jaeger 求积 81~121 采样点，2D FVM 1001×100 网格。

## 3 计算结果（全部程序化注入）

### 3.1 (a) 纯导热：阶跃表面温度与常热流闪温

解析解与数值解之差即网格/时间离散误差本身（2 K 量程为温度尺度）：

| t [s] | 最大绝对偏差 [K] | 相对 2 K 量程 |
|---|---|---|
| 2 | 1.46e-03 | 0.073 % |
| 10 | 2.93e-04 | 0.015 % |
| 30 | 9.75e-05 | 0.005 % |
| 60 | 4.88e-05 | 0.002 % |

常热流 q=2e4 W/m²（界面摩擦热注入通道）：

| t [s] | FVM 表面温度 [°C] | 解析解 [°C] | 相对误差 |
|---|---|---|---|
| 1 | 8.9370 | 8.9643 | 0.249 % |
| 10 | 32.6634 | 32.6720 | 0.025 % |
| 60 | 82.9253 | 82.9288 | 0.004 % |

### 3.2 (b) Stefan 相变界面：对 Neumann 相似解

| t [s] | 熔层厚度 X [mm] | Neumann 2λ√(αt) [mm] | 相对误差 |
|---|---|---|---|
| 1.0 | 0.1282 | 0.1282 | 0.064 % |
| 2.0 | 0.1812 | 0.1812 | 0.007 % |
| 5.0 | 0.2866 | 0.2866 | 0.015 % |
| 10.0 | 0.4053 | 0.4053 | 0.009 % |

熔化带正则化加密（λ 与网格固定，只改 ΔT_melt）：

| ΔT_melt [K] | 最差前沿相对误差 |
|---|---|
| 0.020 | 0.049 % |
| 0.010 | 0.064 % |
| 0.005 | 0.072 % |

界面能量账（`∫q dt = ρLX + 显热`，取 j≥1 的内区以保证与强 Dirichlet 边界一致）：

| t [s] | ∫q dt [J/m²] | 潜热 [J/m²] | 显热 [J/m²] | 闭合误差 |
|---|---|---|---|---|
| 1.0 | 42668 | 42833 | 1314 | -7.57e-13 |
| 2.0 | 60956 | 60532 | 1904 | -7.28e-13 |
| 3.0 | 74983 | 74147 | 2316 | -1.55e-12 |
| 4.0 | 86811 | 85607 | 2683 | -1.89e-12 |
| 5.0 | 97227 | 95731 | 2976 | -2.41e-12 |
| 6.0 | 106650 | 104855 | 3274 | -2.95e-12 |
| 7.0 | 115313 | 113255 | 3537 | -3.45e-12 |
| 8.0 | 123376 | 121091 | 3764 | -3.76e-12 |
| 9.0 | 130950 | 128417 | 4013 | -4.09e-12 |
| 10.0 | 138113 | 135375 | 4217 | -4.19e-12 |

### 3.3 (c) 摩擦热：闪温与 Jaeger 移动热源

恒热流闪温（半无限体，Carslaw & Jaeger 精确解）：

| t [s] | 数值温升 [K] | 解析温升 [K] | 相对误差 |
|---|---|---|---|
| 0.5 | 7.7142 | 7.7529 | 0.499 % |
| 1.0 | 10.9370 | 10.9643 | 0.249 % |
| 5.0 | 24.5047 | 24.5168 | 0.050 % |
| 20.0 | 49.0276 | 49.0337 | 0.012 % |

Jaeger 移动带状热源（a=5 mm，q=1e6 W/m²）：Bessel 积分 vs 独立 2D FVM，
以及高 Peclet 渡越时间渐近解：

| v [m/s] | V=va/2α | 积分峰值 [K] | 2D FVM 峰值 [K] | 积分 vs FVM | 高 Pe 渐近 [K] | 积分 vs 渐近 |
|---|---|---|---|---|---|---|
| 0.005 | 10.9 | 758.38 | 753.63 | 0.626 % | 775.29 | 2.181 % |
| 0.010 | 21.9 | 541.13 | 537.62 | 0.648 % | 548.21 | 1.293 % |
| 0.020 | 43.8 | 384.89 | 382.17 | 0.705 % | 387.65 | 0.712 % |

低 Peclet 端（对数发散极限）：

| v [m/s] | V | 积分中心值 [K] | 对数极限 [K] | 相对误差 |
|---|---|---|---|---|
| 1.0e-03 | 2.19 | 1289.52 | 481.53 | 167.796 % |
| 5.0e-04 | 1.09 | 1872.98 | 1484.42 | 26.176 % |
| 2.0e-04 | 0.44 | 2925.49 | 2810.16 | 4.104 % |
| 1.0e-04 | 0.22 | 3853.25 | 3813.05 | 1.054 % |

说明：对数渐近式只在 V≪1 成立，表中 V≥1 的两行是**渐近式失效**的现场（不是求积误差），
判据只取 V≤0.44 两行（4.1 % / 1.1 %）；高 Pe 端同理只在 V≳10 之后收敛到 <3 %。

Blok 分配自洽性：γ_i+γ_s = 1.000000000000（恒等），
冰/钢 effusivity 比 6.26。

接触导热（CMY，粗糙度 σ=50 μm、斜率 0.2、硬度 10 MPa）：

| 压力 [MPa] | h_c [W/(m²K)] | R'' [m²K/W] |
|---|---|---|
| 0.5 | 1218 | 8.209e-04 |
| 1.0 | 2353 | 4.249e-04 |
| 2.0 | 4546 | 2.200e-04 |
| 5.0 | 10857 | 9.211e-05 |

### 3.4 (d)(e) 双向耦合：两条互补工况

| 工况 | v [m/s] | p [MPa] | μ_cold | μ_wet | F_t(0) [N] | T_s(末) [°C] | μ(末) | F_t(末) [N] | 熔膜 [μm] | 迭代(max) | 残差 | 能量闭合 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| self_limiting (mu_wet = 0) | 0.50 | 5.00 | 0.080 | 0.000 | 40.00 | -1.324 | 0.0156 | 7.80 | 0.00 | 4 | 9.95e-07 | -6.4e-12 |
| boundary_lubrication (mu_wet = 0.03) | 1.00 | 5.00 | 0.080 | 0.030 | 40.00 | 0.000 | 0.0300 | 15.00 | 134.51 | 3 | 1.04e-07 | -2.2e-09 |

物理读法（诚实记录，不粉饰）：

* **自限工况（μ_wet=0）**：界面升温到约
  -1.32 °C 后停在熔点之下，
  μ 单调降到 0.0156，摩擦力从
  40.00 N 降到
  7.80 N（−80.5 %），
  **但没有形成可持续熔层**。这不是数值失败，而是判据 (e) 的直接后果：
  μ(T_m)=0 ⇒ 界面到熔点后热源自关断，接触自行"热刹车"。
* **边界润滑工况（μ_wet=0.03）**：界面被钉在 0 °C，熔层按 Stefan 条件长到
  134.5 μm，法向刚度因串联水膜从
  2.381e+07 降到
  2.346e+07 N/m（−1.43 %），
  摩擦力 40.00 → 15.00 N，
  **闭环成立**：摩擦生热→升温→熔融→μ/k 下降→接触力下降→摩擦热下降。

耦合迭代（判据 d）：两工况最坏 4 步、残差
9.95e-07；冰侧能量闭合最坏 2.22e-09。

### 3.5 (e)(f) 摩擦定律与负控

μ(T) 在 -3~0 °C 区间单调下降、在熔点严格为
0.0；`monotone=True`。

负控 1（无滑移）：界面温度恒为
-10.0 °C，最大摩擦热流
0.0 W/m²，熔膜
0.0 m ⇒ 无摩擦即无温升。
负控 2（无相变）：0.02 s 内界面仍在 −3 °C 以上（未进入润滑带），
μ = 0.080、F_t =
40.00 N =
40.00 N ⇒ 精确退化为纯 Coulomb 力学。

## 4 结果图

| 图 | 内容 |
|---|---|
| `fig1_conduction.png` | (a) 阶跃加热温度剖面 FVM vs erfc；(a2) 常热流闪温 vs 解析 |
| `fig2_stefan.png` | (b) 熔层前沿 vs Neumann；(b2) 熔化带正则化加密收敛 |
| `fig3_jaeger.png` | (c) 移动热源表面温度：Bessel 积分 vs 2D FVM；(c2) 峰值 vs 滑动速度与高 Pe 渐近 |
| `fig4_coupled_history.png` | (d)(e) 界面温度、μ、摩擦力、熔膜厚度时程（两工况） |
| `fig5_convergence.png` | (d) 每步迭代数与残差，对照 5 步 / 1 % 闸门 |
| `fig6_controls.png` | (e) μ(T) 曲线；(f) 两个负控 |

## 5 注意点（踩过的坑 / 局限 / 适用范围）

**踩过的坑（都在代码注释里留了现场）**

1. **有限体积算子的符号**：初版 `_flux_operator` 返回"净流出"而 θ-格式按"净流入"使用，
   θ=1（后向 Euler）时显式项权重为 0 所以看不出来，θ=0.5（Crank-Nicolson）时变成反扩散
   （实测表面温度 174 K vs 解析 9 K，且网格越细越坏）。修正后 θ=0.5/1.0 都对到 <0.3 %。
2. **焓法的 Picard 会静默丢潜热**：表观热容 Picard 迭代在单元一步跨过熔化带时采不到
   巨大 c_eff（实测：0.21 mm 冰被融化而表面只进了 16.4 kJ/m²，超出可得能量的 4 倍）。
   改为"焓残差 Newton"（未知量 H）后能量恒等成立。
3. **温度型 Newton 跨带过冲**：以 T 为未知量时带内 c_eff≈1.7e7，Newton 一步从 0.05 K
   冲到 −79 K；以 H 为未知量（dT/dH 很小）才稳定。
4. **熔化带必须放在熔点之上**：对称带会让"熔点处的固体"有 f_l=0.5，首步就报告半域已融。
5. **Jaeger 积分的两支**：`e^{+Vs}K0(Vs)` 在 Vs>50 溢出，需用渐近级数；对数奇异点用
   `s=e^t` 替换后求积精度到 1e-12。
6. **2D FVM 的三处标定**：能量方程除以 ρc（表面热流项 `qα/(k dz_0)`）、上风边界必须给
   Dirichlet（否则对角为负、解到 1e16）、低 Peclet 时有限域会偏离（见局限）。

**局限（明确不支持的部分）**

* The film drainage (squeeze flow) is not modelled: the melt layer only grows by the Stefan condition, so its thickness is an upper bound once melting starts.
* Coupled cases assume the nominal contact area and the mean pressure are fixed (force-controlled); the displacement-controlled branch is reported as normal_force_at_approach and used only for the audit of the stiffness feedback.
* The ice side is a 1D FVM column: lateral conduction, surface convection to water and radiation are not represented.
* mu(T) collapses to zero at the melting point by construction (criterion e). Sustained melting therefore requires a residual boundary-lubrication friction, which is carried as mu_wet and reported as the second coupled case.

**适用范围**：界面温度在 −20~0 °C、接触压力 1~10 MPa、滑移速度 0.1~2 m/s 的冰-钢结构接触；
Jaeger 参考解适用于接触尺寸远大于热渗透深度（高 Peclet）的情形，低 Peclet 下二维稳态本身
不存在（对数发散），此时须用瞬态闪温解。

## 6 误差判定（逐项回答"是否 < 3 %"）

| 判据 | 被测量的物理量 | 实测 | 闸门 | 结论 |
|---|---|---|---|---|
| (a) | 1D transient conduction vs erfc | 0.073 % | < 2 % | PASS |
| (a2) | constant-flux flash vs 2q√(αt/π)/k | 0.249 % | < 5 % | PASS |
| (b) | Stefan front vs Neumann 2λ√(αt) | 0.064 % | < 5 % | PASS |
| (b2) | Stefan energy ledger closure | 4.19e-12 | < 1e-3 | PASS |
| (c) | frictional flash vs Carslaw-Jaeger | 0.499 % | < 5 % | PASS |
| (c2) | Jaeger integral vs independent 2D FVM | 0.705 % | < 2 % | PASS |
| (d) | coupled DEM-FVM iterations | 4 steps, residual 9.9e-07 | <= 5 steps, < 1 % | PASS |
| (e) | μ(T) monotone, μ(Tmelt) = 0 | monotone=True, μ(0 °C)=0.0 | monotone + zero | PASS |
| (f) | negative controls | zero slip: ΔT=-10.0 °C, no phase change: μ=0.080 | no heating / pure Coulomb | PASS |

* (a) 导热剖面 vs 解析解：0.073 %，**< 3 %（是）**。
* (a2) 常热流闪温：0.249 %，**< 3 %（是）**。
* (b) Stefan 前沿 vs Neumann：0.064 %，**< 3 %（是）**。
* (c) 摩擦闪温 vs Carslaw-Jaeger：0.499 %，**< 3 %（是）**。
* (c2) Jaeger 积分 vs 独立 2D FVM：0.705 %，**< 3 %（是）**；
  对高 Pe 无参数渐近解最坏 2.181 %（V<45 时渐近尚未完全收敛，故闸门取 5 %）。
* (d) 耦合迭代不是"误差"而是收敛判据：4 步、残差 9.9e-07，
  **满足 ≤5 步 / <1 %**。冰侧能量闭合 2.2e-09（<3 %，是）。
* (e) μ(T) 单调性与端点：单调=True、μ(0 °C)=0.0，**满足**。
* 无基准项：接触热阻 CMY 的系数 1.25 与指数 0.95 是文献关联式，本单**未**做独立实验对标，
  仅验证其标度律与单调性（`evidence: REAL` 指公式实现与标度正确，不代表实验确认）。

## 7 复现

```
PYTHONPATH=src python benchmarks/interface_thermal_coupling.py     # 重算 JSON + 6 图 + 摘要
PYTHONPATH=src python scripts/interface_thermal_report.py          # 由 JSON 生成本文
python -m pytest tests/test_interface_thermal_conduction.py \
                 tests/test_interface_thermal_stefan.py \
                 tests/test_interface_thermal_friction.py \
                 tests/test_interface_thermal_coupling.py -q       # 59 passed
```

总体判定：a_conduction=True, b_stefan=True, c_friction_heat=True, d_coupling_convergence=True, e_friction_monotonicity=True, f_negative_controls=True。
