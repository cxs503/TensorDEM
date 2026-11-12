# AUDIT m16736 — `bowden_hughes_thermal` 独立复核（TensorDEM）

> 审计对象：`src/tensordem/bowden_hughes_thermal.py`（2026-11-11 战役波次加入，
> 483 行、8 个公开函数）。该模块对外宣称**导出**了冰-钢摩擦的非单调
> mu(V) 律（低速融水润滑使 mu 下降、高速黏性剪切使其回升，
> 引 Bowden & Hughes 1950 / Evans 1982 / Oksanen 1989）。
> 此前它**没有任何验证文档**，唯一的测试是
> `tests/test_wave_ice_verification.py::TestBowdenHughesFriction`，
> 其断言为 `mu in [0,1]` 与 `len(set(mu)) > 1`。
> 本单按 adversarial-review 技能独立重建执行路径、给闭式解基准、量化偏差，
> 并交付可复现审计台（`benchmarks/bowden_hughes_audit/`）。
> 生成方式：本文所有数字由 `benchmarks/bowden_hughes_audit/build_doc.py`
> 从 `runs/audit.json` 程序化注入（零手抄）。

## ① 案例介绍（物理背景 / 几何 / 参数）

冰-钢滑动接触的摩擦系数随速度变化，是冰区船舶阻力、螺旋桨铣削、冰池
试验摩擦修正的共同物理基础。Bowden-Hughes 思路：摩擦热 q = mu sigma V 在
两体间分配，界面升温到 0 degC 后多余热量用于融冰，形成水膜；水膜承载
面积分数 A 决定摩擦由粗糙峰（mu_asy）还是黏性剪切（mu_film = eta V/(h sigma)）
主导，因而 mu(V) 非单调。

装置即模块默认配置（单点接触斑）：

| parameter | value (shipped default) |
|---|---|
| ice k / rho / c_p | 2.2 W/mK / 917.0 kg/m3 / 2050.0 J/kgK |
| ice diffusivity | 1.1700e-06 m2/s |
| steel k / rho / c_p | 50.0 W/mK / 7800.0 kg/m3 / 470.0 J/kgK |
| water viscosity / density | 0.0018 Pa s / 1000.0 kg/m3 |
| latent heat L_f | 3.3400e+05 J/kg |
| bulk ice temperature | -5.0 degC |
| asperity friction mu_asy | 0.15 |
| contact_time / contact_radius / roughness | 120.0000 s / 0.0020 m / 1.0000e-06 m |
| effusivity e_ice / e_steel | 2033.6347 / 13538.8330 (ratio 6.657) |

审计只测该模块自身；不与实验数据比对（仓内无 Oksanen 1989 / Evans 1982
数字化工件，见 §5 局限 4）。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **重建执行路径**（读码）：`compute_bowden_hughes_equilibrium` 并非其
   docstring 所称的「迭代求解耦合热-力学问题」——没有迭代。它用
   `q_total = mu_asy * sigma * V`（固定取粗糙峰摩擦）算热量；用
   `gamma = e_ice/(e_ice+e_steel)` 做 Blok 分配；两种分支（是否融冰）
   各自把恒等式 `q_ice+q_steel+q_melt = q_total` 直接构造出来；
   膜厚来自质量平衡；水膜承载分数 `A = min(1,h/roughness) * min(1,t_squeeze/t_contact)`；
   最后 `mu_total = (1-A) mu_asy + A mu_film`，且 `mu_film` 被硬夹到 `<= mu_asy`。
2. **零假设对照（tautology control）**：在 4 个 sigma x 3 个体温度 x 13 个
   速度 = 156 个网格点上，分别量
   (i) 模块自带的 `energy_conservation_check` 所检查的那条恒等式的残差，
   (ii) 它**没有**检查的物理闭合 `q = mu_total sigma V` 的残差。
   若 (i) 恒为机器零而 (ii) 显著，则该「能量守恒检查」是对返回值的同义反复。
3. **闭式解基准**（不含任何文献数字，全部可解析推导）：
   * Blok 分配 `q_ice/q_steel = e_ice/e_steel`（两体等界面温度）；
   * 半无限体常热流表面温升 `dT = 2 q sqrt(t/pi)/(e_ice+e_steel)`；
   * Stefan 挤膜半厚时间 `3 eta R^2/(4 sigma h^2)`；
   * 融膜质量平衡 `h = q_melt R/(L_f rho_w V)`。
4. **形状检验**：`mu(V)` 在 V = 1e-4 … 1e2 m/s（241 点）x 4 个 sigma 上取值，
   统计相对 mu_asy 的偏离幅度、最小点位置、以及 regime 计数
   （模块自己的门限：A<0.05 boundary，A<0.5 mixed，否则 hydrodynamic）。
5. **旋钮敏感性**：单独扫描 `contact_time` / `surface_roughness` /
   `contact_radius` / `water_viscosity`，量 mu(V=1 m/s, sigma=1 MPa) 的变化幅度。
6. **时间尺度一致性**：对比模块用于导热的 `contact_time`（默认 120 s）与
   接触斑驻留时间 `2R/V`，并给出由此决定的融冰起始热流阈值。
7. 工具链：`benchmarks/bowden_hughes_audit/run_audit.py`（A–E 五段，纯 numpy，
   本机 10 s 内跑完，LOCAL load15 > 100 时亦符合容量纪律）；
   `tests/test_bowden_hughes_thermal_analytic.py`（28 passed + 3 strict-xfail）。

## ③ 计算结果

### 3.1 零假设对照：「能量守恒检查」是恒真式

| control | measured | verdict |
|---|---|---|
| max \|q - (q_ice+q_steel+q_melt)\| / q  (n = 156 points) | 1.9174e-16 | identity, holds by construction |
| share of points where energy_conservation_check(tol=1e-12) is True | 100.0 % | **tautological** |
| max \|mu_total sigma V - q_total\| / q_total | 0.0170 | closure the check never looks at |
| median closure residual | 4.6306e-06 | small only because mu_total is pinned near mu_asy |

`energy_conservation_check` 在全部 156 个点上、即使把容差收紧到
1e-12，仍然 100 % 返回 True——因为它检查的 `q_ice+q_steel+q_melt == q_total`
是代码在两个分支里**构造**出来的恒等式。真正会失败的是物理闭合
`q_total ?= mu_total sigma V`：模块永远用 `mu_asy` 生成热量，却返回
`mu_total`，最大残差 0.0170（在 V=0.1 m/s、
sigma=0.1 MPa 处，因该处 mu_total 偏离 mu_asy 最远）。**结论：现有测试
所宣称的「能量守恒通过」不构成任何物理证据。**

### 3.2 闭式解基准逐项

| 基准 | 解析式 | 模块实测 | 结论 |
|---|---|---|---|
| Blok 分配比 | `q_ice/q_steel = e_ice/e_steel` | 相对误差 0.0000 | **正确**（唯一严格正确的分支） |
| 半无限体表面温升 | `2 q sqrt(t/pi)/(e_i+e_s)` | 恒高 26.24 % | 误差 > 3 %，超 8.7 倍 |
| Stefan 挤膜时间常数 tau | `3 eta R^2/(4 sigma h^2)` | 恰好 2.000 tau | 常数因子 2（= 半厚时间的 2/3） |
| 融膜质量平衡 | `h = q_melt R/(L rho V)` | 相对误差 1.5465e-16 | **正确** |
| 挤膜标度指数 | `eta^1 R^2 sigma^-1 h^-2` | (1.0, 2.0, -1.0, -2.0) | **正确** |

温升逐点对比（传导分支，未融冰）：

| V [m/s] | sigma [Pa] | q_total [W/m2] | dT module [K] | dT semi-infinite analytic [K] | rel. error |
|---|---|---|---|---|---|
| 1.0000e-04 | 10000.0000 | 0.1500 | 1.503009e-04 | 1.190637e-04 | 26.24 % |
| 1.0000e-04 | 1.0000e+05 | 1.5000 | 0.001503 | 0.001191 | 26.24 % |
| 3.0000e-04 | 10000.0000 | 0.4500 | 4.509028e-04 | 3.571912e-04 | 26.24 % |
| 3.0000e-04 | 1.0000e+05 | 4.5000 | 0.004509 | 0.003572 | 26.24 % |
| 0.0010 | 10000.0000 | 1.5000 | 0.001503 | 0.001191 | 26.24 % |
| 0.0010 | 1.0000e+05 | 15.0000 | 0.015030 | 0.011906 | 26.24 % |
| 0.0030 | 10000.0000 | 4.5000 | 0.004509 | 0.003572 | 26.24 % |
| 0.0030 | 1.0000e+05 | 45.0000 | 0.045090 | 0.035719 | 26.24 % |

误差恒为 26.24 %，与 q、V 无关——因为模块用
`k_eff = mean(k)`、`delta_eff = mean(delta)` 的「平均化」代替了
`2/sqrt(pi)` 因子与 effusivity 求和，二者之比
`(delta_eff/k_eff) / (2 sqrt(t/pi)/(e_i+e_s)) = 1.2624`
（与数值实测的 26.2 % 一致）。

Stefan 挤膜（`h` 取 V=1 m/s、sigma=1 MPa 实测值）。注意两种时间约定差 3 倍：
解 `h(t)=h0/sqrt(1+t/tau)` 得**时间常数** `tau = 3 eta R^2/(4 sigma h0^2)`，
而**半厚时间**是 `3 tau`（令 `t/tau = 3`）。模块返回 `2 tau`，
即半厚时间的 2/3（-33.3 %）：

| sigma [Pa] | h [m] | t_module [s] | tau [s] | 3 tau = half-life [s] | t_module/tau | t_module/half-life |
|---|---|---|---|---|---|---|
| 1.0000e+05 | 8.9264e-07 | 0.134787 | 0.067393 | 0.202180 | 2.000 | 0.667 |
| 1.0000e+06 | 8.9264e-07 | 0.013479 | 0.006739 | 0.020218 | 2.000 | 0.667 |

融膜质量平衡（V 扫描）：

| V [m/s] | q_melt [W/m2] | h module [m] | h from mass balance [m] | rel. error | A_film | mu |
|---|---|---|---|---|---|---|
| 0.0100 | 571.6556 | 3.4231e-07 | 3.4231e-07 | 1.5465e-16 | 2.6146e-04 | 0.149961 |
| 0.1000 | 14071.6556 | 8.4261e-07 | 8.4261e-07 | 0.0000 | 1.0622e-04 | 0.149984 |
| 1.0000 | 1.4907e+05 | 8.9264e-07 | 8.9264e-07 | 0.0000 | 1.0026e-04 | 0.149985 |
| 10.0000 | 1.4991e+06 | 8.9765e-07 | 8.9765e-07 | 1.1795e-16 | 9.9705e-05 | 0.149987 |

注意最后两列：`h` 在 V 变化 4 个数量级时只变 2.622 倍
（因为 `h -> mu sigma R/(L rho)` 与 V 无关），而 `A_film` 恒在 1e-4 量级。

### 3.3 宣称的非单调 mu(V) 不存在

| sigma [Pa] | min mu | V at min [m/s] | dip below mu_asy | max \|mu-mu_asy\|/mu_asy | n in 'boundary' | n in mixed/hydrodynamic |
|---|---|---|---|---|---|---|
| 100000 | 0.147241 | 0.0944 | 1.839 % | 1.839 % | 241 | 0 |
| 500000 | 0.148914 | 0.0126 | 0.724 % | 0.724 % | 241 | 0 |
| 1e+06 | 0.149222 | 0.0063 | 0.519 % | 0.519 % | 241 | 0 |
| 2e+06 | 0.149826 | 0.0032 | 0.116 % | 0.116 % | 241 | 0 |

* 全部 4 x 241 个点里，`regime` **只取 "boundary"**：`mixed` /
  `hydrodynamic` 两个分支是死代码（模块自己的门限是 A_film >= 0.05，
  而全部 4 x 241 点里 A_film 峰值仅 0.029，
  且由挤膜判据 `t_squeeze/t_contact ~ 1e-4` 钉死）。
* mu(V) 相对 mu_asy 的**最大偏离 1.84 %**（sigma=0.1 MPa），1 MPa 处仅 0.52 %；
  「下降段」是这个 1 % 量级数值摆动的 argmin，不是润滑物理。
* 结构上限：`mu_film = min(mu_film, mu_asy)` 使 `mu_total <= mu_asy` 恒成立，
  即模型永不可能表达「高速段摩擦回升并超过干摩擦」这一 docstring 卖点。
  按 `mu_film = eta V/(h sigma)` 反解，达到 mu_asy 需要 V ~ 74 m/s
  （冰区船舶/冰池的物理速度范围是 0.01–10 m/s）。
* 对应地，模块现有的 `test_mu_velocity_nonmonotonic` 断言是
  `decreasing or increasing`——只要最小值不是最大值就通过，
  在 1e-4 量级的数值抖动上即为真，属**近真空断言**。

### 3.4 mu 由旋钮决定，而不是由滑动速度决定

knob 扫描（基准点 V=1 m/s, sigma=1 MPa）：

| knob | range swept | min mu | max mu | mu span / mu_asy | q_melt fraction range |
|---|---|---|---|---|---|
| contact_time | 0.0010 .. 1000.0000 | 0.068425 | 0.150000 | 54.38 % | 0.000 .. 0.998 |
| surface_roughness | 1.0000e-09 .. 1.0000e-04 | 0.149983 | 0.150000 | 0.01 % | 0.994 .. 0.994 |
| contact_radius | 1.0000e-04 .. 0.0100 | 0.149983 | 0.149999 | 0.01 % | 0.994 .. 0.994 |
| water_viscosity | 1.0000e-04 .. 0.0100 | 0.149922 | 0.149999 | 0.05 % | 0.994 .. 0.994 |

* `contact_time` 一个旋钮即可把 mu 从 0.068425 拉到
  0.150000（相对幅度 54.4 %）——
  比整个速度轴（1.84 %）大两个数量级。
* 其余三个旋钮的影响 < 0.06 %：mu 对速度几乎不敏感，对时间尺度极其敏感。

### 3.5 时间尺度不一致：融冰判据提前三个数量级触发

| V [m/s] | residence time 2R/V [s] | conduction time [s] | ratio | q_onset module [W/m2] | q_onset analytic (t_res) [W/m2] | module/analytic |
|---|---|---|---|---|---|---|
| 0.0010 | 4.0000 | 120.0000 | 30.000 | 928.3444 | 34501.8505 | 0.027 |
| 0.0100 | 0.4000 | 120.0000 | 300.000 | 928.3444 | 1.0910e+05 | 0.009 |
| 0.1000 | 0.0400 | 120.0000 | 3000.000 | 928.3444 | 3.4502e+05 | 0.003 |
| 1.0000 | 0.0040 | 120.0000 | 30000.000 | 928.3444 | 1.0910e+06 | 8.509e-04 |
| 10.0000 | 4.0000e-04 | 120.0000 | 3.000e+05 | 928.3444 | 3.4502e+06 | 2.691e-04 |

* 传导用 `contact_time`（120 s），而膜厚/驻留判据隐含的时间是 `2R/V`
  （V=1 m/s 时 4e-3 s），两者差 30000.000 倍。
* 后果：模块的融冰起始热流恒为 `k_ice (-T_bulk)/sqrt(alpha t_contact)` =
  929.1616 W/m2（与 V 无关），在 sigma=1 MPa 下
  只要 V > **0.0062 m/s** 就判为「融冰」；
  而按驻留时间的半无限解析阈值在 V=1 m/s 处是
  1.0910e+06 W/m2，
  模块提前 1175 倍触发。
* 由此得到本单关于内部自洽性的核心结论：**「99.4 % 的摩擦热用于融冰」与
  「水膜只承载 0.01 % 法向载荷」在同一状态里并存**（见 §4 图 2）。
  两个判据用了同一条 `contact_time`，但一个经 `q_melt` 进入膜厚、
  另一个经 `t_squeeze/t_contact` 决定承载分数，故融冰分支在能量上压倒性、
  在力学上却不可见。

## ④ 结果图

* `benchmarks/bowden_hughes_audit/runs/figures/mu_vs_velocity.png`
  —— 4 个 sigma 的 mu(V)（V 轴 6 个数十年，纵轴窗口 0.149–0.1505）：
  曲线是平的，附 mu_asy 参考线；
* `.../film_fraction_vs_melt_fraction.png`
  —— A_film 与「摩擦热中用于融冰的比例」同图（3.5 节的自洽性矛盾）；
* `.../knob_sensitivity.png`
  —— 4 个旋钮对 mu(1 m/s, 1 MPa) 的影响：contact_time 一条线横跨 54 %，
  其余三条近乎水平；
* `.../melting_onset.png`
  —— 融冰起始热流：模块（常数 928 W/m2）vs 半无限解析（随 V 上升）
  vs 摩擦热流 mu_asy sigma V；交点即模块判据提前量。

纯文档/后端任务无场量云图义务：本模块是 0 维热平衡代数模型（无空间离散），
故无云图可画；四图均为 1 维曲线图，已足以承载全部结论。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **坑（本单主发现）**：`energy_conservation_check` 恒真。任何以它为
   验收门的算例都在测一行恒等式。复现只需一行：
   `closure_residual(compute_bowden_hughes_equilibrium(0.1, 1e5, BowdenHughesConfig()))`
   = 1.7e-2，而 `energy_conservation_check(...)` 返回 True。
2. **坑**：模块 docstring 的「非单调 mu(V)」与 `min(mu_film, mu_asy)`
   夹子、以及 `t_squeeze/t_contact` 判据互斥——声称的物理在实现里不可达。
3. **坑**：`contact_time` 同时扮演「传导时间」和「接触持续时间」两个角色，
   而真实接触时间是 `2R/V`；这个混用是 3.5 节全部偏差的根因。
4. **局限**：本单不含实验数据比对。Bowden-Hughes 1950 / Evans 1982 /
   Oksanen 1989 的 mu(V) 数字在仓内没有数字化副本，本单**不手抄文献数字**，
   故 §6 只用可解析推导的基准；文献锚定列为遗留项。
5. **范围**：只审 `compute_bowden_hughes_equilibrium` 及其 5 个私有辅助函数。
   `scripts/generate_wave_ice_figures.py` 对它的调用（波形冰图）未复核，
   但若那些图声称呈现「摩擦系数随速度的非单调性」，其结论同样悬空。
6. **正向结论（不能只报坏消息）**：Blok 分配、融膜质量平衡、挤膜标度指数
   三项与闭式解一致（相对误差 <= 1e-16 / 1e-16 / 0），这部分可以放心用。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 < 3 %）

| 量 | 模块值 | 基准（闭式解） | 误差 | < 3 %？ |
|---|---|---|---|---|
| Blok 热流分配比 q_ice/q_steel | 0.150208 | 0.150208 | 0.0000 | **是** |
| 融膜厚度 h（质量平衡） | 见 §3.2 表 | `q R/(L rho V)` | 1.5465e-16 | **是** |
| 挤膜时间标度指数 | (1,2,-1,-2) | (1,2,-1,-2) | 0 | **是** |
| 挤膜时间常数因子 | `3 eta R^2/(2 sigma h^2)` | tau `= 3 eta R^2/(4 sigma h^2)` | 相对 tau +100.0 %；相对半厚时间 3tau 为 66.7 %（-33.3 %） | 否 |
| 界面温升（传导分支） | 见 §3.2 表 | `2 q sqrt(t/pi)/(e_i+e_s)` | +26.24 % | 否 |
| 融冰起始热流 | 929.1616 W/m2 | 1.0910e+06 W/m2 (V=1 m/s) | 提前 1175 倍（低判） | 否 |
| mu(V) 相对 mu_asy 的偏离幅度 | <= 1.84 % | 无数字化文献锚（见 §5 局限 4）；教科书级定性常识：冰摩擦在 0.01–10 m/s 内由 ~0.1 降到 0.02–0.05 量级 | 模型无法表达该量级的下降 | 否（**定性不可达**，仅定性对照） |
| 物理能量闭合 q = mu sigma V | — | 恒等式 | 最大 0.0170 | 否 |

**显式回答**：本模块的关键物理量**不满足 3 % 标准**。
壳层（Blok 分配、质量平衡、标度律）误差 <= 1e-16，是对的；
但被当作物理结论的三个量——界面温升（+26.2 %）、融冰起始阈值（低判 3 个
数量级）、mu(V) 形状（降幅未实现）——全部落在 3 % 之外，
且 mu(V) 是**定性错误**（宣称的非单调润滑在实现中不存在、regime 恒为
boundary），而非精度不足。按平台「>15 % 不可接受」的红线，该模块当前的
mu(V) 输出**不可用于任何工程结论或论文图表**。

## 处置（disposition）：request changes

1. `tests/test_wave_ice_verification.py::TestBowdenHughesFriction` 的
   `test_mu_velocity_nonmonotonic` 是近真空断言，应改为对
   mu 降幅的量化判据（本单给出的 strict-xfail
   `test_melting_lubrication_depletes_friction` 即模板）。
2. `energy_conservation_check` 应改名或明确其语义（「内部分配恒等式」），
   并把 `closure_residual`（本单已加入模块，纯增量、不改行为）纳入验收。
3. 若要让 mu(V) 具备物理意义，最小改动方向（本单已定位根因，未擅自实施）：
   (a) 统一时间尺度——传导用 `2R/V` 而非 `contact_time`；
   (b) 对 `mu -> q -> h -> A -> mu` 做不动点迭代以闭合能量（现无迭代）；
   (c) 去掉 `min(mu_film, mu_asy)` 夹子，改为 A 阈值的分段模型；
   (d) 用 Oksanen 1989 / Evans 1982 的数字化 mu(V) 做锚定后再进论文图。
   上述 (a)–(d) 会改变数值结果，属重算力 + 需文献数字化的活，故本单列为
   遗留项（本机 load15 > 100，遵守容量纪律只做轻量审计）。
4. 本单交付：审计台（可复现）+ 28 项通过测试（含 3 项 strict-xfail 记录
   上述缺口）+ `closure_residual` 诊断 API + 本文档。

## 可复现命令

```bash
python3 benchmarks/bowden_hughes_audit/run_audit.py            # 生成 runs/audit.json + 4 图（约 10 s）
python3 benchmarks/bowden_hughes_audit/build_doc.py            # 由 audit.json 重建本文档
PYTHONPATH=src python3 -m pytest tests/test_bowden_hughes_thermal_analytic.py -q
# -> 28 passed, 3 xfailed
```
