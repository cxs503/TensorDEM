# TensorDEM 浮冰融化核（threefield.thermal）验证欠账清偿与闭式解消除相消误差

状态：验证欠账清偿 + 一处数值修复（2026-11-04，selfstart m3651）。本单只做**复核与判定**，
不引入新物理模型、不动求解器时间步进、不跑 DEM 算例（本机 load15≈126/128 核，按容量纪律只做轻量活）。

代码与数据：
`scripts/audit_threefield_thermal.py`、`src/tensordem/threefield/thermal.py`、
`tests/test_threefield_thermal_verification.py`、
`docs/threefield-thermal/{thermal_verification.json,cancellation_before_after.csv,closed_form_error.csv,
ode_consistency.csv,rk2_convergence.csv,operational_dt_error.csv,inversion_residual.csv,
vanish_timing.csv,figures/}`。

## 1 案例介绍（物理背景 / 几何 / 参数表）

`src/tensordem/threefield/thermal.py` 是「冰‑桨‑船三场耦合」的**热力学末端**：尾流里的浮冰碎片
被水对流加热融化，质量损失进入耦合账本，直接决定续航与阻力回馈。它的核心是一条可分离的
融化 ODE：

```
Nu = 2 + 0.6 Re^(1/2) Pr^(1/3),   h_c = Nu k_w / D
dD/dt = -A/D - B/√D
A = 4 k_w ΔT /(ρ_i L_f*),  B = 1.2 k_w (u/ν)^(1/2) Pr^(1/3) ΔT /(ρ_i L_f*)
L_f* = L_f + c_p (T_m - T_i)   （include_sensible=True 时）
```

**本单之前，这条 ODE 的两侧从未互相对过账**：闭式解 `analytic_ransom_marshall_time` 只被当作
「参考真值」使用，而它自己从未与任何独立参考比过；逆解 `analytic_diameter_ransom_marshall`、
`B→0` 退化支、`include_sensible=True` 分支**完全没有测试覆盖**；积分器 `integrate_melting`
（RK2 中点）只有一个 5 % 量级的单点对照。这正是本席「验证欠账」类活的口径：公式写得对 ≠ 按
求解器方式用它是对的。

几何与物性（`ThermalProperties` 默认值，物理量纲 SI）：

| 参数 | 符号 | 取值 | 说明 |
|---|---|---|---|
| 水温 / 融点 | T_w / T_m | 0.5 / 0.0 °C | 驱动温差 ΔT = 0.5 K |
| 冰初温 | T_i | −2.0 °C | 显热项 c_p(T_m−T_i) = 4200 J/kg |
| 水导热系数 | k_w | 0.55 W/(m·K) | |
| 水运动黏度 | ν | 1.79e−6 m²/s | |
| 普朗特数 | Pr | 12.5 | |
| 潜热 | L_f | 3.34e5 J/kg | |
| 冰密度 | ρ_i | 917 kg/m³ | |
| 碎片直径 | D0 → D | 0.30→0.27 / 0.05→0.025 / 0.02→0.002 m | 跨越粗碎到细碎 |
| 相对速度 | u | 0, 1e−16 … 0.5 m/s | 14 个数量级（含近静止极限） |
| 耦合运行步长 | melt_dt_s | 0.25 s | `CouplingConfig` 默认值 |

关键判据：**t(D0→D) 的独立参考 = 对 ODE 直接做积分** ∫_D^{D0} dD/(A/D + B/√D)
（`scipy.integrate.quad`，并用 40 位 `mpmath` 复核），只共用 A、B，绝不共用被检的闭式原函数。

## 2 怎么做的（方法 / 流程 / 关键设置）

五组相互独立的检查，全部由 `scripts/audit_threefield_thermal.py` 一次跑完（纯 Python，
数秒，无 DEM 时间步进）：

1. **闭式解 vs 求积**：26 组 (u, D0, D, sensible) 组合上，把 `analytic_ransom_marshall_time`
   与独立求积比较相对误差；参考值优先取 40 位 `mpmath.quad`，并记录 `scipy.quad` 与它的差，
   避免「参考本身不准」。
2. **相消误差前/后对照**：把**修复前**的教科书原函数在审计脚本里原样重实现
   （`textbook_primitive_time`），与仓内现实现并列对参考值打分 —— 修复收益是数字，不是声明。
3. **ODE 一致性**：对闭式解 t(D) 做中心差分得 dt/dD，与 `diameter_rate` 给出的 1/(dD/dt)
   比对，校验原函数的**符号与标度**是否与求解器实际积分的右端项一致（而不是只和自己一致）。
4. **积分器阶**：`integrate_melting` 在几何步长序列 n = 20…1280 上对闭式解求误差，
   log–log 最小二乘拟合观测阶；覆盖 u=0（纯导热）、u=0.25/0.5（对流）、`include_sensible`
   两分支、以及 u=1e−4（跨分支）。
5. **逆解 / 退化支 / 消失时刻**：逆解回代残差；u=0 时闭式解必须严格退化为导热律
   (D0²−D²)/(2A) 且 D(t) 与 `analytic_diameter_conduction` 一致；小碎片完全融尽时，
   消失时刻误差必须以 dt 为上界、质量账必须闭合。

工具链：`numpy`（拟合）、`scipy.integrate.quad`（参考）、`mpmath`（40 位交叉校核）、
`matplotlib`（图，Agg 后端）。

## 3 计算结果（程序化注入，零手抄）

机生成台账 `docs/threefield-thermal/thermal_verification.json`（schema
`tensordem.threefield-thermal-verification/1`）与 7 张 CSV，下表由脚本注入：

<!-- BEGIN:VERDICT -->
| 关键量 | 基准 | 实测最大误差（相对） | < 3 % ? |
|---|---|---|---|
| 闭式融化时间 t(D0→D)（26 组合，u∈[0,0.5]，sensible 两分支） | 40 位 mpmath 求积 | 8.300e-14 | 是 |
| 同上，修复前（教科书原函数） | 40 位 mpmath 求积 | 1.078e+11 | **否**（已修复） |
| 同上，修复后（近静止带 u≤1e−12 专项） | 40 位 mpmath 求积 | 6.930e-15 | 是 |
| 闭式解 ODE 一致性（中心差分 dt/dD vs 1/(dD/dt)） | diameter_rate | 6.724e-09 | 是 |
| RK2 积分器观测阶（6 算例） | 理论阶 2 | 1.987 – 1.993（R²=1.0000） | 是（偏差 1.3e-02） |
| RK2 直径误差 @ melt_dt_s = 0.25 s | 闭式解 | 1.182e-09 | 是 |
| 逆解回代残差 | 闭式解 | 4.268e-15 | 是 |
| u=0 退化支 vs 导热律 | 导热律 (D0²−D²)/(2A) | 0（逐位相等） | 是 |
| 消失时刻误差 | 闭式解融化时间 | 0.815 dt（上界 1 dt） | 是 |
| 消失时质量账闭合 | 初始质量 | 0（逐位相等） | 是 |
<!-- END:VERDICT -->

修复前/后逐点对照（`cancellation_before_after.csv`，D0=0.02 m → D=0.01 m，参考值 40 位求积）：

<!-- BEGIN:CANCELLATION -->
| u [m/s] | ε = B√D0/A | 修复前相对误差 | 修复后相对误差 |
|---|---|---|---|
| 1e-16 | 7.359e-07 | 1.078e+11 | 0.000e+00 |
| 1e-14 | 7.359e-06 | 6.582e+06 | 1.742e-16 |
| 1e-12 | 7.359e-05 | 4.007e+02 | 0.000e+00 |
| 1e-10 | 7.359e-04 | 1.865e-02 | 0.000e+00 |
| 1e-08 | 7.359e-03 | 6.677e-06 | 0.000e+00 |
| 1e-06 | 7.359e-02 | 8.027e-10 | 0.000e+00 |
| 1e-04 | 7.359e-01 | 1.718e-15 | 1.718e-15 |
| 1e-02 | 7.359e+00 | 1.614e-16 | 1.614e-16 |
| 2e-01 | 3.680e+01 | 1.798e-16 | 1.798e-16 |
| 5e-01 | 5.204e+01 | 1.260e-16 | 1.260e-16 |
<!-- END:CANCELLATION -->

## 4 结果图

- `docs/threefield-thermal/figures/closed_form_error.png`：闭式解误差 vs 相对速度（教科书原函数
  与现实现并列，含 1 % 判据线与 1e−12 舍入地板）。
- `docs/threefield-thermal/figures/rk2_convergence.png`：RK2 中点积分器的 log–log 收敛曲线，
  与斜率 2 参考线对照（6 个算例）。
- `docs/threefield-thermal/figures/diameter_history.png`：D(t) 曲线（闭式解 vs 两种步长的 RK2）
  + 消失时刻误差（以 dt 为单位）随步长的变化。

## 5 注意点（踩过的坑 / 局限 / 适用范围）

1. **教科书原函数在 B→0 时完全失效**。`[2s³/(3B) − As²/B² + 2A²s/B³ − 2A³ln(A+Bs)/B⁴]`
   是 (A/B)⁴ 量级大小数之差。实测（40 位参考）：u = 1e−12 m/s 时相对误差 **4.0e+2**，
   u = 1e−14 时 **6.6e+6**，u = 1e−10 时 1.9e−2。u→0 恰好走 `B ≤ 1e-300` 退化支所以没暴露，
   于是**只有「几乎静止但非严格静止」这一条带子是坏的**——最容易被忽略的那条带子。
2. 修复方式（最小改动）：把同一原函数按 A³/B⁴ 因式分解，令 ε = B√D/A，则
   g(ε) = −2ln(1+ε) + 2ε − ε² + (2/3)ε³ = Σ_{k≥4} 2(−1)^k ε^k/k（前四项恒等相消），
   再展开 (A³/B⁴)ε^k = (B/A)^{k−4} D^{k/2}/A 得到 B/A 的级数，首项正是导热律
   (D0²−D²)/(2A)。ε < 0.5 走级数（58 项，截断 < 1e−18），否则走原函数（此时相消轻微）。
   两支与原函数**代数恒等**，因此既有测试的数值变化在 1e−15 量级。
3. **RK2 观测阶 1.987–1.993 而非 2.000**：残差来自步末时刻被 `int(round(duration/dt))`
   量化，以及直径被 `max(D,0)` 截断；R² = 1.0000，1.03e−2 的偏差不是方法降阶。
4. **消失时刻被 dt 量化**：完全融尽的时刻误差实测最大 0.81 dt（硬上界 1 dt），质量账闭合到
   机器精度（相对误差 0）。这意味着耦合账本里「这一碎片何时归零」的分辨率就是 `melt_dt_s`
   = 0.25 s；对 0.03 m 量级碎片（融化时间 1e3–1e5 s 量级）无关紧要，但对亚毫米级细碎必须减小步长。
5. **闭式解只在常相对速度下成立**。`integrate_melting` 允许 u 随 D,t 变化（尾流场），
   此时闭式解不能作为逐点参考；本单的对流算例一律用常数 u，这是判据的适用范围，不是实现的缺陷。
6. **本单未修的既有红测试（如实记录）**：在跑三场测试面时发现
   `tests/test_threefield_lbm_transport.py::test_streaming_is_mass_conserving` 失败，
   断言 `|Σ periodic_stream − Σ f| < 1e-12` 实测 3.6e-12。该文件与本单改动无交集
   （本单只改 `thermal.py`），且失败可稳定复现，属**既有**的机器精度级阈值问题
   （求和顺序 / float32 累加）。它落在 LBM 输运模块的产权范围，本单不擅自改其阈值
   （改阈值可能掩盖真实的守恒量亏损），留作下一张 selfstart 单的线索。
7. **物理量级提醒**：ΔT = 0.5 K、u = 0.25 m/s 时 0.03 m 碎片的 dD/dt ≈ −5.4e−6 m/s（≈19 mm/h），
   Ranz–Marshall 的强制对流项约占 98 %（导热项仅 ~2 %）。因此导热律不能当默认参考，
   本单所有判据都锚在完整求积上。

## 6 误差判定（关键量 vs 基准，逐项列出）

基准：①40 位 `mpmath` 求积（闭式解的参考）；②闭式解本身（积分器/逆解的参考）；
③导热律 (D0²−D²)/(2A)（仅用于 u=0 退化支与量级核对）。

| 关键量 | 基准 | 实测最大误差 | < 3 % ? |
|---|---|---|---|
| 闭式融化时间 t(D0→D)，u ∈ [0, 0.5]，sensible 两分支 | 40 位求积 | 8.3e−14（相对） | 是（7 个数量级余量） |
| 同上，修复前（教科书原函数） | 40 位求积 | 1.08e+11（@u=1e−16） | **否**（已修复） |
| 修复后 u ≤ 1e−12 近静止带 | 40 位求积 | 6.9e−15 | 是 |
| RK2 积分器观测阶（6 个算例） | 理论 2 | 1.987 – 1.993（R²=1.0000） | 是（偏差 1.3e−2 < 3 %） |
| RK2 直径误差 @ 耦合运行步长 0.25 s | 闭式解 | 1.2e−9（相对） | 是 |
| 逆解 `analytic_diameter_ransom_marshall` 回代残差 | 闭式解 | 4.3e−15（相对） | 是 |
| u=0 退化支 vs 导热律 | 导热律 | 0（逐位相等） | 是 |
| 消失时刻误差 | 闭式解融化时间 | ≤ 0.81 dt | 是（理论上界 1 dt） |
| 消失时质量账闭合 | 初始质量 | 0（相对） | 是 |

结论：三场耦合的融化核证据等级由 **WITHHELD → REAL**（闭式解、逆解、退化支、RK2 阶、
消失账五个面均有独立基准可复现），并修复了一处会在近静止条带产生 O(1e2) 量级错误、
且此前任何测试都照不到的数值缺陷。