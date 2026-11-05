# DEM3D 接触-碰撞验证harness的阻尼约定漂移（验证欠账复核，m3031）

**结论先行**：`tensordem.contact_impact`（DEM3D 解析碰撞验证 harness，证据等级 REAL）在
trunk 上与引擎脱耦——它的 5 项 solver 判据全红，实测恢复系数 `e = 1.00032` 而钳位解析
参考值是 `0.71820`（+39.28 %）。根因不是物理，而是 m2626 引入的
`DEM3DConfig.contact_damping_model` 默认值 `"restitution"`：该分支从
`restitution_coefficient` 推导法向阻尼、**忽略 `contact_damping`**；harness 仍按绝对约定
注入 `c_n`，于是它验证的是一个无阻尼对。修复为最小化**显式声明约定**（1 行配置），
本单重做全部解析判据复核并加装回归门。

## 1. 案例介绍

DEM3D 的法向接触是线性弹簧-阻尼器并截断于零力（Hookean，钳位型）：

    F_n = max(0, k_n · δ − c_n · v_n),    δ = 2R − |x_j − x_i|                (1)

验证算例为两等质量自由颗粒对心正碰（键全断、其余格点固定、`drag = 0`），因此
是 1 自由度问题，解析解精确。参数表：

| 参数 | 值 |
|---|---|
| 颗粒半径 R | 0.05 m |
| 密度 ρ | 900 kg/m³ |
| 颗粒质量 m | 0.4712 kg（4/3 π R³ ρ） |
| 有效质量 m_eff | 0.2356 kg（m/2） |
| 法向刚度 k_n | 5.0e4 N/m |
| 法向阻尼 c_n | 由目标恢复系数反解（见 T1） |
| 撞击速度 v0 | 0.5 m/s（每颗粒，相对 1.0 m/s） |
| 时间步 Δt | 引擎 `recommended_dt`（≈1.008e-4 s，68 步/接触） |

**解析参考**（钳位型，本仓库 `contact_impact.impact_analytic`）：力在
`k δ + c δ̇ = 0` 时截断，`θ* = atan2(2ζ√(1−ζ²), 2ζ²−1)`，
`t_force = θ*/ω_d`；几何接触时长 `t_geom = t_force + δ(t*)/(e·v0)`
（力归零后两颗粒仍重叠）。该闭式与独立 RK4 积分一致到 1.2e-11（模块自带 `DerivationTests`）。

## 2. 怎么做的

1. **定位**：`pytest tests/test_dem3d_contact_impact.py` 在 trunk 上 5 red / 12 green，
   失败集合 = 强制律残差、恢复系数闭合、修正映射闭合、时间步收敛、接触时长，全部集中在
   solver 侧，解析侧（`DerivationTests`/`CorrectedMappingTests`）全绿 → 疑点锁定引擎侧。
2. **根因**：读 `src/tensordem/dem3d.py` 的 pair 分支，`gamma_n` 只在
   `contact_damping_model == "absolute"` 时取 `contact_damping`，默认分支由
   `restitution_coefficient = 1.0` 得 `gamma_n = 0`；而 harness `_build_pair` 只传
   `contact_damping=c_n`，未声明约定 → 引擎无阻尼。
3. **修复**：`src/tensordem/contact_impact.py::_build_pair` 显式传
   `contact_damping_model="absolute"`（带注释说明为何必须声明）；不触碰引擎默认值，
   因此其它回归产物不动。
4. **复核**：新增 `scripts/verify_dem3d_contact_impact_convention.py`（程序化产出
   `artifacts/dem3d_contact_impact_convention/{verification.json,cases.csv,convergence.csv,tables.md,fig*.png}`）
   与 `tests/test_dem3d_contact_impact_convention.py`（10 项回归门，含**负控**）。
5. **重生成**：`scripts/benchmark_dem3d_contact_impact.py` 重跑，论文级对照文书
   `docs/dem3d_contact_impact_analytic.md` 逐字复现（git 无 diff），说明修复不是改判据。

## 3. 计算结果

### T1  restitution / contact-duration closure (absolute dashpot)

| $e_{target}$ | $c_n$ [N s/m] | $e$ solver | $e$ clamped | err [%] | $t^{force}$ solver [ms] | $t^{force}$ clamped [ms] | err [%] | $t^{geom}$ err [%] |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 7.4383 | 0.90368 | 0.90203 | +0.183 | 6.6584 | 6.6780 | -0.293 | -0.677 |
| 0.70 | 26.5304 | 0.72210 | 0.71820 | +0.543 | 6.3017 | 6.3696 | -1.067 | -1.197 |
| 0.50 | 55.4153 | 0.55432 | 0.55028 | +0.733 | 5.9445 | 6.0183 | -1.226 | -1.473 |
| 0.30 | 107.9409 | 0.39938 | 0.39714 | +0.564 | 5.5202 | 5.6018 | -1.458 | -0.813 |

### T2  convention negative control (identical state, two conventions)

| $\delta$ [m] | clipped law $F_n$ [N] | engine 'absolute' [N] | err [%] | engine 'restitution' [N] | err [%] |
|---|---|---|---|---|---|
| 1.0e-04 | 60.4153 | 60.4153 | +2.35e-13 | 5.0000 | -91.72 |
| 5.0e-04 | 80.4153 | 80.4153 | +3.53e-14 | 25.0000 | -68.91 |
| 2.0e-03 | 155.4153 | 155.4153 | +5.49e-14 | 100.0000 | -35.66 |
| 5.0e-03 | 305.4153 | 305.4153 | +7.44e-14 | 250.0000 | -18.14 |

### T3  time-step refinement at $e_{target}=0.5$

| $\Delta t$ [s] | steps/contact | $e$ err [%] | $t^{force}$ err [%] | $t^{geom}$ err [%] |
|---|---|---|---|---|
| 1.008e-04 | 68 | +0.733 | -1.226 | -1.473 |
| 5.038e-05 | 137 | +0.334 | -0.389 | -0.748 |
| 2.519e-05 | 275 | +0.146 | -0.389 | -0.386 |

Observed log-log orders over the refinement (dt to dt/4): restitution 1.16, force duration 0.83, geometric duration 0.97 (first order = semi-implicit Euler; the duration residual is discretisation, not a force-law defect).

### T4  unclamped-mapping bias reproduced in the solver

| $e_{target}$ | 0.90 | 0.70 | 0.50 | 0.30 |
|---|---|---|---|---|
| bias [%] | 0.41 | 3.16 | 10.86 | 33.13 |

判据带宽 3 %（仓库工程验收口径）；T1 所有量均在内。T2 为**负控**：同一引擎状态下把约定
换成默认 `"restitution"`，力只剩 `k·δ`（−18 % 到 −92 %），证明该 flag 是**承重**的，
不是装饰。T3 说明 delta 残差是**离散化**（一阶，semi-implicit Euler）而非物理缺陷。

## 4. 结果图

* `artifacts/dem3d_contact_impact_convention/fig1_restitution.png` —— 实测恢复系数 vs
  目标值，同图叠钳位（参考）与未钳位（旧映射）闭式：solver 贴钳位线，旧映射在
  e≤0.7 处偏出 3 % 以上。
* `artifacts/dem3d_contact_impact_convention/fig2_duration_convergence.png` —— log-log
  误差 vs Δt（dt→dt/4）：恢复系数与两个接触时长残差都按一阶下降（斜率
  1.16 /
  0.83 /
  0.97）。
* `artifacts/dem3d_contact_impact_convention/fig3_force_law.png` —— 同一状态的力-重叠
  关系：'absolute' 精确落在式 (1) 上（残差 ≤ 7e-14 相对），默认 'restitution' 退化为
  纯弹簧。

## 5. 注意点 / 局限

* **踩坑**：引擎加了"约定开关"后，所有**直接注入 `contact_damping` 的 harness** 都会静默
  变成无阻尼对——测试若只比"自己算的解析解"，会同时变红；若只比"引擎前后一致"，则完全不报。
  本次红测试正是这套 harness 的判据（3 % 带），说明**带宽判据比自洽判据更能抓脱耦**。
* **负控写法**：`DEM3DConfig` 是 frozen dataclass，构造变体必须 `dataclasses.replace` 后
  重建引擎并重跑 `forces()` 断键（否则键力污染探针）。脚本与测试两处都按此实现。
* **局限**：式 (1) 是线性弹簧-阻尼模型，本验证只证明**实现忠实于所述定律**，不证明冰的
  `k_n, c_n` 标定到实测冰力（那是 material-calibration 线的欠账）；Hertz 非线性分支另有
  `docs/dem3d_hertz_contact_verification.md`。接触时长的 1.2 % 残差为一阶时间离散，需要
  <1 % 时应减半 Δt 或改用事件驱动积分。
* **未做**：`tests/test_fracture_gen_mass_footprint.py` 仍在 collection 处红（需要 owner 裁决
  的质量映射契约，见 `docs/dem_analytic_verification_debt_restoration.md` §4），本单未动。

## 6. 误差判定（vs 解析基准，逐项）

| 关键量 | 基准 | 最大误差 | < 3 % ? |
|---|---|---|---|
| 强制律 `F_n`（T2，含截断） | 式 (1) 闭式 | ≤ 7.4e-14 相对 | 是（机器精度） |
| 恢复系数 `e`（T1，e_target 0.9–0.3） | 钳位闭式 | 0.733 % | 是 |
| 力作用时长 `t_force` | 钳位闭式 `θ*/ω_d` | 1.458 % | 是 |
| 几何接触时长 `t_geom` | 钳位闭式 | 1.473 % | 是 |
| 动量守恒 | 内力等值反向 | 0.0e+00 | 是（精确 0） |
| 旧（未钳位）映射偏差 | 钳位闭式 | 33.13 %（e=0.3） | 否（已知欠账，属旧映射而非本修复） |

**显式回答**：本修复涉及的三项物理量（恢复系数、力作用时长、几何接触时长）相对独立解析
基准的最大误差分别为 0.73 %、
1.46 %、
1.47 %，**均 < 3 %**；强制律残差为机器精度。
旧映射（`contact_tangential.restitution_to_damping`）对钳位型引擎的偏差在 e≤0.7 时
>3 %，这是已并入仓库并保留在测试里的**已知欠账**（`test_known_bias_of_existing_mapping`），
不在本修复范围内，仓库提供 `restitution_to_damping_clamped` 作为修正版。

## 7. 复现

```
python -m pytest tests/test_dem3d_contact_impact.py tests/test_dem3d_contact_impact_convention.py -q
PYTHONPATH=src python scripts/verify_dem3d_contact_impact_convention.py
PYTHONPATH=src python scripts/benchmark_dem3d_contact_impact.py   # 重生成 dem3d_contact_impact_analytic.md
```
