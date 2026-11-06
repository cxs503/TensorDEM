# DEM3D pinned-support pairs: the restitution dashpot was built from the wrong effective mass

Status: **closed** — engine fix in `src/tensordem/dem3d.py`, regression tests in
`tests/test_dem3d_fixed_support_restitution.py`, verification script
`scripts/verify_dem3d_fixed_support_restitution.py`, artifacts in
`docs/artifacts/fixed-support-restitution/`.

Protocol tag: `tensordem-dem3d-fixed-support-restitution-v1`.

## 1 案例介绍（物理背景 / 几何 / 参数）

DEM3D 的法向接触把恢复系数映射成阻尼：

```
zeta = |ln e| / sqrt(pi^2 + ln^2 e),   gamma_n = 2 zeta sqrt(k_n m_eff)
```

`m_eff` 是**相对坐标的折合质量**。对两个都可动的等质量球，两者反向加速，
`m_eff = m/2`；但当一对接触里有一个**被钉住的粒子**（硬支承：每步把速度清零、
位置复位到初始位置，`step()` 里的 `self.fixed` 分支，用于底板、夹具、刚性
压头/支撑颗粒），伙伴质量无穷大，只有自由粒子加速，正确值是 `m`。这正是
`tensordem.restitution.effective_mass(m, None)` 的定义，也是工具/压板分支
（`hertz_reduced_modulus(..., rigid_counterpart=True)`）一贯采用的刚性对应体约定。

修复前 `dem3d` 对所有接触硬编码 `m_eff = cfg.mass / 2.0`（源码注释即
"equal-mass spheres"），于是钉住接触的阻尼被低估 `sqrt(2)` 倍，
`zeta_actual = zeta_target / sqrt(2)`，实测恢复系数系统性**高**于 `e_target`。

| 参数 | 值 |
|---|---|
| 颗粒半径 `R` | 0.025 m |
| 密度 `rho` | 917 kg/m³（冰） |
| 单粒质量 `m` | 6.004e-2 kg（= ρ·4/3πR³） |
| 法向刚度 `k_n` | 5 000 N/m |
| Hertz 模量 / 泊松比 | 1.0e9 Pa / 0.3 |
| 接触模型 | `linear` 与 `hertz` 两支 |
| 恢复系数目标 | `e_target` = 0.5, 0.8 |
| 撞击速度（闭合） | 0.1, 0.25, 0.5, 1.0, 2.0 m/s |
| 时间步 | `dt = 0.6 * recommended_dt` |

几何：2×2×2 点阵只当作容器，`alive[:] = False` 隔离出一对无键接触
（`scripts/verify_dem3d_hertz_restitution.py` 的同一套 harness）。
「钉住」= `sim.fixed[1] = True` 且 `initial_positions[1] = positions[1]`，
自由球以 `speed` 撞向它；对照组两球各以 `speed` 相向（闭合速度 `2*speed`），
`m_eff = m/2`。两组除支承方式外完全同构。

## 2 怎么做的（方法 / 流程 / 工具链）

1. **读码定位**：`docs/dem3d_hertz_restitution_closure.md:122` 把
   "`m_eff = cfg.mass/2` 写死、颗粒—固定支撑情形未验证" 记为已知欠账；
   `dem3d.py` 的接触分支确实只用了 `cfg.mass/2`，且 `fixed` 掩码只在
   `step()` 里清零加速度/速度/位置，接触律看不到它。
2. **修复**（最小改动，`dem3d.py` 接触段）：按对判定
   `pinned = fixed[ci] ^ fixed[cj]`（恰一侧被钉住），把 `m_eff` 由标量
   `m/2` 换成逐对张量 `m`（钉住对）/ `m/2`（自由对）。
   全自由体系**保留原来的标量表达式**（`math.sqrt` 路径），因此既有无键
   轨迹逐位不变；`absolute` 阻尼约定分支与工具/压板分支不受影响。
3. **验证**（三路参考，均不依赖被测实现）：
   - R1 独立 RK4 积分**同一条**钳位律 `m_eff delta'' = -max(k delta + gamma delta', 0)`，
     分别用 `m_eff = m`（正确）与「动力学用 `m`、阻尼由 `m/2` 构造」（修复前引擎）。
   - R2 同 harness 的自由—自由对：修复后钉住对与自由对必须给出同一个实测 e。
   - R3 **修复前引擎本体**：用 `git archive HEAD src` 取出父提交源码，
     以 `TENSORDEM_SRC` 指向该树重跑同一 sweep（复现式负控，而非口头断言）。
4. 工具链：pytest（回归）、CSV/JSON 落盘、matplotlib 出图。

## 3 计算结果（程序化注入，`docs/artifacts/fixed-support-restitution/sweep.csv`）

修复后（`engine = fixed`），`e_pinned` 与 `e_free` 是同一目标下只差支承方式的两个实测值：

| 模型 | e_t | v [m/s] | e_pinned | e_free | pin/free−1 | pin vs RK4(m_eff=m) | pin vs e_t |
|---|---|---|---|---|---|---|---|
| linear | 0.5 | 0.10 | 0.5495 | 0.5512 | −0.31 % | −0.15 % | +9.89 % |
| linear | 0.5 | 0.50 | 0.5517 | 0.5528 | −0.19 % | +0.27 % | +10.35 % |
| linear | 0.5 | 2.00 | 0.5494 | 0.5472 | +0.40 % | −0.16 % | +9.88 % |
| linear | 0.8 | 0.50 | 0.8091 | 0.8097 | −0.08 % | +0.12 % | +1.13 % |
| hertz | 0.5 | 0.10 | 0.5820 | 0.5820 | 0.00 % | −0.05 % | +16.40 % |
| hertz | 0.5 | 0.50 | 0.5820 | 0.5817 | +0.06 % | −0.04 % | +16.41 % |
| hertz | 0.8 | 2.00 | 0.8257 | 0.8259 | −0.01 % | −0.00 % | +3.22 % |

全 20 例（2 模型 × 2 目标 × 5 速度）极值：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| worst \|pinned − free-free\| / free-free | **17.9 %** | **0.541 %** |
| worst \|engine − RK4(m_eff=m)\| | — | **0.319 %** |
| pinned 的 e 相对 `e_target` 偏差（linear, e_t=0.5） | +29.1…29.5 % | +9.9…10.4 % |
| pinned 的 e 相对 `e_target` 偏差（hertz, e_t=0.5） | +34.8 %（至 +35 %） | +16.37…16.41 % |

修复前引擎（R3，父提交源码复跑）与 R1 失配参考一致：`linear e_t=0.5, v=0.5`
实测 **0.6473**，RK4 失配参考 **0.6459**（差 0.2 %），即这 29 % 的偏差
**完全由 `m_eff` 约定错误解释**，不是数值噪声。

图：

| 图 | 内容 |
|---|---|
| `docs/artifacts/fixed-support-restitution/restitution_vs_speed.png` | 实测 e 随闭合速度（对数轴）：钉住 vs 自由 vs 修复前（红叉）|
| `docs/artifacts/fixed-support-restitution/mass_convention_attribution.png` | R1 归因：引擎钉住点 vs RK4(`m`)=黑线 vs RK4(失配 `m/2`)=紫虚线 |

## 4 误差判定（逐项，回答「是否 < 3%」）

| 关键量 | 基准 | 相对误差 | 是否 < 3 % |
|---|---|---|---|
| 钉住对 vs 自由对（质量约定一致性） | 同 harness 自由对 | 0.541 %（worst） | **是** |
| 钉住对 vs 独立 RK4（`m_eff = m`） | RK4 | 0.319 %（worst） | **是** |
| 失配归因：修复前引擎 vs RK4 失配参考 | RK4 | 0.2 %（v=0.5，linear） | **是** |
| 钉住对 e vs `e_target`（linear, e_t=0.8） | `e_target` | 1.13 % | **是** |
| 钉住对 e vs `e_target`（linear, e_t=0.5） | `e_target` | 10.4 % | **否** |
| 钉住对 e vs `e_target`（hertz, e_t=0.5） | `e_target` | 16.4 % | **否** |

**显式判定**：本单要解决的问题（质量约定）**已 < 3 %（0.541 % / 0.319 %）**；
但「engine 实测 e == `e_target`」在 `e_t = 0.5` 上**不 < 3 %**（10.4 % / 16.4 %），
且这不是本单引入的：自由—自由对照在同一速度下同样偏 10.35 % / 16.41 %，
误差随 `dt` 细化不收敛（0.6→0.075 倍步长：0.5517→0.5500），说明它是
**钳位律 + 线性弹簧阻尼 ζ 映射的固有偏差**（力被 `clamp_min(0)` 截断，
阻尼只在压缩段做功，经典 `e = exp(-pi zeta/sqrt(1-zeta^2))` 高估耗散），
与 `docs/dem3d_hertz_restitution_closure.md` 记录的 10.6 % / 16.4 % 一致。
该订正映射属另一件事，本单不声称解决。

## 5 注意点（踩过的坑 / 局限 / 适用范围）

- **坑一（探针错误）**：只设 `sim.fixed[1] = True` 而不同步
  `sim.initial_positions[1]` 时，钉住粒子每步被复位到**点阵原位置**，
  实测速度出现 138 m/s 的伪爆炸。任何「钉住」用法都必须同时固定初始位置。
- **坑二（自反的负控）**：第一版 RK4 负控用「自洽的 `m_eff = m/2`」，
  结果与 `m_eff = m` 几乎相同（0.5503 vs 0.5517）——钳位律的实测 e 只依赖 `zeta`，
  对 `m_eff` 自洽缩放**不变**。真正体现 bug 的是**失配**（动力学 `m`、阻尼 `m/2`）。
  结论：想用「质量标度」做负控，必须固定动力学、只缩放阻尼质量。
- **适用范围**：只覆盖**单侧钉住**的对（硬支承 = 速度清零 + 位置复位）。
  双侧钉住的接触不改变任何动力学（两者加速度都为零），本单只保证不崩、不注入力。
  真实柔性夹具、可动但质量差异大的伙伴颗粒不属此约定。
- **默认数值路径**：`restitution_coefficient = 1.0`（默认）时 `gamma_n = 0`，
  与 `m_eff` 无关；`contact_damping_model = "absolute"` 分支用绝对 `c_n`，
  也不受影响。因此**默认配置与所有 e=1 的既有基准逐位不变**；
  只有同时「e < 1 + 存在钉住接触」的算例轨迹会变——这正是修正对象。
  已回归的相关测试：`test_dem3d_hertz_restitution_verification.py`、
  `test_dem3d_pair_contact_damping.py`、`test_dem3d_contact_impact_convention.py`、
  `test_contact_impact_dashpot_convention.py`、`test_friction_rolling.py`（全绿）。
- **仍未做**：未做全仓测试套件的完整回归（本机 load15 长期 ≥100，容量纪律下
  只跑了直接相关子集 + 新增测试）；未与冰—钢实验 e 数据对照（仓内无实验数据），
  本单只证明「实现忠实于声明的接触律」，不证明物理 e 取值正确。

## 6 复现

```bash
PYTHONPATH=src python scripts/verify_dem3d_fixed_support_restitution.py --render
# 修复前对照（父提交源码）：
git archive HEAD src | tar -x -C /tmp/prefixsrc
PYTHONPATH=src python scripts/verify_dem3d_fixed_support_restitution.py --render \
    --legacy-src /tmp/prefixsrc/src
python -m pytest tests/test_dem3d_fixed_support_restitution.py -q
```