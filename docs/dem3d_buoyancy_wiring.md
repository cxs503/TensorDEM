# DEM3D 重力 + 阿基米德浮力接线：把已验证的浮力核接入求解器（验证欠账清偿）

状态：功能补全 + 求解器级验证完成（2026-11-03，selfstart 单 m3097，TensorDEM）。
证据等级：**REAL**（代码对解析解，非实验标定；冰力学物性未被验证）。
工件：`src/tensordem/dem3d.py`、`src/tensordem/buoyancy_wiring_verification.py`、
`scripts/verify_buoyancy_wiring.py`、`tests/test_dem3d_buoyancy_wiring.py`、
`docs/buoyancy-wiring/buoyancy_wiring_verification.json`、
`docs/buoyancy-wiring/fig1_force_balance.png`、`docs/buoyancy-wiring/fig2_archimedes_settling.png`。

## 1. 案例介绍（欠账是什么，为什么值得还）

`docs/dem3d_buoyancy_float.md` §5 原文记录：

> **The kernel is not yet wired into `IceDEM3D.step`.** `DEM3DConfig` has no gravity
> entry, and adding gravity + buoyancy to the solver changes every existing DEM3D
> benchmark trajectory. That coupling is a separate, heavier increment; this
> delivery provides the verified force drops and the analytic anchors they will be
> scored against.

也就是说：球冠浮力核 `tensordem.buoyancy_wet3d` 已有解析验证（误差 1e-11~1e-15），
但 **求解器根本没有重力，也没有浮力**。后果不是「少一个选项」，而是：

* 破冰 DEM 里最重要的边界条件之一——**浮力支撑**——缺席，冰层在自重下只能靠
  边界固定（`fix_bottom`）或平板支撑伪装的「支撑」，自由漂浮的碎冰块、冰脊龙骨、
  浮冰带（`tensordem.ridge_*`、`floe_buoyancy`）都只能靠外部脚本拼；
* 已经交付并验证过的核**无法被求解器路径复核**：核的验证 ≠ 求解器按核的方式用它
  是否正确（同型判例见 `docs/dem3d_tangential_friction_verification.md` §2）。

本单把这条欠账还掉：给 `DEM3DConfig` 加重力/自由面/水密度/湿阻尼四个**默认关闭**
的开关，把浮力核接进 `forces()` 的体力项，然后用解析基准逐项复核求解器（不是核）。

物理参数表（下文所有算例固定这一套，除非另注）：

| 量 | 符号 | 值 |
|---|---|---|
| 颗粒半径 | `R` | 0.025 m |
| 冰密度 | `rho_p` | 917 kg/m³（对照 700 / 850 / 950） |
| 海水密度 | `rho_w` | 1025 kg/m³ |
| 重力加速度 | `g` | 9.81 m/s² |
| 颗粒质量 | `m = rho_p 4πR³/3` | 0.0600175 kg（rho_p = 917） |
| 自由面 | `z = water_level` | 0.0 m（竖直轴 = `+z`） |
| 晶格 | `nx,ny,nz` | 2×2×2（8 颗粒，仅 1 个自由） |
| 时间步 | `dt` | 4.2433e-4 s（`recommended_dt`，仅由弹性界给出） |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

### 2.1 接线（最小改动，默认关闭）

`src/tensordem/dem3d.py`：

1. `DEM3DConfig` 新增四个字段：`gravity=0.0`（沿 −z）、`water_level=None`
   （自由面高程，`None` = 干态）、`water_density=1025.0`、`wet_drag_coefficient=0.0`
   （只作用在浸没分数上）。`__post_init__` 对非物理输入（负 g、`inf`/`nan` 水面、
   零水密度、负湿阻尼）**抛 ValueError**，不静默降级。
2. `IceDEM3D.body_forces()` 返回 `(N,3)` 体力：重力 `-m g e_z`；
   浮力 `+rho_w g V f_sub(h) e_z`（`f_sub` 直接调用已验证的球冠核
   `buoyancy_wet3d.submerged_fraction_sphere_tensor`）；可选湿阻尼 `-C_wet f_sub v`。
   该函数不触碰摩擦/断裂历史，因此可被 `diagnostics()` 反复调用而不改变轨迹
   （监控纯性，同 `contact_invariants` 的判据）。
3. `forces()` 里体力项只在 `gravity > 0` 或 `water_level is not None` 时才相加；
   默认配置走原路径，逐位不变（§6 Case 5 用 `torch.equal` 证明）。
4. `diagnostics()` 追加 `body_force_ledger()` 的 5 个审计字段
   （`gravity_force_z_N`、`buoyancy_force_z_N`、`wet_drag_force_z_N`、
   `net_body_force_z_N`、`submerged_volume_m3`），供对账用。
5. 竖直轴约定：求解器竖直轴是 `+z`；`buoyancy_wet3d` 内部把它叫 `y`。
   因此本单**不直接复用** `compute_buoyancy_force_3d`（它在 +y 上施力，直接拿来
   会在 y 方向产生虚假侧向力），只复用其几何核并作用在 z 上——这一点写进 docstring。

### 2.2 单颗粒隔离夹具（让解析解可判）

浮力/重力的解析锚（阿基米德吃水、水线面刚度）只对**单个自由颗粒**成立，而
`IceDEM3D` 是一个黏结晶格。夹具 `buoyancy_wiring_verification.isolated_particle()`：

* 2×2×2 晶格，把 7 个颗粒整体平移 `100R` 并置为 `fixed`，自由颗粒单独放在 `z0`；
* 第一次 `forces()` 让所有涉及自由颗粒的键因应变超阈值断裂（随后速度清零、时间归零），
  断言 `alive[涉及 0 号颗粒的键]` 全为 False —— 否则夹具报错（防止「以为隔离了其实没隔离」）；
* 压头（indenter）用 `tool_radius=1e-3, tool_gap=5.0, tool_speed=1e-6` 停到晶格上方
  5 m —— 见 §5 的坑；
* 键/接触刚度降到 500 N/m：算例里没有任何承载键或接触，刚度只进入显式稳定界，
  即只决定 `dt`，不改变被验证的物理。

工具链：`PYTHONPATH=src python scripts/verify_buoyancy_wiring.py --steps 3000` →
JSON + matplotlib 图；测试 `pytest tests/test_dem3d_buoyancy_wiring.py`；
文献/解析锚独立来源是 `buoyancy_wet3d`（球冠积分 + 三次方程根 + 水线面恒等式）。

## 3. 计算结果（程序化注入，零手抄；取自 `docs/buoyancy-wiring/buoyancy_wiring_verification.json`）

### 3.1 Case 1 力恒等式（干态 / 半浸 / 全浸，2 个密度 × 7 个高度 = 14 个态）

| `h/R` | `rho_p` (kg/m³) | 求解器 `F_z` (N) | 解析 `-mg+rho_w gV_sub` (N) | 绝对误差 (N) |
|---|---|---|---|---|
| 1.5 | 917 | -0.588771769695489 | -0.588771769695489 | 0 |
| 0.5 | 917 | -0.485941367579575 | -0.485941367579575 | 0 |
| 0.0 | 917 | -0.294385884847744 | -0.294385884847744 | 0 |
| -0.5 | 917 | -0.072809944038679 | -0.072809944038679 | 0 |
| -2.0 | 917 | +0.069342803846361 | +0.069342803846361 | 0 |

14 个态的最大绝对误差 `5.55e-17 N`；独立 61 点力曲线
（`docs/buoyancy-wiring/fig1_force_balance.png`）最大绝对误差 `3.33e-16 N`，
信号量级 0.589 N ⇒ 相对误差 ~5.7e-16（机器精度）。

### 3.2 Case 2 阿基米德吃水（自由颗粒沉降到解析吃水）

| `rho_p` (kg/m³) | `dt` (s) | 步数 | 物理时间 (s) | 实测中心高 `z` (m) | 解析 `h_eq` (m) | 吃水相对误差 | 误差 / `R` |
|---|---|---|---|---|---|---|---|
| 917 | 4.2433e-4 | 3000 | 1.2730 | -0.014928096 | -0.014929050 | **2.39e-5** | 3.82e-5 |
| 850 | 4.0853e-4 | 3000 | 1.2256 | -0.011866706 | -0.011866877 | **4.64e-6** | 6.84e-6 |
| 917 | 2.1216e-4 | 6000 | 1.2730 | -0.014928116 | -0.014929050 | **2.34e-5** | 3.74e-5 |

第三行是同一物理时间下把步长减半的复算：误差变化 2%，说明 2e-5 量级的残差
不是步长伪影而是收敛平台（残余是阻尼趋近的指数尾巴）。

### 3.3 Case 3 静水恢复刚度（中心差分 vs 水线面恒等式 `K = rho_w g π(R²−h²)`）

| `h/R` | `delta` (m) | 实测 `K` (N/m) | 解析 `K` (N/m) | 相对误差 |
|---|---|---|---|---|
| -0.6 | 0.002 | 12.59368 | 12.63580 | **3.33e-3**（差分截断 `O(delta²)`） |

### 3.4 Case 4 纯重力运动学（干态、无阻尼，`z = z0 − g t²/2`）

| `dt` (s) | 步数 | 时间 (s) | 实测 `z` (m) | 解析 `z` (m) | 绝对误差 (m) | 相对误差 |
|---|---|---|---|---|---|---|
| 4.2433e-4 | 400 | 0.16973 | 0.358342 | 0.358695 | 3.53e-4 | 9.85e-4 |
| 2.1216e-4 | 400 | 0.084865 | 0.464585 | 0.464674 | 8.83e-5 | 1.90e-4 |

固定步数减半步长时误差比 `0.250`（显式 Euler 的局部误差 `O(dt²)`、全局误差
`O(dt)`，而固定步数使 `t ∝ dt`，故总量 `O(dt²)`）——一阶格式的一致收敛指纹。

### 3.5 Case 5 关闭时逐位惰性

| 检查 | 结果 |
|---|---|
| 默认配置 `body_forces()` 最大绝对值 | `0.0 N` |
| 默认配置 vs 水面放到晶格下方 10 m（`gravity=0`）的力向量 | `torch.equal(True)`，最大差 `0.0 N` |
| 默认 `dt` / 加字段后 `dt` | `3.3717702052221915e-5 s` / 同值 |

⇒ 新代码对所有既有 DEM3D 算例（pt-177/178/179 等基准）**逐位无影响**。

### 3.6 Case 6 浸没加权湿阻尼

`z = -0.02 m`（`f_sub = 0.9720`）、`v_z = -0.3 m/s`、`C_wet = 1.5 N·s/m`：
实测阻力 `0.437400 N` vs `-C f_sub v_z = 0.437400 N`，误差 `0.0 N`。

### 3.7 图

* `docs/buoyancy-wiring/fig1_force_balance.png`：净竖向体力随 `h/R` 的曲线
  （解析线 + 求解器虚线重合），标出 `rho_p` = 850 / 917 的阿基米德平衡点。
* `docs/buoyancy-wiring/fig2_archimedes_settling.png`：三条沉降轨迹（两个密度 + 半步长复算）
  收敛到各自解析吃水线（点线）。

## 4. 注意点（坑 / 局限 / 适用范围）

1. **坑：压头会「拖走」被隔离的颗粒。** 第一版沉降算例里颗粒以恒定 0.5 m/s
   无限下沉，看起来像浮力没接上。实际是默认 `tool_radius=0.1 m` 的压头球面
   覆盖了整个 2×2×2 晶格，颗粒一开始就在压头内部，被压头以 `tool_speed=0.5 m/s`
   匀速推下去——速度恰好等于 `tool_speed`。教训：在 `IceDEM3D` 上做「自由颗粒」
   解析基准时，必须同时隔离压头（`tool_gap` 抬高、`tool_speed` 压到 ~1e-6），
   否则任何自由体实验都会被匀速压头污染。
2. **局限（模型边界）**：无附加质量、无自由面变形/辐射、无旋转自由度（因此也没有
   浮心—重心力矩）、水面恒为平面 `z = water_level`、湿阻尼是线性占位
   `-C f_sub v` 而非阻力律、`energy()` 不记重力/浮力势能（机械能账本只覆盖弹性+动能，
   开启体力后**不可用**「机械能守恒」做验收，须改用 `body_force_ledger()` 对账）。
3. **求和约定**：`body_force_ledger()` 的求和包含 `fixed` 颗粒（它们受力但不运动），
   这正是边界反力的物理来源；「净体力」字段不是自由体平衡判据。
4. **重启兼容**：`load_state_dict` 用 `{**asdict(DEM3DConfig()), **saved_config}` 归一化
   （`src/tensordem/dem3d.py` 对应行），旧存档缺这四个键时按默认（关闭）恢复；本单未
   单独跑旧存档回归，属未测路径。
5. **适用范围**：本单只证明「求解器按浮力核的方式实现正确」，**不**证明冰-水物性、
   不做实验对标；`rho_p/rho_w` 用 917/1025 只是几何学的密度比输入。
6. **算力**：本机 load15 一度 ≥ 100，故本单只做轻量算例（单颗粒、≤6000 步，
   全套 ~60 s），未起重算力进程。

## 5. 误差判定（关键量 vs 基准，逐项回答「是否 < 3%」）

| 关键量 | 基准 | 实测偏差 | 是否 < 3% |
|---|---|---|---|
| 净竖向体力（14 个浸没态） | `-m g + rho_w g V_sub` 球冠闭式 | 5.55e-17 N（信号 0.589 N，~9e-17 相对） | **是**（机器精度） |
| 力曲线（61 点） | 同上 | 3.33e-16 N | **是** |
| 自由漂浮吃水 `rho_p=917` | `equilibrium_centre_height` 三次方程根 | 2.39e-5 相对（误差 3.82e-5 `R`） | **是**（远优于 3%） |
| 自由漂浮吃水 `rho_p=850` | 同上 | 4.64e-6 相对 | **是** |
| 半步长复算（`rho_p=917`） | 同上 | 2.34e-5 相对（步长敏感度 2%） | **是** |
| 静水恢复刚度 `K` | `rho_w g π(R²−h²)` | 3.33e-3 相对（差分截断，非模型误差） | **是** |
| 重力自由落体 `z(t)` | `z0 − g t²/2` | 9.85e-4 相对（`dt` 减半 → 1.90e-4） | **是** |
| 浸没加权湿阻尼 | `-C f_sub v` | 0.0 N（精确） | **是** |
| 关闭态对既有算例影响 | 默认配置力向量 | `torch.equal` 逐位相同 | **是**（零扰动） |

结论：接线后的求解器在 8 项可控基准上全部落在 3% 以内（多数为机器精度），
且默认关闭时对既有基准逐位无影响。**未验证**：真实冰-水物性、附加质量、
自由面耦合、旋转力矩——这些不是本单模型的一部分。

## 6. 复现命令

```bash
cd <repo>
PYTHONPATH=src python scripts/verify_buoyancy_wiring.py --steps 3000
PYTHONPATH=src python -m pytest tests/test_dem3d_buoyancy_wiring.py -q
# 回归（默认行为不变）：
PYTHONPATH=src python -m pytest tests/test_dem3d.py tests/test_dem3d_contact_invariants.py \
    tests/test_accounting_output.py tests/test_coupling_restart.py tests/test_reproducibility.py -q
```

本单实测：新测试 `14 passed`（22.8 s）；回归 `42 passed, 38 subtests passed`（22.2 s）。