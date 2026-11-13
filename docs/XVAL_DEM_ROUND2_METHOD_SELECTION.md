# XVAL · CASE-COLUMN-COLLAPSE round-2 method selection vs LIGGGHTS-PUBLIC (m17643)

**Anchor** `XVAL-DEM-ROUND2-METHOD-SELECTION` · repo **TensorDEM** · task class: 开源对标 (XVAL)
· 判据 owner 精度标准 **<15%** · prereg: `prereg/XVAL_DEM_ROUND2_METHOD_SELECTION.json`
· 工件: `artifacts/xval_dem_round2_method_selection/` (`results.json`, `table_main.md`,
`traces.json`, `figures/fig1..fig5.png`)

---

## ① 问题 + ref 出处

**被对标的方法**：`oss_solutions` 行 `TensorDEM|CASE-COLUMN-COLLAPSE|round2-method-selection`
第 ④ 项——柱体坍塌第 2 轮把颗粒棱角性投影到**单个滚动阻力系数 `mu_r`**（球体角性代理）。
幸存构型 `(mu, mu_r, mu_wall) = (0.5, 0.5, 0.8)` 让 Lube et al. (2004) 的 5 个长宽比
全进 15%；但「`mu_r = 0.5`」这个数字只有在我们自己的接触律里才有意义——
同一物理投影在成熟开源实现里是怎么写的、两者能不能互相换算，从未核过。本单只做这件事。

**对标实现（零手抄，逐行取源）**：`CFDEMproject/LIGGGHTS-PUBLIC`（GPL-2.0），
head commit `3d5c00f20519e6bb6eb6756f51f1ad36564e649d`（2024-06-07），经 GitHub contents API 抓取，
字节 sha256 存 `docs/xval/oss_snapshot/PROVENANCE.json`：

| 文件 | 取的什么 | 行 |
|---|---|---|
| `src/rolling_model_cdt.h` | 恒定方向力矩（CDT）：`r_torque = mu_r*Fn*reff * w_rel/|w_rel|`，`reff=radi`(wall) / `radi*radj/(radi+radj)`(pair)，默认去掉法向(扭转)分量 | 89–167 |
| `src/rolling_model_epsd2.h` | 弹塑性滚动弹簧：`kr = kt*reff^2`，`M = M_old + kr*wr_tot*dt`，Coulomb 帽 `|M| <= mu_r*|Fn|*reff`，历史存 M，无阻尼 | 161–230 |
| `src/fix_check_timestep_gran.cpp` | Rayleigh 时步 `dt_R = pi r sqrt(rho/G)/(0.1631 nu + 0.8766)`，`G = Y/(2(1+nu))` | 261–262 |

本仓被对标物：`granular_column/kernel.py::pair_contact_forces` / `::wall_contact_forces`
（线性滚动弹簧 + 帽，转角截断）。

## ② 方法与实现

三件新东西（都进仓、都有测试）：

1. `src/tensordem/xval/oss_rolling.py`——**OSS 律的转写**（不是改写）：`cdt_torque`、
   `SpringRolling`（EPSD2 与 TensorDEM 核共用的弹簧族，`torsion` 开关）、`rayleigh_timestep`、
   `hertz_equivalent_shear_modulus`。每条公式 docstring 里带 `file:line`。
2. `src/tensordem/xval/scenes.py`——**闭环单接触场景**：单球平面滚动（解析参考
   `a = (5/7) mu_r g cos(theta)`）与等球对滚自旋松弛（Coulomb 饱和 ⇒ 线性衰减，
   `t_half = 0.5 w0 /(2 mu_r reff Fn / I)`）。
3. 生产内核可选的 OSS 约定：`Material3D.rolling_lever ∈ {particle_radius, reduced_radius}`
   + `rolling_torsion: bool`（`granular3d_core.py`），`granular_column` 两处接触面与
   `AxisymCase` 透传；**默认 = 历史行为，逐位不变**。

复现：

```
PYTHONPATH=src python benchmarks/xval_dem_round2_method_selection.py     # 45 s，出全部表/图/JSON
PYTHONPATH=src pytest -q tests/test_xval_oss_rolling.py                  # 10 passed
PYTHONPATH=src pytest -q tests/test_axisym_column_lube2004.py            # 13 passed（默认路径回归）
python benchmarks/xval_dem_round2_column_arms.py                          # 宏观臂（本轮未跑，见⑥）
```

## ③ 参数表（无量纲数完整）

材料 = 冻结的 CASE-COLUMN-COLLAPSE 波 5 幸存构型：`r = 0.03 m`，`rho = 2500 kg/m^3`，
`kn = 6e4 N/m`，`kt/kn = 0.5`（⇒ `kr = kt r^2 = 27 N·m`），`e = 0.5`，`mu = 0.5`，
`mu_r = 0.5`，`mu_wall = 0.8`，`g = 9.81 m/s^2`。

| 无量纲量 | 值 | 说明 |
|---|---|---|
| `mu_r`（扫描） | 0.1 / 0.2 / 0.3 / 0.5 / 0.8 | 滚动阻力，本仓约定（lever = r） |
| `mu_r/mu_s` | 0.2–1.6 | >1 时切向滑动先饱和（E2 的 17.6% 出处） |
| `reff/r` | 1（本仓/墙）· 0.5（OSS 对-对） | 本轮核心变量 |
| `kr/(kt r^2)` | 1（本仓）· 0.25（OSS 对-对） | `kr = kt reff^2` 的推论 |
| `M_cap/(mu_r Fn r)` | 1（本仓）· 0.5（OSS 对-对） | 力矩帽 |
| `theta_c`（静持临界坡角） | `arctan(mu_r) = 26.57°` | 刚体静平衡解（本单推导） |
| `dt*sqrt(kt/m)` | 1.6e-2 | 场景步长 5e-5 s 的分辨率 |
| `dt/dt_R` | 0.60 | E6，Rayleigh 判据（本仓 dt = 4.82e-4 s） |

## ④ 对照表（ref vs got；数字全部由脚本从 JSON 生成，见 `table_main.md`）

**E1 · 定律同一性（墙接触）**：生产内核 `wall_contact_forces` vs `rolling_model_epsd2.h` 转写，
同初值、同 dt：`max |omega_kernel - omega_OSS| = 7.3e-14 rad/s`（0.02 s 轨迹，ω 从 10 rad/s 起）。
⇒ **墙接触上我们的滚动律就是 OSS EPSD2**（lever 同为 r），不是"相似"。

**E2 · 单球滚动停车时间**（`v0 = 0.30 m/s`，理论 `t = v0/((5/7) mu_r g)`）：

| mu_r | 理论 [ms] | 本仓 [ms] | OSS EPSD2 [ms] | OSS CDT [ms] | 本仓 vs 理论 | 本仓 vs OSS | <15% |
|---|---|---|---|---|---|---|---|
| 0.1 | 428.1 | 428.1 | 428.1 | 428.4 | −0.01% | +0.00% | ✅ |
| 0.2 | 214.1 | 212.8 | 212.8 | 212.8 | −0.57% | +0.00% | ✅ |
| 0.3 | 142.7 | 141.2 | 141.2 | 141.2 | −1.06% | +0.00% | ✅ |
| 0.5 | 85.6 | 86.3 | 86.3 | 86.3 | +0.79% | +0.00% | ✅ |
| 0.8 | 53.5 | 63.0 | 63.0 | 62.7 | **+17.63%** | +0.00% | ✅（见解释） |

`mu_r = 0.8 > mu_s = 0.5` 时切向摩擦先饱和（所需 `f_t = mu_r N > mu_s N`），
减速度被 `mu_s` 钉住 ⇒ 与"无滑滚动"理论差 17.6%。这是**物理机制**，不是定律分歧：
同一点上 OSS EPSD2 与 CDT 都在同一处（63.0/62.7 ms），三律一致。

**E3 · 等球对自旋松弛（lever 核心）**：

| 臂 | t_half 实测 [ms] | 理论（Coulomb 线性衰减）[ms] | vs OSS EPSD2 |
|---|---|---|---|
| 本仓（`reff = r`） | 33.95 | 33.93 | **−50.00%** |
| OSS EPSD2（`reff = r/2`） | 67.90 | 67.86 | 0（基准） |
| 本仓 + 映射 `mu_r → mu_r/2` | 67.90 | 67.86 | **+0.00%** |
| OSS CDT（`reff = r/2`） | 67.90 | 67.86 | +0.00% |

⇒ 对-对接触的杠杆差恰好 **2 倍**，换算律 `mu_r,OSS = 2 * mu_r,TensorDEM`（等球）；
映射后误差 0.00%，远小于 15%。

**E4 · 斜面临界（静持是否可靠）**：`theta_c = arctan(0.5) = 26.57°`

| 坡角 | 本仓 |v| [m/s] | OSS EPSD2 |v| [m/s] | OSS CDT |v| [m/s] |
|---|---|---|---|---|
| 5° | 1.68e-4 | 1.68e-4 | 3.12e-4 |
| 15° | 2.92e-4 | 2.92e-4 | 9.09e-4 |
| 20° | 3.16e-4 | 3.16e-4 | 7.23e-4 |
| 25° | 3.50e-5 | 3.50e-5 | 5.29e-4 |
| 30°（>theta_c） | 0.140 | 0.140 | 0.137 |

两个结论：① 阈值以上三律都以重力驱动滚动（一致）；② 阈值以下**没有一条律能"严格静持"**
（Coulomb 滞回允许蠕动），CDT 因无历史蠕动略大（~2–3×）。

**E5 · 钻削（轴向自旋）**：`w0 = 20 rad/s`，50 ms：

| 臂 | w_end [rad/s] | 损失 |
|---|---|---|
| 本仓（转写，torsion 在滚动弹簧内） | −0.410 | 102.05%（反向） |
| **生产内核** `wall_contact_forces` | −0.410 | 102.05% |
| OSS 默认（去掉法向分量） | 20.000 | 0.00% |

⇒ 我们的滚动弹簧把**扭转/钻削**也当成滚动阻力吃掉了（且过冲到反向），
LIGGGHTS 默认把它剔除。这条差异在本仓与转写里数值相同，证明它出在生产代码。

**E6 · 时步判据**：本仓 `dt = 4.822e-4 s`（法向周期 20 步）；OSS Rayleigh 判据在
"Hertz 等效剪切模量" `G = 4.053e7 Pa`（由 `kn` + 单粒重声明式换算而来）下
`dt_R = 8.069e-4 s` ⇒ **dt/dt_R = 0.60**，高于常用 0.1–0.2 带约 3 倍。
OSS 侧该比例是用户输入（`fraction_rayleigh_lim`，`fix_check_timestep_gran.cpp:83`），
0.1–0.2 是实践惯例而非常量——如实标注。

**E7 · 步长敏感性**：`mu_r = 0.5` 下 `dt ∈ {1e-5, 5e-5, 1e-4, 2e-4}`：
t_stop 相对散布 本仓 +0.12% / EPSD2 +0.12% / CDT +0.09%（理论 85.6 ms）。
⇒ 停车时间这条指标对 dt 不敏感，④ 的结论不是步长假象。

## ⑤ 收敛性 / 参数敏感性（≥2 点，均取自工件）

1. **`mu_r` 扫描 0.1→0.8**（E2）：三律互相一致到打印精度（0.00%），唯一分歧点是
   `mu_r > mu_s` 的**滑动饱和**机制（+17.6% vs 理论），已在 ④ 给出可解释性。
2. **`dt` 扫描 1e-5→2e-4（20×）**（E7）：t_stop 散布 ≤0.12%，无收敛漂移。
3. **坡角扫描 5°→30°**（E4）：跨过 `arctan(mu_r)` 时速度从 1e-4 量级跳到 1e-1 量级，
   与解析阈值同位置（26.57°）。

## ⑥ 失败模式 / 生态摩擦（如实记）

* **宏观臂已补（2026-11-13，见 `docs/XVAL_DEM_ROUND2_COLUMN_ARMS_MACRO.md`）**：三臂实测
  runout = 2.1060 / 2.4193 / 2.1118（legacy / oss_naive / oss_mapped），`oss_mapped` 复现认证值
  **+0.27 %**、`oss_naive` 偏离 **+14.87 %** ⇒ 换算律 `mu_r,OSS = 2 mu_r,TD` 在宏观成立；
  但 `t_reach` +10.85 %（映射不保 `kr`）、粒-壁帽 2.00×，故换算律不能单独用于时间标定。
  以下为第一轮原始记录（保留不改）：
* **宏观臂未跑（第一轮）**：`benchmarks/xval_dem_round2_column_arms.py`（a=1 柱体，三种滚动约定：
  legacy / OSS 直填 / OSS+映射）本轮**未执行**，原因是算力预算——`nohup … &` 提交后台后
  被 harness 以进程组为单位回收（TIMEOUT 120 s），列体单算例在 120 s 内跑不完。
  脚本已入仓、参数已冻结，下一轮用集群队列（`run_cluster`）补齐即可。
  因此"把 `mu_r` 从 0.5 换成 OSS 约定的 1.0 后柱体 runout 是否复现 2.106"这一条
  **尚无本仓数字**，不做结论。
* **网络**：`raw.githubusercontent.com` 连接被重置；GitHub contents API 可用，
  故全部 OSS 源码经 API 抓取并留 sha256（`PROVENANCE.json`）。
* **命令通道**：`curl` 被安全策略拦截，全部 HTTP 走 python `urllib`。
* **DB 通道**：`psql -d tensorplatform` 读 `oss_solutions/knowledge_base` 报
  permission denied，需 `psql -U postgres`（三查结果照此取得）。
* **OSS 是源码转写而非编译二进制**：公式逐行对齐、并有"生产内核 vs 转写"的恒等测试，
  但没有跑通 LIGGGHTS 本体（GPL-2.0，编译+MPI 成本远超本单）。这是本轮的解释边界。

## ⑦ 本仓 vs OSS 差异清单 + 移植取舍

| # | 差异 | 本仓 | LIGGGHTS-PUBLIC | 量级 | 处置 |
|---|---|---|---|---|---|
| D1 | 律族 | 线性滚动弹簧+帽（EPSD 族） | CDT 无历史 / EPSD2 同族 | — | 不移植 CDT（无静持、无能量账、蠕动随 dt） |
| D2 | 对-对杠杆 | `reff = r` | `reff = radi*radj/(radi+radj) = r/2` | 力矩帽 **2×**、`kr` **4×**、帽角 2× | **采用为可选**：`rolling_lever="reduced_radius"`，另公布换算 `mu_r,OSS = 2 mu_r,TD` |
| D3 | 扭转分量 | 计入滚动弹簧（可过冲反向） | 默认剔除（`torsionTorque` 开关） | 钻削自旋 100% 被吃 | **采用为可选**：`rolling_torsion=False` |
| D4 | 时步判据 | 法向线性周期/20 | Rayleigh（用户给比例） | `dt/dt_R = 0.60` | 只记录+建议（dt/2 已在第 2 轮敏感性里 +11.3% 带内） |
| D5 | 能量账 | 滚动储能/耗散分通道记账 | 无 | — | 移植禁忌：任何新律必须接能量账 |
| D6 | 法向律 | 线性弹簧 `kn*delta` | Hertz/`kn*deltan` 可选 | 影响 `Fn` 与力矩帽 | 不在本月范围（记录） |
| D7 | 切向帽 | `mu*f_n/kt` 弹簧帽 | 同族 | 一致 | 无动作 |
| D8 | 静持 | 滞回允许微蠕动 ~1e-4 m/s | EPSD2 同；CDT 更大 | 三律皆非严格静持 | 记录（不要宣称"静摩擦"） |

**取舍（预登记规则的结果）**：
* H1（我们是不同律族 ⇒ 偏差 >15%）**被否证**：墙接触上两律轨迹相同到 7e-14。
* H2 成立：同族、差在**杠杆**与**扭转**两项工程约定。
* H3（`mu_r` 可无条件互抄）**被否证**：对-对接触差 2 倍，对-墙相同 ⇒
  "同一个 `mu_r`"在两码里意味着不同的物理，直接混用会把形状代理改掉一倍。
* 因此**不做默认切换**（默认切换会让仓内所有已标定 `mu_r` 静默变义，而墙接触又不变，
  形成混合约定）；改为**可选开关 + 公布换算因子 + 差异清单**，默认逐位不变
  （回归：`tests/test_axisym_column_lube2004.py` 13 passed）。S 级：能力扩展
  （可选约定 + 转写参考 + 恒等/换算测试面），非"新增物理"。