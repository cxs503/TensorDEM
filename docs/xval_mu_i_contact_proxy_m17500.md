# XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE — 接触局部 mu(I) 代理 vs 体相 mu(I) 的交叉对标 (m17500)

## ① 问题与 ref 出处（零手抄）

**问题**。柱体坍塌战役（m17111）用**接触局部**惯性数
`I_c = |v_t| / sqrt(f_n/(rho d^2))`（应力尺度 `P ~ f_n/d^2`、剪切率尺度 `dot gamma ~ |v_t|/d`）
配 Jop(2006) 常数代替体相惯性数 `I = dot gamma d / sqrt(P/rho)`，被登记为**负结论**。
本单把它变成可测的交叉对标：同一个伺服剪切盒稳态上，体相路线（REF）与接触代理路线（GOT）差多少？

**ref 出处（机器可查，逐条溯源）**

| 溯源对象 | 位置 |
|---|---|
| 负结论原文 | `oss_solutions` id **421**：repo=**TensorDEM**，feature_slug=`CASE-COLUMN-COLLAPSE`，tag=`mu-i-contact-proxy-negative`；evidence 记 `I_c 均值 0.02-0.03`、`a=1 误差 +17.8%`；transfer 记「下一步改用体相惯性数」 |
| 被测做法源码 | `src/tensordem/granular_column/inertial_friction.py`（`contact_inertial_number` / `mu_of_inertial_number`，`JOP_2006={0.36,0.84,0.30}`） |
| 体相参考路线源码 | `src/tensordem/mu_i_rheology.py` `simulate_servo_shear`（恒应力伺服 + Love-Weber 板压 + 剪切率剖面）；`src/tensordem/mu_i_contact_proxy.py`（本单新增，纯后处理） |
| 定律出处 | Jop, Forterre & Pouliquen (2006) Nature 441:727-730（常数同仓内 `LITERATURE_JOP`） |
| 上述负结论的工程现场 | `docs/case_column_collapse_round5_m17111.md` §②③（tag 同名） |

## ② 方法与实现

同一台设备（恒应力伺服剪切盒，R=5 mm 球、ρ=2500、μ_c=0.5、seed 5279），同一次运行里同时取两种读数：

* **REF（体相 $\mu(I)$，OSS 标准做法）**：`I_bulk = |d<v_x>/dz| d / sqrt(P_slab/rho)`，`mu_bulk = tau/P`；
  $P_\text{slab}$ 用 Love-Weber slab 维里（`bulk_virial_stress`），$d\langle v_x\rangle/dz$ 用 bin 线性回归（`shear_rate_profile`）。
* **GOT（接触代理 $\mu(I_c)$，被判做法）**：逐接触 $|v_t|$（含两体自转的面速度，与 `granular3d_core.contact_forces` 同约定）
  + 法向力 $f_n$ → $I_c$ → $\mu(I_c)$；按 $f_n$ 加权聚合到状态级（摩擦容量 $\sum \mu(I_c) f_n / \sum f_n$），
  与体相 `tau/P` 可直接比较。

实现要点（最小改动、可验证）：
* `src/tensordem/mu_i_contact_proxy.py`：`slip_speed_at_contacts` / `contact_proxy_stats` / `aggregate_proxy` / `shortfall` / `xval_row` / `xval_summary`（纯函数，无 I/O）。
* `ServoShearCase.contact_proxy: bool = False`（**默认关**）：为真时在采样点从既有 pair_out 取 `iu/ju/branch/f_n` 累积接触统计；
  积分器（库仑帽 $\mu|f_n|$）**不动**，历史算例逐位不变（测试 `test_device_diagnostic_is_opt_in`）。
* 窗口：与设备自身压力校验一致取**末 20 %** 样本（尾 60 % 作为数值敏感性对照，见 ⑤）。

## ③ 参数表（无量纲量齐全）

| 量 | 值 | 无量纲 | 值 |
|---|---|---|---|
| r / d | 5 mm / 10 mm | I_target | 0.01, 0.1, 0.6 |
| ρ_p | 2500 kg/m³ | I_bulk 实测 | 0.0095, 0.1090, 0.6008 |
| kn | 1.0e3 N/m（S2: 4.0e3） | kappa = kn/(ρ d³ γ̇²) | 本设备 shear_rate=0（I 控制）故 kappa=None（缺口，见 ⑤） |
| kt/kn | 0.5 | dt*ω0 | 步长取 2π√(m/2kn)/20 |
| e | 0.3 | dt | 2.25e-4 s（S2 减半） |
| μ_c（接触库仑） | 0.5 | 固含率 φ | 0.7257（I=0.1 运行） |
| μ_wall | 1.5 | 配位数 Z | 4.195 |
| P | 2 kPa（S1: 8 kPa） | Jop 常数 | μ_s=0.36, μ_2=0.84, I_0=0.30 |
| 盒子 | 0.12×0.05×~0.06 m | 判据阈值 | 15 % |

## ④ 对照表（ref vs got，误差打勾叉）

`REF` = 体相路线（$\tau/P$ vs $I_\text{bulk}$）｜`GOT` = 接触代理（$\mu(I_c)$ vs $I_c$）｜判据 15 %

| 算例 | I_bulk | mu_bulk | Jop(I_bulk) | 体相 vs Jop | I_c | mu_proxy | **I 映射误差** | **摩擦误差** | shortfall |
|---|---|---|---|---|---|---|---|---|---|
| I=0.01 | 0.0095 | 0.2890 | 0.3748 | +22.9 % ✗ | 0.0241 | 0.3926 | **+152.6 % ✗** | **+35.8 % ✗** | +20.9 % |
| I=0.1 | 0.1090 | 0.3270 | 0.4879 | +33.0 % ✗ | 0.0628 | 0.4285 | **+42.4 % ✗** | **+31.0 % ✗** | +11.4 % |
| I=0.6 | 0.6008 | 0.5167 | 0.6801 | +24.0 % ✗ | 0.2445 | 0.5436 | **+59.3 % ✗** | **+5.2 % ✓** | −15.1 % |
| S2 kn×4 (I=0.1) | 0.0640 | 0.3728 | 0.4444 | +16.1 % ✗ | 0.0397 | 0.4078 | +38.0 % ✗ | +9.4 % ✓ | +16.8 % |
| S1 P=8 kPa (I=0.1) | 0.1205 | 0.3993 | 0.4976 | +19.8 % ✗ | 0.0650 | 0.4315 | +46.1 % ✗ | +8.1 % ✓ | +10.9 % |

**裁决（预注册判据）**：`verdict_mapping = FAIL`（最差 152.6 %）、`verdict_friction = FAIL`（最差 35.8 %，基础扫描 3/3 行超阈）→
**接触局部代理不能替代体相惯性数**，与 oss_solutions#421 的负结论一致。
**同时** `verdict_reference = FAIL`（体相路线自身 vs Jop 常数 16-33 %）：本设备 kn=1e3 处于 Campbell 弹性极限外，**绝对摩擦量比较作废**；
稳健结论是**无量纲映射误差**（不依赖 Jop 常数）：$I_c/I_\text{bulk}=2.54,\ 0.576,\ 0.407$（三档），无常数比例。

**物理解释**（与 421 同向）：低速密集带 $I_c$ 被钉在 $I_0$ 以下，$\mu(I_c)\to\mu_s=0.36<\mu_c=0.5$；
`shortfall` 在 I=0.01 达 **+20.9 %**，即代理闭合让床体比接触律本身"更滑"——正是坍塌 a=1 臂 +17.8 % 超跑的机制。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

| 敏感性 | 观测量 | 结果 |
|---|---|---|
| S3 数值窗口（末 20 % vs 尾 60 %） | I_c、mu_proxy | I_c 漂移 −5.1 %…+11.5 %，mu_proxy 漂移 <0.9 %（5/5 算例）；代理结论对窗口不敏感 |
| S1 围压 2→8 kPa @I=0.1 | I_bulk ×1.106、I_c ×1.035；映射误差 42.4 %→46.1 % | 压力不改变映射缺陷 |
| S2 接触刚度 kn ×4 @I=0.1 | I_bulk ×0.587、I_c ×0.632；shortfall 11.4 %→16.8 % | 刚度（弹性极限）显著移动 $I_c$ 与 shortfall，说明 $I_c$ 不是材料函数 |

## ⑥ 失败模式与工件

* **失败模式 1（已复现的负结论）**：代理把 $\mu$ 上限钉在 $\mu_s$，低速密集带低估摩擦（shortfall +21 %）→ 流态偏滑。
* **失败模式 2（本单新增证据）**：$I_c \leftrightarrow I$ 映射不是常数（0.41–2.54），且随 kn 漂移 → 不能当同一无量纲数移植。
* **设备局限**：I 控制时 `shear_rate=0`，`elastic_limit_regime`/kappa 未生成（REF 路线 16–33 % 偏离的部分原因）；
  3 行中 4/5 算例 `servo_valid=False`（末窗压力偏差 −7.4 %…），故 REF 只作相对比较。**诚实 FAIL 已写入**：这项不能宣称 <15 % 通过。
* **工件**：`src/tensordem/mu_i_contact_proxy.py`、`src/tensordem/mu_i_rheology.py`（+诊断开关）、
  `scripts/xval_mu_i_contact_proxy_m17500.py`、`artifacts/xval_mu_i_contact_proxy_m17500/{results.json,fig_mu_vs_I.png,fig_errors.png}`、
  `tests/test_xval_mu_i_contact_proxy.py`（8 passed）、`prereg/xval_mu_i_contact_proxy_m17500.md`。
* **生态摩擦**：知识引擎 8021 离线（`/tensor/knowledge/search` → `engine_offline`），三查只能落 `oss_solutions`(SQL) + 仓内 grep；
  首次全脚本跑在 120 s shell 上限处被整组收割导致 artifacts 未落盘，改为逐算例即时 JSON（并行 3 进程）后成功。

## 本仓 vs OSS 差异清单 + 移植取舍

| 维度 | OSS/文献标准做法（ref） | 本仓实现（got / REF 侧） | 取舍 |
|---|---|---|---|
| 惯性数来源 | Jop 2006：体相 $\dot\gamma d/\sqrt{P/\rho}$ | 代理：逐接触 $|v_t|/\sqrt{f_n/(\rho d^2)}$（`inertial_friction.py`）；REF 侧：测量剪切率剖面 + Love-Weber 板压 | 保留代理为**可选接触律**（默认 None 时库仑不变），但不作体相替代；跨尺度用法须显式声明 |
| 应力通道 | 板压即 $P$ | 板压 + slab 维里 + 全盒维里三路并行，可查边界层偏差 | 数据齐全，保留 |
| 摩擦上限 | $\mu(I)\in[\mu_s,\mu_2]$ | 库仑帽 $\mu_c|f_n|$ 不变，代理仅诊断 | 不把 $\mu_s<\mu_c$ 的闭合强加给颗粒摩擦（避免 421 的钉死失效） |
| 数值窗口 | 稳态后段 | 末 20 % 判定 + 尾 60 % 对照 | 保留（已量化 <1 % 影响） |
| 边界条件 | 周期性剪切 + 恒应力 | x 周期、y 光滑墙、底/顶摩擦墙 $\mu_w=1.5$ | 与文献装置差异 → REF 的 16-33 % 偏离已有解释，不作精度宣称 |

**移植建议**：把 `contact_proxy` 诊断开关（默认关）+ `mu_i_contact_proxy` 纯函数集保留为**交叉对标基础设施**；
体相 $\mu(I)$ 校准继续用 `brash_ice_rheology.mu_i.calibration_report`（-a2 文学常数门），二者互不替代。
