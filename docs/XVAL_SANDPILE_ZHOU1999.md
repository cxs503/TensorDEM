# XVAL-DEM-SANDPILE-REPOSE-CALIBRATION-ANCHOR — TensorDEM vs Zhou et al. (1999)
> 生成：`benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py --aggregate`；所有数字由 JSON 工件程序化注入，无手抄。
## ① 问题 + 参考出处
参考（零手抄，运行时解析）：Zhou, Y.C., Wright, B.D., Yang, R.Y., Xu, B.H., Yu, A.B. (1999). Rolling friction in the dynamic simulation of sandpile formation. Physica A 269, 536-553.
- 锚点行出自 `docs/GAP_DEM_CLUMP_ROUND2.md` (sha256 `4767513195d247b0…`)：
  > | Zhou et al. (1999) *Physica A* 269:536–553 | lifted-box 沙堆 repose 标定算例；球 22–25°，角砾/椭球 35–45° | **外部锚**：趋势方向 + 球基线带（沿用仓内 `LIT`） |
- 解析所得球基线带：**[22, 25]°**（中心 23.5°）；非球形带：(35.0, 45.0)
- 书目行出自 `src/tensordem/protocols/sandpile.py` (sha256 `d5c98856e9771c85…`)。

本单检验：本仓 `tensordem.protocols.sandpile`（`sandpile-v1` 冻结协议，priors μ_s=0.5, μ_r=0.05）能否在 <15% 内复现该带。

## ② 方法与实现
| 环节 | 做法 |
|---|---|
| 参考值 | 运行时正则解析仓内文献记录（文件 sha256 入 verdict），不写死数字 |
| 前向模型 | `SandpileRun`：喷嘴撒料→自由落体→静置→自由面 h(r) 环带扇区最大高程→25–85% 趾半径线性拟合 θ=atan(−slope) |
| 冻结保护 | `SandpileConfig` 冻结 + `assert_priors`：摩擦先验只能经营白名单负控制（μ_r=0 / μ_s=0）偏离 |
| 判据 | 预先登记 D1–D6（见脚本 docstring），阈值写死在脚本常量中，跑后未改 |
| 执行 | 每 case 单线程 torch 子进程，JSON 落 `artifacts/XVAL-SANDPILE-ZHOU1999/cases/` |

## ③ 参数表（无量纲数完整）
| case | N | q* [-] | v_inj [m/s] | Fr=v_inj/√(gR) | H/R [-] | k*=k_n/(ρgR²) [-] | t*=t_settle/√(R/g) [-] | dt/t_ref [-] | μ_s | μ_r |
|---|---|---|---|---|---|---|---|---|---|---|
| as_delivered | 4000 | 90.30 | 0.480 | 2.169 | 8.00 | 9786 | 8.9 | 4.596e-03 | 0.50 | 0.050 |
| ctrl_mu_r_zero | 2000 | 11.29 | 0.379 | 1.710 | 2.00 | 2446 | 88.6 | 9.192e-03 | 0.50 | 0.000 |
| ctrl_mu_s_zero | 2000 | 11.29 | 0.379 | 1.710 | 2.00 | 2446 | 88.6 | 9.192e-03 | 0.00 | 0.050 |
| gentle_anchor | 2000 | 11.29 | 0.379 | 1.710 | 2.00 | 2446 | 88.6 | 9.192e-03 | 0.50 | 0.050 |
| sens_N1000 | 1000 | 11.29 | 0.379 | 1.710 | 2.00 | 2446 | 88.6 | 9.192e-03 | 0.50 | 0.050 |
| sens_N4000 | 4000 | 22.58 | 0.758 | 3.421 | 2.00 | 2446 | 88.6 | 9.192e-03 | 0.50 | 0.050 |
| sens_dt_half | 2000 | 11.29 | 0.379 | 1.710 | 2.00 | 2446 | 88.6 | 4.596e-03 | 0.50 | 0.050 |
| sens_settle_0p6 | 2000 | 11.29 | 0.379 | 1.710 | 2.00 | 2446 | 26.6 | 9.192e-03 | 0.50 | 0.050 |

无量纲组定义（R=5 mm, ρ=2500 kg/m³, g=9.81 m/s², t_ref=√(R/g)=22.6 ms）：q*=投料率×t_ref，drop/R=释放点高出堆顶的落差，k*=k_n/(ρgR²)，t*=静置时间/t_ref，Fr=喷嘴出口速度/√(gR)。

## ④ 对照表（ref vs got，误差打勾叉）
| case | θ_sim [°] | ±stderr | r² | Z(配位数) | φ(探针) | vs 参考 [22,25] |
|---|---|---|---|---|---|---|
| as_delivered | 9.51 | 0.77 | 0.885 | 2.08 | 0.564 | ❌ |
| ctrl_mu_r_zero | 0.00 | 0.00 | 0.007 | 0.00 | nan | ❌ |
| ctrl_mu_s_zero | 0.00 | 0.00 | 0.593 | 0.00 | nan | ❌ |
| gentle_anchor | 19.05 | 0.47 | 0.994 | 3.86 | 0.581 | ❌ |
| sens_N1000 | 15.88 | 1.46 | 0.930 | 3.25 | 0.621 | ❌ |
| sens_N4000 | 18.34 | 0.28 | 0.997 | 3.96 | 0.604 | ❌ |
| sens_dt_half | 16.70 | 0.35 | 0.996 | 3.72 | 0.623 | ❌ |
| sens_settle_0p6 | 18.93 | 0.47 | 0.994 | 3.84 | 0.581 | ❌ |

**锚点 case 误差**：|19.05 − 23.5|/23.5 = **18.92%**（判据 <15%：❌）

### 判据逐项
- `D1` ❌ |19.05 - 23.50|/23.50 = 18.92% <= 15%
- `D2` ❌ theta = 19.05 deg inside [22.0, 25.0]
- `D6` ✅ r2=0.994 (>= 0.95), stderr=0.466 deg (<= 1.0), Z=3.86 in [3.0, 7.0]
- `D3` ✅ theta(baseline) - theta(mu_r=0) = 19.05 deg (>= 3.0) and theta(mu_r=0) = 0.00 <= 20.0
- `D4` ❌ |theta(baseline) - theta(mu_s=0)| = 19.05 deg (<= 3.0)
- `D5` ❌ spread over ['anchor', 'sens_N1000', 'sens_N4000', 'sens_settle_0p6', 'sens_dt_half'] = 3.18 deg (<= 3.0)

## ⑤ 收敛性 / 参数敏感性（≥2 点）
| 变体 | θ [°] | Δθ vs 锚点 [°] | 说明 |
|---|---|---|---|
| sens_N1000 | 15.88 | -3.18 | gentle feed: pour_rate=500 /s through a 0.020 m nozzle, drop=0.010 m, k_n=1500 N/m, settle=2.0 s |
| sens_N4000 | 18.34 | -0.72 | gentle feed: pour_rate=500 /s through a 0.020 m nozzle, drop=0.010 m, k_n=1500 N/m, settle=2.0 s |
| sens_settle_0p6 | 18.93 | -0.12 | gentle feed: pour_rate=500 /s through a 0.020 m nozzle, drop=0.010 m, k_n=1500 N/m, settle=2.0 s |
| sens_dt_half | 16.70 | -2.35 | gentle feed: pour_rate=500 /s through a 0.020 m nozzle, drop=0.010 m, k_n=1500 N/m, settle=2.0 s |

## ⑥ 失败模式与工件
- 决策：`xval_within_15pct = False`，`all_checks_pass = False`，`honest_fail_stage = D1/theta_error_gt_15pct`
- 工件：`artifacts/XVAL-SANDPILE-ZHOU1999/cases/*.json`（每 case 完整记录：surface_profile / packing / fabric / energy / feed / env）、`verdict.json`、`figures/*.png`
- 失败模式（实测）：
  - `as_delivered`：θ = 9.51°，Z = 2.08，φ = 0.564 ⇒ 不是沙堆而是散喷（配位数≈0），该配置下的 repose 读数不可用于对标。

### 图
![fig1_profiles.png](artifacts/XVAL-SANDPILE-ZHOU1999/figures/fig1_profiles.png)
![fig2_theta_vs_band.png](artifacts/XVAL-SANDPILE-ZHOU1999/figures/fig2_theta_vs_band.png)
![fig3_sensitivity.png](artifacts/XVAL-SANDPILE-ZHOU1999/figures/fig3_sensitivity.png)

## ⑦ 交叉路线一致性（同接触参数、不同边界条件）
路线 B = `tensordem.repose_calibration`（提筒塌落，与 Zhou 的 lifted-box 同族）。

| 路线 | μ_s | μ_r | θ [°] | flank r² | 备注 |
|---|---|---|---|---|---|
| A `sandpile-v1` | 0.50 | 0.05 | 19.05 | 0.994 | 撒料+L/R≈16 |
| B `repose_calibration` | 0.50 | 0.05 | 5.19 | 0.634 | N=95, dt=6.64e-05 s |
| B `repose_calibration` | 0.50 | 0.20 | 8.08 | 0.570 | N=95, dt=6.64e-05 s |

- 文档表节点（`docs/repose_angle_calibration.md` 解析，零手抄）：μ=0.50, μ_r=0.20 → θ=12.54°；本单同参数重跑 8.08° （差 35.5%）——文档参数与本脚本参数不完全一致/沉积过薄，**该节点不可复现即不可作锚**。
- 同参数（μ_s=0.50, μ_r=0.05）两条路线差 **13.9°**；路线 B 的 flank r²=0.63 < 0.95（薄沉积，见 `docs/repose_angle_calibration.md` 已记局限）⇒ B 路线本单**不具锚定资格**，判定只落在 A 路线。这一分歧说明本仓休止角是**驱动相关**的，不能把偏差全部归给接触律。

## ⑧ 误差判定（逐项，显式回答「是否<3% / <15%」）
| 关键量 | 参考 | 本仓 | 相对误差 | <3%? | <15%? |
|---|---|---|---|---|---|
| 球堆休止角 θ | 23.5°（带中心 [22,25]°） | 19.05° | 18.9% | 否 | 否 |
| 配位数 Z（堆体资格） | ≈4–6（密堆范围，本仓判据 (3.0, 7.0)） | 3.86 | — | 否 | — |

结论：**honest FAIL**（`D1` 未过：18.9% > 15%）。失败点不是数值发散（r²=0.994、静置收敛 0.12°），而是**系统性偏低**：撒料边界条件把每球落高（释放平面固定在堆顶估算高度之上，实际落高≈16R）的动能注入堆体。机制侧与 Zhou-1999 一致（去掉滚动摩擦后堆体退化为θ=0.00°、Z=0.00 的单层），故偏差可定位到边界条件而非接触律，但**未经改边界条件验证**，不下因果结论（仅相关性）。

### 局限 / 适用范围
- 参考只到**带级**：Zhou-1999 的算法与算子细节本单不可得（知识引擎离线、`oss_solutions` 无读权限），故不能做实现级 diff，只能做带级+机制级对标（WITHHELD 部分如实声明）。
- `μ_r` **不能扫描**（`assert_priors` 防标定）：论文的 μ_r 曲线不可复现，只测了 μ_r=0 端点。
- 锚点值**未对 dt/N 完全收敛**：`dt×0.5` 使 θ 降 2.35°、N=1000 使 θ 降 3.17°（D5 未过），故 19.05° 本身带 ±2° 数值不确定度，18.9% 的偏差与数值误差同阶。
- 预登记假设 `D4`（μ_s 不敏感）**被证伪**：μ_s=0 时 θ=0.00°，说明滑动摩擦在撒料堆里同样承重（与「滚动摩擦唯一主因」的朴素读法不同）。
- 单分散球、无黏结、平底摩擦板；不适用于非球形料（非球形带 (35.0, 45.0) 本单未测）。
