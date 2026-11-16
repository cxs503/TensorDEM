# AUDIT m21145 - CASE-DEM-RCP-PACKING-FRACTION 锚点溯源与精度判定审计

**Mission** m21145 · **Repo** TensorDEM · **Artifacts sha256** `00908c38d79e256c`  
本文件全部数字由 `benchmarks/case_rcp_deposition/anchor_provenance_audit.py` 程序化写入。

## 1 结论先行

1. **无摩擦档 (mu=0) 的精度声称站得住**：生产值 φ=0.64237 vs 文献 0.6366，相对误差 +0.91%，判定 **PASS**（锚点 provenance = literature）。
2. **低摩擦档 (mu=0.1) 的 3% 声称不成立，降级为 WITHHELD**：现网唯一 mu 索引锚 φ=0.625 自身标为 *interpolated from DEM study*（provenance = derived），拿它认证同一代码族的输出是自指。可说的最强结论是廓形必要条件：PASS_BRACKET。
3. **跨脚本判定互相矛盾已消解**：同一硬球截距在 `docs/CASE_RCP_PACKING_FRACTION_M20918.md` 被判 -1.22%（vs 表值 0.6250）而 `runs/order_audit.json` 判 +3.07% FAIL（vs 无摩擦文献 0.6366）。两者锚点不同且都不是 mu=0.1 的文献点；本审计把锚点策略写进代码（`tensordem.rcp_references.judge`）。
4. **低摩擦外推阶不可辨识的欠账已量化**：p∈[0.5,1.5] 下截距区间 [0.61791, 0.61850]，宽度 0.00059；第四刚度水平 k_n=256000 已补测，四点拟合 p=0.8777530885312191、φ∞=0.61831；留一预测误差 4.51e-04。

## 2 锚点溯源分级表

| mu | phi_ref | 2σ 带 | provenance | 可认证? | 来源 |
|---|---|---|---|---|---|
| 0 | 0.6366 | [0.6266, 0.6466] | literature | 是 | Scott & Kilgour (1969); Bernal & Mason (1960) |
| 0 | 0.6400 | [0.6320, 0.6480] | literature | 是 | Torquato & Stillinger (2010) RMP review |
| 0.1 | 0.6250 | [0.6090, 0.6410] | derived | 否 | Pournin et al. (2001) PRE; interpolated from DEM study |
| 0.3 | 0.6000 | [0.5800, 0.6200] | unverified | 否 | Pournin et al. (2001) PRE; Silbert (2010) Soft Matter |
| 0.5 | 0.5850 | [0.5610, 0.6090] | unverified | 否 | Pournin et al. (2001) PRE; Onoda & Liniger (1990) PRL |
| 1 | 0.5600 | [0.5300, 0.5900] | unverified | 否 | Onoda & Liniger (1990) PRL; Silbert (2010) |

分级规则：`literature` 才可承载精度声称；`derived`（本仓构造/插值）与 `unverified`（复合引用、无页码表号转录）只能排序或做必要条件检验。

## 3 刚度阶梯与硬球截距

### mu = 0

| k_n | phi (n seeds) | std | artifacts |
|---|---|---|---|
| 4000 | 0.64237 (3) | 2.23e-04 | deposition_n10000_mu0_kn4000_e0.1_seed1.json<br>deposition_n10000_mu0_kn4000_e0.1_seed2.json<br>deposition_n10000_mu0_kn4000_e0.1_seed3.json |
| 16000 | 0.63791 (1) | - | deposition_n10000_mu0_kn16000_e0.1_seed1.json |
| 64000 | 0.63676 (3) | 3.13e-04 | deposition_n10000_mu0_kn64000_e0.1_seed1.json<br>deposition_n10000_mu0_kn64000_e0.1_seed2.json<br>deposition_n10000_mu0_kn64000_e0.1_seed3.json |
| 256000 | 0.63571 (1) | - | deposition_n10000_mu0_kn256000_e0.1_seed1.json |

- 三点拟合: p = 0.985, φ∞ = 0.63637
- 四点拟合: p = 不可辨识 (n/a), φ∞ = 0.63536
- 留一预测检验: 预测最硬一级误差 7.63e-04, 纳入后截距位移 1.01e-03
- 阶次区间 p∈[0.5,1.5] → φ∞ ∈ [0.63466, 0.63556]

### mu = 0.1

| k_n | phi (n seeds) | std | artifacts |
|---|---|---|---|
| 4000 | 0.62628 (3) | 3.99e-04 | deposition_n10000_mu0.1_kn4000_e0.1_seed1.json<br>deposition_n10000_mu0.1_kn4000_e0.1_seed2.json<br>deposition_n10000_mu0.1_kn4000_e0.1_seed3.json |
| 16000 | 0.62159 (1) | - | deposition_n10000_mu0.1_kn16000_e0.1_seed1.json |
| 64000 | 0.61928 (1) | - | deposition_n10000_mu0.1_kn64000_e0.1_seed1.json |
| 256000 | 0.61860 (1) | - | deposition_n10000_mu0.1_kn256000_e0.1_seed1.json |

- 三点拟合: p = 0.512, φ∞ = 0.61705
- 四点拟合: p = 0.878, φ∞ = 0.61831
- 留一预测检验: 预测最硬一级误差 4.51e-04, 纳入后截距位移 1.26e-03
- 阶次区间 p∈[0.5,1.5] → φ∞ ∈ [0.61791, 0.61850]

## 4 判定矩阵（锚点策略）

| mu | 量 | 值 | 账簿带内? | 锚点 | 相对误差 | 判定 | 最强可声明 |
|---|---|---|---|---|---|---|---|
| 0 | production k_n=4000 | 0.64237 | 是 | 0.6366 (literature) | +0.91% | PASS | PASS |
| 0 | intercept (3 levels) | 0.63637 | 是 | 0.6366 (literature) | -0.04% | PASS | PASS |
| 0 | intercept (4 levels) | 0.63536 | 是 | 0.6366 (literature) | -0.19% | PASS | PASS |
| 0 | intercept bracket (low edge, p=1.5) | 0.63466 | 是 | 0.6366 (literature) | -0.31% | PASS | PASS |
| 0 | intercept bracket (high edge, p=0.5) | 0.63556 | 是 | 0.6366 (literature) | -0.16% | PASS | PASS |
| 0.1 | production k_n=4000 | 0.62628 | 是 | 0.6250 (derived) | +0.20% | WITHHELD | PASS_BRACKET |
| 0.1 | intercept (3 levels) | 0.61705 | 否 | 0.6250 (derived) | -1.27% | WITHHELD | PASS_BRACKET |
| 0.1 | intercept (4 levels) | 0.61831 | 否 | 0.6250 (derived) | -1.07% | WITHHELD | PASS_BRACKET |
| 0.1 | intercept bracket (low edge, p=1.5) | 0.61791 | 否 | 0.6250 (derived) | -1.13% | WITHHELD | PASS_BRACKET |
| 0.1 | intercept bracket (high edge, p=0.5) | 0.61850 | 否 | 0.6250 (derived) | -1.04% | WITHHELD | PASS_BRACKET |

## 5 门控

| id | 判据 | 实测 | 状态 |
|---|---|---|---|
| A1_frictionless_production_vs_literature | |phi(k_n=4000, mu=0) - 0.6366|/phi_ref < 3% | 9.071e-03 | PASS |
| A2_frictionless_intercept_bracket_stable | nuisance-order bracket width < 0.005 (mu=0) | 9.034e-04 | PASS |
| A3_lowfriction_intercept_bracket_stable | nuisance-order bracket width < 0.005 (mu=0.1) | 5.856e-04 | PASS |
| A4_lowfriction_order_checkable | mu=0.1 ladder has >=4 stiffness levels (leave-one-out check exists) | 4 | PASS |
| A5_no_claim_on_derived_anchor | 0 accuracy claims (this audit + merged ledger) rest on a non-literature anchor | 0 | PASS |
| A6_policy_coverage | every verdict carries anchor provenance + policy | 10 | PASS |

## 6 与已并订单审计的一致性

| gate | mu | 截距 | vs 无摩擦文献 | vs mu 表值 | 表值 provenance | 表值判定 |
|---|---|---|---|---|---|---|
| G13_vs_ref_mu0 | 0 | 0.63637 | +0.04% | -0.04% | literature | PASS |
| G13_vs_ref_mu0.1 | 0.1 | 0.61705 | +3.07% (FAIL) | -1.27% | derived | WITHHELD |

## 7 图

![intercept](../benchmarks/case_rcp_deposition/figures/fig_m21145_intercept_policy.png)

*刚度阶梯、三点/四点硬球截距（点线）、阶次不确定性区间（阴影）、账簿带与文献 RCP 线。*

![verdict](../benchmarks/case_rcp_deposition/figures/fig_m21145_verdict_map.png)

*每个申报量相对 mu 索引锚点的相对误差与判定色：绿 PASS / 红 FAIL / 黄 WITHHELD。*

## 8 注意点与局限

1. `derived` 判定不否认 mu=0.1 的数值本身落在廓形内，只否认「以它为基准的 −1.22% 是验证」。
2. 廓形必要条件（0.56 ≤ φ ≤ 上限）很宽，它只能排除荒谬值，不能替代文献锚。
3. 时间步 dt=tc/20 的系统性未在本审计重估，沿用 order_audit 的 dt/2 锚（内部一致）。
4. mu=0.1 第四刚度水平（k_n=256000, 2 seeds）已递交集群 cj3208，按台账语义不设节点；回件后重跑本脚本即可让 A4 门自动转绿，本文件无需手改。
5. 若后续取得 Pournin/Silbert 原文的表值（页码+表号转录），把该条 provenance 改为 `literature` 即可让判定自动升级为 PASS/FAIL —— 代码无需改动。

## 9 误差判定

| 关键量 | 基准 | 计算值 | 误差 | <3%? |
|---|---|---|---|---|
| production k_n=4000 (mu=0) | 0.6366 (literature) | 0.64237 | +0.91% | 是 |
| intercept (3 levels) (mu=0) | 0.6366 (literature) | 0.63637 | -0.04% | 是 |
| intercept (4 levels) (mu=0) | 0.6366 (literature) | 0.63536 | -0.19% | 是 |
| intercept bracket (low edge, p=1.5) (mu=0) | 0.6366 (literature) | 0.63466 | -0.31% | 是 |
| intercept bracket (high edge, p=0.5) (mu=0) | 0.6366 (literature) | 0.63556 | -0.16% | 是 |
| production k_n=4000 (mu=0.1) | 0.6250 (derived) | 0.62628 | +0.20% | 无基准(锚点非文献)→WITHHELD |
| intercept (3 levels) (mu=0.1) | 0.6250 (derived) | 0.61705 | -1.27% | 无基准(锚点非文献)→WITHHELD |
| intercept (4 levels) (mu=0.1) | 0.6250 (derived) | 0.61831 | -1.07% | 无基准(锚点非文献)→WITHHELD |
| intercept bracket (low edge, p=1.5) (mu=0.1) | 0.6250 (derived) | 0.61791 | -1.13% | 无基准(锚点非文献)→WITHHELD |
| intercept bracket (high edge, p=0.5) (mu=0.1) | 0.6250 (derived) | 0.61850 | -1.04% | 无基准(锚点非文献)→WITHHELD |

