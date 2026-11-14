# AUDIT — CASE-COLUMN-COLLAPSE：realisation（初始构型）轴的几何可容许性裁决（m20259）

**锚点**：`CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌；ref: Lube et al. 2004 *JFM* 508:175-199, doi:10.1017/S0022112004009036；容差 < 15 %）
**性质**：验证欠账修复 + 门禁缺陷修复。轮 1–6 与 m17534/m17917/m18261/m18385 都把
「单种子、单一初始构型（HCP+gap, jitter=0），无系综散布」列为未完成；m19922 预登记了 15 例系综波却只归档 1 例
（`n_cases: 1`），且其机械裁决把 NaN 读成 pass。本单回答：① 系综轴能否用位置抖动实现；② 裁决为何不许把 NaN 当通过。

**产出**：`src/tensordem/granular_column/admissibility.py`（新，纯几何可容许性）、
`benchmarks/case_column_collapse_ensemble_admissible_m20259.py`（可容许系综波，积分前过几何门）、
`scripts/analyze_column_collapse_ensemble_m20259.py`（5 条预登记假设 + 机器裁决 + 2 图）、
`tests/test_column_collapse_ensemble_admissible_m20259.py`（9 项）、
`docs/artifacts/column_collapse_ensemble_m20259/`（csv/summary/manifest/2 png）、
`scripts/analyze_column_collapse_ensemble_m19922.py`（改：三态裁决）。

## ① 案例介绍
圆柱颗粒柱（$r_p=0.03$ m、8 粒/柱半径、HCP 格隙 2 % = 1.18 mm）抽壁坍塌，参考 Lube et al. (2004)：
$r_\infty/r_i = 1+1.24a\ (a<1.7)$、$1+1.6\sqrt a\ (a>1.7)$、$t_\infty=3\sqrt{h_i/g}$。
冻结闭合臂（未改一参数）：$\mu=0.5,\mu_r=0.5,\mu_w=0.8,e=0.5,k_n=6\times10^4$ N/m,$\rho=2500$ kg/m³,$g=9.81$。

## ② 怎么做的
1. 纯几何（零积分）普查初始态：`gap_ij=|x_i-x_j|-2r_p`、穿透对数、最大穿透深度；
2. 抖幅包络线扫描 jitter∈{0,0.0025,0.005,0.01,0.02,0.03,0.05,0.10}，判据 = 穿透对数 0（非可调容差）；
3. 补跑**可容许**系综（a=0.5，jitter 0.0025/0.005 × 2 seeds，成员积分前过门，有穿透即 abort）；
4. 用修正后的三态逻辑对 m19922 原波重新裁决（fail ≻ insufficient ≻ pass，阈值未动）。

## ③ 计算结果（`cases_ensemble.csv` 逐行可复算）
**表 1 初始态几何普查（seed 101；区间 = a=0.5 与 a=2.0）**

| jitter [$r_p$] | 最小间隙 [mm] | 穿透对数 | 最大穿透 [mm] | 可容许 |
|---|---|---|---|---|
| 0 (HCP) | +1.18 | 0 | 0 | 是 |
| 0.005 | +0.25…+0.41 | 0 | 0 | 是 |
| 0.01 | −0.37…−0.70 | 9…39 | 0.37…0.70 | 否 |
| 0.05 | −6.6…−8.3 | 1417…5622 | 6.6…8.3 | 否 |
| 0.10 | −14.2…−17.8 | 1854…7411 | 14.2…17.8 | 否 |

→ **jitter_max_admissible = 0.005 r_p，首个被拒档 0.01 r_p**（两个端 aspect 同一包络）。

**表 2 已归档成员跑距（a=0.5，基准 1.620）**

| 波 | jitter | 穿透对数 | $r_\infty/r_i$ | err | $t_{95}$ err | stop |
|---|---|---|---|---|---|---|
| m19922 lattice | 0 | 0 | 1.573 | −2.9 % | +38.4 % | at_rest |
| m19922 | 0.05 (s101) | 1420 | 1.567 | −3.3 % | −4.7 % | at_rest |
| m19922 | 0.05 (s202) | 1384 | 1.563 | −3.5 % | +1.3 % | at_rest |
| m19922 | 0.10 (s303) | 1870 | 2.299 | **+41.9 %** | +20.6 % | at_rest |
| m19922 | 0.10 (s404) | 1847 | 2.347 | **+44.9 %** | +30.0 % | at_rest |
| m20259 | 0.0025 (s505) | 0 | 1.412 | −12.8 % | +16.8 % | at_rest |
| m20259 | 0.0025 (s606) | 0 | 1.433 | −11.5 % | +9.7 % | at_rest |
| m20259 | 0.005 (s707) | 0 | 1.471 | −9.2 % | +20.4 % | at_rest |
| m20259 | 0.005 (s808) | 0 | 1.436 | −11.4 % | +8.2 % | at_rest |

峰值前缘速度：晶格 0.84 m/s → 10 % 抖幅 1.84/1.95 m/s（弹性能在 $t=0$ 被释放的直接证据）。

## ④ 结果图
![fig_admissibility_scan.png](../artifacts/column_collapse_ensemble_m20259/fig_admissibility_scan.png)
**fig_admissibility_scan.png** — 初始态最大穿透 vs 抖幅；可容许带止于 0.005 $r_p$，预登记抖幅深在不可容许区。

![fig_runout_vs_jitter.png](../artifacts/column_collapse_ensemble_m20259/fig_runout_vs_jitter.png)
**fig_runout_vs_jitter.png** — a=0.5 跑距误差 vs 抖幅（阴影 ±15 %）：不可容许段 +42…+45 %，可容许段 −9…−13 %。

## ⑤ 注意点
1. **预登记系综波几何上不成立**：2 % 格隙只够吸收 0.005 $r_p$ 抖动；0.05/0.10 抖幅造出 1400–1850 对穿透（深达 14.2 mm），
   是**高能初态**而非「同一根柱子的另一种实现」，故 m19922 的 H_BAND/H_JITTER 否证**不能**读成「物理上对初值极敏感」。
2. **可容许抖动也吃掉裕度**：跑距从 −2.9 % 平移到 −9.2…−12.8 %（A_SCALE 10.3 % 被否证），15 % 判据只剩 2.2 pp。
3. **时间律仍不可裁决**：12 个成员 $t_{95}$ 误差 −4.7…+38.4 %、符号不稳，与 m18261 H1/H3/H5 否证一致；本单新增一条未控轴 = 初态。
4. **未完成（如实申报）**：可容许波只跑 a=0.5；a=1.0/2.0 未跑 —— 集群作业 `cj2594`（6 分片）在节点上因
   `benchmarks/case_column_collapse_ensemble_m19922.py` 不在该 checkout 路径而 6 分片全部 `can't open file` 失败。
   `summary.json` 用 `incomplete_aspects: [1.0, 2.0]` 显式登记，pytest 断言它与行集一致。
5. **正确生成器**：抖动无法同时做到「可容许」与「无序」；需沉积/沉降生成目标 $\phi$ 的无穿透堆积。

## ⑥ 误差判定（显式回答「是否 < 3 %」）

| 关键量 | 基准 | 实测 | < 3 % ? | < 15 % ? |
|---|---|---|---|---|
| 跑距 a=0.5 jitter=0 | Lube 律 | −2.9 % | 是（边缘） | 是 |
| 跑距 a=0.5 可容许最差成员 | 同上 | −12.8 % | 否 | 是（裕度 2.2 pp） |
| 跑距 a=0.5 可容许均值 (n=4) | 同上 | −11.2 % | 否 | 是（裕度 3.8 pp） |
| 跑距 a=0.5 不可容许 10 % 抖幅 | 同上 | +41.9/+44.9 % | 否 | **否** |
| $t_{95}$ a=0.5，12 成员 | $3\sqrt{h_i/g}$ | −4.7…+38.4 % | 否 | 3/12 |

**总回答：否。** 唯一 <3 % 的是晶格跑距（−2.9 %），而可容许抖动把它推到 −12.8 %。

## ⑦ 裁决
| 假设 | 裁决 | 证据 |
|---|---|---|
| A_ADMISS 可容许成员穿透对数=0 | **supported** | 4/4 |
| A_BAND 可容许成员全在 15 % 内 | **supported** | 最差 12.8 % |
| A_SPREAD σ/mean ≤ 5 % | **supported** | 1.72 % (n=4) |
| A_SCALE 晶格→最大可容许抖幅漂移 ≤5 % | **falsified** | 10.3 % |
| A_GAP m19922 抖幅不可容许 | **supported** | 4/4，最大穿透 14.2 mm |
| m19922 H_BAND / H_SPREAD / H_JITTER 重裁 | **falsified ×3** | 44.9 % / 22.5 % / 49.2 % |

**门禁缺陷（已修）**：`_mean_sigma` 对 n=1 返回 σ=0（单成员「系综」报零散布）；H_JITTER 把 NaN shift 当 pass；
H_SPREAD 把「无法测量」记成 FAIL。改为三态（pass/fail/insufficient），NaN 与成员不足一律 insufficient，
fail 支配 insufficient 支配 pass，阈值未动。

## 复现
```bash
PYTHONPATH=src python3 benchmarks/case_column_collapse_ensemble_admissible_m20259.py --aspects 0.5 --nshards 4 --shard 0
PYTHONPATH=src python3 scripts/analyze_column_collapse_ensemble_m20259.py
PYTHONPATH=src python3 -m pytest tests/test_column_collapse_ensemble_admissible_m20259.py tests/test_column_collapse_ensemble_m19922.py -q  # 20 passed
```
