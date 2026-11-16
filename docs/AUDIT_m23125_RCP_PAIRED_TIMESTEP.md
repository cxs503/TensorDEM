# AUDIT m23125 · RCP 沉积算例池：时间步偏差门的配对（CRN）估计与种子预算

锚点：`CASE-DEM-RCP-PACKING-FRACTION`（随机密堆体积分数，Scott 1960 / Bernal & Mason 1960，验收带 0.62–0.66，10000 球）。

## ① 案例与要解决的问题

m22824 §3 对四个 `(μ, k_n)` 档做时间步偏差门，判据为

`|Δ| + t·SE_Δ < 5e-4`（Δ = φ(dt/2) − φ(dt)，m22558/m21401 预登记），

结果 3/4 档 `UNDECIDABLE`，可认证 0/4。原因是**区间太宽**：种子散布（σ≈3e-4）与容许带同量级，按非配对 Welch 估计要 ~5 种子/侧才收敛。

本审计的实质增量是**换估计量，不是放宽容差**：粗/细两次算例在同一 `seed` 下**共享同一条初始构型**（`DepositionSpec.box_width` / `initial_height` 与 `dt_frac` 无关，`initial_configuration` 只吃 `(n, box_width, height, radius, seed)`），于是种子的随机性是一对算例的**共模参数**，在逐种子差 `Δ_i` 中相消。容差 `TOL_BIAS`、置信度与三值裁决词汇全部从 `seed_bias_audit` 原样导入，未新增第六种判词。

## ② 怎么做的

1. **前提先验证，不靠读代码**：`tests/test_rcp_paired_timestep.py` 断言同一 seed、`dt_frac`=1/20 与 1/40 的 `DepositionSpec` 几何相同，且 `initial_configuration` 返回**逐字节相同**的坐标与相同的拒斥候选数；存档算例另做**指纹核对**——run JSON 的 `placement` 记录（`drop_height`、`initial_height`、`rejected_candidates`）在同一对的两次算例上必须数值全等，不等的对被丢弃并计数（`PAIRING_PREMISE_FAILED`），不静默平均。

2. **估计量**：`Δ_i = φ_fine,i − φ_coarse,i`，`SE = sd(Δ)/√n`，自由度 `n−1`，t 分布 95 % 区间；裁决沿用 `classify_bias`（区间完全落带内 CONVERGED / 完全落带外 BIASED / 跨带边 UNDECIDABLE）。**点估计永不认证**：`certified` 额外要求 `resolved_from == "interval"`，即 `n_pairs ≥ 2`。

3. **配对收益可核查**：`variance_reduction()` 在同一批配对数据上同时给出配对 SE 与非配对 Welch SE，比值 `SE_unpaired/SE_paired` 直接报告（本单存档数据上该比值见下表）；`seeds_needed_paired(σ, margin)` 给出配对估计量所需的种子预算，与非配对的 `seeds_needed` 并列。

4. **单命令复现**：`PYTHONPATH=src python benchmarks/case_rcp_deposition/paired_timestep_audit.py`；本文件由 `benchmarks/case_rcp_deposition/m23125_doc.py` 程序化生成。

## ③ 结果（数字由脚本注入）

### 3.1 逐档裁决：配对 vs 非配对（同一批算例）

| μ | $k_n$ | 种子(粗/细) | 配对数 | Δ 配对 | 配对 95 % 区间 | 配对裁决 | Δ 非配对 | 非配对裁决 | 可认证 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4000 | 3/1 | 1 | +2.477e-05 | — | **CONVERGED** | +2.576e-04 | UNDECIDABLE | 否 |
| 0 | 64000 | 3/3 | 3 | -7.298e-04 | [-1.699e-03, +2.392e-04] | **UNDECIDABLE** | -7.298e-04 | UNDECIDABLE | 否 |
| 0.1 | 4000 | 3/1 | 1 | -3.835e-04 | — | **CONVERGED** | +4.370e-05 | UNDECIDABLE | 否 |
| 0.1 | 64000 | 1/1 | 1 | +1.188e-04 | — | **CONVERGED** | +1.188e-04 | CONVERGED | 否 |

汇总：`{"n_certified_paired": 0, "n_converged_by_point_estimate_only": 3, "n_levels": 4, "n_pairs_premise_failed": 0, "verdicts_paired": {"BIASED": 0, "CONVERGED": 3, "NOT_RUN": 0, "UNDECIDABLE": 1}, "verdicts_unpaired": {"BIASED": 0, "CONVERGED": 1, "NOT_RUN": 0, "UNDECIDABLE": 3}}`

### 3.2 配对买到多少区间

| μ | $k_n$ | 配对数 | SE 配对 | SE 非配对(Welch) | 比值 | 配对数预算(带内 5e-4) |
|---|---|---|---|---|---|---|
| 0 | 4000 | 1 | — | — | — | None |
| 0 | 64000 | 3 | +2.252e-04 | +4.395e-04 | 1.95 | 5 |
| 0.1 | 4000 | 1 | — | — | — | None |
| 0.1 | 64000 | 1 | — | — | — | None |

### 3.3 锚点验收（φ 本身，未被本单改动）

| μ | $k_n$ | φ(粗, 均值) | 相对文献 0.6366 | 在 0.62–0.66 带内 |
|---|---|---|---|---|
| 0 | 4000 | 0.64237 | +0.91% | 是 |
| 0 | 64000 | 0.63676 | +0.03% | 是 |
| 0.1 | 4000 | 0.62628 | -1.62% | 是 |
| 0.1 | 64000 | 0.61928 | -2.72% | 否 |

![forest](benchmarks/case_rcp_deposition/figures/fig_m23125_paired_forest.png)

*配对（蓝）与非配对（灰）的 Δ 95 % 区间，绿带为容差 |Δ|<5e-4。区间越短，越可能离开 UNDECIDABLE——而带的位置一点没动。*

![scatter](benchmarks/case_rcp_deposition/figures/fig_m23125_pair_scatter.png)

*左：逐种子配对的 Δ_i；右：同一批数据上 SE_unpaired/SE_paired，即配对把区间收窄的倍数。*

## ④ 注意点与局限

1. **配对合法性的边界**：配对要求两次算例的**初始构型**相同。若将来有人改动 `initial_configuration` 使其依赖 `dt_frac`，或新旧摘要的放置规则不同，指纹核对会把这些对判 `PAIRING_PREMISE_FAILED` 并剔出估计——这是设计使然，不是失败。

2. **配对不能消除系统性偏差**：dt→0 的偏差本身被 Δ 直接测量，配对只降方差。因此 Δ 显著不为零时判 `BIASED` 是正确的物理结论，不是数值噪声。

3. **种子数仍受算力约束**：本单只补 μ=0 的两档（承载文献对标的档）；μ=0.1 两档仍是 1–3 对，其裁决保持 `UNDECIDABLE` 或点估计，不认证。

4. **跨摘要**：存档算例横跨 3 个实现摘要，m23050 已判 `EQUIVALENT_PATH` （可达函数逐字节相同）；本审计沿用该结论，未重新取证。

## ⑤ 误差判定（逐项，显式回答是否 <3 %）

| 量 | 计算值 | 基准 | 相对误差 | <3 %？ |
|---|---|---|---|---|
| φ(μ=0, $k_n$=4000) | 0.64237 | 0.6366 (Scott 1960; Bernal & Mason 1960) | +0.91% | 是 |
| φ(μ=0, $k_n$=64000) | 0.63676 | 0.6366 (Scott 1960; Bernal & Mason 1960) | +0.03% | 是 |
| φ(μ=0.1, $k_n$=4000) | 0.62628 | 0.6366 (Scott 1960; Bernal & Mason 1960) | -1.62% | 是 |
| φ(μ=0.1, $k_n$=64000) | 0.61928 | 0.6366 (Scott 1960; Bernal & Mason 1960) | -2.72% | 是 |

本审计的新增量是**估计量与种子预算**，不是新的物理量；Δ 与文献无可比性（Δ 是同一物理量在两个时间步下的差，基准是 0）。φ 的文献误差判定行如上表，未被本单改写。

## ⑥ 待办（交给下一单，附量化验收）

1. 本单已递交的配对补齐算例（`plan_m23125.json`，10 个作业）回件后重跑 `m23125_doc.py` 即可刷新本文；若 μ=0 两档达到 `seeds_needed_paired` 预算，裁决应从 `UNDECIDABLE` 变成有证据的 `CONVERGED`/`BIASED`。

2. 建议新条目 `CASE-DEM-RCP-TIMESTEP-PAIRED-CERT`（域：DEM·堆积）：验收=四个 `(μ,k_n)` 档在配对估计量下全部 `certified`（`resolved_from=interval` 且 verdict∈{CONVERGED,BIASED}），并给出逐档所需种子数与算例成本。


### 递交台账

```json
{
 "mission": 23125,
 "anchor": "CASE-DEM-RCP-PACKING-FRACTION",
 "note": "m23125 补齐『时间步偏差门』的配对（common-random-numbers）种子预算：m22824 §3 在 4 档中 3 档判 UNDECIDABLE，原因是粗/细两侧种子数不足，而两侧实际共享同一初始构型（同种子）。本账只补齐 μ=0 的两档——承载文献对标的档；μ=0.1 两档留待后续单。",
 "jobs": [
  {
   "jid": "cj4599",
   "tag": "deposition_n10000_mu0_kn4000_e0.1_seed2_dtf0.025",
   "why": "μ=0 kn=4000 细时间步 seed2（配对补齐）"
  },
  {
   "jid": "cj4600",
   "tag": "deposition_n10000_mu0_kn4000_e0.1_seed3_dtf0.025",
   "why": "μ=0 kn=4000 细时间步 seed3（配对补齐）"
  },
  {
   "jid": "cj4597",
   "tag": "deposition_n10000_mu0_kn4000_e0.1_seed4_dtf0.025",
   "why": "μ=0 kn=4000 细时间步 seed4"
  },
  {
   "jid": "cj4598",
   "tag": "deposition_n10000_mu0_kn4000_e0.1_seed5_dtf0.025",
   "why": "μ=0 kn=4000 细时间步 seed5"
  },
  {
   "jid": "cj4603",
   "tag": "deposition_n10000_mu0_kn4000_e0.1_seed4",
   "why": "μ=0 kn=4000 粗时间步 seed4"
  },
  {
   "jid": "cj4604",
   "tag": "deposition_n10000_mu0_kn4000_e0.1_seed5",
   "why": "μ=0 kn=4000 粗时间步 seed5"
  },
  {
   "jid": "cj4601",
   "tag": "deposition_n10000_mu0_kn64000_e0.1_seed4",
   "why": "μ=0 kn=64000 粗时间步 seed4"
  },
  {
   "jid": "cj4602",
   "tag": "deposition_n10000_mu0_kn64000_e0.1_seed5",
   "why": "μ=0 kn=64000 粗时间步 seed5"
  },
  {
   "jid": "cj4605",
   "tag": "deposition_n10000_mu0_kn64000_e0.1_seed4_dtf0.025",
   "why": "μ=0 kn=64000 细时间步 seed4"
  },
  {
   "jid": "cj4606",
   "tag": "deposition_n10000_mu0_kn64000_e0.1_seed5_dtf0.025",
   "why": "μ=0 kn=64000 细时间步 seed5"
  }
 ]
}
```
