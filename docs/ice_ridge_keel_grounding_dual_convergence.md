# 冰脊龙骨接地：颗粒数 × 网格双维收敛研究（W28）

> 唯一数值来源：`papers/artifacts/w28/ice-ridge-keel-grounding/dual_convergence.json`（本文件由脚本从工件渲染，零手抄）　·　生成器 `scripts/run_w28_dual_convergence.py`　·　mission 19906

**总体裁决：`HONEST_FAIL`**　C1 FAIL / C2 PASS / C3 FAIL / C4 WITHHELD / C5 PASS

## 1. 结论摘要

- **颗粒维（W27 遗留问题）确认不收敛**：主臂（W27 原键刚度律）四档颗粒（[25000, 50000, 100000, 200000]）的龙骨足尺阻力 $R(b=H_k)$ 的 GCI = **35.07%**、观测阶 $p_N$ = **1.2**（按间距口径 $p_a$=2.41）；与 W27 的 39.6% 同量级，且四档 log–log 斜率 = **0.5149**（正斜率 ⇒ 随颗粒数单调发散，不存在收敛极限）。
- **受控诊断臂给出根因与补救**：把键刚度律换成间距一致（$k_n=Ew_b/\Lambda$）后，四档 $R$ 的相对散布降到 **7.73%**（噪声量级），格点割线模量 $k$ 在四档上重合（主臂则按 $N^{0.541}$ 抬升）。这一条把 W27 的“模量随颗粒数漂移”从推测变为受控实验结论。
- **本轮的诚实缺口**：耦合（梁–龙骨）块的牛顿迭代在集群节点上**第一步即中止**（24/24 算例 `solution_finite=False`、平衡残差≈0.99），因此耦合块的 $R/A_c/q_{max}$ 三量**不作为证据**，网格维与贯入–接触力 Spearman 判据本轮 **NOT_EVALUATED / WITHHELD**；已登记为求解器缺陷与下一步。

## 2. 方法

两个离散化维度在同一耦合问题内：龙骨（颗粒，[25000, 50000, 100000, 200000] 档）给出压力–沉降律 $p(w)$ 与足尺阻力；结构（Euler–Bernoulli 梁，单元数 [32, 64, 128]）坐在**无拉**龙骨地基上，以 $w_0$=0.008 mm 的刚性压板驱动，量取 $R=\int p\,dx$、$A_c$、$q_{max}$。收敛判据预声明为 Roache GCI（安全系数 1.25）+ 观测阶，两臂只差键刚度长度一个变量。

## 3. 结果（程序化注入）

### 3.1 颗粒维收敛（DEM 足尺阻力 $R(b=H_k)$，四档）

| 臂 | 四档 $R$ [MN/m] | GCI(细) | $p_N$ | $p_a$ | Richardson 极限 [MN/m] | 不确定带 |
|---|---|---|---|---|---|---|
| legacy_bond_length | 52.274 / 71.375 / 115.355 / 146.364 | 35.07% | 1.2 | 2.41 | 37.61 | 48.75% |
| spacing_consistent | 1.377 / 1.325 / 1.276 / 1.280 | 92.38% | 0.072 | 0.144 | 2.395 | 53.12% |

### 3.2 龙骨模量与键律

| 臂 | $k$ 四档 [GPa/m] | log–log 斜率 | 键律 |
|---|---|---|---|
| legacy_bond_length | 1.849 / 2.643 / 4.319 / 5.479 | 0.541 | legacy_bond_length |
| spacing_consistent | 0.047 / 0.048 / 0.047 / 0.047 | 0.00294 | spacing_consistent |

### 3.3 Kerr 对账（最细颗粒档）

| 臂 | 足尺宽 $b$ [m] | DEM [MN/m] | Kerr 封闭解 [MN/m] | 相对偏差 | 角色 |
|---|---|---|---|---|---|
| legacy_bond_length | 1.0 | 41.320 | 46.347 | 12.17% | identification |
| legacy_bond_length | 2.0 | 74.701 | 90.180 | 20.72% | validation |
| legacy_bond_length | 4.0 | 146.364 | 177.847 | 21.51% | validation |
| spacing_consistent | 1.0 | 0.363 | 0.392 | 7.96% | identification |
| spacing_consistent | 2.0 | 0.655 | 0.770 | 17.55% | validation |
| spacing_consistent | 4.0 | 1.280 | 1.526 | 19.20% | validation |

### 3.4 耦合块 24 算例（登记为缺陷，不作证据）

| 臂 | $N$ | $h$ [m] | $R$ [MN/m] | $A_c$ [m²/m] | $q_{max}$ [MPa] | 牛顿步数 | 有限解 |
|---|---|---|---|---|---|---|---|
| legacy_bond_length | 24960 | 0.500 | 37.967 | 3.000 | 14.794 | 1 | **否** |
| legacy_bond_length | 24960 | 0.250 | 33.777 | 2.500 | 14.794 | 1 | **否** |
| legacy_bond_length | 24960 | 0.125 | 31.682 | 2.250 | 14.794 | 1 | **否** |
| legacy_bond_length | 50065 | 0.500 | 54.253 | 3.000 | 21.143 | 1 | **否** |
| legacy_bond_length | 50065 | 0.250 | 48.269 | 2.500 | 21.143 | 1 | **否** |
| legacy_bond_length | 50065 | 0.125 | 45.278 | 2.250 | 21.143 | 1 | **否** |
| legacy_bond_length | 99720 | 0.500 | 88.620 | 3.000 | 34.549 | 1 | **否** |
| legacy_bond_length | 99720 | 0.250 | 78.858 | 2.500 | 34.549 | 1 | **否** |
| legacy_bond_length | 99720 | 0.125 | 73.978 | 2.250 | 34.549 | 1 | **否** |
| legacy_bond_length | 200260 | 0.500 | 112.506 | 3.000 | 43.833 | 1 | **否** |
| legacy_bond_length | 200260 | 0.250 | 100.087 | 2.500 | 43.833 | 1 | **否** |
| legacy_bond_length | 200260 | 0.125 | 93.877 | 2.250 | 43.833 | 1 | **否** |
| spacing_consistent | 24960 | 0.500 | 0.956 | 3.000 | 0.372 | 1 | **否** |
| spacing_consistent | 24960 | 0.250 | 0.850 | 2.500 | 0.372 | 1 | **否** |
| spacing_consistent | 24960 | 0.125 | 0.797 | 2.250 | 0.372 | 1 | **否** |
| spacing_consistent | 50065 | 0.500 | 0.979 | 3.000 | 0.381 | 1 | **否** |
| spacing_consistent | 50065 | 0.250 | 0.871 | 2.500 | 0.381 | 1 | **否** |
| spacing_consistent | 50065 | 0.125 | 0.817 | 2.250 | 0.381 | 1 | **否** |
| spacing_consistent | 99720 | 0.500 | 0.956 | 3.000 | 0.373 | 1 | **否** |
| spacing_consistent | 99720 | 0.250 | 0.850 | 2.500 | 0.373 | 1 | **否** |
| spacing_consistent | 99720 | 0.125 | 0.798 | 2.250 | 0.373 | 1 | **否** |
| spacing_consistent | 200260 | 0.500 | 0.970 | 3.000 | 0.378 | 1 | **否** |
| spacing_consistent | 200260 | 0.250 | 0.863 | 2.500 | 0.378 | 1 | **否** |
| spacing_consistent | 200260 | 0.125 | 0.809 | 2.250 | 0.378 | 1 | **否** |

## 4. 判据逐条

| 判据 | 阈值 | 实测 | 裁决 |
|---|---|---|---|
| C1 颗粒维 GCI<5% 且阶≥1.5 | 5% / 1.5 | GCI 35.07%、$p_N$ 1.2（主臂）；网格维 NOT_EVALUATED | **FAIL** |
| C2 Kerr 解析 vs FEM 挠度 | 3% | 0.001% | **PASS** |
| C3 颗粒龙骨 vs Kerr 阻力 | 15% | 21.51%（主臂）/ 19.20%（诊断臂） | **FAIL** |
| C4 贯入–接触力 Spearman | 0.99 | 耦合块未收敛，数值不可用 | **WITHHELD** |
| C5 12 算例全部落工件（含失败） | 12 | 24（主 12 / 诊断 12）× 6 贯入级，失败算例含在册 | **PASS** |

## 5. 误差判定（对锚）

- Kerr 解析 vs 独立混合 Galerkin FEM 挠度偏差 **0.001%**（<3% ✔）。
- 本单梁求解器（线性极限）vs Hetenyi 无限长梁解析解：单测阈值 1%，实测通过。
- DEM 足尺阻力 vs Kerr 封闭解：主臂 **21.51%**、诊断臂 **19.20%**（>15%，本轮不合格；W27 在 100k 时为 0.6%）。
- ISO 19906:2019 A.8 / IACS UR I：原文不可得，工件内 `WITHHELD`，**无任何计算量依赖规范数值**。

## 6. 注意点与下一步

- the coupled beam-on-keel Newton iteration aborted at its first iteration on the cluster node for every one of the 24 cases, so R, A_c and q_max of the coupled block are NOT evidence and are kept only for traceability
- the same configuration converges locally in 6 Newton iterations with equilibrium residual < 1e-3 (see tests/test_w28_dual_convergence.py and the W28 dev log); the cluster node produced a non-finite assembly/step at iteration 1, so the failure is environment dependent and NOT yet diagnosed
- add an environment fingerprint (numpy/scipy versions + a deterministic coupled smoke case with checksum) to the runner, bisect the node-vs-local divergence, and re-run the coupled block before any convergence claim is made
- the keel is a bonded lattice with frozen topology (strain ~4e-3): no bond breakage, no block rotation, no rate effects; the freeze-bond modulus 1 GPa is a stiffness choice, so absolute pressures are model dependent -- the *convergence* statements are not
- the structure is a plane-strain beam (unit width) with free ends; the overhang past the contact area lifts off (no-tension) and is not carried by buoyancy
- the local keel constitutive law entering the coupled solve is the confined (wide-lid) compression curve; footing edge effects enter only through the reconciliation block
- the rigid-platen edge makes the contact pressure discontinuous, which caps the observed order of the uniform mesh family (remedy demonstrated with a platen-edge-graded family, reported separately)
- the no-tension switch is smoothed with a C1 ramp of width 0.1 w0; the ramp sensitivity block shows the spread (see extra_evidence.ramp_sensitivity)
- ISO 19906:2019 A.8 and IACS UR I values are WITHHELD: no standard number enters any computation in this artifact

## 7. 复现

```
PYTHONPATH=src python -m pytest tests/test_w28_dual_convergence.py -q
PYTHONPATH=src python scripts/run_w28_dual_convergence.py --jobs 4
```

```
PYTHONPATH=src python -m pytest tests/test_w28_dual_convergence.py -q
```

交付清单（sha256 / git blob）见 `papers/artifacts/w28/ice-ridge-keel-grounding/manifest.json`。
