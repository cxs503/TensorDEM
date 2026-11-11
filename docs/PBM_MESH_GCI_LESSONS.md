# 可复用规则：给有限体积 PBM 做网格无关性（GCI）时的四条硬规则

来源：节点 `p488w1-n09-pbm-gci`（TensorDEM，2026-11-11），实现 `src/tensordem/erosion_pbm/mesh_gci.py`、
研究档 `artifacts/p488w1_n09/metrics.json`、回归 `tests/test_erosion_pbm_mesh_gci.py`。
适用范围：线性破碎/ coagulation 算子的固定枢轴（Kumar–Ramkrishna）形式，以及任何"守恒量 + 组分网格"的格式。
不适用：移动枢轴法、矩方法（QMOM/DQMOM）、把守恒量当收敛量的其它格式。

## R1 初始/边界数据必须在每个网格上都是同一个物理量

* **规则**：进料/边界颗粒用两枢轴权重放置（`initial_particle`），不要"放进包含它的那个 bin 并给单位权重"。
* **机制**：单位权重把被追踪的质量钉在 bin 代表上，`Σ v_k n_k(0) = v_pivot ≠ v_0`，偏差 `O(h)` 且随网格改变。
* **证据**：`q=2/3/24` 时初始质量偏高 18.9% / 12.2% / 1.5%；改两枢轴放置后 `Σ v_k n_k(0)` 精确 `= v_0`（1e-16），
  观测阶从"假 p≈1"变回真实 p=2.0，`q=3` 的 `M0` 偏差从 +7.6% 降到 −0.1%。
* **反例**：任何"守恒量逐位相同所以收敛了"的论断，若初始条件本身带 `O(h)` 偏差，就是被污染的。

## R2 用矩阵指数推进强非正态线性算子时，先怀疑库，不要先怀疑自己

* **规则**：`A = -diag(S) + M diag(S)`（`M` 稀疏）时用均匀化级数 `exp(At)n0 = e^{-λt} Σ (λt)^k/k! P^k n0`，
  以**算子作用** `Pv = v + (M(Sv)-Sv)/λ` 施加；不要用 `scipy.linalg.expm`，也不要给 `solve_ivp` 配 `atol=1e-30`。
* **机制**：`cond(A) ~ 1e10` 时 Padé 分母病态，`expm` 会**无 warning 返回 NaN**；`atol=1e-30` 让微小尾部 bin 支配步长。
* **证据**：`K=72/144` 上 `expm` 返回 NaN；`solve_ivp(DOP853,atol=1e-30)` 在 `K=144` 用了 79 s；
  均匀化 + 稀疏作用 5 个时间点 15 ms，截断尾 ≤6e-16。
* **回归守卫**：`test_propagator_conserves_mass_to_machine_precision` 断言漂移 `<1e-12` 且 `tail < 1e-12`。

## R3 可变网格的加密参数化：用"整体梯度比"，不用"每 bin 增长比"

* **规则**：`ln V_k = ln v_min + L(e^{β k/N}-1)/(e^β-1)`，`β = ln R`；加密 `N→fN` 时 `R` 不变。
  网格"梯度"指标用**对数宽**之比（`R^{1-1/N}`），不要用算术宽之比。
* **机制**：每 bin 增长比固定时，`N` 变化会改变整条网格的*形状*，三档不再是同一族 ⇒ 序列非单调、`p=nan`。
* **证据**：旧参数化（`γ=1.05`，`K=72/144/288`）给出 `monotone=False`、`R` 从 2.21 漂到 1.22；
  新参数化 `p` 回到 1.05–1.31，且 `R=4` 族 `GCI ≤ 5.3e-3`。

## R4 Richardson 外推的符号与"退化序列"必须被测试钉死

* **规则**：`f_ex = f_f + (f_f - f_m)/(r^p-1)`；若代码里 `d = f_m - f_f`，写 `f_ex = f_f - d/(r^p-1)`。
  三档逐位相同/非单调时**打旗**（`certified=False`），GCI 仍给数值但标注不认证。
* **机制**：符号写反时 GCI **幅度**碰巧不变（`|d/(r^p-1)|`），只有外推值会偏离真值 `2Ch^p` ⇒ 只测 GCI 抓不到。
* **证据**：初版符号错，合成 `f(h)=f0+ch²` 序列的外推值偏 `2ch²`；`test_richardson_recovers_first/second_order`
  与 `test_richardson_flags_a_degenerate_sequence` 现为常驻回归守卫。

## 附：一条不能省的对照组

"GCI 很小"永远是可疑的，除非同时给出**故意粗化**的对照：本节点用 `N=10` 档把 `M0` 误差抬到 22.3%，
证明度量对网格缺陷敏感（`C5_negative_control`）。缺了这条，小 GCI 无法与"量具失灵"区分。

最后核验：2026-11-11，`python -m pytest tests/test_erosion_pbm_mesh_gci.py -q` → 43 passed。