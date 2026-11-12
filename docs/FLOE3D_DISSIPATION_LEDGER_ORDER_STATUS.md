# floe3d 接触耗散账本：假设未证实（honest FAIL）与可复用测量面

**状态：HONEST FAIL —— 假设「步初速度记账 → O(dt)；改中点速度 → O(dt²)」在本仓的
1-DOF 阻尼撞击算例上没有被证实。** 失败如实登记，不粉饰；下面给出可复现的测量、
数字、两个候选根因和下一步。

## 1. 起点

m16361 第 3 轮 `docs/H1_BROKEN_ICE_DEM_ROUND3.md` §5.5 记：「`energy_residual` 在冲击段
由 0.093 升到 0.165，高于 `case3_repose` 的 0.05 有效门；判读为冲击期耗散项按步初速度
记账的 O(dt) 误差随耗散功线性增长」。本单把这条**判读**变成可测问题。

## 2. 做了什么

1. 核对 `polyhedron_floe3d/dynamics.py:step()`：drag / buoyancy 的功已经用踢半步的中点
   速度记账（`dE = F·(v_n+v_{n+1})/2 dt`），而接触/墙耗散用步初速度 `P(v_n) dt` ——
   两条账的口径确实不一致，代码事实成立。
2. 实现中点口径：`_apply_pair` / `_apply_wall` 只收集功率系数（`c_n`、`mu f_n`、滚动上限
   `cap_r`）与几何，踢半步之后由 `_settle_dissipation` 用中点速度统一结算；力律仍在踢前
   求值、不读账本。`dissipation_ledger="begin"` 保留修复前口径（逐位复现旧算术）作为对照，
   默认 `midpoint`。
3. 建独立基准：1-DOF `m x'' + c x' + k x = 0`，`c = 2 zeta sqrt(k m)`，
   `D = c ∫ xdot² dt` 用 `scipy.integrate.quad`（epsabs 1e-14）独立求积，
   `e = exp(-pi zeta/sqrt(1-zeta²))`；二者都不经过 DEM 积分器。
   工具：`scripts/verify_floe3d_energy_ledger_order.py`（案例 A 有闭式解；案例 B 三冰排
   跌入方箱，量账本恒等式残差），产物 `artifacts/floe3d_energy_ledger/{results.json,fig_ledger_order.png}`。

## 3. 测量结果（`artifacts/floe3d_energy_ledger/results.json`）

案例 A（`zeta = 0.10`、`v0 = 0.4 m/s`、`kn = 2e5`、每接触期 237…3789 步）：

| 口径 | D 相对闭式解的误差（dt 由 2e-4 到 1.25e-5） | log-log 阶 |
|---|---|---|
| midpoint（默认） | 9.20e-2, 9.52e-2, 9.36e-2, 9.45e-2, 9.37e-2 | **-0.00** |
| begin（修复前） | 8.94e-2, 9.40e-2, 9.30e-2, 9.42e-2, 9.35e-2 | **-0.01** |

→ 误差**不随 dt 下降**（阶 ≈ 0），两种口径差 0.4 % 量级；`e` 实测 ≈ 0.9998 而闭式解
0.729，能量落差 ≈ 2× 解析耗散。案例 B（无摩擦、3 冰排、多体接触）残差同样阶 ≈ 0
（4.79e-2 … 4.81e-2）——账本恒等式在多体情形不收敛。

**结论：本单未能把 m16361 的「O(dt) 记账误差」判读证实。** 门槛逐条：G1/G2/G4/G5/G6/G8
FAIL，只有 G3（begin 阶 < 1.5，无意义地成立）、G7（轨迹逐位不变）PASS。

## 4. 两个候选根因（下一个切片该查哪）

1. **测量面自身**：`e` 实测 ≈ 1 而能量落差 = 2× 解析耗散，说明「分离时刻」的判读可能
   取在仍在接触的步（`pe_spring` 在 step 开头写、`velocity` 同刻取值，索引错一位就会
   同时解释 e≈1 与 2× 落差）。先修测量面再谈记账口径。
2. **接触账本更深的缺口**：`_apply_pair` 把「切向力 × 切向速度」整额记为耗散，即便处于
   粘着（`|f_t| < mu f_n`、力来自切向弹簧 `xi`）——粘着期的功其实是存进 `0.5 kt |xi|²`
   （已计入 `pe_spring`），再记一次耗散就是**重复记账**。这是有界项，不会随 dt 收敛到 0，
   与「残差恒定不降」一致，也与案例 B 摩擦变体残差更大（5.2e-2 vs 4.8e-2）一致。

## 5. 已交付的可用部分

* `scripts/verify_floe3d_energy_ledger_order.py`：闭式基准 + 双口径 + 阶拟合 + 图，**可复用**。
* `tests/test_floe3d_energy_ledger_order.py`（4 条，绿）：解析基准自洽、**账本不改轨迹**、
  耗散非负、非法口径名拒收。原有 `test_polyhedron_floe3d_dynamics.py`、
  `test_polyhedron_floe3d_kernel_ledger.py` 全绿（合计 14 passed）。
* 代码改动仅是把「两条账口径不一致」这件事变成可选口径 + 单一结算点，**不改变任何轨迹**
  （G7 逐位相等），因此不改变任何已发布的力学结论；但它**没有**被证明更准，故本单不声明
  精度增益。

## 复现

```bash
PYTHONPATH=src python scripts/verify_floe3d_energy_ledger_order.py   # 退出码 1（有 FAIL 门）
PYTHONPATH=src python -m pytest tests/test_floe3d_energy_ledger_order.py -q
```
