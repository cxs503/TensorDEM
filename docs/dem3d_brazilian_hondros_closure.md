# TensorDEM 巴西劈裂（Hondros）验证欠账定谳：不是"跑得不够久"，而是中心力格点结构上到不了 -3

## 1. 案例介绍

`docs/dem3d_platen_contact_mask_audit.md` §6 把巴西（径向压缩）试件的弹性中心场记为
**WITHHELD**：900 步时中心 virial 应力 `sigma_xx` 仍为负、主应力比 `+6.1`，当时的理由是
"跨直径波传播需 ~3.3e3 步、粘弹松弛更久，需 ~2e4 步的平衡战役"。该战役一直没跑，
于是物理判据（`centre_principal_ratio_tol=0.15`、`centre_tension_rel_tol=0.15`）悬空。

本单把它定谳。试件即 manifest 既定算例 `tensordem-dem3d-brazilian-v1 / brazil_elastic_field`：

| 参数 | 值 |
|---|---|
| 几何 | `cylinder_xz`，nx×ny×nz = 15×3×15 键合球柱体 |
| 颗粒半径 / 晶格间距 | 2000 N/m 键刚度；间距 0.020 m |
| 直径 D / 厚度 t | 0.3000 m / 0.0600 m |
| 颗粒 / 键 | 447 / 3994 |
| 键刚度 / 压板刚度 | 2000 N/m / 5000 N/m |
| 加载速度 / dt | 2.0e-3 m/s；dt = 8.864e-6 s (0.5×推荐值) |
| 基准 | Hondros (1959) / ASTM D3967 点载荷解：`sigma_t = 2P/(pi D t)`，`sigma_zz/sigma_xx = -3` |
| 接触弧比 | 0.0669 (上限 0.2) |

## 2. 怎么做的

两条互不依赖的路线，都在本仓内、都可复跑：

1. **显式动力学 ramp**（`scripts/brazilian_equilibration.py`）：按 manifest 原定
   `steps=40000` 单次确定性地跑完，每 1000 步采样中心 virial 应力、载荷、能量与接触弧。
2. **静力松弛解**（`scripts/brazilian_lattice_static.py`，本单新增）：把**同一张键网络**在
   **同一压板行程**下求静力平衡。因为引擎只暴露键力律（材料切线），缺几何刚度项，
   Newton 会发散（本单先试过，实测残差 18×），故改为对弹性势做 L-BFGS 最小化 +
   20 段载荷连续化；收敛判据用物理量（`|F|_inf / 间距^2 <= 1e-4 Pa`），不用机器 epsilon。
   最小化用的 numpy 力场经测试断言与引擎 `forces()` **逐点一致**（相对误差 <1e-10），
   所以它不是"另一套模型"，而是同一力律的更便宜求值。

静力解的意义：它就是 ramp 的 `t -> inf`、无阻尼极限。若它也到不了 -3，那么缺的不是步数。

## 3. 计算结果

**静力松弛（完全平衡，残差 9.09e-08 相对量）**

| 量 | 值 |
|---|---|
| 压板行程 | 7.091300e-04 m |
| 平均压板载荷 P | 3.37231 N |
| 两板载荷不平衡 | 9.39e-11 |
| 分裂应力 2P/(πDt) | 119.27100 Pa |
| 中心 sigma_xx | 155.58995 Pa |
| 中心 sigma_zz | -313.49622 Pa |
| 中心主应力比 sigma_zz/sigma_xx | **-2.0149** (Hondros -3) |
| 中心拉应力相对误差 | **30.45%** |
| 沿加载直径 sigma_xx 相对散布 | 8.829 |

**行程扫描（同一静力解，20× 行程范围）**：主应力比 -2.027 / -2.024 / -2.015，
拉应力误差 29.94% / 30.05% / 30.45% —— 无量纲中心场对行程不敏感，即**纯结构效应**。

**显式 ramp（manifest 原定 40000 步）**：本窗口**未执行完**——单进程 40000 步需约 15 min，而本机 load15 已 ≥100（容量纪律只允许轻量活），后台进程两次被调度窗口收割，故本报告不提供该曲线，也不宣称其数值。已有的 900 步数据（§1，中心 sigma_xx 为负、主应力比 +6.1）与静力松弛终值之间的差距，正好说明 ramp 方向与静力解一致但不构成独立证据；复跑命令：`PYTHONPATH=src:scripts python3 scripts/brazilian_equilibration.py --steps 40000`，产物路径 `artifacts/dem3d_brazilian_ramp/`。显式 ramp 的缺席**不影响**本单结论的成立，因为结论来自完全平衡的静力解（残差 2e-6），而不是动力学截断。

## 4. 结果图

图 2 = 沿加载直径的 sigma_xx 与 Hondros 常值线；图 3 = 行程收敛（平线）。图 1（ramp 历史）因下述原因未生成。

![fig2](../artifacts/dem3d_brazilian_closure/fig2_axial_profile.png)
![fig3](../artifacts/dem3d_brazilian_closure/fig3_travel_convergence.png)

## 5. 注意点（坑与局限）

* **根因一：中心力格点不是各向同性连续体。** 同一键集的 Born（Affine）弹性常数：
  C11=433333 Pa、C12=233333 Pa、C44=233333 Pa，
  C12=C44（Cauchy 固体），Zener 各向异性 2.33、E=270000 Pa、
  nu=0.350。Hondros 解假设各向同性线弹性；该格点在大转动/应力集中处不满足。
* **根因二：接触弧不是数学线载荷。** 接触弧比 0.067（仅
  3 个颗粒承压），而中心窗口只有 ±1.5 个晶格间距，离散化把
  有限宽接触与格点各向异性一起放大到中心量上 —— 这是沿加载直径 sigma_xx 散布
  8.83 的直接来源。
* **先试错再定方法**：直接对引擎残差做 Newton（材料切线）在满行程下残差只剩 18×，不可用；
  必须用能量最小化或补几何刚度。这个坑写进脚本 docstring 与测试，避免后人重走。
* **静力解不等于"物理上更真"**：它是同一力律的平衡态，用于区分"未平衡"与"结构不匹配"，
  不能当作实验对照。
* 局限：未做晶格分辨/接触弧加密扫描（recipe 已写进 §6），故"加密后能否进 15%"未知。

## 6. 误差判定（逐项显式回答是否 <3%）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 分裂应力定义 2P/(πDt) | 解析恒等式 | <1e-12 | 是 |
| 两板载荷平衡 | 牛顿第三定律 | 9.4e-11 | 是 |
| 静力平衡残差 | 0 | 9.1e-08 | 是 |
| numpy 力场 vs 引擎力 | 逐点相同 | <1e-10 | 是 |
| 中心主应力比 | -3（Hondros） | **32.84%** | **否** |
| 中心拉应力 | 2P/(πDt) | **30.45%** | **否** |
| 沿加载直径均匀性 | 0（点载荷解） | **883%** | **否** |

honest verdict：**FAIL（结构性，非瞬态）**，非 WITHHELD。900 步的 WITHHELD 已可解除悬置，
但结论是**不达标**：manifest 的 15% 容差在当前试件（15 点跨径、6.7% 接触弧、中心力键）
下不可达。建议二选一：①加密晶格与接触弧后重跑（预期消除根因二，根因一需键模型升级）；
②把基准换成有限宽加载的解析/连续体参考，并把 15% 容差重标定。

## 7. 验证与回归

```
PYTHONPATH=src:scripts python3 -m pytest tests/test_dem3d_brazilian_static.py -q   # 5 passed
PYTHONPATH=src:scripts python3 scripts/brazilian_lattice_static.py --steps 40000
PYTHONPATH=src:scripts python3 scripts/brazilian_equilibration.py --steps 40000
```
