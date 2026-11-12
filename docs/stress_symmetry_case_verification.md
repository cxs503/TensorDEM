# 2-D 平面剪切的应力序参量：私有 Virial 求和并入共享仪器的验证账

**案件**：`CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT`（`tensordem.elastic_limit_shear`）  
**仪器**：`tensordem.granular2d.stress_arrays` → `tensordem.granular3d.stress.contact_outer`  
**产出脚本**：`scripts/verify_stress_symmetry_case.py`（数据） / `scripts/write_stress_symmetry_doc.py`（本文，零手抄）  
**工程参数**：$N=400$，剪切率 $\dot\gamma\in\{0.1, 0.3, 1\}$，摩擦系数 $\mu\in\{0.5, 0, 0.3\}$；其余取 case 默认值（$\phi=0.8$，$k_n=10000$，$k_t/k_n=0.5$，restitution $=0.7$，累计应变 $=3$，平均窗起始应变 $=1$）。

## 1. 案例介绍（物理背景与几何/参数）

Lees–Edwards 平面剪切：二维圆盘体系在周期剪切盒中被 SLLOD 方程驱动，是「弹性极限 / Bagnold 惯性极限」的判据工况。门禁关注的关键量是无量纲剪应力 $\tau^* = \tau/(\rho d^2\dot\gamma^2)$ 对 $\kappa$ 的log–log 斜率（弹性极限 $n\to 0$）与有效摩擦 $\mu_{eff}=\tau/P$。

问题的来源不是物理，而是**序参量的记账**：原实现把剪切应力写成

$$\tau = -\frac{1}{A}\Big[\sum_p m v_x v_y + \sum_c F_{i,x}(r_i-r_j)_y\Big] = \sigma_{yx}$$

即**原始 Virial 张量的一个非对角分量**，再叠加**对称**的动能项。对有心力（无摩擦）接触，原始 Virial 对称，$\sigma_{xy}=\sigma_{yx}$，该选择不可见；一旦存在 Coulomb 切向力，原始 Virial 不对称，二者相差

$$\sigma_{xy}-\sigma_{yx}=\frac{1}{A}\sum_c (l_x f_y - l_y f_x)=\frac{1}{A}\sum_c |l|\,f_t ,$$

正是共享仪器文档中所说的「非中心部分 $\sigma-\sigma^{T}$」。把其中一个分量称为「剪应力」并与对称动能项相加，是任何能量门禁都看不见的记账缺陷。

## 2. 怎么做的（方法/流程/工具链）

1. 新增 `tensordem/granular2d/stress_arrays.py`：把二维 $(l,f)$ 向量**零填充到三维**后交给已独立验证的三维仪器 `granular3d.stress.contact_outer`，返回面平均张量与两个非对角分量、对称/反对称部分、二维压强 $-(\sigma_{xx}+\sigma_{yy})/2$。**不重新推导任何公式**，二维与三维案例因此不可能各自漂移。
2. `elastic_limit_shear.pair_stress` 改为**经由仪器**求值（保留原返回契约 `(sigma_yx, P)`，历史结果可复现）；新增 `pair_stress_components` 暴露对称剪切、反对称（杠杆臂）部分与压强。
3. 力核 `_forces` 每步同时累加对称非对角和 $\sum(d_x f_y+d_y f_x)$ 与反对称和 $\sum(d_x f_y-d_y f_x)$；结果字典新增 `tau_symmetric`、`tau_minus_symmetric`、`tau_symmetry_defect`、`stress_asymmetry(_relative)`、`stress_instrument`。**门禁条件不变**（不做静默改判）。
4. 三组摩擦系数（0.5 默认 / 0.3 / 0.0 对照）各扫三个剪切率，量化差距、并对两个序参量分别拟合 log–log 速率指数，回答「门禁结论是否受修正影响」。

## 3. 计算结果（全部由 JSON 程序化注入）

| $\mu$ | $\dot\gamma$ | $\tau=\sigma_{yx}$ | 对称 $\tau_{sym}$ | $\tau-\tau_{sym}$ | 相对缺陷 | 逐样本 $\langle|\sigma_{xy}-\sigma_{yx}|\rangle/|\tau_{sym}|$ | $P$ | $\mu_{eff}$(旧/新) | 门禁 |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.10 | 103.6669 | 103.7645 | -0.0975 | 0.094% | 5.966% | 224.59 | 0.4616 / 0.4620 | PASS |
| 0.50 | 0.30 | 112.3274 | 111.5386 | 0.7888 | 0.707% | 4.519% | 229.44 | 0.4896 / 0.4861 | PASS |
| 0.50 | 1.00 | 117.0500 | 119.2807 | -2.2307 | 1.870% | 6.142% | 235.77 | 0.4965 / 0.5059 | PASS |
| 0.00 | 0.10 | 0.2610 | 0.2610 | 0.0000 | 0.000% | 0.000% | 0.96 | 0.2708 / 0.2708 | FAIL |
| 0.00 | 0.30 | 1.9275 | 1.9275 | 0.0000 | 0.000% | 0.000% | 7.13 | 0.2702 / 0.2702 | FAIL |
| 0.00 | 1.00 | 14.0431 | 14.0431 | 0.0000 | 0.000% | 0.000% | 55.80 | 0.2517 / 0.2517 | FAIL |
| 0.30 | 0.10 | 29.6235 | 29.6971 | -0.0736 | 0.248% | 4.836% | 75.99 | 0.3898 / 0.3908 | PASS |
| 0.30 | 0.30 | 36.4860 | 36.2152 | 0.2708 | 0.748% | 4.492% | 89.38 | 0.4082 / 0.4052 | PASS |
| 0.30 | 1.00 | 54.1106 | 54.5282 | -0.4176 | 0.766% | 4.210% | 129.11 | 0.4191 / 0.4223 | PASS |

两个序参量的速率指数（三点 log–log 斜率，由 `loglog_slope` 计算）：

| $\mu$ | $n$（历史 $\sigma_{yx}$） | $n$（对称 $\tau_{sym}$） | $|\Delta n|$ |
|---|---|---|---|
| 0.50 | 0.05244 | 0.06045 | 0.00801 |
| 0.00 | 1.72949 | 1.72949 | 0.00000 |
| 0.30 | 0.26269 | 0.26512 | 0.00243 |

库函数 `shear_rate_exponent`（同一 rate sweep，$\mu=0.5$）返回：`n_historical = 0.05244`，`n_symmetric = 0.06045`。

图：`artifacts/stress_symmetry_case/fig_stress_symmetry.png`（左：两序参量随剪切率的幂律；右：历史序参量的记账缺陷 vs $\mu$）。

## 4. 误差判定

本项**没有外部实验基准**（Campbell 的系数表 paywalled，案例说明里已声明该腿为 WITHHELD），因此判据是**内部一致性与解析恒等式**：

* 恒等式 $\sigma_{ab}=(1/A)\sum_c l_a f_b$：`pair_stress` 与仪器一致到 $10^{-13}$ 相对（`tests/test_stress_symmetry_case.py::test_pair_stress_is_the_instrument`）；
* 杠杆臂恒等式 $\sigma_{xy}-\sigma_{yx}=\sum_c|l|f_t/A$：解析钉扎 （`::test_tangential_gap_is_the_lever_arm_torque`），有心力极限下差距为 0 （`::test_central_forces_give_a_symmetric_virial`）；
* 无摩擦实跑（$\mu=0$）：两序参量在 $10^{-12}$ 相对内相同，即差距确实全部来自切向力，不是数值噪声（`::test_frictionless_run_has_no_ordinate_gap`）。

实测缺陷幅度：$\mu=0.5$ 时时间平均意义的相对缺陷 $|\tau-\tau_{sym}|/|\tau_{sym}|$ 最大为 $1.870\%$；逐样本 $\langle|\sigma_{xy}-\sigma_{yx}|\rangle/|\tau_{sym}|$ 最大为 $6.142\%$。两者之差说明：**瞬时反对称部分很大（$\sim$5–6%），但其时间平均几乎抵消**，所以历史序参量的系统性偏差是亚百分到百分量级，而不是与瞬时量同阶。

「是否 <3%？」逐项回答：

* 时间平均的序参量缺陷：**是**（最大 $1.870\% < 3\%$）。
* 速率指数 $n$：$\mu=0.5$ 时 $|\Delta n|=0.00801$，相对 $n_{hist}=0.05244$ 为 $15.27\%$ 量级 —— **该项不满足 3%**（斜率本身接近 0，微小位移即放大），必须如实登记：结论（弹性极限 $n\to0$、速率弱依赖）不变，但**发表的指数数值应以对称序参量重算**。
* 逐样本反对称幅度：**不满足 3%**（5–6%），它是物理的（切向力力矩），不是误差；把它当误差抹掉才是错的。

## 5. 注意点（坑与适用范围）

* 二维压强是 $-(\sigma_{xx}+\sigma_{yy})/2$；把零填充后的三维张量送进 `granular3d.stress.pressure`（$-tr/3$）会得到同一状态的 **2/3**，两者绝不可混用——已在模块文档与实现里显式区分。
* 差距的**瞬时**值与**时间平均**值不是一回事：只报平均会低估反对称部分，只报瞬时会把抵消误读成系统偏差。本账两者都报。
* 门禁条件未改：本轮只把序参量挂到共享仪器上并量化，**不静默改判**。若后续决定以对称序参量作为门禁量，需要单独一轮（历史 benchmark JSON 需重算）。
* 仍未接线：`bpm_calibration_feasibility.virial_stress_tensor`（torch 逐粒子 Love–Weber 私有和）依旧自算；该项在总账里明确保留为 open。

## 6. 复现

```
PYTHONPATH=src python3 scripts/verify_stress_symmetry_case.py   # ~10 min, 9 points
PYTHONPATH=src python3 scripts/write_stress_symmetry_doc.py     # regenerates this file
python3 -m pytest tests/test_stress_symmetry_case.py -q
python3 -m pytest tests/test_elastic_limit_shear.py tests/test_elastic_limit_stress_decomposition.py -q
```
