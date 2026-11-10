# 散货舱颗粒卸料动力学验证面（LW3-N06, TensorDEM）

> 模块 `src/tensordem/bulk_cargo.py` · 驱动 `scripts/m8875_bulk_cargo_surface.py` ·
> 图 `scripts/m8875_bulk_cargo_figures.py` · 表 `scripts/m8875_bulk_cargo_tables.py` ·
> 用例 `tests/test_bulk_cargo_discharge.py`（18 项） ·
> 产物 `artifacts/bulk_cargo/{case,results}.json` + `artifacts/bulk_cargo/raw/*.json`

## 1. 案例介绍

### 1.1 物理背景

散货船把干散货（矿砂 / 煤 / 粮）装在一个箱形货舱里，靠舱底开孔（或抓斗）卸出。
三个宏观量决定这个舱能不能卸干净，而且**三者都是颗粒堆积的涌现性质**，没有不含
经验常数的解析预测：

1. **休止角** `theta_r`：自然堆积能维持的最陡自由面。它决定舱底/污水沟里卸不掉
   的**残货量**，也是 DEM 参数（滑动摩擦 `mu`、滚动阻力 `mu_r`）唯一能被工程侧
   测量的宏观对应量。
2. **卸料流率** `W(D)`：Beverloo 经验律
   `W = C rho_b sqrt(g) (D - k d)^(5/2)`（Beverloo, Leniger & van de Velde 1961），
   `C ~ 0.58`、`k ~ 1.4`（近球形颗粒）。它的**可证伪内容**是指数 `5/2` 与等效孔径
   修正 `k d`，以及与装料高度无关。
3. **舱壁压力** `p(z)`：深舱内 Janssen 分层平衡
   `sigma_v(z) = p_inf (1 - exp(-z / lambda))`，`lambda = R_h/(mu_w K)`。它**饱和**
   而不是像静水压那样随深度线性增长，是内壳板/斜板的设计载荷。

本节点把这三条定律放进**同一个**理想化棱柱货舱的三维离散元模型里测出来，再逐条
对照理论，从而填补 TensorDEM「散货装卸零验证面」的空白。

### 1.2 几何与参数

* 货舱：矩形平面 `L × B`，四面垂直舷壁，平底中央一个圆形卸料孔（可"关闭"=装料
  状态），孔的边缘按圆环（rim arc）接触处理，避免等效孔径被系统性放大。
* 颗粒：单分散球，`d = 2r = 10 mm`、`rho = 2500 kg/m³`、`mu = 0.5`（节点指定），
  `mu_rolling` 作为**材料角状度杠杆**扫描（0.05 圆珠 / 0.20 次棱角矿 / 0.50 棱角矿），
  `e = 0.3`、`k_n = 1e4 N/m`、`k_t/k_n = 0.5`、`g = 9.81 m/s²`。
* 接触律取 `tensordem.granular3d_core`：法向线性弹簧-阻尼、切向 Cundall–Strack
  增量弹簧（饱和于 `mu|F_n|`）、各向同性滚动弹簧（饱和于 `mu_r r |F_n|`）。
* 边界全解析（四壁、底板含孔洞+孔缘、供料漏斗四壁）：每颗粒携带**切向/滚动弹簧
  状态**，因此舱壁**静摩擦**（Janssen 饱和的物理来源）是被模拟出来的，而不是用
  速度相关滑块替代。墙-颗粒滚动阻力同样饱和于 `mu_r r |F_n|`——这一项缺失会让
  理想球在舱底上**无耗散地滚走**，正是"卸料堆摊平"这一失效模式的成因（见 §5）。

### 1.3 数值设置（CFL/稳定性自查）

`dt` 取接触周期判据（最硬弹簧每周期 20 步），并与 Rayleigh 表面波估计
`dt_R = pi r sqrt(rho/G)` 比较（`G = k_n/(2r)`）。全部用例的
`dt/dt_R ≈ 0.10`（见 `case.json` 的 `numerics` 段），即时间步由接触周期控制、
同时远低于表面波极限；该比值是**报告项不是放宽项**。

## 2. 方法

三个测量各自独立、且都只从粒子轨迹/质量守恒取数：

1. **休止角（倾倒法）**：方形供料漏斗底部开启，货柱自由落下在舱底堆成锥体；
   流完后静置 `settle_time`。自由面按径向环带取颗粒中心的**上分位高度**（分位给
   出的是一个**常数偏移**，因此不偏置斜率），在 `[max(0.2 r_pile, 漏斗脚印+r),
   0.85 r_pile]` 区间最小二乘拟合 `z = z0 - tan(theta) r`，并对 +x / +y 两个切面
   各做一次独立拟合作为交叉检查。
2. **Beverloo 流率（质量守恒）**：装料→堵塞静置→开孔，逐窗口统计
   `dm_discharged / window`，`W` 为窗口均值 ± 1 σ。**Beverloo 值只出现在对照字段**
   `W_theory_kg_s`，绝不出现在测量路径上（回归钉：
   `test_discharge_mass_balance_is_consistent` 要求窗口和 == 卸出质量）。
   另做 `log W = log A + p log(D - k d)` 拟合，检验指数是否为 2.5。
3. **Janssen 压力（静力平衡）**：堵塞满舱静置后，取静置后段的时间平均：
   `sigma_h(z)` = 水平层内四壁法向力 / 壁面积；`sigma_v(z)` = 切片力平衡
   `(W_above - F_wall,friction,above)/A`（不含任何应力张量假设）。

判据 C1–C5 的机械化判定在 `scripts/m8875_bulk_cargo_surface.py::evaluate()`，
数字全部落在 `results.json`。

### Criteria verdicts

| criterion | measured | reference | deviation | verdict |
|---|---|---|---|---|
| C1 repose angle | 0.02 deg | 30.0 deg | 99.9 % | FAIL |
| C2 Beverloo curve (worst point) | worst 84.8 % | 0 % | 84.8 % | FAIL |
| C3 Janssen profile (worst slab) | worst 28.7 % | 0 % | 28.7 % | FAIL |
| C3 identity p_inf/lambda = rho_b g | 13985.1 Pa/m | 14969.7 Pa/m | 6.6 % | PASS |
| C4a frictionless repose | 0.00 deg | < 5 deg | - | PASS |
| C4b blocked orifice W | 0.0000 kg/s | <= 5 % of the D/d=4 rate (0.00 %) | - | PASS |
| C5 end-to-end JSON keys | beverloo_curve, janssen_pressure, repose_angle | repose_angle/beverloo_curve/janssen_pressure | - | PASS |

### Poured-heap cases (repose angle)

| case | mu | mu_r | theta [deg] | tan(theta) | flank r2 | sections x/y [deg] | n particles |
|---|---|---|---|---|---|---|---|
| R4_frictionless | 0.00 | 0.00 | 0.00 | 0.000 | -220.950 | nan / nan | 1372 |
| R1_beads | 0.50 | 0.05 | 0.02 | 0.000 | 0.021 | nan / nan | 1372 |
| R2_ore | 0.50 | 0.20 | 5.26 | 0.092 | 0.789 | nan / 14.0 | 1372 |
| R3_angular | 0.50 | 0.35 | 10.27 | 0.181 | 0.885 | 9.2 / 18.3 | 1372 |

### Beverloo discharge cases

| case | D [mm] | D/d | W measured [kg/s] | +/- 1 sigma | W Beverloo [kg/s] | deviation | rho_b [kg/m3] | windows [kg/s] |
|---|---|---|---|---|---|---|---|---|
| W_D040 | 40.0 | 4.0 | 0.0458 | 0.0916 | 0.3022 | 84.8 % | 1526 | 0.183, 0.000, 0.000, 0.000 |
| W_D048 | 48.0 | 4.8 | 0.3534 | 0.1550 | 0.5909 | 40.2 % | 1526 | 0.314, 0.216, 0.308, 0.576 |
| W_D056 | 56.0 | 5.6 | 0.5252 | 0.2209 | 1.0021 | 47.6 % | 1526 | 0.445, 0.301, 0.530, 0.825 |
| W_D064 | 64.0 | 6.4 | 0.9839 | 0.2792 | 1.5496 | 36.5 % | 1526 | 0.720, 0.956, 1.276 |

### Janssen pressure profile (plugged hold, mu_w = 0.40)

| z below surface [mm] | sigma_h [Pa] | sigma_v [Pa] | Janssen theory [Pa] (lambda from measured K) | hydrostatic [Pa] | in criterion |
|---|---|---|---|---|---|
| 101.9 | 31 | 1461 | 1135 | 1525 | yes |
| 91.1 | 389 | 1243 | 1046 | 1364 | yes |
| 80.4 | 382 | 1134 | 951 | 1204 | yes |
| 69.7 | 547 | 995 | 850 | 1043 | yes |
| 59.0 | 269 | 808 | 741 | 883 | yes |
| 48.3 | 296 | 677 | 626 | 722 | yes |
| 37.5 | 307 | 512 | 502 | 562 | yes |
| 26.8 | 195 | 346 | 370 | 401 | yes |
| 16.1 | 144 | 192 | 229 | 241 | no (surface slice) |
| 5.4 | 101 | 9 | 79 | 80 | no (surface slice) |



## 3. 结果图

* `docs/bulk_cargo/fig_repose.png` —— (a) 倾倒堆自由面与锥拟合；(b) 休止角随
  `mu_r` 的变化 + 文献带。
* `docs/bulk_cargo/fig_beverloo.png` —— (a) `W(D)` 测量 vs Beverloo（标注逐点偏差）；
  (b) 幂律指数拟合。
* `docs/bulk_cargo/fig_janssen.png` —— (a) `sigma_h / sigma_v / 静水压` 沿深度分布
  与 Janssen 拟合；(b) 无量纲恒等式 `p_inf/lambda = rho_b g`。
* `docs/bulk_cargo/fig_controls.png` —— (a) `D < 2d` 负控；(b) 逐窗口流率平稳性。

## 4. 注意点（踩坑与适用范围）

1. **墙-颗粒滚动阻力必须显式建模。** 首版只做了颗粒-颗粒滚动阻力，理想球在舱底
   上无耗散滚动，倾倒料被摊成单层"薄饼"（`apex ≈ 1 层`，`r_98 ≈ 0.38 m`），测得
   休止角恒为 0°。加上墙侧滚动弹簧后才出现可测的锥面。
2. **漏斗壁坐标要跟着料柱走。** 首版把料柱平移了 `0.5 s` 而漏斗壁仍在 `0, s`，
   结果约 2/3 的颗粒被"墙"切成深穿透（首步速度即 24 m/s），仿真瞬间发散。此类
   错误只有在"初始几何自查 + 首步速度"两层检查下才暴露。
3. **薄堆是分辨率瓶颈。** 本仓库已验证的 `repose_angle_calibration`（另一节点）已
   记录：`mu_r = 0.2` 时球堆角度饱和在 `arctan(mu_r)` 附近，且 100 mm 半径的堆只有
   2–4 个粒径厚，侧翼仅被 ~6 颗粒分辨；本节点的堆同样受此限制（见 §5 的诚实结论）。
   提高分辨率的手段是更多颗粒 / 更小粒径 / 更大堆，代价是 CPU 时间。
4. **Beverloo 的适用前提。** 需要 `h >~ 3D`（料层足够深）且 `D << 舱宽`。本节点用
   `D/B = 0.33…0.53`、`h/D = 1.9…3`，属于"小舱 + 大孔"的偏离区；这也是逐点偏差
   必须逐点报告的原因，而不是靠调 `C` 吸收。
5. **单分散球、无黏聚、无粒径分布**，`mu_r` 固定；结论只能外推到同类材料参数，
   不能当作某具体货种的标定值。

## 5. 误差判定

表格见 §2（`C1`–`C5` 逐条给出实测/基准/偏差/判定）。关键量的<3% 回答：

* **是否 <3%？** —— 数值闭合量（窗口质量守恒、`p_inf/lambda = rho_b g`）达 3% 量级；
  **物理量本身（休止角 0.02–10.3° vs 文献 25–35°、绝对流率偏低 36–85%、Janssen 剖面
  最深切片偏低 29%）没有达到 3%，也没有达到本节点的 10%/10%/15% 线**。本条如实记
  FAIL，不粉饰。

## 6. 本次判定的物理解释（每一项 FAIL 的机制，可直接续接）

### 6.1 C1 休止角 FAIL：堆太薄 + 滚动律饱和（引擎级限制，非本节点实现错误）

实测 `mu_r = 0.05 / 0.20 / 0.35` 对应 `theta_r = 0.02° / 5.26° / 10.27°`，单调上升，
负控（`mu = mu_r = 0`）为 `0.00°` 且 `tan(theta) = 0`，说明**涌现机制本身是通的**
（摩擦产生坡度、去掉摩擦坡度消失）。但绝对角度比真实货种（25–35°）低一个量级。
这与本仓库另一已验证节点 `docs/repose_angle_calibration.md` §5 的记录一致：
单分散球堆的角度在 `mu_r` 上饱和，100 mm 半径的堆只有 2–4 个粒径厚，侧翼仅由
~6 颗粒分辨。本节点 1372 颗料的堆在 `theta ~ 10°` 时侧翼也只有 3–4 层。
**结论**：不是"把 C1 调过线"，而是要换分辨率（`r = 2 mm`、`R_toe/2r ≈ 25`）或
换滚动律；这是 TensorDEM 侧的引擎级 gap，已在本报告量化。

### 6.2 C2 Beverloo FAIL：几何把小舱效应带进来了

三个流得动的孔（`D/d = 4.8 / 5.6 / 6.4`）实测 `W = 0.35 / 0.53 / 0.98 kg/s`，
Beverloo 给 `0.59 / 1.00 / 1.55`，偏差 40%/48%/37%；`D/d = 4.0` 基本起拱
（首窗 0.18 kg/s 后归零）。诊断：

* 本舱 `D/B = 0.33…0.53`、`h/D = 1.9…3`，**不满足 Beverloo 的"深料层 + 孔远小于
  舱宽"前提**；窗口序列（见 `fig_controls.png` 右）显示流率仍在爬升，说明测到的是
  发育中的流，不是稳态。
* 等效系数 `C_eff = W_meas/(rho_b sqrt(g) (D-kd)^{5/2}) ≈ 0.35`（文献 0.58）——
  高壁摩擦（`mu_w = 0.4`）+ 墙侧滚动阻力 + 小舱共同压低流率。
* 幂律指数拟合为 **4.52**（Beverloo 主张 2.50，`r2 = 0.92`）——**不是**接近 2.5。
  机制：`D/d = 4.0` 那一点被起拱压到接近 0，而 `D/d >= 5.6` 的窗口仍在爬升（料层浅、
  孔大，整床像"局部放空"而不是稳态孔流），于是曲线在小孔端被压低、在大孔端被抬高，
  拟合指数被拉陡。这说明本几何下测到的**不是 Beverloo 稳态**，指数偏差是同一几何
  根因的第二个指纹（与逐点偏差互为佐证）。

**下一步（可机械执行）**：`B = 0.24 m`、`h = 0.16 m`、`D = 0.03…0.06 m`
（`D/B ≤ 0.25`、`h/D ≥ 2.7`），料量 ~6000 颗，单例约 15 CPU-min。

### 6.3 C3 Janssen：无量纲恒等式 PASS（6.6%），剖面 FAIL（最深偏低 29%）

`p_inf/lambda` 与 `rho_b g` 相差 6.6%：Janssen 分层平衡的**动量关系**是成立的。
但实测 `sigma_v` 比 Janssen 理论剖面**更接近静水压**（102 mm 处 1461 Pa vs 理论
1135 Pa，静水 1525 Pa），即壁面摩擦只承担了约 4% 的自重，Janssen 饱和没有真正
发育。机制：本舱料层只有 0.13 m，而 `lambda_theory = R_h/(mu_w K_meas) = 0.16 m`
——`h/lambda ≈ 0.8`，**比饱和长度还浅**；且料床是由"落料+静置"形成的密实床，
侧压力系数与壁摩擦状态都没被充分调动（`K_meas = 0.49`，`sigma_h` 只有
`sigma_v` 的 3–40%）。
**下一步**：`h ≥ 3 lambda`（例如 `R_h ≈ 0.03 m`、`mu_w = 0.4`、`K ≈ 0.5` 时需
`h ≈ 0.5 m`，或按比例缩小 `R`），并采用分层装料/振动压实以建立 Janssen 应力状态。

### 6.4 负控与端到端（PASS）

`D < 2d`（`D/d = 0.8`）时**零卸出**（起拱完全堵塞）；`mu = mu_r = 0` 时堆角 0°。
这两个负控证明验证面能区分"有物理"和"没物理"，也正是 C1/C2 的 FAIL 属**真实
物理/分辨率**问题、而非求解器空转的证据。