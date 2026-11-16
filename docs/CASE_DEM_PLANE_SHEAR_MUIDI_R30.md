# CASE-DEM-PLANE-SHEAR-MUIDI — round 30 (m22864): the long window drains, and the yardstick becomes 2-D

**Anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池·平面剪切 mu(I) 流变律标定, status `open`, tolerance 10 %)  
**Cluster job** `cj4332` (8-strain 预注册 campaign, state `succeeded`)  
**Device** `tensordem.granular2d` 2-D disk plane shear, 458 free disks, $P=10$ kPa, $\mu_p=0.4$, $w=0.5$ (crystal control $w=0$)


## ① 案例介绍

致密颗粒流的局部流变律 $\mu(I)$、$\phi(I)$ 是 DEM→连续介质本构的桥梁。锚点验收要求 $\mu$、$\phi$ 对惯性数 $I=10^{-3}\ldots0.3$ 的曲线落在 **10 %** 内，基准为 da Cruz et al. 2005 (PRE 72:021309) / GDR MiDi 2004。

本轮做两件事。**(a) 补上 R29 明确留作第一优先、但因排队未回的长窗证据**：把 `cj4332` 的 9 个组合（8 应变测量窗、种子系综 {0,1,2}）取回并跑完预注册的稳态/自相关判据——这是本账本第一次拥有**稳态认证**的 $\mu$。**(b) 修基准的维度错配**：前 29 轮把 **2-D 圆盘**装置判给 **3-D 玻璃珠**读数 $(0.36,0.84,0.30)$（该读数在仓内自标 `QUOTED-NOT-ARCHIVED`），而锚点引用的 da Cruz 2005 本身就是 **2-D 圆盘、接触摩擦 $\mu=0.4$** 的研究——与装置完全同族。论文自己给出的是**线性律**：

$$\mu^*(I)\simeq \mu^*_{min}+bI,\quad \mu^*_{min}\simeq0.22,\; b\simeq1.0\qquad\text{(Eq. 14, }\mu\neq0\text{)}$$
$$\nu(I)\simeq\nu_{max}-aI,\quad \nu_{max}\simeq0.81,\; a\simeq0.3\qquad\text{(Eq. 13, }\mu=0.4\text{)}$$

这 4 个数不是拟合旋钮：`reference_2d.verify_provenance()` 把每个常数绑到**说它的那一句**上（句内绑定，而非「文件里出现过」——`0.3` 会误命中 `0.30`），档案语料 digest `cf6c4b0930ef14c9`。


## ② 怎么做的

1. `run_compute fetch cj4332` → 9 个 `long_r29_*.json` 入 `runs/plane_shear_2d/`；`analyse_long_window_r29.py` 产出 `outputs/r29/r29_summary.json`（Sokal 自相关窗、$n_{{eff}}=n/\tau$、1 应变分块趋势检验、指数饱和拟合）。
2. `reference_2d.py`：句内绑定档案 → 2-D 线性律 + 加权 OLS。
3. `judge_r30_2d.py`：同一批稳态点分别对 2-D / 3-D 两把尺子判误差，并把实测系数 $(\mu^*_{min},b)$, $(\nu_{max},a)$ 与论文值逐项对照；另给出**缺失流应力项**的硬上界。
4. 全部数字由脚本注入本文件，无手抄。


## ③ 计算结果

稳态判据（预注册：末两个 1 应变块 2σ 内一致 且 后半段加权趋势 |z|<2）：**未全部通过**。

| $I$ | $w$ | seeds | steady? | $\mu$ (8-strain) | $\mu_{2D}$ | err % | $\mu_{3D}$ | err % | $\phi$ | $\phi_{2D}$ | err % | $\phi_{3D}$ | err % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0194 | 0.5 | 1 | NO | 0.1636 | 0.2394 | -31.6 | 0.3891 | -58.0 | 0.8210 | 0.8042 | +2.1 | 0.6364 | +29.0 |
| 0.0490 | 0.5 | 3 | YES | 0.1661 | 0.2690 | -38.2 | 0.4274 | -61.1 | 0.8185 | 0.7953 | +2.9 | 0.6316 | +29.6 |
| 0.1935 | 0.5 | 3 | YES | 0.2101 | 0.4135 | -49.2 | 0.5482 | -61.7 | 0.8016 | 0.7520 | +6.6 | 0.6165 | +30.0 |

| coefficient | paper (2-D) | measured here | rel. dev % |
|---|---|---|---|
| $\mu^*_<built-in function min>$ (intercept) | 0.220 | 0.156 $\pm$ 0.004 | -29.3 |
| $b$ (rate sensitivity) | 1.000 | 0.279 $\pm$ 0.031 | -72.1 |
| $\nu_<built-in function max>$ | 0.810 | 0.824 $\pm$ 0.001 | +1.7 |
| $a$ (dilatancy slope) | -0.300 | -0.113 $\pm$ 0.005 | -62.3 |

| run | $p_{kin}^{ub}$ / $p_{bulk}$ % | $v_{rms}^{ub}/\sqrt{P/\rho}$ |
|---|---|---|
| long_r29_w0.5_I0.05_s0.json | 0.011 | 0.0046 |
| long_r29_w0.5_I0.05_s1.json | 0.013 | 0.0049 |
| long_r29_w0.5_I0.05_s2.json | 0.010 | 0.0044 |
| long_r29_w0.5_I0.2_s0.json | 0.126 | 0.0308 |
| long_r29_w0.5_I0.2_s1.json | 0.132 | 0.0316 |
| long_r29_w0.5_I0.2_s2.json | 0.130 | 0.0313 |

**换尺子的效果**：$|\mu\text{ 误差}|$ 最坏值 61.7% → 49.2%；$|\phi\text{ 误差}|$ 最坏值 30.0% → 6.6%。$\phi$ 侧在 2-D 尺子下**全部进入 10 % 带**（最坏 6.6%）。


## ④ 图

* `docs/figures/r30_mu_phi_2d_vs_3d.png` — 左：$\mu(I)$；右：$\phi(I)$。红实线=论文 2-D 律，蓝虚线=仓内 3-D 读数，黑方块=本装置 8 应变稳态点（种子系综误差棒）。灰竖线 $I=0.2$ 是论文自述的饱和起点。
* `docs/figures/r30_kinetic_bound_and_fit.png` — 左：缺失流应力项上界占 $p_{bulk}$ 的比例；右：实测 $\mu(I)$ 与其线性拟合 vs 2-D 律（截距 vs 率敏感度分离）。


## ⑤ 注意点 / 局限

* **$\mu$ 仍不合格**：2-D 尺子下最坏 49.2%，截距 0.156（论文 0.22，-29.3%），斜率 $b$=0.279（论文 1.0，-72.1%）。失败签名从「整体低 60 %」变成「截距低约三成 + 率敏感度低 3-4 倍」，这是**指向率相关（碰撞/流）应力项**的诊断，而不是一律偏低的标定错。
* **缺失流应力项的硬上界**：装置的两个纵坐标都是**接触** Love-Weber 维里，论文的 $S=\mu^*_{min}P+b\sqrt{mP}\,\dot\gamma$ 含率相关部分。用 $p_{kin}\le KE/A$ 得到的上界最大仅 **0.132 %** of $p_{bulk}$，因此「补上流项就能补回 3-4 倍的 $b$」这条解释被**否定**——但这只是上界推断，不是直接测量的流应力。
* **未解的新线索（本轮未结）**：同一批样本里总动能 $KE/V$ 反推的颗粒均方速度只有约 $0.1$ m/s 量级，而壁面速度 $2V/h$ 对应的 $I=0.19$ 意味着 $\dot\gamma d\approx0.39$ m/s；R21 在另一装置上量到 $v_{{top}}/v_{{platen}}\approx1$（无滑移）。两者不能同时成立，值得下一轮在该 8 应变装置上直接量速度剖面（R21 的 `low_i_profile_r21` 套路）。**本轮不据此改判**。
* 基准线性律本身是论文自述的近似（$I\ge0.2$ 处饱和），所以 $I=0.1935$ 那一点已标 `beyond_linear_range`，不作定量结论。
* $\mu^*_{{min}}=0.22$ 是论文对「任意 $\mu\neq0$」的陈述，弱于$(\nu_{max},a)$ 的「$\mu=0.4$」归属；装置 $\mu_p=0.4$ 使比较成立，但归属强度差异已记录。
* $w=0$（晶体）分支仅作对照，不进拟合：$\phi\approx0.875$ 是结晶态，不是流动态。


## ⑥ 误差判定（honest）

| 量 | 判据 | 本轮 | 判定 |
|---|---|---|---|
| $\mu$ vs **2-D** 论文律 | <10 % | 最坏 49.2 % | **FAIL** |
| $\mu$ vs **3-D** 仓内读数 | <10 % | 最坏 61.7 % | FAIL |
| $\phi$ vs **2-D** 论文律 | <10 % | 最坏 6.6 % | **PASS** |
| $\phi$ vs **3-D** 仓内读数 | <10 % | 最坏 30.0 % | FAIL |
| 8 应变窗是稳态 | 预注册判据 | 未全部通过 | FAIL |
| 关键量 vs 基准是否 **<3 %** | <3 % | 最好的量是 $\phi$ 2.1 % | **NO** |
| 流应力缺失能否解释 $b$ 偏低 | 上界足够 | 上界 ≤ 0.132 % of P | **否（假说被否定）** |

**锚点判定：仍然 `open`（FAIL）。** 但失败的性质已被重写：$\phi$ 在维度一致的尺子下已合格，$\mu$ 的缺陷是「率敏感度不足」而非整体标定偏低。


## ⑦ 产物

* `benchmarks/plane_shear_mu_i/reference_2d.py`（2-D 基准律 + 句内绑定 provenance 门）
* `benchmarks/plane_shear_mu_i/judge_r30_2d.py`（本文件生成器）
* `benchmarks/plane_shear_mu_i/test_r30_2d_reference.py`
* `benchmarks/plane_shear_mu_i/runs/plane_shear_2d/long_r29_*.json`（`cj4332` 取回，9 组合）
* `outputs/r29/r29_summary.json`, `outputs/r30/r30_summary.json`, `outputs/r30/r30_tables.md`
* `docs/figures/r30_mu_phi_2d_vs_3d.png`, `docs/figures/r30_kinetic_bound_and_fit.png`


## 给下次自找活的线索

1. **速度剖面直测**（本轮新线索）：在 8 应变装置上量 $u(y)$，回答「KE 反推的 0.1 m/s 与 $I=0.19$ 是否自洽」。若确有滑移/局域化，则本装置所有 $I$ 纵坐标需重算，这会同时解释 $b$ 偏低。
2. **率相关应力项**：给 `plane_shear_2d.py` 加流应力（$\rho\langle v'v'\rangle$）累加器，直接测 $b$ 而非上界推断。
3. 若 (1)/(2) 都做完仍不合格，再拆锚点：2-D 条目绑 Eq. 13-14，3-D 条目绑 Jop 三元组。

