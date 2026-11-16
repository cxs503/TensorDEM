# CASE-DEM-PLANE-SHEAR-MUIDI — R34: 谁的压强进 `I`，谁的应力进 `mu`？——估计量配对的验证欠账（外加一条自我否证）

锚点：`CASE-DEM-PLANE-SHEAR-MUIDI`（算例池·平面剪切 mu(I) 流变律标定，`open`，容差 10 %，$I=10^{-3}\ldots0.3$）
本单：m23200（自找活·proposer 席）｜仓 TensorDEM｜分支 `agent/m23200`
预登记：`benchmarks/plane_shear_mu_i/prereg_r34.json`（**跑之前**写定，四条预测 + 两条决策规则）

---

## 1. 为什么做这件事（缺口在哪）

R24–R33 判 $\mu(I)$ 时用的是一对**来源不同**的估计量，而且从没被审计过：

* 纵坐标 `mu_bulk` = 全胞所有接触的 Love-Weber 应力（接触应力）；
* 横坐标 `I_measured` = `profile_r31.compare_to_nominal` 给的 $\dot\gamma_{meas} d/\sqrt{p/ho}$，
  其中 $p$ 是 **round-31 剖面仪的「中心带接触压强」**（`p_contact_total`）。

这两个是不同的观测量。稳态均匀剪切**必须满足动量平衡**，所以 ``p_wall``、体维里 ``p_bulk``、
剖面 ``p_contact_total`` 应当一致；若不一致，这个不一致就是**估计量散度**。而尺子是线性的
（$\mu=\mu_{min}+bI$，$b=1$），几十个百分点的压强误差就搬到 $\mu_{ref}$ 上 $0.01$–$0.03$，
正好是 R30–R33 想归因给物理（过软、无滚动阻力、薄层分支、$b$ 太扁）的那点残差的量级。

于是本轮问一个可判定的问题（预登记）：**换一个同样合法的估计量配对，判定的误差会不会跨过
10 % 的验收带？** 会 → 单配对的 PASS/FAIL 在锚点容差下**不可辨识**，诚实结论是「带散度的
WITHHELD」，不是 FAIL。

本轮**不加物理、不改装置、不动尺子、不碰任何归档 JSON**：只把所有合法观测通道列出来，逐对判。

## 2. 怎么做的

| 步骤 | 内容 |
|---|---|
| 装置 | `plane_shear_2d.RoughWallShearCell`（R33 修好壁面平动后的 `wall_translate=True`） |
| 协议 | R31/R33 短协议：$n=310$，$W=0.24$，$gap=0.14$，settle 20，窗 0.6 应变（**诊断，非锚点证据**） |
| 点 | $I_{target}\in\{0.05\times4\text{ seeds},\;0.10\times2\text{ seeds}\}$ —— 预登记，6 点 |
| 通道 | `wall`（壁面牵引）、`bulk`（Love-Weber）、`profile_contact`、`profile_total`（剖面接触/含动能） |
| 配对 | 4×4 = 16 种 $(\mu\text{通道}, p\text{通道})$；其中 4 种自配对（同通道），1 种是 R24–R33 实际用的 `(bulk, profile_contact)`（**非对角**，命名为 legacy） |
| 判据 | `reference_2d`（da Cruz 2005 Eq.13-14，出处句子绑定、防篡改）$\mu=0.22+1.00I$、$\nu=0.81-0.30I$，容差 10 % |
| 新工具 | `pairing_audit_r34.py`（跑点 + 台账 + 16 配对 + 表 A–G + 图，零手抄）、`test_pairing_audit_r34.py`（10 测试） |

**协议保真**：`point_I0.05_s0.json` 与 R33 归档的 `smoke_filled_n310_wfixed_I0.05_s0.json`
**逐位相同**（`config_digest` 相等，`mu_bulk`/`phi_gap`/`p_wall_over_P`/`I_measured` 全等，
差 0 ulp），唯一区别是 R34 **保留**了 `profile` 块（R33 的提取器把它丢了）。这一条已入测试
`test_r34_rerun_reproduces_the_archived_r33_point_bit_for_bit`。

## 3. 计算结果

# R34 — stress-estimator pairing audit (generated, do not hand-edit)

pre-registration: `prereg_r34.json` (6 points)

## Table A — the ledger: four normal-stress channels of the same run

| I_target | seed | p_wall/P | p_bulk/P | p_prof_contact/P | p_prof_total/P | kinetic share of p | gamma_meas/gamma_nom |
|---|---|---|---|---|---|---|---|
| 0.050 | 0 | 0.676 | 0.648 | 0.653 | 0.658 | 0.007 | 0.631 |
| 0.050 | 1 | 0.862 | 0.853 | 0.853 | 0.859 | 0.007 | 0.555 |
| 0.050 | 2 | 0.343 | 0.353 | 0.357 | 0.362 | 0.012 | 0.260 |
| 0.050 | 3 | 0.678 | 0.689 | 0.692 | 0.696 | 0.007 | 0.562 |
| 0.100 | 0 | 0.620 | 0.544 | 0.545 | 0.553 | 0.015 | 0.171 |
| 0.100 | 1 | 0.590 | 0.601 | 0.604 | 0.614 | 0.016 | 0.147 |

## Table B — the four self-consistent pairings vs the 2-D ruler

| I_target | seed | pairing | I | mu | mu_ref(I) | err % | in 10 % band? |
|---|---|---|---|---|---|---|---|
| 0.050 | 0 | (wall,wall) | 0.0424 | 0.290 | 0.262 | +10.4 | no |
| 0.050 | 0 | (bulk,bulk) | 0.0433 | 0.307 | 0.263 | +16.4 | no |
| 0.050 | 0 | (profile_contact,profile_contact) | 0.0431 | 0.307 | 0.263 | +16.5 | no |
| 0.050 | 0 | (profile_total,profile_total) | 0.0429 | 0.304 | 0.263 | +15.8 | no |
| 0.050 | 1 | (wall,wall) | 0.0331 | 0.358 | 0.253 | +41.5 | no |
| 0.050 | 1 | (bulk,bulk) | 0.0332 | 0.355 | 0.253 | +40.0 | no |
| 0.050 | 1 | (profile_contact,profile_contact) | 0.0332 | 0.355 | 0.253 | +40.0 | no |
| 0.050 | 1 | (profile_total,profile_total) | 0.0331 | 0.352 | 0.253 | +39.0 | no |
| 0.050 | 2 | (wall,wall) | 0.0248 | 0.394 | 0.245 | +60.8 | no |
| 0.050 | 2 | (bulk,bulk) | 0.0244 | 0.346 | 0.244 | +41.4 | no |
| 0.050 | 2 | (profile_contact,profile_contact) | 0.0243 | 0.346 | 0.244 | +41.5 | no |
| 0.050 | 2 | (profile_total,profile_total) | 0.0241 | 0.339 | 0.244 | +38.7 | no |
| 0.050 | 3 | (wall,wall) | 0.0377 | 0.323 | 0.258 | +25.2 | no |
| 0.050 | 3 | (bulk,bulk) | 0.0374 | 0.310 | 0.257 | +20.4 | no |
| 0.050 | 3 | (profile_contact,profile_contact) | 0.0373 | 0.310 | 0.257 | +20.5 | no |
| 0.050 | 3 | (profile_total,profile_total) | 0.0372 | 0.308 | 0.257 | +19.6 | no |
| 0.100 | 0 | (wall,wall) | 0.0236 | 0.316 | 0.244 | +29.9 | no |
| 0.100 | 0 | (bulk,bulk) | 0.0252 | 0.388 | 0.245 | +58.1 | no |
| 0.100 | 0 | (profile_contact,profile_contact) | 0.0252 | 0.388 | 0.245 | +58.1 | no |
| 0.100 | 0 | (profile_total,profile_total) | 0.0250 | 0.378 | 0.245 | +54.4 | no |
| 0.100 | 1 | (wall,wall) | 0.0208 | 0.405 | 0.241 | +68.0 | no |
| 0.100 | 1 | (bulk,bulk) | 0.0206 | 0.364 | 0.241 | +51.1 | no |
| 0.100 | 1 | (profile_contact,profile_contact) | 0.0206 | 0.364 | 0.241 | +51.1 | no |
| 0.100 | 1 | (profile_total,profile_total) | 0.0204 | 0.356 | 0.240 | +48.1 | no |

## Table C — the legacy pairing (rounds 24-33) vs its off-diagonal neighbours

| I_target | seed | pairing | I | mu | err % |
|---|---|---|---|---|---|
| 0.050 | 0 | (wall,profile_contact) | 0.0431 | 0.290 | +10.1 |
| 0.050 | 0 | (bulk,wall) | 0.0424 | 0.307 | +16.8 |
| 0.050 | 0 | (bulk,bulk) | 0.0433 | 0.307 | +16.4 |
| 0.050 | 0 | (bulk,profile_contact) **legacy** | 0.0431 | 0.307 | +16.5 |
| 0.050 | 0 | (bulk,profile_total) | 0.0429 | 0.307 | +16.6 |
| 0.050 | 0 | (profile_contact,profile_contact) | 0.0431 | 0.307 | +16.5 |
| 0.050 | 0 | (profile_total,profile_contact) | 0.0431 | 0.304 | +15.7 |
| 0.050 | 1 | (wall,profile_contact) | 0.0332 | 0.358 | +41.4 |
| 0.050 | 1 | (bulk,wall) | 0.0331 | 0.355 | +40.1 |
| 0.050 | 1 | (bulk,bulk) | 0.0332 | 0.355 | +40.0 |
| 0.050 | 1 | (bulk,profile_contact) **legacy** | 0.0332 | 0.355 | +40.0 |
| 0.050 | 1 | (bulk,profile_total) | 0.0331 | 0.355 | +40.1 |
| 0.050 | 1 | (profile_contact,profile_contact) | 0.0332 | 0.355 | +40.0 |
| 0.050 | 1 | (profile_total,profile_contact) | 0.0332 | 0.352 | +39.0 |
| 0.050 | 2 | (wall,profile_contact) | 0.0243 | 0.394 | +61.2 |
| 0.050 | 2 | (bulk,wall) | 0.0248 | 0.346 | +41.2 |
| 0.050 | 2 | (bulk,bulk) | 0.0244 | 0.346 | +41.4 |
| 0.050 | 2 | (bulk,profile_contact) **legacy** | 0.0243 | 0.346 | +41.5 |
| 0.050 | 2 | (bulk,profile_total) | 0.0241 | 0.346 | +41.6 |
| 0.050 | 2 | (profile_contact,profile_contact) | 0.0243 | 0.346 | +41.5 |
| 0.050 | 2 | (profile_total,profile_contact) | 0.0243 | 0.339 | +38.6 |
| 0.050 | 3 | (wall,profile_contact) | 0.0373 | 0.323 | +25.4 |
| 0.050 | 3 | (bulk,wall) | 0.0377 | 0.310 | +20.3 |
| 0.050 | 3 | (bulk,bulk) | 0.0374 | 0.310 | +20.4 |
| 0.050 | 3 | (bulk,profile_contact) **legacy** | 0.0373 | 0.310 | +20.5 |
| 0.050 | 3 | (bulk,profile_total) | 0.0372 | 0.310 | +20.5 |
| 0.050 | 3 | (profile_contact,profile_contact) | 0.0373 | 0.310 | +20.5 |
| 0.050 | 3 | (profile_total,profile_contact) | 0.0373 | 0.308 | +19.6 |
| 0.100 | 0 | (wall,profile_contact) | 0.0252 | 0.316 | +29.0 |
| 0.100 | 0 | (bulk,wall) | 0.0236 | 0.388 | +59.1 |
| 0.100 | 0 | (bulk,bulk) | 0.0252 | 0.388 | +58.1 |
| 0.100 | 0 | (bulk,profile_contact) **legacy** | 0.0252 | 0.388 | +58.1 |
| 0.100 | 0 | (bulk,profile_total) | 0.0250 | 0.388 | +58.2 |
| 0.100 | 0 | (profile_contact,profile_contact) | 0.0252 | 0.388 | +58.1 |
| 0.100 | 0 | (profile_total,profile_contact) | 0.0252 | 0.378 | +54.2 |
| 0.100 | 1 | (wall,profile_contact) | 0.0206 | 0.405 | +68.1 |
| 0.100 | 1 | (bulk,wall) | 0.0208 | 0.364 | +51.0 |
| 0.100 | 1 | (bulk,bulk) | 0.0206 | 0.364 | +51.1 |
| 0.100 | 1 | (bulk,profile_contact) **legacy** | 0.0206 | 0.364 | +51.1 |
| 0.100 | 1 | (bulk,profile_total) | 0.0204 | 0.364 | +51.2 |
| 0.100 | 1 | (profile_contact,profile_contact) | 0.0206 | 0.364 | +51.1 |
| 0.100 | 1 | (profile_total,profile_contact) | 0.0206 | 0.356 | +48.0 |

## Table E — is the device in steady, homogeneous state? (per run)

| I_target | seed | gamma_meas/gamma_nom | profile r^2 | p_wall/p_bulk - 1 | mu_wall/mu_bulk - 1 | mu_bulk (mean of ratios) vs ratio of means |
|---|---|---|---|---|---|---|
| 0.050 | 0 | 0.631 | 0.979 | +4.4 | -5.5 | 0.3125 vs 0.3066 |
| 0.050 | 1 | 0.555 | 0.862 | +1.0 | +1.0 | 0.3538 vs 0.3546 |
| 0.050 | 2 | 0.260 | 0.970 | -2.9 | +13.9 | 0.3224 vs 0.3456 |
| 0.050 | 3 | 0.562 | 0.993 | -1.7 | +4.1 | 0.3154 vs 0.3100 |
| 0.100 | 0 | 0.171 | 0.586 | +14.0 | -18.4 | 0.4052 vs 0.3876 |
| 0.100 | 1 | 0.147 | 0.673 | -1.9 | +11.3 | 0.3682 vs 0.3636 |

## Table F — ensemble mean +/- sd of the judged mu error per channel

| I_target | channel | n seeds | mean err % | sd err % | half-range % |
|---|---|---|---|---|---|
| 0.050 | bulk | 4 | +29.6 | 12.973 | 12.486 |
| 0.050 | profile_contact | 4 | +29.6 | 12.950 | 12.471 |
| 0.050 | profile_total | 4 | +28.3 | 12.322 | 11.627 |
| 0.050 | wall | 4 | +34.5 | 21.665 | 25.211 |
| 0.100 | bulk | 2 | +54.6 | 4.911 | 3.473 |
| 0.100 | profile_contact | 2 | +54.6 | 4.922 | 3.480 |
| 0.100 | profile_total | 2 | +51.2 | 4.392 | 3.106 |
| 0.100 | wall | 2 | +48.9 | 26.936 | 19.046 |

## Table G — can the ruler slope be re-measured from this ensemble?

weighted OLS ``mu = a0 + a1 I`` over all runs (bulk channel, achieved ``I``);
the ruler is ``mu = 0.22 + 1.00 I``.

| quantity | value | ruler |
|---|---|---|
| n points | 6 | - |
| intercept a0 | 0.435 +/- 0.033 | 0.22 |
| slope a1 | -2.939 +/- 1.049 | 1.00 |
| achieved I range | 0.0206 .. 0.0433 | 0.001 .. 0.3 |
| z(slope - 1.00) | -3.75 | 0 = consistent with the ruler |

*caveat (fixed before the runs)*: this pooled fit is **not** a measurement of the ruler slope `b`. The runs do not form a controlled `I` sweep -- the `I_target = 0.1` points never reach their nominal rate (see Table E) and land at a *lower* achieved `I` with a *higher* `mu`, so pooling them is confounded. A non-zero `z` here is evidence that the achieved `I` is not the state variable while the cell is not rate-controlled, i.e. a device-state warning, not a rheological result.


## Table D — pre-registered predictions (fixed before the runs)

| prediction | observed | held? |
|---|---|---|
| P1 kinetic share > 0.15 | max 0.016 | no |
| P2 pressure spread > 1.15 | max 1.140 | no |
| P3 error spread > 10 pp | max 28.2 pp | yes |
| P4 falsifier: spread < 2 pp | min 2.4 pp | no |

P1/P2/P3/P4 are the pre-registered statements verbatim from `prereg_r34.json`; the observed columns are computed from the archived run JSON, not typed.


### 3.1 结论一：预登记的「动能应力配对」假设被**否证**（P1/P2 fail）

剖面的**动能（颗粒温度）部分只占法向应力的 0.7 %–1.6 %**（表 A 最后一列），所以
「接触应力 vs 全应力」根本不是问题；四个压强通道在 $I_{target}=0.05$ 时彼此相差 −3 %…+4 %
（最大/最小 = 1.02–1.06，远小于预登记的 1.15 门槛）。

⇒ **R34 自己提的 harness 缺陷假设（动能应力导致 `I` 与 `mu` 定义不一致）不成立**，如实入账，
不做后续粉饰。P3 表面上「成立」（误差极差最大 28 pp），但那只出现在 $I_{target}=0.10$ 的两点
——那里胞体**根本没进稳态**（见 3.2），所以不能当成估计量散度的证据；在 $I_{target}=0.05$
的 4 个种子上，通道散度只有 2.4–6.1 pp。

### 3.2 结论二：真正的大项是**装置状态**，不是估计量——单点 $p_{wall}/P$ 只在 0.34…0.86

表 A 第一列：同协议、同 $I_{target}=0.05$ 的 4 个种子，常压伺服达到的围压是
**0.343 / 0.676 / 0.678 / 0.862 × P**——**没有一个达到设定值**。表 E 更直接：

* $I_{target}=0.10$ 的两点，闭合率 $\dot\gamma_{meas}/\dot\gamma_{nom}$ 只有 **0.171 / 0.147**，
  剖面 $r^2$ 掉到 0.586 / 0.673，壁-体压强失配 14 % —— 胞体在高率段**根本没被剪切到名义率**，
  它落到 $I\approx0.021$–$0.025$，即**名义 $I$ 的 1/4**。
* $I_{target}=0.05$ 的种子 2 也掉到 $p/P=0.343$、闭合率 0.260。

⇒ 锚点要求的 $I\le0.3$ 上端，本装置在现配置下**不是率控制的**；这些点不能用来读 $\mu(I)$。
这一条给 cj4610 战役（$I_{target}\in\{0.03,0.05,0.1,0.2\}$，1 seed/点）一个硬警告：
**它有一半的分片会落在「名义 I ≠ 达成 I」的区域**，且 1 seed/点分辨不了下面的散度。

### 3.3 结论三：实现散度（种子间）$\sigma\approx13$ pp，**大于** 10 % 容差

表 F（bulk 通道，$I_{target}=0.05$，4 seeds）：**+29.6 ± 13.0 pp**（半极差 12.5 pp）；
壁面通道更差（+34.5 ± 21.7 pp）。也就是说：

* R33 报的 +18.8 % 是这**整个分布的下沿**（单 seed 样本），不是「残差」的点估计；
* 「残差 20 %」的**量级**没有被确立，被确立的是「均值 +30 %、宽度 ±13 pp，且宽度主要来自
  伺服/固结是否成功（$p/P$ 0.34…0.86）」，而不是物理；
* 反过来，FAIL 在**均值**上是稳健的（4 seeds 全部 $>+16$ pp，无一进入 10 % 带）——所以这一条
  不能读成「其实 PASS 了」。

$I_{target}=0.10$ 两点 +54.6 ± 4.9 pp（仅 2 seeds，宽度不可靠，但都不在带内）。

### 3.4 结论四：尺子的斜率**不可从本数据复测**（表 G 是混杂探测器，不是 $b$ 的测量）

把 6 点合起来拟合 $\mu=a_0+a_1I$ 得 $a_1=-2.9\pm1.1$（$z=-3.75$ 相对尺子的 $b=1$）。这**不是**
「$b$ 测得为负」——合并没有意义：$I_{target}=0.10$ 的点达成 $I$ **更低**却 $\mu$ **更高**
（因为它们是准静态局域化态，不是惯性态）。所以 R30 的「$b=0.279$，只有论文 1/3」结论
**在本数据上也同样不可辨识**；要测 $b$ 必须先有一段**率受控**的 $I$ 区间。


## 4. 注意点 / 局限

1. **本轮的 6 个点全部是诊断，不是锚点证据**：$n=310$、settle 20、窗 0.6 应变，稳态未达
   （表 E 的闭合率和 $r^2$ 自己说了）。本机算力门限下只能做这个尺度；满尺度战役 cj4610
   递交后仍在排队（递交时前方 96 个作业，本单收尾仍未回），故「修复后锚点是否落进 10 %」
   = **NOT_RUN**，不得记为 PASS。
2. **$p_{wall}/P<0.9$ 的 4 个点全部来自同一个协议**，因此 3.3 的散度是「协议散度」，不能直接
   当作装置本征散度：需要满尺度 + 伺服门的对照才能分离。诚实地说，本轮只能给出
   「协议与装置状态项 ≥13 pp」这个下界。
3. **2-D 装置没有伺服有效性门**。3-D 路径 R5 已加了 `pressure_control_tol`（持续窗口内
   伺服误差进容差才计测量），2-D 的 `PlaneShear2DCase` **没有**这个字段——所以 R24–R33 的
   2-D 点没有一个是「伺服达成」意义上合格的。这是本轮发现的可直接修的结构缺口（见 §5）。
4. 表 C 里 `(wall, profile_contact)` 这只非对角组合在 $I_t=0.05$ s0 给出 **+10.1 %**，而同一
   次运行的 legacy 配对给 +16.5 %——即**在 10 % 边界上，选哪个通道就能决定 PASS/FAIL**。
   这不足以判 PASS（其它 5 点同组合都远超 10 %），但它说明「贴边通过」不可信。
5. 拟合只对 `pairing_audit_r34.py` 与归档 JSON 负责；`provenance_archive` 与 `reference_2d`
   未改一字（`test_r30_2d_reference.py` 仍绿）。

## 5. 未完成 / 建议（honest）

* **未完成**：满尺度（$n=1220$，$W=0.40$，settle 60，窗 1.0 应变）的 4-seed 集成 + 伺服门；
  cj4610 回执到手后应立即用它复核本轮的散度是否缩到 <10 %。
* **建议新增总账条目**（本单不改锚点）：
  - `CASE-DEM-PLANE-SHEAR-MUIDI-SERVO-GATE-2D`｜2-D 装置补上 R5 那样的
    `pressure_control_tol` + 持续窗口判据，验收：12 个越界探针中「$p/P<0.9$ 却计入测量」= 0；
  - `CASE-DEM-PLANE-SHEAR-MUIDI-ENSEMBLE`｜每点 $\ge4$ seeds，验收：判定的 $\mu$ 误差
    半极差 < 10 pp 才允许出 PASS/FAIL，否则 WITHHELD；
  - `CASE-DEM-PLANE-SHEAR-MUIDI-HIGH-I-FEASIBILITY`｜$I\ge0.1$ 的率控制可行性（改 $P$、改 gap、
    或改走 3-D 路径），验收：闭合率 $\dot\gamma_{meas}/\dot\gamma_{nom}\ge0.8$ 且 $r^2\ge0.95$。

## 6. 误差判定（逐项，是否 <3 % / <10 %）

| 关键量 | 基准 | 结果 | <3 %? | <10 %? |
|---|---|---|---|---|
| 协议保真（vs R33 归档点） | 逐位相同 | `config_digest` 相等、4 个标量全等 | **是（0 ulp）** | 是 |
| legacy 配对复现（vs R33 表 B 的 +18.8 %） | R33 已发表数 | +18.8 %（`test_legacy_pairing...`） | 是（±0.15 pp） | 是 |
| 动能应力占比 | 预登记 >15 % | 0.7–1.6 % | 否（假设被否证） | — |
| 压强通道一致度 | 预登记 1.15 | 1.02–1.14（$I_t$=0.05: ≤1.06） | 否（但 <5 % 已足够） | 是 |
| $\mu$ vs 尺子（bulk 通道均值，4 seeds） | $\mu=0.22+1.00I$ | **+29.6 %**（±13.0 pp） | 否 | **否** |
| $\phi$ vs 尺子（同点） | $\nu=0.81-0.30I$ | 0.7997–0.8010 vs 0.797–0.807（表 A/表 B 的 $\phi$ 列） | 部分 | 是 |

**判定**：$\mu$ 门在本轮诊断协议下 **FAIL（均值）**；但由于实现散度 ±13 pp 大于容差，
「残差 20 %」这一**量级**结论记为 **WITHHELD**；$I\ge0.1$ 的锚点区间记为 **NOT_SUPPORTED**
（本轮无率受控点）。$\phi$ 门在同点 10 % 内，仍按 R32 的薄/厚层歧义保持 WITHHELD。

## 7. 复现

```bash
cd benchmarks/plane_shear_mu_i
PYTHONPATH=../../src python pairing_audit_r34.py --I 0.05 --seed 0     # 1 点 ~65 s
PYTHONPATH=../../src python pairing_audit_r34.py --all                  # 预登记 3 点
PYTHONPATH=../../src python pairing_audit_r34.py --judge-only           # 只重判归档
PYTHONPATH=../../src python -m pytest test_pairing_audit_r34.py -q      # 10 passed
```

工件：`runs/pairing_r34/{point_*.json, tables_r34.md, judged_r34.json, fig_pairing_r34.png}`。

## 8. 图

`runs/pairing_r34/fig_pairing_r34.png`（三面板，程序化生成）
* (a) $\mu$ 对**达成** $I$：6 点（bulk/wall 两种标记）叠在 da Cruz 尺子与 ±10 % 带上——
  全部点在带上方，且 4 个种子的水平位置相差近 2 倍（达成 $I$ = 0.024–0.043）。
* (b) 每个通道的判定误差：通道间几乎重合，种子间大幅分离 —— 视觉上就是「散度不是通道来的」。
* (c) 同一次运行的四个压强通道 / 设定值：$I_t=0.10$ 的两点贴近 0.55–0.62，伺服没到设定。
