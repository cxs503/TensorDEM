# XVAL-DEM-ROLLING-FRICTION-AS-SHAPE-SURROGATE — 滚动摩擦作为形状代理的**标定面**交差核验（m17903）

* **判定：honest FAIL（3/5 判据不过 15% 门）**——判据 C2（文献对应点）、C3（灵敏度）、C4（μ_r=0 端可反演）不过；
  通过项：C1 单调性（Spearman=1.000）、C5 本协议内样本外迁移（最差 10.65 % < 15 %）。
* **锚点**：`feature_ledger` slug `XVAL-DEM-ROLLING-FRICTION-AS-SHAPE-SURROGATE`，
  标题「literature/Wensrich-Katterfeld-2012 的 rolling-friction-as-shape-surrogate 做法」，验收 <15 %。
* **新增能力**（此前不可得）：滚动摩擦**标定面** θ(μ_r) 的生成、饱和拟合、单调反演、样本外迁移打分
  → `src/tensordem/protocols/sandpile_calibration.py`（先验门只对 `mu_s/mu_r` 开洞，且标定点**不可能**被读成 baseline）。
* 工件（全部程序化生成，零手抄）：
  `artifacts/xval_rolling_shape_surrogate/result.json`、`points/*.json`（8 点含完整协议记录）、
  `fig_shape_surrogate_calibration.png`。
* 复现：
  `PYTHONPATH=src python benchmarks/xval_rolling_shape_surrogate/run_sweep.py point --mu-r 0.3 --tag ref_mu0p30`
  → `PYTHONPATH=src python benchmarks/xval_rolling_shape_surrogate/run_sweep.py aggregate`
  → `PYTHONPATH=src python benchmarks/xval_rolling_shape_surrogate/figures.py`

## ① 案例介绍（物理背景与基准出处）

Wensrich & Katterfeld (2012, *Powder Technol.* **217**:409–417) 的做法：**用球 + 单一滚动摩擦系数 μ_r 替代非球形颗粒**，
μ_r **不对由单接触几何推导，而由宏观观测量标定**。本仓登记在案的那条文献对应（`docs/GAP_DEM_CLUMP_ROUND2.md` 第 32 行，
sha256 `4767513195d2…`，运行时由 `parse_wk2012_reference` 正则解析，**不手抄**）：

> 球的 repose 随 μ_r 单调升（μ_r 0→0.5 对应 21°→32°）→ 灵敏度 22 °/单位 μ_r

前两轮（m17638/m17676）只跑了**单个先验点**（μ_r=0.05）去撞 22–25° 带，从未产出可反演的标定面；
且冻结协议 `protocols/sandpile.py` 的先验门**禁止**标定（手工调 μ_r 会被 `assert_priors` 直接拒绝前一步）——
所以"文献做法"在本仓其实**做不了**。本单的工作就是把这一条补上并交差核验它是否成立。

**协议条件**（参考条件，8 个点共用）：N=1600 球，r=5 mm，ρ=2500 kg/m³，μ_s=0.5（先验），
pour_rate=890 grains/s（喷嘴出口速度 v0=0.30 m/s，非暴力抛撒），落高 5 mm，静置 0.15 s，dt=1.038e-4 s（临界步长的 0.1），
seed=11。堆高 h/R≈10–15（≥3 层判有效），侧壁拟合 r²≥0.987。

## ② 怎么做的

1. **协议开洞（最小改动）**：`SandpileConfig` 新增 `calibration_sweep` 字段；`assert_priors` 仅对 `mu_s/mu_r` 豁免，
   且 `calibration_sweep=True` 时 `control` 只能是 `baseline`（负控制不可借道）。
   标定点记录强制带 `kind="calibration_sweep"`、`calibration_disabled=False`、`prior_guard="bypassed:calibration_sweep"`、
   `priors_match_file=False` → **结构上不可能被误读成受门保护的 baseline**。
2. **标定面**：μ_r ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5} 各跑一次完整冻结协议（同一测量算子 `repose_angle_from_profile`），
   得到 θ(μ_r)；有效点判据先验声明：r²≥0.9 且 h/R≥3，不合格点**报告但不进拟合**。
3. **代理当作预测量来考**：饱和拟合 θ = θ∞ − A·exp(−μ_r/μ_c)，单调反演文献角 → μ_r\*，并在**未参与拟合的条件**
   （seed=23、N=800）上做样本外打分（唯一的非循环精度口径）。
4. **工具链**：numpy/torch CPU，单位核；8 个点合计 ~19 min 墙钟（每点 120–159 s），`result.json` 内嵌每点 sha256 与 git rev。

## ③ 计算结果（数字表，程序化注入）

**标定面（参考条件，有效点）**

| μ_r | θ_DEM [°] | stderr [°] | r² | h/R | φ_bulk |
|---|---|---|---|---|---|
| 0.05 | 21.06 | 1.27 | 0.987 | 10.0 | 0.859 |
| 0.10 | 21.65 | 1.21 | 0.989 | 10.7 | 0.766 |
| 0.20 | 30.46 | 0.86 | 0.996 | 13.4 | 0.690 |
| 0.30 | 31.81 | 0.98 | 0.989 | 14.1 | 0.672 |
| 0.50 | 36.84 | 1.27 | 0.987 | 14.8 | 0.668 |

**被拒点（不合格，如实报告）**：μ_r=0 → θ=0.067°，r²=0.161，h/R=4.5 —— 无滚动阻力的球床**不成堆**（平面铺层），
该角是几何假象，**不得**当作"文献 μ_r=0 ⇒ 21°"的测量值；它把拟合降为 5 点。

**饱和拟合**：θ∞=41.04°，A=24.94°，μ_c=0.2811，RMSE=1.24°（5 点）。

**文献对应点逐项（门 <15 %）**

| 文献点 | θ_ref | θ_DEM | 来源 | 相对误差 | <15 % |
|---|---|---|---|---|---|
| μ_r=0 → 21° | 21.0 | 16.10 | 拟合外推 | **23.3 %** | ✘ |
| μ_r=0.5 → 32° | 32.0 | 36.84 | 实测 | **15.14 %** | ✘（临界超门） |

**灵敏度**：文献 22.0 °/单位 μ_r；实测 41.49 → **+88.6 %** ✘。
即本协议的 repose 对 μ_r **比文献对应陡 1.9 倍**。

**反演**：μ_r=32° → μ_r\*=0.3075（bracket 0.3–0.5）；μ_r=21° **不可达**（实测最小 21.06°，μ_r=0.05）✘。

**样本外迁移（唯一非循环精度）**：拟合只在 {0.05,0.1,0.2,0.3,0.5}×seed11×N1600 上做，预测：
μ_r=0.3, N=800 → 实测 29.34°，预测 32.46°（−10.65 %）；μ_r=0.3, seed=23 → 实测 31.79°，预测 32.46°（−2.1 %）。
最差 **10.65 % < 15 %** ✔（C5）。

**协议散布**：μ_r=0.3 处 (N=1600,seed11)=31.81°/（N=800）=29.34°/（seed23）=31.79° → 极差 2.47°（8.0 %）。

**判据逐条**：C1 单调 ✔（Spearman 1.000）｜C2 文献点 ✘｜C3 灵敏度 ✘｜C4 μ_r=0 端可反演 ✘｜C5 迁移 ✔ →
`all_checks_pass=false`。**是否 <3 %：否**（唯一对得上的量是 C5 的 10.65 %，其余量级 15–89 %）；不粉饰。

## ④ 结果图

`artifacts/xval_rolling_shape_surrogate/fig_shape_surrogate_calibration.png`（三联，matplotlib，数据全部读自 result.json）：
(a) 标定面 + 饱和拟合 + 文献 0→0.5 对应线 + 样本外点；(b) 两个文献锚点的 ref-vs-got 柱状与逐项误差百分比；
(c) 反演 μ_r\* 与不可达角的标注。

## ⑤ 注意点（坑与适用域）

* **先验门是真门**：没有 `calibration_sweep` 标志，`mu_r=0.3` 会在第一步之前被拒；本次开洞只放 `mu_s/mu_r` 两个旋钮，
  且标定点与 baseline 在记录层**不可混淆**（`prior_guard`/`kind`/`priors_match_file` 三重标记，tests 覆盖）。
* **μ_r=0 端是物理退化，不是测量噪声**：无滚动阻力的球在重力下滚散为铺层，repose 角**没有定义**；
  这也说明"文献 μ_r=0 ⇒ 21°"这条对应在**本参数集**下根本不可实现 ⇒ 该文献对应**不是材料无关的**。
* **φ_bulk 在浅堆不可信**（μ_r=0.05 给出 0.859 > 球体上限 0.74），沉积体积估计在薄层失效——不影响 θ（θ 走侧壁线性拟合，
  r²≥0.987），但报告中不得引用 φ。
* **比较口径**：文献数值来自仓内文献表（原文 PDF 不在本环境，**WITHHELD**），且是**另一套接触参数/协议**下的对应；
  本单检验的是"该对应是否可移植"，答案是否。任何"与原文表一致"的结论都不成立。
* **单点成本**：N=1600 每点 120–159 s；N=800 与 seed 变体同量级。9 并发在 load≈100 的 128 核机上可行。

## ⑥ 结论与后续

滚动摩擦作为形状代理，在本仓**可以作为标定面被生成、被反演、并在本协议内做样本外预测（10.65 %）**；
但它**不能**复现文献登记的那条 μ_r→repose 对应（端点 23.3 %/15.14 %，灵敏度 +88.6 %），
且文献下端点在本参数集**不可达**。因此本轮结论是 **honest FAIL**，且失败点定位到
"文献对应不是材料无关的 + 无滚动阻力时球床不成堆"这两条**物理**原因，而非数值实现缺陷。

后续（建议，不自行改锚点）：
1. 把「标定面 + 样本外迁移」登记为 benchmark case（`tensor_benchmark_cases`，anchor_slug 同上，metric=transfer_worst_rel_error，tolerance=15 %）；
2. 若要复原文献对应，需要**同一套接触参数**（μ_s、k_n、滚动律）下的文献数值，当前只能 WITHHELD；
3. 形状端本身仍缺：用同一个 2-D 多边形/3-D 多面体引擎测 `a/b` 形状族的 repose，才能把"代理 vs 真形状"直接对撞
   （`docs/GAP_DEM_CLUMP_ROUND2.md` 的 C3 仍 FAIL，侧墙几何是已知阻塞点）。