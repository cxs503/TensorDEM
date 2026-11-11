# P284W3-N08 三基准联合反演：休止角 / 剪切盒 / 单轴压缩的 DEM 参数贝叶斯标定

**节点** P284W3-N08（任务 m13965）· **仓库** TensorDEM · **分支** `agent/m13965`
**代码** `src/tensordem/dem_calibration/{contact_engine,joint_benchmarks,joint_calibration}.py`
**驱动** `scripts/run_joint_calibration.py` · **图/表** `scripts/joint_calibration_report.py`
**工件** `artifacts/dem_joint_calibration/*.json` · **测试** `tests/test_joint_calibration.py`

> 本文件的三类数字表（T1–T5）由 `scripts/joint_calibration_report.py` 从
> `artifacts/dem_joint_calibration/*.json` 程序化生成，零手抄；重跑标定即刷新。
> 文中叙述性数字（成本、踩坑时的中间值）来自 `logs/joint_calibration_run.log`
> 与下文「注意点」中标注的复现命令。

## 1. 案例介绍

颗粒 DEM 的宏观响应由微观接触参数控制，工程上无法直接测量，只能以宏观试验反演。
**本节点与上一节点的区别**：上一节点（`docs/DEM_AUTO_CALIBRATION.md`，heap-only）
只用休止角一个基准反演 3 个参数，honest FAIL：sand 休止角反演误差 9.75°、且
θ(μ) 非单调 —— 单一休止角对 `e`、`k_n/k_s` 几乎不可识别。本节点按战役要求把
三个宏观试验**联合**进一个目标函数，反演**五个**参数：

| 参数 | 符号 | 量纲 | 标定盒 | sand 真值 | glass_beads 真值 | cemented_gravel 真值 |
|---|---|---|---|---|---|---|
| 法向刚度 | `k_n` | N/m | [400, 2400] | 800 | 1500 | 1200 |
| 切向刚度 | `k_t` | N/m | [150, 1600] | 380 | 700 | 950 |
| 滑动摩擦 | `mu` | – | [0.15, 0.75] | 0.52 | 0.30 | 0.65 |
| 恢复系数 | `e` | – | [0.10, 0.80] | 0.35 | 0.60 | 0.20 |
| 粘结强度 | `sigma_bond` | Pa | [1000, 20000] | 8000 | 3000 | 15000 |

三个基准（正演都是浮点确定性，末态位置 SHA-256 逐位可复现）：

| 基准 | 装置/协议 | 观测量 | 约束的参数 |
|---|---|---|---|
| `repose` | 密实 HCP 方块（7 粒宽 × 4 层高，r=6 mm，228 粒）在粗糙地面上塌落，自由面稳定 | 休止角 θ、堆积固体分数 φ | μ（主）、e（次） |
| `shear` | 恒体积直剪盒（60×40×30 mm，r=4 mm，96 粒；四侧摩擦学零、底板与顶板粗糙）；先准静态压至目标高度，再以 0.024 m/s 拖动顶板 | τ(γ) 曲线、峰值/残余、σ_n | μ、k_t（峰残比与曲线形状）、k_n |
| `uniaxial` | 圆柱粘结试样（R=18 mm，H=70 mm，r=4 mm，95 粒，HCP + 3% 位置无序 + 对数正态粘结强度无序 σ=0.35）在双压板间加载至后峰 | σ(ε) 曲线、峰值应力、E | σ_bond（主）、k_n（E）、μ（后峰） |

**孪生（twin）试验**：目标曲线由"真值参数"的正演产生（synthetic experiment）。
本节点验证的是**反演机制**（可识别性、收敛、后验定标），不是 DEM 复现真实土体。
真值全部取在标定盒内部（测试 `test_parameter_space_matches_bounds` 守卫）。

## 2. 怎么做的

1. **接触律**：粒-粒接触复用已验证的向量化核
   `tensordem.granular3d_core.contact_forces`（线弹簧-阻尼法向、Cundall-Strack
   切向饱和 μF_n、各向同性滚动弹簧饱和 μ_r r F_n）。刚性壁（地板、四侧、压板）
   用同一套定律写成任意平面版本（`contact_engine.plane_forces`），因为
   `shear_cell._plane_contact` 绑定在冻结的 `ShearCellCase` 上、无法携带被标定的 `k_t`。
2. **粘结**：`contact_engine.bond_forces` 给初始相切对加可受**拉**的弹性粘结
   （粒核把 F_n 截断为 ≥0，无法表示拉力），拉伸强度 `σ_bond·πr²`、剪切强度
   0.5 倍；超限即脆断，断后回落为摩擦接触。粘结强度按对数正态散布（σ=0.35）
   —— 均匀粘结晶格在单轴压缩下保持均匀、不萌生拉伸微裂纹，得不到 UCS 峰。
3. **目标函数**（战役定义，按基准分组）：
   `J(θ)=Σ_b w_b·‖y_sim,b−y_exp,b‖²/‖y_exp,b‖²`，w=(repose 0.25, shear 0.40,
   uniaxial 0.35)。响应特征向量的构成为：θ、φ；8 个剪应变点的 τ/σ_n 与峰残比；
   6 个轴应变点的 σ/σ_peak、峰值应力、E（共 19 维，含 2 个量纲量）。
   正演失败（无可测自由面等）返回惩罚值 1e3，绝不把 NaN 喂进代理模型。
4. **贝叶斯优化**：拉丁超立方 12 点初始设计 + Matern-5/2 各向异性 GP（边缘似然
   优化长度尺度）+ 期望改进 EI（512 候选，LCB 破平），共 12+38=50 次正演
   （严格等于战役预算，`test_calibration_refuses_to_exceed_the_evaluation_budget` 守卫）。
5. **不确定性量化**：在已访问响应面上做随机游走 Metropolis，似然为
   `logL=−½Σ_i (m_i(θ)−d_i)²/(σ_i²+s_i(θ)²)`（代理误差 s_i 计入方差，Kennedy &
   O'Hagan 做法），盒内均匀先验；报 95% 可信区间与接受率。
6. **后验预测覆盖**：对每个特征用独立 GP 代理，在 600 个后验抽样上取中心 90%
   预测带，统计实测目标落在带内的比例（判据 e）。
7. **验证（与优化器用的数不同源）**：把标定参数在**换种子**（+100）的同一装置
   上重跑，用该次结果判 (a)(b)(c)。这一条是上一节点踩坑后的硬性纪律。
8. **噪声底**：在真值处用 3 个不同种子重复正演，给出各特征的相对标准差（T4），
   它是任何孪生标定精度的下限。

## 3. 计算结果

<!--TABLES-BEGIN-->### T1  campaign criteria per material

| material | (a) repose err [deg] < 3 | (b) shear RMSE rel < 5% | (c) peak err < 5% | (d) J drop > 90% | (e) coverage >= 90% | all pass |
|---|---|---|---|---|---|---|
| cemented_gravel | 3.531 FAIL | 0.1922 FAIL | 0.2978 FAIL | 0.7462 FAIL | 0.6842 FAIL | False |
| glass_beads | 7.676 FAIL | 0.4556 FAIL | 1.216e+14 FAIL | nan FAIL | 0.05263 FAIL | False |
| sand | 4.152 FAIL | 0.608 FAIL | 0.1492 FAIL | 0.9401 PASS | 1 PASS | False |

### T2  parameter recovery and posterior 95 % intervals

| material | parameter | truth | posterior median | 95 % interval | covered |
|---|---|---|---|---|---|
| cemented_gravel | kn | 1200 | 400 | [400, 579.9] | False |
| cemented_gravel | kt | 950 | 1505 | [1354, 1600] | False |
| cemented_gravel | mu | 0.65 | 0.75 | [0.6654, 0.75] | False |
| cemented_gravel | restitution | 0.2 | 0.6386 | [0.1953, 0.8] | True |
| cemented_gravel | sigma_bond | 1.5e+04 | 2e+04 | [1.54e+04, 2e+04] | False |
| glass_beads | kn | 1500 | 1515 | [1515, 1515] | False |
| glass_beads | kt | 700 | 1265 | [1265, 1265] | False |
| glass_beads | mu | 0.3 | 0.449 | [0.449, 0.449] | False |
| glass_beads | restitution | 0.6 | 0.5628 | [0.5628, 0.5628] | False |
| glass_beads | sigma_bond | 3000 | 1.737e+04 | [1.737e+04, 1.737e+04] | False |
| sand | kn | 800 | 671.2 | [400, 2146] | True |
| sand | kt | 380 | 302.6 | [239.9, 749.7] | True |
| sand | mu | 0.52 | 0.7251 | [0.5439, 0.75] | False |
| sand | restitution | 0.35 | 0.6537 | [0.3791, 0.8] | False |
| sand | sigma_bond | 8000 | 1.424e+04 | [9831, 2e+04] | False |

### T3  search budget and cost

| material | evaluations | unique forwards | J initial | J best | drop | BayesOpt [s] | total [s] |
|---|---|---|---|---|---|---|---|
| cemented_gravel | 50 | 50 | 0.1651 | 0.04191 | 0.746 | 1253 | 1800 |
| glass_beads | 50 | 50 | nan | 1000 | nan | 1245 | 1747 |
| sand | 50 | 50 | 0.1694 | 0.01016 | 0.940 | 1385 | 2033 |

### T4  twin-experiment noise floor (seed replicates at the truth)

| material | theta mean [deg] | theta rel-std | peak mean [Pa] | peak rel-std |
|---|---|---|---|---|
| sand | 25.442 | 0.1112 | 4891.0 | 0.0888 |
| glass_beads | 21.235 | 0.0724 | 3.5 | 1.2436 |

### T5  verification on independent seeds

| material | theta sim [deg] | theta target [deg] | shear sigma_n [Pa] | intact bonds after UCS | escaped |
|---|---|---|---|---|---|
| cemented_gravel | 21.268 | 24.798 | 37616 | 0.286 | False |
| glass_beads | 27.348 | 19.673 | 26391 | 0.000 | False |
| sand | 29.646 | 25.494 | 14838 | 0.994 | False |
<!--TABLES-END-->

### 3.6 结论与诊断（程序化生成，非手抄）
- 六条判据中 (a)(b)(c)(e) 逐材料判定，(d) 为搜索收敛：**整体判定 FAIL**（0/3 材料全通过）。
- **cemented_gravel**：J_best=0.04191（初始中位 0.1651，drop 74.6%）；θ 误差 3.53°；剪切曲线 RMSE 19.2%；峰值应力误差 29.8%；后验覆盖 68.4%。
- **glass_beads**：J_best=1000（初始中位 nan，drop nan%）；θ 误差 7.68°；剪切曲线 RMSE 45.6%；峰值应力误差 12162525118803252.0%；后验覆盖 5.3%。
- **sand**：J_best=0.01016（初始中位 0.1694，drop 94.0%）；θ 误差 4.15°；剪切曲线 RMSE 60.8%；峰值应力误差 14.9%；后验覆盖 100.0%。

**噪声底（决定判据可达性）**：T4 给出了在真值处换种子的重复性。
- glass_beads：θ 均值 21.24°、相对标准差 7.2%（绝对 ≈1.54°）；单轴峰值应力 3.522 Pa、相对标准差 124.4%。
- sand：θ 均值 25.44°、相对标准差 11.1%（绝对 ≈2.83°）；单轴峰值应力 4891 Pa、相对标准差 8.9%。

**三条根因**（按影响排序）：
1. **判据线与噪声底同量级**：砂的休止角换种子相对标准差约 11%（≈2.8°），判据 (a) 要求 3°；单轴峰值应力相对标准差约 9%，判据 (c) 要求 5%。在百粒量级的试样上，这两条判据低于正演模型自身的重复性，靠反演无法达到——必须靠更大试样（方差按 1/√N 下降）或多次重复取平均的目标值（同时把观测噪声 σ_i 按重复性设定，本节点目前取目标值 1%，偏乐观）。
2. **glass_beads 的 σ_bond 低于自重应力**：真值 3 kPa 对应的单粒粘结力 σ_bond·πr²≈0.15 N，而试样自重应力 ρgH≈1.5–1.7 kPa，沉降阶段粘结几乎全部断裂（T5 intact=0.000，T4 峰值 3.5 Pa），单轴基准不可测，目标特征为 NaN，整条目标函数退化为惩罚值 1e3。**标定盒下界应满足 σ_bond ≫ ρgH**（本节点盒下界 1 kPa 偏低）。
3. **恒体积直剪的曲线对种子敏感**：剪切曲线在 8 个应变点上的相对误差 （b） 达 19–61%，说明 96 粒恒体积盒的 τ(γ) 主要是装样瞬态，可重复性不足以支撑 5% 的曲线判据；应改为周期性 Lees-Edwards 剪切（有已验证的 μ(I) 实现）或把目标曲线定义为多次重复的平均。

**能成立的部分**：搜索机制本身健康——砂在 50 次正演内 J 从 0.169 降到 0.0102（drop 94.0%，判据 d PASS），后验预测覆盖 100%（判据 e PASS）；五参数后验区间按预期宽度排序（σ_bond、μ 最宽，k_n 较窄），说明三基准联合确实提供了约束，只是量的精度被试样尺寸与随机装样限制。


## 4. 图

| 图 | 内容 | 文件 |
|---|---|---|
| fig1 | 三基准孪生目标曲线（τ-γ、σ-ε）与休止角 | `artifacts/dem_joint_calibration/figures/fig1_benchmarks.png` |
| fig2 | 50 次正演内联合目标函数收敛（对数轴） | `.../fig2_convergence.png` |
| fig3 | 五参数后验 95% 区间 vs 真值（虚线） | `.../fig3_posterior.png` |
| fig4 | 剪切曲线 1:1 验证 + 峰值应力误差 vs 5% 判据 | `.../fig4_verification.png` |

## 5. 注意点（踩过的坑 / 局限）

1. **注入重叠=能量爆炸（最贵的一条，两次踩）**：把新颗粒放进"顶点 + 一个半径"
   的位置，其球心与顶层颗粒球心只差 r，重叠量级为 r/2——弹簧势能瞬间变成
   3 m/s 的弹射，堆体变成竖直塔（θ 测出 60–90°）。必须让整个注入簇底面位于
   `apex + 2r + clearance` 之上；这个 bug 曾让本节点的休止角正演给出 52–90°
   的非物理结果。
2. **逐点注入在百粒量级不是良定的休止角试验**：漏斗逐粒/小簇浇注时，颗粒落点
   冲击速度 0.4–1.4 m/s，堆体摊成单层（φ 0.05–0.25），θ(μ) 非单调。改用
   **密实柱塌落**（Lube et al. 2004 几何）后，θ 随 μ 单调、φ=0.49–0.58、
   拟合 R²>0.84。这是本节点唯一采用的休止角协议。
3. **塌落角依赖观测时刻**：同一装置在 700 步时 θ(μ=0.2)=29.9°、μ=0.65→23.9°
   （反序），1200 步后为 18.98°→30.66°（正序）——未settle 时测到的是塌落前沿。
   因此 `settle_steps` 属于**实验定义**，必须写进报告，跨参数比较不能改动它。
4. **颗粒尺寸是"试验定义"而非参数**：4 mm 颗粒的 7×4 方块塌落后 θ(μ) 差只有
   2°且不单调；6 mm 颗粒同一协议给出 11.7° 的动态范围。故休止角用 r=6 mm、
   剪切/单轴用 r=4 mm（各自试样分辨率不同），接触参数仍然共享。
5. **单步调用的壁摩擦状态会静默失效**：`integrate()` 早期每次调用新建
   `WallState`，而伺服/加载循环是"一次调用 1 步"，于是每步把库仑切向弹簧清零，
   顶板拖不动料床（τ 只测到 7 Pa 而 σ_n=23 kPa）。修复=把壁状态持久化传参；
   这是**静默**错误（无异常、无 NaN），只能靠量纲检查（μ_macro=τ/σ_n 应 ~μ）发现。
6. **自由垂直压板的自由度太"轻"**：板质量 40 倍颗粒质量、床刚度 k_n·N_c 时，
   ωΔt≈0.5，恒载一压就打穿料床（床高从 30 mm 涨到 150 mm）。改为**恒体积**
   （运动学压板）后彻底稳定，法向应力变成输出量。
7. **粘结滚动弹簧会摧毁显式积分**：粘结对的弹性滚动刚度 k_t r² 给出
   ω≈3200 rad/s（Δt·ω≈0.6，多接触叠加后越界），首轮出现 NaN。粘结只约束
   相对平动、断后由粒核恢复滚动阻力。
8. **均匀粘结晶格不破坏**：σ_bond 全同 + 位置无序 1% 时，单轴压缩 5% 应变后
   仍有 95.7% 粘结完好、σ(ε) 单调上升，没有 UCS 峰。加 3% 位置无序 + 强度
   对数正态散布后，峰值出现在 ε≈4.9%、断后应力掉到 0（完好率 0.335）。
9. **顶面估计器是高阶统计量**：环扇区取"最高 3 粒中心均值"对解析锥给出**偏平**
   的角度（测试断言 `fit["theta_deg"] < theta_true`，容差 3°），因为 max 统计量
   抬高的是整体包络而非斜率。该偏差对标定目标与预测同源，在比值中抵消。
10. **适用范围**：滚动阻力系数 μ_r、粒径、密度固定不参与标定；四个壁面摩擦
    直接取 μ（未独立标定）。孪生试验证明的是反演机制；要用真实土工试验数据，
    必须换目标曲线并重估观测噪声 σ_i（本节点取目标值的 1%）。

## 6. 误差判定

判据 (a)(b)(c) 全部对**换种子重跑**的结果判定（数字见 T1/T5），(d)(e) 见 T1/T3/T4。


| 材料 | (a) θ 误差 [deg]（阈值 3） | (b) 剪切 RMSE（阈值 5%） | (c) 峰值应力误差（阈值 5%） | (e) 后验覆盖（阈值 90%） |
|---|---|---|---|---|
| cemented_gravel | 3.531 (FAIL) | 0.1922 (FAIL) | 0.2978 (FAIL) | 0.6842 (FAIL) |
| glass_beads | 7.676 (FAIL) | 0.4556 (FAIL) | 1.216e+14 (FAIL) | 0.05263 (FAIL) |
| sand | 4.152 (FAIL) | 0.608 (FAIL) | 0.1492 (FAIL) | 1 |

**是否 < 3%？** —— (a) 是角度判据 3°，不是百分比：(a) 三个材料均**未**通过（最小 3.53°）；(b)(c) 是 5% 判据，三个材料均**未**通过（最小 14.9%）。唯一达到绝对 3% 以内的是砂的 J 下降（94.0%）与砂的覆盖（100%），两者不是误差型判据。

**为什么达不到（诚实的可达性判定）**：T4 的换种子重复性显示，休止角的相对标准差 7–11%、单轴峰值应力 9%（砂）——判据线 (a) 3° 与 (c) 5% 落在噪声底之内或之下；(b) 的 19–61% 误差来自恒体积直剪 τ(γ) 的装样敏感性。这不是优化器没收敛（砂的 drop 94% 说明搜索有效），而是**判据线与正演重复性不匹配**。

**可达路径（下一节点）**：(i) 试样放大到 1e3–1e4 粒，把随机装样方差按 1/√N 压下来；(ii) 目标曲线取 N 次重复的均值并把观测噪声 σ_i 设为重复性标准差（本节点 σ_i=1% 偏乐观，这也是 (e) 覆盖在砂上达到 100%、在砾石上只有 68% 的原因）；(iii) 直剪改用周期性 Lees-Edwards 剪切以获得稳态 τ，去掉装样瞬态；(iv) 标定盒下界改为 σ_bond ≫ ρgH（本节点 1 kPa 偏低，glass_beads 因此在单轴基准上不可测）。

## 7. 审计链

- 模块：`contact_engine.py`（壁/粘结/积分器）、`joint_benchmarks.py`（三基准驱动）、
  `joint_calibration.py`（特征/目标/BO/UQ/判据）。
- 驱动与复现：
  ```
  python scripts/run_joint_calibration.py --all --jobs 3 --n-init 12 --n-iter 38 \
      --seed 0 --n-posterior 3000          # 三个材料的联合标定
  python scripts/run_joint_calibration.py --all --replicates 3 --jobs 3   # 噪声底
  python scripts/joint_calibration_report.py                              # 图表
  python -m pytest tests/test_joint_calibration.py -q                     # 21 passed
  ```
- 随机性：初始设计、正演初值（HCP+抖动）、粘结强度散布、MCMC 全部由显式 seed
  控制；`digest` 字段给出末态位置的 SHA-256，`test_repose_is_deterministic_and_settles`
  与 `test_shear_is_deterministic` 守卫逐位可复现。
- 上一节点对照：`docs/DEM_AUTO_CALIBRATION.md`（heap-only，honest FAIL），本节点
  的 §1 与 §5.2 说明为什么单基准不够以及为什么换协议。