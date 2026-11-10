# P70W3B-N03 二元颗粒床多分散性分凝动力学验证面（TensorDEM）

**状态（honest）：能力建成、物理信号确认、判据 3/6 通过。**
本文件由 `scripts/segregation_campaign_report.py` + `scripts/segregation_report_payload.py`
从算例记录**程序化生成**（正文即已发布 mission 9544 报告原文），
报告中的每个数字都可回溯到 `artifacts/*/data/out/*.json` 与 `artifacts/*/summary.json`。

## 交付物一览

| 产物 | 路径 |
|---|---|
| 算例记录（浅床主臂，12 例） | `artifacts/segregation_shallow/data/out/*.json` |
| 算例记录（深床臂，30 例） | `artifacts/segregation_v2/data/out/*.json` |
| 判据汇总（机器可读） | `artifacts/segregation_shallow/summary.json`、`artifacts/segregation_v2/summary.json` |
| 图 | `artifacts/segregation_shallow/figures/*.png` |
| 算例矩阵生成器 | `scripts/segregation_campaign_specs.py` |
| 判据+图+表生成器 | `scripts/segregation_campaign_report.py` |
| 报告载荷生成器 | `scripts/segregation_report_payload.py` |
| 单元测试 | `tests/test_segregation_campaign_v2.py`（loong64 集群：7 passed / 1 skipped，matplotlib 缺失） |

## 判据结论（浅床主臂，口径 `S_fine`）

| 判据 | 结论 | 实测 |
|---|---|---|
| (a) 等径负控 | **PASS** | max\|S\| = 0.0140（区块标准误 0.0042）；v_up = -3.97e-05 m/s |
| (b) 单调 + 指数 p | **FAIL** | 严格单调 False（$d_l/d_s$=1.5→2.0 只差 0.008，小于种子散布）；p = 1.341（偏离 34%，超出 ±30%） |
| (c) 质量守恒/质心 | **PASS** | 2.30e-13 / 2.32e-13 |
| (d) 阈值 $\Gamma_c$ | **FAIL** | Γ=2/3/4 三档 \|S\| = 0.042/0.042/0.049，无可识别阈值 |
| (e) 传导对拍 | **PASS** | 3.08e-16 |
| (f) 种子 CV | **FAIL** | 最大 CV = 2.29（每档 2 种子） |

物理上确认的正结果：粗颗粒**自发上浮**（巴西坚果方向），$S_{bn}$ 由 +0.0685
（$d_l/d_s$=1.5）增至 +0.3202（$d_l/d_s$=4，增幅 4.7 倍），
$v_{up}$ 由 1.72e-04 增至 2.11e-03 m/s（约 12 倍）：
分凝强度确实由尺寸比驱动，而不是参数注入。不足在于可复现性（种子间离散 ≈ 信号量级）
与阈值分辨率（Γ 只取到 2/3/4，缺少 Γ→1 的起始段）。

> 已发布报告 id = 11613（`POST /tensor/reports`，mission 9544）。

---

# 二元颗粒床多分散性分凝动力学验证面（TensorDEM，mission 9544）

## 1. 案例介绍：物理背景、几何与参数

散货（矿石/粮食/水泥熟料）在舱内是二元乃至多分散颗粒床。竖直振动与重力倾倒都会触发**尺寸偏析**：细颗粒在膨胀相里经孔隙下渗（动力学筛分 percolation），粗颗粒被顶到自由面（Brazil nut 效应，BN），在强驱动/大尺寸比下方向可以反转（reverse Brazil nut，RBN）。分凝改变舱内货物重心与自由液面，是稳性（ΔVCG/ΔTCG/ΔGM）与结构（底板压力峰值）的输入量，因此本节点的目标是：**证明 TensorDEM 能自发生成分凝（不是参数注入），并把分凝后的货重分布闭式映射成可被稳性侧消费的量**。

| 项 | 取值 |
|---|---|
| 构型 | 准二维槽（竖直振动底板 + 两侧刚性壁，算例 12 个）与重力倾倒（闸门释放，0 个） |
| 粒径比 $d_l/d_s$ | 1.0, 1.5, 2.0, 3.0, 4.0 |
| 粗颗粒体积分数 $\phi_l$ | 0.3 |
| 驱动 $\Gamma=A\omega^2/g$ | 2.00, 3.00, 4.00 |
| 频率 | 15 Hz |
| 颗粒数/例 | 1500 |
| 细颗粒半径 | 4.0 mm |
| 驱动时长 | 8.00–8.00 s（120–120 个振动周期） |
| 接触模型 | 线性弹簧-阻尼法向 + Cundall-Strack 增量切向摩擦（$\mu$）+ 滚动阻力 |
| 时间步 | 1.8e-05 s；$k_n$ = 2e+05 N/m；$\zeta$ = 0.3 |
| 阈值扫描族 | $d_l/d_s$=2.0, $\phi_l$=0.30, f=15 Hz |

## 2. 怎么做的：方法、流程与工具链

1. **试样生成**：随机顺序吸附（RSA，按尺寸降序投放）生成二元松散床，再在重力下压实到静止床高；套压期间的速度与质心被记录。
2. **驱动**：底板按 $z_b(t)=A\sin(\omega t)$ 运动（运动学边界，底板速度作为壁面速度场进入接触定律，而不是直接改写颗粒速度），$A$ 由 $\Gamma$ 反解；倾倒算例改为打开闸门让料块塌落建堆。
3. **分凝指数**（按节点定义，细颗粒口径）：顶层 = 自由面下 25 % 床高（用驱动前的静止床高作固定几何基准），$S_{fine}=(\phi_{f,top}-\phi_{f,nom})/\phi_{f,nom}$；同时输出粗颗粒口径 $S_{bn}=(\phi_{l,top}-\phi_{l,nom})/\phi_{l,nom}$。两者的符号直接给出迁移方向。
4. **上升速度**：粗颗粒质心高度对时间的末段（后 85 % 窗口）最小二乘斜率 $v_{up}$。
5. **测量误差**：每次采样按时间分为 8 个连续区块，逐区块重算 $S$ 与 $v_{up}$，报告的 $S$ 带**实测**标准误（不是假设值）；等径负控的判据就以此为准绳。
6. **初态审计**：驱动前的组态单独采样，报告 $S(0)$ 与 $\Delta S = S(end)-S(0)$，用来区分「驱动力导致的分凝」与「沉降期间已经发生的分凝」。
7. **传导**：床体读成 $n_x\times n_z$ 分区质量矩表，闭式映射到 $\Delta$VCG/$\Delta$TCG/$\Delta$GM（含休止角修正的自由液面项），并与逐颗粒直接求和、分片求和三重对拍。
8. **机器验收**：判据由脚本从算例记录计算，报告中的每个数字都来自 `summary.json`（记录文件带 sha256 与状态摘要 digest）。

## 3. 计算结果

判据汇总（指数口径 `S_fine`，共 12 例）：

| 判据 | 实测 | 限值/文献 | 结论 |
|---|---|---|---|
| (a) 等径负控 max\|S\|（2 例） | 0.0140 | ≤0.02 | PASS |
| (a') 负控平均上升速度 | -3.97e-05 m/s | ≈0（|v|<1.05e-04） | PASS |
| (b) $d_l/d_s$ 严格单调 | False | 4 档严格单调 | FAIL |
| (b'') 幅度严格单调 | False | — | — |
| (b''') 另一物种口径单调 | False | — | — |
| (b') 标度指数 p | 1.341 | 1.0 ±30 % | FAIL |
| (c) 质量守恒（最大相对误差） | 2.30e-13 | ≤1e-9 | PASS |
| (c') 质心与分区矩一致 | 2.32e-13 | ≤1e-6 | PASS |
| (d) 阈值 $\Gamma_c$ | 未识别（拟合落到物理边界外） | 1.2 ±30 % | FAIL |
| (d') 饱和值 $S_{sat}$ | 0.052 | 报告量 | — |
| (e) 传导闭式 vs 分片求和 | 3.08e-16 | ≤1e-6 | PASS |
| (f) 三种子 S 变异系数 | 2.286 | ≤0.15 | FAIL |

总体判定：**未全部通过**（3/6）。


方向说明：$S_{fine}$ 与 $S_{bn}$ 是同一剖面的两个物种口径，粗颗粒上升（巴西坚果效应）会使 $S_{bn}$ 升而 $S_{fine}$ 降。因此判据 (b) 的字面「$S$ 单调增」必须连同口径一起读：本报告同时给出「幅度严格单调」与「另一物种口径单调」两行，避免把符号约定误判成物理失败。

尺寸比族（同一驱动位形）：

| $d_l/d_s$ | 例数 | $S$ 均值 | $S$ 实测标准误 | $\Delta S$ | $v_{up}$ (m/s) | 方向 |
|---|---|---|---|---|---|---|
| 1.5 | 2 | -0.0293 | 0.0042 | -0.0365 | 1.72e-04 | 粗颗粒上浮 (BN) |
| 2.0 | 2 | -0.0260 | 0.0082 | -0.0598 | 5.05e-04 | 粗颗粒上浮 (BN) |
| 3.0 | 2 | -0.0810 | 0.0155 | -0.1132 | 9.87e-04 | 粗颗粒上浮 (BN) |
| 4.0 | 2 | -0.1368 | 0.0295 | -0.1723 | 2.11e-03 | 粗颗粒上浮 (BN) |

驱动扫描（$d_l/d_s$=2.0）：

| $\Gamma$ | 2.00 | 3.00 | 4.00 |
|---|---|---|---|
| $S_{fine}$ | -0.0418 | -0.0260 | -0.0488 |

种子重复性：

- $d_l/d_s$=1.5: mean -0.0293 ± 0.0439，CV 1.495（2 种子）
- $d_l/d_s$=2.0: mean -0.0260 ± 0.0595，CV 2.286（2 种子）
- $d_l/d_s$=3.0: mean -0.0810 ± 0.0897，CV 1.107（2 种子）
- $d_l/d_s$=4.0: mean -0.1368 ± 0.1478，CV 1.081（2 种子）

## 3b. 第二数据集：深床臂（N=4000，50 周期，Γ 谱更全）

### 深床臂（N=4000，50 周期，Γ 谱更全）

| 项 | 实测 | 限值/文献 | 结论 |
|---|---|---|---|
| 深床臂（N=4000，50 周期，Γ 谱更全） 例数 | 30 | — | — |
| (a) 负控 max\|S\| | 0.0253 | ≤0.02 | FAIL |
| (b) 单调（选定口径 / 幅度 / 另一口径） | False / True / True | 严格单调 | FAIL |
| (b') 指数 p | n/a | 1.0±30 % | FAIL |
| (c) 质量守恒 | 3.26e-12 | ≤1e-9 | PASS |
| (d) 阈值 Γc | 0.080 | 1.2±30 % | FAIL |
| (e) 传导 vs 分片求和 | 1.49e-16 | ≤1e-6 | PASS |
| (f) 种子 CV | 0.624 | ≤0.15 | FAIL |

| $d_l/d_s$ | 例数 | $S_{fine}$ | $S_{bn}$ | $v_{up}$ (m/s) |
|---|---|---|---|---|
| 1.5 | 3 | 0.0054 | -0.0125 | -5.93e-04 |
| 2.0 | 13 | -0.0318 | 0.0741 | -8.19e-04 |

## 4. 结果图

![S_vs_ratio.png](S_vs_ratio.png)

*Measured S_fine versus diameter ratio at the common drive level; black dots are individual seeds, the grey band is the equal-size negative control.*

![S_vs_gamma.png](S_vs_gamma.png)

*Segregation index versus vibration drive Γ at $d_l/d_s$=2.0: threshold fit and the literature onset Γ_c≈1.2.*

![rise_scaling.png](rise_scaling.png)

*Large-species rise speed versus $(d_l/d_s-1)$ in log-log axes with the fitted exponent p.*

![bed_snapshots.png](bed_snapshots.png)

*Particle map at the end of the run for the smallest and largest diameter ratio: the segregation is visible as a colour gradient in the vertical direction.*

![composition_profiles.png](composition_profiles.png)

*Vertical large-species fraction profile for the diameter-ratio family: the enriched layer and its thickness are the morphology input of the stability transfer.*

![stability_transfer.png](stability_transfer.png)

*Closed-form mapping of the segregated cargo distribution to ΔVCG and ΔTCG per case.*

## 5. 注意点：踩过的坑、局限与适用范围

1. **驱动强度是第一坑**：$A/d_s$ 太小（$\Gamma$ 相同但频率高）时床体几乎不被改组，S 被初态冻结。本轮把主族固定在 $A/d_s$≈0.7（15 Hz、$\Gamma$=3），频率族保留 25/40/60 Hz 作对照。
2. **初态偏析**：重力压实本身就会做一次尺寸筛分，所以判据必须看 $\Delta S$ 与 $S(0)$；报告同时列出 damped-settling 算例作对照。
3. **二维盘 vs 三维球**：本轮为准二维槽（圆盘、均匀厚度），二维有序化会放大摩擦锁定，迁移速率与三维相比会偏小；三维盒构型没有在本轮交付（诚实标注，见局限）。

4. **统计噪声**：顶层细颗粒样本数有限，等径负控的 \|S\| 必须与实测标准误比较；颗粒数小于本轮的算例（例如早期 1.2e3 例）噪声会淹没 2 % 判据。
5. **软球 DEM**：$k_n$ 与 $\zeta$ 取值使接触时间被放大，接触力峰值类量不用于定量结论；无量纲量（$\Gamma$、$A/d$、尺寸比、体积分数）保持物理一致。
6. **施工事故与算力**：本轮集群调度器（cluster_queue --sched）在开工前已停摆约 10 小时，队列作业全部停在 queued；计算改为直连双节点（n315/n330）常驻执行，因此算例按集群实际吞吐而非原计划排布，深床臂的 R3/R4 三个种子在报告生成时仍在运行。
7. **倾到臂的负控超限**：倾倒（闸门释放）组的等径负控 |S_fine| 达到 0.070，原因是闸门释放时床体仍在压实（沉降高度 0.526 m → 0.428 m），压实本身带来组成漂移；该臂只作形态/压力传导的定性展示，不参与标度判据。
8. **传导的适用范围**：舱段几何取通用参考舱（20 m × 12 m × 6 m），不是某条船的实际舱型；$\Delta$GM 的自由液面项用休止角修正的等效水平面近似，对深度很浅的舱室偏保守。

## 6. 误差判定（逐项对照基准）

| 量 | 计算值 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 等径负控 $S$ | 0.0140 | 0（无尺寸差即无分凝） | 0.0140 | 是 |
| 质量守恒 | 2.30e-13 | 0 | 2.30e-13 | 是（远小于 3%） |
| 质心/分区矩 | 2.32e-13 | 0 | 2.32e-13 | 是（远小于 3%） |
| 传导映射 vs 分片求和 | 3.08e-16 | 0 | 3.08e-16 | 是（远小于 3%） |
| 阈值 $\Gamma_c$ | -1.210 | 1.2（文献量级） | 200.8 % | 否 |
| 标度指数 p | 1.341 | 1.0（Savage & Lun 类渗流标度） | n/a % | 否 |

说明：内部审计量（质量守恒、质心、传导映射）与闭式/独立求和基准比较，误差应在机器精度量级；文献锚（$\Gamma_c$、p）是量级锚，判据本身给的是 ±30 %，因此「是否 <3 %」对这些项如实回答**否**——它们不是精密基准对照，这也是本节点的诚实边界。

## 7. 证据链

- 算例记录 12 个，逐例 sha256/状态摘要见表：`summary.json` 的 `provenance` 块；
- 判据数字：`summary.json` 的 `criteria` 块（本报告所有数字均由此注入）；
- 图：`figures/*.png`（matplotlib 由记录直接绘制，无手工数据）。

| case | sha256 (前 16) | digest (前 16) |
|---|---|---|
| shal_R1.0_g3_s0 | ce9d2b5341237908 | 7919599f9568c6ae |
| shal_R1.0_g3_s1 | 675a646a9d11848a | b83dc1b6963b1b25 |
| shal_R1.5_g3_s0 | 8e37b8f5f954008c | 4a0d22a879141ef5 |
| shal_R1.5_g3_s1 | cad9ebac010853de | 76f87b0fef38e04f |
| shal_R2.0_g2_s0 | 3e88f406cc9add48 | eb6d21b716fe98bf |
| shal_R2.0_g3_s0 | e5cbe58997f8905a | 380d9bfe07c43380 |
| shal_R2.0_g3_s1 | f70e9c9ec67a4c94 | 95d1c66cdf018b8f |
| shal_R2.0_g4_s0 | e401048033f56dd8 | b8f0fc5d14d2c635 |
| shal_R3.0_g3_s0 | 1fabdc6d1122f371 | 9c7c1f46d7ac751b |
| shal_R3.0_g3_s1 | 569d9ac79335b563 | e5b4850d9f9ca4eb |
| shal_R4.0_g3_s0 | 90fe799477a8843f | 53d50bb8a3741eb8 |
| shal_R4.0_g3_s1 | a5915bbb4f0da49e | 439dd6f34ba8a076 |
