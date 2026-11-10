# FEM → DEM 断裂交接链（二）：碎块**形状分辨**带粒团 + 运动学传递 + 交接后动力守恒

**仓** TensorDEM · **战役节点** p302w2-n03-fem-dem-fracture-handoff · **模块** `tensordem.fem_dem_seeding` · **前序节点** p302w1-n03（`tensordem.fem_dem_fracture`，本节点在其契约之上推进，不重复其已验证内容）

**产物** `artifacts/fem_dem_seeding/p302w2_n03/failure_faces.json`（断裂面契约，含裂纹折线 / 断裂单元 / 碎块多边形）、`artifacts/fem_dem_seeding/p302w2_n03/fem_node_state.json`（**新增**：FEM 节点挠度与 slope 场，运动学传递的唯一输入）、`artifacts/fem_dem_seeding/p302w2_n03/dem_bodies.json`（DEM 体初值）、`artifacts/fem_dem_seeding/p302w2_n03/report_data.json`（本文件全部数字的唯一来源，程序化注入）。

**复现** `PYTHONPATH=src python scripts/run_fem_dem_seeding.py` → `PYTHONPATH=src python scripts/make_fem_dem_seeding_doc.py`；测试 `PYTHONPATH=src pytest tests/test_fem_dem_seeding_handoff.py -q`（33 用例）。

## 1. 案例介绍（物理背景 / 几何 / 参数）

浮冰板的弯曲破坏是连续介质问题，碎冰块之后的漂移、碰撞与堆积是离散颗粒问题。层冰（level ice）在结构面非均匀接触压力下压弯，弯曲应力 $\sigma=6|M|/t^2$ 达到弯曲强度 $\sigma_f$ 时开裂，板被裂纹网络切成若干碎块；**碎块一旦形成，就只剩离散体还能描述它**——这正是交接（handoff）的物理必要性：连续性求解器必须把自己的终态，无质量损失、无几何漂移、无运动学断点地交给离散求解器。

本节点沿用前序节点已经验证过的同一块冰板与同一套断裂工况（保证两次交接的差异只来自**交接本身**，而不来自工况），并对其交接内容做三处推进：形状分辨、运动学传递、交接后动力守恒。

| 量 | 值 | 来源 |
|---|---|---|
| 冰板尺寸 [m] | 10.0 × 10.0 | meta.plate |
| 厚度 t [m] | 0.5 | meta.plate |
| 板面积 [m²] | 100.0 | meta.plate |
| 杨氏模量 E [Pa] | 5.000e+09 | meta.material |
| 泊松比 ν | 0.3 | meta.material |
| 密度 ρ_ice [kg/m³] | 917.0 | meta.material |
| 弯曲强度 σ_f [Pa] | 7.000e+05 | meta.material |
| FEM 网格 | 40 × 40 ACM 板单元（h = 0.25 m） | summary.case |
| 首裂压力 q_first [Pa] | 3.501e+04 | summary.case |
| 加载速率 q̇ [Pa/s] | 5.000e+03 | summary.case |
| 载荷因子的时间率 λ̇ [1/s] | 1.428e-01 | summary.case |
| 碎块数 N_floe | 7 | summary.case |
| 裂纹总长 [m] | 54.0 | summary.case |
| 海水密度 ρ_w [kg/m³] | 1025.0 | summary.case |

厚度与密度决定碎块的水动力时间尺度：浮冰垂荡（heave）的固有角频率为 $\omega=\sqrt{\rho_w g/(\rho_i t)}$，**与碎块尺寸无关**，本工况 $\omega$ = 4.6822 rad/s，周期 1.3419 s。这个「与尺寸无关」恰恰是检验交接质量的利器：若某个碎块拿到的质量与其面积/厚度不自洽，它的周期就会偏离同一条频率线。

## 2. 怎么做的（方法 / 流程 / 关键设置）

### 2.1 连续侧（复用 p302w1-n03，未改动）

Adini–Clough–Melosh 12 自由度的 Kirchhoff 板单元 + Winkler 浮力基础 $k=\rho_w g$；接触压力按裁剪矩形精确积分；自由边界；裂纹追踪为冻结场 mode-I 格点行走，裂纹沿单元边行走因此碎块划分是**精确的连通分量划分**（面积守恒到舍入误差的来源）。本节点只**读取**它的两个产物，不改其判据。

### 2.2 交接契约：两份文件，全靠文件驱动

* `failure_faces.json`（前序契约，schema `tensordem.failure_faces/1`）：`cracks` 裂纹折线、`fractured_elements`/`crack_cells` 断裂单元、`floes` 碎块多边形 + 面积/质心/等效直径/周长/质量、`fsd`、`conservation` 闭合账本；读取端**只用文件本身**重建账本并校验，被手改的工件无法悄悄进入 DEM。
* `fem_node_state.json`（**本节点新增**，schema `tensordem.fem_node_state/1`）：FEM 全部 1681 个节点（41 × 41 网格）的 `(x, y, w, w,x, w,y)` 与载荷因子、载荷率。有了它，离散侧的运动学初值不再依赖进程内回调，而是与几何一样来自磁盘。

### 2.3 形状分辨的碎块带粒团（clump）

前序交接给每个碎块一个**等面积圆盘**（$D=\sqrt{4A/\pi}$，质心处）。它让质量/面积账本闭合，但对接触模型是错误的：长条或凹形碎块会被一个圆盘替换，圆盘既溢出碎块，也压到邻居身上。本节点把每个碎块播成**内接、互不重叠的圆盘团**，规则可审计：

1. 圆盘中心取三角格点（最密排列，面填充分数 $\pi/(2\sqrt3)=0.907$），格距 $2r$ ⇒ 同团内两两不相交；
2. 只保留落在碎块内、且到边界距离 $\ge r$ 的格点（栅格 EDT 判据，保守）；
3. 每个圆盘半径取 $\min(r,\ \text{该点到碎块边界的精确距离})$（点—线段精确距离，不依赖栅格分辨率）⇒ **每个圆盘都在自己碎块内部**，因此也不可能越过断裂面压到邻块；
4. 半径为 $r$ 的格距固定后，用**单调函数二分**求满足目标盘数的 $r$（盘数与形状有关，是碎块的属性而非调用者的承诺）；
5. 碎块质量按圆盘面积占比分摊，$\sum_j m_j = m_\text{floe}$ 由归一化**恒等成立**（形状覆盖不足只损失几何分数，绝不损失质量）。
生产设置：目标盘半径 0.10 m（= 0.4 × FEM 单元 0.25 m）。

### 2.4 运动学传递（准静态 FEM 的"速度=0"是错的）

断裂发生的那一刻，板正在弯。把碎块初速设为 0，等于抹掉了 FEM 用真金白银算出来的运动学状态。对每个碎块，取其内部 FEM 节点，做**刚体平面最小二乘拟合**

$$w(x,y) \approx w_0 + p\,(x-x_0) + q\,(y-y_0)$$

其中 $(x_0,y_0)$ 是该碎块自身节点的形心——因此 $w_0$ 就是碎块的**平动**（DEM 需要的 $z_0$），而 $p,q$ 是它的**倾角**。乘以载荷率 $\dot\lambda$ 得到初值：

$$z_0=w_0,\quad v_z=\dot\lambda\,w_0,\quad \omega_x=\dot\lambda\,(-q),\quad \omega_y=\dot\lambda\,p$$

同时报告一个决定"离散化是否合法"的数字——**刚性残差** $\varepsilon_\text{rig}=\mathrm{rms}(w-\hat w)/\Delta w_\text{floe}$，即刚体平面无法复现的那部分自身弯曲。它小，说明碎块确实在平动+倾斜，刚体碎块 DEM 没有丢东西；它大，说明碎块还在弯，任何质量账本都发现不了的建模错误已经发生。
另外做一次**独立交叉检验**：拟合斜率 $p,q$ 必须与该碎块 FE 节点 slope 场 $(w_{,x},w_{,y})$ 的均值一致——两个不同的插值给出同一个数才算证据。

### 2.5 交接后动力守恒

* **垂荡**：$\ddot z=-\omega^2 z$，$\omega^2=\rho_w g A/m$（刚度用碎块自身足迹面积、质量用交接分配的质量），velocity-Verlet 积分，从零交叉交点测周期，与闭式解比；同时检查动能+势能漂移。
* **平面漂移**：$\dot v=-(v-U)/\tau$（$\tau$=20.0 s，环境流速 [0.15, -0.05] m/s），RK4 与指数闭式比对。板弯曲求解本身没有膜运动学，故平面速度是环境输入（如实声明）。
* **账本再闭合**：积分跑完之后重新称质量、量面积、查碎块间穿透。

## 3. 计算结果（判据数字）

### 3.1 机器可算判据（15/15 通过）

| 判据 | 通过 | 实测 | 目标 |
|---|---|---|---|
| C1_fracture_artefact_written | ✅ | n_floe=7, crack_total_length_m=5.400e+01 | sigma_max >= sigma_flexure, failure_faces.json written with fracture faces |
| C2_mass_conservation | ✅ | 0.000e+00 | < 1e-6 kg |
| C2b_disc_mass_conservation | ✅ | -4.584e-10 | < 1e-6 kg |
| C3_area_conservation | ✅ | 0.000e+00 | < 1e-3 m^2 |
| C3b_boundary_identity | ✅ | rim_residual_m=0.000e+00, overused_segments=0 | rim rebuilt from floe edges equals plate perimeter, no edge used >2x |
| C4_floe_count_in_range | ✅ | 7 | [3, 30] |
| C5_end_to_end | ✅ | n_floe=7, n_bodies=7 | DEM body per floe read from the files alone |
| W1_shape_no_spill | ✅ | 0.000e+00 | clump material does not leave its floe |
| W2_no_initial_overlap | ✅ | n_overlapping_pairs=0, baseline_pairs=9 | no inter-floe interpenetration at t=0 |
| W3_rigidity_residual | ✅ | 1.002e-01 | < 0.15 of the plate deflection range |
| W4_heave_period_closed_form | ✅ | max_period_error=1.028e-05, period_spread_s=2.220e-16 | measured heave period vs sqrt(rho_w g/(rho_i t)) < 3 % |
| W5_post_run_conservation | ✅ | mass_residual_kg=0.000e+00, area_residual_m2=0.000e+00 | books still closed after integration |
| W0_shape_improved_per_floe | ✅ | mean_iou_clump=6.331e-01, mean_iou_baseline=3.368e-01, min_gain=1.969e-01 | clump IoU above the equivalent disc on every floe |
| W6_raster_resolved | ✅ | 8.469e-04 | < 5 % change when h is halved |
| W7_drift_closed_form | ✅ | 1.228e-13 | RK4 vs exponential < 1e-4 |

判据里 C1–C5 对应战役表的机器可算判据；W0–W7 是本节点自己提出的**强化判据**（形状、重叠、刚性、闭式频率、积分后守恒、栅格收敛、漂移闭式）。

### 3.2 破块与带粒团（逐块）

| floe | A [m²] | m [kg] | N_disc | r_min [m] | r_max [m] | 覆盖 | IoU(团) | 溢出(团) [m²] | IoU(等效盘) | 溢出(等效盘) [m²] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 46.125 | 21148.3 | 1333 | 0.096 | 0.096 | 0.831 | 0.831 | 0.0000 | 0.562 | 12.938 |
| 1 | 1.250 | 573.1 | 28 | 0.075 | 0.075 | 0.396 | 0.396 | 0.0000 | 0.129 | 0.968 |
| 2 | 0.312 | 143.3 | 9 | 0.075 | 0.075 | 0.522 | 0.522 | 0.0000 | 0.325 | 0.156 |
| 3 | 4.188 | 1920.0 | 120 | 0.076 | 0.087 | 0.677 | 0.677 | 0.0000 | 0.258 | 2.469 |
| 4 | 46.438 | 21291.6 | 1321 | 0.079 | 0.096 | 0.826 | 0.826 | 0.0000 | 0.511 | 15.019 |
| 5 | 1.125 | 515.8 | 33 | 0.071 | 0.089 | 0.669 | 0.669 | 0.0000 | 0.347 | 0.543 |
| 6 | 0.562 | 257.9 | 16 | 0.075 | 0.075 | 0.510 | 0.510 | 0.0000 | 0.226 | 0.355 |

总盘数 2860；平均 IoU 由 0.337（等效盘）升到 0.633（带粒团），**逐块**提升（最小增益 0.197）。等效盘的溢出合计 32.45 m²（占板面积 32.4%），带粒团溢出 0.0000 m²。

### 3.3 守恒账本（交接前 / 交接后 / 积分后）

| 量 | 带粒团 | 等效盘 | 积分后 |
|---|---|---|---|
| 体数 | 7 | 7 | 7 |
| 面积和 [m²] | 100.000000000000 | 100.000000000000 | 100.000000000000 |
| 面积残差 [m²] | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 质量和 [kg] | 45850.000000 | 45850.000000 | 45850.000000 |
| 质量残差 [kg] | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 盘质量和残差 [kg] | -4.584e-10 | 0.000e+00 | -4.584e-10 |
| 材料面积和 [m²] | 81.218 | 100.000 | — |

### 3.4 边界恒等式（几何守恒的最强形式）

| 量 | 值 |
|---|---|
| 碎块边界线段总数 | 334 |
| 只出现一次的边（板外缘）[m] | 40.000000000 |
| 板周长 [m] | 40.000000000 |
| 外缘残差 [m] | 0.000e+00 |
| 被两碎块共享的边（断裂面）[m] | 21.750 |
| 被 3 次以上使用的边 | 0 |
| 单次使用边数 / 共享边数 | 160 / 174 |

这条恒等式是逐线段（multiset）核对的：任何一条边若被复制或漏掉，面积总分仍可能对，但线段计数立刻暴露——这正是它比"面积和相等"更强的地方。（本实现第一版把线段键写成四个坐标的排序，水平与竖直的单元边在共用角点时被判为同一条，判据当场报出 5 条超用边——说明这条检查有牙。）

### 3.5 运动学初值（逐块）

| floe | N_node | 刚性残差 [%] | z₀ [mm] | v_z [mm/s] | pitch [mrad] | roll [mrad] | 斜率差 p−θx | 斜率差 q−θy | 自然斜率尺度 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 738 | 1.30 | 199.5 | 28.49 | 1.641 | 5.906 | 1.397e-05 | 6.986e-05 | 6.618e-03 |
| 1 | 20 | 0.95 | 203.2 | 29.02 | 1.556 | 5.961 | 8.300e-05 | 1.494e-04 | 7.855e-03 |
| 2 | 5 | 0.29 | 189.4 | 27.05 | 0.000 | 6.122 | 1.865e-03 | 3.110e-06 | 1.095e-02 |
| 3 | 67 | 0.35 | 188.3 | 26.89 | 1.140 | 6.527 | 8.632e-05 | 2.056e-04 | 1.159e-02 |
| 4 | 743 | 1.07 | 166.6 | 23.80 | 1.700 | 6.704 | 1.456e-05 | 2.943e-05 | 7.058e-03 |
| 5 | 18 | 0.32 | 175.2 | 25.02 | 1.820 | 6.693 | 3.772e-05 | 8.458e-05 | 7.905e-03 |
| 6 | 9 | 0.29 | 178.3 | 25.47 | 0.000 | 6.696 | 1.343e-03 | 1.152e-05 | 1.784e-02 |

全板刚性残差（按节点加权、以全场挠度极差归一）10.02%；全板挠度极差 0.0800 m。拟合斜率与 FE slope 均值之差，最大为碎块自然斜率尺度 $\Delta w/\sqrt A$ 的 17.0%（floe 2，该块只有 5 个节点、几乎不倾斜），其余碎块均在 2% 以内——两个独立的插值给出同一个倾角。

### 3.6 垂荡动力（闭式对照）

| floe | A [m²] | m [kg] | ρ_body [kg/m³] | ω [rad/s] | T_测 [s] | T_闭式 [s] | 相对误差 | 振幅 [mm] | 能量漂移 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 46.125 | 21148.3 | 917.00 | 4.682228 | 1.341908 | 1.341922 | 1.028e-05 | 199.56 | 6.163e-05 |
| 1 | 1.250 | 573.1 | 917.00 | 4.682228 | 1.341908 | 1.341922 | 1.028e-05 | 203.27 | 6.163e-05 |
| 2 | 0.312 | 143.3 | 917.00 | 4.682228 | 1.341908 | 1.341922 | 1.028e-05 | 189.48 | 6.163e-05 |
| 3 | 4.188 | 1920.0 | 917.00 | 4.682228 | 1.341908 | 1.341922 | 1.028e-05 | 188.35 | 6.163e-05 |
| 4 | 46.438 | 21291.6 | 917.00 | 4.682228 | 1.341908 | 1.341922 | 1.028e-05 | 166.67 | 6.163e-05 |
| 5 | 1.125 | 515.8 | 917.00 | 4.682228 | 1.341908 | 1.341922 | 1.028e-05 | 175.25 | 6.163e-05 |
| 6 | 0.562 | 257.9 | 917.00 | 4.682228 | 1.341908 | 1.341922 | 1.028e-05 | 178.43 | 6.163e-05 |

最大周期误差 1.028e-05；碎块之间的周期差异 2.220e-16 s（= 机器精度，因为 $\omega$ 与尺寸无关）；最大能量漂移 6.163e-05。对照：若用等效盘（同样质量、同样足迹面积），周期误差 1.028e-05——**垂荡频率对两种表示都不敏感**，这正说明"动力守恒通过"不能替代"形状正确"，两者必须分开考。

### 3.7 形状分数随带粒团分辨率的收敛

| floe | 目标半径 [m] | N_disc | 实际半径均值 [m] | 覆盖 | IoU | 溢出 [m²] |
|---|---|---|---|---|---|---|
| 4 | 0.40 | 89 | 0.349 | 0.732 | 0.732 | 0.0000 |
| 4 | 0.25 | 212 | 0.225 | 0.725 | 0.725 | 0.0000 |
| 4 | 0.15 | 598 | 0.141 | 0.803 | 0.803 | 0.0000 |
| 4 | 0.10 | 1321 | 0.096 | 0.826 | 0.826 | 0.0000 |
| 4 | 0.07 | 2556 | 0.070 | 0.847 | 0.847 | 0.0000 |
| 0 | 0.40 | 82 | 0.327 | 0.599 | 0.599 | 0.0000 |
| 0 | 0.25 | 213 | 0.222 | 0.713 | 0.713 | 0.0000 |
| 0 | 0.15 | 586 | 0.140 | 0.787 | 0.787 | 0.0000 |
| 0 | 0.10 | 1333 | 0.096 | 0.831 | 0.831 | 0.0000 |
| 0 | 0.07 | 2515 | 0.070 | 0.840 | 0.840 | 0.0000 |

分数单调升向圆盘最密堆积极限 0.9069 而**永远不能到达**：不相交圆盘无法无缝铺满多边形，宽度 $r$ 的边界带必然无盘。把这一点写成数字，而不是含糊地说"带粒团很好"。

### 3.8 平面漂移闭式对照

$\tau$ = 20.0 s 的线性阻尼弛豫，RK4（dt = 0.05 s，60 s）与 $v=U+(v_0-U)\,e^{-t/\tau}$ 的最大相对偏差 1.228e-13；t = 60 s 时 v = 0.150242 m/s（理论 0.150242，目标流速 0.1581 m/s）。

## 4. 结果图

**图 1** `artifacts/fem_dem_seeding/p302w2_n03/figures/fig1_footprint_vs_body.png` — Continuum floe footprint (grey outline) and the DEM material seeded into it: left, the inscribed non-overlapping disc clump of this campaign; right, the single equivalent disc of the first handoff. The dashed square is the plate rim; the baseline disc crosses it.

![fig1_footprint_vs_body.png](artifacts/fem_dem_seeding/p302w2_n03/figures/fig1_footprint_vs_body.png)

**图 2** `artifacts/fem_dem_seeding/p302w2_n03/figures/fig2_shape_fidelity.png` — Per-floe shape scores. IoU is the symmetric-difference score of the body's material against its footprint; coverage is the fraction of the floe the discs fill. The clump beats the equivalent disc on every fragment of any elongation.

![fig2_shape_fidelity.png](artifacts/fem_dem_seeding/p302w2_n03/figures/fig2_shape_fidelity.png)

**图 3** `artifacts/fem_dem_seeding/p302w2_n03/figures/fig3_kinematics.png` — Left: rigidity residual, i.e. the part of each floe's own deflection field that the best-fitting rigid plane cannot reproduce, normalised by that floe's deflection range. Right: the initial heave velocity (bars) and heave offset (line) transferred from the FE node field at the loading rate.

![fig3_kinematics.png](artifacts/fem_dem_seeding/p302w2_n03/figures/fig3_kinematics.png)

**图 4** `artifacts/fem_dem_seeding/p302w2_n03/figures/fig4_heave_dynamics.png` — Left: velocity-Verlet heave of every seeded floe. The frequency is set by thickness and density only, so all fragments move together -- a size-independent period proves the handed-over mass and buoyancy stiffness are consistent. Right: measured vs analytic period against the 3 % line.

![fig4_heave_dynamics.png](artifacts/fem_dem_seeding/p302w2_n03/figures/fig4_heave_dynamics.png)

**图 5** `artifacts/fem_dem_seeding/p302w2_n03/figures/fig5_ledger_and_overlap.png` — Left: absolute mass, disc-mass and area residuals of both representations and of the ledger re-closed after integration (log scale; the 1e-18 floor marks an exact zero). Right: material leaving its own floe and initial inter-floe interpenetration -- the two defects the equivalent disc cannot avoid.

![fig5_ledger_and_overlap.png](artifacts/fem_dem_seeding/p302w2_n03/figures/fig5_ledger_and_overlap.png)

**图 6** `artifacts/fem_dem_seeding/p302w2_n03/figures/fig6_shape_convergence.png` — IoU of the clump against its floe footprint as the clump is refined (log axis). The score rises towards the densest-disc-packing limit of 0.907 and cannot cross it: a disjoint disc clump can never tile a polygon, which is why the exact footprint polygon travels with the body.

![fig6_shape_convergence.png](artifacts/fem_dem_seeding/p302w2_n03/figures/fig6_shape_convergence.png)

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **圆盘团永远铺不满**：最密排列上限 0.9069，实际 IoU 0.40–0.83。覆盖不足只影响几何分数，不影响质量（质量按面积占比归一化）。因此**精确多边形足迹必须随体一起交付**，供形状分辨的接触模型使用；带粒团是给圆盘型 DEM 引擎用的表示。
2. **单单元宽的碎块无法用圆盘描述**：floe 1/6 的最窄处就是一个 FEM 单元（0.25 m），内接半径上限 0.125 m，IoU 只有 0.40/0.51。这是形状本身的限制，不是实现缺陷——把数字摆出来，比把阈值调到能过更诚实。
3. **刚性残差是离散化的合法性判据**：本工况全板 10.02%，最大单块 1.30%（逐块见 3.5 表）。若换到更薄、更强、开裂后仍大幅弯曲的工况，这个数字会上升，届时刚体碎块 DEM 不再成立，需要壳单元碎块或子结构。它**没有**解析基准，故只能给出量级判断（见第 6 节）。
4. **垂荡忽略附加质量**：$C_a=0$ 时周期 1.342 s；实际浮冰 $C_a\approx0.5$–1 会使周期长 5–15%。本节点把 $C_a$ 作为显式参数（默认 0）暴露，未做标定。
5. **平面运动学来自环境输入**：板弯曲是垂直方向的，膜方向速度不由该求解器产生；3.8 的漂移检查验证的是积分器与闭式的一致性，不是冰-流耦合的物理。
6. **栅格测量误差已量化**：全部形状分数在 h = 0.031 m 与其一半上重测，最大相对变化 8.469e-04（< 5% 判据）；多边形面积在栅格上**精确**（误差 0.000e+00 m²），因为碎块多边形边全部对齐 FEM 格点。
7. **裂纹网络仍是冻结场模式 I 生成器**（前序节点的已声明局限）：无裂尖应力重分布、无动力、无内聚区。故本节点交付的是**碎块几何与运动学的数据总线**，不是强度预测。
8. **DEM 侧只做了"一步"**：垂荡 + 平面漂移 + 穿透审计，没有做碎块间接触力学的长时演化（那属于碎冰场 DEM 的其它节点）。本节点的判据是"初值自洽且守恒"，不是"碎冰场演化正确"。

## 6. 误差判定（关键量 vs 基准，逐项回答"是否 < 3%"）

| 关键量 | 基准 | 实测误差 | < 3% ? |
|---|---|---|---|
| 垂荡周期（7 个碎块，最大） | 闭式 $2\pi\sqrt{\rho_i t/(\rho_w g)}$ | 1.028e-05 | 是 |
| 积分器能量漂移（velocity-Verlet, 400 步/周期） | 能量守恒（同阶收敛验证） | 6.163e-05 | 是 |
| 平面漂移速度 | 指数闭式 | 1.228e-13 | 是 |
| 形状分数对栅格分辨率的敏感性 | h → h/2 自收敛 | 8.469e-04 | 是 |
| 质量守恒（体/盘/积分后） | 板质量 ρ A t | 0.000e+00 | 是 |
| 面积守恒（体/积分后） | 板面积 | 0.000e+00 | 是 |
| 边界恒等式（外缘重建） | 板周长 | 0.000e+00 | 是 |
| 拟合斜率 vs FE slope 均值（节点数 ≥ 18 的碎块，最大） | 同一 FE 场的另一插值（按碎块自然斜率 $\Delta w/\sqrt A$ 归一） | 1.902e-02 | 是 |
| 同上，退化碎块（< 18 节点，pitch≈0） | 同上；两个接近 0 的斜率相比无物理意义 | 1.703e-01（仍 < 自然斜率的 20%，见 3.5） | 不适用（已声明） |
| 带粒团形状 IoU | 最密圆盘堆积极限 0.907 | 0.633（比值 69.8%） | 否（结构性上限，非数值误差，已量化） |
| 刚性残差（刚体化误差） | 无解析基准（离散化合法性指标） | 10.02% | 无基准——原因：它是"刚体假设与真实弯曲场之差"，没有独立参考量 |

**逐项回答"是否 < 3%"**：数值类误差（周期、能量、漂移、栅格、质量、面积、边界、斜率交叉检验）全部 < 3%，多数在机器精度量级；形状 IoU 与刚性残差**不是**数值误差而是模型表示的结构性代价，已给出绝对值与上限（0.907）并与前序基线（等效盘 IoU 0.337）逐块对照，未做"阈值内即通过"的粉饰。

## 7. 与本次交接相关的判据数字汇总（供机器核对）

| 判据 | 数值 |
|---|---|
| C2 质量残差 [kg] | 0.000e+00 |
| C3 面积残差 [m²] | 0.000e+00 |
| C4 N_floe | 7 |
| C5 端到端体数 | 7 |
| W0 平均 IoU（团 / 等效盘） | 0.633 / 0.337 |
| W2 初始穿透对数（团 / 等效盘） | 0 / 9 |
| W3 刚性残差（全板） | 10.02 |
| W4 最大垂荡周期误差 | 1.028e-05 |
| W6 栅格收敛（h→h/2） | 8.469e-04 |
| W7 漂移闭式偏差 | 1.228e-13 |
