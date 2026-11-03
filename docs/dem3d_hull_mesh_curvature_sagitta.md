# 三维船体三角网格接触的矢高偏差（验证欠账清偿）

**结论一句话**：`tensordem.hull_mesh_contact` 的三维三角网格接触此前只在**平面**上验收
（`docs/m669_hull_mesh_contact.md` 记 "sphere_contact_force_error = 0.0"），而真实船体是曲面：
内接三角形把曲面替换成**落在曲面之内**的弦平面，压入量被系统性低估
`delta_facet = delta_true - s`，`s = R (1 - cos(pi/N))` 为矢高，相对力误差恰为 `-s/delta`，
且当 `delta <= s` 时**接触完全丢失**。本单用独立闭式基准复核该内核：内核几何与闭式解最大偏差
`3.68e-15 m`，丢失接触阈值与矢高相对偏差 `<1e-4`，并给出可直接使用的网格分辨率判据
`N >= pi / arccos(1 - delta*tol/R)`（本扫描 28 个工况全部满足 3%），
以及把 `hull_radius = s` 当作常量偏移即可**精确抵消中弦偏差**（残差 `6.1e-13`）。

## 1. 案例介绍

**物理背景**：船-冰 DEM 的载荷入口是船体表面接触。三维内核（m669 合并）用 Ericson
最近点三角形算法 + 空间哈希求球-网格罚函数接触力，网格来自 STL/OBJ 船体。
冰载荷对压入深度是**线性**映射（罚弹簧），因此任何几何离散偏差都按同一比例直接进入载荷；
更严重的是，冰-船接触的典型压入量 `delta/r_p ~ 1e-3..1e-2` 与常见 STL 分段数下的矢高同量级，
会出现「明明贴着船壳却零接触力」的静默丢载。

**几何与参数**（全部 SI）：以半径 `R` 的圆柱侧面（船首圆弧的三维对应）为被离散对象，
顶点精确落在圆周上（与二维 2D polyline 审计同一定义），每圈 `N` 条弦、每条弦两个三角形。

| 符号 | 含义 | 取值 |
|---|---|---|
| `R / r_p` | 曲率半径 / 颗粒半径 | {2, 5, 10, 20} |
| `r_p` | 颗粒半径 | 1 m |
| `N` | 圆周弦段数（facets） | {4, 8, 16, 32, 64, 128, 256, 512} |
| `delta / r_p` | 真实压入深度 | {0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2} |
| 采样位置 | 一条弦内的相对位置（0 = 顶点，0.5 = 中弦） | {0, 0.25, 0.5, 0.75, 1.0} |
| 罚刚度 | `k = 1 N/m`（使力与压入量数值等价） | 1 |

## 2. 方法与关键关系

1. **被测**：`mesh_contact_forces`（已合并内核，不改动），球心置于
   `r = R + r_p - delta`、弦内角度 `theta = f * 2*pi/N`。
2. **独立闭式基准**：`closed_form_penetration` 用纯标量代数求点到内接多边形**所有闭合弦段**的
   最小距离（`_point_segment_distance`），不调用任何内核。
3. **二维交叉校验**：同一截面用 `hull_contact.polyline_contact_forces`（二维 polyline 内核）
   再算一遍，检验两条独立实现的几何是否一致。
4. **修正假设**：内核已有 `hull_radius` 偏移参数（`contact_r = r_p + hull_radius`）。
   由于偏差是**与 `delta` 无关的常量** `s`，令 `hull_radius = s` 应精确抵消中弦偏差。

解析关系（可精确验证，非渐近）：

    delta_facet = delta_true - s,     s = R (1 - cos(pi/N)),
    rel. force error = -s / delta,    丢失去接触 iff delta <= s,
    N_min = pi / arccos(1 - delta * tolerance / R).

复现：`python scripts/audit_hull_mesh_curvature_sagitta.py --output docs/hull-mesh-curvature`
（数据表与图全部由该脚本程序化写出，见 `docs/hull-mesh-curvature/tables.md`）。

## 3. 计算结果

### 3.1 中弦相对力误差（实测）vs 闭式 `-s/delta`

| R/rp | N | delta/rp | s/delta | 实测 mid-chord 误差 | 闭式预测 |
|---|---|---|---|---|---|
| 10 | 32 | 0.05 | 0.963 | -96.31% | -96.31% |
| 10 | 64 | 0.05 | 0.241 | -24.09% | -24.09% |
| 10 | 128 | 0.01 | 0.301 | -30.12% | -30.12% |
| 20 | 128 | 0.02 | 0.301 | -30.12% | -30.12% |
| 2 | 32 | 0.05 | 0.193 | -19.26% | -19.26% |

全扫描最不利的非丢载工况：`R/r_p = 2`、`N = 32`、
`delta/r_p = 0.01`，实测误差 `-96.31%`，
闭式预测 `-96.31%`。1120 个位置样本中有
`301` 个（`26.9%`）因 `delta <= s` **完全没有接触力**（其中中弦位置 105/224 = 46.9%）。

### 3.2 丢失去接触的阈值 = 矢高

| R/rp | N | 实测丢失 delta [m] | 矢高 s [m] | 相对差 |
|---|---|---|---|---|
| 2 | 16 | 3.8429e-02 | 3.8429e-02 | 5.8e-15 |
| 5 | 16 | 9.6074e-02 | 9.6074e-02 | 4.6e-15 |
| 10 | 16 | 1.9215e-01 | 1.9215e-01 | 4.6e-15 |
| 20 | 16 | 3.8429e-01 | 3.8429e-01 | 4.6e-15 |

全部 16 个工况的最大相对偏差 `9.2e-14`。

### 3.3 满足 3% 力误差所需的弦段数（闭式 vs 暴力搜索）

| R/rp | delta/rp | 闭式 N | 暴力搜索 N | N 处的实测误差 |
|---|---|---|---|---|
| 10 | 0.002 | 907 | 907 | -3.00% |
| 10 | 0.01 | 406 | 406 | -2.99% |
| 10 | 0.05 | 182 | 182 | -2.98% |
| 10 | 0.2 | 91 | 91 | -2.98% |
| 20 | 0.01 | 574 | 574 | -3.00% |

28 个工况中闭式解与暴力搜索**全部一致**，且在 `N` 处实测误差最大 `2.999%`
（全部 <= 3%），`N-1` 处全部 > 3%。

### 3.4 `hull_radius = s` 常量修正

| 位置 | 修正前 | 修正后 |
|---|---|---|
| 中弦 | `-s/delta`（系统性低估，最不利） | 残差 `6.1e-13`（机器精度） |
| 顶点 | `0`（无偏差） | `+s/delta`（最大 `+96.3%`，对称过载） |

即：常量偏移把**单向低估**换成围绕零的对称锯齿，均值≈0、峰值仍为 `s/delta`；
所以它只能当「无偏化」手段，不能代替足够密的网格——峰值仍由 `N_min` 判据控制。

### 3.5 检查项

| 检查 | 数值 | 判定 |
|---|---|---|
| kernel_geometry_matches_closed_form_penetration | max_abs_deviation_m=3.68e-15, tolerance_m=1.00e-09 | PASS |
| 3d_mesh_kernel_equals_2d_polyline_kernel | max_abs_difference_N=3.67e-15, tolerance_N=1.00e-09 | PASS |
| dropout_threshold_equals_sagitta | max_relative_difference=9.22e-14, tolerance=1.00e-04 | PASS |
| worst_mid_chord_underestimate_over_sweep | value=-1.00e+00, note=most negative relative force error found (systematic underload) | PASS |
| hull_radius_s_offset_cancels_mid_chord_bias | max_abs_mid_chord_error_after_offset=6.12e-13, tolerance=1.00e-09 | PASS |
| required_facets_criterion_delivers_3pct | cases=28, max_abs_error_at_required=3.00e-02 | PASS |
| analytic_reference_consistency | note=delta_facet = delta_true - s (cases with delta > s), max_abs_deviation_m=4.33e-15, tolerance_m=1.00e-12 | PASS |

## 4. 结果图

- **fig1_error_vs_facets.png**：中弦相对误差随弦段数变化（实测点 vs 闭式虚线 `s/delta`），
  含 3% 容差线；`docs/hull-mesh-curvature/fig1_error_vs_facets.png`。
- **fig2_angular_profile.png**：一条弦内的误差剖面（顶点 0 → 中弦最大），
  修正前后对比：单向低估 → 对称锯齿。
- **fig3_required_facets.png**：达到 3% 所需弦段数 vs 设计压入量，标出常见 STL 的 16 段/圈。

## 5. 注意点（局限与适用范围）

- 本单**只做几何 + 接触律的静态评估**，不积分动力学：因此参考值是精确的，
  但传导到「整船冰阻力」的实际误差还取决于压入量分布（真实接触多在顶点附近还是中弦附近）。
- 结论针对**内接**三角网格（顶点在曲面上，STL 导出船体的常规做法）。外接网格误差反号。
- 圆柱截面代表船首圆弧；球面/双曲率面片的矢高由局部曲率半径控制，同一判据可用但需取
  局部最小曲率半径。
- `hull_radius = s` 修正对 `s` 依赖 `N` 与 `R`：网格一改就要重算；且它不能消除顶点处的对称
  过载，故只作为「零均值化」的补充手段。
- 修改 `docs/m669_hull_mesh_contact.md` 中「force error 0.0」的适用范围只补了一行注记
  （该数字仅对平面成立），未改动其验收数据。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 内核压入量（全部 1120 个采样） | 内接多边形闭式最近距离 | max `3.68e-15 m`（相对约 1e-15） | 是 |
| 丢失去接触阈值 | 矢高 `s` | `9.2e-14`（相对） | 是 |
| 3D 网格内核 vs 2D polyline 内核 | 二维独立实现 | max `3.67e-15 N` | 是 |
| `hull_radius = s` 中弦残差 | 真实压入量 | max `6.1e-13`（相对） | 是 |
| 网格分辨率判据 | 暴力搜索最小 N | 28/28 一致；N 处实测误差 <= 3.0% | 是 |

**是否 < 3%：是**（上表五项关键量均满足；其中四项与闭式解一致到机器精度）。
局限：该判定是「内核几何 == 内接网格几何」，而非「内接网格误差 < 3%」；后者由 3.3 节的
`N_min` 判据按设计压入量保证，若沿用常见 16 段/圈 STL 且 `delta/r_p <~ 0.02`，则误差会
显著超过 3%（见 3.1 表），属于必须显式报告的限制。
