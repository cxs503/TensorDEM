# XVAL-DEM-ROUND2-METHOD-SELECTION · macro arm — does the `mu_r → 2 mu_r` mapping survive a column collapse?

**Round 2 supplement** to `docs/XVAL_DEM_ROUND2_METHOD_SELECTION.md` §⑥ (the
first round left exactly this item open: *"宏观臂未跑 … 因此「把 `mu_r` 从 0.5
换成 OSS 约定的 1.0 后柱体 runout 是否复现 2.106」这一条尚无本仓数字，不做结论"*).
**Repo** TensorDEM · **Anchor** `XVAL-DEM-ROUND2-METHOD-SELECTION` (gap-closure
of its declared residual) · **判据** owner 精度标准 **< 15 %**
**Artefacts** `artifacts/xval_dem_round2_method_selection/column_arms.json`,
`artifacts/xval_dem_round2_column_arms/` (`table_macro.md`, `verdict.json`,
`fig1_runout_arms.png`, `fig2_effective_rolling_cap.png`)
**Drivers** `benchmarks/xval_dem_round2_column_arms.py` (frozen matrix),
`…_arm.py` (one arm per process), `…_report.py` (tables/figures/verdict),
`tests/test_rolling_lever_conventions.py` (5 property tests).
Reproduce:

```
PYTHONPATH=src python benchmarks/xval_dem_round2_column_arms_arm.py legacy_mu0.5      # ~50 s each
PYTHONPATH=src python benchmarks/xval_dem_round2_column_arms_arm.py oss_naive_mu0.5
PYTHONPATH=src python benchmarks/xval_dem_round2_column_arms_arm.py oss_mapped_mu1.0
PYTHONPATH=src python benchmarks/xval_dem_round2_column_arms_report.py                # table + 2 figures
PYTHONPATH=src pytest -q tests/test_rolling_lever_conventions.py tests/test_axisym_column_lube2004.py
```

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

**几何与工况**：CASE-COLUMN-COLLAPSE 的冻结 `a = h_i/r_i = 1` 柱体，轴对称柱坐标 DEM；
颗粒在重力下坍落，量测最终铺展比 `r_inf/r_i`（runout）与铺展时间 `t_reach`。

**为什么这条要单跑**：第一轮在**单接触场景**上证明了两码的差别只在工程约定
（杠杆臂 `reff`、扭转分量），并给出换算 `mu_r,OSS = 2 mu_r,TD`（等球）。
但"单接触等价"推不出"宏观等价"：柱体同时含 **粒-粒** 与 **粒-壁** 两类接触，
而两者的杠杆臂在 OSS 约定下**不同**（`r/2` vs `r`）。这条宏观臂就是来裁决的。

**参数表（全冻结，取自 `xval_dem_round2_column_arms.py::BASE`）**

| 量 | 值 | 量 | 值 |
|---|---|---|---|
| `aspect_ratio` a | 1.0（实测 1.0783） | `radius` r | 0.03 m |
| `density` | 2500 kg/m³ | `kn` | 6.0e4 N/m |
| `kt/kn` | 0.5 | `mu` / `mu_wall` | 0.5 / 0.8 |
| `restitution` e | 0.5 | `grains_per_radius` | 8（⇒ 1955 粒） |
| `g` | 9.81 m/s² | `dt` | 自动（≈4.82e-4 s，法向周期/20） |
| `duration_factor` / `min_time` | 8.0 / 0.5 s | `seed` | 12345 |
| 停止判据 | `rest_ke_tol=1e-4`, `rest_hold=200` | runout 分位 | `p995`（主）/`max`（尾） |

三个臂（`mu_r` 单位无量纲）：

| 臂 | `rolling_lever` | `mu_r` | 物理含义 |
|---|---|---|---|
| `legacy_mu0.5` | `particle_radius`（r） | 0.5 | 本仓认证构型（冻结基线） |
| `oss_naive_mu0.5` | `reduced_radius`（对-对 r/2） | 0.5 | 把 OSS 参数**直接抄进来**、不换算 |
| `oss_mapped_mu1.0` | `reduced_radius` | 1.0 | 按第一轮换算律 `2 × 0.5` 施加 |

## ② 怎么做的

1. 冻结脚本 `benchmarks/xval_dem_round2_column_arms.py`（第一轮已入仓、参数未改）定义三臂；
   为了避开单次 shell 切片 120 s 的回收，改用 `…_arm.py` 逐臂一进程跑（每臂 ≈50 s），
   参数与臂表 `import` 自冻结脚本，**不重打**任何数字。
2. 每臂写 `arm_<name>.json`；合并成 `column_arms.json`（与原驱动同样的 schema，
   `runout_rel_diff_vs_legacy` 由合并器从三臂实测值算出）。
3. `…_report.py` 从 JSON 生成表、`verdict.json` 与两张图（**零手抄**：本文所有数字
   都出自这两个 JSON）。
4. 解析面板（fig2）来自 `granular3d_core.Material3D` 的杠杆/帽算术：
   对-对 `mu_r Fn reff` 在映射下**恒等**，`kr = kt reff²` 在映射下**变为 1/4**；
   粒-壁 `reff = r`（两约定相同）⇒ 帽**翻倍**、`kr` 不变。5 条 property 测试钉住这套算术。
5. 集群队列 `cj1753`（`run_cluster`, repo=TensorDEM ref=main）同题递交，因排队 304 未在
   本单时间窗内派发；本单数字全部为本机逐臂串行产出（`elapsed_s` 记录在 JSON 内），
   **不依赖**该作业。

## ③ 计算结果（表：`artifacts/xval_dem_round2_column_arms/table_macro.md`，由脚本生成）

| arm | lever | mu_r | runout r∞/ri | err vs Lube (2.240) | diff vs legacy | in ±15% |
|---|---|---|---|---|---|---|
| `legacy_mu0.5` | particle_radius | 0.50 | 2.1060 | -5.98 % | +0.00 % | yes |
| `oss_naive_mu0.5` | reduced_radius | 0.50 | 2.4193 | +8.01 % | **+14.87 %** | yes |
| `oss_mapped_mu1.0` | reduced_radius | 1.00 | 2.1118 | -5.72 % | **+0.27 %** | yes |

配套量（同 JSON）：

| arm | `t_reach` [s] | runout_max（尾） | elapsed [s] | n |
|---|---|---|---|---|
| legacy | 0.7843 | 2.4416 | 49.3 | 1955 |
| oss_naive | 1.0007 | 2.5107 | 49.6 | 1955 |
| oss_mapped | 0.8693 | 2.7089 | 51.2 | 1955 |

## ④ 图

* `fig1_runout_arms.png` — 三臂 runout 柱状 + Lube 2004 a=1 参考线（2.240）与 ±15% 带 +
  认证 legacy 值（2.106）：一眼看出 `oss_naive` 把结果推到带内**另一侧**、`oss_mapped` 与
  认证线重合。
* `fig2_effective_rolling_cap.png` — 解析面板：对-对力矩帽 1.00×（映射恒等）、
  对-对 `kr` 0.25×、粒-壁帽 **2.00×**、粒-壁 `kr` 1.00×。这张图解释了 ③ 里
  `t_reach` 偏差（+10.9 %）为何大于 runout 偏差（+0.27 %）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

* **`oss_naive` 在 ±15 % 带内 —— 这是陷阱不是许可**。它相对 **Lube 文献值** 只差 +8.01 %
  （比认证基线 −5.98 % 的绝对值还小），但相对**本仓认证结果**偏 +14.87 %。也就是说：
  不换算就抄参数，会让"与文献的对标"看起来变好、而物理其实已经改了一倍（力矩帽）。
  因此本臂的判据必须是**复现认证值**，不是"越靠近文献越好"。这条与第一轮 H3 一致。
* **映射只对 Coulomb 饱和区恒等**：`kr` 变 1/4 ⇒ 弹性（未饱和）段更软。宏观后果就是
  `t_reach` +10.9 %（runout 只 +0.27 %）。若工程上用 `t_reach` 当标定量，
  换算律**不够**，必须同时改 `kt_ratio` 或接受 10 % 量级的时间偏差。
* **粒-壁不一致**：映射把壁面力矩帽翻倍（2.00×）。本算例（`mu_wall = 0.8`）runout 仍复现
  到 0.27 %，说明该量对壁面滚动帽不敏感；但**不能**由此推断所有含壁算例都不敏感。
* **单种子、单长宽比、轴对称（2D 柱坐标）**：无 3D 侧向铺展、无多 seed 统计，因此
  ±0.27 % 是"这个算例上的复现"，不是"映射精确到 0.3 %"。runout_max 尾指标在 mapped 臂
  明显更大（2.709 vs 2.442，+10.9 %），提示少量逃逸颗粒——分位数 `p995` 的结论不自动
  传到极端尾部。
* **OSS 仍未编译执行**（第一轮 F4 的边界照旧）：本单是"本仓三臂"的宏观一致性检验，
  不是与 LIGGGHTS 二进制的对照。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 本单值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| runout（`legacy_mu0.5`） | 认证值 2.106 | 2.1060 | **+0.00 %** | ✅ |
| runout（`legacy_mu0.5`） | Lube 2004 2.240 | 2.1060 | −5.98 % | ❌（> 3 %，但 < 15 %，与第一轮认证一致；口径：目标律按名义 a、分母按实测 a，见 `AUDIT_COLUMN_COLLAPSE_TIME_LAW_m17534.md`） |
| runout（`oss_mapped_mu1.0`） | 认证值 2.106 | 2.1118 | **+0.27 %** | ✅ |
| runout（`oss_mapped_mu1.0`） | Lube 2004 2.240 | 2.1118 | −5.72 % | ❌（同口径说明） |
| runout（`oss_naive_mu0.5`） | 认证值 2.106 | 2.4193 | +14.87 % | ❌（仍在 owner 15 % 带内，但已近边缘） |
| `t_reach`（`oss_mapped`） | legacy 0.7843 s | 0.8693 s | +10.85 % | ❌（映射不保弹簧刚度，见 ⑤） |
| 对-对力矩帽（解析） | legacy | mapped | 0.00 % | ✅ |
| 对-对 `kr`（解析） | legacy | mapped | −75.00 % | ❌（定义域差异，非误差） |

**结论（预登记口径）**：§⑥ 的悬案答案是 **"是"**——按 `mu_r,OSS = 2 mu_r,TD` 换算后，
柱体 runout 复现认证值到 **+0.27 %**（< 3 %）；而**不换算**直接把 OSS 参数抄进来会把
runout 推离认证值 **+14.87 %**。第一轮 D2/D3 的"可选开关 + 换算因子"处置，
在宏观层面得到独立支持；同时新增一条边界：**换算律不覆盖 `kr`（弹性段）与粒-壁帽**，
以 `t_reach` 为标定量时不能只换 `mu_r`。