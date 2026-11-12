# H1-BROKEN-ICE-DEM · 第 3 轮：休止角**量测**修复（C2 的判据侧欠账）

锚点：`H1-BROKEN-ICE-DEM`（in_progress，票据见 `feature_ledger`）。
上游：`docs/H1_BROKEN_ICE_DEM_ROUND2.md`（第 2 轮实施）、板审计 `docs/AUDIT_M14948_H1_BROKEN_ICE_DEM.md`（m14948/m15029，
`request changes`：C1 PASS / **C2 honest-FAIL（0–16.27°，未静止）** / C3 demo 级；另有红测试 `energy_residual 212.08` 与
run_artifacts 未提交）。

本轮**不调物理参数**。第 2 轮把 C2 判为 FAIL 时用的数字是 `0–16.27°`，本轮先问一个更前置的问题：
**这个数字本身可不可信？** 结论（下节，合成基准可复核）：**不可信**——原拟合把「坡面」与「堆积裙」平均在一起，
对一个**已知 30° 的锥体**只报 24.1°（带裙）甚至 16.1°（裙更厚），误差 6–14° **与物理无关**。
所以 C2 的 FAIL 在此之前是**测量欠账**，不能用来接受或否定物理。修好量测后重跑 DEM 分片。

---

## 1. 缺陷：自由面拟合把「坡」和「裙」一起平均

原实现 `study._pile_metrics` 对所有环带（annulus）做**一条**最小二乘直线，取其斜率。真实堆积体（倾倒/坍塌成型）
在径向上恒有三段：

| 区域 | 形态 | 该不该进休止角 |
|---|---|---|
| 顶/平台 | `h ≈ const` | 否 |
| **坡面（flank）** | `h = H − (r−r₀)tanα` | **是，休止角即此段** |
| 裙/脚（apron） | `h ≈ 一个粒径` | 否 |

一条直线穿过三段 = 坡角被平坦裙拉低。合成基准实测（`docs` 表格与 `artifacts/h1_broken_ice_dem/fig4_estimator_ground_truth.png`）：

| 真值 | 原拟合（无裙） | 原拟合（40% 颗粒在裙上） | 本轮 BIC 坡拟合（无裙） | 本轮（带裙） |
|---|---|---|---|---|
| 30.0° | 28.67° | **24.09°** | 30.0° | 28.05° |
| 40.0° | 38.47° | **33.02°** | 40.0° | 37.75° |
| 45.0° | 43.44° | **37.76°** | 40.0°* | 42.70° |

\* `*` = 该格 `angle_deg` 受「隆起高度须 ≥ 一个粒径」的门（见 §3）影响时改读 `provisional_angle_deg`；表中数值均由
`benchmarks/h1_broken_ice_dem/make_figures_round3.py` 从固定 seed 合成堆**程序化产出**（`artifacts/h1_broken_ice_dem/round3_figures.json`），零手抄。

13 个真值（15–45°，步长 2.5°）上的**最坏误差**：原拟合带裙 **7.24°**，本轮坡拟合带裙 **2.30°**。

## 2. 修复：BIC 分段 + Theil–Sen 稳健斜率（`pile_surface.py`）

新增 `src/tensordem/polyhedron_floe3d/pile_surface.py`：

* `bin_free_surface(...)`：环带自由面剖面（每带取 `top` 的 90 分位，与旧报告可比；`<3` 体的环带被丢弃，避免给单点算分位）。
* `_two_segment_break(...)`：对「坡段 + 裙段」分段线性模型按 **BIC**（`n ln(SSE/n) + k ln n`）与「整段单线」比选。
  BIC 的惩罚项是关键：干净的锥体不会被切成两段短线（否则会系统性低估 ~1.5°），有裙的堆才会断在坡脚。
* `fit_flank(...)`：坡段斜率用 **Theil–Sen**（成对斜率中位数），单个噪声环带不能像最小二乘那样把角度拉走；
  返回 `angle_deg / n_flank_bins / apex_radius_m / toe_radius_m / flank_height_drop_m / apron_angle_deg / apron_body_share / reason`。
  **裙不是被丢掉而是被报告**（`apron_angle_deg`、`apron_body_share`），审计者可核。
* 拒绝规则（全部结构性，**不引用 25–40° 文献带**）：
  * 无下降坡段（单层散开）= `reason = no_descending_flank`，`angle_deg = nan`。旧实现会给出一个「0° 的休止角」——
    这是把「看不见堆」误报成角度。
  * 坡段环带 < 3 = 同上拒绝（两个环带不能定斜率）。
  * 坡段隆起 `flank_height_drop_m < 一个粒径` = 拒绝（高度不及一个颗粒的不是堆）；同时把 `provisional_angle_deg` 一并写出，
    信息不丢。

接入：`study._pile_metrics` 改为调用 `pile_surface.fit_flank(..., min_drop=median(bounding_radius))`，并把
`repose_flank` / `repose_profile` / `flank_reason` 写进每个 run 的指标，`case3_repose` 的 `valid` 判据不变
（`settled ∧ energy_residual<0.05 ∧ n_contacts>0 ∧ angle 有限`）。

## 3. 稳定性：步长必须与接触刚度同变（第 2 轮复现命令的坑）

`case3_repose` 的默认 `kn=2e5`、`dt=1e-4`。审计建议「`n_steps ≥ 12000` 或 `dt=2e-4` 配 `kn ≤ 5e4`」——两者必须同变：

| 组合 | 最小体质量 | 接触周期 | 每周期步数 | 判读 |
|---|---|---|---|---|
| kn=2e5, dt=1e-4 | 0.0576 kg | 3.37 ms | **33.7** | 可用（第 2 轮用此） |
| kn=2e5, dt=2e-4 | 0.0576 kg | 3.37 ms | **16.9** | 不可用：欠分辨，冲击期能量账发散 |
| kn=5e4, dt=2e-4 | 0.0576 kg | 6.74 ms | **33.7** | 本轮采用：同分辨率、步数减半 |

数字由 `benchmarks/h1_broken_ice_dem/diagnose_repose.py` 打印的 `steps_per_contact_period` 给出（读数见 §5）。

## 4. 复现

```bash
# 量测单元测试（合成真值，11 项，<10 s）
PYTHONPATH=src python -m pytest tests/test_h1_repose_estimator.py -q
# H1 三判据回归（含 C1 浮沉、C3 航道）
PYTHONPATH=src python -m pytest tests/test_h1_broken_ice_dem.py -q
# 图（fig4 = 量测标定；fig5 = DEM 休止角 vs μ）
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/make_figures_round3.py
# DEM 分片（每个 μ 一片，集群）
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py --cases repose \
    --repose-shard 2 --repose-bodies 45 --repose-steps 6000 --repose-dt 2e-4 --repose-kn 5e4 \
    --out repose_shard_2.json
```

## 5. 注意点 / 局限

1. **本轮修的是量测，不是物理**：坡拟合让「坡角」可测，但它不能把一个散开的单层变成一个堆。若 DEM 末态仍是单层散开，
   `flank_reason = no_descending_flank` 会如实报「没有堆」，而不是像旧实现那样报 0°。
2. **浅堆有分辨率下限**：隆起不足一个粒径（≈ 本算例的 0.05 m）时拒绝给角度。合成基准里 <37.5° 的锥体在 n=400 时即触发该门
   （锥高 `R tanα` 太小），故此类算例只能读 `provisional_angle_deg`。这条门是**物理分辨率**而非调参：一个堆若还不如一个颗粒高，
   其「坡角」不确定度大于数值本身。
3. **n=45 的统计误差**：7 个环带 × 每带 ~6 体，合成基准上带裙误差可达 **3.9°**（n=400 时 0.2°，n=120 时 ~2°）。
   因此 n=45 的角度只能对「是否落在 25–40° 带内」做量级判读，不能当作 ±2° 的测量值。
4. **接触步的墙钟远高于自由飞行步**：`diagnose_repose.py` 实测 27 体自由飞行 ~9 ms/步，进入接触后升到 ~0.2 s/步
   （接触对 <10 时）。这是**性能账**（多面体 SAT + 四面体切割的每对成本），本轮未拆解，已在建议清单中列为下一切片。
5. **能量账在冲击期超门**：kn=5e4/dt=2e-4 下 `energy_residual` 在冲击段（t≈0.25–0.30 s）由 0.093 升到 0.165，
   高于 `case3_repose` 的 `0.05` 有效门。该门因此会在「冲击主导」的算例上结构性判 invalid——这是**判据门标定**问题
   （冲击期耗散项在 `_apply_pair` 内按步初速度计，O(dt) 记账误差随耗散功线性增长），不是积分发散。
   本轮**如实报告不改门**（改门=改判据，须审计席位裁定）。
6. **文献带仍是 WITHHELD**：25–40° 是冰堆积量级带，本仓无冰区休止角原始数据集；即使落带也只能说「量级相容」。