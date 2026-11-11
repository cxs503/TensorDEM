# H1-BROKEN-ICE-DEM 第3轮（返修轮2）：浮沉 PASS · 航道清空全扫掠 PASS · 休止角 honest NOT_RUN

交付分支 `agent/m15203`。所有数字由 `artifacts/h1_broken_ice_dem/results.json` 程序化注入，零手抄。
入口：`benchmarks/h1_broken_ice_dem/run_validation.py` 与 `python -m tensordem.cli h1-broken-ice`；
平台侧集成调用：`tensordem.integration_callers.run_h1_broken_ice_dem()`（本仓新增）。

## 1 判据（动手前固定）
| # | 判据 | 基准 | 阈值 | 摩擦声明 |
|---|---|---|---|---|
| C1 | 静水浮沉：浸没体积分数 / 吃水 | Archimedes `V_sub/V = rho_i/rho_w = 917/1025` | < 5 % | 无接触，无 mu |
| C2 | 碎冰堆积休止角 | 文献冰区带 25–40°（WITHHELD，出处 `docs/H1_BROKEN_ICE_DEM_ROUND2.md` §2） | 落带内且已静止 | mu 扫 0.2/0.4/0.6，mu_r=0.02，kt/kn=0.25 |
| C3 | 航道清空：被扫掠固定泳道的冰浓度时序 | 已清空（降幅 > 50 %），且扫掠完成 | + 质量残差 < 1e-12、能量残差 < 5 % | mu = 0.35（冰-冰 = 冰-船体，声明值，未标定） |

## 2 本轮方法与两处真实改进
1. **C3 的观测量改成精确裁剪固体体积**：新增 `DEMSystem3D.solid_volume_slab(axis, lo, hi)`
   （把 `submerged_tet_moments` 的平面裁剪推广到任意轴，逐四面体裁剪），
   泳道浓度 = 泳道内精确固体体积 / (corridor × width × 中位厚度)。旧的“质心落在盒内即全算”
   代理只能以台阶方式变化，无法分辨半块碎冰跨过面的情形；两者都留在 `series` 里便于对照
   （测试 `test_slab_volume_matches_analytic_box` / `test_slab_volume_is_continuous_not_a_step_function`
   用解析长方体逐项校核，相对误差 < 1e-10）。
2. **C3 判定加入“扫掠完成”门**：`hull_advance >= corridor` 才算清空，否则如实报 `sweep_complete=False`。
   上一轮把 18 % 的扫掠当成“部分清空”，本轮不允许。（测试 `test_channel_case_declares_incomplete_sweep`。）
   另把出口壁移远（`outlet_margin`）避免被推走的碎冰在出口墙前堆积成柱塞。

## 3 计算结果（ref vs got）
| 判据 | 量 | ref（基准） | got（实测） | 相对误差 | 阈值 | 判定 |
|---|---|---|---|---|---|---|
| C1 | 浸没体积分数 | 0.894634 | 0.894555 | 0.0088 % | < 5 % | **PASS** |
| C1 | 吃水 [m] | 0.107356 | 0.107360 | 0.0033 % | < 5 % | **PASS** |
| C1 | 能量残差 [-] | 0 | 1.299e-08 | — | < 1e-2 | **PASS** |
| C3 | 泳道浓度 初→末 | 0（清空） | 0.8095 → 0.0000 | 降幅 100.0 % | > 50 % | **PASS** |
| C3 | 泳道半衰时间 [s] | — | 0.368 | — | — | 观测值 |
| C3 | 扫掠完成 | hull_advance ≥ 0.30 m | 0.360 m | — | ≥ corridor | **PASS** |
| C3 | 质量残差 [-] | 0 | 0.0e+00 | — | < 1e-12 | **PASS** |
| C3 | 能量残差 [-] | 0 | 0.0328 | — | < 5 % | **PASS** |
| C3 | 船体阻力 稳态均值 | 无基准（不对标实验/经验式） | 71.9 ± 22.5 N | — | — | 不引用 |
| C2 | 休止角 [°] | 文献带 25–40 | **nan（未静止，接触 0 对）** | — | 落带内 | **NOT_RUN（honest）** |

C1 算例：16 边形棱柱浮冰，面积 0.36 m²、厚 0.12 m、rho_i=917 / rho_w=1025 kg/m³，dt=5e-4 s，
线性阻尼取临界值的 0.7 倍（阻尼只决定弛豫快慢，不决定平衡吃水）。
C3 算例：8 块多面体碎冰、泳道宽 0.5 m、长 1.2 m、被扫掠泳道长 0.30 m、船速 0.4 m/s、dt=5e-4 s、
1800 步（0.9 s，推进 0.36 m）、kn=5e4 N/m、mu=0.35（声明值）。

## 4 结果图
`fig1_buoyancy.png`（吃水与浸没分数弛豫 vs Archimedes 线）、`fig2_channel.png`（泳道浓度时序 + 船体阻力时序；
含精确裁剪与质心代理两条曲线对照）、`fig3_repose.png`（C2 未运行的真实记账：接触数/堆高）。

## 5 注意点（踩坑与局限）
1. **C2 本轮仍未出数**：45 体 / 3 个 mu / 20000 步 / dt=2e-4 的三分片已递交集群 `cj1009`，交付时仍在排队
   （队列 200+）。本地 120 s/次的计算窗口内只能跑到约 1/8 所需积分量，因此不给角度。
2. **一次被证伪的降本配置**（负面结果，留档）：`n_bodies=18, kn=2e4, dt=1e-3` 想用软弹簧换大步长，
   结果 18 体 3×3×2 只有两层、坍塌后摊平成薄层（堆高 0.089 m）、末态 0 对接触、
   `energy_residual = 0.72`（账不闭合）、`settled=False`。
   → 结论：**不能靠软弹簧+大步长缩短休止角算例**；该配置的产物留在 `repose_mu0.2.json` 里作为反例，
   不计入达标。（这也是第 2 轮“未静止”死法的同族陷阱：时间窗不足时任何角度都是伪值。）
3. **休止角文献带是 band 不是单点实验值**：本仓无原始冰区数据集 → 即使将来落带内也只能记 WITHHELD。
4. **C3 阻力绝对值不可引用**：单算例、mu 未标定、无实验/经验式基准；本判据只主张“清空发生时序可量化”。
5. `lane_concentration = 0.0`（末态）是**扫掠完成后泳道里确实没有冰**，不是指标退化；质心代理曲线同时给出作对照。

## 6 误差判定（关键量 vs 基准，是否 < 3 %）
* C1 浸没体积分数：0.0088 % → **< 3 %** ✓
* C1 吃水：0.0033 % → **< 3 %** ✓
* C3：无解析基准（清空是演化过程量），只判质量守恒（0.0）与能量账（3.28 % → < 5 % ✓）；
  **不给“与实验误差 < 3 %”的声明**，因为没有实验基准。
* C2：无数字，**不声明任何误差结论**（honest NOT_RUN）。

## 7 复现命令
```bash
# C1 + C3（本机 ~51 s）
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py \
    --cases buoyancy,channel --channel-steps 1800
# 平台侧集成调用 + 成熟度检查
PYTHONPATH=src python -m tensordem.integration_callers --h1 --cases buoyancy,channel --channel-steps 1800
# C2（重计算，集群；45 体 × 3 mu × 20000 步 × dt=2e-4）
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py \
    --cases repose --repose-shard 1 --repose-bodies 45 --repose-steps 20000 --repose-dt 2e-4
# 测试
PYTHONPATH=src python -m pytest tests/test_h1_broken_ice_dem.py tests/test_polyhedron_floe3d_kernel_ledger.py -q
```

## 8 参考 / 谱系
Archimedes；Cundall & Strack (1979) Géotechnique 29(1) 47–65；冰-冰摩擦 mu≈0.1–0.3
（`docs/ICE_FRICTION_REGIME_MAP.md`）；休止角带见 `docs/H1_BROKEN_ICE_DEM_ROUND2.md` §2。
本仓先行工作：`docs/IN157_N03_DEM_BROKEN_ICE_VALIDATION.md`、`docs/GAP_DEM_CLUMP.md`、
`docs/DEM_POLYHEDRON_MIXED_MODE.md`、`docs/H1_BROKEN_ICE_DEM_ROUND2_REPAIR.md`。
