# H1-BROKEN-ICE-DEM · 第 3 轮交付物落地 + 断点可容许性修复（m18018 / 原 PR#1387）

锚点：`H1-BROKEN-ICE-DEM`（C2 休止角，`in_progress`）。上游：`docs/H1_BROKEN_ICE_DEM_ROUND3.md`。

## 1. 为什么还要重做这一轮

原 PR #1387（m15203）已由 owner 裁定关闭，但其**代码交付物其实已在 main**：`pile_surface.py`
（`bin_free_surface` / `fit_flank` / BIC 分段 + Theil–Sen）、`study._pile_metrics` 的接入、
`integration_callers.run_h1_broken_ice_dem` / `check_h1_maturity`、`tests/test_h1_repose_estimator.py`。
机器复核（本单）找到两处**真实缺口**，都不在代码文本层面：

| 缺口 | 事实 | 后果 |
|---|---|---|
| 证据不在树内 | 第 3 轮 §1 的表格由 `benchmarks/.../make_figures_round3.py` 写到 `artifacts/`，该目录被合并门禁滤除 | 从 checkout 无法复现「最坏 7.24°/2.30°」；表格与代码漂移无人发现 |
| 断点搜索无结构约束 | BIC 断点可把**更陡**的外段当「裙」丢弃，报出偏小的内段斜率 | 干净 40° 锥体被报 **35.10°**；13 格合成基准中 **9 格**因隆起门拒答（干净锥体被判「没有堆」） |

## 2. 做了什么

1. **证据入树（零手抄链）**
   `scripts/render_h1_round3_evidence.py` → 调用同一生成器（`make_figures_round3.py`）→
   `runs/h1_broken_ice_dem/round3/round3_figures.json` + `fig4_estimator_ground_truth.png`，并写入
   provenance（生成器 sha256 `022165a3…`、估计器 sha256、commit、UTC 时间、重跑命令）。
   同一次运行还记录 **pre-fix 消融列**（`fig1_pre_fix_ablation`，`require_flatter_apron=False`，同一批 profile）。
2. **断点可容许性（`pile_surface._two_segment_break`）** 两条结构判据：
   * **裙不得比坡陡**：裙是压平的脚；外段比内段更陡 → 那是同一锥体的凸性，不是裙，该断点不可容许；
   * **裙须由 ≥2 个环带分辨**：单环带是一个点，无法被证明是脚，且会截断干净锥体的坡。
   `fit_flank(..., require_flatter_apron=False)` 保留修复前行为（测试基线）；默认 `min_apron_bins=2`。
3. **测试**：`tests/test_h1_round3_estimator_evidence.py`（6 项，读入树工件 + 活体重算）与
   `tests/test_h1_repose_estimator.py` 新增 4 项（3 个参数化 + 结构签名检查）。

## 3. 结果（同一批固定 seed 合成堆，13 个真值 15–45°/2.5°，n=400）

| 真值 | 原拟合（无裙） | 原拟合（40% 裙） | BIC 坡拟合（无裙，**修复前**） | BIC 坡拟合（无裙，**修复后**） | BIC 坡拟合（带裙，修复前后**相同**） |
|---|---|---|---|---|---|
| 30.0° | 28.67° | 24.09° | 拒答（nan） | 29.17° | 28.05° |
| 40.0° | 38.47° | 33.02° | **35.10°** | 39.05° | 37.75° |
| 45.0° | 43.44° | 37.76° | 39.95° | 44.03° | 42.70° |

| 量 | 修复前 | 修复后 | 判据 |
|---|---|---|---|
| 无裙最坏误差（13 真值） | **5.05°** | **0.97°** | 测试断言 ≤1.5° |
| 无裙平均误差 | 4.93° | 0.79° | — |
| 无裙拒答格数 | **9 / 13** | **0 / 13** | 测试断言 =0 |
| 带裙最坏误差 | 2.30° | 2.30°（逐位相同） | 测试断言差异 atol=1e-12 |
| 带裙最大误差 vs 原拟合（7.24°） | — | 2.30° | 与第 2 轮同口径 |

所有数字来自 `runs/h1_broken_ice_dem/round3/round3_figures.json`，测试逐格核对，零手抄。

## 4. 复现

```bash
PYTHONPATH=src python scripts/render_h1_round3_evidence.py     # 重生成工件（~10 s）
PYTHONPATH=src python -m pytest tests/test_h1_round3_estimator_evidence.py -q   # 6 项
PYTHONPATH=src python -m pytest tests/test_h1_repose_estimator.py -q            # 15 项
```

## 5. 注意点 / 局限（如实）

1. **未改物理参数**，只改量测与量测的证据链；不是对 C2 物理结论的背书。
2. **C2 的 DEM 分片证据不在树内**（`repose_shard_*.json` 属集群工件），因此本修复对物理判读的影响**未测**；
   树内 `case3_repose(18 体/400 步)` demo 在两种设置下同为「无堆、0 接触」，无区分力。`fig2`
   （DEM 休止角 vs μ）因此仍为 `null`——工件里显式记为 `fig2: null`，不用空图充数。
3. **15° 附近仍有量测下限**：n=45 时 4 格因隆起 < 一个粒径拒答（第 3 轮 §5 第 2 条同结论）；n=120 时最坏 1.79°。
4. **文献带仍 WITHHELD**：25–40° 只是冰堆积量级带，本仓无冰区休止角原始数据集。
5. **重叠声明**：`runs/h1_broken_ice_dem/results_c1c3.json`（C1/C3 证据）与
   `tests/test_h1_broken_ice_dem.py` 的 3 个 skip 属另一条在开分支（ROUND7 gate，m17381）的范围，本单**不碰**；
   本单工件全部放在 `runs/h1_broken_ice_dem/round3/` 子目录，无文件级冲突。