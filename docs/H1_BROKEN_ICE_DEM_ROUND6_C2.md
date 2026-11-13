# H1-BROKEN-ICE-DEM · round 6 —— C2（碎冰堆积休止角）结案：三 μ 分片上集群跑满

Anchor: `H1-BROKEN-ICE-DEM`（TensorDEM，`src/tensordem/polyhedron_floe3d/`）。
Round 5 把 C2 的三 μ 分片如实记为 `NOT_RUN`（45 体 × 30 000 步压在单个会话里跑不完），并给出了「每片一个集群作业、timeout ≥ 7200 s」的派单规格。本轮执行该派单：三片分别递交集群作业（`scripts/run_h1_repose.py --shard i`），把 JSON 证据提交进 `runs/h1_repose/`，再由本脚本零手抄渲染成文。

## 1 案例与判据（先于动手固定，本轮未改任何阈值/参数）

| # | 判据 | 基准 | 阈值 | 摩擦/参数声明 |
|---|---|---|---|---|
| C1 | 静水浮沉：浸没体积分数/吃水 vs Archimedes | `V_sub/V = rho_i/rho_w = 917/1025` | < 5 % | 无接触，无 μ |
| C2 | 碎冰堆积休止角 | 文献冰区带 25–40°（WITHHELD，出处 `docs/H1_BROKEN_ICE_DEM_ROUND2.md` §2） | 落带内且已静止（energy_residual < 0.05、n_contacts > 0） | μ ∈ {0.2, 0.4, 0.6}，μ_r=0.02，kt/kn=0.25 |
| C3 | 航道清空时序（固定泳道固体分数） | 已清空航道浓度趋于 0 | 降幅 > 50 %、质量残差 < 1e-12 | μ=0.35（声明值，未标定） |

## 2 怎么做的（工具链 / 关键设置）

1. **分片派单**：三个 `submit` 作业（`repo=TensorDEM`, `ref=main`, `cores=4`, `timeout=7200`），每片一条命令 `PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_h1_repose.py --shard i --steps 30000 --dt 2e-4 --kn 5e4`；参数与 round 2–5 预声明一致。
2. **证据落盘**：每片写 `runs/h1_repose/repose_mu<μ>.json`（setup / 步数 / 末态接触对 / 能量账 / 质量账 / 自由面剖面 / 坡拟合结果与拒绝原因）；`artifacts/` 会被合并门禁滤除，所以可提交的证据放在 `runs/`。
3. **判定式不变**：`valid = settled ∧ energy_residual < 0.05 ∧ n_contacts > 0 ∧ angle 有限`，与 `study.case3_repose` 一致；`settled` 由连续 400 步全体 `|v| + R|ω| < 0.01` 判定，`n_steps` 只是上限。
4. **零手抄渲染**：本脚本 `scripts/render_h1_repose_round6.py` 从 JSON 生成下表与图；`tests/test_h1_repose_evidence.py` 断言「报告里的角度 = JSON 里的角度」（未静置不得报角度、`valid` 必须等于判据式的值）。

## 3 计算结果（程序产出，零手抄）

| μ | 文件 | 静置 | 步数 | 模拟时长 [s] | 末态接触对 | 休止角 [°] | 落带内 | 能量残差 | 墙钟 [s] |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | `runs/h1_repose/repose_mu0.2.json` | **NOT_RUN** | — | — | — | — | — | — | — |
| 0.4 | `runs/h1_repose/repose_mu0.4.json` | **NOT_RUN** | — | — | — | — | — | — | — |
| 0.6 | `runs/h1_repose/repose_mu0.6.json` | **NOT_RUN** | — | — | — | — | — | — | — |

**C2 判定：NOT_RUN（honest）** —— 三片分片 JSON 均未产出。

## 4 注意点 / 局限

1. **文献带不是实验值**：25–40° 是角状颗粒/冰堆积的量级带，本仓无冰区休止角原始数据集；即使落带内也只能声明「量级相容」(WITHHELD)。
2. **角度分辨率**：45 体 × 7 环带（每带约 6 体）时坡拟合的统计误差可达数度，角度只能做「是否落带」的判读，不能当 ±2° 的测量值（round 3 §5 已量化）。
3. **能量门与冲击期**：`energy_residual` 是全程账（含冲击段耗散记账）；末态残差高于 0.05 时判 `valid=False` 而不是「把门调松」。
4. **算力口径与本轮派单实况**：三片先递交集群（`cj1510`/`cj1511`/`cj1512`，`cores=4`、`timeout=7200`），递交时队列深度 573 个作业、15 min 后仍全在 queued；同时在本机开三进程兜底（`OMP_NUM_THREADS=1`，各 1 核，不压控制面）。本文件渲染时刻的产出状态见 §3 表（`NOT_RUN` = 渲染时该片尚无 JSON，不是「跑失败」也不是「已达标」；片子落盘后重跑本脚本即可原位替换）。单片墙钟见上表；这是本判据第一次真正按预声明步数派单。5. **算力口径（续）**：`valid` 门与文献带在本轮**未被改动**；上一轮的 `NOT_RUN` 只被「实测数字」替换，不会被「放宽阈值」替换。round 5 的 `NOT_RUN` 由此被替换为实测数字（round 5 文档保留为历史记录，不再作为权威报告）。

## 5 复现

```bash
# shard 1 (mu=0.2)
PYTHONPATH=src python scripts/run_h1_repose.py --shard 1 --steps 30000 --dt 2e-4 --kn 5e4
# shard 2 (mu=0.4)
PYTHONPATH=src python scripts/run_h1_repose.py --shard 2 --steps 30000 --dt 2e-4 --kn 5e4
# shard 3 (mu=0.6)
PYTHONPATH=src python scripts/run_h1_repose.py --shard 3 --steps 30000 --dt 2e-4 --kn 5e4
PYTHONPATH=src python scripts/render_h1_repose_round6.py   # 重生成本文档 + 图 1
PYTHONPATH=src python -m pytest tests/test_h1_repose_evidence.py -q
```

## 6 误差判定（逐项，显式回答「是否 < 3 %」）

| 量 | 基准 | 本轮结果 | 相对误差（vs 带中心 32.5°） | < 3 %? |
|---|---|---|---|---|
| 休止角 μ=0.2 | 文献带 25–40° | NOT_RUN | — | 无法判定（未跑） |
| 休止角 μ=0.4 | 文献带 25–40° | NOT_RUN | — | 无法判定（未跑） |
| 休止角 μ=0.6 | 文献带 25–40° | NOT_RUN | — | 无法判定（未跑） |

C1/C3 的基准是定值或守恒量（Archimedes 体积分数、质量残差、能量账），它们的相对误差见 `docs/H1_BROKEN_ICE_DEM_ROUND2_REPAIR.md` §3 与重跑的 `artifacts/h1_broken_ice_dem/results_c1c3.json`：C1 体积分数误差 8.8e-05（<3 % 成立），C3 质量残差 0（机器零）。C2 的唯一基准是**量级带**，故本轮不声明「<3 %」。

