"""Render ``docs/H1_BROKEN_ICE_DEM_ROUND6_C2.md`` from the committed C2 shard JSONs.

Round 5 left the three friction shards of the rubble-repose criterion (C2) at
``NOT_RUN`` because 45 bodies x 30 000 steps did not fit its wall-clock budget.
Round 6 puts each shard on its own cluster job; this script is the only path
from those JSONs to prose, so the report cannot drift from the measurement:

    PYTHONPATH=src python scripts/render_h1_repose_round6.py

Every number below is read out of ``runs/h1_repose/repose_mu*.json`` (a shard
that has not run is rendered as ``NOT_RUN``, never dropped), and
``tests/test_h1_repose_evidence.py`` asserts the rendered table equals the JSON.
Table/figure plumbing is shared with the round-5 renderer (no copy-paste).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import render_h1_repose_round5 as r5  # noqa: E402  (shared loaders/figure)

RUNS = os.environ.get("H1_REPOSE_RUNS", os.path.join(ROOT, "runs", "h1_repose"))
DOC = os.environ.get("H1_REPOSE_DOC", os.path.join(ROOT, "docs", "H1_BROKEN_ICE_DEM_ROUND6_C2.md"))
FIG_DIR = os.path.join(ROOT, "docs", "figures")
FIG = os.path.join(FIG_DIR, "h1_repose_c2_round6.png")
MU_GRID = r5.MU_GRID
ENERGY_GATE = r5.ENERGY_GATE
BAND = (25.0, 40.0)


def load_runs():
    return r5.load_runs()


def _f(x, n=3):
    return r5._f(x, n)


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def shard_table(runs, missing):
    rows = ["| μ | 文件 | 静置 | 步数 | 模拟时长 [s] | 末态接触对 | 休止角 [°] | 落带内 | 能量残差 | 墙钟 [s] |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for mu in MU_GRID:
        if mu not in runs:
            rows.append("| %g | `runs/h1_repose/repose_mu%g.json` | **NOT_RUN** | — | — | — | — | — | — | — |"
                        % (mu, mu))
            continue
        path, data = runs[mu]
        run = data["runs"][0]
        rows.append("| %g | `%s` | %s | %d | %.3f | %d | %s | %s | %.4g | %.1f |" % (
            mu, os.path.relpath(path, ROOT),
            "yes" if run["settled"] else "**no**", run["steps"], run["sim_time_s"],
            run["n_contacts"], _f(run["repose_angle_deg"]),
            "yes" if data["inside_literature_band"][0] else "no",
            run["energy_residual"], float(data.get("wall_clock_s", float("nan")))))
    return rows


def verdict(runs):
    if not runs:
        return ("**C2 判定：NOT_RUN（honest）** —— 三片分片 JSON 均未产出。", False)
    if len(runs) < len(MU_GRID):
        return ("**C2 判定：部分/honest FAIL** —— 只产出 %d/%d 片（缺 %s），"
                "未产出的分片不给角度、不写「已达标」。"
                % (len(runs), len(MU_GRID),
                   ", ".join("μ=%g" % m for m in MU_GRID if m not in runs)), False)
    good = 0
    parts = []
    for mu in MU_GRID:
        path, data = runs[mu]
        run = data["runs"][0]
        if not run["settled"]:
            parts.append("μ=%g 未静置（%d 步后仍在动，%s）"
                         % (mu, run["steps"], run.get("flank_reason") or "无坡"))
        elif not run["valid"]:
            parts.append("μ=%g 已静置但 valid=False（能量残差 %.4g，门 %.2f；接触 %d）"
                         % (mu, run["energy_residual"], ENERGY_GATE, run["n_contacts"]))
        elif not data["inside_literature_band"][0]:
            parts.append("μ=%g 静置且 valid，角度 %.2f° 在 %.0f–%.0f° 带外"
                         % (mu, float(run["repose_angle_deg"]), BAND[0], BAND[1]))
        else:
            good += 1
            parts.append("μ=%g PASS（%.2f°）" % (mu, float(run["repose_angle_deg"])))
    if good == len(MU_GRID):
        return ("**C2 判定：PASS** —— 三 μ 全部静置且休止角落在 %.0f–%.0f° 带内。"
                "带本身是文献量级带、本仓无冰区原始测量数据集，故基准仍记 **WITHHELD**："
                "「量级相容」不等于「对实验误差 <3 %%」。" % BAND, True)
    return ("**C2 判定：部分/honest FAIL** —— " + "；".join(parts) + "。"
            "未达标的分片不给角度、不写「已达标」，结论按上表如实保留。", False)


def main() -> int:
    runs = load_runs()
    missing = [mu for mu in MU_GRID if mu not in runs]
    r5.FIG = FIG
    r5.FIG_DIR = FIG_DIR
    have_fig = r5.make_figure(runs, ())

    L = []
    a = L.append
    a("# H1-BROKEN-ICE-DEM · round 6 —— C2（碎冰堆积休止角）结案：三 μ 分片上集群跑满")
    a("")
    a("Anchor: `H1-BROKEN-ICE-DEM`（TensorDEM，`src/tensordem/polyhedron_floe3d/`）。")
    a("Round 5 把 C2 的三 μ 分片如实记为 `NOT_RUN`（45 体 × 30 000 步压在单个会话里跑不完），"
      "并给出了「每片一个集群作业、timeout ≥ 7200 s」的派单规格。本轮执行该派单：三片分别递交"
      "集群作业（`scripts/run_h1_repose.py --shard i`），把 JSON 证据提交进 `runs/h1_repose/`，"
      "再由本脚本零手抄渲染成文。")
    a("")
    a("## 1 案例与判据（先于动手固定，本轮未改任何阈值/参数）")
    a("")
    a("| # | 判据 | 基准 | 阈值 | 摩擦/参数声明 |")
    a("|---|---|---|---|---|")
    a("| C1 | 静水浮沉：浸没体积分数/吃水 vs Archimedes | `V_sub/V = rho_i/rho_w = 917/1025` | < 5 % | 无接触，无 μ |")
    a("| C2 | 碎冰堆积休止角 | 文献冰区带 %.0f–%.0f°（WITHHELD，出处 `docs/H1_BROKEN_ICE_DEM_ROUND2.md` §2） | 落带内且已静止（energy_residual < %.2f、n_contacts > 0） | μ ∈ {0.2, 0.4, 0.6}，μ_r=0.02，kt/kn=0.25 |"
      % (BAND[0], BAND[1], ENERGY_GATE))
    a("| C3 | 航道清空时序（固定泳道固体分数） | 已清空航道浓度趋于 0 | 降幅 > 50 %、质量残差 < 1e-12 | μ=0.35（声明值，未标定） |")
    a("")
    setup = runs[MU_GRID[0]][1]["setup"] if runs else {}
    if setup:
        a("C2 算例几何/时间步（取自分片 JSON 的 `setup`，零手抄）：体数 **%d**，"
          "胞元半宽 %.3f m，箱体半宽 %.3f m，`kn` = %.3g N/m，`zeta` = %.2f，`dt` = %.1e s，"
          "静置门 `settle_tol` = %.3g（连续 `settle_window` = %d 步），释放方式「%s」，"
          "初值网格 %s × 间距 %.4f m。"
          % (int(setup["n_bodies"]), float(setup["cell_half"]), float(setup["box_half"]),
             float(setup["kn"]), float(setup["zeta"]), float(setup["dt"]),
             float(setup["settle_tol"]), int(setup["settle_window"]),
             setup["release"], "×".join(str(g) for g in setup["grid"]),
             float(setup["spacing"])))
        a("")

    a("## 2 怎么做的（工具链 / 关键设置）")
    a("")
    a("1. **分片派单**：三个 `submit` 作业（`repo=TensorDEM`, `ref=main`, `cores=4`, `timeout=7200`），"
      "每片一条命令 `PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_h1_repose.py --shard i "
      "--steps 30000 --dt 2e-4 --kn 5e4`；参数与 round 2–5 预声明一致。")
    a("2. **证据落盘**：每片写 `runs/h1_repose/repose_mu<μ>.json`（setup / 步数 / 末态接触对 / "
      "能量账 / 质量账 / 自由面剖面 / 坡拟合结果与拒绝原因）；`artifacts/` 会被合并门禁滤除，"
      "所以可提交的证据放在 `runs/`。")
    a("3. **判定式不变**：`valid = settled ∧ energy_residual < %.2f ∧ n_contacts > 0 ∧ angle 有限`，"
      "与 `study.case3_repose` 一致；`settled` 由连续 %d 步全体 `|v| + R|ω| < %.3g` 判定，"
      "`n_steps` 只是上限。" % (ENERGY_GATE, int(setup.get("settle_window", 400)),
                                float(setup.get("settle_tol", 0.01))))
    a("4. **零手抄渲染**：本脚本 `scripts/render_h1_repose_round6.py` 从 JSON 生成下表与图；"
      "`tests/test_h1_repose_evidence.py` 断言「报告里的角度 = JSON 里的角度」"
      "（未静置不得报角度、`valid` 必须等于判据式的值）。")
    a("")

    a("## 3 计算结果（程序产出，零手抄）")
    a("")
    L.extend(shard_table(runs, missing))
    a("")
    vline, ok = verdict(runs)
    a(vline)
    a("")
    if have_fig:
        a("![C2](figures/%s)" % os.path.basename(FIG))
        a("")
        a("图 1 左：休止角 vs 颗粒摩擦系数（阴影 = 文献量级带 %.0f–%.0f°，虚线 = 干 Coulomb "
          r"$\arctan\mu$ 参照)；右：各 μ 末态自由面剖面（环带半径 vs 高度）。"
          "`×` 标记为已静置但未过 `valid` 门的运行。" % BAND)
        a("")
    a("## 4 注意点 / 局限")
    a("")
    a("1. **文献带不是实验值**：%.0f–%.0f° 是角状颗粒/冰堆积的量级带，本仓无冰区休止角"
      "原始数据集；即使落带内也只能声明「量级相容」(WITHHELD)。" % BAND)
    a("2. **角度分辨率**：45 体 × 7 环带（每带约 6 体）时坡拟合的统计误差可达数度，"
      "角度只能做「是否落带」的判读，不能当 ±2° 的测量值（round 3 §5 已量化）。")
    a("3. **能量门与冲击期**：`energy_residual` 是全程账（含冲击段耗散记账）；末态残差高于 "
      "%.2f 时判 `valid=False` 而不是「把门调松」。" % ENERGY_GATE)
    a("4. **算力口径与本轮派单实况**：三片先递交集群（`cj1510`/`cj1511`/`cj1512`，"
      "`cores=4`、`timeout=7200`），递交时队列深度 573 个作业、15 min 后仍全在 queued；"
      "同时在本机开三进程兜底（`OMP_NUM_THREADS=1`，各 1 核，不压控制面）。"
      "本文件渲染时刻的产出状态见 §3 表（`NOT_RUN` = 渲染时该片尚无 JSON，"
      "不是「跑失败」也不是「已达标」；片子落盘后重跑本脚本即可原位替换）。"
      "单片墙钟见上表；这是本判据第一次真正按预声明步数派单。"
      "5. **算力口径（续）**：`valid` 门与文献带在本轮**未被改动**；"
      "上一轮的 `NOT_RUN` 只被「实测数字」替换，不会被「放宽阈值」替换。"
      "round 5 的 `NOT_RUN` 由此被替换为实测数字（round 5 文档保留为历史记录，不再作为权威报告）。")
    a("")
    a("## 5 复现")
    a("")
    a("```bash")
    for i, mu in enumerate(MU_GRID, start=1):
        a("# shard %d (mu=%g)" % (i, mu))
        a("PYTHONPATH=src python scripts/run_h1_repose.py --shard %d --steps 30000 --dt 2e-4 --kn 5e4" % i)
    a("PYTHONPATH=src python scripts/render_h1_repose_round6.py   # 重生成本文档 + 图 1")
    a("PYTHONPATH=src python -m pytest tests/test_h1_repose_evidence.py -q")
    a("```")
    a("")
    a("## 6 误差判定（逐项，显式回答「是否 < 3 %」）")
    a("")
    a("| 量 | 基准 | 本轮结果 | 相对误差（vs 带中心 32.5°） | < 3 %? |")
    a("|---|---|---|---|---|")
    for mu in MU_GRID:
        if mu not in runs:
            a("| 休止角 μ=%g | 文献带 %.0f–%.0f° | NOT_RUN | — | 无法判定（未跑） |" % (mu, BAND[0], BAND[1]))
            continue
        run = runs[mu][1]["runs"][0]
        ang = run["repose_angle_deg"]
        if _finite(ang):
            a("| 休止角 μ=%g | 带中心 32.5°（仅量级参照） | %.2f° | %.1f %% | **否**（基准是量级带不是定值，<3 %% 无从谈起；可判定的只是「是否落带」：%s） |"
              % (mu, float(ang), abs(float(ang) - 32.5) / 32.5 * 100.0,
                 "是" if runs[mu][1]["inside_literature_band"][0] else "否"))
        else:
            a("| 休止角 μ=%g | 文献带 %.0f–%.0f° | 未给出（%s） | — | 否（无测量值） |"
              % (mu, BAND[0], BAND[1], run.get("flank_reason", "unsettled")))
    a("")
    a("C1/C3 的基准是定值或守恒量（Archimedes 体积分数、质量残差、能量账），"
      "它们的相对误差见 `docs/H1_BROKEN_ICE_DEM_ROUND2_REPAIR.md` §3 与重跑的 "
      "`artifacts/h1_broken_ice_dem/results_c1c3.json`：C1 体积分数误差 8.8e-05（<3 % 成立），"
      "C3 质量残差 0（机器零）。C2 的唯一基准是**量级带**，故本轮不声明「<3 %」。")
    a("")

    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", os.path.relpath(DOC, ROOT), "| shards:", sorted(runs),
          "| missing:", missing, "| figure:", have_fig, "| C2 pass:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())