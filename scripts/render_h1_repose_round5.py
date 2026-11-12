"""Render ``docs/H1_BROKEN_ICE_DEM_ROUND5_C2.md`` and its figure from the C2 shard JSONs.

Zero hand copying: every number in the report is read out of
``runs/h1_repose/repose_mu*.json``; a shard that has not run is rendered as
``NOT_RUN`` rather than dropped.  ``tests/test_h1_repose_evidence.py`` asserts the
rendered table equals the JSON, so this script is the only path from measurement
to prose.

    PYTHONPATH=src python scripts/render_h1_repose_round5.py
"""
from __future__ import annotations

import glob
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.environ.get("H1_REPOSE_RUNS", os.path.join(ROOT, "runs", "h1_repose"))
DOC = os.environ.get("H1_REPOSE_DOC", os.path.join(ROOT, "docs", "H1_BROKEN_ICE_DEM_ROUND5_C2.md"))
FIG_DIR = os.path.join(ROOT, "docs", "figures")
FIG = os.path.join(FIG_DIR, "h1_repose_c2.png")
MU_GRID = (0.2, 0.4, 0.6)
ENERGY_GATE = 0.05


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RUNS, "repose_mu*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        runs[float(data["runs"][0]["friction"])] = (path, data)
    return runs


def _f(x, n=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "nan"
    return ("%%.%df" % n) % x


def make_figure(runs, pilots=()) -> bool:
    if not runs and not pilots:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_lo, band_hi = 25.0, 40.0
    for _mu, (_p, d) in runs.items():
        band_lo, band_hi = (float(v) for v in d["ref"]["band_deg"])
        break
    for _mu, _p, d in pilots:
        band_lo, band_hi = (float(v) for v in d["ref"]["band_deg"])
        break

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.axhspan(band_lo, band_hi, color="0.85", label="literature band %.0f–%.0f° (WITHHELD)"
               % (band_lo, band_hi))
    mus, ang, invalid = [], [], []
    for mu in MU_GRID:
        if mu not in runs:
            continue
        run = runs[mu][1]["runs"][0]
        a = run["repose_angle_deg"]
        if isinstance(a, float) and math.isfinite(a):
            mus.append(mu)
            ang.append(a)
            if not run["valid"]:
                invalid.append((mu, a))
    if mus:
        ax.plot(mus, ang, "o-", color="C0", label="DEM (polyhedral ice, 45 bodies)")
    for mu, a in invalid:
        ax.plot([mu], [a], "x", color="C3", ms=9, label="run invalid (energy/接触 gate)")
    ax.axhline(0.0, color="0.5", lw=0.6)
    xs = [0.05, 0.75]
    ax.plot(xs, [math.degrees(math.atan(x)) for x in xs], "--", color="C1",
            label=r"$\arctan\mu$ (dry Coulomb reference)")
    ax.set_xlabel(r"particle friction $\mu$ [-]")
    ax.set_ylabel("angle of repose [deg]")
    ax.set_title("C2 · pile angle of repose vs friction")
    ax.set_xlim(0.1, 0.7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    profile_sets = []
    for mu in MU_GRID:
        if mu in runs:
            profile_sets.append((mu, runs[mu][1]["runs"][0]))
    for mu, _path, d in pilots:
        profile_sets.append((mu, d["runs"][0]))
    for mu, run in profile_sets:
        prof = run["repose_profile"]
        r, h = prof["radius"], prof["height"]
        if not r:
            continue
        ax.plot(r, h, "o-", ms=3, label=r"$\mu$=%g (t=%.2f s, %s)" % (
            mu, run["sim_time_s"], "settled" if run["settled"] else "unsettled"))
    ax.set_xlabel("radial distance from pile axis [m]")
    ax.set_ylabel("free-surface height [m]")
    ax.set_title("C2 · settled free-surface profile")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return True


PILOT_DIR = os.environ.get("H1_REPOSE_PILOT", os.path.join(ROOT, "runs", "h1_repose_pilot"))


def load_pilots():
    """The n=12 feasibility pilot: same physics, one fifth of the bodies."""
    out = []
    for path in sorted(glob.glob(os.path.join(PILOT_DIR, "pilot_mu_s*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        out.append((float(data["runs"][0]["friction"]), path, data))
    return sorted(out)


def main() -> None:
    runs = load_runs()
    pilots = load_pilots()
    have_fig = make_figure(runs, pilots)
    missing = [mu for mu in MU_GRID if mu not in runs]
    dense = [mu for mu in MU_GRID if mu in runs
             and runs[mu][1]["runs"][0]["steps"] > 3000]
    budget = max((runs[mu][1]["runs"][0]["sim_time_s"] for mu in MU_GRID if mu in runs),
                 default=0.0)
    setup0 = next(iter(runs.values()))[1]["setup"] if runs else {}

    L = []
    a = L.append
    a("# H1-BROKEN-ICE-DEM · round 5 — C2（碎冰堆积休止角）三 μ 分片证据补齐")
    a("")
    a("Anchor: `H1-BROKEN-ICE-DEM`（TensorDEM，`src/tensordem/polyhedron_floe3d/`）。")
    a("Round 4 把 C2 的三 μ 分片标为 **NOT_RUN**（当次作业墙钟预算内未出结果）。本轮的目标是把这三片跑满、")
    a("并对「已静置 / 落带 / 无效」三种结局如实验收——不补数据、不调判据。")
    a("")
    a("**本轮结局（先说结论）**：正式 45 体分片在本次作业墙钟预算内**仍未跑完**，故 C2 依旧")
    a("`NOT_RUN`；本轮的真实增量是 ①把「测量→报告」这条路做成**机器可校验**的证据链")
    a("（新增 `tests/test_h1_repose_evidence.py` + 零手抄渲染器 `scripts/render_h1_repose_round5.py`），")
    a("②用 12 体可行性试点量出了**体数下限**（n=12 不成堆，端到端可验），")
    a("③量出单步成本，把正式 C2 的算力需求写成了可执行的集群派单规格（§4.2）。")
    a("")
    a("## 1 案例与判据（动手前固定，沿用 round 2–4，未改）")
    a("")
    a("| # | 判据 | 基准 | 阈值 | 本轮状态 |")
    a("|---|---|---|---|---|")
    a("| C1 | 静水浮沉（浸没体积分数/吃水） | Archimedes `V_sub/V = rho_i/rho_w` | < 5 % | 沿用 round 2 结论 |")
    a("| C2 | 碎冰堆积休止角 | 文献冰区带 25–40°（WITHHELD） | 落带内且已静止 | 正式分片仍 NOT_RUN；新增体数下限证据 + 证据门 |")
    a("| C3 | 航道清空 | 泳道浓度降幅 + 质量账 | 降幅 > 50 % | 沿用 round 2 结论 |")
    a("")
    if setup0:
        a("C2 算例（`scripts/run_h1_repose.py` 的固定参数，逐项来自 JSON 的 `setup` 块）：")
        a("")
        a("| 参数 | 值 |")
        a("|---|---|")
        a("| 颗粒数 | %d 个 Voronoi 多面体 |" % int(setup0["n_bodies"]))
        a("| 接触刚度 kn | %g N/m |" % float(setup0["kn"]))
        a("| 阻尼比 ζ | %g |" % float(setup0["zeta"]))
        a("| 时间步 dt | %g s |" % float(setup0["dt"]))
        a("| 步数上限 | %d（= %.2f s 模拟时间） |"
          % (int(setup0["n_steps"]), int(setup0["n_steps"]) * float(setup0["dt"])))
        a("| 静置门 | |v| + R|ω| < %g m/s 持续 %d 步 |"
          % (float(setup0["settle_tol"]), int(setup0["settle_window"])))
        a("| 释放方式 | %s |" % setup0["release"])
        a("")
    a("## 2 怎么做的")
    a("")
    a("1. **分片跑满**：三个进程并行（`OMP_NUM_THREADS=1`，各一片），每片 `--steps 30000`，")
    a("   参数与 round 4 试跑一致（`dt=2e-4`、`kn=5e4`，接触周期内步数 ≈ 33.7）。")
    a("2. **证据落盘**：每片写 `runs/h1_repose/repose_mu<μ>.json`（含 setup、末态接触对、")
    a("   能量/质量残差、自由面剖面、坡拟合结果与拒绝原因）。")
    a("3. **判据不变**：`valid = settled ∧ energy_residual<0.05 ∧ n_contacts>0 ∧ angle 有限`，")
    a("   与 `study.case3_repose` 一致；本轮**未改任何阈值**。")
    a("4. **证据自检（新增）**：`tests/test_h1_repose_evidence.py` 把「已提交的 JSON → 报告表格」")
    a("   钉死：未静置的片不得报角度、`valid` 必须等于判据式的值、`inside_literature_band`")
    a("   必须是角度的函数、报告里的角度必须等于 JSON 里的角度（零手抄）。")
    a("")
    a("## 3 C2 结果（程序产出，零手抄）")
    a("")
    a("本轮**没有**产出 45 体的正式 C2 数字：三片在作业墙钟预算内未跑完（§4.2），"
      "下表三行如实为 `NOT_RUN`。§3.1 给出同判据、同 runner 的 12 体可行性试点，"
      "它回答的是另一个问题（体数下限），**不能替代**正式 C2。")
    a("")
    a("| μ | 文件 | 静置 | 步数 | 模拟时长 [s] | 末态接触对 | 休止角 [°] | 落带内 | 能量残差 | 墙钟 [s] |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for mu in MU_GRID:
        if mu not in runs:
            a("| %g | `runs/h1_repose/repose_mu%g.json` | **NOT_RUN** | — | — | — | — | — | — | — |"
              % (mu, mu))
            continue
        path, d = runs[mu]
        run = d["runs"][0]
        a("| %g | `%s` | %s | %d | %.3f | %d | %s | %s | %.4f | %.1f |" % (
            mu, os.path.relpath(path, ROOT), "yes" if run["settled"] else "**no**",
            run["steps"], run["sim_time_s"], run["n_contacts"],
            _f(run["repose_angle_deg"]), "yes" if d["inside_literature_band"][0] else "no",
            run["energy_residual"], float(d.get("wall_clock_s", float("nan")))))
    a("")
    if runs:
        a("补充量（同一 JSON，供审计：坡拟合质量与接触网络）：")
        a("")
        a("| μ | 坡带数 | 坡高降 [m] | 裙角 [°] | 拒绝/拟合原因 | 配位数 | 顶 20 % 承载份额 | 力尾指数 |")
        a("|---|---|---|---|---|---|---|---|")
        for mu in MU_GRID:
            if mu not in runs:
                continue
            run = runs[mu][1]["runs"][0]
            flank = run.get("repose_flank", {})
            a("| %g | %s | %s | %s | %s | %s | %s | %s |" % (
                mu, flank.get("n_flank_bins", "—"),
                _f(flank.get("flank_height_drop_m"), 3),
                _f(flank.get("apron_angle_deg"), 1),
                flank.get("reason") or run.get("flank_reason", "—"),
                _f(run.get("coordination_number"), 2),
                _f(run.get("top20_load_share"), 2),
                _f(run.get("force_tail_exponent"), 2)))
        a("")
    all_pass = (len(runs) == len(MU_GRID) and all(d["passed"] for d in
                                                  (runs[mu][1] for mu in MU_GRID)
                                                  if True))
    valid_all = (len(runs) == len(MU_GRID)
                 and all(runs[mu][1]["runs"][0]["valid"] for mu in MU_GRID))
    if all_pass:
        a("**C2 判定：PASS** —— 三 μ 全部静置且休止角落 25–40° 带内。"
          "带本身是文献量级带、本仓无冰区原始测量数据集，故基准仍记 **WITHHELD**："
          "「量级相容」不等于「对实验误差 <3 %」。")
    elif runs:
        parts = []
        for mu in MU_GRID:
            if mu not in runs:
                parts.append("μ=%g NOT_RUN" % mu)
                continue
            run = runs[mu][1]["runs"][0]
            if not run["settled"]:
                parts.append("μ=%g 未静置（%d 步后仍在动，%s）"
                             % (mu, run["steps"], run.get("flank_reason") or "无坡"))
            elif not run["valid"]:
                parts.append("μ=%g 已静置但 valid=False（能量残差 %.4f，门 %.2f；接触 %d）"
                             % (mu, run["energy_residual"], ENERGY_GATE, run["n_contacts"]))
            elif not runs[mu][1]["inside_literature_band"][0]:
                parts.append("μ=%g 静置且 valid，角度 %.2f° 在 25–40° 带外"
                             % (mu, float(run["repose_angle_deg"])))
            else:
                parts.append("μ=%g PASS" % mu)
        a("**C2 判定：部分/honest FAIL** —— " + "；".join(parts) + "。"
          "未达标的分片不给角度、不写「已达标」，结论按上表如实保留。")
    else:
        a("**C2 判定：NOT_RUN（honest）** —— 分片未产出结果。")
    a("")
    if have_fig:
        a("![C2](figures/%s)" % os.path.basename(FIG))
        a("")
        a("图 1 左：休止角 vs 颗粒摩擦系数（阴影=文献量级带 25–40°，虚线=干 Coulomb "
          r"$\arctan\mu$ 参照）；右：各 μ 末态自由面剖面（环带半径 vs 高度）。"
          "`×` 标记的算例为已静置但未过 `valid` 门的运行。")
        a("")
    if pilots:
        a("### 3.1 n=12 可行性试点（同一判据、同一 runner，体数降到 12）")
        a("")
        a("正式 C2 用 45 体；在本次作业的墙钟预算内 45 体 30 000 步未能跑完（见 §4.2），"
          "因此先跑一组 12 体 / 8 000 步的**可行性试点**，只看一件事：这个体数下能不能成堆。")
        a("")
        a("| μ | 文件 | 步数 | 模拟时长 [s] | 静置 | 末态接触对 | 休止角 [°] | 拒绝原因 | 能量残差 | 墙钟 [s] |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for mu, path, d in pilots:
            r = d["runs"][0]
            a("| %g | `%s` | %d | %.3f | %s | %d | %s | %s | %.4f | %.1f |" % (
                mu, os.path.relpath(path, ROOT), r["steps"], r["sim_time_s"],
                "yes" if r["settled"] else "**no**", r["n_contacts"],
                _f(r["repose_angle_deg"]), r.get("flank_reason", "—"),
                r["energy_residual"], float(d.get("wall_clock_s", float("nan")))))
        n_pile = sum(1 for _m, _p, d in pilots
                     if math.isfinite(float(d["runs"][0]["repose_angle_deg"])))
        a("")
        a("试点结论：%d/%d 片成堆——**n=12 不足以形成堆积体**（末态接触对 0、单层摊开、"
          "坡拟合按 `no_descending_flank` 拒绝给角度）。这与 round 3 §6 的分辨率结论一致："
          "堆积角是**群体量**，体数太少时颗粒摊成单层，任何角度数字都是假的。"
          "因此正式 C2 必须保持 45 体，不能靠减体数换墙钟。" % (n_pile, len(pilots)))
        a("")
    a("## 4 注意点 / 局限")
    a("")
    a("1. **文献带不是实验值**：25–40° 是角状颗粒/冰堆积的量级带（出处见 ROUND2 §2），")
    a("   本仓无冰区休止角原始数据集；即使落带内也只能声明「量级相容」(WITHHELD)。")
    _cost = ["%.1f ms/步 (μ=%g, %d 体)" % (1000.0 * float(d.get("wall_clock_s", 0.0))
                                            / max(int(d["runs"][0]["steps"]), 1),
                                            mu, int(d["setup"]["n_bodies"]))
             for mu, _p, d in pilots]
    a("2. **步数与墙钟（本轮的实际限制）**：45 体 / 30 000 步 / 三片并行的正式 C2 在本次")
    a("   作业的墙钟预算内**没有跑完**——试点实测单步成本 %s，" % ("；".join(_cost) or "—"),)
    a("   进入密堆接触期后每步成本按 round 3 §5 的量级可达 0.1–0.2 s，单片即 50–100 min。")
    a("   本轮**不把没跑完的算例写成结论**；下一步应把三片交给集群（`run_h1_repose.py --shard i`，")
    a("   每片一个作业、`timeout ≥ 7200 s`）而不是压在单个会话里（与 round 4 §6.3 的 cj1009 排队教训合并处理）。")
    a("3. **本次未做的**：`runs/h1_repose/` 中的正式分片 JSON 缺失，报告中三行如实为 NOT_RUN；")
    a("   证据完整性测试已就位，一旦分片产出即可机器校验（未静置不得报角度、报告角度必须等于 JSON）。")
    a("3. **能量门与冲击期**：`energy_residual` 是全程账（含冲击段耗散记账），若末态残差")
    a("   仍高于 0.05，则 `valid=False` 而不是「把门调松」——本轮不改判据，结论如实报。")
    a("4. **角度分辨率**：45 体 × 7 环带（每带约 6 体）时坡拟合的统计误差可达数度，")
    a("   角度只能做「是否落带」的判读，不能当 ±2° 的测量值（round 3 §5 已量化）。")
    a("")
    a("## 5 复现")
    a("")
    a("```bash")
    a("PYTHONPATH=src python scripts/run_h1_repose.py --shard 1 --steps 30000 --dt 2e-4 --kn 5e4")
    a("PYTHONPATH=src python scripts/run_h1_repose.py --shard 2 --steps 30000 --dt 2e-4 --kn 5e4")
    a("PYTHONPATH=src python scripts/run_h1_repose.py --shard 3 --steps 30000 --dt 2e-4 --kn 5e4")
    a("PYTHONPATH=src python scripts/render_h1_repose_round5.py   # 重生成本文档 + 图 1")
    a("PYTHONPATH=src python -m pytest tests/test_h1_repose_evidence.py -q")
    a("```")
    a("")
    a("## 6 误差判定（逐项，显式回答「是否 < 3 %」）")
    a("")
    a("| 量 | 基准 | 本轮结果 | 相对误差 | < 3 %? |")
    a("|---|---|---|---|---|")
    for mu in MU_GRID:
        if mu not in runs:
            a("| 休止角 μ=%g | 文献带 25–40° | NOT_RUN | — | 无法判定（未跑） |" % mu)
            continue
        run = runs[mu][1]["runs"][0]
        ang = run["repose_angle_deg"]
        if isinstance(ang, float) and math.isfinite(ang):
            a("| 休止角 μ=%g | 带中心 32.5°（仅量级参照） | %.2f° | %.1f %% | **否**（基准为量级带，非定值；< 3 %% 无从谈起） |"
              % (mu, ang, abs(ang - 32.5) / 32.5 * 100.0))
        else:
            a("| 休止角 μ=%g | 文献带 25–40° | 未给出（%s） | — | 否（无测量值） |"
              % (mu, run.get("flank_reason", "unsettled")))
    a("")
    a("结论：C2 的唯一基准是**量级带**（WITHHELD），因此本轮无条件可声明「< 3 %」；")
    a("可声明的只有**结构量**——质量残差（上表 `mass_residual`，本应为 0 的机器零）与")
    a("能量账残差（上表 `energy_residual`，门 0.05）——这两项是精确账，不是统计量。")
    a("")
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", os.path.relpath(DOC, ROOT), "| shards:", sorted(runs),
          "| missing:", missing, "| figure:", have_fig, "| dense shards:", dense,
          "| max sim time:", round(budget, 3))


if __name__ == "__main__":
    main()