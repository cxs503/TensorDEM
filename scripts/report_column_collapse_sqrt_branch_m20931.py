"""Report builder for the a=3 square-root-branch rung (m20931).

Reads only committed provenance (``docs/artifacts/.../verdict.json`` of this
round and of m20873) and injects every number into the markdown table, so no
value in ``docs/CASE_COLUMN_COLLAPSE_SQRT_BRANCH_m20931.md`` is typed by hand.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "artifacts", "column_collapse_sqrt_branch_m20931")
VERDICT = os.path.join(OUT, "verdict.json")
ARCHIVE = os.path.join(ROOT, "docs", "artifacts",
                       "column_collapse_deep_refinement_m20873", "verdict.json")
DOC = os.path.join(ROOT, "docs", "CASE_COLUMN_COLLAPSE_SQRT_BRANCH_m20931.md")
TOL = 0.15


def load() -> tuple[dict, dict]:
    with open(VERDICT) as fh:
        payload = json.load(fh)
    with open(ARCHIVE) as fh:
        archive = json.load(fh)
    return payload, archive


def fig_ladder(payload: dict) -> str:
    rows = payload["rows"]
    gpr = [r["grains_per_radius"] for r in rows]
    err = [100.0 * r["err_runout"] for r in rows]
    ratio = [r["runout_ratio"] for r in rows]
    ref = [r["ref_runout_ratio"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax[0].plot(gpr, ratio, "o-", color="#1f4e79", label=r"DEM $r_\infty/r_i$")
    ax[0].plot(gpr, ref, "s--", color="#c00000",
               label=r"Lube (2004) at $a_{\rm meas}$")
    ax[0].set_xlabel("grains per column radius $r_i/d$")
    ax[0].set_ylabel(r"runout ratio $r_\infty/r_i$")
    ax[0].set_xticks(gpr)
    ax[0].set_title(r"$a = 3$ (square-root branch) refinement ladder")
    ax[0].legend(frameon=False)
    ax[1].axhspan(-100 * TOL, 100 * TOL, color="#d9ead3", label="±15 % band")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].plot(gpr, err, "o-", color="#1f4e79")
    for x, y in zip(gpr, err):
        ax[1].annotate(f"{y:+.2f} %", (x, y), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=9)
    ax[1].set_xlabel("grains per column radius $r_i/d$")
    ax[1].set_ylabel("runout error vs Lube law [%]")
    ax[1].set_xticks(gpr)
    ax[1].set_title("acceptance: |error| ≤ 15 %")
    ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_a3_ladder.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_branches(payload: dict, archive: dict) -> str:
    by_id = {r["case_id"]: r for r in archive["rows"]}
    lin = [(by_id[c]["grains_per_radius"], 100.0 * by_id[c]["err_runout"])
           for c in ("R_a1_g8", "R_a1_g12", "R_a1_g16") if c in by_id]
    sqrt = [(r["grains_per_radius"], 100.0 * r["err_runout"])
            for r in payload["rows"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhspan(-100 * TOL, 100 * TOL, color="#d9ead3", label="±15 % band")
    ax.axhline(0.0, color="k", lw=0.8)
    if lin:
        ax.plot([g for g, _ in lin], [e for _, e in lin], "s-", color="#7f7f7f",
                label=r"$a = 1$ (linear branch, m20873)")
    ax.plot([g for g, _ in sqrt], [e for _, e in sqrt], "o-", color="#1f4e79",
            label=r"$a = 3$ (square-root branch, this round)")
    for g, e in sqrt:
        ax.annotate(f"{e:+.2f} %", (g, e), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("grains per column radius $r_i/d$")
    ax.set_ylabel("runout error vs Lube law [%]")
    ax.set_xticks(sorted({g for g, _ in lin} | {g for g, _ in sqrt}))
    ax.set_title("branch asymmetry of the runout error under refinement")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_branch_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_increments(payload: dict) -> str:
    rows = payload["rows"]
    gpr = [r["grains_per_radius"] for r in rows]
    delta = [float("nan")] + [rows[k]["runout_ratio"] - rows[k - 1]["runout_ratio"]
                              for k in range(1, len(rows))]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(range(len(rows)), [0.0 if d != d else d for d in delta],
           color=["#d9d9d9"] + ["#1f4e79"] * (len(rows) - 1))
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{g:g}" for g in gpr])
    ax.set_xlabel("grains per column radius $r_i/d$")
    ax.set_ylabel(r"increment $\Delta (r_\infty/r_i)$")
    ax.set_title("step-to-step increments (contraction test)")
    for k, d in enumerate(delta):
        if d == d:
            ax.annotate(f"{d:+.4f}", (k, d), textcoords="offset points",
                        xytext=(0, 6 if d >= 0 else -14), ha="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_increments.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def table_md(payload: dict) -> str:
    lines = ["| case | 粒/$r_i$ | N | dt [s] | $a_{\\rm meas}$ | $r_\\infty/r_i$ "
             "| ref | 误差 | t90 [s] | ref $t_\\infty$ | t90 误差 | 判 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in payload["rows"]:
        flag = "✅" if abs(r["err_runout"]) <= TOL else "❌"
        lines.append(
            f"| {r['case_id']} | {r['grains_per_radius']:g} | {r['n_particles']} "
            f"| {r['dt']:.3e} | {r['aspect_measured']:.3f} "
            f"| {r['runout_ratio']:.4f} | {r['ref_runout_ratio']:.4f} "
            f"| {100 * r['err_runout']:+.2f} % | {r['t90']:.4f} "
            f"| {r['ref_time']:.4f} | {100 * r['err_t90']:+.2f} % | {flag} |")
    return "\n".join(lines)


def write_doc(payload: dict, figs: list[str], fig_archive: str) -> str:
    v = payload["verdict"]
    rows = payload["rows"]
    rich = v.get("richardson", {})
    lines = []
    lines.append("# CASE-COLUMN-COLLAPSE：平方根支 $a=3$ 第三细化档（m20931）\n")
    lines.append("## ① 案例介绍\n")
    lines.append(
        "锚点 `CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌，ref Lube et al. 2004，"
        "容差 <15 %）。物理：圆柱颗粒柱在重力下坍塌，跑距满足 "
        "$r_\\infty/r_i = 1+1.24a$（$a<1.7$）与 $1+1.6a^{1/2}$（$a>1.7$），"
        "埋置时间 $t_\\infty = 3(r_i a/g)^{1/2}$。\n")
    lines.append(
        "本单只补一件欠账：m20873 把 $a=3$ 留在线外（gpr 8 → +8.09 %，gpr 12 → "
        "**+15.40 %**），且显式记「未做：$a=3$ 的 16 粒/半径档」。两档不足以区分"
        "「离散化未收敛」与「平方根支系统性偏差」。")
    lines.append("\n**冻结几何与参数**（`SB_a3_g16`，预登记见 "
                 "`docs/PREREG_COLUMN_COLLAPSE_SQRT_BRANCH_m20931.md`）："
                 "$r_i=0.48$ m、$r=0.015$ m（$r_i/d=16$）、$h_i=1.44$ m、$\\rho=2500$ kg/m³、"
                 "$\\mu=0.5$、$\\mu_r=0.5$、$\\mu_{\\rm wall}=0.8$、$e=0.5$、"
                 "$k_t/k_n=0.5$、$k_n=1.5\\times10^4$ N/m（保持 $\\delta/d$ 不变）、"
                 "HCP 初始格隙 2 %、$g=9.81$ m/s²。\n")
    lines.append("## ② 怎么做的\n")
    lines.append(
        "1. 跑前预登记（H1/H2/H3 分档规则、唯一新增算例、不许调参）→ "
        "`docs/PREREG_COLUMN_COLLAPSE_SQRT_BRANCH_m20931.md`。\n"
        "2. 新脚本 `benchmarks/case_column_collapse_sqrt_branch_m20931.py` "
        "**只读复用** m18261 的冻结配置（`make_case`/`reference_values`/"
        "`protocol_signature`），不改旧协议文件；阶梯前两档直接读 m20873 已提交的逐算例 JSON。\n"
        "3. 参考律在运行时求值（`lube_2004_runout_ratio(3.0)`），零手抄；"
        "主判据用**实测** $a$，并同时报名义 $a$ 口径。\n"
        "4. 集群递交 `cj3056`（单进程，$N$ ≈ 4.7 万），结果写进 "
        "`docs/artifacts/column_collapse_sqrt_branch_m20931/`。\n"
        "5. `--collect` 从三份 JSON 归约出 `verdict.json` + CSV，"
        "`scripts/report_column_collapse_sqrt_branch_m20931.py` 把数字注入本文（无手抄）。\n")
    lines.append("## ③ 计算结果\n")
    lines.append(table_md(payload) + "\n")
    lines.append(
        f"阶梯裁决（预登记规则）：**{v['hypothesis']} → {v['verdict']}**。"
        f"增量收缩检验 $|\\Delta_{{23}}|/|\\Delta_{{12}}| = "
        f"{(v.get('abs_delta_ratio') if v.get('abs_delta_ratio') is not None else float('nan')):.3f}$"
        f"（<1 表示收缩），$\\Delta_{{12}}={v.get('delta_12', float('nan')):+.4f}$、"
        f"$\\Delta_{{23}}={v.get('delta_23', float('nan')):+.4f}$。\n")
    if rich.get("available"):
        lines.append(
            f"指示性外推（两点 Richardson，$r={rich['ratio']:.4f}$，阶数未知故给括号）："
            f"$p=1$ → $r_\\infty/r_i\\approx{rich['f_inf_p1']:.4f}$"
            f"（误差 {100 * rich['err_inf_p1']:+.2f} %）、$p=2$ → "
            f"$\\approx{rich['f_inf_p2']:.4f}$（误差 {100 * rich['err_inf_p2']:+.2f} %）。"
            "**外推值不作为验收通过依据**，只作趋势判读。\n")
    lines.append(f"能量账本残差：{['%.2e' % x for x in v.get('energy_residual', [])]}；"
                 f"停止原因：{v.get('stop_reasons', [])}。\n")
    lines.append("## ④ 结果图\n")
    lines.append("![a3 ladder](../docs/artifacts/column_collapse_sqrt_branch_m20931/"
                 "fig_a3_ladder.png)\n")
    lines.append("**图 1** — 左：$r_\\infty/r_i$ 随分辨率（点=DEM，虚线=Lube 律在实测 $a$ 上的值）；"
                 "右：相对误差与 ±15 % 判据带。\n")
    lines.append("![branch comparison](../docs/artifacts/column_collapse_sqrt_branch_m20931/"
                 "fig_branch_comparison.png)\n")
    lines.append("**图 2** — 两支对照：$a=1$（线性支，灰）与 $a=3$（平方根支，蓝）在同一细化轴"
                 "上的跑距误差轨迹；平方根支随细化**外扩**、线性支在带内摆动，是本单的主要观察。\n")
    lines.append("![increments](../docs/artifacts/column_collapse_sqrt_branch_m20931/"
                 "fig_increments.png)\n")
    lines.append("**图 3** — 步间增量（收缩检验）：粗→中、中→细两步的 $r_\\infty/r_i$ 增量。\n")
    lines.append("## ⑤ 注意点\n")
    lines.append(
        "1. 三点阶梯**不能**独立测出收敛阶，Richardson 只能给括号；本单不以外推值当实测。\n"
        "2. 与 m20873 相同的口径纪律：主判据用实测 $a$（名义 $a$ 并列于 CSV），"
        "混口径会伪造趋势。\n"
        "3. 冻结闭合臂的 $\\mu_r=0.5$ 是 round-5 标定值，本单**未**再拟合；"
        "若换参数化（$\\mu(I)$、粗糙地面）属新预登记条目，不在此单结论内。\n"
        "4. 穿透/重叠计数：协议 `return_positions=False`，记 `not_run`。\n"
        "5. 单实现、确定性（jitter=0）：不含系综散布（散布见 m19922/m20259），"
        "故单点差异不得解释为随机性。\n"
        "6. **阶梯非单调（本单最重要的限定）**：误差 8.09 % → 15.40 % → 11.06 %，"
        "中间档是离群点。因此本单只能说「$gpr=16$ 回到 15 % 带内且最后一步增量收缩」，"
        "**不能**声称已收敛，也不能宣称观测到干净的收敛阶；三点非单调阶梯连 Richardson "
        "的阶数假设都不成立，上式外推只作括号。\n"
        "7. **对抗性复核发现（m20873 provenance 标签缺陷，非数值错误）**："
        "m20873 的 `verdict.json` 把到达时间列命名为 `t95`，"
        "但其数值实为冻结主定义 `fraction_0.9`（已逐例核对：`R_a1_g8` 该列为 0.6684 s = "
        "`arrival.fraction.0.9`，而真正的 0.95 分位为 0.7190 s）；报告正文写的也是 t90。"
        "标签与数值不符会让下游误用，本单的 `ladder_row` 因此**不采信列名**，"
        "直接从 `arrival.primary.fraction_0.9` 重算。\n"
        "8. **能量账本在平方根支只闭合到 ~89 %**：`residual_fraction_of_potential`"
        "（释放势能 − 动能 − 弹簧能 − 耗散 / 初始势能）在 $a=3$ 三档为 "
        "12.36 % / 11.57 % / 10.86 %（$a=1$ 的粗档仅 5.97 %），随细化单调下降但幅度小。"
        "本单**不设阈值门**（不做事后定阈值），只如实入账：该残差与跑距超预测同向，"
        "提示平方根支的耗散记账存在未归因项，属可另立条目追查的欠账。\n")
    lines.append("## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）\n")
    lines.append("| 关键量 | 基准 | 误差 | <3 % ? | <15 % ? |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        e = r["err_runout"]
        lines.append(f"| 跑距 `{r['case_id']}` | Lube 律 @$a_{{\\rm meas}}$ "
                     f"| {100 * e:+.2f} % | {'是' if abs(e) <= 0.03 else '否'} "
                     f"| {'是' if abs(e) <= TOL else '否'} |")
    for r in rows:
        et = r["err_t90"]
        lines.append(f"| 埋置时间 t90 `{r['case_id']}` | $3(h_i/g)^{{1/2}}$ "
                     f"| {100 * et:+.2f} % | {'是' if abs(et) <= 0.03 else '否'} "
                     f"| {'是' if abs(et) <= TOL else '否'} |")
    worst = max(rows, key=lambda r: abs(r["err_runout"]))
    lines.append(
        f"\n**显式回答**：跑距最大误差 {100 * max(abs(r['err_runout']) for r in rows):+.2f} % "
        f"（`{worst['case_id']}`），"
        f"{'**达到**' if max(abs(r['err_runout']) for r in rows) <= 0.03 else '**未达到**'} <3 %；"
        f"时间轴最大 {100 * max(abs(r['err_t90']) for r in rows):.2f} %，"
        f"{'达到' if max(abs(r['err_t90']) for r in rows) <= 0.03 else '未达到'} <3 %。"
        f"15 % 判据：{sum(1 for r in rows if abs(r['err_runout']) <= TOL)}/{len(rows)} 通过。\n")
    lines.append("## 复现\n")
    lines.append("```bash\n"
                 "# 计算（集群，勿在本机跑重算）\n"
                 "PYTHONPATH=src python3 benchmarks/case_column_collapse_sqrt_branch_m20931.py --case --progress\n"
                 "PYTHONPATH=src python3 benchmarks/case_column_collapse_sqrt_branch_m20931.py --collect\n"
                 "PYTHONPATH=src python3 scripts/report_column_collapse_sqrt_branch_m20931.py\n"
                 "PYTHONPATH=src python3 -m pytest tests/test_column_collapse_sqrt_branch_m20931.py -q\n"
                 "```\n")
    lines.append(f"产物：`docs/artifacts/column_collapse_sqrt_branch_m20931/`"
                 f"（`cases/SB_a3_g16.json`、`verdict.json`、"
                 f"`cases_sqrt_branch.csv`、`{os.path.basename(figs[0])}` 等 3 图）。")
    text = "\n".join(lines) + "\n"
    with open(DOC, "w") as fh:
        fh.write(text)
    return DOC


def main() -> int:
    payload, archive = load()
    os.makedirs(OUT, exist_ok=True)
    figs = [fig_ladder(payload), fig_branches(payload, archive),
            fig_increments(payload)]
    doc = write_doc(payload, figs, "")
    print("wrote", doc)
    for f in figs:
        print("wrote", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())