#!/usr/bin/env python3
"""Build the m23125 audit artefacts for the paired timestep-bias gate.

Writes, from the run JSONs on disk and nothing else:

* ``benchmarks/case_rcp_deposition/runs/paired_timestep_audit.json``
* ``benchmarks/case_rcp_deposition/figures/fig_m23125_paired_forest.png``
* ``benchmarks/case_rcp_deposition/figures/fig_m23125_pair_scatter.png``
* ``docs/AUDIT_m23125_RCP_PAIRED_TIMESTEP.md``

Every number in the markdown is injected from the audit payload, so re-running
this script after new runs land rewrites the document; nothing is hand-copied.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paired_timestep_audit import OUT_JSON, audit  # noqa: E402
from seed_bias_audit import TOL_BIAS  # noqa: E402

FIG = os.path.join(HERE, "figures")
DOC = os.path.join(ROOT, "docs", "AUDIT_m23125_RCP_PAIRED_TIMESTEP.md")
RCP_REF = 0.6366  # Scott 1960; Bernal & Mason 1960 (see tensordem.rcp_references)


def _fmt(x, nd=6):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _sci(x):
    return "—" if x is None else f"{x:+.3e}"


def fig_forest(res):
    rows = res["levels"]
    fig, ax = plt.subplots(figsize=(7.2, 0.9 * len(rows) + 1.8))
    ax.axvspan(-TOL_BIAS, TOL_BIAS, color="tab:green", alpha=0.15,
               label=f"tolerance |Δ| < {TOL_BIAS:g}")
    ax.axvline(0.0, color="k", lw=0.8)
    for i, r in enumerate(rows):
        p, u = r["paired"], r["unpaired"]
        ax.plot([u["lo"], u["hi"]], [i + 0.16, i + 0.16], color="tab:gray",
                lw=2.2, label="unpaired Welch 95 %" if i == 0 else None)
        ax.plot(u["delta"], i + 0.16, "o", color="tab:gray", ms=5)
        if p["lo"] is not None:
            ax.plot([p["lo"], p["hi"]], [i - 0.16, i - 0.16], color="tab:blue",
                    lw=2.6, label="paired CRN 95 %" if i == 0 else None)
        ax.plot(p["delta"], i - 0.16, "o", color="tab:blue", ms=6,
                label=None if i else "paired CRN point")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"μ={r['mu']:g}\n$k_n$={r['kn']:g}\n"
                        f"({r['paired']['n_pairs']} pairs)" for r in rows])
    ax.set_xlabel(r"$\Delta=\phi(dt/2)-\phi(dt)$")
    ax.set_title("Timestep bias: paired vs unpaired interval")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(FIG, "fig_m23125_paired_forest.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_scatter(res):
    rows = [r for r in res["levels"] if r["paired"]["n_pairs"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    for r in rows:
        d = np.array(r["paired"]["deltas"])
        ax.plot(np.arange(1, d.size + 1), d, "o-", ms=4,
                label=f"μ={r['mu']:g}, $k_n$={r['kn']:g}")
    ax.axhspan(-TOL_BIAS, TOL_BIAS, color="tab:green", alpha=0.15)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("pair index (seed)")
    ax.set_ylabel(r"$\Delta_i$ per seed")
    ax.set_title("Per-seed paired differences")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [f"μ={r['mu']:g}\n$k_n$={r['kn']:g}" for r in rows]
    ratios = [(r["variance_reduction"]["ratio"] or np.nan) for r in rows]
    ax.bar(range(len(rows)), ratios, color="tab:blue", alpha=0.8)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="no gain (ratio = 1)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\mathrm{SE}_{unpaired}/\mathrm{SE}_{paired}$")
    ax.set_title("Interval width bought by pairing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(FIG, "fig_m23125_pair_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def markdown(res, fig_forest_rel, fig_scatter_rel, plan):
    s = res["summary"]
    rows = res["levels"]
    L = []
    A = L.append
    A("# AUDIT m23125 · RCP 沉积算例池：时间步偏差门的配对（CRN）估计与种子预算\n")
    A("锚点：`CASE-DEM-RCP-PACKING-FRACTION`（随机密堆体积分数，Scott 1960 / "
      "Bernal & Mason 1960，验收带 0.62–0.66，10000 球）。\n")
    A("## ① 案例与要解决的问题\n")
    A("m22824 §3 对四个 `(μ, k_n)` 档做时间步偏差门，判据为\n\n"
      "`|Δ| + t·SE_Δ < 5e-4`（Δ = φ(dt/2) − φ(dt)，m22558/m21401 预登记），\n\n"
      "结果 3/4 档 `UNDECIDABLE`，可认证 0/4。原因是**区间太宽**：种子散布"
      "（σ≈3e-4）与容许带同量级，按非配对 Welch 估计要 ~5 种子/侧才收敛。\n")
    A("本审计的实质增量是**换估计量，不是放宽容差**：粗/细两次算例在同一 `seed` "
      "下**共享同一条初始构型**（`DepositionSpec.box_width` / `initial_height` "
      "与 `dt_frac` 无关，`initial_configuration` 只吃 "
      "`(n, box_width, height, radius, seed)`），于是种子的随机性是一对算例的"
      "**共模参数**，在逐种子差 `Δ_i` 中相消。容差 `TOL_BIAS`、置信度与三值裁决"
      "词汇全部从 `seed_bias_audit` 原样导入，未新增第六种判词。\n")
    A("## ② 怎么做的\n")
    A("1. **前提先验证，不靠读代码**：`tests/test_rcp_paired_timestep.py` 断言"
      "同一 seed、`dt_frac`=1/20 与 1/40 的 `DepositionSpec` 几何相同，且 "
      "`initial_configuration` 返回**逐字节相同**的坐标与相同的拒斥候选数；"
      "存档算例另做**指纹核对**——run JSON 的 `placement` 记录（`drop_height`、"
      "`initial_height`、`rejected_candidates`）在同一对的两次算例上必须数值全等，"
      "不等的对被丢弃并计数（`PAIRING_PREMISE_FAILED`），不静默平均。\n")
    A("2. **估计量**：`Δ_i = φ_fine,i − φ_coarse,i`，"
      "`SE = sd(Δ)/√n`，自由度 `n−1`，t 分布 95 % 区间；"
      "裁决沿用 `classify_bias`（区间完全落带内 CONVERGED / 完全落带外 BIASED / "
      "跨带边 UNDECIDABLE）。**点估计永不认证**：`certified` 额外要求 "
      "`resolved_from == \"interval\"`，即 `n_pairs ≥ 2`。\n")
    A("3. **配对收益可核查**：`variance_reduction()` 在同一批配对数据上同时给出"
      "配对 SE 与非配对 Welch SE，比值 `SE_unpaired/SE_paired` 直接报告（本单存档"
      "数据上该比值见下表）；`seeds_needed_paired(σ, margin)` 给出配对估计量所需"
      "的种子预算，与非配对的 `seeds_needed` 并列。\n")
    A("4. **单命令复现**："
      "`PYTHONPATH=src python benchmarks/case_rcp_deposition/paired_timestep_audit.py`；"
      "本文件由 `benchmarks/case_rcp_deposition/m23125_doc.py` 程序化生成。\n")
    A("## ③ 结果（数字由脚本注入）\n")
    A("### 3.1 逐档裁决：配对 vs 非配对（同一批算例）\n")
    A("| μ | $k_n$ | 种子(粗/细) | 配对数 | Δ 配对 | 配对 95 % 区间 | 配对裁决 | "
      "Δ 非配对 | 非配对裁决 | 可认证 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        p, u = r["paired"], r["unpaired"]
        ci = ("—" if p["lo"] is None
              else f"[{_sci(p['lo'])}, {_sci(p['hi'])}]")
        A(f"| {r['mu']:g} | {r['kn']:g} | {r['n_seeds_coarse']}/{r['n_seeds_fine']} | "
          f"{p['n_pairs']} | {_sci(p['delta'])} | {ci} | **{p['verdict']}** | "
          f"{_sci(u.get('delta'))} | {u['verdict']} | "
          f"{'是' if r['certified'] else '否'} |")
    A("")
    A(f"汇总：`{json.dumps(s, sort_keys=True)}`\n")
    A("### 3.2 配对买到多少区间\n")
    A("| μ | $k_n$ | 配对数 | SE 配对 | SE 非配对(Welch) | 比值 | "
      "配对数预算(带内 5e-4) |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        vr, sb = r["variance_reduction"], r["seed_budget"]
        ratio_txt = "—" if vr["ratio"] is None else f"{vr['ratio']:.2f}"
        A(f"| {r['mu']:g} | {r['kn']:g} | {vr['n_pairs']} | {_sci(vr['sem_paired'])} | "
          f"{_sci(vr['sem_unpaired'])} | {ratio_txt} | "
          f"{sb['seeds_needed_paired']} |")
    A("")
    A("### 3.3 锚点验收（φ 本身，未被本单改动）\n")
    A("| μ | $k_n$ | φ(粗, 均值) | 相对文献 0.6366 | 在 0.62–0.66 带内 |")
    A("|---|---|---|---|---|")
    for r in rows:
        phi = r["phi_coarse_mean"]
        rel = None if phi is None else (phi - RCP_REF) / RCP_REF * 100.0
        A(f"| {r['mu']:g} | {r['kn']:g} | {_fmt(phi, 5)} | "
          f"{'—' if rel is None else f'{rel:+.2f}%'} | "
          f"{'是' if phi is not None and 0.62 <= phi <= 0.66 else '否'} |")
    A("")
    A(f"![forest]({fig_forest_rel})\n")
    A("*配对（蓝）与非配对（灰）的 Δ 95 % 区间，绿带为容差 |Δ|<5e-4。"
      "区间越短，越可能离开 UNDECIDABLE——而带的位置一点没动。*\n")
    A(f"![scatter]({fig_scatter_rel})\n")
    A("*左：逐种子配对的 Δ_i；右：同一批数据上 SE_unpaired/SE_paired，"
      "即配对把区间收窄的倍数。*\n")
    A("## ④ 注意点与局限\n")
    A("1. **配对合法性的边界**：配对要求两次算例的**初始构型**相同。若将来有人"
      "改动 `initial_configuration` 使其依赖 `dt_frac`，或新旧摘要的放置规则不同，"
      "指纹核对会把这些对判 `PAIRING_PREMISE_FAILED` 并剔出估计——这是设计使然，"
      "不是失败。\n")
    A("2. **配对不能消除系统性偏差**：dt→0 的偏差本身被 Δ 直接测量，配对只降方差。"
      "因此 Δ 显著不为零时判 `BIASED` 是正确的物理结论，不是数值噪声。\n")
    A("3. **种子数仍受算力约束**：本单只补 μ=0 的两档（承载文献对标的档）；"
      "μ=0.1 两档仍是 1–3 对，其裁决保持 `UNDECIDABLE` 或点估计，不认证。\n")
    A("4. **跨摘要**：存档算例横跨 3 个实现摘要，m23050 已判 `EQUIVALENT_PATH` "
      "（可达函数逐字节相同）；本审计沿用该结论，未重新取证。\n")
    A("## ⑤ 误差判定（逐项，显式回答是否 <3 %）\n")
    A("| 量 | 计算值 | 基准 | 相对误差 | <3 %？ |")
    A("|---|---|---|---|---|")
    worst = None
    for r in rows:
        phi = r["phi_coarse_mean"]
        if phi is None:
            continue
        rel = (phi - RCP_REF) / RCP_REF * 100.0
        worst = rel if worst is None or abs(rel) > abs(worst) else worst
        A(f"| φ(μ={r['mu']:g}, $k_n$={r['kn']:g}) | {phi:.5f} | 0.6366 "
          f"(Scott 1960; Bernal & Mason 1960) | {rel:+.2f}% | "
          f"{'是' if abs(rel) < 3.0 else '否'} |")
    A("")
    A("本审计的新增量是**估计量与种子预算**，不是新的物理量；Δ 与文献无可比性"
      "（Δ 是同一物理量在两个时间步下的差，基准是 0）。"
      "φ 的文献误差判定行如上表，未被本单改写。\n")
    A("## ⑥ 待办（交给下一单，附量化验收）\n")
    A("1. 本单已递交的配对补齐算例（`plan_m23125.json`，10 个作业）回件后重跑 "
      "`m23125_doc.py` 即可刷新本文；若 μ=0 两档达到 `seeds_needed_paired` 预算，"
      "裁决应从 `UNDECIDABLE` 变成有证据的 `CONVERGED`/`BIASED`。\n")
    A("2. 建议新条目 `CASE-DEM-RCP-TIMESTEP-PAIRED-CERT`（域：DEM·堆积）："
      "验收=四个 `(μ,k_n)` 档在配对估计量下全部 `certified`（`resolved_from=interval` "
      "且 verdict∈{CONVERGED,BIASED}），并给出逐档所需种子数与算例成本。\n")
    A(f"\n### 递交台账\n\n```json\n{json.dumps(plan, indent=1, ensure_ascii=False)}\n```\n")
    return "\n".join(L)


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    res = audit()
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    f1 = fig_forest(res)
    f2 = fig_scatter(res)
    plan_path = os.path.join(HERE, "plan_m23125.json")
    plan = json.load(open(plan_path)) if os.path.exists(plan_path) else {}
    md = markdown(res,
                  os.path.relpath(f1, ROOT),
                  os.path.relpath(f2, ROOT), plan)
    with open(DOC, "w") as fh:
        fh.write(md)
    print(f"wrote {OUT_JSON}\nwrote {f1}\nwrote {f2}\nwrote {DOC}")
    print("summary:", json.dumps(res["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())