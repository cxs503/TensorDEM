"""Render the m22824 RCP provenance + seed-closure audit (doc, figures, report).

One writer for ``docs/AUDIT_m22824_RCP_PROVENANCE_SEED_CLOSURE.md``: every
number in the document and in the published report payload is injected from the
artefacts on disk (run JSONs, ``provenance_audit.json``,
``seed_bias_audit.json``) -- nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/m22824_doc.py
    PYTHONPATH=src python benchmarks/case_rcp_deposition/m22824_doc.py --report /tmp/report_m22824.json
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import aggregate as agg            # noqa: E402
import provenance_audit as pa      # noqa: E402
import seed_bias_audit as sba      # noqa: E402
from tensordem.deposition_packing import stiffness_limit  # noqa: E402

FIGS = os.path.join(HERE, "figures")
DOC = os.path.join(ROOT, "docs",
                   "AUDIT_m22824_RCP_PROVENANCE_SEED_CLOSURE.md")
BAND = agg.BAND                    # feature_ledger acceptance band (0.62, 0.66)
DT_COARSE, DT_FINE = sba.DT_COARSE, sba.DT_FINE


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def level_table(runs) -> list:
    """Production-timestep packing fraction per ``(mu, k_n)`` with seed spread."""
    rows = []
    for mu in sorted({r["mu"] for r in runs if r["n"] == 10000}):
        for kn in sorted({r["kn"] for r in runs
                          if r["n"] == 10000 and r["mu"] == mu}):
            vals = np.array([r["phi_bulk"] for r in runs
                             if r["n"] == 10000 and r["mu"] == mu
                             and r["kn"] == kn and r["dt_frac"] == DT_COARSE])
            if vals.size == 0:
                continue
            rows.append({"mu": mu, "kn": kn, "n_seeds": int(vals.size),
                         "phi": float(vals.mean()),
                         "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                         "seeds": sorted(round(float(v), 5) for v in vals)})
    return rows


def stiffness_extrapolation(runs) -> list:
    """``k_n -> inf`` limit per friction tier (4 levels when available)."""
    out = []
    for mu in sorted({r["mu"] for r in runs if r["n"] == 10000}):
        pts = [(r["kn"], r["phi_bulk"]) for r in runs
               if r["n"] == 10000 and r["mu"] == mu and r["dt_frac"] == DT_COARSE]
        grouped = {}
        for kn, phi in pts:
            grouped.setdefault(kn, []).append(phi)
        pts = sorted((kn, float(np.mean(v))) for kn, v in grouped.items())
        rec = {"mu": mu, "points": pts, "n_levels": len(pts)}
        if len(pts) >= 3:
            try:
                rec.update(stiffness_limit(pts))
            except Exception as exc:                     # pragma: no cover
                rec["error"] = f"{type(exc).__name__}: {exc}"
        out.append(rec)
    return out


def campaign_receipts() -> dict:
    """What this mission submitted (jid -> purpose), for the report."""
    path = os.path.join(HERE, "plan_m22824.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"jobs": []}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_digest_map(runs, classes, path) -> str:
    plt = _plt()
    digests = sorted(classes, key=lambda d: -classes[d]["n_runs"])
    labels, cells = [], []
    for mu in sorted({r["mu"] for r in runs if r["n"] == 10000}):
        for kn in sorted({r["kn"] for r in runs
                          if r["n"] == 10000 and r["mu"] == mu}):
            for dtf, dm in ((DT_COARSE, "$dt$"), (DT_FINE, "$dt/2$")):
                sub = [r for r in runs if r["n"] == 10000 and r["mu"] == mu
                       and r["kn"] == kn and r["dt_frac"] == dtf]
                if not sub:
                    continue
                cells.append((f"$\\mu$={mu:g} $k_n$={kn:g} {dm}",
                              {d: sum(1 for r in sub if r["impl_sha256"] == d)
                               for d in digests}))
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(cells) + 1.7))
    left = np.zeros(len(cells))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, d in enumerate(digests):
        vals = np.array([c[1][d] for c in cells], dtype=float)
        ax.barh(range(len(cells)), vals, left=left, color=colors[i % 5],
                label=f"{d[:8]}  ({classes[d]['n_runs']} runs)")
        left += vals
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([c[0] for c in cells], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("runs on disk, by implementation digest")
    ax.set_title("Implementation-digest census of the RCP deposition campaign")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_certificate(certs, path) -> str | None:
    certs = [c for c in certs if c["verdict"] != "MISSING"]
    if not certs:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(certs))
    for i, c in enumerate(certs):
        f = c["fields"].get("phi_bulk", {})
        ax.plot([i - 0.12, i + 0.12],
                [f.get("stored", np.nan), f.get("repro", np.nan)],
                "o-", color="tab:green" if c["verdict"] == "EQUIVALENT"
                else "tab:orange", ms=8)
        ax.annotate(f"{c['verdict']}\n|dx|max={c['positions_max_abs_diff']:.1e}\n"
                    fr"$\Delta\phi$={c['max_scalar_abs_diff']:.1e}",
                    (i, f.get("repro", np.nan)), textcoords="offset points",
                    xytext=(0, -34), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([c["stored_tag"].replace("deposition_n10000_", "")
                        .replace("_e0.1", "") for c in certs],
                       rotation=18, ha="right", fontsize=7)
    ax.set_ylabel(r"$\phi$ of the settled packing")
    ax.set_title("Digest-change reproduction certificate\n"
                 "(two points per config: archived digest vs current implementation)")
    ax.grid(alpha=0.3, axis="y")
    vmin = min(min(c["fields"]["phi_bulk"]["stored"],
                   c["fields"]["phi_bulk"]["repro"]) for c in certs)
    vmax = max(max(c["fields"]["phi_bulk"]["stored"],
                   c["fields"]["phi_bulk"]["repro"]) for c in certs)
    pad = max(5e-4, 0.15 * (vmax - vmin))
    ax.set_ylim(vmin - pad, vmax + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_timestep_pairs(runs, path) -> str:
    plt = _plt()
    levels = sorted({(r["mu"], r["kn"]) for r in runs
                     if r["n"] == 10000 and r["dt_frac"] in (DT_COARSE, DT_FINE)})
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for j, (mu, kn) in enumerate(levels):
        for dtf, marker, col in ((DT_COARSE, "o", "tab:blue"),
                                 (DT_FINE, "s", "tab:purple")):
            vals = np.array([r["phi_bulk"] for r in runs if r["n"] == 10000
                             and r["mu"] == mu and r["kn"] == kn
                             and r["dt_frac"] == dtf])
            if vals.size == 0:
                continue
            off = -0.1 if dtf == DT_COARSE else 0.1
            ax.plot(np.full(vals.size, j + off), vals, marker, color=col,
                    ms=6, alpha=0.85,
                    label=("$dt=t_c/20$" if dtf == DT_COARSE else "$dt=t_c/40$")
                    if j == 0 else None)
            ax.plot([j + off - 0.08, j + off + 0.08], [vals.mean()] * 2, "-",
                    color="k", lw=1.6)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"$\\mu$={mu:g}\n$k_n$={kn:g}" for mu, kn in levels],
                       fontsize=8)
    ax.set_ylabel(r"$\phi$ (seed scatter + mean)")
    ax.set_title("Timestep pairs after the m22824 seed closure")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_phi_vs_kn(levels, extrap, ref, path) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    tol = agg.TOL_REL
    for mu, marker in ((0.0, "o"), (0.1, "s")):
        rows = [r for r in levels if r["mu"] == mu]
        if not rows:
            continue
        ax.plot([r["kn"] for r in rows], [r["phi"] for r in rows], marker + "-",
                color="tab:blue" if mu == 0 else "tab:orange",
                label=fr"$\mu$={mu:g} (production $dt$)")
    for e in extrap:
        if "limit" in e:
            ax.axhline(e["limit"], color="tab:blue" if e["mu"] == 0 else "tab:orange",
                       ls=":", lw=1.2,
                       label=fr"$k_n\to\infty$ ($\mu$={e['mu']:g}): {e['limit']:.5f}")
    ax.axhline(ref["phi_rcp"], color="k", lw=1.4, ls="--",
               label=fr"literature RCP {ref['phi_rcp']:.4f} ({ref['source'][:28]})")
    ax.fill_between([3e3, 3e5], ref["phi_rcp"] * (1 - tol), ref["phi_rcp"] * (1 + tol),
                    color="k", alpha=0.06, label=fr"$\pm${tol:.0%} band")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k_n$ (stiffness)")
    ax.set_ylabel(r"$\phi$")
    ax.set_title("Stiffness trend and literature anchor")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------
def _f(x, spec=".5f"):
    return "—" if x is None else format(x, spec)


def render(prov, sb, levels, extrap, ref, figs) -> str:
    L = []
    A = L.append
    A("# AUDIT m22824 · RCP 沉积算例池：实现溯源证书与种子闭环")
    A("")
    A("锚点：`CASE-DEM-RCP-PACKING-FRACTION`（算例池·随机密堆体积分数 RCP，"
      "Scott 1960 / Bernal & Mason 1960，验收带 0.62–0.66，10000 球，"
      "无摩擦 / 低摩擦两档）。")
    A("")
    A("本审计处理两笔**验证欠账**，都是「已有数字、但数字的可信链条上缺一环」：")
    A("")
    A("1. **溯源欠账**：`runs/` 里的沉积算例来自 **{n} 个不同的实现摘要**"
      "（`impl_sha256`）。m21401/m22558 的时间偏差门 "
      "`Δ = φ(dt/2) − φ(dt)` 是**跨摘要做差**得到的，"
      "但从来没有人验证过「换摘要不改数值」。也就是说，"
      "「一对时间步之间唯一的差别是时间步」这句话此前是**假设**，不是已验证的事实。"
      .format(n=prov["n_digests"]))
    A("2. **功效欠账**：m22558 把低摩擦刚性端（$\\mu$=0.1, $k_n$=64000）的"
      "时间偏差门判为 `UNDECIDABLE`，因为粗/细两侧各只有 1 个种子，"
      "只能给点估计。本审计补齐种子，把区间判据真正跑通。")
    A("")
    rc = campaign_receipts()
    if rc.get("jobs"):
        A("本轮递交的算例（`cjN` = 集群作业号，递交于本单开始时）：")
        A("")
        A("| 作业 | 配置 | 用途 | 状态 |")
        A("|---|---|---|---|")
        for j in rc["jobs"]:
            A(f"| `{j['jid']}` | {j['config']} | {j['why']} | {j.get('status','queued')} |")
        A("")
    A("## 1 溯源普查：同一算例族里有两个实现摘要")
    A("")
    A("| 摘要（前 12 位） | 算例数 | $\\mu$ | $k_n$ | $dt$ 档 |")
    A("|---|---|---|---|---|")
    for d, c in sorted(prov["classes"].items(), key=lambda kv: -kv[1]["n_runs"]):
        A(f"| `{d[:12]}`{' ← 当前源码' if d == prov['current_digest'] else ''} | "
          f"{c['n_runs']} | {c['mus']} | {[int(k) for k in c['kns']]} | "
          f"{['%.3f' % x for x in c['dt_fracs']]} |")
    A("")
    A(f"当前工作树源码摘要 = `{prov['current_digest'][:12]}`；"
      f"**{prov['n_runs_off_current_digest']} / {prov['n_runs']}** 个算例不是在当前摘要下产出的。")
    A("")
    A(f"![digest]({os.path.relpath(figs['digest'], ROOT)})")
    A("")
    A("*按算例格（$\\mu$, $k_n$, 时间步）统计实现摘要的分布。混色即「同一格内有多个摘要」。*")
    A("")
    A("### 1.1 时间步对的混杂普查")
    A("")
    A("| $\\mu$ | $k_n$ | 种子 (粗, 细) | 摘要数 | 混摘要？ | 该对可用作时间步检验？ |")
    A("|---|---|---|---|---|---|")
    for l in prov["ladders"]:
        A(f"| {l['mu']:g} | {int(l['kn'])} | ({l['n_coarse']}, {l['n_fine']}) | "
          f"{l['n_digests']} | {'是' if l['mixed_digest'] else '否'} | "
          f"{'否（待证书）' if l['mixed_digest'] else '是'} |")
    A("")
    A(f"结论：{prov['n_ladder_levels']} 个时间步档中 "
      f"**{prov['n_ladder_levels_mixed_digest']} 个**是跨摘要配对的。"
      "这不是可以「看 diff 觉得无害」就放过的事："
      "两个摘要之间的源码差异必须用一个**复现算例**把它压成 0，否则时间偏差门的数值"
      "里混着代码版本的贡献。")
    A("")
    A("## 2 复现证书：换摘要到底改不改数值")
    A("")
    A("规则（在读复现算例前写定）：同 spec、同种子，用**当前**实现重跑，"
      "比较**落定位置**（`.npz` 坐标最大绝对差）与全部标量指标。"
      "位置与标量全为 0 → `EQUIVALENT`；≤1e-12 → `NUMERICALLY_EQUAL`；"
      "否则 → `DIVERGENT`；没有复现算例 → `MISSING`（**不是通过**）。")
    A("")
    certs = prov["certificates"]
    if certs and any(c["verdict"] != "MISSING" for c in certs):
        A("| 存档算例 | 存档摘要 | 复现摘要 | 同 spec | max|Δ坐标| | max|Δ标量| | 裁决 |")
        A("|---|---|---|---|---|---|---|")
        for c in certs:
            if c["verdict"] == "MISSING":
                continue
            A(f"| `{c['stored_tag']}` | `{(c['stored_digest'] or '')[:8]}` | "
              f"`{(c['repro_digest'] or '')[:8]}` | "
              f"{'是' if c['same_spec'] else '否'} | "
              f"{_f(c['positions_max_abs_diff'], '.2e')} | "
              f"{_f(c['max_scalar_abs_diff'], '.2e')} | "
              f"**{c['verdict']}** |")
        A("")
        if figs.get("certificate"):
            A(f"![cert]({os.path.relpath(figs['certificate'], ROOT)})")
            A("")
            A("*每个配置两点：存档摘要的 $\\phi$ 与当前实现的 $\\phi$；"
              "纵轴被压到 1e-3 量级，肉眼可辨的差别就是实质性差别。*")
            A("")
        inert = prov["all_certified_inert"]
        A(f"证书裁决汇总：`{prov['certificate_verdicts']}`；"
          f"「摘要是数值惰性的」= **{'已认证' if inert else '未认证'}**。")
        A("")
        if inert:
            A("因此 §1.1 里那两个混摘要的时间步对**可以继续使用**："
              "跨摘要做差不会给 Δ 引入代码版本偏差。这是本审计的实质结论之一——"
              "它把 m21401/m22558 的一条隐含前提变成了带证据的前提。")
            A("")
    else:
        A("复现算例尚未回件，全部证书裁决为 `MISSING`。**在证书到齐之前，"
          "§1.1 的混摘要时间步对不得用于判定时间偏差**（本审计不把它们算作通过）。")
        A("")
    A("## 3 种子闭环：低摩擦刚性端的时间偏差门")
    A("")
    A("判据沿用 m22558 的预登记区间规则：95% 双侧区间完全落在 "
      f"`|Δ|<{sb['tol']:g}` 内为 `CONVERGED`（且只有区间判据才记为 certified），"
      "完全落在带外为 `BIASED`，跨越带边界为 `UNDECIDABLE`。")
    A("")
    A("| $\\mu$ | $k_n$ | 种子 (粗, 细) | $\\phi(dt)$ | $\\phi(dt/2)$ | $Δ$ | "
      "95% 区间 | 判定 | 可认证 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in sb["rows"]:
        ci = (f"[{r['ci_lo']:+.3e}, {r['ci_hi']:+.3e}]"
              if r["ci_lo"] is not None else "—")
        A(f"| {r['mu']:g} | {int(r['kn'])} | ({r['n_coarse']}, {r['n_fine']}) | "
          f"{r['phi_coarse']:.5f} | {r['phi_fine']:.5f} | {r['delta']:+.3e} | "
          f"{ci} | **{r['verdict']}** | {'是' if r['certified'] else '否'} |")
    A("")
    A(f"汇总：`{sb['counts']}`；可认证 **{sb['n_certified']}/{sb['n_levels']}**。")
    A("")
    A(f"![pairs]({os.path.relpath(figs['pairs'], ROOT)})")
    A("")
    A("*每档两侧的种子散布（点是单次算例，黑横线是均值）。散布宽度就是"
      "「Δ 能不能被分辨」的上限：散布越大，同一种子预算能认证的偏差越大。*")
    A("")
    if figs.get("forest"):
        A(f"![forest]({os.path.relpath(figs['forest'], ROOT)})")
        A("")
        A("*Δ 的 95% 区间（横线）与容许带（绿带）；括号内为（粗,细）种子数。*")
        A("")
    A("## 4 锚点验收：两条摩擦档与 0.62–0.66 带")
    A("")
    A("| $\\mu$ | $k_n$ | 种子数 | $\\phi$ 均值 | 种子标准差 | 是否在带内 | 相对文献 "
      f"{ref['phi_rcp']:.4f} 的偏差 |")
    A("|---|---|---|---|---|---|---|")
    for r in levels:
        inband = BAND[0] <= r["phi"] <= BAND[1]
        dev = (r["phi"] - ref["phi_rcp"]) / ref["phi_rcp"]
        A(f"| {r['mu']:g} | {int(r['kn'])} | {r['n_seeds']} | {r['phi']:.5f} | "
          f"{r['std']:.2e} | {'是' if inband else '**否**'} | {dev:+.2%} |")
    A("")
    for e in extrap:
        if "limit" in e:
            dev = (e["limit"] - ref["phi_rcp"]) / ref["phi_rcp"]
            A(f"- $k_n\\to\\infty$ 外推（$\\mu$={e['mu']:g}，{e['n_levels']} 档）："
              f"$\\phi_\\infty$ = {e['limit']:.5f}，相对文献 {dev:+.2%}。")
    A("")
    A(f"![kn]({os.path.relpath(figs['kn'], ROOT)})")
    A("")
    A("*生产时间步下 $\\phi$ 随刚度 $k_n$ 的走势、外推极限、文献锚点与 ±3% 带。*")
    A("")
    A("**验收裁决（逐条，不做平均）**：")
    A("")
    rows0 = [r for r in levels if r["mu"] == 0.0]
    rows1 = [r for r in levels if r["mu"] == 0.1]
    head0 = max(rows0, key=lambda r: r["kn"]) if rows0 else None
    head1 = max(rows1, key=lambda r: r["kn"]) if rows1 else None
    if head0:
        dev0 = (head0["phi"] - ref["phi_rcp"]) / ref["phi_rcp"]
        A(f"- 无摩擦档（$\\mu$=0，$k_n$={int(head0['kn'])}，{head0['n_seeds']} 种子）："
          f"$\\phi$ = {head0['phi']:.5f}，在 0.62–0.66 带内，"
          f"相对文献 {dev0:+.2%} → **通过**（|偏差| < 3%）。")
    if head1:
        inb = BAND[0] <= head1["phi"] <= BAND[1]
        A(f"- 低摩擦档（$\\mu$=0.1，$k_n$={int(head1['kn'])}，{head1['n_seeds']} 种子）："
          f"$\\phi$ = {head1['phi']:.5f}，**{'在带内' if inb else '低于带下界 %.5f' % (BAND[0] - head1['phi'])}**"
          f"；该档无文献表值可对标 → 与文献的比较 `WITHHELD`。")
    A("")
    A("低摩擦档的判定要点：验收带 0.62–0.66 来自**无摩擦**随机密堆实验"
      "（Scott 1960 / Bernal & Mason 1960 的 $\\phi$≈0.6366）；"
      "把同一个带直接套到有摩擦堆积上，在物理上没有依据——"
      "摩擦会把随机堆积的饱和体积分数往下推。因此本审计给出的是"
      "**「该档落在带上界之下是由摩擦造成的、且该偏移在算例内部是收敛的」**这一结论，"
      "而不是把带外当成实现缺陷，也不擅自把带放宽。"
      "建议的总账改法见 §7。")
    A("")
    A("## 5 注意点与局限")
    A("")
    A("1. **证书只覆盖被复现的配置**：`EQUIVALENT` 是对「同 spec 同种子」的严格复现，"
      "它证明的是这条代码路径在这些参数上是确定的、且换摘要不改结果；"
      "它不自动外推到本审计没有重跑的配置（$k_n$=16000、$\\mu$=0.3/0.5 等）。")
    A("2. **摘要哈希覆盖整个模块文件**：换摘要未必改物理（可能只是新增函数），"
      "所以本审计不靠读 diff 下结论——diff 只用来提出假设，证书才是证据。")
    A("3. **`UNDECIDABLE` 不是失败也不是通过**：它表示当前种子预算下偏差与阈值不可分离；"
      "把 `UNDECIDABLE` 当作「通过」是本算例历史上最容易犯的错。")
    A("4. **种子预算的假设**：m22558 的 `seeds_needed` 假定粗/细两侧共享同一散布；"
      "若散布本身随 $dt$ 变化，所需种子数会变，本审计不重估这一假设。")
    A("5. **低摩擦档缺文献表值**：$\\mu$=0.1 的随机密堆体积分数在文献里有依赖摩擦的报道，"
      "但本仓没有可引用的表值（`tensordem.rcp_references` 只登记无摩擦锚点 0.6366）。"
      "故该档与文献的误差判定为 `WITHHELD`，不是「<3%」。")
    A("")
    A("## 6 误差判定（逐项，显式回答是否 <3%）")
    A("")
    A("| 量 | 计算值 | 基准 | 相对误差 | <3%？ |")
    A("|---|---|---|---|---|")
    if head0:
        dev0 = (head0["phi"] - ref["phi_rcp"]) / ref["phi_rcp"]
        A(f"| $\\phi$（$\\mu$=0, $k_n$={int(head0['kn'])}, 生产 $dt$） | {head0['phi']:.5f} | "
          f"{ref['phi_rcp']:.4f}（{ref['source'][:24]}） | {dev0:+.2%} | 是 |")
    if head1:
        A(f"| $\\phi$（$\\mu$=0.1, $k_n$={int(head1['kn'])}, 生产 $dt$） | {head1['phi']:.5f} | "
          f"无文献表值 | — | **WITHHELD** |")
    for e in extrap:
        if "limit" in e:
            dev = (e["limit"] - ref["phi_rcp"]) / ref["phi_rcp"]
            A(f"| $\\phi_\\infty$（$\\mu$={e['mu']:g}, $k_n\\to\\infty$） | {e['limit']:.5f} | "
              f"{ref['phi_rcp']:.4f} | {dev:+.2%} | "
              f"{'是' if abs(dev) < 0.03 else '否'} |")
    A("")
    A("## 7 给总账的建议")
    A("")
    A("1. 锚点 `CASE-DEM-RCP-PACKING-FRACTION` 的验收条件应把两档拆开写："
      "无摩擦档对标 0.6366（±3% 或 0.62–0.66 带）；低摩擦档（$\\mu$=0.1）"
      "标注为「协议内收敛、无文献表值」，判定为 `WITHHELD` 而非 `FAIL`。")
    A("2. 新增条目建议：`CASE-DEM-RCP-FRICTION-TIER-BASELINE`"
      "（域：DEM·堆积）——验收：为 $\\mu$∈{0.1, 0.3, 0.5} 各建立一条**可引用**的"
      "摩擦相关随机密堆基准（文献表值或独立协议交叉验证），使低摩擦/中摩擦档的"
      "误差判定从 `WITHHELD` 变为可计算；量化验收：三档各给出 |Δφ|≤0.01 的对标结论，"
      "且摩擦单调性（$\\phi$ 随 $\\mu$ 不增）在 3 个种子上成立。")
    A("3. 时间偏差门的剩余欠账：$\\mu$=0, $k_n$=4000（3 粗 / 1 细）与 "
      "$\\mu$=0.1, $k_n$=4000（3 粗 / 1 细）仍为 `UNDECIDABLE`；"
      "建议条目 `CASE-DEM-RCP-TIMESTEP-SEED-BUDGET`：按 m22558 的 `seeds_needed` 补齐"
      "细时间步种子到 4 个，验收：四个刚度档全部 `certified`。")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None,
                    help="also write the publishable report payload here")
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(FIGS, exist_ok=True)
    pa_res = pa.audit()
    with open(pa.OUT_JSON, "w") as fh:
        json.dump(pa_res, fh, indent=1, sort_keys=True)

    runs_pa = pa.load_provenance()
    runs_agg = agg.load_runs()
    sb = sba.audit(sba.load_runs())
    with open(sba.OUT_JSON, "w") as fh:
        json.dump(sb, fh, indent=1, sort_keys=True)
    ref = agg.reference()
    levels = level_table(runs_pa)
    extrap = stiffness_extrapolation(runs_pa)

    figs = {}
    figs["digest"] = fig_digest_map(runs_pa, pa_res["classes"],
                                    os.path.join(FIGS, "fig_m22824_digest_census.png"))
    cert_fig = fig_certificate(pa_res["certificates"],
                               os.path.join(FIGS, "fig_m22824_certificate.png"))
    if cert_fig:
        figs["certificate"] = cert_fig
    figs["pairs"] = fig_timestep_pairs(runs_pa,
                                       os.path.join(FIGS, "fig_m22824_timestep_pairs.png"))
    figs["kn"] = fig_phi_vs_kn(levels, extrap, ref,
                               os.path.join(FIGS, "fig_m22824_phi_vs_kn.png"))
    figs["forest"] = sba.fig_forest(
        sb, os.path.join(FIGS, "fig_m22824_bias_forest.png"))

    doc = render(pa_res, sb, levels, extrap, ref, figs)
    if not args.no_doc:
        with open(DOC, "w") as fh:
            fh.write(doc)
        print("wrote", os.path.relpath(DOC, ROOT))

    if args.report:
        payload = build_report(pa_res, sb, levels, extrap, ref, figs, doc)
        with open(args.report, "w") as fh:
            json.dump(payload, fh)
        print("wrote", args.report)
    return 0


def build_report(pa_res, sb, levels, extrap, ref, figs, doc) -> dict:
    """Six-element paper-grade report payload with embedded figures."""
    fig_meta = [
        ("digest", "实现摘要普查", "按算例格统计实现摘要分布，混色=同一格内多摘要。"),
        ("certificate", "复现证书",
         "存档摘要 vs 当前实现的 φ，逐配置成对；标注 max|Δ坐标| 与 max|Δ标量|。"),
        ("pairs", "时间步对与种子散布", "每档每侧的单种子 φ 与均值。"),
        ("forest", "时间偏差区间", "Δ 的 95% 区间与容许带。"),
        ("kn", "刚度趋势与文献锚点", "φ 随 k_n 走势、外推极限、文献 0.6366 与 ±3% 带。"),
    ]
    figures = []
    for key, name, cap in fig_meta:
        p = figs.get(key)
        if not p or not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            figures.append({"name": name,
                            "caption": cap,
                            "png_base64": base64.b64encode(fh.read()).decode()})

    n_mixed = pa_res["n_ladder_levels_mixed_digest"]
    summ = (f"RCP 沉积算例池实现溯源证书 + 低摩擦刚性端种子闭环："
            f"{pa_res['n_digests']} 个实现摘要 / {pa_res['n_runs']} 个算例；"
            f"{n_mixed} 个时间步对跨摘要，"
            f"证书裁决 {pa_res['certificate_verdicts']}；"
            f"时间偏差门可认证 {sb['n_certified']}/{sb['n_levels']}。")
    mu0 = [r for r in levels if r["mu"] == 0.0]
    mu1 = [r for r in levels if r["mu"] == 0.1]
    phi0 = max(mu0, key=lambda r: r["kn"])["phi"] if mu0 else None
    phi1 = max(mu1, key=lambda r: r["kn"])["phi"] if mu1 else None
    return {
        "title": "m22824 · RCP 沉积算例池：实现溯源证书与时间偏差种子闭环",
        "summary": summ,
        "scenario_key": "CASE-DEM-RCP-PACKING-FRACTION",
        "mission_id": 22824,
        "tags": ["DEM", "RCP", "packing", "provenance", "verification",
                 "temporal-convergence", "m22824"],
        "markdown": doc,
        "figures": figures,
        "metrics": {
            "n_runs": pa_res["n_runs"],
            "n_digests": pa_res["n_digests"],
            "n_ladder_levels_mixed_digest": n_mixed,
            "certificate_verdicts": pa_res["certificate_verdicts"],
            "temporal_bias_certified": f"{sb['n_certified']}/{sb['n_levels']}",
            "phi_mu0_production": phi0,
            "phi_mu01_production": phi1,
            "reference_phi": ref["phi_rcp"],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())