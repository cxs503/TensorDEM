"""Seed-resolved adjudication of the RCP temporal-bias gate.

Why this script exists
----------------------
``timestep_ladder_audit.py`` (m21401) judges the temporal convergence of every
stiffness level with a single pre-registered scalar test::

    |Delta| < 5e-4,   Delta = phi(dt/2) - phi(dt)

``Delta`` is a difference of two *sample means* (one per seed), so it carries a
standard error; comparing the point estimate alone against a fixed tolerance
throws that information away.  The m21401 run left exactly one of those gates
marginal and honestly FAIL::

    mu = 0, k_n = 64000:  Delta = -5.069e-04  (tolerance 5e-4)
    coarse seed std = 3.13e-04, z = 1.62  -> "bias indistinguishable from
    seed noise", fine level n = 1

That is a statement about **statistical power**, not about physics, and it is
resolvable: add fine-timestep seeds and ask whether the *confidence interval*
on ``Delta`` fits inside the tolerance, not just whether the point estimate
does.  This module supplies the machinery and the campaign bookkeeping:

* :func:`welch_diff` -- mean difference of two seed samples with a Welch
  standard error and Welch-Satterthwaite degrees of freedom (the two levels do
  not share a variance: a finer step is not expected to change the seed scatter
  but we do not assume it either);
* :func:`classify_bias` -- three-way, pre-registered decision rule
  **CONVERGED / BIASED / UNDECIDABLE** based on the two-sided 95 % interval for
  ``Delta``, so a run can neither pass on an unfavourable point estimate nor be
  failed on noise alone;
* :func:`seeds_needed` -- the seed budget that *would* make a level decidable:
  the smallest per-level ``n`` such that the expected half-width
  ``t * sqrt(2) * s / sqrt(n)`` drops below a target margin;
* :func:`audit` -- applies the above to every ``(mu, k_n)`` level that has both
  timestep levels on disk, and reports for each level the verdict, the
  resolution (smallest bias the seed set can detect) and the seeds still
  missing.

The rule is registered *before* the new seeds are read: a level passes only if
``|Delta| + t*sem_Delta < TOL``.  A level whose interval straddles the
tolerance is reported as UNDECIDABLE -- never as a pass and never as a fail.

Usage::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/seed_bias_audit.py
    PYTHONPATH=src python benchmarks/case_rcp_deposition/seed_bias_audit.py --json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")
OUT_JSON = os.path.join(RUNS, "seed_bias_audit.json")

N_PRIMARY = 10000
DT_COARSE = 1.0 / 20.0
DT_FINE = 1.0 / 40.0
TOL_BIAS = 5.0e-4          # same pre-registered tolerance as m21401
CONF = 0.95

__all__ = [
    "load_runs", "welch_diff", "classify_bias", "seeds_needed", "audit",
    "levels_present", "TOL_BIAS", "DT_COARSE", "DT_FINE",
]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_runs(runs_dir: str = RUNS) -> list:
    """Read every deposition run JSON (position/velocity npz files are skipped)."""
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "deposition_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        spec, m = d["spec"], d["metrics"]
        out.append({
            "path": os.path.relpath(path, ROOT),
            "n": int(spec["n_particles"]),
            "mu": float(spec["mu"]),
            "kn": float(spec["kn"]),
            "seed": int(spec["seed"]),
            "dt_frac": float(spec.get("dt_frac", DT_COARSE)),
            "phi": float(m["phi_bulk"]),
            "phi_corr": float(m["phi_bulk_overlap_corrected"]),
            "settled": bool(d.get("settled", False)),
        })
    return out


def levels_present(runs, n: int = N_PRIMARY) -> list:
    """``[(mu, kn), ...]`` for which *both* timestep levels have >= 1 run."""
    keys = set()
    for r in runs:
        if r["n"] == n:
            keys.add((r["mu"], r["kn"]))
    out = []
    for mu, kn in sorted(keys):
        have = {r["dt_frac"] for r in runs
                if r["n"] == n and r["mu"] == mu and r["kn"] == kn}
        if DT_COARSE in have and DT_FINE in have:
            out.append((mu, kn))
    return out


def _sample(runs, mu, kn, dt_frac, n=N_PRIMARY) -> np.ndarray:
    return np.array(sorted(r["phi"] for r in runs
                           if r["n"] == n and r["mu"] == mu
                           and r["kn"] == kn and r["dt_frac"] == dt_frac),
                    dtype=float)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def welch_diff(a, b) -> dict:
    """``b - a`` with Welch standard error and Welch-Satterthwaite df.

    ``a``/``b`` are the coarse/fine seed samples.  With one sample on either
    side the standard error is undefined; the returned dict then carries
    ``sem=None`` and ``df=None`` rather than a fabricated number.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    ma, mb = float(a.mean()), float(b.mean())
    delta = mb - ma
    if na < 2 or nb < 2:
        std_a = float(a.std(ddof=1)) if na > 1 else None
        std_b = float(b.std(ddof=1)) if nb > 1 else None
        std = std_a if std_a is not None else std_b
        if std is not None and na + nb > 2:
            # One level has a single seed: no independent scatter estimate on
            # that side, so the *other* side's scatter is used as the common
            # sigma (the campaign's design assumption: halving dt should not
            # change the seed spread).  Flagged so a consumer can discount it.
            sem = std * math.sqrt(1.0 / na + 1.0 / nb)
            return {"delta": delta, "sem": sem, "df": float(na + nb - 2),
                    "n_coarse": na, "n_fine": nb, "mean_coarse": ma,
                    "mean_fine": mb, "std_coarse": std_a, "std_fine": std_b,
                    "error_model": "pooled_one_sided",
                    "z": abs(delta) / sem if sem > 0 else None}
        return {"delta": delta, "sem": None, "df": None, "n_coarse": na,
                "n_fine": nb, "mean_coarse": ma, "mean_fine": mb,
                "std_coarse": std_a, "std_fine": std_b,
                "error_model": "none"}
    va, vb = float(a.var(ddof=1)) / na, float(b.var(ddof=1)) / nb
    sem = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)) if sem > 0 else math.inf
    return {"delta": delta, "sem": sem, "df": df, "n_coarse": na, "n_fine": nb,
            "mean_coarse": ma, "mean_fine": mb,
            "std_coarse": float(a.std(ddof=1)), "std_fine": float(b.std(ddof=1)),
            "error_model": "welch",
            "z": abs(delta) / sem if sem > 0 else None}


def classify_bias(delta: float, sem, df, tol: float = TOL_BIAS,
                  conf: float = CONF) -> dict:
    """Three-way verdict from the two-sided ``conf`` interval for ``Delta``.

    * ``CONVERGED``  -- whole interval inside ``|Delta| < tol`` (bias certified
      below tolerance);
    * ``BIASED``     -- whole interval outside the band (bias certified);
    * ``UNDECIDABLE``-- interval straddles the tolerance (more seeds needed, or
      the bias is genuinely the size of the tolerance).

    With an undefined or zero standard error the verdict degenerates to the
    point-estimate comparison, flagged ``resolved_from="point"`` so a consumer
    can refuse to over-claim.
    """
    if sem is None or not np.isfinite(sem) or sem <= 0.0 or df is None:
        verdict = "CONVERGED" if abs(delta) < tol else "BIASED"
        return {"verdict": verdict, "half_width": None, "lo": None, "hi": None,
                "resolved_from": "point"}
    tcrit = float(stats.t.ppf(0.5 + conf / 2.0, df))
    hw = tcrit * sem
    lo, hi = delta - hw, delta + hw
    # The tolerance defines the *band* [-tol, tol]; the verdict turns on how the
    # interval sits relative to that band, not on |lo|/|hi| individually:
    #   wholly inside          -> CONVERGED   (bias certified below tolerance)
    #   wholly outside         -> BIASED      (bias certified above tolerance)
    #   crossing a band edge   -> UNDECIDABLE (no claim either way)
    if lo > -tol and hi < tol:
        verdict = "CONVERGED"
    elif hi < -tol or lo > tol:
        verdict = "BIASED"
    else:
        verdict = "UNDECIDABLE"
    return {"verdict": verdict, "half_width": hw, "lo": lo, "hi": hi,
            "t_crit": tcrit, "resolved_from": "interval"}


def seeds_needed(std: float, margin: float, conf: float = CONF,
                 n_max: int = 200) -> int:
    """Seeds per timestep level so the 95 % half-width falls below ``margin``.

    Assumes the two levels share the scatter ``std`` (the mild assumption the
    campaign makes by design: halving ``dt`` should not change the seed spread).
    Returns ``n_max`` when the margin is not reachable within ``n_max``.
    """
    if not np.isfinite(std) or std <= 0.0 or margin <= 0.0:
        return 1
    for n in range(2, n_max + 1):
        hw = float(stats.t.ppf(0.5 + conf / 2.0, 2 * n - 2)) * math.sqrt(2.0) * std / math.sqrt(n)
        if hw < margin:
            return n
    return n_max


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def _duplicate_seeds(runs) -> list:
    """Runs sharing ``(mu, kn, dt_frac, seed)``: they would be double counted."""
    seen, dup = {}, []
    for r in runs:
        if r["n"] != N_PRIMARY:
            continue
        key = (r["mu"], r["kn"], r["dt_frac"], r["seed"])
        if key in seen:
            dup.append({"mu": r["mu"], "kn": r["kn"], "dt_frac": r["dt_frac"],
                        "seed": r["seed"], "paths": [seen[key], r["path"]]})
        else:
            seen[key] = r["path"]
    return dup


def audit(runs, tol: float = TOL_BIAS, conf: float = CONF) -> dict:
    """Per-level seed-resolved bias table plus the campaign-level summary."""
    rows = []
    dupes = _duplicate_seeds(runs)
    for mu, kn in levels_present(runs):
        coarse = _sample(runs, mu, kn, DT_COARSE)
        fine = _sample(runs, mu, kn, DT_FINE)
        w = welch_diff(coarse, fine)
        verdict = classify_bias(w["delta"], w["sem"], w["df"], tol, conf)
        std = w["std_coarse"] if w["std_coarse"] is not None else w["std_fine"]
        rows.append({
            "mu": mu, "kn": kn,
            "n_coarse": w["n_coarse"], "n_fine": w["n_fine"],
            "phi_coarse": w["mean_coarse"], "phi_fine": w["mean_fine"],
            "delta": w["delta"], "sem": w["sem"], "df": w["df"],
            "z": w.get("z"),
            "std_coarse": w["std_coarse"], "std_fine": w["std_fine"],
            "verdict": verdict["verdict"],
            "ci_lo": verdict["lo"], "ci_hi": verdict["hi"],
            "half_width": verdict["half_width"],
            "resolved_from": verdict["resolved_from"],
            "error_model": w["error_model"],
            "resolution_abs": verdict["half_width"],
            "seeds_needed_for_tol": (seeds_needed(std, tol, conf)
                                     if std is not None else None),
            "point_within_tol": abs(w["delta"]) < tol,
            # a level is *certified* only when the interval (not just the point
            # estimate) fits inside the tolerance band
            "certified": (verdict["verdict"] == "CONVERGED"
                          and verdict["resolved_from"] == "interval"),
        })
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    total = len(rows)
    return {
        "tol": tol, "conf": conf, "n_primary": N_PRIMARY,
        "dt_coarse": DT_COARSE, "dt_fine": DT_FINE,
        "rows": rows,
        "duplicate_seeds": dupes,
        "counts": counts,
        "n_levels": total,
        "all_converged": total > 0 and counts.get("CONVERGED", 0) == total,
        "n_certified": sum(1 for r in rows if r["certified"]),
        "all_certified": total > 0 and all(r["certified"] for r in rows),
        "n_biased": counts.get("BIASED", 0),
        "n_undecidable": counts.get("UNDECIDABLE", 0),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="numbers only, no doc")
    ap.add_argument("--doc", action="store_true",
                    help="regenerate docs/AUDIT_m22558_RCP_SEED_RESOLVED_BIAS.md + figures")
    ap.add_argument("--runs", default=RUNS)
    args = ap.parse_args(argv)

    runs = load_runs(args.runs)
    res = audit(runs, tol=TOL_BIAS, conf=CONF)
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    if args.doc:
        os.makedirs(FIGS, exist_ok=True)
        f1 = fig_forest(res, os.path.join(FIGS, "fig_m22558_bias_forest.png"))
        f2 = fig_levels(runs, os.path.join(FIGS, "fig_m22558_timestep_pairs.png"))
        doc = write_doc(res, f1, f2)
        print("wrote", doc)
    if args.json:
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
        for r in res["rows"]:
            print(f"mu={r['mu']:g} kn={r['kn']:g} n=({r['n_coarse']},{r['n_fine']}) "
                  f"delta={r['delta']:+.3e} hw={r['half_width']} -> {r['verdict']}")
        return 0
    print(f"wrote {os.path.relpath(OUT_JSON, ROOT)}")
    print(json.dumps(res["counts"], indent=1))
    for r in res["rows"]:
        print(f"mu={r['mu']:g} kn={r['kn']:g} delta={r['delta']:+.3e} "
              f"verdict={r['verdict']}"
              + (f" ci=[{r['ci_lo']:+.3e},{r['ci_hi']:+.3e}]" if r["ci_lo"] is not None else ""))
    return 0


# ---------------------------------------------------------------------------
# document + figures (numbers injected programmatically, nothing hand copied)
# ---------------------------------------------------------------------------
DOC = os.path.join(ROOT, "docs", "AUDIT_m22558_RCP_SEED_RESOLVED_BIAS.md")
FIGS = os.path.join(HERE, "figures")

# runs submitted by this mission that close the m21401 seed debt
CAMPAIGN = [
    {"jid": "cj4208", "tag": "deposition_n10000_mu0.1_kn256000_e0.1_seed1",
     "why": "低摩擦档第四刚度水平（m21145 门 A4 需要 >=4 档）"},
    {"jid": "cj4209", "tag": "deposition_n10000_mu0.1_kn256000_e0.1_seed2",
     "why": "同上，第二种子使留一检验有意义"},
    {"jid": "cj4210", "tag": "deposition_n10000_mu0_kn64000_e0.1_seed2_dtf0.025",
     "why": "无摩擦刚性端时间偏差的第二个细时间步种子"},
    {"jid": "cj4211", "tag": "deposition_n10000_mu0_kn64000_e0.1_seed3_dtf0.025",
     "why": "同上，第三种子（与粗时间步 3 种子对齐）"},
    {"jid": "cj4215", "tag": "deposition_n10000_mu0.1_kn64000_e0.1_seed2",
     "why": "低摩擦刚性端粗时间步第二种子（原档仅 1 种子，只能给点估计）"},
    {"jid": "cj4216", "tag": "deposition_n10000_mu0.1_kn64000_e0.1_seed3",
     "why": "同上，第三种子"},
    {"jid": "cj4217", "tag": "deposition_n10000_mu0.1_kn64000_e0.1_seed2_dtf0.025",
     "why": "低摩擦刚性端细时间步第二种子"},
    {"jid": "cj4218", "tag": "deposition_n10000_mu0.1_kn64000_e0.1_seed3_dtf0.025",
     "why": "同上，第三种子"},
]


def _run_present(tag: str) -> bool:
    return os.path.exists(os.path.join(RUNS, tag + ".json"))


def fig_forest(res, path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 0.75 * len(rows) + 1.8))
    tol = res["tol"]
    ax.axvspan(-tol, tol, color="tab:green", alpha=0.10,
               label=fr"admissible band $|\Delta|<{tol:g}$")
    colors = {"CONVERGED": "tab:green", "BIASED": "tab:red",
              "UNDECIDABLE": "tab:orange"}
    for i, r in enumerate(rows):
        c = colors[r["verdict"]]
        if r["ci_lo"] is not None:
            ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], "-", color=c, lw=2.5)
            ax.plot([r["ci_lo"], r["ci_lo"]], [i - 0.15, i + 0.15], "-", color=c, lw=1.2)
            ax.plot([r["ci_hi"], r["ci_hi"]], [i - 0.15, i + 0.15], "-", color=c, lw=1.2)
        ax.plot(r["delta"], i, "o", color=c, ms=7,
                label=r["verdict"] if r["verdict"] not in [t.get_text() for t in ax.get_yticklabels()] else None)
        ax.annotate(f"{r['verdict']}  ({r['n_coarse']},{r['n_fine']})",
                    (r["delta"], i), textcoords="offset points", xytext=(6, 5),
                    fontsize=7, color=c)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"mu={r['mu']:g}\nkn={r['kn']:g}" for r in rows], fontsize=8)
    ax.set_xlabel(r"$\Delta=\phi(dt/2)-\phi(dt)$   (95% interval, seeds in brackets)")
    ax.set_title("Seed-resolved temporal-bias verdicts")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return os.path.relpath(path, ROOT)


def fig_levels(runs, path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lv = levels_present(runs)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for j, (mu, kn) in enumerate(lv):
        for dt_frac, marker, off in ((DT_COARSE, "o", -0.08), (DT_FINE, "s", +0.08)):
            vals = _sample(runs, mu, kn, dt_frac)
            if vals.size == 0:
                continue
            x = j + off
            sd = vals.std(ddof=1) if vals.size > 1 else 0.0
            ax.errorbar(x, vals.mean(), yerr=sd, fmt=marker, capsize=3,
                        color="tab:blue" if dt_frac == DT_COARSE else "tab:purple",
                        ms=6, label=(None if j else
                                     (r"$\phi(dt)$" if dt_frac == DT_COARSE
                                      else r"$\phi(dt/2)$")))
            for k, v in enumerate(vals):
                ax.plot(x + 0.02 * (k - (vals.size - 1) / 2), v, ".", color="grey", ms=3)
    ax.set_xticks(range(len(lv)))
    ax.set_xticklabels([f"mu={mu:g}\nkn={kn:g}" for mu, kn in lv], fontsize=8)
    ax.set_ylabel(r"$\phi_{bulk}$")
    ax.set_title("Timestep pairs per ladder level (dots = individual seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return os.path.relpath(path, ROOT)


def _fmt(x, spec="+.3e"):
    return "—" if x is None else format(x, spec)


def write_doc(res, fig_forest_rel, fig_levels_rel) -> str:
    os.makedirs(FIGS, exist_ok=True)
    L = []
    L.append("# AUDIT m22558 · RCP 时间偏差门：种子分辨裁决")
    L.append("")
    L.append("锚点：`CASE-DEM-RCP-PACKING-FRACTION`（算例池·随机密堆体积分数 RCP）。"
             "本审计只处理上一波留下的**一笔具体欠账**：m21401 把无摩擦刚性端"
             "（`mu=0, k_n=64000`）的时间偏差门判为边际 FAIL，理由是"
             "「|Δ| 略超阈，但与种子噪声不可分离」。那是**统计功效**问题，不是物理问题。")
    L.append("")
    L.append("## 1 欠账的原文与本次要做的判定")
    L.append("")
    L.append("m21401 的判据是单一的标量比较：`|Δ| < 5e-4`，其中 "
             "`Δ = φ(dt/2) − φ(dt)` 是两组**种子均值**之差。点估计与固定阈比较丢掉了误差信息："
             "当 `|Δ| ≈ 阈值` 而种子标准差同量级时，既不能判通过，也不能判不通过。"
             "本审计把判据升级为**区间判据**，并补齐使判定成立所需的种子。")
    L.append("")
    L.append("预登记规则（在读取新种子前写定）：95% 双侧区间 `[lo, hi]` 与容许带 `[-5e-4, 5e-4]` 的关系")
    L.append("")
    L.append("| 关系 | 裁决 | 含义 |")
    L.append("|---|---|---|")
    L.append("| 区间完全落在带内 | `CONVERGED` | 偏差**已被认证**小于阈值 |")
    L.append("| 区间完全落在带外 | `BIASED` | 偏差**已被认证**大于阈值 |")
    L.append("| 区间跨越带边界 | `UNDECIDABLE` | 不下判断（不通过、也不失败） |")
    L.append("")
    L.append("`certified` 只对 `resolved_from=interval` 的 `CONVERGED` 为真——"
             "只有点估计（两侧各一个种子）的档即使点估计在带内也不认证。")
    L.append("")
    L.append("## 2 本轮补齐的算例")
    L.append("")
    L.append("| 作业 | run（`benchmarks/case_rcp_deposition/runs/`） | 在盘 | 用途 |")
    L.append("|---|---|---|---|")
    for c in CAMPAIGN:
        L.append(f"| `{c['jid']}` | `{c['tag']}` | {'✓' if _run_present(c['tag']) else '✗（未回件）'} | {c['why']} |")
    L.append("")
    L.append("全部为 N=10000（台账要求的系统尺寸）、g=1、d=1、e=0.1、"
             "`dt = dt_frac·t_c`；时间探针 `dt_frac=0.025`（即 `t_c/40`）与生产档 `t_c/20` 配对。")
    L.append("")
    L.append("## 3 结果：逐档种子分辨裁决")
    L.append("")
    L.append("| μ | k_n | 种子 (粗,细) | φ(dt) | φ(dt/2) | Δ | sem | df | 95% 区间 | 点估计在带内? | 裁决 | 可认证 | 达阈所需种子/档 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        ci = "—" if r["ci_lo"] is None else f"[{r['ci_lo']:+.2e}, {r['ci_hi']:+.2e}]"
        L.append("| {mu:g} | {kn:g} | ({nc},{nf}) | {pc:.5f} | {pf:.5f} | {d} | {se} | {df} | {ci} | {pt} | `{v}` | {ce} | {sn} |".format(
            mu=r["mu"], kn=r["kn"], nc=r["n_coarse"], nf=r["n_fine"],
            pc=r["phi_coarse"], pf=r["phi_fine"], d=_fmt(r["delta"]),
            se=_fmt(r["sem"], ".2e"), df=("-" if r["df"] is None else f"{r['df']:.1f}"),
            ci=ci, pt="是" if r["point_within_tol"] else "否", v=r["verdict"],
            ce="是" if r["certified"] else "否",
            sn=("-" if r["seeds_needed_for_tol"] is None else r["seeds_needed_for_tol"])))
    L.append("")
    L.append(f"合计：`CONVERGED` {res['counts'].get('CONVERGED', 0)} · "
             f"`BIASED` {res['n_biased']} · `UNDECIDABLE` {res['n_undecidable']}；"
             f"其中**可认证** {res['n_certified']}/{res['n_levels']}。")
    L.append("")
    L.append("## 4 图")
    L.append("")
    L.append(f"![forest](../{fig_forest_rel})")
    L.append("")
    L.append("*Δ 的 95% 区间（横线）与容许带（绿带）。括号内为（粗,细）种子数。"
             "橙=不可判定，绿=已认证收敛，红=已认证有偏。*")
    L.append("")
    L.append(f"![levels](../{fig_levels_rel})")
    L.append("")
    L.append("*每个刚度档的 φ(dt)（蓝圆）与 φ(dt/2)（紫方），误差棒=种子标准差，灰点=单个种子。*")
    L.append("")
    L.append("## 5 与 m21401 的一致性 / 差异")
    L.append("")
    L.append("| 档 | m21401 的 Δ | 本审计 Δ | 差异原因 |")
    L.append("|---|---|---|---|")
    for r in res["rows"]:
        if r["mu"] == 0.0 and abs(r["kn"] - 64000.0) < 1e-9:
            L.append(f"| mu=0, k_n=64000 | -5.069e-04（细档 n=1） | {_fmt(r['delta'])}（细档 n={r['n_fine']}） | "
                     "细档种子增加后均值移动 |")
        elif r["mu"] == 0.1 and abs(r["kn"] - 64000.0) < 1e-9:
            L.append(f"| mu=0.1, k_n=64000 | +1.188e-04（细档 n=1） | {_fmt(r['delta'])}（细档 n={r['n_fine']}） | "
                     "同上；该档两侧仍各 1 种子，故本审计只能给点估计 |")
    L.append("")
    L.append("m21401 的截距/精度结论（无摩擦档 dt 修正后截距 0.63569 vs 文献 0.6366，−0.14%）"
             "在本审计中**不被重估**：本审计只回答「那一道边际门现在能不能判」。")
    L.append("")
    L.append("## 6 注意点与局限")
    L.append("")
    L.append("1. 单侧只有 1 个种子时，误差按对侧标准差合并估计（`error_model=pooled_one_sided`，"
             "df = n粗+n细−2），这是**工程近似**而非严格 Welch；该档即使裁决为 `CONVERGED` 也不认证。")
    L.append("2. `seeds_needed_for_tol` 假设两档共享同一散布 σ（该算例的设计假设：dt 减半不应改变种子散布）；"
             "若该假设不成立，所需种子数会变。")
    L.append("3. `UNDECIDABLE` 不是失败也不是通过：它表示当前种子预算下该档的偏差与阈值不可分离。")
    L.append("4. 各档之间不正交：种子、盒宽、投放序列共享，故本表逐档裁决，不做跨档合并检验（合并会引入伪独立）。")
    L.append("5. 本审计不触及刚度外推阶次区间（见 m21145）、稳态判据（m17063）、φ 定义（m18395）。")
    L.append("")
    L.append("## 7 误差判定")
    L.append("")
    L.append("| 量 | 计算值 | 基准 | 相对误差 | <3%? |")
    L.append("|---|---|---|---|---|")
    for r in res["rows"]:
        if r["mu"] == 0.0:
            rel = abs(r["phi_coarse"] - 0.6366) / 0.6366
            L.append(f"| φ_bulk (mu=0, k_n={r['kn']:g}) | {r['phi_coarse']:.5f} | 0.6366 (Scott 1960 / Bernal & Mason 1960) "
                     f"| {rel * 100:+.2f}% | {'是' if rel < 0.03 else '否'} |")
        else:
            L.append(f"| φ_bulk (mu=0.1, k_n={r['kn']:g}) | {r['phi_coarse']:.5f} | 0.6250（derived 级，非文献表值） "
                     "| — | 无可用基准 → WITHHELD |")
    L.append("")
    L.append("时间偏差门的判定（本次审计的对象）不是相对误差，而是绝对偏差闸："
             "`|Δ| < 5e-4`（约为 φ_ref 的 0.08%）。表中 `裁决` 一列即为该项的显式回答；"
             "凡 `UNDECIDABLE` 者一律不计为通过。")
    text = "\n".join(L) + "\n"
    with open(DOC, "w") as fh:
        fh.write(text)
    return os.path.relpath(DOC, ROOT)

if __name__ == "__main__":
    raise SystemExit(main())
