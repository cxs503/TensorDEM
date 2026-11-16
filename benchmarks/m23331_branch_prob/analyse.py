#!/usr/bin/env python3
"""m23331 estimator: branch probability, its CI, and the conditional flow rate.

Reads the per-record JSON written by ``run_case.py`` and applies the verdict rule
of ``prereg.md`` §3 **exactly as registered** — no threshold is chosen after the
data are seen.  Everything here is pure post-processing of stored records, so it
is pinned by ``tests/test_m23331_branch_prob.py`` against synthetic inputs.

    python3 benchmarks/m23331_branch_prob/analyse.py --dir runs/m23331_branch_prob \
        --figures docs/figures/m23331
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
Z95 = 1.959963984540054
#: two-sided t(0.975, df) for the small samples of this mission (df = n-1)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}
LEDGER_TOL = 0.10


# --------------------------------------------------------------------------- #
# statistics (implemented here so the estimator has no scipy dependency)
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _hypergeom_pmf(a: int, row1: int, col1: int, n: int) -> float:
    return (math.comb(col1, a) * math.comb(n - col1, row1 - a)
            / math.comb(n, row1))


def fisher_exact(a: int, b: int, c: int, d: int) -> dict:
    """Two-sided Fisher exact test on a 2x2 table [[a, b], [c, d]].

    ``p`` is the sum of the probabilities of all tables with the same margins
    whose probability is <= that of the observed table (the standard
    "point probability" definition), with a 1e-9 relative slack so that
    numerically equal tables are not dropped.
    """
    for v in (a, b, c, d):
        if v < 0:
            raise ValueError("counts must be non-negative")
    n = a + b + c + d
    if n == 0:
        return {"p_value": 1.0, "odds_ratio": float("nan"), "n": 0}
    row1, col1 = a + b, a + c
    lo = max(0, row1 + col1 - n)
    hi = min(row1, col1)
    obs = _hypergeom_pmf(a, row1, col1, n)
    p = sum(_hypergeom_pmf(x, row1, col1, n) for x in range(lo, hi + 1)
            if _hypergeom_pmf(x, row1, col1, n) <= obs * (1.0 + 1e-9))
    if b * c == 0:
        odds = float("inf") if a * d > 0 else float("nan")
    else:
        odds = (a * d) / (b * c)
    return {"p_value": min(1.0, float(p)), "odds_ratio": odds, "n": n}


def branch_probability(flows: list[bool]) -> dict:
    """Fraction FLOWING with its Wilson CI."""
    n = len(flows)
    k = int(sum(1 for f in flows if f))
    lo, hi = wilson(k, n)
    return {"n": n, "n_flowing": k, "p_flowing": (k / n if n else float("nan")),
            "ci95": [lo, hi], "pure": bool(n > 0 and k in (0, n))}


def classify_branch(p_f: dict, p_q: dict, fisher: dict) -> dict:
    """Apply ``prereg.md`` §3 verbatim.

    B1 SPELLING-DOMINATED : one arm pure at p = 1, the other pure at p = 0.
    B2 SEED-DOMINATED     : both mixed, or both pure and equal (Fisher p >= 0.05).
    B3 INCONCLUSIVE       : everything else.
    """
    f_pure, q_pure = p_f["pure"], p_q["pure"]
    # Amendment A1 (see prereg.md §6): a pure-vs-pure split is only called
    # SPELLING-DOMINATED when the exact test on its own 2x2 table supports it.
    # Without this guard a 4/4-vs-0/1 table (Fisher p = 0.20) would be reported
    # as a significant factor effect, which its own p-value contradicts.
    opposite = (f_pure and q_pure and p_f["n_flowing"] != p_q["n_flowing"]
                and fisher["p_value"] < 0.05)
    same = (f_pure and q_pure and p_f["n_flowing"] == p_q["n_flowing"])
    mixed = (not f_pure) and (not q_pure)
    if (f_pure and q_pure and p_f["n_flowing"] != p_q["n_flowing"]
            and fisher["p_value"] >= 0.05):
        return {"verdict": "B3_INCONCLUSIVE", "why":
                "the arms are pure and opposite, but the exact test on this table "
                f"gives p = {fisher['p_value']:.3g} >= 0.05: with few records per arm "
                "the split is indistinguishable from seed luck"}
    if opposite:
        return {"verdict": "B1_SPELLING_DOMINATED", "why":
                "one arm is pure FLOWING and the other pure ARRESTED: the branch "
                "is selected by the dt value, not by the packing realization"}
    if same:
        return {"verdict": "B2_SEED_DOMINATED", "why":
                "both arms are pure and identical, Fisher p = "
                f"{fisher['p_value']:.3g} >= 0.05: dt is irrelevant at this ULP gap"}
    if mixed and fisher["p_value"] >= 0.05:
        return {"verdict": "B2_SEED_DOMINATED", "why":
                "both arms mixed and indistinguishable (Fisher p = "
                f"{fisher['p_value']:.3g} >= 0.05): the branch is a property of the "
                "packing realization"}
    return {"verdict": "B3_INCONCLUSIVE", "why":
            "neither arm is pure and the arms differ (Fisher p = "
            f"{fisher['p_value']:.3g}): more seeds are needed to separate the factors"}


def conditional_stats(rates: list[float], reference: float) -> dict:
    """Mean / sd / 95 % CI of the FLOWING records and the ledger test."""
    r = np.asarray([x for x in rates if x > 0.0], dtype=np.float64)
    n = int(r.size)
    out = {"n": n, "reference_kg_s": float(reference)}
    if n == 0:
        out.update({"mean_kg_s": None, "verdict": "NOT_DECIDABLE",
                    "why": "no FLOWING record in this arm"})
        return out
    mean = float(np.mean(r))
    out["mean_kg_s"] = mean
    out["rel_error_vs_reference"] = mean / reference - 1.0 if reference > 0 else None
    if n < 2:
        out.update({"sd_kg_s": None, "ci95_halfwidth_kg_s": None,
                    "ci95_halfwidth_rel": None, "verdict": "NOT_DECIDABLE",
                    "why": "a single FLOWING record cannot be both a point estimate "
                           "and its own uncertainty"})
        return out
    sd = float(np.std(r, ddof=1))
    half = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    err = abs(mean / reference - 1.0) if reference > 0 else float("nan")
    acc_ok = bool(err < LEDGER_TOL)
    prec_ok = bool(half / mean < LEDGER_TOL)
    # "accuracy" and "precision" are separate sentences: an unbiased mean whose
    # CI is wider than the tolerance is INCONCLUSIVE, not FAIL
    verdict = ("PASS" if acc_ok and prec_ok else
               "INCONCLUSIVE_CI_TOO_WIDE" if acc_ok else "FAIL")
    out.update({
        "sd_kg_s": sd,
        "cv": sd / mean,
        "ci95_halfwidth_kg_s": half,
        "ci95_halfwidth_rel": half / mean,
        "ci95_kg_s": [mean - half, mean + half],
        "accuracy_ok": acc_ok,
        "precision_ok": prec_ok,
        "verdict": verdict,
    })
    return out


def required_n(cv: float, tol: float = LEDGER_TOL, nmax: int = 200) -> int | None:
    """Smallest ``n`` whose 95 % t-interval half-width is below ``tol``.

    ``half/mean = t(n-1)·CV/√n`` — the number a campaign must budget for if it
    wants to *grade* a case at ``tol`` rather than merely sample it.
    """
    if not (cv > 0.0):
        return 1
    for n in range(2, nmax + 1):
        t = T975.get(n - 1, 1.96)
        if t * cv / math.sqrt(n) < tol:
            return n
    return None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def load_records(dirs) -> list[dict]:
    """Load records from one or more directories, one record per ``(arm, seed)``.

    A duplicate ``(arm, seed)`` (same recipe recorded twice, e.g. an archive and
    a fresh run) is kept once — the fresh run wins — and the disagreement is
    returned so it can be reported as a reproducibility check instead of being
    averaged in twice.
    """
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    by_key, dups = {}, []
    files = []
    for d in dirs:
        files.extend(sorted(Path(d).glob("*.json")))
    for f in files:
        rec = json.loads(f.read_text())
        bp = rec.get("branch_prob")
        if not bp:
            continue
        ds = rec.get("discharge_status", {})
        bp = dict(bp)
        # status is taken from the record's own classifier output when present,
        # so a record can never be summarised under a different label than the
        # one `beverloo.discharge_status` produced for it
        if ds.get("status"):
            bp["status"] = ds["status"]
        bp["arrested"] = bool(ds.get("arrested", bp.get("arrested", ds.get("status") == "ARRESTED")))
        bp["path"] = str(f)
        bp["flow_rate_kg_s"] = rec["flow_rate_kg_s"]
        bp["window_particles"] = rec.get("window_particles", [])
        bp["flow_rate_windows_kg_s"] = rec.get("flow_rate_windows_kg_s", [])
        bp["n_exited"] = rec.get("particles_exited")
        bp["status_reasons"] = ds.get("reasons", [])
        bp["rate_over_reference"] = ds.get("rate_over_reference")
        bp["empty_window_fraction"] = ds.get("empty_window_fraction")
        key = (bp["arm"], int(bp["seed"]))
        bp["source"] = bp.get("source", "prospective")
        if key in by_key:
            prev = by_key[key]
            rel = abs(bp["flow_rate_kg_s"] / prev["flow_rate_kg_s"] - 1.0)
            fresh = bp["source"] == "prospective"
            dups.append({"arm": key[0], "seed": key[1],
                         "kept": prev.get("path"), "dropped": bp["path"],
                         "rel_diff_W": rel, "agree_1e-12": bool(rel < 1e-12)})
            if fresh:
                bp["duplicates"] = dups
                by_key[key] = bp
            continue
        by_key[key] = bp
    recs = [by_key[k] for k in sorted(by_key)]
    for r in recs:
        r["duplicates"] = dups
    return recs


def analyse(recs: list[dict]) -> dict:
    arms = {}
    for arm in sorted({r["arm"] for r in recs}):
        sel = [r for r in recs if r["arm"] == arm]
        ref = float(np.mean([r["w_reference_kg_s"] for r in sel]))
        flows = [not r["arrested"] for r in sel]
        arms[arm] = {
            "seed": [r["seed"] for r in sel],
            "dt_s": sel[0]["dt_s"],
            "dt_bits": sel[0]["dt_bits"],
            "w_reference_kg_s": ref,
            "records": [{"seed": r["seed"], "status": r["status"],
                         "flow_rate_kg_s": r["flow_rate_kg_s"],
                         "rel_error_vs_beverloo": r["rel_error_vs_beverloo"],
                         "rate_over_reference": r["rate_over_reference"],
                         "window_particles": r["window_particles"],
                         "n_exited": r["n_exited"],
                         "bed_state_sha256": r["bed_state_sha256"],
                         "reasons": r["status_reasons"]} for r in sel],
            "branch": branch_probability(flows),
            "conditional": conditional_stats([r["flow_rate_kg_s"] for r in sel if not r["arrested"]], ref),
        }
        c = arms[arm]["conditional"]
        arms[arm]["required_n"] = (required_n(c["cv"]) if c.get("cv") else None)
    out = {"arms": arms, "n_records": len(recs)}
    if "F" in arms and "Q" in arms:
        a, b = arms["F"]["branch"], arms["Q"]["branch"]
        n1, n2 = a["n"], b["n"]
        tab = [[a["n_flowing"], n1 - a["n_flowing"]],
               [b["n_flowing"], n2 - b["n_flowing"]]]
        fe = fisher_exact(tab[0][0], tab[0][1], tab[1][0], tab[1][1])
        out["table"] = tab
        out["fisher"] = fe
        out["branch_verdict"] = classify_branch(a, b, fe)
        # prereg.md §3 (as corrected before any record existed): the anchor grade
        # is quoted on the canonical spelling Q (= Material.dt today) when it has
        # n >= 2 FLOWING records, otherwise on F.  Both arms always get a row.
        pick = "Q" if arms["Q"]["conditional"].get("n", 0) >= 2 else "F"
        out["anchor"] = {"arm": pick, "conditional": arms[pick]["conditional"],
                         "ledger_tolerance": LEDGER_TOL,
                         "why": "canonical spelling T_c/20 with n >= 2 FLOWING records"
                         if pick == "Q" else
                         "canonical arm Q has < 2 FLOWING records: graded on the legacy "
                         "spelling F, and this is reported as such"}
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(res: dict, recs: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # ---- fig 1: every record against the Beverloo reference ±10 % ----------
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ref = float(np.mean([a["w_reference_kg_s"] for a in res["arms"].values()]))
    ax.axhspan(0.9 * ref, 1.1 * ref, color="0.85", zorder=0,
               label="Beverloo 1961 ±10 % (ledger tolerance)")
    ax.axhline(ref, color="k", lw=1.0, ls="--", zorder=1, label="Beverloo 1961 (C=0.58, k=1.4)")
    for i, arm in enumerate(sorted(res["arms"])):
        sel = [r for r in recs if r["arm"] == arm]
        for j, r in enumerate(sel):
            x = i + (j - (len(sel) - 1) / 2) * 0.10
            if r["arrested"]:
                ax.plot(x, r["flow_rate_kg_s"], "v", color="crimson", ms=8, zorder=3,
                        label="ARRESTED (arch)" if (i == 0 and j == 0) else None)
            else:
                ax.plot(x, r["flow_rate_kg_s"], "o", color="tab:blue", ms=7, zorder=3,
                        label="FLOWING" if (i == 0 and j == 0) else None)
            ax.annotate(f"s{r['seed']}", (x, r["flow_rate_kg_s"]),
                        textcoords="offset points", xytext=(3, 4), fontsize=7)
        c = res["arms"][arm]["conditional"]
        if c.get("mean_kg_s"):
            ax.plot([i - 0.32, i + 0.32], [c["mean_kg_s"]] * 2, color="tab:green", lw=2.2, zorder=4,
                    label="mean of FLOWING" if i == 0 else None)
    ax.set_xticks(range(len(res["arms"])))
    ax.set_xticklabels([f"{a} (dt={res['arms'][a]['dt_s']:.17e})".replace("e-0", "e-")
                        for a in sorted(res["arms"])])
    ax.set_ylabel("$W$ [kg/s]")
    ax.set_title("m23331 — 12 frozen-case records: no single record is a measurement")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    p = out_dir / "m23331_records_vs_reference.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # ---- fig 2: branch probability with Wilson CI --------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for i, arm in enumerate(sorted(res["arms"])):
        b = res["arms"][arm]["branch"]
        ax.bar([i], [b["p_flowing"]], width=0.5, color="tab:blue", alpha=0.8)
        lo, hi = b["ci95"]
        ax.errorbar([i], [b["p_flowing"]], yerr=[[b["p_flowing"] - lo], [hi - b["p_flowing"]]],
                    fmt="none", ecolor="k", capsize=6, lw=1.4)
        ax.text(i, min(1.0, hi + 0.04), f"{b['n_flowing']}/{b['n']}", ha="center", fontsize=9)
    ax.set_xticks(range(len(res["arms"])))
    ax.set_xticklabels(sorted(res["arms"]))
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("$p_{\\mathrm{flow}}$ (95 % Wilson CI)")
    bv = res.get("branch_verdict", {}).get("verdict", "n/a")
    ax.set_title(f"branch probability per dt spelling — {bv}")
    ax.grid(axis="y", alpha=0.3)
    p = out_dir / "m23331_branch_probability.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # ---- fig 3: window particle series (the arch signature) ----------------
    fig, axes = plt.subplots(1, len(res["arms"]), figsize=(10.5, 3.6), sharey=True)
    if len(res["arms"]) == 1:
        axes = [axes]
    for ax, arm in zip(axes, sorted(res["arms"])):
        for r in [r for r in recs if r["arm"] == arm]:
            w = r["window_particles"]
            if not w:
                continue
            ax.plot(range(1, len(w) + 1), w, marker="o", ms=3.5, lw=1.2,
                    ls="-" if not r["arrested"] else "--",
                    color="tab:blue" if not r["arrested"] else "crimson",
                    label=f"s{r['seed']} {'ARR' if r['arrested'] else 'FLOW'}")
        ax.set_title(f"arm {arm} — particles leaving per 0.05 s window")
        ax.set_xlabel("measurement window")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("particles exited")
    p = out_dir / "m23331_window_series.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def render_doc(res: dict, fig_paths: list[Path]) -> str:
    L = []
    A = L.append
    bv = res.get("branch_verdict", {})
    A("# CASE-DEM-SILO-BEVERLOO / m23331 — branch probability of the frozen case\n")
    A("**Pre-registration**: `benchmarks/m23331_branch_prob/prereg.md` (written before "
      "any record existed). **Kernel**: `tensordem.silo_flow`. **Reference**: Beverloo, "
      "Leniger & van de Velde (1961), *Chem. Eng. Sci.* **15**, 260–269, Eq. (4), "
      "`C = 0.58`, `k = 1.4`, never refitted.\n")
    A(f"**Records**: {res['n_records']} distinct `(arm, seed)` records "
      f"(sources `{res.get('sources')}`); the pre-registered prospective grid is "
      f"2 spellings × seeds 1–6 = 12 records.\n")
    A("## Verdict\n")
    if bv:
        A(f"* **Branch verdict**: `{bv['verdict']}` — {bv['why']}.")
        c = res.get("anchor", {}).get("conditional", {})
        if c.get("mean_kg_s"):
            A(f"* **Conditional flow rate** (arm `{res['anchor']['arm']}`, "
              f"`dt = {res['arms'][res['anchor']['arm']]['dt_s']:.17e}` s — "
              f"{'the canonical spelling `T_c/20` = `Material.dt` today' if res['anchor']['arm'] == 'Q' else 'the legacy spelling `0.05·T_c`; canonical `T_c/20` has < 2 FLOWING records'}): "
              f"`W = {c['mean_kg_s']:.6g} kg/s` over "
              f"`n = {c['n']}` FLOWING records, CI95 half-width "
              f"`{c.get('ci95_halfwidth_rel', float('nan')):.2%}`, "
              f"error vs 1961 `{c['rel_error_vs_reference']:+.2%}` → **{c['verdict']}** "
              f"(ledger tolerance ±{res['anchor']['ledger_tolerance']:.0%}, with "
              f"`|err| + CI 半宽 < 10 %` required).")
        else:
            A("* **No conditional flow rate** can be quoted.")
    else:
        A("* Arms missing — see per-arm tables.")
    A("\n## Per-arm tables\n")
    for arm, a in res["arms"].items():
        b = a["branch"]
        A(f"### arm `{arm}` — `dt = {a['dt_s']:.17e}` s (`{a['dt_bits']}`)\n")
        A(f"`p_flow = {b['n_flowing']}/{b['n']} = {b['p_flowing']:.3f}` "
          f"(95 % Wilson [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}])\n")
        A("| seed | status | W [kg/s] | W/W_1961 − 1 | W/W_ref | windows (particles) |")
        A("|---|---|---|---|---|---|")
        for r in a["records"]:
            rr = r["rate_over_reference"]
            rr_txt = "n/a" if rr is None else f"{rr:.3f}"
            A(f"| {r['seed']} | {r['status']} | {r['flow_rate_kg_s']:.6g} | "
              f"{r['rel_error_vs_beverloo']:+.2%} | {rr_txt} | "
              f"{r['window_particles']} |")
        c = a["conditional"]
        errs = [abs(r["rel_error_vs_beverloo"]) for r in a["records"] if not r["reasons"]]
        A(f"* conditional: `n = {c.get('n')}`, mean "
          f"`{'n/a' if not c.get('mean_kg_s') else f'{c[chr(109)+chr(101)+chr(97)+chr(110)+chr(95)+chr(107)+chr(103)+chr(95)+chr(115)]:.6g}'}` kg/s, "
          f"CV `{'n/a' if not c.get('cv') else f'{c[chr(99)+chr(118)]:.2%}'}`, "
          f"CI95 half-width `{'n/a' if c.get('ci95_halfwidth_rel') is None else f'{c[chr(99)+chr(105)+chr(57)+chr(53)+chr(95)+chr(104)+chr(97)+chr(108)+chr(102)+chr(119)+chr(105)+chr(100)+chr(116)+chr(104)+chr(95)+chr(114)+chr(101)+chr(108)]:.2%}'}`, "
          f"verdict **{c['verdict']}**, `required_n = {a.get('required_n')}`.")
        if errs:
            A(f"* largest single-record |error| in this arm: `{max(errs):.2%}` "
              f"(tolerance {LEDGER_TOL:.0%}) — a one-record verdict is a lottery "
              f"of width `{max(errs) - min(errs):.2%}`.")
        A("")
    A("## Single-record verdict lottery\n")
    A("The ledger tolerance applied to *one* record, which is how every earlier "
      "round of this anchor was graded:\n")
    A("| arm | seed | W [kg/s] | rel. vs 1961 | single-record verdict |")
    A("|---|---|---|---|---|")
    for arm, a in res["arms"].items():
        for r in a["records"]:
            v = "PASS" if abs(r["rel_error_vs_beverloo"]) < LEDGER_TOL else "FAIL"
            A(f"| {arm} | {r['seed']} | {r['flow_rate_kg_s']:.6g} | "
              f"{r['rel_error_vs_beverloo']:+.2%} | {v} |")
    A("")
    if res.get("duplicates") or res.get("sources"):
        A("## Provenance and reproducibility\n")
        A(f"* record sources: `{res.get('sources')}` "
          f"(`archive` = harvested read-only files, `prospective` = fresh runs).")
        for d in res.get("duplicates", []):
            A(f"* duplicate `{d['arm']}_s{d['seed']}`: `{d['kept']}` vs `{d['dropped']}`, "
              f"relative `W` difference `{d['rel_diff_W']:.2e}` "
              f"({'agree <1e-12' if d['agree_1e-12'] else 'DISAGREE'}).")
        A("")
    if res.get("note"):
        A("## Run status\n")
        A(res["note"])
        A("")
    if res.get("table"):
        t, fe = res["table"], res["fisher"]
        A("## 2 × 2 table and exact test\n")
        A("| | FLOWING | ARRESTED |")
        A("|---|---|---|")
        A(f"| F (`0.05·T_c`) | {t[0][0]} | {t[0][1]} |")
        A(f"| Q (`T_c/20`) | {t[1][0]} | {t[1][1]} |")
        A(f"\nTwo-sided Fisher exact: `p = {fe['p_value']:.4g}`, odds ratio "
          f"`{fe['odds_ratio']:.3g}`.\n")
    A("## Figures\n")
    for p in fig_paths:
        A(f"* `{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}`")
    A("\n*Generated by `benchmarks/m23331_branch_prob/analyse.py` — numbers are read "
      "from the run records, never hand-typed.*\n")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["runs/m23331_branch_prob"])
    ap.add_argument("--figures", default="docs/figures/m23331")
    ap.add_argument("--out", default="runs/m23331_branch_prob/summary.json")
    ap.add_argument("--doc", default="docs/CASE-DEM-SILO-BEVERLOO_m23331_branch_prob.md")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--note", default="", help="run-status text quoted verbatim in the doc")
    args = ap.parse_args(argv)

    recs = load_records([Path(d) for d in args.dir])
    if not recs:
        print(f"no records in {args.dir}")
        return 1
    res = analyse(recs)
    figs = [] if args.no_figures else figures(res, recs, ROOT / args.figures)
    res["figures"] = [str(p) for p in figs]
    res["duplicates"] = recs[0].get("duplicates", []) if recs else []
    res["note"] = args.note
    res["sources"] = sorted({r.get("source", "prospective") for r in recs})
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    doc = render_doc(res, figs)
    (ROOT / args.doc).write_text(doc)
    print(json.dumps(res.get("branch_verdict", {}), indent=1))
    for arm, a in res["arms"].items():
        print(f"{arm}: p_flow={a['branch']['n_flowing']}/{a['branch']['n']} "
              f"cond={a['conditional'].get('mean_kg_s')} "
              f"err={a['conditional'].get('rel_error_vs_reference')} "
              f"{a['conditional'].get('verdict')}")
    if res.get("fisher"):
        print(f"fisher p={res['fisher']['p_value']:.4g}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())