"""Round 13 analysis: full-range (I = 1e-3 ... 0.3) rough-vs-smooth mu(I)/phi(I).

Reads every JSONL the case has produced -- ``runs/rough_wall/`` (round 11),
``runs/round12/`` (round 12 partial) and ``runs/round13/`` (round 13, the
ledger's full acceptance range) -- dedups by (mode, I, seed) with the newest
round winning, aggregates per (mode, I) over the seeds, and compares every
mu estimator and the bed solid fraction against the reference law of
``reference_law.py``.

Outputs (under ``runs/round13/``): ``analysis_r13.json``, ``tables_r13.md``
and the figures ``fig_mu_vs_I.png``, ``fig_phi_vs_I.png``,
``fig_seed_spread.png``.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "round13"
OUT.mkdir(parents=True, exist_ok=True)

import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from reference_law import ref_mu, ref_phi, reference  # noqa: E402

ESTIMATORS = ("mu_wall", "mu_full", "mu_full_zz", "mu_bulk", "mu_bulk_mid")
LABEL = {"mu_wall": "platen traction", "mu_full": "full-cell virial (p)",
         "mu_full_zz": "full-cell virial (sigma_zz)",
         "mu_bulk": "slab virial (both)", "mu_bulk_mid": "slab virial (midpoint)"}
BAND = 0.10  # acceptance band of the ledger row

#: round precedence for the (mode, I, seed) dedup: later round wins.
ROUND_DIRS = (HERE / "runs" / "rough_wall",
             HERE / "runs" / "round12",
             HERE / "runs" / "round13")


def load_records():
    """All records, newest round winning per (mode, I, seed); FAILs kept."""
    latest = {}
    for d in ROUND_DIRS:
        if not d.exists():
            continue
        for path in sorted(d.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec.get("wall_mode"), rec.get("I_target"), rec.get("seed"))
                latest[key] = rec
    recs = list(latest.values())
    fails = [r for r in recs if r.get("status") == "FAIL"]
    ok = [r for r in recs if r.get("status") != "FAIL"]
    return ok, fails


def _num(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def aggregate(recs):
    """{(mode, I): {estimator: (mean, sd, n), ...}} over the seeds."""
    groups = defaultdict(list)
    for rec in recs:
        groups[(rec["wall_mode"], float(rec["I_target"]))].append(rec)
    agg = {}
    for key, items in groups.items():
        entry = {"n_seeds": len(items), "seeds": sorted(r["seed"] for r in items)}
        for est in ESTIMATORS:
            vals = [v for v in (_num(r.get(est)) for r in items) if v is not None]
            entry[est] = ({"mean": float(np.mean(vals)),
                           "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                           "n": len(vals)} if vals else None)
        for extra in ("phi_bed_steady", "gamma_dot_ratio", "profile_r2",
                     "P_rel_err_final", "I_measured_rate", "I_imposed_rate",
                     "mu_wall_final", "dilatancy", "n_particles"):
            vals = [v for v in (_num(r.get(extra)) for r in items) if v is not None]
            entry[extra] = ({"mean": float(np.mean(vals)),
                             "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                             "n": len(vals)} if vals else None)
        entry["servo_valid"] = sum(bool(r.get("servo_valid")) for r in items)
        agg[key] = entry
    return agg


def tables(agg, fails, ok):
    lines = ["# Round 13 tables - full-range (1e-3..0.3) rough vs smooth wall",
             "",
             "3 seeds per point (5279, 6113, 7919), single device "
             "`ServoShearCase`, only `wall_mode` changes.  Reference: "
             "da Cruz 2005 / GDR MiDi 2004 as quoted in-repo "
             "(mu_s, mu_2, I0) = (0.36, 0.84, 0.30); phi magnitudes "
             "PROVENANCE-UNVERIFIED (see `reference_law.py`).",
             ""]
    modes = sorted({k[0] for k in agg})
    Is = sorted({k[1] for k in agg})

    lines += ["## Table 1 - mu(I), platen traction (historical estimator)", "",
              "| I | mu_ref | " + " | ".join(f"{m} mean +- sd (err)" for m in modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    for I in Is:
        row = [f"{I:g}", f"{ref_mu(I):.4f}"]
        for m in modes:
            e = agg.get((m, I), {}).get("mu_wall")
            if not e:
                row.append("n/a")
            else:
                err = 100.0 * (e["mean"] / ref_mu(I) - 1.0)
                row.append(f"{e['mean']:.4f} +- {e['sd']:.4f} ({err:+.1f}%)")
        lines.append("| " + " | ".join(row) + " |")

    for est in ESTIMATORS[1:]:
        lines += ["", f"## Table 1b - mu(I), {LABEL[est]}", "",
                  "| I | mu_ref | " + " | ".join(modes) + " |",
                  "|---|---|" + "---|" * len(modes)]
        for I in Is:
            row = [f"{I:g}", f"{ref_mu(I):.4f}"]
            for m in modes:
                e = agg.get((m, I), {}).get(est)
                row.append("n/a" if not e else
                         f"{e['mean']:.4f} +- {e['sd']:.4f} "
                         f"({100.0 * (e['mean'] / ref_mu(I) - 1.0):+.1f}%)")
            lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Table 2 - servo and kinematics", "",
              "| I | mode | seeds | servo valid | P err (final) | "
              "rate meas/imposed | profile R^2 | I meas |",
              "|---|---|---|---|---|---|---|---|"]
    for I in Is:
        for m in modes:
            e = agg.get((m, I))
            if not e:
                continue
            pr = e["P_rel_err_final"]["mean"] if e.get("P_rel_err_final") else None
            rr = e["gamma_dot_ratio"]["mean"] if e.get("gamma_dot_ratio") else None
            r2 = e["profile_r2"]["mean"] if e.get("profile_r2") else None
            im = e["I_measured_rate"]["mean"] if e.get("I_measured_rate") else None
            lines.append(
                f"| {I:g} | {m} | {e['n_seeds']} | {e['servo_valid']}/{e['n_seeds']} | "
                f"{(f'{100 * pr:+.1f}%' if pr is not None else 'n/a')} | "
                f"{(f'{rr:.3f}' if rr is not None else 'n/a')} | "
                f"{(f'{r2:.3f}' if r2 is not None else 'n/a')} | "
                f"{(f'{im:.4f}' if im is not None else 'n/a')} |")

    lines += ["", "## Table 3 - phi(I), full-gap bed definition", "",
              "| I | phi_ref (PROVENANCE-UNVERIFIED) | " +
              " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    for I in Is:
        row = [f"{I:g}", f"{ref_phi(I):.4f}"]
        for m in modes:
            e = agg.get((m, I), {}).get("phi_bed_steady")
            row.append("n/a" if not e else
                     f"{e['mean']:.4f} +- {e['sd']:.4f} "
                     f"({100.0 * (e['mean'] / ref_phi(I) - 1.0):+.1f}%)")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Table 4 - outcome vs the ledger acceptance (<10 %)", "",
              "| estimator | mode | points inside 10 % | worst |err| | mean err |",
              "|---|---|---|---|---|"]
    for est in ESTIMATORS:
        for m in modes:
            errs = []
            for I in Is:
                e = agg.get((m, I), {}).get(est)
                if e:
                    errs.append(100.0 * (e["mean"] / ref_mu(I) - 1.0))
            if not errs:
                continue
            inside = sum(abs(x) <= 100 * BAND for x in errs)
            lines.append(f"| {LABEL[est]} | {m} | {inside}/{len(errs)} | "
                         f"{max(abs(x) for x in errs):.1f}% | "
                         f"{float(np.mean(errs)):+.1f}% |")

    by_mi = defaultdict(list)
    for r in ok:
        by_mi[(r["wall_mode"], float(r["I_target"]))].append(r)
    lines += ["", "## Table 5 - seed-to-seed spread (sample identification)", "",
              "| I | mode | mu_wall seeds | spread (max-min)/ref |",
              "|---|---|---|---|"]
    for I in Is:
        for m in modes:
            vals = sorted((_num(r.get("mu_wall")) for r in by_mi.get((m, I), [])))
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                continue
            spread = 100.0 * (max(vals) - min(vals)) / ref_mu(I)
            lines.append(f"| {I:g} | {m} | " +
                         ", ".join(f"{v:.4f}" for v in vals) +
                         f" | {spread:.1f}% |")
    if fails:
        lines += ["", "## Failures (honest FAIL records)", ""]
        for rec in fails:
            lines.append(f"* {rec['wall_mode']} I={rec['I_target']:g} "
                         f"seed={rec['seed']}: {rec.get('error')}")
    return "\n".join(lines) + "\n"


def figures(agg, ok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted({k[0] for k in agg})
    Is = sorted({k[1] for k in agg})
    Igrid = np.logspace(-3, math.log10(0.3), 200)

    # --- mu(I) -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Igrid, ref_mu(Igrid), "k-", lw=2, label="reference mu(I)")
    ax.fill_between(Igrid, ref_mu(Igrid) * (1 - BAND), ref_mu(Igrid) * (1 + BAND),
                   color="0.85", label="+/-10 % band")
    styles = {"smooth": ("tab:blue", "o"), "rough": ("tab:red", "s")}
    for m in modes:
        xs, ys, yerr = [], [], []
        for I in Is:
            e = agg.get((m, I), {}).get("mu_wall")
            if e:
                xs.append(I)
                ys.append(e["mean"])
                yerr.append(e["sd"])
        if xs:
            c, mk = styles[m]
            ax.errorbar(xs, ys, yerr=yerr, marker=mk, color=c, capsize=3,
                        label=f"{m} wall (platen traction)")
    ax.set_xscale("log")
    ax.set_xlabel("inertial number I")
    ax.set_ylabel("mu = tau / |sigma_n|")
    ax.set_title("plane shear mu(I): rough vs smooth wall, 3 seeds")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mu_vs_I.png", dpi=150)
    plt.close(fig)

    # --- phi(I) -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Igrid, ref_phi(Igrid), "k-", lw=2,
            label="reference phi(I) (PROVENANCE-UNVERIFIED)")
    for m in modes:
        xs, ys, yerr = [], [], []
        for I in Is:
            e = agg.get((m, I), {}).get("phi_bed_steady")
            if e:
                xs.append(I)
                ys.append(e["mean"])
                yerr.append(e["sd"])
        if xs:
            c, mk = styles[m]
            ax.errorbar(xs, ys, yerr=yerr, marker=mk, color=c, capsize=3,
                        label=f"{m} wall (bed solid fraction)")
    ax.set_xscale("log")
    ax.set_xlabel("inertial number I")
    ax.set_ylabel("phi (bed solid fraction)")
    ax.set_title("plane shear phi(I): rough vs smooth wall, 3 seeds")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phi_vs_I.png", dpi=150)
    plt.close(fig)

    # --- seed spread -------------------------------------------------------
    by_mi = defaultdict(list)
    for r in ok:
        by_mi[(r["wall_mode"], float(r["I_target"]))].append(r)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in modes:
        c, mk = styles[m]
        for seed in (5279, 6113, 7919):
            xs, ys = [], []
            for I in Is:
                for r in by_mi.get((m, I), []):
                    if r.get("seed") == seed:
                        v = _num(r.get("mu_wall"))
                        if v is not None:
                            xs.append(I)
                            ys.append(v)
            if xs:
                ax.plot(xs, ys, marker=mk, ls=":", color=c, alpha=0.6,
                        label=f"{m}, seed {seed}")
    ax.plot(Igrid, ref_mu(Igrid), "k-", lw=2, label="reference mu(I)")
    ax.set_xscale("log")
    ax.set_xlabel("inertial number I")
    ax.set_ylabel("mu (platen traction)")
    ax.set_title("per-seed mu(I): sample-to-sample spread")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_seed_spread.png", dpi=150)
    plt.close(fig)


def main():
    ok, fails = load_records()
    agg = aggregate(ok)
    (OUT / "tables_r13.md").write_text(tables(agg, fails, ok))
    figures(agg, ok)
    payload = {
        "reference": reference(),
        "n_ok": len(ok),
        "n_fail": len(fails),
        "fails": fails,
        "aggregate": {f"{m}|{I:g}": v for (m, I), v in agg.items()},
    }
    (OUT / "analysis_r13.json").write_text(json.dumps(payload, indent=1))
    print(f"ok={len(ok)} fail={len(fails)} -> {OUT}")


if __name__ == "__main__":
    main()
