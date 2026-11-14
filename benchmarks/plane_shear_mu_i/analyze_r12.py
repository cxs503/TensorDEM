"""Round 12 analysis: does a rough (grafted) wall bring mu(I) / phi(I) onto the
da Cruz 2005 / GDR MiDi 2004 reference?

Reads the JSONL produced by ``rough_wall_sweep.py`` (2 wall modes x 4 target
inertial numbers x 3 seeds), aggregates per (mode, I) over seeds, and compares
every available mu estimator against the reference law:

* ``mu_wall``     -- platen traction, tau_wall / p_wall  (the historical device)
* ``mu_full``     -- full-cell Love-Weber virial with momentum-balance
                     normalisation, tau / p
* ``mu_full_zz``  -- same, but the normal stress is taken from sigma_zz
* ``mu_bulk``     -- slab virial, legacy "both endpoints" rule
* ``mu_bulk_mid`` -- slab virial, midpoint rule (m17126; unbiased on a thin slab)

Outputs (all under ``runs/round12/``):
``analysis_r12.json``, ``tables.md`` and the report figures
``fig_mu_vs_I.png``, ``fig_phi_vs_I.png``, ``fig_estimator_bias.png``.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "rough_wall"
RUNS_R12 = HERE / "runs" / "round12"
OUT = HERE / "runs" / "round12"
OUT.mkdir(parents=True, exist_ok=True)

import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from reference_law import ref_mu, ref_phi  # noqa: E402

ESTIMATORS = ("mu_wall", "mu_full", "mu_full_zz", "mu_bulk", "mu_bulk_mid")
LABEL = {"mu_wall": "platen traction", "mu_full": "full-cell virial (p)",
         "mu_full_zz": "full-cell virial (sigma_zz)",
         "mu_bulk": "slab virial (both)", "mu_bulk_mid": "slab virial (midpoint)"}
BAND = 0.10  # acceptance band of the ledger row


def load_records(paths=None):
    """Load every JSONL record, round-12 re-runs superseding round-11 ones.

    The round-12 sweep re-runs the whole 24-point plan (same protocol, same
    seeds -> deterministic), so a (mode, I, seed) point that exists in both
    ``runs/rough_wall/`` (round 11) and ``runs/round12/`` is kept once, from
    the newer file.  FAIL records are kept as-is for the honest-FAIL table.
    """
    if paths is None:
        paths = sorted(RUNS.glob("*.jsonl")) + sorted(RUNS_R12.glob("*.jsonl"))
    recs = []
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_file"] = path.name
            recs.append(rec)
    latest = {}
    for rec in recs:
        if rec.get("status") == "FAIL":
            continue
        key = (rec.get("wall_mode"), rec.get("I_target"), rec.get("seed"))
        latest[key] = rec
    fails = [r for r in recs if r.get("status") == "FAIL"]
    return list(latest.values()) + fails


def _num(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def aggregate(recs):
    """{(mode, I): {estimator: (mean, sd, n), ...}} over the seeds."""
    groups = defaultdict(list)
    for rec in recs:
        if rec.get("status") == "FAIL":
            continue
        groups[(rec["wall_mode"], float(rec["I_target"]))].append(rec)
    agg = {}
    for key, items in groups.items():
        entry = {"n_seeds": len(items), "seeds": sorted(r["seed"] for r in items)}
        for est in ESTIMATORS:
            vals = [v for v in (_num(r.get(est)) for r in items) if v is not None]
            if vals:
                entry[est] = {"mean": float(np.mean(vals)),
                              "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              "n": len(vals)}
            else:
                entry[est] = None
        for extra in ("phi_bed_steady", "phi_bed_final", "gamma_dot_ratio",
                      "profile_r2", "P_rel_err_final", "I_measured_rate",
                      "I_imposed_rate", "mu_wall_final", "dilatancy",
                      "n_particles"):
            vals = [v for v in (_num(r.get(extra)) for r in items) if v is not None]
            if vals:
                entry[extra] = {"mean": float(np.mean(vals)),
                                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                "n": len(vals)}
            else:
                entry[extra] = None
        entry["servo_valid"] = sum(bool(r.get("servo_valid")) for r in items)
        agg[key] = entry
    return agg


def tables(agg, fails):
    lines = ["# Round 12 tables - rough vs smooth wall, mu(I) and phi(I)",
             "",
             "3 seeds per point (5279, 6113, 7919), single device "
             "`ServoShearCase`, only `wall_mode` changes.",
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
    if fails:
        lines += ["", "## Failures (honest FAIL records)", ""]
        for rec in fails:
            lines.append(f"* {rec['wall_mode']} I={rec['I_target']:g} "
                         f"seed={rec['seed']}: {rec.get('error')}")
    return "\n".join(lines) + "\n"


def figures(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted({k[0] for k in agg})
    Is = np.array(sorted({k[1] for k in agg}), dtype=float)
    style = {"smooth": ("o", "#1f77b4"), "rough": ("s", "#d62728")}

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    grid = np.logspace(-3, 0.0, 200)
    ax.plot(grid, ref_mu(grid), "k-", lw=1.6,
            label="da Cruz 2005 / GDR MiDi 2004")
    ax.fill_between(grid, ref_mu(grid) * 0.9, ref_mu(grid) * 1.1,
                    color="k", alpha=0.10, label="+-10 % band")
    for m in modes:
        mk, col = style.get(m, ("o", None))
        xs, ys, es = [], [], []
        for I in Is:
            e = agg.get((m, I), {}).get("mu_bulk_mid")
            if e:
                xs.append(I); ys.append(e["mean"]); es.append(e["sd"])
        ax.errorbar(xs, ys, yerr=es, marker=mk, color=col, capsize=3, lw=1.3,
                    label=f"{m} wall - slab virial (midpoint)")
    ax.set_xscale("log"); ax.set_xlabel("inertial number I  (imposed rate)")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("Plane shear mu(I): grafted (rough) vs flat (smooth) wall")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_mu_vs_I.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(grid, ref_phi(grid), "k-", lw=1.6, label="reference phi(I)*")
    ax.fill_between(grid, ref_phi(grid) * 0.9, ref_phi(grid) * 1.1,
                    color="k", alpha=0.10, label="+-10 % band")
    for m in modes:
        mk, col = style.get(m, ("o", None))
        xs, ys, es = [], [], []
        for I in Is:
            e = agg.get((m, I), {}).get("phi_bed_steady")
            if e:
                xs.append(I); ys.append(e["mean"]); es.append(e["sd"])
        ax.errorbar(xs, ys, yerr=es, marker=mk, color=col, capsize=3, lw=1.3,
                    label=f"{m} wall")
    ax.set_xscale("log"); ax.set_xlabel("inertial number I  (imposed rate)")
    ax.set_ylabel(r"$\phi$ (full-gap bed)")
    ax.set_title("Plane shear phi(I)  (*reference magnitudes PROVENANCE-UNVERIFIED)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_phi_vs_I.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.8 / max(1, len(modes) * len(Is))
    x0 = np.arange(len(ESTIMATORS))
    for k, m in enumerate(modes):
        for j, I in enumerate(Is):
            errs = []
            for est in ESTIMATORS:
                e = agg.get((m, I), {}).get(est)
                errs.append(100.0 * (e["mean"] / ref_mu(I) - 1.0) if e else np.nan)
            off = (k * len(Is) + j - 0.5 * (len(modes) * len(Is) - 1)) * width
            ax.bar(x0 + off, errs, width=width,
                   color=style.get(m, ("o", "C0"))[1],
                   alpha=0.35 + 0.5 * j / max(1, len(Is) - 1),
                   label=f"{m}, I={I:g}")
    ax.axhspan(-10, 10, color="k", alpha=0.08, label="+-10 % band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x0); ax.set_xticklabels([LABEL[e] for e in ESTIMATORS],
                                          fontsize=7, rotation=12)
    ax.set_ylabel("error vs reference mu(I)  [%]")
    ax.set_title("Which estimator carries the mu deficit?")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_estimator_bias.png", dpi=160)
    plt.close(fig)
    return ["fig_mu_vs_I.png", "fig_phi_vs_I.png", "fig_estimator_bias.png"]


def main() -> int:
    recs = load_records()
    fails = [r for r in recs if r.get("status") == "FAIL"]
    agg = aggregate(recs)
    md = tables(agg, fails)
    (OUT / "tables.md").write_text(md)
    figs = figures(agg)
    payload = {
        "n_records": len(recs),
        "n_failed": len(fails),
        "modes": sorted({k[0] for k in agg}),
        "I_targets": sorted({k[1] for k in agg}),
        "reference": {"law": "mu_s + (mu_2 - mu_s)/(I0/I+1)",
                      "mu_s": 0.36, "mu_2": 0.84, "I0": 0.30,
                      "status": "constants as quoted in rounds 1-5"},
        "acceptance_band_pct": 100 * BAND,
        "aggregate": {f"{m}|{I:g}": v for (m, I), v in sorted(agg.items())},
        "figures": figs,
    }
    (OUT / "analysis_r12.json").write_text(json.dumps(payload, indent=2, default=float))
    print(md)
    print("wrote", OUT / "tables.md", OUT / "analysis_r12.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())