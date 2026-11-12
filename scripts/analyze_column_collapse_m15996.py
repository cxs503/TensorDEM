"""m15996 — collect CASE-COLUMN-COLLAPSE round-2 grid points and plot.

Reads the per-case provenance JSONs written by
``benchmarks/case_column_collapse_lube2004.py`` (nothing is typed by hand) and
emits ``artifacts/case_column_collapse_m15996/{summary.json,grid.csv,*.png}``.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "artifacts/case_column_collapse_m15996"
CASES = "runs/case_column_collapse_lube2004/cases"


def load() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(CASES, "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        c, crit = r["case"], r["criteria"]
        rows.append({
            "case": c["label"],
            "a": c["aspect_ratio"], "a_meas": r["aspect_measured"],
            "gpr": c["grains_per_radius"], "mu": c["mu"],
            "mu_r": c["mu_rolling"], "dt_scale": c["dt_scale"],
            "n": r["n_particles"], "dt": r["dt"], "t_end": r["time_end"],
            "stop": r["stop_reason"], "wall": r.get("wall_clock_s"),
            "ratio": r["runout_ratio_p995"], "ratio_max": r["runout_ratio_max"],
            "ref_ratio": crit["runout_ratio"]["reference"],
            "err": crit["runout_ratio"]["relative_error"],
            "pass_ratio": crit["runout_ratio"]["pass"],
            "t99": r["t_reach"],
            "ref_t": crit["emplacement_time"]["reference"],
            "err_t": crit["emplacement_time"]["relative_error"],
            "pass_t": crit["emplacement_time"]["pass"],
        })
    return rows


def main() -> int:
    rows = load()
    if not rows:
        print("no cases found", file=sys.stderr)
        return 1
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "grid.csv"), "w") as fh:
        keys = sorted({k for r in rows for k in r})
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"{'case':>28} {'a':>5} {'N':>6} {'ratio':>7} {'ref':>6} "
          f"{'err%':>7} {'t99':>6} {'ref_t':>6} {'err_t%':>7} stop")
    for r in sorted(rows, key=lambda r: (r["a"], r["gpr"], r["mu"])):
        print(f"{r['case']:>28} {r['a']:5.2f} {r['n']:6d} {r['ratio']:7.3f} "
              f"{r['ref_ratio']:6.3f} {100*r['err']:+7.1f} {r['t99']:6.3f} "
              f"{r['ref_t']:6.3f} {100*r['err_t']:+7.1f} {r['stop']}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a_sorted = sorted({r["a"] for r in rows})
    ax[0].plot(a_sorted, [next(r["ref_ratio"] for r in rows if r["a"] == a) for a in a_sorted],
               "k-", lw=1.5, label="Lube+2004 law")
    for r in rows:
        col = "tab:blue" if r["mu_r"] == 0.4 else "tab:red"
        ax[0].plot(r["a"], r["ratio"], "o", ms=6, color=col)
    ax[0].set_xlabel("a = h_i/r_i"); ax[0].set_ylabel(r"$r_\infty/r_i$")
    ax[0].legend(); ax[0].set_title("runout vs Lube+2004")

    for r in rows:
        ax[1].plot(r["a"], 100 * r["err"], "o", color="tab:blue")
        ax[1].plot(r["a"], 100 * r["err_t"], "s", color="tab:orange", alpha=0.7)
    ax[1].axhspan(-15, 15, color="tab:green", alpha=.12)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel("a"); ax[1].set_ylabel("relative error [%]")
    ax[1].set_title("runout (o) and t99 (s); band = ±15 %")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "grid_vs_a.png"), dpi=130)

    lad = sorted([r for r in rows if r["a"] == 2.0], key=lambda r: r["gpr"])
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot([r["gpr"] for r in lad], [100 * r["err"] for r in lad], "o-")
    ax2.axhspan(-15, 15, color="tab:green", alpha=.12)
    ax2.set_xlabel("grains per column radius"); ax2.set_ylabel("runout error [%]")
    ax2.set_title("resolution ladder at a = 2.0")
    fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "resolution_ladder.png"), dpi=130)

    summary = {"n_cases": len(rows),
               "pass_ratio": sum(r["pass_ratio"] for r in rows),
               "pass_time": sum(r["pass_t"] for r in rows),
               "rows": rows}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nn_cases={len(rows)} pass_ratio={summary['pass_ratio']} pass_time={summary['pass_time']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
