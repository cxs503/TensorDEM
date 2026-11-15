"""Round 20 figures for CASE-DEM-PLANE-SHEAR-MUIDI (no hand-copied numbers).

Reads ``runs/high_n_r20/result.json`` (written by ``high_n_coverage.py
--assemble``) and emits

* ``fig1_mu_vs_I.png``    measured mu (wall / full / bulk) vs the reference law
                          and the two-triple envelope, log-I abscissa;
* ``fig2_error_band.png`` per-point residual of every estimator/abscissa cell
                          against the 10 % acceptance band;
* ``fig3_phi_vs_I.png``   the three phi estimators vs the reference phi(I), with
                          the provenance-withheld note in the caption;
* ``fig4_noise.png``      mu noise floor of the final 25 % window (round-18 P1).

Usage::

    python figures_r20.py [runs/high_n_r20]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parents[1] / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference_law import I_0, MU_2, MU_S, ref_mu, ref_phi  # noqa: E402
from reference_provenance import CANDIDATES  # noqa: E402

ARM_STYLE = {"mu_wall": ("o", "wall (tau/p at platen)"),
             "mu_full": ("s", "full cell (tau/<p>)"),
             "mu_bulk": ("^", "bulk slab (tau/p)")}


def _fgrid():
    import numpy as np
    return np.logspace(math.log10(8e-4), math.log10(0.4), 200)


def fig_mu(rows, out):
    import numpy as np
    I = _fgrid()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    lo = [min(c.mu(i) for c in CANDIDATES) for i in I]
    hi = [max(c.mu(i) for c in CANDIDATES) for i in I]
    ax.fill_between(I, lo, hi, color="0.82", label="reference envelope (2 triples)")
    ax.plot(I, [ref_mu(i) for i in I], "k-", lw=2,
            label=f"PRIMARY_REPO mu_s={MU_S},mu_2={MU_2},I0={I_0}")
    for c in CANDIDATES:
        if c.name != "PRIMARY_REPO":
            ax.plot(I, [c.mu(i) for i in I], "k--", lw=1.2, label=c.name)
    for arm, (mark, lab) in ARM_STYLE.items():
        xs = [r["I_abscissa_best"] for r in rows if not math.isnan(r["I_abscissa_best"])]
        ys = [r[arm] for r in rows if not math.isnan(r["I_abscissa_best"])]
        es = [r[arm] * r["mu_std_rel_pct"] / 100.0 for r in rows
              if not math.isnan(r["I_abscissa_best"])]
        ax.errorbar(xs, ys, yerr=es, fmt=mark + "-", ms=6, lw=1.2,
                    capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("inertial number I (measured bulk rate, fallback imposed)")
    ax.set_ylabel(r"effective friction $\mu$")
    ax.set_title("CASE-DEM-PLANE-SHEAR-MUIDI r20: mu(I), N>1000")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_error(verdict, out):
    import numpy as np
    rows = verdict["rows"]
    xs = [r["I_abscissa_best"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(-10, 10, color="0.85", label="acceptance band +/-10 %")
    ax.axhline(0, color="k", lw=1)
    for arm, (mark, lab) in ARM_STYLE.items():
        ys = [r[f"err_pct__{arm}__I_abscissa_best"] for r in rows]
        ax.plot(xs, ys, mark + "-", ms=6, lw=1.2, label=lab)
    for k, v in enumerate(verdict["mu_verdicts"]):
        if v["abscissa"] != "I_abscissa_best":
            ys = [r[f"err_pct__{v['arm']}__{v['abscissa']}"] for r in rows]
            ax.plot(xs, ys, ":", lw=1.0, alpha=0.7,
                    label=f"{v['arm']} @ imposed" if v["abscissa"] == "I_imposed_rate" else None)
    ax.set_xscale("log")
    ax.set_xlabel("I (measured rate, fallback imposed)")
    ax.set_ylabel("residual vs PRIMARY_REPO [%]")
    ax.set_title("mu residual against the 10 % band")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_phi(rows, out):
    I = _fgrid()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(I, [ref_phi(i) for i in I], "k-", lw=1.6,
            label="reference phi(I) [magnitudes PROVENANCE-UNVERIFIED]")
    for key, mark, lab in (("phi_bed_steady", "o", "bed slab (steady)"),
                           ("phi_extent_final", "s", "extent (shell corrected)"),
                           ("phi_container_final", "^", "container")):
        xs = [r["I_abscissa_best"] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, mark + "-", ms=6, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("I (measured rate, fallback imposed)")
    ax.set_ylabel(r"solid fraction $\phi$")
    ax.set_title("phi(I): dilatancy, estimator spread")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_noise(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [r["I_target"] for r in rows]
    ys = [r["mu_std_rel_pct"] for r in rows]
    ax.plot(xs, ys, "o-")
    ax.axhline(10, color="r", ls="--", label="10 % band")
    for x, y, r in zip(xs, ys, rows):
        ax.annotate(f"N={r['n_particles']:.0f}", (x, y), fontsize=7,
                    xytext=(0, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("I target")
    ax.set_ylabel("mu std of final 25 % window [%]")
    ax.set_title("noise floor at the higher particle count")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "high_n_r20"
    verdict = json.loads((runs / "result.json").read_text())
    rows = verdict["rows"]
    fig_mu(rows, runs / "fig1_mu_vs_I.png")
    fig_error(verdict, runs / "fig2_error_band.png")
    fig_phi(rows, runs / "fig3_phi_vs_I.png")
    fig_noise(rows, runs / "fig4_noise.png")
    print("wrote", ", ".join(sorted(p.name for p in runs.glob("fig*.png"))))


if __name__ == "__main__":
    main()