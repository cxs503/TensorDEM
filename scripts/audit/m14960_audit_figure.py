"""Figure for the m14960 cross-audit: why the single-window mu is not a steady state.

(a) tau(gamma) of the two headline runs with the averaged window marked;
(b) mu computed from the last 50/40/25/10 % of the window, against the 0.36-0.40
    literature band and the 10 % gate.

    PYTHONPATH=src python scripts/audit/m14960_audit_figure.py --snapshot /tmp/audit_m14960
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "artifacts", "audit_m14960")
MU_LIT, BAND = 0.38, (0.36, 0.40)
FRACS = (0.50, 0.40, 0.25, 0.10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    snap = ap.parse_args().snapshot
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for tag, col in (("size_small", "C3"), ("size_medium", "C0")):
        d = json.load(open(os.path.join(snap, "artifacts", "shear_cell_m14960", tag + ".json")))
        lv = d["levels"][0]
        g = np.asarray(lv["shear_strain_history"], float)
        t = np.asarray(lv["shear_stress_history"], float)
        n = t.size
        ax[0].plot(g, t, color=col, lw=0.9,
                   label="%s (N=%d), window mu=%.4f" % (tag, d["bed"]["n_particles"], lv["mu"]))
        i0 = int(0.5 * n)
        ax[0].axvspan(g[i0], g[-1], color=col, alpha=0.08)
        mus = [float(t[int((1 - f) * n):].mean() / lv["sigma_pa"]) for f in FRACS]
        ax[1].plot([100 * f for f in FRACS], mus, "o-", color=col,
                   label="%s (stored %.4f)" % (tag, lv["mu"]))
        ax[1].hlines(lv["mu"], 8, 52, color=col, ls=":", lw=1.0)

    ax[1].axhspan(*BAND, color="0.85", zorder=0, label="literature band 0.36-0.40")
    ax[1].axhline(MU_LIT * 1.10, color="r", ls="--", lw=0.9, label="+10 % gate")
    ax[1].axhline(MU_LIT * 0.90, color="r", ls="--", lw=0.9)
    ax[0].set_xlabel(r"shear strain $\gamma$ [-]")
    ax[0].set_ylabel(r"platen traction $\tau$ [Pa]")
    ax[0].set_title("(a) shear traction is still drifting (shaded = averaged window)")
    ax[1].set_xlabel("length of averaging window: last x % of the run")
    ax[1].set_ylabel(r"$\mu_{macro}=\bar\tau/\sigma_n$")
    ax[1].set_title("(b) mu depends on the window -> not a steady state")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "window_sensitivity.png")
    fig.savefig(p, dpi=140)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())