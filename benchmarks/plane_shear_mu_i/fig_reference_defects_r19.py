"""Round 19 (m21228) figure: the two reference-handling defects in the round-18
higher-N sweep, drawn from the shipped code's own formulas.

Panel (a): ``ref_phi(I)`` as re-declared inside ``high_n_sweep.py`` before this
round (``0.58 + 0.06/(I0/I+1)``, growing with I = *compaction*, unphysical) vs
``reference_law.ref_phi`` (decreasing with I = dilatancy, the round-2/round-7
fix).  Panel (b): the ledger band "error < 10 %" as it was enforced
(``TOL = 0.10`` compared against a residual already in percent, i.e. 0.1 %)
against the 10 % band the ledger asks for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parents[1] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reference_law import I_0, PHI_MAX, PHI_MIN, ref_phi  # noqa: E402

OUT = HERE / "runs" / "high_n_reference_audit"


def old_ref_phi(I):                       # as shipped in round 18 (m21111)
    return 0.58 + (0.64 - 0.58) / (0.30 / I + 1.0)


def main() -> int:
    I = np.logspace(-3, 0, 300)
    new, old = ref_phi(I), old_ref_phi(I)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    ax1.axhspan(PHI_MIN, PHI_MAX, color="0.9", zorder=0,
                label=f"reference band [{PHI_MIN}, {PHI_MAX}]")
    ax1.plot(I, old, "-", color="tab:red", lw=1.8,
             label=r"round-18 local: $0.58+0.06/(I_0/I+1)$ (grows = compaction)")
    ax1.plot(I, new, "k-", lw=1.8,
             label=r"reference_law: $\phi_{max}-(\phi_{max}-\phi_{min})/(I_0/I+1)$")
    for Iv in (0.01, 0.3):
        ax1.annotate(f"{old_ref_phi(Iv):.3f}", (Iv, old_ref_phi(Iv)),
                     textcoords="offset points", xytext=(4, 6), fontsize=8,
                     color="tab:red")
        ax1.annotate(f"{ref_phi(Iv):.3f}", (Iv, ref_phi(Iv)),
                     textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("inertial number $I$")
    ax1.set_ylabel(r"$\phi$")
    ax1.set_title("(a) dilatancy sign: phi(I) shrinks as I grows")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=7.5, loc="lower right")

    err = np.linspace(0, 20, 400)          # residual, in percent
    ax2.axvspan(0, 0.1, color="tab:red", alpha=0.25,
                label="band actually enforced: $TOL=0.10$ on a percent residual (0.1 %)")
    ax2.axvspan(0, 10.0, color="tab:green", alpha=0.12,
                label="ledger band: error < 10 %")
    ax2.axvline(10.0, color="tab:green", lw=1.5)
    ax2.axvline(0.1, color="tab:red", lw=1.5)
    for e, txt in ((6.0, "6 % error\n(spurious FAIL)"), (15.0, "15 % error\n(FAIL, both)")):
        ax2.annotate(txt, (e, 0.5), ha="center", fontsize=8.5)
    ax2.set_xlim(0, 20)
    ax2.set_yticks([])
    ax2.set_xlabel("|residual| [%]  (as stored in ``err_mu_pct``)")
    ax2.set_title("(b) tolerance unit: 0.1 % enforced where 10 % was required")
    ax2.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig_reference_defects_r19.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    data = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 19,
        "defect": "round-18 higher-N sweep: inverted ref_phi + duplicated triple + TOL unit",
        "I": [1e-3, 1e-2, 3e-2, 1e-1, 3e-1],
        "phi_round18_local": [old_ref_phi(x) for x in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1)],
        "phi_reference_law": [ref_phi(x) for x in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1)],
        "phi_sign_agreement": "round-18 local grows with I; reference must decrease",
        "tol_old": 0.10, "tol_new_pct": 10.0,
        "figure": png.name,
    }
    (OUT / "result.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
