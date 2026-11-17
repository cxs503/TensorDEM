"""m24435 (see-through of m22550 / PR#1722): render the drained 9-point
polydispersity sweep against the archived mu(I)/phi(I) law.

Reads docs/m22550/outputs/r26/r26_sweep_table.json (produced by
benchmarks/plane_shear_mu_i/analyse_polydispersity_r26.py once cj4227 drained)
and emits docs/figures/r26_mu_phi_vs_law.png.  Zero hand-copied numbers: every
ordinate is read from the table.
"""
import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
tab = json.loads((REPO / "docs/m22550/outputs/r26/r26_sweep_table.json").read_text())

# reference law (single source of truth: reference_law.py)
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
PHI_MAX, PHI_MIN = 0.64, 0.58


def ref_mu(I):
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def ref_phi(I):
    return PHI_MAX - (PHI_MAX - PHI_MIN) / (I_0 / I + 1.0)


def group(w):
    rows = [r for r in tab.values() if abs(r["polydispersity"] - w) < 1e-12]
    rows.sort(key=lambda r: r["I_measured"])
    return (np.array([r["I_measured"] for r in rows]),
            np.array([r["mu_wall"] for r in rows]),
            np.array([r["phi_gap"] for r in rows]),
            np.array([r["psi6_mean"] for r in rows]))


fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
Ic = np.logspace(math.log10(1e-3), math.log10(0.3), 200)
ax[0].plot(Ic, ref_mu(Ic), "k-", lw=2, label=r"law $\mu_s+(\mu_2-\mu_s)/(I_0/I+1)$")
ax[1].plot(Ic, ref_phi(Ic), "k-", lw=2, label=r"law $\phi_{max}-(\phi_{max}-\phi_{min})/(I_0/I+1)$")
styles = {0.0: ("o", "tab:blue"), 0.25: ("s", "tab:orange"), 0.5: ("^", "tab:green")}
for w in (0.0, 0.25, 0.5):
    I, mu, phi, psi = group(w)
    mk, col = styles[w]
    ax[0].plot(I, mu, mk, color=col, ms=7, label=f"$w={w:g}$ ($\\psi_6$={psi[0]:.2f})")
    ax[1].plot(I, phi, mk, color=col, ms=7, label=f"$w={w:g}$")
for a, yl, lo, hi in ((ax[0], r"$\mu_{wall}$", 0.0, 0.9), (ax[1], r"$\phi_{gap}$", 0.5, 0.95)):
    a.set_xscale("log"); a.set_xlabel(r"measured $I$")
    a.set_ylabel(yl); a.set_ylim(lo, hi)
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
ax[0].set_title(r"$\mu_{wall}(I)$: polydispersity lifts $\mu$ off the frictionless branch")
ax[1].set_title(r"$\phi_{gap}(I)$: above the (3-D, provenance-unverified) reference")
fig.tight_layout()
out = REPO / "docs/figures/r26_mu_phi_vs_law.png"
fig.savefig(out, dpi=140)
print("wrote", out)

# console summary (for the report, programmatic)
print(f"{'w':>4} {'I_t':>5} {'I_m':>7} {'mu':>7} {'ref_mu':>7} {'err%':>7} {'phi':>7} {'ref_phi':>8} {'psi6':>6}")
for r in sorted(tab.values(), key=lambda r: (r["polydispersity"], r["I_measured"])):
    print(f"{r['polydispersity']:>4g} {r['I_target']:>5g} {r['I_measured']:>7.4f} "
          f"{r['mu_wall']:>7.4f} {r['ref_mu']:>7.4f} {100*r['mu_rel_err']:>7.1f} "
          f"{r['phi_gap']:>7.4f} {r['ref_phi']:>8.4f} {r['psi6_mean']:>6.3f}")
