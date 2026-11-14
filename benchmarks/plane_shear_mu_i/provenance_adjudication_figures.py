"""Figures for the round-13 provenance / consistency adjudication."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference_provenance as rp          # noqa: E402

RUNS = HERE / "runs"
OUT = RUNS / "round13"
OUT.mkdir(parents=True, exist_ok=True)
TOL = 0.10

rep = json.loads((RUNS / "provenance_adjudication.json").read_text())

I_REF = np.exp(np.linspace(np.log(1e-3), np.log(0.6), 400))


# --------------------------------------------------------------------------- #
# Fig 1 - measured arms vs the disputed reference envelope
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8.0, 5.4))

lo, hi = rp.envelope_bounds(I_REF)
ax.fill_between(I_REF, lo, hi, color="0.85", zorder=0,
                label="disputed reference envelope (8 corners)")
ax.semilogx(I_REF, rp.PRIMARY_REPO.mu(I_REF) * (1 + TOL), color="C3",
            ls=":", lw=1.2, zorder=1)
ax.semilogx(I_REF, rp.PRIMARY_REPO.mu(I_REF) * (1 - TOL), color="C3",
            ls=":", lw=1.2, label=f"+-{int(TOL*100)} % band on PRIMARY_REPO")
ax.semilogx(I_REF, rp.PRIMARY_REPO.mu(I_REF), color="C3", lw=2.2,
            label="PRIMARY_REPO 0.36/0.84/0.30 (reference_law.py)")
ax.semilogx(I_REF, rp.TEST_GDR_MIDI.mu(I_REF), color="C0", lw=2.2, ls="--",
            label="TEST_GDR_MIDI 0.379/0.641/0.279 (pytest suite)")

markers = ["o", "s", "^", "D", "v"]
for k, a in enumerate(rep["arms"]):
    style = "-" if "law-consistent" in a["label"] else "--"
    lw = 2.4 if "law-consistent" in a["label"] else 1.2
    ax.semilogx(a["I"], a["mu"], marker=markers[k % len(markers)], ls=style, lw=lw,
                ms=7, alpha=0.95 if lw > 2 else 0.65,
                label=f"measured: {a['label']}")

ax.set_xlabel("inertial number  $I$")
ax.set_ylabel(r"effective friction  $\mu$")
ax.set_title("CASE-DEM-PLANE-SHEAR-MUIDI  r13\n"
             "measured $\\mu(I)$ vs the disputed reference envelope (archival, no new runs)")
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=6.8, loc="upper left", framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT / "fig1_mu_vs_I_arms.png", dpi=160)
plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2 - residuals of the law-consistent arm against both candidates
# --------------------------------------------------------------------------- #
law = next(a for a in rep["arms"] if a["label"].startswith("p-consistent, measured"))
fig, ax = plt.subplots(figsize=(8.0, 4.8))
ax.axhspan(-100 * TOL, 100 * TOL, color="C2", alpha=0.16, zorder=0,
           label=f"+-{int(TOL*100)} % acceptance band")
ax.axhline(0.0, color="0.4", lw=0.9)
for v, col, mk in zip(law["verdicts"], ("C3", "C0"), ("o", "s")):
    rel = 100.0 * np.asarray(v["rel_err"])
    ax.semilogx(law["I"], rel, mk + "-", color=col, lw=1.8, ms=8,
                label=f"{v['candidate']}  ({v['n_within_tol']}/{v['n_points']} in band)")
    for x, y in zip(law["I"], rel):
        ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8 if y > 0 else -14), ha="center", fontsize=7.5,
                    color=col)
ax.set_xlabel("inertial number  $I$  (law-consistent: $p=-\\mathrm{tr}\\,\\sigma/3$ for both axes)")
ax.set_ylabel(r"$\mu_{measured}/\mu_{ref}-1$   [%]")
ax.set_title("r13: law-consistent pairing, residual vs both disputed candidates\n"
             "no corner of the envelope passes every point (0/8)")
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig2_law_consistent_residuals.png", dpi=160)
plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3 - kinematic admissibility + the normaliser-mixing artefact
# --------------------------------------------------------------------------- #
cons = rep["consistency"]
idx = np.arange(len(cons))
labels = [f"I={c['I_target']:g}" for c in cons]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
ax = axes[0]
w = 0.38
ax.bar(idx - w / 2, [c["rate_ratio"] for c in cons], w, label="bulk/platen rate ratio",
       color="C0", alpha=0.85)
ax.bar(idx + w / 2, [c["profile_r2"] for c in cons], w, label="velocity-profile $R^2$",
       color="C1", alpha=0.85)
ax.axhline(0.90, color="C3", ls="--", lw=1.3, label="admissibility threshold 0.90")
for i, c in enumerate(cons):
    if not c["servo_valid"]:
        ax.annotate("servo\nFAIL", (i, 0.03), ha="center", fontsize=7,
                    color="C3", fontweight="bold")
ax.set_xticks(idx)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel("dimensionless")
ax.set_title("kinematic admissibility: only 1/4 points is a bulk shear measurement")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

ax = axes[1]
ax.bar(idx, [100 * (c["mixing_inflation"] - 1.0) for c in cons], 0.55, color="C4",
       alpha=0.9)
for i, c in enumerate(cons):
    ax.annotate(f"+{100*(c['mixing_inflation']-1):.0f}%", (i, 100 * (c['mixing_inflation'] - 1) + 0.7),
                ha="center", fontsize=8)
ax.set_xticks(idx)
ax.set_xticklabels(labels)
ax.set_ylabel(r"spurious inflation of $\mu$  [%]")
ax.set_title(r"normaliser mixing: $\mu=\tau/p$ with $I$ from $\sigma_{zz}$"
             "\n" r"inflates $\mu$ by $|\sigma_{zz}|/p$ without moving $I$")
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

fig.tight_layout()
fig.savefig(OUT / "fig3_admissibility_and_normaliser.png", dpi=160)
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob('*.png'))], sep="\n  ")