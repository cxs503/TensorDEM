#!/usr/bin/env python3
"""m16194 audit figure: why m16136 has zero gated points (no reference comparison possible).

Two panels, all numbers read from docs/audit-m16136/evidence_verification.json
(i.e. from the audit's own re-execution of the ladder, not from the audited report):

  left  - tau*(kappa) on log-log axes, every rung marked as gate-rejected; the Bagnold
          plateau (tau* = const, n = 2) is drawn as the reference expectation;
  right - the three gate diagnostics vs kappa with their declared thresholds
          (energy balance <= 0.30, max overlap/d <= 0.60, steady drift <= 0.30),
          showing that the energy gate misses by 3-5 orders of magnitude.

Writes runs/audit_m16136/fig_gate_diagnostics.png and prints the base64 length.
"""
from __future__ import annotations

import base64
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE.parents[1] / "runs" / "audit_m16136"
EV = json.loads((HERE / "evidence_verification.json").read_text())
P = EV["repro"]["points"]
tags = [t for t in ("kn1e2", "kn1e3", "kn1e4", "kn1e5", "kn1e6") if "kappa" in P[t]]
kap = [P[t]["kappa"] for t in tags]
tst = [P[t]["tau_star"] for t in tags]
en = [P[t]["gate"]["energy_balance_rel"] for t in tags]
ovl = [P[t]["gate"]["max_overlap_over_d"] for t in tags]
drift = [P[t]["gate"]["steady_drift"] if isinstance(P[t]["gate"]["steady_drift"], (int, float))
         else float("nan") for t in tags]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.4))

ax.loglog(kap, tst, "x", ms=9, mew=2, color="#b02a37", label="m16136 rungs (all gate-rejected)")
ax.axhline(tst[0], ls=":", color="#1b6ca8", lw=1.2,
           label=r"Bagnold expectation $\tau^*=$const ($n=2$), anchored at kn1e2")
ax.set_xlabel(r"$\kappa=k_n/(\rho d^3\dot\gamma^2)$")
ax.set_ylabel(r"$\tau^*=\tau/(\rho d^2\dot\gamma^2)$")
ax.set_title("tau*(kappa): 0 of 5 rungs admitted -> no fit, no reference check")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
for k, v, t in zip(kap, tst, tags):
    ax.annotate(t, (k, v), textcoords="offset points", xytext=(4, 6), fontsize=8,
                color="#b02a37")

bx.loglog(kap, en, "o-", color="#6f42c1", label="energy-balance rel. error (tol 0.30)")
bx.loglog(kap, ovl, "s--", color="#b02a37",
          label="max overlap / d (tol 0.60)")
bx.loglog(kap, [d if d == d else 1e-2 for d in drift], "^-.", color="#2c7a4b",
          label="steady drift (tol 0.30)")
bx.axhline(0.30, color="#6f42c1", lw=0.8, ls=":", alpha=0.8)
bx.axhline(0.60, color="#b02a37", lw=0.8, ls=":", alpha=0.8)
nan_tags = [t for t, d in zip(tags, drift) if d != d]
if nan_tags:
    bx.annotate("steady drift = NaN (n/a) at " + ",".join(nan_tags),
                (kap[0], 1e-2), textcoords="offset points", xytext=(4, -14),
                fontsize=8, color="#2c7a4b")
bx.set_xlabel(r"$\kappa$")
bx.set_ylabel("gate diagnostic (log)")
bx.set_title("Gate diagnostics vs thresholds: energy misses by 2-5 decades")
bx.grid(True, which="both", alpha=0.3)
bx.legend(fontsize=8)

fig.suptitle("m16194 board audit of m16136 — CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT "
             "(re-executed ladder, commit 00e704d9)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = RUN / "fig_gate_diagnostics.png"
fig.savefig(out, dpi=140)
b64 = base64.b64encode(out.read_bytes()).decode()
(RUN / "fig_gate_diagnostics.b64").write_text(b64)
print(json.dumps({"fig": str(out), "bytes": out.stat().st_size,
                  "b64_chars": len(b64), "tags": tags, "tau_star": tst,
                  "energy_balance_rel": en, "max_overlap_over_d": ovl,
                  "steady_drift": drift}, indent=1))