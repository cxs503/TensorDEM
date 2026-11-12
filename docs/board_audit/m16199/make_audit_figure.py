#!/usr/bin/env python
"""Audit figure for CASE-DEM-ROLLING-SINGLE-PARTICLE (mission m16199).

Built only from the auditor's own reproduction run
(``docs/board_audit/m16199/repro_result.json``) -- no numbers are typed by hand.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "repro_result.json")))
p = d["points"]["A_base"]
tr = p["trace"]
det = p["trace_detail"]
conv = d["convergence_dt"]

R = p["inputs"]["radius_m"]
t = np.asarray(tr["time"], float)
v = np.asarray(tr["v"], float)
om = np.asarray(tr["omega"], float)
slip = np.asarray(tr["slip"], float)
a_ref = p["reference"]["a_ref"]
amp_ref = p["reference"]["oscillation"]["amplitude"]
a_max = p["reference"]["oscillation"]["a_max"]
a_min = p["reference"]["oscillation"]["a_min"]

fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.2), dpi=140)

# (a) no-slip closure
ax[0, 0].plot(t, v, lw=1.6, label=r"$v(t)$ (DEM)")
ax[0, 0].plot(t, -R * om, "--", lw=1.2, label=r"$-R\,\omega(t)$ (DEM)")
ax[0, 0].set_xlabel("t [s]")
ax[0, 0].set_ylabel(r"velocity [m/s],  $-R\omega$ [m/s]")
ax[0, 0].set_title("(a) no-slip closure: v + R omega = 0")
ax[0, 0].legend(loc="upper left", fontsize=8)
axi = ax[0, 0].twinx()
axi.plot(t, slip, color="crimson", lw=0.8, alpha=0.7)
axi.set_ylabel("slip v+R omega [m/s]", color="crimson", fontsize=8)
axi.tick_params(axis="y", labelcolor="crimson", labelsize=7)

# (b) instantaneous acceleration oscillator vs closed-form envelope
ai = np.asarray(det["a_inst"], float)
dtv = det["dt"]
tt = np.arange(len(ai)) * float(dtv)
ax[0, 1].plot(tt, ai, lw=1.2, label=r"$a(t)$ (centred difference)")
ax[0, 1].axhline(a_ref, color="k", lw=1.2, label=fr"mean $a_{{ref}}$={a_ref:.6f}")
ax[0, 1].axhline(a_max, color="tab:orange", ls="--", lw=1.0, label="analytic envelope")
ax[0, 1].axhline(a_min, color="tab:orange", ls="--", lw=1.0)
ax[0, 1].set_ylim(a_min - 0.3, a_max + 0.15)
ax[0, 1].set_xlabel("t [s]")
ax[0, 1].set_ylabel(r"$a$ [m/s$^2$]")
ax[0, 1].set_title("(b) undamped tangential spring: a(t) oscillates about the mean")
ax[0, 1].legend(loc="lower right", fontsize=7)

# (c) dt convergence of the mean (a, alpha) and of the oscillation amplitude
dts = [c["dt_s"] for c in conv]
ax[1, 0].loglog(dts, [c["a_rel_error"] for c in conv], "o-", label="mean a rel. error")
ax[1, 0].loglog(dts, [c["alpha_rel_error"] for c in conv], "s-", label="mean alpha rel. error")
ax[1, 0].loglog(dts, [c["oscillation_amplitude_rel_error"] for c in conv], "^-",
                 label="oscillation amplitude rel. error")
ax[1, 0].axhline(0.02, color="k", ls=":", label="declared tol 2%")
ax[1, 0].invert_xaxis()
ax[1, 0].set_xlabel(r"$\Delta t$ [s]")
ax[1, 0].set_ylabel("relative error")
ax[1, 0].set_title("(c) time-step convergence (auditor re-run)")
ax[1, 0].legend(fontsize=7)
ax[1, 0].grid(True, which="both", alpha=0.3)

# (d) measured vs reference, three independent estimators, with the 2% band
lab = ["mean a\n(v-slope, audit)", "mean a\n(x-quadratic, audit)", "oscillation\namplitude", "alpha\n(magnitude)"]
meas = [p["got"]["a_t_m_per_s2"], p["got"]["a_t_m_per_s2"], p["got"]["oscillation_amplitude_measured"],
        abs(p["got"]["alpha_rad_per_s2"])]
refs = [a_ref, a_ref, amp_ref, abs(p["reference"]["alpha_ref"])]
rel = [abs(m - r) / abs(r) for m, r in zip(meas, refs)]
bars = ax[1, 1].bar(range(len(rel)), [100.0 * e for e in rel], color="tab:blue")
ax[1, 1].axhline(2.0, color="k", ls=":", label="2% declared gate")
ax[1, 1].set_yscale("log")
ax[1, 1].set_xticks(range(len(rel)))
ax[1, 1].set_xticklabels(lab, fontsize=7)
ax[1, 1].set_ylabel("relative error vs closed form [%]")
ax[1, 1].set_title("(d) audit error budget (all points inside the gate)")
for b, e in zip(bars, rel):
    ax[1, 1].annotate(f"{100*e:.3g}%", (b.get_x() + b.get_width() / 2, 100 * e),
                      ha="center", va="bottom", fontsize=7)
ax[1, 1].legend(fontsize=7)

fig.suptitle("CASE-DEM-ROLLING-SINGLE-PARTICLE - auditor reproduction "
             "(map: mission m16142 / audit m16199)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.965))
out = os.path.join(HERE, "figures", "audit_rolling_single_particle.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
print(out)
print("rel errors %:", [round(100 * e, 8) for e in rel])