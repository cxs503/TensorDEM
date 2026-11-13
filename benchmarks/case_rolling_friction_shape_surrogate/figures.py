#!/usr/bin/env python3
"""Figure for XVAL-DEM-ROLLING-FRICTION-AS-SHAPE-SURROGATE.

Reads artifacts/case_rolling_friction_shape_surrogate/result.json (zero
hand-typed numbers) and writes fig_shape_surrogate.png.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[2] / "artifacts" / "case_rolling_friction_shape_surrogate"
d = json.loads((HERE / "result.json").read_text())
rows = d["rows"]
m3 = d["M3_ngon_static"]
sweep = d["sensitivity_hexagon"]
G = d["params"]["g_m_s2"]
f = d["params"]["inertia_factor"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

# (a) Surrogate dynamics: DEM vs closed form, per n-gon (fine dt).
ax = axes[0, 0]
fine = [r for r in rows if r["dt"] == d["params"]["dt_fine_s"]]
for r in fine:
    ax.plot([r["M2_a_ref_m_s2"], r["M2_a_dem_m_s2"]],
            [r["mu_r_eff"], r["mu_r_eff"]], "-", color="0.6", lw=1)
    ax.plot([r["M2_a_ref_m_s2"]], [r["mu_r_eff"]], "s", ms=8, mfc="tab:blue", mec="k",
            label=f"n={r['n']} (ref)")
    ax.plot([r["M2_a_dem_m_s2"]], [r["mu_r_eff"]], "o", ms=8, mfc="tab:red", mec="k",
            label=f"n={r['n']} (DEM)" if r["n"] == fine[0]["n"] else None)
    ax.annotate(f"{r['M2_rel_err_pct']:.2e}%", (r["M2_a_dem_m_s2"], r["mu_r_eff"]),
                textcoords="offset points", xytext=(6, 8), fontsize=8)
ax.set_xlabel(r"down-slope acceleration $a$ (m/s$^2$) at $\theta_{ref}$")
ax.set_ylabel(r"surrogate $\mu_{r,eff}=\tan(\pi/n)$")
ax.set_title("(a) Surrogate sphere: DEM vs closed form per n-gon")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.3)

# (b) Surrogate mapping + threshold matching.
ax = axes[0, 1]
ns = sorted({r["n"] for r in rows})
mus = [math.tan(math.radians(180.0 / n)) for n in ns]
thcs = [180.0 / n for n in ns]
ax.plot(ns, mus, "o-", color="tab:blue", label=r"$\mu_{r,eff}=\tan(\pi/n)$")
ax2 = ax.twinx()
ax2.plot(ns, thcs, "s--", color="tab:orange", label=r"$\theta_c=\pi/n$")
ax2.plot(ns, [math.degrees(math.atan(m)) for m in mus], "s:", color="tab:red",
         label=r"$\arctan(\mu_{r,eff})$ (sphere)")
ax.set_xlabel("n-gon sides n")
ax.set_ylabel(r"$\mu_{r,eff}$", color="tab:blue")
ax2.set_ylabel(r"$\theta_c$ (deg)", color="tab:orange")
ax.set_title("(b) Shape surrogate mapping & threshold match")
ax.grid(alpha=0.3)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

# (c) Convergence: M2 error vs dt (log-log).
ax = axes[1, 0]
for n in ns:
    sub = [r for r in rows if r["n"] == n]
    ax.loglog([r["dt"] for r in sub], [r["M2_rel_err_pct"] for r in sub], "o-",
            label=f"n={n}")
ax.axhline(15.0, color="tab:red", ls=":", lw=1.5, label="15 % gate")
ax.set_xlabel(r"$\Delta t$ (s)")
ax.set_ylabel(r"M2 rel err (%)")
ax.set_title("(c) Time-step convergence of M2")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

# (d) Sensitivity: M2 error vs theta_ref (hexagon).
ax = axes[1, 1]
ax.plot([s["theta_ref_deg"] for s in sweep], [s["rel_err_pct"] for s in sweep],
        "o-", color="tab:green")
for s in sweep:
    ax.annotate(f"{s['rel_err_pct']:.2e}%", (s["theta_ref_deg"], s["rel_err_pct"]),
              textcoords="offset points", xytext=(4, 6), fontsize=8)
ax.axhline(15.0, color="tab:red", ls=":", lw=1.5, label="15 % gate")
ax.set_xlabel(r"$\theta_{ref}$ (deg)")
ax.set_ylabel(r"M2 rel err (%)")
ax.set_title("(d) Sensitivity: hexagon surrogate across $\theta_{ref}$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Rolling friction as a shape surrogate (m17823, round 2) — "
             f"verdict {d['verdict']}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(HERE / "fig_shape_surrogate.png", dpi=150)
print("wrote", HERE / "fig_shape_surrogate.png")
