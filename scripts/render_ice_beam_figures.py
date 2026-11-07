"""Render the figures of the ice-beam anchor report from the machine-readable JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ice-beam-bending"
report = json.loads((OUT / "ice_beam_report.json").read_text(encoding="utf-8"))
primary = report["primary"]
anchor = primary["anchor"]
spec = primary["spec"]
figures = []


def save(fig, name, caption):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figures.append({"name": Path(name).stem, "path": str(path.relative_to(ROOT)),
                    "caption": caption})


# 1. deflection profile -------------------------------------------------------
x = np.asarray(anchor["deflection"]["x_over_L"])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(x, 1e3 * np.asarray(anchor["deflection"]["w_dem_m"]), "o", ms=2.5,
        label="DEM lattice equilibrium")
ax.plot(x, 1e3 * np.asarray(anchor["deflection"]["w_timoshenko_m"]), "-",
        label="Timoshenko (plane stress, E)")
ax.plot(x, 1e3 * np.asarray(anchor["deflection"]["w_euler_bernoulli_m"]), "--",
        label="Euler-Bernoulli")
ax.plot(x, 1e3 * np.asarray(primary["anchor"]["deflection"]["w_dem_m"]), ".", ms=0.1)
ax.set_xlabel("x / L")
ax.set_ylabel("mid-plane deflection w [mm]")
ax.set_title(f"DEM vs beam theory, P = {anchor['probe_load_N']:.0f} N, "
             f"d = {1000 * spec['particle_diameter_m']:.1f} mm")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
save(fig, "deflection_profile.png",
     "Mid-plane deflection profile: DEM rate-free lattice equilibrium against the "
     "Timoshenko and Euler-Bernoulli closed forms at 10 % of the analytic failure "
     "load. The DEM profile has the analytic shape but is 11 % small, i.e. the "
     "lattice is stiffer in bending than E I.")

# 2. stress profile -----------------------------------------------------------
layers = anchor["stress"]["layers"]
z = np.asarray([row["z_over_h"] for row in layers])
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.plot(1e-3 * np.asarray([row["sigma_xx_dem_Pa"] for row in layers]), z, "o-",
        ms=4, label="DEM virial $\\sigma_{xx}$")
ax.plot(1e-3 * np.asarray([row["sigma_xx_theory_Pa"] for row in layers]), z, "--",
        label="beam theory $M(z_{na}-z)/I$")
ax.axvline(0.0, color="k", lw=0.6)
ax.set_ylabel("z / h")
ax.set_xlabel("$\\sigma_{xx}$ [kPa]")
ax.set_title("Midspan bending stress through the depth")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
save(fig, "stress_profile.png",
     "Bending stress through the beam depth at midspan (the load line): the DEM "
     "virial stress per lattice layer against the linear beam-theory profile. The "
     "outermost layer reads low because a surface node has half the bonds, and the "
     "whole column carries the local load application, which is why the clean "
     "checks are the transmitted section moment (exact) and the quarter-section "
     "profile (20 % gradient deficit).")

# 3. load-deflection + failure load -------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.0))
probe = anchor["probe_load_N"]
w_mid = anchor["deflection"]["midspan_dem_m"]
w_mid_ts = anchor["deflection"]["midspan_timoshenko_m"]
ax.plot([0.0, 1e3 * w_mid], [0.0, probe], "-", label="DEM (linear, elastic probe)")
ax.plot([0.0, 1e3 * w_mid_ts], [0.0, probe], "--", label="Timoshenko")
ax.axhline(anchor["first_failure"]["first_break_load_from_elastic_probe_N"],
           color="r", lw=0.9,
           label=f"DEM first-bond load "
                 f"{anchor['first_failure']['first_break_load_from_elastic_probe_N']:.0f} N")
ax.axhline(anchor["first_failure"]["analytic_flexural_failure_load_N"],
           color="k", ls=":", lw=0.9,
           label=f"analytic "
                 f"{anchor['first_failure']['analytic_flexural_failure_load_N']:.0f} N")
ax.set_xlabel("midspan deflection [mm]")
ax.set_ylabel("central load P [N]")
ax.set_title("Load-deflection response and predicted ultimate load")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
save(fig, "load_deflection.png",
     "Load-deflection response of the elastic lattice (linear in load) together "
     "with the DEM first-bond failure load and the analytic first-fibre load.")

# 4. crack path ---------------------------------------------------------------
events = primary["cascade_events"]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
for ax, ev, title in ((axs[0], events, "load controlled"),
                      (axs[1], primary["displacement_controlled_events"],
                       "displacement controlled")):
    if not ev:
        continue
    scatter = ax.scatter([e["x_over_L"] for e in ev], [e["z_over_h"] for e in ev],
                         c=[e["event"] for e in ev], cmap="viridis", s=16)
    fig.colorbar(scatter, ax=ax, label="event index")
    ax.set_xlabel("x / L")
    ax.set_ylabel("z / h")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle(f"Brittle damage cascade, d = {1000 * primary['crack_case_diameter_m']:.1f} mm",
             fontsize=9)
save(fig, "crack_path.png",
     "Location of every broken bond in the load-controlled and "
     "displacement-controlled damage cascades. Both initiate exactly at the "
     "midspan bottom fibre, but at h/d = 7 the strength criterion spreads the "
     "damage along the tension fibre instead of driving it through the section.")

# 5. resolution convergence ---------------------------------------------------
table = report["resolution_table"]
h_over_d = np.asarray([row["depth_particles"] for row in table], dtype=float)
fig, ax = plt.subplots(figsize=(5.8, 4.0))
ax.plot(1.0 / h_over_d, 100 * np.asarray(
    [row["rel_error_midspan_vs_timoshenko_plane_stress"] for row in table]), "o-",
    label="vs plane-stress $E$")
ax.plot(1.0 / h_over_d, 100 * np.asarray(
    [row["rel_error_midspan_vs_timoshenko_plane_strain"] for row in table]), "s--",
    label="vs plate modulus $E/(1-\\nu^2)$")
for row in table:
    ax.annotate(f"{1000 * row['particle_diameter_m']:.1f} mm", 
                (1.0 / row["depth_particles"],
                 100 * row["rel_error_midspan_vs_timoshenko_plane_stress"]),
                textcoords="offset points", xytext=(4, 4), fontsize=6)
ax.set_xlabel("particle diameter / depth  (d / h)")
ax.set_ylabel("midspan deflection error [%]")
ax.set_title("Discretisation bias of the lattice bending stiffness")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(left=0)
save(fig, "resolution_convergence.png",
     "Midspan deflection error versus particle size over depth for both "
     "beam-theory references: the lattice bending bias decays roughly as d/h and "
     "is still 11 % at the specified 5 mm particles, so the anchor is not "
     "resolution converged at the task geometry.")

# 6. negative control ---------------------------------------------------------
control = report["negative_control"]
fig, ax = plt.subplots(figsize=(5.8, 3.8))
for key, label in (("bonded", "bonded beam"), ("unbonded", "no bonds (packing)")):
    samples = control[key]["samples"]
    ax.semilogy([1e6 * s["time_s"] for s in samples],
                [abs(s["midspan_deflection_m"]) for s in samples], "o-", ms=3,
                label=label)
ax.set_xlabel("simulated time [$\\mu$s]")
ax.set_ylabel("|midspan deflection| [m]")
ax.set_title(f"Paired negative control, P = {control['load_N']:.0f} N")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
save(fig, "negative_control.png",
     "Paired negative control at the same load and supports: the bonded beam is in "
     "quasi-static equilibrium while the bond-free packing is in free fall; with "
     "every bond removed the assembled stiffness block has no non-zero entries at "
     "all, so no equilibrium can exist.")

(OUT / "figures.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
report["figures"] = figures
(OUT / "ice_beam_report.json").write_text(json.dumps(report, indent=2, sort_keys=True),
                                          encoding="utf-8")
print(json.dumps(figures, indent=2))