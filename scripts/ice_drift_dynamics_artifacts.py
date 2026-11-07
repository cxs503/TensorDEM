"""Generate acceptance JSON + figures for the IN27-S6 sea-ice drift closure.

Run:  python scripts/ice_drift_dynamics_artifacts.py
Writes artifacts/ice_drift_dynamics/{acceptance.json,*.png}
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.ice_drift_dynamics import (  # noqa: E402
    DriftEnvironment,
    FloeInertia,
    coriolis_parameter,
    drift_case_matrix,
    ekman_coupled_drift,
    ekman_decay_depth,
    ekman_profile,
    ekman_surface_current,
    ekman_transport,
    nansen_angle_from_drag_frequency,
    simulate_free_drift,
    solve_ekman_finite_difference,
    solve_steady_drift,
    steady_drift_analytic,
    water_drag,
    added_mass_tensor,
)

OUT = ROOT / "artifacts" / "ice_drift_dynamics"
OUT.mkdir(parents=True, exist_ok=True)

K_EDDY = 0.05           # IOBL eddy viscosity [m^2/s]
INERTIA = FloeInertia()  # h = 1.5 m, c_a = 0.5
REF_PHI, REF_WIND = 75.0, 10.0
OBS_BAND = (20.0, 40.0)  # observed Nansen angle band [deg]
WF_BAND = (0.01, 0.03)   # observed wind factor band [-]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})

acceptance = {
    "node": "IN27-S6",
    "domain": "sea-ice drift dynamics closure",
    "repository": "TensorDEM",
    "module": "src/tensordem/ice_drift_dynamics.py",
    "tests": "tests/test_ice_drift_dynamics.py",
    "parameters": {
        "ice_thickness_m": INERTIA.ice_thickness,
        "ice_density_kg_m3": INERTIA.ice_density,
        "added_mass_coefficient": INERTIA.added_mass_coefficient,
        "air_density_kg_m3": 1.30,
        "water_density_kg_m3": 1025.0,
        "cd_air": 1.8e-3,
        "cd_water": 5.5e-3,
        "eddy_viscosity_m2_s": K_EDDY,
        "coriolis_includes_added_mass": True,
    },
    "criteria": {},
}

# ---------------------------------------------------------------------------
# Case suite
# ---------------------------------------------------------------------------
rows = []
for phi, wind in drift_case_matrix():
    env = DriftEnvironment(latitude_deg=phi, wind_speed=wind)
    ana = steady_drift_analytic(env, INERTIA, include_added_mass_coriolis=True)
    num = solve_steady_drift(env, INERTIA, include_added_mass_coriolis=True,
                             u_guess=(ana["u"], ana["v"]))
    cpl = ekman_coupled_drift(env, INERTIA, eddy_viscosity=K_EDDY,
                              include_added_mass_coriolis=True)
    rows.append({
        "latitude_deg": phi,
        "wind_speed_m_s": wind,
        "f_1_s": env.f,
        "wind_stress_N_m2": ana["tau_wind"],
        "wind_only_angle_deg": ana["angle_right_deg"],
        "wind_only_speed_m_s": ana["speed"],
        "wind_only_wind_factor": ana["wind_factor"],
        "drag_frequency_1_s": ana["drag_frequency"],
        "arctan_f_over_cd_deg": nansen_angle_from_drag_frequency(env.f, ana["drag_frequency"]),
        "ekman_coupled_angle_deg": cpl["angle_right_deg"],
        "ekman_coupled_speed_m_s": cpl["speed"],
        "ekman_coupled_wind_factor": cpl["wind_factor"],
        "ekman_surface_current_m_s": cpl["surface_current_speed"],
        "ekman_surface_current_dir_deg": cpl["surface_current_dir_deg"],
        "ekman_decay_depth_m": cpl["decay_depth"],
        "momentum_residual_N_m2": ana["residual"],
        "coupled_converged": cpl["coupling_converged"],
        "coupled_iterations": cpl["coupling_iterations"],
        "relative_wind_speed_deficit": (ana["speed"] - num["speed"]) / ana["speed"],
    })
acceptance["cases"] = rows

angles = [r["ekman_coupled_angle_deg"] for r in rows]
wfs = [r["ekman_coupled_wind_factor"] for r in rows]

acceptance["criteria"]["a_nansen_angle_20_40_deg"] = {
    "definition": "turning angle of the Ekman-coupled free drift to the right of the wind",
    "values_deg": angles,
    "min": min(angles), "max": max(angles), "mean": float(np.mean(angles)),
    "band_deg": list(OBS_BAND),
    "pass": bool(min(angles) >= OBS_BAND[0] and max(angles) <= OBS_BAND[1]),
    "reference_case_deg": next(r["ekman_coupled_angle_deg"] for r in rows
                               if r["latitude_deg"] == REF_PHI and r["wind_speed_m_s"] == REF_WIND),
}
acceptance["criteria"]["b_wind_factor_1_3_percent"] = {
    "definition": "U_ice / U_wind (Ekman-coupled free drift)",
    "values_percent": [100.0 * w for w in wfs],
    "min_percent": 100.0 * min(wfs), "max_percent": 100.0 * max(wfs),
    "mean_percent": 100.0 * float(np.mean(wfs)),
    "band_percent": [100.0 * WF_BAND[0], 100.0 * WF_BAND[1]],
    "pass": bool(min(wfs) >= WF_BAND[0] and max(wfs) <= WF_BAND[1]),
}

# --- criterion (c): Ekman spiral -------------------------------------------
ek = []
for phi in (60.0, 75.0, 85.0):
    f = coriolis_parameter(phi)
    for K in (0.02, K_EDDY):
        fd = solve_ekman_finite_difference(K, f, 0.234 + 0.0j, n=6001, n_decay_depths=10.0)
        ek.append({
            "latitude_deg": phi,
            "f_1_s": f,
            "eddy_viscosity_m2_s": K,
            "decay_depth_analytic_m": fd["decay_depth_analytic"],
            "decay_depth_numeric_m": fd["decay_depth_numeric"],
            "decay_depth_rel_error": abs(fd["decay_depth_numeric"] - fd["decay_depth_analytic"])
            / fd["decay_depth_analytic"],
            "surface_veer_right_deg": fd["surface_veer_right_deg"],
            "surface_speed_m_s": fd["surface_speed"],
            "transport_numeric_m2_s": [fd["transport_numeric"].real, fd["transport_numeric"].imag],
            "transport_analytic_m2_s": [fd["transport_analytic"].real, fd["transport_analytic"].imag],
            "transport_rel_error": abs(fd["transport_numeric"] - fd["transport_analytic"])
            / abs(fd["transport_analytic"]),
            "profile_rel_error": fd["rel_error_profile"],
        })
acceptance["criteria"]["c_ekman_decay_depth"] = {
    "definition": "D_E = sqrt(2K/f) recovered by an independent tridiagonal FD solve",
    "cases": ek,
    "max_decay_depth_rel_error": max(e["decay_depth_rel_error"] for e in ek),
    "max_veer_deviation_deg": max(abs(abs(e["surface_veer_right_deg"]) - 45.0) for e in ek),
    "max_transport_rel_error": max(e["transport_rel_error"] for e in ek),
    "pass": bool(max(e["decay_depth_rel_error"] for e in ek) < 1e-2
                 and max(abs(abs(e["surface_veer_right_deg"]) - 45.0) for e in ek) < 0.2
                 and max(e["transport_rel_error"] for e in ek) < 1e-2),
}

# --- criterion (d): no-Coriolis negative control ----------------------------
env0 = DriftEnvironment(latitude_deg=0.0, wind_speed=10.0)
ana0 = steady_drift_analytic(env0, INERTIA, include_added_mass_coriolis=True)
num0 = solve_steady_drift(env0, INERTIA, include_added_mass_coriolis=True)
sim0 = simulate_free_drift(env0, INERTIA, duration=20 * 86400.0, dt=600.0)
simR = simulate_free_drift(DriftEnvironment(latitude_deg=REF_PHI, wind_speed=REF_WIND),
                           INERTIA, duration=40 * 86400.0, dt=600.0, include_added_mass_coriolis=True)
numR = solve_steady_drift(DriftEnvironment(latitude_deg=REF_PHI, wind_speed=REF_WIND),
                          INERTIA, include_added_mass_coriolis=True)
acceptance["criteria"]["d_no_coriolis_negative_control"] = {
    "definition": "f = 0 -> turning angle exactly 0",
    "analytic_angle_deg": ana0["angle_right_deg"],
    "newton_angle_deg": num0["angle_right_deg"],
    "rk4_final_angle_deg": sim0["final_angle_right_deg"],
    "analytic_cross_wind_component_m_s": ana0["v"],
    "pass": bool(abs(ana0["angle_right_deg"]) < 1e-10 and abs(num0["angle_right_deg"]) < 1e-8
                 and abs(sim0["final_angle_right_deg"]) < 1e-6),
}

# --- criterion (e): coverage ------------------------------------------------
acceptance["criteria"]["e_case_matrix_coverage"] = {
    "cases": len(rows),
    "latitude_deg": sorted({r["latitude_deg"] for r in rows}),
    "wind_speed_m_s": sorted({r["wind_speed_m_s"] for r in rows}),
    "pass": bool(len(rows) >= 4
                 and min(r["latitude_deg"] for r in rows) == 60.0
                 and max(r["latitude_deg"] for r in rows) == 85.0
                 and min(r["wind_speed_m_s"] for r in rows) == 5.0
                 and max(r["wind_speed_m_s"] for r in rows) == 15.0),
}

# --- supporting numbers -----------------------------------------------------
acceptance["verification"] = {
    "analytic_momentum_residual_max_N_m2": max(abs(r["momentum_residual_N_m2"]) for r in rows),
    "arctan_identity_max_error_deg": max(abs(r["arctan_f_over_cd_deg"] - r["wind_only_angle_deg"])
                                         for r in rows),
    "relative_wind_speed_deficit_percent": [
        100.0 * r["relative_wind_speed_deficit"] for r in rows],
    "rk4_vs_newton_speed_rel_diff": abs(simR["final_speed"] - numR["speed"]) / numR["speed"],
    "rk4_vs_newton_angle_diff_deg": abs(simR["final_angle_right_deg"] - numR["angle_right_deg"]),
    "added_mass_tensor_diag_kg": np.diag(added_mass_tensor(50.0)).tolist(),
}
sensitivity = []
for h in (1.0, 1.5, 2.0, 3.0):
    for K in (0.02, 0.03, 0.05, 0.08):
        env = DriftEnvironment(latitude_deg=REF_PHI, wind_speed=REF_WIND)
        cpl = ekman_coupled_drift(env, FloeInertia(ice_thickness=h), eddy_viscosity=K)
        sensitivity.append({"ice_thickness_m": h, "eddy_viscosity_m2_s": K,
                            "angle_deg": cpl["angle_right_deg"],
                            "wind_factor_percent": 100.0 * cpl["wind_factor"]})
acceptance["sensitivity"] = sensitivity
acceptance["all_pass"] = all(v.get("pass", True) for v in acceptance["criteria"].values()
                             if isinstance(v, dict))

# ---------------------------------------------------------------------------
# Figure 1: Nansen angle vs wind speed
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
winds = np.linspace(3.0, 18.0, 61)
ax.axhspan(*OBS_BAND, color="tab:green", alpha=0.12, label="observed Nansen band 20-40 deg")
for phi, ls in zip((60.0, 70.0, 75.0, 85.0), ("-", "-", "-", "-")):
    coupled, wind_only = [], []
    for w in winds:
        env = DriftEnvironment(latitude_deg=phi, wind_speed=float(w))
        coupled.append(ekman_coupled_drift(env, INERTIA, eddy_viscosity=K_EDDY)["angle_right_deg"])
        wind_only.append(steady_drift_analytic(env, INERTIA,
                                              include_added_mass_coriolis=True)["angle_right_deg"])
    ax.plot(winds, coupled, ls, lw=1.8, label=f"Ekman-coupled, phi={phi:.0f} deg")
    ax.plot(winds, wind_only, ":", lw=1.2, color=ax.lines[-1].get_color(),
            label=f"wind-only, phi={phi:.0f} deg")
ax.plot([r["wind_speed_m_s"] for r in rows], angles, "ko", ms=5, zorder=5,
        label="acceptance cases (12)")
ax.set_xlabel("wind speed $U_{wind}$ [m/s]")
ax.set_ylabel("turning angle to the right of the wind [deg]")
ax.set_title("Nansen turning angle: wind-only vs Ekman-coupled free drift")
ax.legend(fontsize=6.5, ncol=2, loc="upper right")
fig.savefig(OUT / "nansen_angle_vs_wind.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Ekman spiral (hodograph + profile + FD overlay)
# ---------------------------------------------------------------------------
f75 = coriolis_parameter(75.0)
tau = 0.234 + 0.0j
D = ekman_decay_depth(K_EDDY, f75)
zz = np.linspace(0.0, 3.0 * D, 220)
w = ekman_profile(K_EDDY, f75, tau, zz)
fd = solve_ekman_finite_difference(K_EDDY, f75, tau, n=4001, n_decay_depths=8.0)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
ax = axes[0]
ax.plot(w.real * 100, w.imag * 100, "-", lw=1.6)
ax.plot(w.real[0] * 100, w.imag[0] * 100, "o", ms=6, label="surface ($w_0$)")
ax.plot([0], [0], "k+")
ax.plot(fd["w"].real[::200] * 100, fd["w"].imag[::200] * 100, "x", ms=5,
        label="FD solve")
ax.set_xlabel("u [cm/s]"), ax.set_ylabel("v [cm/s]")
ax.set_title(f"Ekman hodograph ($\\tau$ = {tau.real:.3f} N/m$^2$)\nsurface veer "
             f"{fd['surface_veer_right_deg']:.2f} deg right")
ax.legend(fontsize=7)
ax = axes[1]
ax.plot(np.abs(w) * 100, zz, "-", lw=1.6)
ax.plot(fd["z"], np.abs(fd["w"]) * 100, "--", lw=1.0, label="FD")
ax.axvline(abs(w[0]) * 100 / math.e, color="tab:red", ls=":", lw=1.0)
ax.axhline(D, color="tab:red", ls=":", lw=1.0,
           label=f"$D_E=\\sqrt{{2K/f}}$={D:.2f} m")
ax.invert_yaxis()
ax.set_xlabel("$|w|$ [cm/s]"), ax.set_ylabel("depth z [m]")
ax.set_title("speed decay (e-fold at $D_E$)")
ax.legend(fontsize=7)
ax = axes[2]
ax.plot(zz, np.degrees(np.angle(w)), lw=1.6)
ax.axhline(-45.0, color="tab:red", ls=":", lw=1.0, label="-45 deg (surface)")
ax.axhline(-90.0, color="tab:orange", ls=":", lw=1.0, label="-90 deg (transport)")
ax.set_xlabel("depth z [m]"), ax.set_ylabel("direction [deg]")
ax.set_title("veering with depth (NH, right of stress)")
ax.legend(fontsize=7)
fig.savefig(OUT / "ekman_spiral.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: wind factor vs angle (acceptance plane)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.add_patch(plt.Rectangle((WF_BAND[0] * 100, OBS_BAND[0]),
                           (WF_BAND[1] - WF_BAND[0]) * 100,
                           OBS_BAND[1] - OBS_BAND[0],
                           color="tab:green", alpha=0.12,
                           label="acceptance box"))
for r in rows:
    ax.plot(r["ekman_coupled_wind_factor"] * 100, r["ekman_coupled_angle_deg"], "o",
            color="tab:blue")
    ax.annotate(f"{r['latitude_deg']:.0f}N/{r['wind_speed_m_s']:.0f}",
                (r["ekman_coupled_wind_factor"] * 100, r["ekman_coupled_angle_deg"]),
                textcoords="offset points", xytext=(4, -8), fontsize=6)
ax.set_xlabel("wind factor $U_{ice}/U_{wind}$ [%]")
ax.set_ylabel("Nansen angle [deg]")
ax.set_title("free-drift state vs observed wind-factor / turning-angle box")
ax.legend(fontsize=8)
fig.savefig(OUT / "wind_factor_vs_angle.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: transient approach + no-Coriolis control, incl. added-mass effect
# ---------------------------------------------------------------------------
envR = DriftEnvironment(latitude_deg=REF_PHI, wind_speed=REF_WIND)
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
ax = axes[0]
for ca, ls in ((0.0, "-"), (0.5, "--"), (1.5, "-.")):
    sim = simulate_free_drift(envR, FloeInertia(added_mass_coefficient=ca),
                              duration=30 * 86400.0, dt=300.0)
    ax.plot(sim["t"] / 86400.0, sim["speed"] * 100, ls, lw=1.4,
            label=f"$c_a$={ca} (spin-up {sim['spin_up_time'] / 86400.0:.1f} d)")
    ax.axhline(numR["speed"] * 100, color="k", lw=0.8, ls=":")
ax.set_xlabel("time [d]"), ax.set_ylabel("$|U_{ice}|$ [cm/s]")
ax.set_title("spin-up: virtual mass slows the transient, not the steady state")
ax.legend(fontsize=7)
ax = axes[1]
sim_f = simulate_free_drift(envR, INERTIA, duration=30 * 86400.0, dt=300.0,
                            include_added_mass_coriolis=True)
sim_0 = simulate_free_drift(env0, INERTIA, duration=30 * 86400.0, dt=300.0)
ax.plot(sim_f["u"] * 100, sim_f["v"] * 100, lw=1.4, label="75N (Coriolis on)")
ax.plot(sim_0["u"] * 100, sim_0["v"] * 100, lw=1.4, label="equator (f=0 control)")
ax.plot(numR["u"] * 100, numR["v"] * 100, "k*", ms=9, label="Newton steady state")
ax.set_xlabel("u [cm/s]"), ax.set_ylabel("v [cm/s]")
ax.set_title("trajectory in velocity space (wind toward +x)")
ax.legend(fontsize=7)
fig.savefig(OUT / "drift_transient.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: sensitivity map of the turning angle
# ---------------------------------------------------------------------------
hs = sorted({s["ice_thickness_m"] for s in sensitivity})
Ks = sorted({s["eddy_viscosity_m2_s"] for s in sensitivity})
grid = np.zeros((len(hs), len(Ks)))
for s in sensitivity:
    grid[hs.index(s["ice_thickness_m"]), Ks.index(s["eddy_viscosity_m2_s"])] = s["angle_deg"]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
ax.set_xticks(range(len(Ks))), ax.set_xticklabels([f"{k:g}" for k in Ks])
ax.set_yticks(range(len(hs))), ax.set_yticklabels([f"{h:g}" for h in hs])
ax.set_xlabel("eddy viscosity $K$ [m$^2$/s]"), ax.set_ylabel("ice thickness h [m]")
for i in range(len(hs)):
    for j in range(len(Ks)):
        inside = OBS_BAND[0] <= grid[i, j] <= OBS_BAND[1]
        ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=7,
                color="w" if inside else "0.35")
ax.set_title(f"Nansen angle [deg] at {REF_PHI:.0f}N, {REF_WIND:.0f} m/s\n"
             "(white = inside the 20-40 deg band)")
fig.colorbar(im, ax=ax, label="angle [deg]")
fig.savefig(OUT / "sensitivity_angle_map.png")
plt.close(fig)

(OUT / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
print(json.dumps({k: v.get("pass") for k, v in acceptance["criteria"].items()}, indent=2))
print("all_pass:", acceptance["all_pass"])
print("angles:", [round(a, 2) for a in angles])
print("wind factors %:", [round(100 * w, 2) for w in wfs])
print("wrote", OUT)