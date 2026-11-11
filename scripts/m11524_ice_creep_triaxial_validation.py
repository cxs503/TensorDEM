"""Driver for ice creep triaxial chamber and size-effect validation (IL70-N06).

Runs the full validation suite:
1. Stress exponent n from triaxial creep at multiple deviatoric stresses
2. Arrhenius temperature dependence: activation energy Q
3. Size effect homogenization: D/d_p convergence
4. Triaxial creep chamber: sigma_3, sigma_1 combinations

Generates results.json and 4 figures in artifacts/ice_creep_il70n06/.

Usage:
    PYTHONPATH=src python scripts/m11524_ice_creep_triaxial_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensordem.ice_creep import (
    GlenCreepLaw,
    LatticeCreepParameters,
    norton_uniaxial_rate,
    R_GAS,
)
from tensordem.ice_creep_triaxial import (
    triaxial_constant_stress,
    size_effect_scan,
    temperature_sweep,
    arrhenius_activation_energy,
)

OUT_DIR = REPO_ROOT / "artifacts" / "ice_creep_il70n06"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
})

# Problem sizes for the driver (smaller than full validation, but enough to show trends)
N_CELLS = 4
T_END = 300.0
DT = 10.0


def trim_array(arr, limit=100):
    """Downsample array for JSON serialization."""
    arr = np.asarray(arr).ravel()
    if arr.size <= limit:
        return arr.tolist()
    idx = np.unique(np.linspace(0, arr.size - 1, limit).astype(int))
    return arr[idx].tolist()


def jsonable(obj, limit=100):
    """Recursively convert numpy/scipy output into JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, limit) for v in obj]
    if isinstance(obj, np.ndarray):
        return trim_array(obj, limit)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- #
# 1. Stress exponent n from triaxial creep
# --------------------------------------------------------------------------- #
print("=" * 70)
print("1. Stress exponent n from triaxial creep chamber")
print("=" * 70)

law = GlenCreepLaw(a_ref=3.5e-24, n=3.0, q_over_r=7.4e3)
params = LatticeCreepParameters(law=law, temperature_c=-10.0)

# Scan deviatoric stresses
seq_list = [0.5e6, 0.8e6, 1.2e6, 1.8e6]
triaxial_results = []
rates_measured = []

for seq in seq_list:
    print(f"  sigma_1 = {seq/1e6:.1f} MPa, sigma_3 = 0 ...", end=" ", flush=True)
    res = triaxial_constant_stress(
        sigma_1=seq, sigma_3=0.0, n_cells=N_CELLS,
        params=params, t_end=T_END, dt=DT,
    )
    rate = abs(res["rate_measured"])
    rates_measured.append(rate)
    triaxial_results.append({
        "sigma_1_MPa": seq / 1e6,
        "sigma_3_MPa": 0.0,
        "seq_MPa": res["seq"] / 1e6,
        "rate_measured": rate,
        "rate_norton_expected": res["norton_rate_expected"],
        "t": trim_array(res["t"]),
        "strain_deviatoric": trim_array(res["strain_deviatoric"]),
    })
    print(f"rate = {rate:.3e} 1/s")

# Fit stress exponent
log_s = np.log(np.array(seq_list))
log_r = np.log(np.array(rates_measured))
design = np.vstack([log_s, np.ones_like(log_s)]).T
coef, *_ = np.linalg.lstsq(design, log_r, rcond=None)
n_fitted = float(coef[0])
print(f"\n  Fitted stress exponent: n = {n_fitted:.3f}")
print(f"  Acceptance: n in [2.5, 3.5] -> {'PASS' if 2.5 <= n_fitted <= 3.5 else 'WIDE_RANGE'}")

# --------------------------------------------------------------------------- #
# 2. Arrhenius temperature dependence
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("2. Arrhenius temperature dependence: activation energy Q")
print("=" * 70)

temps_c = [-20.0, -10.0]
temp_results = []
rates_temp = []

for t_c in temps_c:
    print(f"  T = {t_c}°C ...", end=" ", flush=True)
    p = LatticeCreepParameters(law=law, temperature_c=t_c)
    res = triaxial_constant_stress(
        sigma_1=1.0e6, sigma_3=0.0, n_cells=N_CELLS,
        params=p, t_end=T_END, dt=DT,
    )
    rate = abs(res["rate_measured"])
    rates_temp.append(rate)
    temp_results.append({
        "temperature_C": t_c,
        "temperature_K": t_c + 273.15,
        "rate_measured": rate,
        "fluidity": p.fluidity(),
    })
    print(f"rate = {rate:.3e} 1/s, A(T) = {p.fluidity():.3e}")

# Fit activation energy
temps_k = np.array([t_c + 273.15 for t_c in temps_c])
Q_fitted = arrhenius_activation_energy(temps_k, np.array(rates_temp))
Q_expected = law.q_over_r * R_GAS
print(f"\n  Fitted activation energy: Q = {Q_fitted/1000:.1f} kJ/mol")
print(f"  Expected (from law):     Q = {Q_expected/1000:.1f} kJ/mol")
print(f"  Acceptance: Q in [50, 80] kJ/mol -> {'PASS' if 50 <= Q_fitted/1000 <= 80 else 'CHECK'}")

# --------------------------------------------------------------------------- #
# 3. Size effect homogenization
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("3. Size effect homogenization: D/d_p convergence")
print("=" * 70)

size_result = size_effect_scan(
    sigma0=1.0e6,
    ratios=(6, 8, 10),
    params=params,
    t_end=T_END,
    dt=DT,
)

print("  D/d_p  |  rate (1/s)   |  rate/norton")
print("  " + "-" * 45)
for i, r in enumerate(size_result["ratios"]):
    rate = size_result["rates_measured"][i]
    ratio = rate / size_result["rate_norton"] if size_result["rate_norton"] > 0 else 0
    print(f"  {r:5d}  |  {rate:.3e}  |  {ratio:.3f}")

conv = size_result["convergence_ratio"]
print(f"\n  Convergence ratio (last two): {conv:.2%}")
print(f"  Acceptance: <= 5% -> {'PASS' if conv <= 0.05 else 'LATTICE_EFFECT'}")

# --------------------------------------------------------------------------- #
# 4. Triaxial creep chamber: sigma_3 scan
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("4. Triaxial creep chamber: confining pressure effect")
print("=" * 70)

sigma_1_fixed = 2.0e6
sigma_3_list = [0.0, 0.5e6, 1.0e6]
triax_confined = []

for s3 in sigma_3_list:
    seq = sigma_1_fixed - s3
    print(f"  sigma_1 = {sigma_1_fixed/1e6:.1f} MPa, sigma_3 = {s3/1e6:.1f} MPa, seq = {seq/1e6:.1f} MPa ...", end=" ", flush=True)
    res = triaxial_constant_stress(
        sigma_1=sigma_1_fixed, sigma_3=s3, n_cells=N_CELLS,
        params=params, t_end=T_END, dt=DT,
    )
    rate = abs(res["rate_measured"])
    triax_confined.append({
        "sigma_1_MPa": sigma_1_fixed / 1e6,
        "sigma_3_MPa": s3 / 1e6,
        "seq_MPa": seq / 1e6,
        "p_MPa": res["p"] / 1e6,
        "rate_measured": rate,
        "rate_norton_expected": res["norton_rate_expected"],
    })
    print(f"rate = {rate:.3e} 1/s")

# --------------------------------------------------------------------------- #
# Write results JSON
# --------------------------------------------------------------------------- #
results = {
    "mission": "IL70-N06",
    "title": "Ice creep: stress exponent n, Arrhenius Q, size effect, triaxial chamber",
    "stress_exponent": {
        "n_fitted": n_fitted,
        "n_expected": 3.0,
        "acceptance_range": [2.5, 3.5],
        "pass": 2.5 <= n_fitted <= 3.5,
        "seq_list_MPa": [s / 1e6 for s in seq_list],
        "rates_measured": rates_measured,
        "triaxial_results": triaxial_results,
    },
    "arrhenius": {
        "Q_fitted_J_mol": Q_fitted,
        "Q_fitted_kJ_mol": Q_fitted / 1000,
        "Q_expected_J_mol": Q_expected,
        "Q_expected_kJ_mol": Q_expected / 1000,
        "acceptance_range_kJ": [50, 80],
        "pass": 50 <= Q_fitted / 1000 <= 80,
        "temperatures_C": temps_c,
        "rates_measured": rates_temp,
        "temperature_results": temp_results,
    },
    "size_effect": {
        "D_over_dp": size_result["ratios"],
        "rates_measured": size_result["rates_measured"],
        "rate_norton": size_result["rate_norton"],
        "convergence_ratio": size_result["convergence_ratio"],
        "acceptance_threshold": 0.05,
        "pass": size_result["convergence_ratio"] <= 0.05,
    },
    "triaxial_confined": {
        "sigma_3_list_MPa": [s / 1e6 for s in sigma_3_list],
        "results": triax_confined,
    },
}

results_path = OUT_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump(jsonable(results), f, indent=2)
print(f"\n✓ Results written to {results_path}")


# --------------------------------------------------------------------------- #
# Generate figures
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("Generating figures ...")
print("=" * 70)

# Figure 1: Strain-time curves for different deviatoric stresses
fig, ax = plt.subplots(figsize=(8, 5))
for i, res in enumerate(triaxial_results):
    t = np.array(res["t"])
    eps = np.array(res["strain_deviatoric"])
    ax.plot(t, eps * 100, label=f"σ = {res['seq_MPa']:.1f} MPa")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Deviatoric strain (%)")
ax.set_title("Triaxial creep: strain-time curves at different deviatoric stresses")
ax.legend()
fig.savefig(OUT_DIR / "fig1_strain_time.png")
print("  ✓ fig1_strain_time.png")
plt.close(fig)

# Figure 2: Log-log fit for stress exponent n
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(seq_list, rates_measured, "o-", label="DEM lattice")
# Fit line
log_s_fit = np.linspace(np.log(min(seq_list)), np.log(max(seq_list)), 50)
log_r_fit = coef[0] * log_s_fit + coef[1]
ax.loglog(np.exp(log_s_fit), np.exp(log_r_fit), "--", label=f"Fit: n = {n_fitted:.2f}")
ax.set_xlabel("Deviatoric stress σ (Pa)")
ax.set_ylabel("Steady-state creep rate (1/s)")
ax.set_title(f"Stress exponent n = {n_fitted:.2f} (Glen n ≈ 3)")
ax.legend()
fig.savefig(OUT_DIR / "fig2_stress_exponent.png")
print("  ✓ fig2_stress_exponent.png")
plt.close(fig)

# Figure 3: Arrhenius plot
fig, ax = plt.subplots(figsize=(7, 5))
inv_t = 1.0 / temps_k
ax.semilogy(inv_t, rates_temp, "o-", label="DEM lattice")
# Fit line
inv_t_fit = np.linspace(min(inv_t), max(inv_t), 50)
ln_rates = np.log(np.array(rates_temp))
design_arr = np.vstack([inv_t, np.ones_like(inv_t)]).T
coef_arr, *_ = np.linalg.lstsq(design_arr, ln_rates, rcond=None)
ln_r_fit = coef_arr[0] * inv_t_fit + coef_arr[1]
ax.semilogy(inv_t_fit, np.exp(ln_r_fit), "--", label=f"Fit: Q = {Q_fitted/1000:.1f} kJ/mol")
ax.set_xlabel("1/T (1/K)")
ax.set_ylabel("Steady-state creep rate (1/s)")
ax.set_title(f"Arrhenius: activation energy Q = {Q_fitted/1000:.1f} kJ/mol")
ax.legend()
fig.savefig(OUT_DIR / "fig3_arrhenius.png")
print("  ✓ fig3_arrhenius.png")
plt.close(fig)

# Figure 4: Size effect convergence
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(size_result["ratios"], size_result["rates_measured"], "o-", label="DEM lattice")
ax.axhline(size_result["rate_norton"], color="k", linestyle="--", label="Norton prediction")
ax.set_xlabel("D / d_p (specimen size / particle size)")
ax.set_ylabel("Steady-state creep rate (1/s)")
ax.set_title("Size effect: homogenization convergence")
ax.legend()
fig.savefig(OUT_DIR / "fig4_size_effect.png")
print("  ✓ fig4_size_effect.png")
plt.close(fig)

print("\n" + "=" * 70)
print("Validation complete.")
print("=" * 70)
