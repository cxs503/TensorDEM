#!/usr/bin/env python3
"""DEM-FEM Coupling Benchmark Suite (CPU).

Hertz: DEM3D spherical indenter (honest-FAIL: discrete lattice ≠ continuum).
Beam: Simply-supported beam under uniform load via finite differences.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tensordem.dem3d import DEM3DConfig, IceDEM3D


# ===== Hertz functions =====
def hertz_force(delta, R, E_eff):
    return (4.0 / 3.0) * E_eff * math.sqrt(R) * delta ** 1.5


def hertz_stiffness(delta, R, E_eff):
    return 2.0 * E_eff * math.sqrt(R * delta)


def fit_power_law(x, y):
    mask = (x > 1e-12) & (y > 1e-12)
    coeffs = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(math.exp(coeffs[1])), float(coeffs[0])


# ===== Beam FD solver =====
def solve_beam_ss(n_elements, L, E, I, q):
    """Simply-supported beam with uniform load q.
    BCs: w(0)=w(L)=0, w''(0)=w''(L)=0. Uses ghost points.
    """
    n = n_elements + 1
    dx = L / n_elements
    ni = n - 2  # interior nodes
    
    K = np.zeros((ni, ni))
    for i in range(ni):
        gi = i + 1
        row = i
        for offset, coeff in [(-2, 1), (-1, -4), (0, 6), (1, -4), (2, 1)]:
            gj = gi + offset
            if gj == 0 or gj == n - 1:
                continue
            elif gj < 0:
                col = 1 - 1
                K[row, col] += coeff * (-1)
            elif gj >= n:
                col = (n - 2) - 1
                K[row, col] += coeff * (-1)
            else:
                col = gj - 1
                K[row, col] += coeff
    K *= E * I / dx**4
    
    F = np.full(ni, q)
    w_interior = np.linalg.solve(K, F)
    w = np.zeros(n)
    w[1:n-1] = w_interior
    return w


def beam_max_deflection_ana(q, L, E, I):
    """Uniform load: δ_max = 5*q*L⁴/(384*E*I)"""
    return 5.0 * q * L**4 / (384.0 * E * I)


def beam_stiffness_uniform(q, L, E, I):
    """Effective stiffness k = q*L / δ_max = 384*E*I/(5*L³)"""
    return 384.0 * E * I / (5.0 * L**3)


# ===== Hertz benchmark =====
def run_hertz_benchmark(nx, ny, nz, radius=0.02, k_bond=1e5, k_contact=1e5,
                        tool_radius=0.1, tool_speed=5.0, max_indent=0.05,
                        sample_every=50, device="cpu"):
    config = DEM3DConfig(
        nx=nx, ny=ny, nz=nz, radius=radius, density=2000.0,
        bond_stiffness=k_bond, contact_stiffness=k_contact,
        breaking_strain=0.5, shear_breaking_strain=0.5,
        contact_damping=0.0, drag=0.0,
        tool_radius=tool_radius, tool_speed=tool_speed, tool_gap=0.0,
        device=device, geometry="box",
        fix_x_edges=False, fix_bottom=True,
        top_platen_enabled=False, bottom_platen_enabled=False,
    )
    sim = IceDEM3D(config)
    initial_tool_z = float(sim.tool_start[2].item())
    E_dem = k_bond / (2.0 * radius)
    nu_dem = 0.25
    E_eff = E_dem / (2.0 * (1.0 - nu_dem ** 2))
    
    deltas, forces, times = [], [], []
    max_steps = min(int(max_indent / (tool_speed * sim.dt)) + 500, 10000)
    
    t0 = time.time()
    for step in range(max_steps):
        sim.step()
        if step % sample_every != 0:
            continue
        tool_z = float(sim.tool_position[2].item())
        delta = initial_tool_z - tool_z
        fz = float(sim.last_reaction[2].item())
        if fz > 0.1 and delta > 0:
            deltas.append(delta)
            forces.append(fz)
            times.append(float(sim.time))
        if delta > max_indent * 1.05:
            break
    
    return {
        "displacements": deltas, "forces": forces, "times": times,
        "E_dem": E_dem, "E_eff": E_eff, "nu_dem": nu_dem,
        "dt": sim.dt, "n_particles": len(sim.positions),
        "n_bonds": len(sim.pairs), "broken_bonds": sim.broken_bonds,
        "elapsed_s": time.time() - t0, "n_steps": step + 1,
    }


# ===== Beam benchmark =====
def run_beam_benchmark():
    L = 1.0       # m
    b = 0.05      # width (m)
    h = 0.1       # height (m)
    E = 2e11      # Pa (steel)
    I = b * h**3 / 12.0  # m⁴
    
    loads_q = np.linspace(1000, 50000, 15)  # N/m (uniform load)
    
    results = []
    for n_elem in [20, 40, 80]:
        deflections = []
        for q in loads_q:
            w = solve_beam_ss(n_elem, L, E, I, q)
            deflections.append(float(np.max(np.abs(w))))
        
        deflections = np.array(deflections)
        analytical = np.array([beam_max_deflection_ana(q, L, E, I) for q in loads_q])
        
        # Fit linear: δ = c * q
        coeffs = np.polyfit(loads_q, deflections, 1)
        c_num = coeffs[0]
        c_ana = 1.0 / beam_stiffness_uniform(1.0, L, E, I)  # per unit load
        err = abs(c_num - c_ana) / c_ana * 100
        
        results.append({
            "n_elements": n_elem,
            "loads_N_m": loads_q.tolist(),
            "deflections_m": deflections.tolist(),
            "analytical_m": analytical.tolist(),
            "numerical_compliance": float(c_num),
            "analytical_compliance": float(c_ana),
            "stiffness_error_pct": float(err),
        })
    
    # Convergence
    err_coarse = results[0]["stiffness_error_pct"]
    err_fine = results[-1]["stiffness_error_pct"]
    change = abs(err_fine - err_coarse) / max(err_coarse, 1e-12) * 100 if err_coarse > 0 else 0
    
    return {
        "parameters": {"L_m": L, "b_m": b, "h_m": h, "E_Pa": E, "I_m4": I},
        "resolutions": results,
        "convergence": {
            "error_coarse_pct": err_coarse,
            "error_fine_pct": err_fine,
            "error_change_pct": change,
        },
    }


def main():
    print("=" * 60)
    print("DEM-FEM Coupling Benchmark Suite (CPU)")
    print("=" * 60)
    
    device = "cpu"
    
    # ===== Hertz =====
    print("\n=== Hertz Contact (DEM3D) ===")
    radius = 0.02; k_bond = 1e5; tool_radius = 0.1; max_indent = 0.05
    
    r = run_hertz_benchmark(4, 3, 4, radius=radius, k_bond=k_bond,
                            k_contact=1e5, tool_radius=tool_radius,
                            tool_speed=5.0, max_indent=max_indent,
                            sample_every=50, device=device)
    
    d = np.array(r["displacements"]); f = np.array(r["forces"])
    print(f"Points: {len(d)}, Steps: {r['n_steps']}, Time: {r['elapsed_s']:.1f}s")
    
    if len(d) > 5:
        a_fit, b_fit = fit_power_law(d, f)
        print(f"Fit: F = {a_fit:.2e} * δ^{b_fit:.4f}")
    else:
        a_fit, b_fit = 0.0, 0.0
    
    E_eff = r["E_eff"]; R = tool_radius
    delta_max = float(d[-1]) if len(d) > 0 else 0
    F_max = float(f[-1]) if len(f) > 0 else 0
    k_num = F_max / delta_max if delta_max > 0 else 0
    k_hertz = hertz_stiffness(delta_max, R, E_eff) if delta_max > 0 else 0
    stiff_err = abs(k_num - k_hertz) / max(k_hertz, 1e-12) * 100
    
    work = float(np.trapezoid(f, d)) if len(d) > 1 else 0
    energy = (8.0/15.0) * E_eff * math.sqrt(R) * delta_max ** 2.5 if delta_max > 0 else 0
    energy_drift = abs(work - energy) / max(abs(work), abs(energy), 1e-12) * 100
    
    exp_ok = bool(1.45 <= b_fit <= 1.55)
    stiff_ok = bool(stiff_err <= 3.0)
    eng_ok = bool(energy_drift <= 1.0)
    print(f"Exponent: {b_fit:.4f} (PASS={exp_ok})")
    print(f"Stiffness err: {stiff_err:.1f}% (PASS={stiff_ok})")
    print(f"Energy drift: {energy_drift:.1f}% (PASS={eng_ok})")
    
    # Mesh convergence
    conv_hertz = []
    for nz in [4, 6]:
        ny_h = max(3, nz // 2)
        cr = run_hertz_benchmark(nz, ny_h, nz, radius=radius, k_bond=k_bond,
                                 k_contact=1e5, tool_radius=tool_radius,
                                 tool_speed=5.0, max_indent=max_indent,
                                 sample_every=50, device=device)
        cd = np.array(cr["displacements"]); cf = np.array(cr["forces"])
        k = float(cf[-1] / cd[-1]) if len(cd) > 0 and cd[-1] > 0 else 0.0
        conv_hertz.append({"nz": nz, "n_particles": cr["n_particles"],
                          "stiffness": k, "elapsed_s": cr["elapsed_s"]})
    
    stiff_change = float('inf')
    if len(conv_hertz) >= 2 and conv_hertz[0]["stiffness"] > 0:
        stiff_change = abs(conv_hertz[1]["stiffness"] - conv_hertz[0]["stiffness"]) / conv_hertz[0]["stiffness"] * 100
    conv_ok = bool(stiff_change < 5.0)
    print(f"Mesh conv: {stiff_change:.1f}% (PASS={conv_ok})")
    
    # ===== Beam =====
    print("\n=== Beam Deflection (FD) ===")
    beam = run_beam_benchmark()
    for res in beam["resolutions"]:
        print(f"  n={res['n_elements']}: error={res['stiffness_error_pct']:.4f}%")
    
    # ===== Output =====
    hertz_pass = exp_ok and stiff_ok and eng_ok and conv_ok
    
    output = {
        "benchmark": "hertz_contact_sphere_halfspace",
        "solver": "TensorDEM DEM3D (bonded-sphere as half-space analogue)",
        "device": device,
        "honest_fail_note": (
            "TensorFEM is not available in this workspace. "
            "DEM3D bonded-sphere assembly used as elastic half-space analogue. "
            f"Discrete lattice does not reproduce continuum Hertz contact: "
            f"exponent={b_fit:.4f} (target 1.5±0.05), "
            f"stiffness_error={stiff_err:.1f}% (target ≤3%). "
            "Beam deflection benchmark (FD Euler-Bernoulli) provided as fallback."
        ),
        "hertz": {
            "parameters": {
                "dem_particle_radius_m": radius, "bond_stiffness_N_m": k_bond,
                "contact_stiffness_N_m": 1e5, "tool_radius_m": tool_radius,
                "tool_speed_m_s": 5.0, "max_indentation_m": max_indent,
                "E_dem_Pa": r["E_dem"], "E_eff_Pa": r["E_eff"],
                "dt_s": r["dt"], "n_particles": r["n_particles"],
                "n_bonds": r["n_bonds"], "broken_bonds": r["broken_bonds"],
            },
            "fitting": {
                "fitting_exponent": round(b_fit, 6),
                "fitting_coefficient": round(a_fit, 6),
                "exponent_target": 1.5, "exponent_tolerance": 0.05,
                "exponent_pass": exp_ok,
            },
            "stiffness": {
                "numerical_secant_N_m": round(k_num, 4),
                "hertz_tangent_N_m": round(k_hertz, 4),
                "stiffness_error_pct": round(stiff_err, 4),
                "stiffness_pass": stiff_ok,
                "at_displacement_m": round(delta_max, 8),
            },
            "energy": {
                "work_done_J": round(work, 8),
                "stored_energy_J": round(energy, 8),
                "energy_drift_pct": round(energy_drift, 4),
                "energy_pass": eng_ok,
            },
            "mesh_convergence": {
                "resolutions": conv_hertz,
                "stiffness_change_pct": round(stiff_change, 4),
                "convergence_pass": conv_ok,
            },
            "data": {
                "displacements_m": d.tolist(),
                "forces_N": f.tolist(),
                "times_s": r["times"],
            },
            "overall_pass": hertz_pass,
        },
        "beam_fallback": {
            "description": "Simply-supported beam under uniform load (FD Euler-Bernoulli)",
            "analytical": "δ_max = 5*q*L⁴/(384*E*I)",
            "parameters": beam["parameters"],
            "resolutions": beam["resolutions"],
            "convergence": beam["convergence"],
        },
        "overall_pass": hertz_pass,
    }
    
    out = Path(__file__).resolve().parent.parent / "artifacts" / "dem_fem_hertz_cpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nOutput: {out}")
    
    # Curve data
    if len(d) > 2:
        df = np.linspace(d.min(), d.max(), 100)
        curve = {
            "delta_m": df.tolist(),
            "F_fit_N": (a_fit * df ** b_fit).tolist(),
            "F_hertz_N": [hertz_force(x, R, E_eff) for x in df],
            "F_measured_N": f.tolist(),
            "delta_measured_m": d.tolist(),
        }
        cp = Path(__file__).resolve().parent.parent / "artifacts" / "hertz_curve_data.json"
        with open(cp, "w") as fp:
            json.dump(curve, fp, indent=2)
    
    return output


if __name__ == "__main__":
    main()
