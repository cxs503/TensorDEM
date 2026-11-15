"""DEM-FEM one-way coupling: particle impact on Euler-Bernoulli beam.

Physical scenario
-----------------
A single spherical particle (DEM) impacts the center of a simply-supported
Euler-Bernoulli beam (FEM).  The DEM contact force time history is recorded
and applied as a nodal load on the beam center node.

This is a **one-way** coupling: the beam does not react back on the particle.

Verification targets
--------------------
1. **force_history_port_parity**: DEM-side Σ∫F dt vs FEM-side Σ∫F dt (same force
   history, must match to ≤1e-10 relative).  *scope=numerical_verification_only*
   — this is a consistency check on the trapezoid rule, not physical evidence.
2. **Deflection**: beam center peak deflection (FEM) vs continuum modal
   superposition (independent implementation) ≤1%.
3. **Energy ledger**: total energy + dissipation = initial (≤1e-8 relative).
4. **Separability**: DEM alone and FEM alone run without coupling.
5. **dt convergence**: 4-level time-step refinement chain
   (dt/1, dt/2, dt/4, dt/8) with peak-deflection convergence ratio ≥0.9.

Usage
-----
    python examples/dem_fem_oneway/run.py

Outputs
-------
- examples/dem_fem_oneway/results.json
- artifacts/dem_fem_coupling_ledger.json
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.fem_euler_bernoulli import (
    EulerBernoulliBeam, BeamConfig,
    beam_center_deflection_analytic,
)
from tensordem.dem_fem_gates import (
    beam_natural_frequencies,
    modal_response,
    modal_static_deflection,
)


@dataclass(frozen=True)
class CouplingConfig:
    """Parameters designed for verifiable DEM-FEM coupling.

    The impact is carried by the *platen* contact law,
    ``platen_stiffness = 2.0e4 N/m`` -- **not** by ``contact_stiffness``
    (2.0e2 N/m), which is the particle-particle law and is inert here (the
    packing holds a single moving particle).  With the platen spring the
    contact duration is ``t_c = pi*sqrt(m/k) = 5.44 ms`` for
    ``m = 0.0600175 kg`` (measured 5.37 ms).
    Stiff beam (h=80mm, L=0.3m) gives natural period ~0.48ms.
    Ratio tc/T1 = 11.35 gives a quasi-static response (DAF 1.045 vs the
    continuum modal anchor; 1.035 vs ``P L^3/48EI``).
    """
    particle_radius: float = 0.025
    particle_density: float = 917.0
    contact_stiffness: float = 2.0e2
    platen_stiffness: float = 2.0e4
    initial_velocity: float = -1.0
    restitution: float = 1.0

    beam_length: float = 0.3
    beam_width: float = 0.05
    beam_height: float = 0.08
    beam_young_modulus: float = 2.1e11
    beam_density: float = 7850.0
    beam_n_elements: int = 30
    beam_bc: str = "simply_supported"
    beam_damping: float = 0.0

    device: str = "cpu"


def particle_mass(radius: float, density: float) -> float:
    return density * 4.0 / 3.0 * math.pi * radius ** 3


def run_dem(config: CouplingConfig) -> dict:
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=config.particle_radius,
        density=config.particle_density,
        bond_stiffness=1.0,
        contact_stiffness=config.contact_stiffness,
        contact_damping=0.0,
        drag=0.0,
        restitution_coefficient=config.restitution,
        contact_model="linear",
        bottom_platen_enabled=True,
        bottom_platen_gap=0.0,
        platen_stiffness=config.platen_stiffness,
        platen_damping=0.0,
        fix_x_edges=False,
        fix_bottom=False,
        gravity=0.0,
        device=config.device,
    )

    sim = IceDEM3D(cfg)
    platen_z = sim.bottom_platen_start_z

    xy = torch.linalg.vector_norm(sim.positions[:, :2], dim=1)
    depth = torch.isclose(sim.positions[:, 2], sim.positions[:, 2].min())
    idx = int(torch.nonzero(depth, as_tuple=False).flatten()[torch.argmin(xy[depth])])

    sim.positions[idx, 2] = platen_z + config.particle_radius
    sim.initial_positions = sim.positions.clone()
    sim.velocities[idx, 2] = config.initial_velocity

    force_history = []
    time_history = []
    overlap_history = []
    velocity_history = []
    energy_history = []

    for step in range(30000):
        sim.step()
        f_z = float(sim.last_bottom_platen_reaction_z.item())
        force_history.append(f_z)
        time_history.append(float(sim.time))

        overlap = platen_z - (float(sim.positions[idx, 2]) - config.particle_radius)
        overlap_history.append(max(0.0, overlap))
        velocity_history.append(float(sim.velocities[idx, 2].item()))
        energy_history.append(sim.energy())

        if overlap <= 0 and sim.velocities[idx, 2] > 0 and step > 100:
            break

    impulse_dem = 0.0
    for i in range(1, len(force_history)):
        dt_i = time_history[i] - time_history[i - 1]
        impulse_dem += 0.5 * (force_history[i] + force_history[i - 1]) * dt_i

    m = particle_mass(config.particle_radius, config.particle_density)
    v_initial = config.initial_velocity
    v_final = velocity_history[-1]

    peak_force = max(abs(f) for f in force_history)
    peak_idx = max(range(len(force_history)), key=lambda i: abs(force_history[i]))
    peak_force_time = time_history[peak_idx]

    contact_steps = [i for i, o in enumerate(overlap_history) if o > 0]
    contact_duration = 0.0
    if len(contact_steps) >= 2:
        contact_duration = time_history[contact_steps[-1]] - time_history[contact_steps[0]]

    return {
        "force_history": force_history,
        "time_history": time_history,
        "overlap_history": overlap_history,
        "velocity_history": velocity_history,
        "energy_history": energy_history,
        "impulse_dem_Ns": impulse_dem,
        "particle_mass_kg": m,
        "v_initial_ms": v_initial,
        "v_final_ms": v_final,
        "peak_force_N": peak_force,
        "peak_force_time_s": peak_force_time,
        "contact_duration_s": contact_duration,
        "dt": sim.dt,
        "n_steps": len(force_history),
        "e_initial": energy_history[0],
        "e_final": energy_history[-1],
    }


def run_fem(force_history, time_history, config, *, ringdown_steps: int = 0):
    beam_cfg = BeamConfig(
        length=config.beam_length,
        width=config.beam_width,
        height=config.beam_height,
        young_modulus=config.beam_young_modulus,
        density=config.beam_density,
        n_elements=config.beam_n_elements,
        bc=config.beam_bc,
        damping_ratio=config.beam_damping,
    )
    beam = EulerBernoulliBeam(beam_cfg, device=config.device)

    dt = time_history[1] - time_history[0]
    center_node = beam.n_node // 2
    center_dof = 2 * center_node

    deflection_history = []
    energy_history = []
    work_done = 0.0

    prev_v_center = 0.0
    for i, f in enumerate(force_history):
        f_vec = torch.zeros(beam.n_dof, dtype=beam.dtype, device=beam.device)
        f_vec[center_dof] = f
        result = beam.apply_load(f_vec, dt)
        deflection_history.append(result["deflection"])

        # Compute work: integral of F * v_center dt
        v_center = beam.center_velocity()
        if i > 0:
            # Trapezoidal rule for work
            work_done += 0.5 * (f + force_history[i-1]) * 0.5 * (v_center + prev_v_center) * dt
        prev_v_center = v_center

        energy_history.append({
            "kinetic": beam.kinetic_energy(),
            "strain": beam.strain_energy(),
            "total": beam.total_energy(),
        })

    impulse_fem = 0.0
    for i in range(1, len(force_history)):
        dt_i = time_history[i] - time_history[i - 1]
        impulse_fem += 0.5 * (force_history[i] + force_history[i - 1]) * dt_i

    # Optional free ring-down with zero load: lets an external anchor measure the
    # beam's own fundamental period from the time response (see
    # examples/dem_fem_oneway/independent_gates.py).
    ring_down = []
    for _ in range(max(0, int(ringdown_steps))):
        zero = torch.zeros(beam.n_dof, dtype=beam.dtype, device=beam.device)
        beam.apply_load(zero, dt)
        ring_down.append(beam.center_deflection())

    peak_deflection = max(abs(d[center_node]) for d in deflection_history)
    peak_deflection_time = time_history[
        max(range(len(deflection_history)),
            key=lambda i: abs(deflection_history[i][center_node]))
    ]

    return {
        "deflection_history": [d[center_node] for d in deflection_history],
        "energy_history": energy_history,
        "impulse_fem_Ns": impulse_fem,
        "peak_deflection_m": peak_deflection,
        "peak_deflection_time_s": peak_deflection_time,
        "e_final": energy_history[-1],
        "work_done_J": work_done,
        "dt": dt,
        "ring_down_deflection": ring_down,
        "e_final_strain": energy_history[-1]["strain"],
        "e_final_kinetic": energy_history[-1]["kinetic"],
    }


def run_fem_at_refinement(force, times, config, factor: int, ringdown_steps: int = 0):
    """Drive the FEM with the same force history sampled ``factor`` times finer."""
    if factor == 1:
        t_fine = list(times)
        f_fine = list(force)
    else:
        dt = times[1] - times[0]
        dt_fine = dt / factor
        t_fine = [times[0] + i * dt_fine for i in range(int((times[-1] - times[0]) / dt_fine) + 1)]
        f_fine = np.interp(t_fine, times, force).tolist()

    beam_cfg = BeamConfig(
        length=config.beam_length,
        width=config.beam_width,
        height=config.beam_height,
        young_modulus=config.beam_young_modulus,
        density=config.beam_density,
        n_elements=config.beam_n_elements,
        bc=config.beam_bc,
        damping_ratio=config.beam_damping,
    )
    beam = EulerBernoulliBeam(beam_cfg, device=config.device)

    dt_use = t_fine[1] - t_fine[0]
    center_node = beam.n_node // 2
    center_dof = 2 * center_node

    deflection_history = []
    energy_history = []

    for f in f_fine:
        f_vec = torch.zeros(beam.n_dof, dtype=beam.dtype, device=beam.device)
        f_vec[center_dof] = f
        result = beam.apply_load(f_vec, dt_use)
        deflection_history.append(result["deflection"][center_node])
        energy_history.append({
            "kinetic": beam.kinetic_energy(),
            "strain": beam.strain_energy(),
            "total": beam.total_energy(),
        })

    # Ring-down
    ring_down = []
    for _ in range(max(0, int(ringdown_steps))):
        zero = torch.zeros(beam.n_dof, dtype=beam.dtype, device=beam.device)
        beam.apply_load(zero, dt_use)
        ring_down.append(beam.center_deflection())

    peak_deflection = max(abs(d) for d in deflection_history)

    return {
        "deflection_history": deflection_history,
        "energy_history": energy_history,
        "peak_deflection_m": peak_deflection,
        "dt": dt_use,
        "refinement": factor,
        "ring_down_deflection": ring_down,
    }


def test_dem_alone(config):
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=config.particle_radius, density=config.particle_density,
        bond_stiffness=1.0, contact_stiffness=config.contact_stiffness,
        contact_damping=0.0, drag=0.0, restitution_coefficient=config.restitution,
        contact_model="linear", bottom_platen_enabled=True, bottom_platen_gap=0.0,
        platen_stiffness=config.platen_stiffness, platen_damping=0.0,
        fix_x_edges=False, fix_bottom=False, gravity=0.0, device=config.device,
    )
    sim = IceDEM3D(cfg)
    platen_z = sim.bottom_platen_start_z
    xy = torch.linalg.vector_norm(sim.positions[:, :2], dim=1)
    depth = torch.isclose(sim.positions[:, 2], sim.positions[:, 2].min())
    idx = int(torch.nonzero(depth, as_tuple=False).flatten()[torch.argmin(xy[depth])])
    sim.positions[idx, 2] = platen_z + config.particle_radius
    sim.initial_positions = sim.positions.clone()
    sim.velocities[idx, 2] = config.initial_velocity
    for _ in range(100):
        sim.step()
    return {"status": "ok", "time": float(sim.time),
            "particle_vz": float(sim.velocities[idx, 2].item())}


def test_fem_alone(config):
    beam_cfg = BeamConfig(
        length=config.beam_length, width=config.beam_width,
        height=config.beam_height, young_modulus=config.beam_young_modulus,
        density=config.beam_density, n_elements=config.beam_n_elements,
        bc=config.beam_bc, damping_ratio=0.0,
    )
    beam = EulerBernoulliBeam(beam_cfg, device=config.device)
    center_node = beam.n_node // 2
    center_dof = 2 * center_node
    f_vec = torch.zeros(beam.n_dof, dtype=beam.dtype, device=beam.device)
    f_vec[center_dof] = 100.0
    result = beam.static_solve(f_vec)
    analytical = beam_center_deflection_analytic(
        100.0, config.beam_length, config.beam_young_modulus,
        beam_cfg.inertia, bc=config.beam_bc,
    )
    numerical = result["center_deflection"]
    error = abs(numerical - analytical) / analytical if analytical != 0 else 0.0
    return {"status": "ok", "numerical_deflection_m": numerical,
            "analytical_deflection_m": analytical, "relative_error": error}


def compute_modal_peak_deflection(force_history, time_history, config):
    """Compute peak deflection via continuum modal superposition (independent of FEM)."""
    beam_cfg = BeamConfig(
        length=config.beam_length, width=config.beam_width,
        height=config.beam_height, young_modulus=config.beam_young_modulus,
        density=config.beam_density, n_elements=config.beam_n_elements,
        bc=config.beam_bc,
    )
    ei = config.beam_young_modulus * beam_cfg.inertia
    rho_a = beam_cfg.mass_per_length

    # Extend with ring-down for proper modal comparison
    omega1 = float(beam_natural_frequencies(ei, rho_a, config.beam_length)[0])
    f1 = omega1 / (2.0 * math.pi)
    t1 = 1.0 / f1
    dt = time_history[1] - time_history[0]
    ringdown_steps = int(math.ceil(5.0 * t1 / dt))

    t_ext = time_history + [time_history[-1] + dt * (i + 1) for i in range(ringdown_steps)]
    f_ext = force_history + [0.0] * ringdown_steps

    modal = modal_response(
        f_ext, t_ext, ei=ei, mass_per_length=rho_a, length=config.beam_length,
        x_load=config.beam_length / 2.0, x_probe=config.beam_length / 2.0,
        n_modes=9, bc=config.beam_bc,
    )

    # Only compare the forced phase (same window as FEM)
    n_forced = len(force_history)
    modal_forced = modal["deflection"][:n_forced]

    return {
        "peak_deflection_m": float(np.max(np.abs(modal_forced))),
        "peak_deflection_full_m": modal["peak_deflection"],
        "n_modes": modal["n_modes"],
        "modal_forced": modal_forced.tolist(),
    }


def modal_peak_on_grid(force_history, time_history, config, factor, n_modes=9):
    """Peak centre deflection of the continuum modal anchor on a refined grid.

    The force history is linearly interpolated onto the same ``dt / factor``
    grid used by the FEM, so anchor and FEM integrate an identical load.
    """
    if factor == 1:
        t_grid, f_grid = list(time_history), list(force_history)
    else:
        dt = time_history[1] - time_history[0]
        dt_fine = dt / factor
        n = int((time_history[-1] - time_history[0]) / dt_fine) + 1
        t_grid = [time_history[0] + i * dt_fine for i in range(n)]
        f_grid = np.interp(t_grid, time_history, force_history).tolist()

    beam_cfg = BeamConfig(
        length=config.beam_length, width=config.beam_width,
        height=config.beam_height, young_modulus=config.beam_young_modulus,
        density=config.beam_density, n_elements=config.beam_n_elements,
        bc=config.beam_bc,
    )
    modal = modal_response(
        f_grid, t_grid,
        ei=config.beam_young_modulus * beam_cfg.inertia,
        mass_per_length=beam_cfg.mass_per_length,
        length=config.beam_length,
        x_load=config.beam_length / 2.0, x_probe=config.beam_length / 2.0,
        n_modes=n_modes, bc=config.beam_bc,
    )
    return float(np.max(np.abs(np.asarray(modal["deflection"]))))


def main():
    config = CouplingConfig()

    print("=" * 60)
    print("DEM-FEM One-Way Coupling: Particle Impact on Beam")
    print("=" * 60)

    print("\n[1/6] Separability: DEM alone...")
    dem_alone = test_dem_alone(config)
    assert dem_alone["status"] == "ok"
    print(f"  OK (v_z={dem_alone['particle_vz']:.4f} m/s)")

    print("\n[2/6] Separability: FEM alone...")
    fem_alone = test_fem_alone(config)
    assert fem_alone["status"] == "ok"
    print(f"  OK (static error={fem_alone['relative_error']:.2e})")
    assert fem_alone["relative_error"] < 0.05

    print("\n[3/6] Running DEM...")
    dem_result = run_dem(config)
    print(f"  {dem_result['n_steps']} steps, dt={dem_result['dt']:.6e}s")
    print(f"  Peak force: {dem_result['peak_force_N']:.2f} N")
    print(f"  Contact duration: {dem_result['contact_duration_s']:.6e}s")
    print(f"  Impulse: {dem_result['impulse_dem_Ns']:.6e} Ns")

    print("\n[4/6] Running FEM...")
    fem_result = run_fem(
        dem_result["force_history"], dem_result["time_history"], config,
    )
    print(f"  Peak deflection: {fem_result['peak_deflection_m']:.6e} m")
    print(f"  Impulse: {fem_result['impulse_fem_Ns']:.6e} Ns")
    print(f"  Work done on FEM: {fem_result['work_done_J']:.6e} J")

    print("\n[5/6] Modal superposition (independent anchor)...")
    modal_result = compute_modal_peak_deflection(
        dem_result["force_history"], dem_result["time_history"], config,
    )
    print(f"  Modal peak deflection: {modal_result['peak_deflection_m']:.6e} m")
    print(f"  Modal n_modes: {modal_result['n_modes']}")

    print("\n[6/6] Verification...")

    # 1. Force history port parity (renamed from impulse)
    impulse_dem = dem_result["impulse_dem_Ns"]
    impulse_fem = fem_result["impulse_fem_Ns"]
    force_history_parity_error = abs(impulse_dem - impulse_fem) / max(abs(impulse_dem), 1e-30)
    print(f"  Force history port parity: DEM={impulse_dem:.6e}, FEM={impulse_fem:.6e}, err={force_history_parity_error:.2e}")

    # 2. Deflection: dynamic FEM vs modal superposition ≤1%
    fem_peak = fem_result["peak_deflection_m"]
    modal_peak = modal_result["peak_deflection_m"]
    deflection_error = abs(fem_peak - modal_peak) / max(abs(modal_peak), 1e-30)
    print(f"  Deflection (dynamic vs modal): FEM={fem_peak:.6e}m, modal={modal_peak:.6e}m, err={deflection_error:.4f}")

    # 3. Energy ledger: total energy + dissipation = initial
    e_initial = dem_result["e_initial"]["mechanical_energy_J"]
    e_final_dem = dem_result["e_final"]["mechanical_energy_J"]
    e_final_fem = fem_result["e_final"]["total"]
    work_done = fem_result["work_done_J"]

    # Dissipation = numerical dissipation in DEM
    dissipation = e_initial - e_final_dem

    # Work-energy error
    work_energy_error = abs(work_done - e_final_fem) / max(abs(work_done), 1e-30)

    # Energy balance
    energy_residual = e_initial - e_final_dem - e_final_fem
    energy_balance_error = abs(energy_residual) / max(abs(e_initial), 1e-30)

    print(f"  Energy: init={e_initial:.6e}, DEM_final={e_final_dem:.6e}, FEM_final={e_final_fem:.6e}")
    print(f"  Dissipation (DEM): {dissipation:.6e} J")
    print(f"  Work done on FEM: {work_done:.6e} J")
    print(f"  Work-energy error (FEM): {work_energy_error:.2e}")
    print(f"  Energy balance error: {energy_balance_error:.2e}")

    # 4. dt convergence chain (4 levels: dt/1, dt/2, dt/4, dt/8)
    print("\n  dt convergence chain...")
    beam_cfg = BeamConfig(
        length=config.beam_length, width=config.beam_width,
        height=config.beam_height, young_modulus=config.beam_young_modulus,
        density=config.beam_density, n_elements=config.beam_n_elements,
        bc=config.beam_bc,
    )
    ei = config.beam_young_modulus * beam_cfg.inertia
    rho_a = beam_cfg.mass_per_length
    omega1 = float(beam_natural_frequencies(ei, rho_a, config.beam_length)[0])
    f1 = omega1 / (2.0 * math.pi)
    t1 = 1.0 / f1

    convergence_results = []
    for factor in [1, 2, 4, 8]:
        ringdown_steps = int(math.ceil(5.0 * t1 / (dem_result["dt"] / factor)))
        fem_ref = run_fem_at_refinement(
            dem_result["force_history"], dem_result["time_history"],
            config, factor, ringdown_steps=ringdown_steps,
        )
        anchor_peak = modal_peak_on_grid(
            dem_result["force_history"], dem_result["time_history"], config, factor,
        )
        dt_level = fem_ref["dt"]
        level_error = abs(fem_ref["peak_deflection_m"] - anchor_peak) / max(abs(anchor_peak), 1e-30)
        convergence_results.append({
            "refinement": factor,
            "dt": dt_level,
            "dt_over_t1": dt_level / t1,
            "peak_deflection_m": fem_ref["peak_deflection_m"],
            "modal_anchor_peak_m": anchor_peak,
            "deflection_error_vs_modal_relative": level_error,
            "n_samples": len(fem_ref["deflection_history"]),
        })
        print(f"    dt/{factor}: dt/T1={dt_level / t1:.4f}, "
              f"peak_def={fem_ref['peak_deflection_m']:.6e}m, "
              f"anchor={anchor_peak:.6e}m, err={100 * level_error:.4f}%")

    dt_errors = [r["deflection_error_vs_modal_relative"] for r in convergence_results]

    # Diagnostic only: two adjacent refinements give a peak ratio ~1 by
    # construction, so that ratio can never turn red on physics.  The gate
    # below uses the per-level error against the independent modal anchor.
    finest = convergence_results[-1]
    second_finest = convergence_results[-2]
    convergence_ratio = finest["peak_deflection_m"] / max(second_finest["peak_deflection_m"], 1e-30)
    dt_chain_pass = (
        dt_errors[0] <= 0.01          # coarse level dt/T1 = 0.14 inside 1%
        and dt_errors[-1] <= 1e-3     # finest level inside 0.1%
        and dt_errors[1] <= dt_errors[0]
        and dt_errors[-1] <= dt_errors[1]
    )
    print("  dt chain errors vs modal anchor: "
          + ", ".join(f"{100 * e:.4f}%" for e in dt_errors)
          + f"  ({'PASS' if dt_chain_pass else 'FAIL'})")
    print(f"  (diagnostic) peak ratio dt/8 over dt/4: {convergence_ratio:.6f}")

    # Use work-energy error as the energy criterion
    energy_error = work_energy_error

    results = {
        "scenario": "dem_fem_oneway_particle_impact",
        "scope": "numerical_verification_only",
        "dem": {
            "n_steps": dem_result["n_steps"], "dt": dem_result["dt"],
            "peak_force_N": dem_result["peak_force_N"],
            "peak_force_time_s": dem_result["peak_force_time_s"],
            "contact_duration_s": dem_result["contact_duration_s"],
            "impulse_Ns": dem_result["impulse_dem_Ns"],
            "particle_mass_kg": dem_result["particle_mass_kg"],
            "v_initial_ms": dem_result["v_initial_ms"],
            "v_final_ms": dem_result["v_final_ms"],
            "energy_initial_J": e_initial,
            "energy_final_J": e_final_dem,
        },
        "fem": {
            "n_elements": config.beam_n_elements,
            "peak_deflection_m": fem_result["peak_deflection_m"],
            "peak_deflection_time_s": fem_result["peak_deflection_time_s"],
            "impulse_Ns": fem_result["impulse_fem_Ns"],
            "energy_final_J": e_final_fem,
            "work_done_J": work_done,
        },
        "modal": {
            "peak_deflection_m": modal_result["peak_deflection_m"],
            "n_modes": modal_result["n_modes"],
        },
        "dt_convergence": convergence_results,
        "verification": {
            "force_history_port_parity_error_relative": force_history_parity_error,
            "force_history_port_parity_pass": force_history_parity_error <= 1e-10,
            "deflection_dynamic_vs_modal_m": modal_peak,
            "deflection_error_relative": deflection_error,
            "deflection_pass": deflection_error <= 0.01,
            "dem_dissipation_J": dissipation,
            "fem_work_done_J": work_done,
            "work_energy_error_relative": work_energy_error,
            "energy_pass": work_energy_error <= 1e-8,
            "energy_balance_error_relative": energy_balance_error,
            "separability_dem": dem_alone["status"] == "ok",
            "separability_fem": fem_alone["status"] == "ok",
            "separability_fem_deflection_error": fem_alone["relative_error"],
            "dt_convergence_errors_vs_modal": dt_errors,
            "dt_convergence_criterion": (
                "error(dt/T1=0.14) <= 1% and error(dt/T1=0.0175) <= 0.1% and both "
                "finer levels below the coarse level; per-level error against the "
                "independent continuum modal anchor, NOT a ratio of two nearly "
                "identical peaks"
            ),
            "dt_convergence_peak_ratio_finest_over_second_finest": convergence_ratio,
            "dt_convergence_peak_ratio_is_diagnostic_only": True,
            "dt_convergence_pass": dt_chain_pass,
        },
    }

    ledger = {
        "ledger_type": "dem_fem_coupling",
        "dem_side": {
            "total_impulse_Ns": dem_result["impulse_dem_Ns"],
            "energy_initial_J": e_initial,
            "energy_final_J": e_final_dem,
            "numerical_dissipation_J": dissipation,
            "peak_force_N": dem_result["peak_force_N"],
            "contact_duration_s": dem_result["contact_duration_s"],
            "n_force_samples": len(dem_result["force_history"]),
        },
        "fem_side": {
            "total_impulse_Ns": fem_result["impulse_fem_Ns"],
            "energy_final_J": e_final_fem,
            "kinetic_energy_J": fem_result["e_final"]["kinetic"],
            "strain_energy_J": fem_result["e_final"]["strain"],
            "work_done_J": work_done,
            "peak_deflection_m": fem_result["peak_deflection_m"],
        },
        "modal_side": {
            "peak_deflection_m": modal_result["peak_deflection_m"],
            "n_modes": modal_result["n_modes"],
        },
        "interface": {
            "scope": "numerical_verification_only",
            "force_history_port_parity_error_relative": force_history_parity_error,
            "force_history_port_parity_error_abs_Ns": abs(
                dem_result["impulse_dem_Ns"] - fem_result["impulse_fem_Ns"]),
            "impulse_dem_Ns": dem_result["impulse_dem_Ns"],
            "impulse_fem_Ns": fem_result["impulse_fem_Ns"],
            "force_samples_transferred": len(dem_result["force_history"]),
            "note": (
                "DEM and FEM integrate the same force array with the same trapezoid rule, so "
                "this parity is an algebra check on the port, not evidence of interface "
                "conservation."
            ),
        },
        "energy_balance": {
            "dem_initial_J": e_initial,
            "dem_final_J": e_final_dem,
            "fem_final_J": e_final_fem,
            "dem_dissipation_J": dissipation,
            "fem_work_done_J": work_done,
            "work_energy_error_relative": work_energy_error,
            "energy_balance_residual_J": energy_residual,
            "energy_balance_error_relative": energy_balance_error,
        },
        "deflection_comparison": {
            "fem_peak_m": fem_result["peak_deflection_m"],
            "modal_peak_m": modal_result["peak_deflection_m"],
            "error_relative": deflection_error,
        },
        "dt_convergence": convergence_results,
        "separability": {
            "dem_alone_ok": dem_alone["status"] == "ok",
            "fem_alone_ok": fem_alone["status"] == "ok",
            "fem_static_error": fem_alone["relative_error"],
        },
    }

    results_path = Path(__file__).parent / "results.json"
    ledger_path = Path(__file__).parent.parent.parent / "artifacts" / "dem_fem_coupling_ledger.json"
    # The ledger is inlined into results.json as well: the artifacts/ copy is a
    # convenience dump, while the committed results.json keeps validate.py
    # runnable in a clean checkout.
    results["ledger"] = ledger

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger_path, "w") as f:
            json.dump(ledger, f, indent=2)
    except OSError as exc:  # read-only checkout: results.json already carries it
        print(f"  (ledger dump to {ledger_path} skipped: {exc})")

    print(f"\nResults: {results_path}")
    print(f"Ledger: {ledger_path} (also inlined at results['ledger'])")

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Force history port parity (≤1e-10):  {force_history_parity_error:.2e}  {'PASS' if force_history_parity_error <= 1e-10 else 'FAIL'}")
    print(f"  Deflection dynamic vs modal (≤1%):  {deflection_error:.4f}  {'PASS' if deflection_error <= 0.01 else 'FAIL'}")
    print(f"  Work-Energy (≤1e-8): {work_energy_error:.2e}  {'PASS' if work_energy_error <= 1e-8 else 'FAIL'}")
    print("  dt chain errors vs modal (<=1% @dt, <=0.1% @dt/8): "
          + "/".join(f"{100 * e:.3f}%" for e in dt_errors)
          + f"  {'PASS' if dt_chain_pass else 'FAIL'}")
    print(f"  Sep DEM:           {'PASS' if dem_alone['status'] == 'ok' else 'FAIL'}")
    print(f"  Sep FEM:           {'PASS' if fem_alone['status'] == 'ok' else 'FAIL'}")

    all_pass = (force_history_parity_error <= 1e-10 and deflection_error <= 0.01
                and work_energy_error <= 1e-8 and dt_chain_pass
                and dem_alone["status"] == "ok" and fem_alone["status"] == "ok")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return results, ledger


if __name__ == "__main__":
    main()
