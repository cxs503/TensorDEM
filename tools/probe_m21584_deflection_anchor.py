"""Probe: reproduce the ledger's dt-convergence chain error values.

Read-only diagnostic for GAP-ICE-ONEWAY-CLAIM items (2)(3).  Runs the shipped
DEM-FEM one-way case once, then recomputes the peak-deflection error against the
continuum modal anchor for four time-step refinements and several modal
truncations, and compares with the ledger-quoted chain
(0.908 / 0.102 / 0.041 / 0.043 % at dt/T1 = 0.140 / 0.070 / 0.035 / 0.0175).
No files are written into the repo.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location("dem_fem_run", ROOT / "examples/dem_fem_oneway/run.py")
run = importlib.util.module_from_spec(spec)
sys.modules["dem_fem_run"] = run
spec.loader.exec_module(run)

from tensordem.fem_euler_bernoulli import BeamConfig
from tensordem.dem_fem_gates import modal_response, beam_natural_frequencies

cfg = run.CouplingConfig()
dem = run.run_dem(cfg)

beam_cfg = BeamConfig(
    length=cfg.beam_length, width=cfg.beam_width, height=cfg.beam_height,
    young_modulus=cfg.beam_young_modulus, density=cfg.beam_density,
    n_elements=cfg.beam_n_elements, bc=cfg.beam_bc,
)
ei = cfg.beam_young_modulus * beam_cfg.inertia
rho_a = beam_cfg.mass_per_length
omega1 = float(beam_natural_frequencies(ei, rho_a, cfg.beam_length)[0])
t1 = 2.0 * math.pi / omega1
dt = dem["dt"]

times, force = dem["time_history"], dem["force_history"]
grids = {}
for factor in (1, 2, 4, 8):
    if factor == 1:
        t_f, f_f = list(times), list(force)
    else:
        dt_fine = dt / factor
        t_f = [times[0] + i * dt_fine for i in range(int((times[-1] - times[0]) / dt_fine) + 1)]
        f_f = np.interp(t_f, times, force).tolist()
    grids[factor] = (t_f, f_f)

target = {1: 0.908, 2: 0.102, 4: 0.041, 8: 0.043}
fem_peaks = {}
for factor in (1, 2, 4, 8):
    if factor == 1:
        fem_peaks[factor] = run.run_fem(force, times, cfg)["peak_deflection_m"]
    else:
        ringdown = int(math.ceil(5.0 * t1 / (dt / factor)))
        fem_peaks[factor] = run.run_fem_at_refinement(force, times, cfg, factor,
                                                      ringdown_steps=ringdown)["peak_deflection_m"]

print(f"dt/T1 = {dt/t1:.4f} (level 1);  FEM peaks: "
      + ", ".join(f"dt/{f}={fem_peaks[f]:.6e}" for f in (1, 2, 4, 8)))
print(f"\n{'n_modes':>7} | " + " | ".join(f"dt/{f} err [%] (ledger {target[f]:.3f})" for f in (1, 2, 4, 8)))
for n_modes in (5, 6, 7, 8, 9, 15, 29):
    cells = []
    for factor in (1, 2, 4, 8):
        t_f, f_f = grids[factor]
        modal = modal_response(
            f_f, t_f, ei=ei, mass_per_length=rho_a, length=cfg.beam_length,
            x_load=cfg.beam_length / 2.0, x_probe=cfg.beam_length / 2.0,
            n_modes=n_modes, bc=cfg.beam_bc,
        )
        peak = float(np.max(np.abs(np.asarray(modal["deflection"])[:len(f_f)])))
        err = 100.0 * abs(fem_peaks[factor] - peak) / peak
        cells.append(f"{err:>7.3f}          ")
    print(f"{n_modes:>7} | " + " | ".join(cells))