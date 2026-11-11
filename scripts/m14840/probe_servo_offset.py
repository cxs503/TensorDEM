"""m14840 probe: is the +3..+5 % normal-stress offset a *controller* artifact?

Two claims of the merged m14721 doc are put to a numerical test:

H1 (offset law)  The drag-servo ``vz += dN/m_p dt; vz *= exp(-dt/tau)`` has a
    discrete steady state ``dN = m_p * (1 - exp(-dt/tau))/dt * vz``.  So the
    measured load offset is *predictable* from the platen creep velocity, i.e.
    it is a control discretisation error, not "physical creep of the bed"
    (doc §4.3).

H2 (shear-rate bookkeeping)  ``ShearCellCase.shear_rate`` uses the *nominal*
    ``fill_height`` (0.05 m) while the compacted bed is only ~0.03 m tall, so
    the reported gamma_dot -- and the inertial number I quoted as "quasi-static
    I <~ 0.3" -- are underestimated by the ratio H_nominal / H_actual.

Run: PYTHONPATH=src python scripts/m14840/probe_servo_offset.py
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace

import numpy as np

from tensordem.shear_cell import ShearCellCase, prepare_bed, simulate_shear_cell

CASE = ShearCellCase(normal_stress=4000.0, diagnostics=True, sample_every=25)


def main() -> int:
    bed = prepare_bed(CASE)
    out = simulate_shear_cell(CASE, bed=bed)

    dt = CASE.timestep
    tau_damp = CASE.servo_damping_time
    n = out["n_particles"]
    m_p = CASE.servo_plate_mass_frac * n * CASE.mass
    gain = (1.0 - math.exp(-dt / tau_damp)) / dt          # dN per (m/s)

    d = out["diagnostics"]
    phase = np.asarray(d["phase"])
    vz = np.asarray(d["vz"], dtype=float)
    sig = np.asarray(d["sigma"], dtype=float)
    shear = phase == "shear"
    half = np.zeros_like(shear)
    half[np.nonzero(shear)[0][len(np.nonzero(shear)[0]) // 2:]] = True

    vz_mean = float(vz[shear & half].mean())
    sig_mean = float(sig[shear & half].mean())
    dN_code = (sig_mean - CASE.normal_stress) * CASE.area       # N, from data
    dN_law = m_p * gain * vz_mean                                # N, predicted

    bed_h = float(np.mean(np.asarray(d["bed_top"], dtype=float)[shear & half]))
    gamma_nominal = CASE.shear_rate
    gamma_actual = CASE.shear_velocity / bed_h
    i_nom = gamma_nominal * 2 * CASE.radius / math.sqrt(CASE.normal_stress / CASE.density)
    i_act = gamma_actual * 2 * CASE.radius / math.sqrt(sig_mean / CASE.density)

    rep = {
        "n_particles": n,
        "plate_mass_kg": m_p,
        "servo_gain_N_per_mps": gain,
        "vz_mean_mps": vz_mean,
        "sigma_measured_pa": sig_mean,
        "sigma_target_pa": CASE.normal_stress,
        "sigma_rel_err": sig_mean / CASE.normal_stress - 1.0,
        "dN_from_data_N": dN_code,
        "dN_drag_law_N": dN_law,
        "dN_law_vs_data_rel": (dN_law - dN_code) / dN_code if dN_code else None,
        "bed_height_mean_m": bed_h,
        "fill_height_nominal_m": CASE.fill_height,
        "gamma_dot_nominal": gamma_nominal,
        "gamma_dot_actual": gamma_actual,
        "inertial_nominal": i_nom,
        "inertial_actual": i_act,
        "mu_macro": out["mu_macro"],
        "valid": out["valid"],
    }
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())