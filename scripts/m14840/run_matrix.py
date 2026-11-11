"""m14840 evidence driver: servo-mode x normal-stress matrix + stationarity run.

Writes JSON to artifacts/shear_cell_m14840/<tag>.json.  One bed preparation is
shared by all stress levels of a mode (the same protocol as
``measure_macro_friction``), so the runs stay comparable.

Run: PYTHONPATH=src python scripts/m14840/run_matrix.py <tag> <mode> <strain>
     tag    free label (also the output file name)
     mode   drag|bedref (servo)
     strain shear_strain_target
     stresses  comma list, default 2000,4000,8000
     consol    consolidation time (s), default 0.04 = the m14721 protocol
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from tensordem.shear_cell import ShearCellCase, prepare_bed, simulate_shear_cell

OUT = Path("artifacts/shear_cell_m14840")


def main() -> int:
    tag, mode, strain = sys.argv[1], sys.argv[2], float(sys.argv[3])
    stresses = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [2000.0, 4000.0, 8000.0]
    OUT.mkdir(parents=True, exist_ok=True)
    consol = float(sys.argv[5]) if len(sys.argv) > 5 else 0.04
    base = ShearCellCase(diagnostics=True, sample_every=25, servo_mode=mode,
                         shear_strain_target=strain, consol_time=consol)
    bed = prepare_bed(base)
    rows = []
    for sig in stresses:
        case = replace(base, normal_stress=sig)
        out = simulate_shear_cell(case, bed=bed)
        d = out["diagnostics"]
        phase = np.asarray(d["phase"])
        shear = phase == "shear"
        idx = np.nonzero(shear)[0]
        half = np.zeros_like(shear)
        half[idx[len(idx) // 2:]] = True
        vz = np.asarray(d["vz"], dtype=float)
        m_p = case.servo_plate_mass_frac * out["n_particles"] * case.mass
        gain = (1.0 - math.exp(-case.timestep / case.servo_damping_time)) / case.timestep
        tau_hist = np.asarray(out["shear_stress_history"], dtype=float)
        rows.append({
            "sigma_target": sig,
            "sigma": out["normal_stress_measured"],
            "sigma_rel": out["normal_stress_rel_err"],
            "sigma_sem": out["normal_stress_sem"],
            "tau": out["shear_stress"],
            "tau_sem": out["shear_stress_sem"],
            "mu": out["mu_macro"],
            "mu_sem": out["shear_stress_sem"] / out["normal_stress_measured"],
            "tau_floor": out["floor_shear_stress"],
            "tau_platen_vs_floor_rel": out["tau_platen_vs_floor_rel"],
            "r2": out["r_squared"],
            "drift": out["shear_stress_drift_rel"],
            "valid": out["valid"],
            "rz_no": out["invalid_reason"],
            "n_contact": float(np.mean(np.asarray(d["n_contact"], dtype=float)[shear & half])),
            "n_above_max": int(np.max(np.asarray(d["n_above"], dtype=float)[shear])),
            "vz_platen": float(vz[shear & half].mean()),
            "vmax_bed": float(np.max(np.asarray(d["vmax"], dtype=float)[shear])),
            "dN_data_N": (out["normal_stress_measured"] - sig) * case.area,
            "dN_drag_law_N": m_p * gain * float(vz[shear & half].mean()),
            "bed_height": out["bed_height_mean"],
            "shear_rate_actual": out["shear_rate_actual"],
            "inertial_number": out["inertial_number"],
            "tau_window_quarters": [float(np.mean(q)) for q in np.array_split(tau_hist, 4)],
            "shear_strain": case.shear_strain_target,
            "shear_rate_nominal": case.shear_rate,
            "timestep": case.timestep,
            "n_particles": out["n_particles"],
            "bed_prep": {"steps": bed["steps"], "packing_fraction": bed["packing_fraction"],
                         "bed_top": bed["bed_top"]},
        })
        print(tag, sig, "rel=%.4f" % out["normal_stress_rel_err"],
              "mu=%.4f" % out["mu_macro"], "drift=%.3f" % out["shear_stress_drift_rel"], flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps(
        {"mode": mode, "strain": strain, "consol_time": consol, "tag": tag,
         "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())