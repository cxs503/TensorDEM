"""CASE-DEM-PLANE-SHEAR-MUIDI: plane-shear mu(I) rheology-law calibration.

Reuses the in-repo constant-stress servo shear cell
(``tensordem.mu_i_rheology.simulate_servo_shear``) and sweeps the inertial
number I over [1e-3, 0.3] to confront the measured mu(I) and phi(I) curves
with the glass-bead reference of da Cruz et al. (2005) PRE 72:021309 /
GDR MiDi (2004) JFM 509:165:

    mu(I)  = mu_s + (mu_2-mu_s)/(I_0/I+1),  mu_s=0.36, mu_2=0.84, I_0=0.30
    phi(I) = phi_c + (phi_j-phi_c)/(I_0/I+1), phi_c=0.58, phi_j=0.64

Contact friction is set to the literature quasi-static limit (0.36) so the
high-I rise must emerge from the dynamics (the brash_ice_rheology protocol).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"
OUT.mkdir(parents=True, exist_ok=True)

# --- reference (da Cruz 2005 / GDR MiDi 2004), glass beads -----------------
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
PHI_C, PHI_J = 0.58, 0.64


def ref_mu(I: float) -> float:
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def ref_phi(I: float) -> float:
    return PHI_C + (PHI_J - PHI_C) / (I_0 / I + 1.0)


# --- device configuration (dense regime, P/kn = 1 as in IN102-N02) --------
P = 6.25e4          # Pa, servo-held normal stress
KN = 6.25e4         # N/m
MU_CONTACT = 0.5    # calibrated micro-friction (see report)
MU_ROLLING = 0.05   # calibrated (rolling resistance matters)
E = 0.3
R = 0.005
RHO = 2500.0
W, D, H = 0.12, 0.05, 0.06
I_TARGETS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
SHEAR_STRAIN = 0.2
SEED = 5279


def run_point(I: float) -> dict:
    case = ServoShearCase(inertial_number=I, mu=MU_CONTACT,
                          mu_rolling=MU_ROLLING, restitution=E, kn=KN,
                          normal_stress=P, shear_strain=SHEAR_STRAIN,
                          settle_steps=2500, min_shear_steps=400,
                          packing_gap=0.02, jitter=0.01, seed=SEED,
                          radius=R, density=RHO, box_width=W, box_depth=D,
                          fill_height=H)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case)
    wc = time.perf_counter() - t0
    v_part = 4.0 / 3.0 * math.pi * R ** 3
    phi_bed = res["n_particles"] * v_part / (W * D * res["plate_height"])
    return {
        "I_target": I,
        "I_measured": res["inertial_number"],
        "mu_measured": res["mu_effective"],
        "phi_device": res["solid_fraction"],
        "phi_bed": phi_bed,
        "coordination": res["coordination_number"],
        "normal_stress_pa": res["normal_stress_measured_pa"],
        "shear_stress_pa": res["shear_stress_pa"],
        "n_particles": res["n_particles"],
        "plate_height": res["plate_height"],
        "shear_steps": res["shear_steps"],
        "settle_steps": res["settle_steps"],
        "dt": res["dt"],
        "max_velocity": res["max_velocity"],
        "wall_clock_s": wc,
        "ref_mu": ref_mu(res["inertial_number"]),
        "ref_phi": ref_phi(res["inertial_number"]),
    }


def main() -> None:
    rows = []
    for I in I_TARGETS:
        rec = run_point(I)
        rows.append(rec)
        print(f"I={I:g} -> I={rec['I_measured']:.4g} mu={rec['mu_measured']:.4f} "
              f"(ref {rec['ref_mu']:.4f}, {100*(rec['mu_measured']/rec['ref_mu']-1):+.1f}%) "
              f"phi_bed={rec['phi_bed']:.4f} (ref {rec['ref_phi']:.4f}) "
              f"wc={rec['wall_clock_s']:.0f}s", flush=True)
        # incremental save
        (OUT / "sweep_partial.json").write_text(json.dumps(rows, indent=1))
    payload = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "device": "tensordem.mu_i_rheology.simulate_servo_shear",
        "config": {"P_pa": P, "kn": KN, "mu_contact": MU_CONTACT,
                   "mu_rolling": MU_ROLLING, "e": E, "r": R, "rho": RHO,
                   "box": [W, D, H], "shear_strain": SHEAR_STRAIN,
                   "seed": SEED, "packing_gap": 0.02},
        "reference": {"mu_s": MU_S, "mu_2": MU_2, "I_0": I_0,
                      "phi_c": PHI_C, "phi_j": PHI_J,
                      "source": "da Cruz et al. 2005 PRE 72:021309; "
                               "GDR MiDi 2004 JFM 509:165"},
        "rows": rows,
    }
    (OUT / "sweep.json").write_text(json.dumps(payload, indent=1))
    print("WROTE", OUT / "sweep.json", flush=True)


if __name__ == "__main__":
    main()
