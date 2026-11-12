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

Round 2 (m16427) adds the steady-state bed bookkeeping of the device
(``phi_bed_steady`` / ``dilatancy``): round 1 judged ``phi(I)`` on a bed height
that is frozen at the end of the settle phase, hence I-independent by
construction.  ``--arm stiff`` repeats the sweep with a 10x stiffer contact
(P/kn = 0.1 instead of 1) to test whether the low bed solid fraction is a
contact-softness artefact.

Round 3 (m16497) makes the two owed controls runnable:

* ``stiff``  -- stiffness arm; the step budget is now scaled with the contact
  stiffness (``stiffness_step_scale``).  ``dt = 2*pi*sqrt(0.5*m/kn)/20`` shrinks
  as ``1/sqrt(kn)``, so a *fixed step count* spans a 3.16x shorter physical time
  at kn = 10x and the servo phase used to abort with
  ``servo did not reach the target normal stress`` (round-2 note D3).  The
  physical settle duration must be held fixed instead of the step count.
* ``thick``  -- bed-thickness arm (8d wide x 9 layers instead of 5d x 6) to test
  whether the thin bed (m14960 measured a 19 % low packing at 3 layers) is what
  depresses the bed solid fraction.
* ``stiff_thick`` -- the two controls combined (interaction).

No physics, contact law or reference constants are changed by any of the arms;
the arms are pure experiment-design controls on the same device.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"
OUT.mkdir(parents=True, exist_ok=True)

# --- reference (da Cruz 2005 / GDR MiDi 2004), glass beads -----------------
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
# phi(I) constants: NO in-repo literature anchor exists for these two numbers
# (grep of brash_ice_rheology.mu_i / granular_column.scaling returns the mu(I)
# trio only), so the phi curve is a hand-copied reference and is reported as
# PROVENANCE-UNVERIFIED (see analyse.py -> phi_reference_status).
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

# base step budgets of the collisionally resolved protocol (round-1 values)
SETTLE_STEPS_BASE = 2500
MIN_SHEAR_STEPS_BASE = 400

ARMS = {
    "baseline": {"kn": 6.25e4, "label": "P/kn=1, 5d x 6 layers (round-1 configuration)"},
    "stiff": {"kn": 6.25e5, "label": "P/kn=0.1 (10x stiffer contact), 5d x 6 layers"},
    "thick": {"kn": 6.25e4, "label": "P/kn=1, 8d x 9-layer bed",
              "box_depth": 0.08, "fill_height": 0.09},
    "stiff_thick": {"kn": 6.25e5, "label": "P/kn=0.1, 8d x 9-layer bed",
                    "box_depth": 0.08, "fill_height": 0.09},
}


def stiffness_step_scale(kn: float, kn_ref: float = KN) -> float:
    """Steps per unit physical time scale as ``sqrt(kn)``.

    ``ServoShearCase.timestep = 2*pi*sqrt(0.5*m/kn)/20``, so a stiffer contact
    needs ``sqrt(kn/kn_ref)`` more steps to cover the same physical duration.
    With a fixed step count the servo phase of the stiff arm aborted
    (``servo did not reach the target normal stress``): the crate cannot relax
    over the same *step* budget at a shorter dt.
    """
    if not (kn > 0.0 and kn_ref > 0.0):
        raise ValueError("kn and kn_ref must be positive")
    return math.sqrt(kn / kn_ref)


def run_point(I: float, kn: float = KN, box_depth: float = D,
              fill_height: float = H, step_scale: float | None = None) -> dict:
    scale = stiffness_step_scale(kn) if step_scale is None else float(step_scale)
    settle_steps = int(round(SETTLE_STEPS_BASE * scale))
    min_shear_steps = int(round(MIN_SHEAR_STEPS_BASE * scale))
    case = ServoShearCase(inertial_number=I, mu=MU_CONTACT,
                          mu_rolling=MU_ROLLING, restitution=E, kn=kn,
                          normal_stress=P, shear_strain=SHEAR_STRAIN,
                          settle_steps=settle_steps,
                          min_shear_steps=min_shear_steps,
                          packing_gap=0.02, jitter=0.01, seed=SEED,
                          radius=R, density=RHO, box_width=W,
                          box_depth=box_depth, fill_height=fill_height)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case)
    wc = time.perf_counter() - t0
    v_part = 4.0 / 3.0 * math.pi * R ** 3
    phi_bed = res["n_particles"] * v_part / (W * box_depth * res["plate_height"])
    return {
        "I_target": I,
        "I_measured": res["inertial_number"],
        "mu_measured": res["mu_effective"],
        "phi_device": res["solid_fraction"],
        "phi_bed": phi_bed,
        "phi_bed_settled": res["phi_bed_settled"],
        "phi_bed_steady": res["phi_bed_steady"],
        "plate_height_settled": res["plate_height_settled"],
        "plate_height_steady": res["plate_height_steady"],
        "dilatancy": res["dilatancy"],
        "kn": kn,
        "coordination": res["coordination_number"],
        "normal_stress_pa": res["normal_stress_measured_pa"],
        "normal_stress_applied_pa": res["normal_stress_applied_pa"],
        "pressure_control_error": (res["normal_stress_measured_pa"]
                                   / res["normal_stress_applied_pa"] - 1.0),
        "settle_pressure_pa": res["settle_pressure_pa"],
        "shear_stress_pa": res["shear_stress_pa"],
        "n_particles": res["n_particles"],
        "box_depth": box_depth,
        "fill_height": fill_height,
        "layers": fill_height / (2.0 * R),
        "box_depth_d": box_depth / (2.0 * R),
        "settle_compression": 1.0 - res["plate_height_settled"] / fill_height,
        "step_scale": scale,
        "plate_height": res["plate_height"],
        "shear_steps": res["shear_steps"],
        # actual accumulated strain: the requested ``shear_strain`` is a lower
        # bound, because ``min_shear_steps`` overrides it at low I (round 4a)
        "strain_achieved": res["shear_strain_achieved"],
        "settle_steps": res["settle_steps"],
        "dt": res["dt"],
        "max_velocity": res["max_velocity"],
        "wall_clock_s": wc,
        "ref_mu": ref_mu(res["inertial_number"]),
        "ref_phi": ref_phi(res["inertial_number"]),
    }


def payload_for(arm_name: str, rows: list[dict], points) -> dict:
    arm = ARMS[arm_name]
    kn = float(arm["kn"])
    box_depth = float(arm.get("box_depth", D))
    fill_height = float(arm.get("fill_height", H))
    return {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 3,
        "device": "tensordem.mu_i_rheology.simulate_servo_shear",
        "arm": arm_name,
        "arm_label": arm["label"],
        "config": {"P_pa": P, "kn": kn, "mu_contact": MU_CONTACT,
                   "mu_rolling": MU_ROLLING, "e": E, "r": R, "rho": RHO,
                   "box": [W, box_depth, fill_height], "shear_strain": SHEAR_STRAIN,
                   "seed": SEED, "packing_gap": 0.02,
                   "layers": fill_height / (2.0 * R),
                   "box_depth_d": box_depth / (2.0 * R),
                   "step_scale": stiffness_step_scale(kn),
                   "settle_steps_requested": int(round(SETTLE_STEPS_BASE
                                                       * stiffness_step_scale(kn))),
                   "min_shear_steps_requested": int(round(MIN_SHEAR_STEPS_BASE
                                                          * stiffness_step_scale(kn)))},
        "points": list(points),
        "reference": {"mu_s": MU_S, "mu_2": MU_2, "I_0": I_0,
                      "phi_c": PHI_C, "phi_j": PHI_J,
                      "source": "da Cruz et al. 2005 PRE 72:021309; "
                                "GDR MiDi 2004 JFM 509:165"},
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=tuple(ARMS), default="baseline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    arm = ARMS[args.arm]
    kn = float(arm["kn"])
    box_depth = float(arm.get("box_depth", D))
    fill_height = float(arm.get("fill_height", H))
    out_name = args.out or f"sweep_{args.arm}.json"
    rows = []
    for I in I_TARGETS:
        rec = run_point(I, kn=kn, box_depth=box_depth, fill_height=fill_height)
        rows.append(rec)
        print(f"[{args.arm}] I={I:g} -> I={rec['I_measured']:.4g} "
              f"mu={rec['mu_measured']:.4f} "
              f"(ref {rec['ref_mu']:.4f}, {100*(rec['mu_measured']/rec['ref_mu']-1):+.1f}%) "
              f"p/p_tgt={1.0+rec['pressure_control_error']:.3f} "
              f"phi_bed_settled={rec['phi_bed_settled']:.4f} "
              f"phi_bed_steady={rec['phi_bed_steady']:.4f} (ref {rec['ref_phi']:.4f}) "
              f"wc={rec['wall_clock_s']:.0f}s", flush=True)
        # incremental save
        (OUT / f"partial_{args.arm}.json").write_text(json.dumps(rows, indent=1))
    payload = payload_for(args.arm, rows, I_TARGETS)
    (OUT / out_name).write_text(json.dumps(payload, indent=1))
    print("WROTE", OUT / out_name, flush=True)


if __name__ == "__main__":
    main()