"""CASE-DEM-PLANE-SHEAR-MUIDI: contact-friction calibration probe.

The provenance adjudication (round 13) showed that *no* corner of the
reference-law envelope rescues the acceptance band: the measured ``mu(I)``
curve (0.33 -> 0.49 over I = 1e-3 .. 0.3) is too flat against every admissible
reading of da Cruz 2005 / GDR MiDi 2004 (which demand a rise to ~0.64-0.84).

This probe tests the single most plausible *physics* lever for that flatness:
the microscopic contact friction ``mu_contact``.  In the Jop ``mu(I)`` law the
high-I limit ``mu_2`` is a *dynamic* friction that the contacts must be able to
transmit; if ``mu_contact`` is too low the macroscopic ``mu`` is capped and the
curve saturates early.  We therefore re-run the two worst-failing points
(I = 0.1, 0.3) at ``mu_contact`` = 0.5 / 0.6 / 0.7 / 0.8 and ask whether the
high-I ``mu`` moves toward the reference within the 10 % band.

This is a *probe*, not a new acceptance: it reports the measured ``mu`` at each
``mu_contact`` and the error against the reference, and leaves the verdict to
the reader.  It is deliberately small (2 I points x 4 frictions) so it can be
run on a single core in a few minutes.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402

# --- reference (da Cruz 2005 / GDR MiDi 2004), the benchmark yardstick ------
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
TOL = 0.10
P = 6.25e4
KN = 6.25e4
R = 0.005
RHO = 2500.0
W, D, H = 0.12, 0.05, 0.06
SHEAR_STRAIN = 0.2
SEED = 5279


def ref_mu(I: float) -> float:
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def run_point(I: float, mu_contact: float) -> dict:
    case = ServoShearCase(inertial_number=I, mu=mu_contact,
                          mu_rolling=0.05, restitution=0.3, kn=KN,
                          normal_stress=P, shear_strain=SHEAR_STRAIN,
                          settle_steps=2500, min_shear_steps=400,
                          packing_gap=0.02, jitter=0.01, seed=SEED,
                          radius=R, density=RHO, box_width=W,
                          box_depth=D, fill_height=H)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case)
    wc = time.perf_counter() - t0
    return {
        "I_target": I,
        "mu_contact": mu_contact,
        "I_measured": res["inertial_number"],
        "mu_measured": res["mu_effective"],
        "mu_bulk": res.get("bulk_mu", float("nan")),
        "ref_mu": ref_mu(I),
        "mu_err_pct": 100.0 * (res["mu_effective"] / ref_mu(I) - 1.0),
        "normal_stress_pa": res["normal_stress_measured_pa"],
        "shear_stress_pa": res["shear_stress_pa"],
        "n_particles": res["n_particles"],
        "wall_clock_s": wc,
    }


def main():
    out = []
    for I in (0.1, 0.3):
        for mu_c in (0.5, 0.6, 0.7, 0.8):
            row = run_point(I, mu_c)
            out.append(row)
            print("I=%.1f mu_c=%.1f -> mu=%.4f ref=%.4f err=%+.1f%% "
                  "(%.0fs)" % (I, mu_c, row["mu_measured"], row["ref_mu"],
                                row["mu_err_pct"], row["wall_clock_s"]))
            sys.stdout.flush()
    path = Path(__file__).resolve().parent / "runs" / "calib_contact_friction.json"
    path.write_text(json.dumps({
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "probe": "contact-friction calibration (mu_contact sweep)",
        "reference": {"mu_s": MU_S, "mu_2": MU_2, "I_0": I_0},
        "rows": out,
    }, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
