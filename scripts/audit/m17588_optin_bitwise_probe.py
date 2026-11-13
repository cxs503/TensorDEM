#!/usr/bin/env python3
"""m17588 audit probe — does the opt-in contact-proxy diagnostic leave the
integrator bit-identical?  The m17500 test suite only asserts the dataclass
default flag (`test_device_diagnostic_is_opt_in`), never the trajectory.

Runs one short constant-stress servo shear case twice (diagnostic OFF / ON) and
compares every integrator-level quantity that the case returns, bit-for-bit.

Run against the m17500 deliverable worktree:
  PYTHONPATH=/root/tp-agent/work/m17500/src python3 this_file.py <n_steps>
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200

BASE = dict(radius=0.005, density=2500.0, mu=0.5, restitution=0.3, kn=1.0e3,
            normal_stress=2000.0, seed=5279, settle_steps=60,
            shear_rate=20.0, shear_strain=0.6, min_shear_steps=STEPS)

# quantities that must not move when the diagnostic is switched on
CMP = ("shear_rate_measured", "mu_wall_measured", "normal_stress_measured_pa",
       "coordination", "phi", "dt", "pressure_control_error_final",
       "bulk_pressure_balance_error")


def run(**kw):
    case = ServoShearCase(**BASE, **kw)
    r = simulate_servo_shear(case, return_series=False)
    return r


def main():
    off = run(contact_proxy=False)
    on = run(contact_proxy=True)
    diffs = {}
    for k in CMP:
        a, b = off.get(k), on.get(k)
        if a is None or b is None:
            diffs[k] = {"present": False, "off": a, "on": b}
            continue
        af, bf = float(a), float(b)
        if math.isnan(af) and math.isnan(bf):
            diffs[k] = {"both_nan": True, "bit_equal": True}
            continue
        # bit-for-bit: compare raw float64 patterns, not a tolerance
        bit = np.float64(af).tobytes() == np.float64(bf).tobytes()
        diffs[k] = {"off": af, "on": bf, "bit_equal": bit,
                    "rel": (bf - af) / af if af else None}
    diag_present = "contact_proxy" in on
    traj_bit = all(v.get("bit_equal", False) for v in diffs.values())
    out = {"n_steps": STEPS, "diagnostic_key_present_when_on": diag_present,
           "diagnostic_key_present_when_off": "contact_proxy" in off,
           "integrator_bit_identical": traj_bit, "per_quantity": diffs,
           "proxy_when_on": on.get("contact_proxy")}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()