#!/usr/bin/env python
"""Independent probe of CASE-DEM-ROLLING-SINGLE-PARTICLE (audit m16199).

Not part of the mission. Two cross-checks that do NOT reuse the case's own
estimator (LS slope of v(t)):

  P1  displacement estimator: fit x(t) = x0 + v0 t + a t^2 / 2 over the steady
      window (an independent functional form from v-slope) and compare a with
      the closed-form a = g sin(theta) / (1 + I/(mR^2)).
  P2  angle scaling: at theta = 10/15/25 deg, a / (g sin theta) must equal
      1/(1 + I/(mR^2)) = 5/7, i.e. the inertia factor is measured, not assumed.
  P3  rolling-resistance branch: mu_r = 0.1 changes a by
      -mu_r cos(theta)/(sin(theta)) relative to mu_r = 0 (CDT torque really
      enters the equations, so the match is not a closed-form tautology).
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from tensordem.rolling_resistance import (  # noqa: E402
    RollingResistanceConfig,
    SphereInclineConfig,
    simulate_sphere_on_incline,
)


def quad_slope(t: torch.Tensor, x: torch.Tensor) -> float:
    """LS a from x = c0 + c1 t + c2 t^2 -> a = 2 c2 (independent of v-slope fit)."""
    A = torch.stack([torch.ones_like(t), t, t * t], dim=1)
    coef = torch.linalg.lstsq(A, x).solution
    return float(2.0 * coef[2])


def run(theta_deg: float, mu_r: float = 0.0, dt: float = 1e-5, duration: float = 0.16, settle: float = 0.06):
    cfg = SphereInclineConfig(k_n=2e5, dt=dt, mu=0.9)
    res = simulate_sphere_on_incline([theta_deg], RollingResistanceConfig(model="cdt", mu_r=mu_r),
                                     config=cfg, duration=duration, settle_time=settle)
    m = res["steady_mask"]
    a_quad = quad_slope(res["time"][m], res["x"][m, 0])
    t = res["time"][m]
    v = res["v"][m, 0]
    a_ls = float(((t - t.mean()) * (v - v.mean())).sum() / ((t - t.mean()) ** 2).sum())
    th = math.radians(theta_deg)
    a_ref = cfg.gravity * (math.sin(th) - mu_r * math.cos(th)) / (1.0 + cfg.inertia_factor)
    return {
        "theta_deg": theta_deg, "mu_r": mu_r, "I_over_mR2": cfg.inertia_factor,
        "a_quad": a_quad, "a_vslope": a_ls, "a_ref": a_ref,
        "err_quad": abs(a_quad - a_ref) / abs(a_ref),
        "quad_vs_vslope": abs(a_quad - a_ls) / abs(a_ref),
        "a_over_g_sin": a_ls / (cfg.gravity * math.sin(th)),
    }


out = {"P1_P2_angle_scaling": [], "P3_mu_r_branch": None}
for th in (10.0, 15.0, 25.0):
    out["P1_P2_angle_scaling"].append(run(th))
base = out["P1_P2_angle_scaling"][1]["a_vslope"]
roll = run(15.0, mu_r=0.1)
out["P3_mu_r_branch"] = {
    "a_mu_r0": base, "a_mu_r0.1": roll["a_vslope"], "a_ref_mu_r0.1": roll["a_ref"],
    "rel_err": roll["err_quad"] if False else abs(roll["a_vslope"] - roll["a_ref"]) / roll["a_ref"],
    "measured_shift_vs_mu_r0": (roll["a_vslope"] - base) / base,
    "predicted_shift": -0.1 * math.cos(math.radians(15.0)) / math.sin(math.radians(15.0)),
}
out["inertia_factor_target"] = 0.4
out["all_angle_scaling_within_1pct"] = all(
    abs(p["a_over_g_sin"] - 1.0 / 1.4) / (1.0 / 1.4) < 0.01 for p in out["P1_P2_angle_scaling"])
out["all_quad_within_1pct"] = all(p["err_quad"] < 0.01 for p in out["P1_P2_angle_scaling"])
print(json.dumps(out, indent=1))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_independent.json"), "w") as fh:
    json.dump(out, fh, indent=1)