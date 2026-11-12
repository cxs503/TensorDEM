#!/usr/bin/env python3
"""Audit probe: engine tangential state machine vs exact Mindlin--Deresiewicz.

Runs the *implementation* (``tensordem.contact_tangential.mindlin_tangential_step``)
through a steady-state stick-slip cycle and compares it, pointwise, with the
closed-form Mindlin--Deresiewicz branches in ``tensordem.mindlin_partial_slip``
(until now only ever compared with closed-form Cundall--Strack, never with the
engine).

Outputs ``artifacts/case_mindlin_tangential/engine_vs_md.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensordem import hertz_mindlin  # noqa: E402
from tensordem.dem3d import hertz_effective_radius, hertz_reduced_modulus  # noqa: E402
from tensordem.contact_tangential import (  # noqa: E402
    MindlinTangentialConfig,
    mindlin_tangential_step,
)
from tensordem.mindlin_partial_slip import (  # noqa: E402
    gross_slip_displacement,
    partial_slip_magnitude,
    reversal_loop_area,
)

ART = REPO / "artifacts" / "case_mindlin_tangential"

FN = 10.0      # N   quasi-static normal load (constant through the cycle)
MU = 0.3
E = 9.0e9      # Pa  ice
NU = 0.33
R = 0.05       # m   sphere on rigid plane
E_STAR = float(hertz_reduced_modulus(E, NU, rigid_counterpart=True))
R_STAR = float(hertz_effective_radius(R, rigid_counterpart=True))
G_STAR = float(hertz_mindlin.hertz_reduced_shear_modulus(E, NU, rigid_counterpart=True))
A = (3.0 * FN * R_STAR / (4.0 * E_STAR)) ** (1.0 / 3.0)
KT0 = 8.0 * G_STAR * A
DS = float(gross_slip_displacement(FN, MU, KT0))
CAP = MU * FN


def g_branch(s: float) -> float:
    """Mindlin--Deresiewicz steady-state reversal shape, g(0)=0, g(2 ds)=1."""
    x = min(max(s / (2.0 * DS), 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 1.5


def run_cycle(backbone: str, n_steps: int = 400, dt: float = 1.0):
    """Virgin load 0 -> delta_s, reverse by 2 delta_s, reload by 2 delta_s.

    Displacement driven (the contact-point velocity is prescribed).  Engine sign
    convention: ``F = -k (delta - delta_r) + F_r``, so a forward displacement
    gives a negative force.
    """
    cfg = MindlinTangentialConfig(k_t0=KT0, mu_s=MU, backbone="mindlin",
                                  gamma_t=0.0, unloading=backbone)
    F = torch.zeros(1, 1, dtype=torch.float64)
    delta = torch.zeros(1, 1, dtype=torch.float64)
    F_r = torch.zeros(1, 1, dtype=torch.float64)
    d_r = torch.zeros(1, 1, dtype=torch.float64)
    fn = torch.tensor([FN], dtype=torch.float64)

    v_load = DS / n_steps
    v_rev = -2.0 * DS / n_steps
    segments = [("load", v_load, n_steps),
                ("reverse", v_rev, n_steps),
                ("reload", -v_rev, n_steps)]

    trace = []
    for name, v, steps in segments:
        s = 0.0
        for _ in range(steps):
            v_t = torch.tensor([[v]], dtype=torch.float64)
            out, F, delta, F_r, d_r, sliding = mindlin_tangential_step(
                F, delta, F_r, d_r, v_t, fn, cfg, dt)
            s += abs(v) * dt
            trace.append(dict(segment=name, delta=float(delta[0, 0]),
                              force=float(out[0, 0]), excursion=s,
                              sliding=bool(sliding[0])))
    return trace


def exact_force(p) -> float:
    """Exact Mindlin--Deresiewicz force on the same kinematic path."""
    if p["segment"] == "load":
        mag = float(partial_slip_magnitude(torch.tensor(abs(p["delta"])),
                                           FN, MU, KT0))
        return -mag                      # forward displacement -> negative force
    if p["segment"] == "reverse":        # starts at -cap, moves toward +cap
        return -CAP + 2.0 * CAP * g_branch(p["excursion"])
    return CAP - 2.0 * CAP * g_branch(p["excursion"])   # reload: +cap -> -cap


def exact_loop_area() -> float:
    """Exact closed steady-state cycle area ``1.2 (mu Fn)^2 / k_t`` (m16810)."""
    return 1.2 * CAP ** 2 / KT0


def _legacy_open_integral() -> float:
    """Analytic work absorbed per closed cycle (trapezoid on the exact branches).

    Half-cycle = reverse leg (from +cap-bound state to -cap-bound state).  The
    loading/unloading branches differ only through ``g_branch`` vs the virgin
    backbone, so the loop area is the area between them.
    """
    n = 4001
    acc = 0.0
    prev = None
    for i in range(n):
        s = 2.0 * DS * i / (n - 1)
        # displacement excursion from the saturation point, expressed on delta
        f_rev = CAP - 2.0 * CAP * g_branch(s)
        f_load = CAP               # forward-motion branch is saturated everywhere
        if prev is not None:
            acc += 0.5 * ((f_load - f_rev) + prev) * (2.0 * DS / (n - 1))
        prev = f_load - f_rev
    return acc


def loop_area_from_trace(trace) -> float:
    """Area enclosed by the reverse + reload legs (the steady-state cycle).

    The virgin loading leg is excluded: it is not part of the periodic loop.
    """
    last_load = [p for p in trace if p["segment"] == "load"][-1]
    pts = [(last_load["delta"], last_load["force"])]
    pts += [(p["delta"], p["force"]) for p in trace if p["segment"] != "load"]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return -area         # positive = work absorbed by the contact per cycle


def gross_slip_excursion(trace, seg: str) -> float | None:
    for p in trace:
        if p["segment"] == seg and abs(p["force"]) >= 0.999 * CAP:
            return p["excursion"]
    return None


def main() -> dict:
    res = {"delta_s_m": DS, "k_t0_N_per_m": KT0, "cap_N": CAP,
           "a_m": A, "e_star_pa": E_STAR, "g_star_pa": G_STAR}
    for backbone in ("linear", "mindlin"):
        tr = run_cycle(backbone)
        errs = [abs(p["force"] - exact_force(p)) / CAP for p in tr]
        jumps = [abs(b["force"] - a["force"]) for a, b in zip(tr, tr[1:])]
        res[backbone] = {
            "max_abs_err_over_cap": max(errs),
            "mean_abs_err_over_cap": sum(errs) / len(errs),
            "max_step_jump_over_cap": max(jumps) / CAP,
            "loop_area_J": loop_area_from_trace(tr),
            "reversal_gross_slip_excursion_m": gross_slip_excursion(tr, "reverse"),
            "trace": tr,
        }
    res["exact_md"] = {
        "loop_area_J": exact_loop_area(),
        "legacy_reversal_loop_area_J": _legacy_open_integral(),
        "reversal_gross_slip_excursion_m": 2.0 * DS,
        "reversal_loop_area_J": float(reversal_loop_area(FN, MU, KT0)),
    }
    for backbone in ("linear", "mindlin"):
        res[backbone]["loop_area_rel_err_vs_md"] = (
            res[backbone]["loop_area_J"] / res["exact_md"]["loop_area_J"] - 1.0)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "engine_vs_md.json").write_text(json.dumps(res, indent=2))
    slim = {k: v for k, v in res.items() if k not in ("mindlin", "linear")}
    for backbone in ("mindlin", "linear"):
        slim[backbone] = {k: v for k, v in res[backbone].items() if k != "trace"}
    print(json.dumps(slim, indent=2))
    return res


if __name__ == "__main__":
    main()