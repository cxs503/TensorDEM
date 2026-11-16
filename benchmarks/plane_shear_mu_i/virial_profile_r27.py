"""Round 27 (m22662) — where does the 24 % pressure mismatch come from?

The observation being explained
-------------------------------
In every archived run of this benchmark the two pressure estimators disagree:

* ``p_wall`` (traction on the pinned wall rows) reproduces the imposed pressure
  (``sigma_yy/P = 0.97 ... 1.08``), while
* ``p_bulk`` (Love-Weber virial over all disk-disk contacts, divided by
  ``W * gap_eff * thickness``) sits at ``0.72 ... 0.82`` of it.

``mu_wall = tau_wall/p_wall`` and ``I_bulk = gammadot d/sqrt(p_bulk/rho)`` are then
built on two *different* pressures, and the spread of the resulting ``mu`` over
the four (tau, p) pairings is 51-180 % - five to eighteen times the ledger's
10 % tolerance.  So the ordinate is ambiguous until this is resolved.

What is computed here (all on one short run of the probe cell)
-------------------------------------------------------------
For a force-balanced assembly the discrete virial identity is

    sum_c l x f_c  =  sum_p x_p x F_p^ext ,

and for this cell the only external forces are the constraint forces that hold
the two pinned rows, ``F_p = -sum_{contacts on p} f``.  Hence

    sigma_yy * V = sum_p y_p F_p^y = (y_top - y_bottom) * F^y

so the volume that makes the *virial* equal the *wall traction* is
``W * (y_top - y_bottom) * thickness`` - the centre-to-centre wall distance, not
``gap_eff = y_top - y_bottom - 2r``.  Three candidate normalisations are
therefore reported side by side, plus the identity residual that decides which
one is right:

  A ``W * gap_eff * th``      what ``measure()`` divides by today
  B ``W * (y_top-y_bottom) * th``   what the identity demands (2r larger)
  C ``W * (gap_eff+4r) * th``  the outer surface of the wall rows

It also prints a layer-resolved ``sigma_yy(y)`` / ``sigma_xy(y)`` profile: if the
deficit were a boundary-layer effect the interior profile would sag; if it is a
normalisation error the profile is flat and only the prefactor is wrong.

Run: ``python benchmarks/plane_shear_mu_i/virial_profile_r27.py`` (one core,
~1-2 minutes; writes ``outputs/r27/virial_profile.json``).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "plane_shear_mu_i"
sys.path.insert(0, str(BENCH))

import plane_shear_2d as ps  # noqa: E402

OUT = REPO / "outputs" / "r27"


def virial_terms(cell):
    """Raw virial sums and the wall tractions, in N and N/m, no normalisation."""
    sysm = cell.system
    case = cell.case
    pos = sysm.pos
    contacts = sysm.contacts
    i = np.asarray(contacts["i"], np.int64)
    j = np.asarray(contacts["j"], np.int64)
    f = np.asarray(contacts["f_on_i"], np.float64)
    branch = pos[j] - pos[i]
    branch[:, 0] -= case.width * np.round(branch[:, 0] / case.width)
    W = np.einsum("ma,mb->ab", branch, f)          # sum_c l x f

    force = sysm._contact_force
    # constraint force holding each pinned disk = -sum of contact forces on it
    F_bot = -force[cell.bottom_idx]
    F_top = -force[cell.top_idx]
    y_bot = float(np.mean(pos[cell.bottom_idx, 1]))
    y_top = float(np.mean(pos[cell.top_idx, 1]))

    # normalisation areas (2-D: thickness is the out-of-plane unit length)
    th = case.thickness
    A_gap = case.width * cell.gap_eff * th
    A_cc = case.width * (y_top - y_bot) * th
    A_outer = case.width * (cell.gap_eff + 4.0 * case.radius) * th
    return {
        "W": W, "f_total_yy": float(np.sum(f[:, 1])),
        "p_wall": 0.5 * abs(float(np.sum(force[cell.top_idx, 1])))
                  / (case.width * th) * 2.0 * 0.5 + 0.0,   # replaced below
        "tau_wall": 0.5 * (abs(float(np.sum(force[cell.top_idx, 0])))
                           + abs(float(np.sum(force[cell.bottom_idx, 0]))))
                    / (case.width * th),
        "p_bot": abs(float(np.sum(force[cell.bottom_idx, 1]))) / (case.width * th),
        "p_top": abs(float(np.sum(force[cell.top_idx, 1]))) / (case.width * th),
        "identity_yy": float(np.sum(pos[cell.bottom_idx, 1] * F_bot[:, 1])
                             + np.sum(pos[cell.top_idx, 1] * F_top[:, 1])),
        "areas": {"gap_eff": A_gap, "centre_to_centre": A_cc, "outer": A_outer},
        "y": {"bottom": y_bot, "top": y_top},
    }


def layer_profile(cell, n_bands: int = 8) -> list[dict]:
    """``sigma_yy`` and ``sigma_xy`` resolved in horizontal bands across the gap."""
    sysm = cell.system
    case = cell.case
    pos = sysm.pos
    contacts = sysm.contacts
    i = np.asarray(contacts["i"], np.int64)
    j = np.asarray(contacts["j"], np.int64)
    f = np.asarray(contacts["f_on_i"], np.float64)
    branch = pos[j] - pos[i]
    branch[:, 0] -= case.width * np.round(branch[:, 0] / case.width)
    y_c = 0.5 * (pos[i, 1] + pos[j, 1])
    lo = float(np.min(pos[cell.bottom_idx, 1]))
    hi = float(np.max(pos[cell.top_idx, 1]))
    edges = np.linspace(lo, hi, n_bands + 1)
    out = []
    band_area = case.width * (hi - lo) / n_bands * case.thickness
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (y_c >= a) & (y_c < b)
        if not np.any(sel):
            continue
        Wb = np.einsum("ma,mb->ab", branch[sel], f[sel])
        out.append({"y": float(0.5 * (a + b)),
                    "sigma_yy_over_P": float(-Wb[1, 1] / band_area / case.normal_stress),
                    "sigma_xy_over_P": float(0.5 * (abs(Wb[0, 1]) + abs(Wb[1, 0]))
                                             / band_area / case.normal_stress),
                    "n_contacts": int(sel.sum())})
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    case = ps.PlaneShear2DCase(I_target=0.05, n_particles=458, width=0.30, gap=0.15,
                               normal_stress=10000.0, settle_time_units=20.0,
                               shear_settle_strain=0.2, strain_measure=0.1,
                               init_mode="random", seed=0, polydispersity=0.0)
    cell = ps.RoughWallShearCell(case)
    P = case.normal_stress

    # compaction (short: the question is a prefactor, not a critical state)
    cell.v_wall = 0.0
    cell._set_wall_kinematics(0.0)
    settle_steps = int(math.ceil(case.settle_time_units
                                 * case.diameter / case.velocity_scale / cell.dt))
    for _ in range(settle_steps):
        cell.advance(200)
    # a little shear so the state is not a freshly compacted pile
    cell.v_wall = cell.case.wall_speed
    cell._set_wall_kinematics(0.0)
    rows, prof = [], []
    for k in range(60):
        cell.advance(200)
        t = virial_terms(cell)
        rows.append({"W_yy": float(t["W"][1, 1]), "identity_yy": t["identity_yy"],
                     "p_bot": t["p_bot"], "p_top": t["p_top"],
                     "tau_wall": t["tau_wall"],
                     "areas": t["areas"], "y": t["y"]})
        if k == 59:
            prof = layer_profile(cell)

    def m(key):
        return float(np.mean([r[key] for r in rows]))

    areas = rows[-1]["areas"]
    Wyy = m("W_yy")
    p_virial = {k: -Wyy / a for k, a in areas.items()}
    p_wall = 0.5 * (m("p_bot") + m("p_top"))
    tau_wall = m("tau_wall")
    result = {
        "case": {"I_target": case.I_target, "P": P, "n": case.n_particles,
                 "W": case.width, "gap": case.gap, "mu_p": case.mu_p},
        "samples": len(rows),
        "p_wall": {"mean": p_wall, "over_P": p_wall / P},
        "p_virial": {k: {"mean": v, "over_P": v / P} for k, v in p_virial.items()},
        "tau_wall": tau_wall,
        "mu_with_wall_pressure": tau_wall / p_wall,
        "mu_with_virial_pressure_gap_eff": tau_wall / p_virial["gap_eff"],
        "identity_check": {
            "virial_W_yy": Wyy,
            "sum_pinned_yF_y": m("identity_yy"),
            "rel_difference": float(abs(Wyy - m("identity_yy"))
                                    / max(1e-30, abs(m("identity_yy")))),
        },
        "predicted_normalisation": {
            "A_cc_over_A_gap_eff": areas["centre_to_centre"] / areas["gap_eff"],
            "explains_observed_deficit": (
                p_virial["centre_to_centre"] / P if p_wall > 0 else float("nan")),
        },
        "layer_profile": prof,
    }
    (OUT / "virial_profile.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: v for k, v in result.items() if k != "layer_profile"},
                     indent=1, default=float))
    print("layer profile (sigma_yy/P):",
          [round(r["sigma_yy_over_P"], 3) for r in prof])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())