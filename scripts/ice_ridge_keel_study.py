"""Campaign study: freeze-bonded keel porosity -> structure -> drag resistance.

Runs the acceptance campaign of
:mod:`tensordem.ice_ridge_keel_freezebond` and writes
``artifacts/ice_ridge_keel/study.json`` (numbers, zero hand-copying) plus the
figures used by the report.  Every block is defensive: a part that raises is
recorded as a failure and the rest of the campaign still runs.

Usage::

    PYTHONPATH=src python scripts/ice_ridge_keel_study.py [--quick]
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import traceback

import numpy as np

from tensordem import ice_ridge_keel_freezebond as keel

OUT = os.path.join("artifacts", "ice_ridge_keel")
QUICK = os.environ.get("KEEL_QUICK", "0") == "1"

TEMPS = [-2.0, -5.0, -8.0, -12.0, -16.0, -20.0]
HK = [2.0, 4.0, 6.0]

CFG_BASE = dict(n_particles=520, width_m=6.0, length_m=6.0, porosity=0.34,
                dt=6.0e-4, drag_velocity=0.45, drag_distance_m=0.6,
                release_steps=200, lid_preload=1.0)
CONSOLIDATE = dict(preload=1.0, steps=2500 if not QUICK else 1200, buoyant=False,
                   viscosity=20.0, overlap_guard=0.08, final_relax_steps=400)


def _spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(np.sum(rx * ry) / math.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))


def part_bond_mapping():
    """Criterion 1: F_break = tau_adfreeze * A_contact, ten parameter sets."""
    sets = []
    # (T [degC], contact time [s], Ra [um], salinity [g/kg], r_i, r_j, penetration)
    # Seven freshwater sets + three sea-ice sets (S = 32 g/kg).  The saline sets
    # are what the liquidus transposition used to zero out; they are kept in the
    # campaign so the fix stays audited.
    combos = [
        (-2.0, 3.6e3, 2.5, 0.0, 0.30, 0.30, 2.0e-3),
        (-5.0, 3.6e3, 2.5, 0.0, 0.20, 0.45, 1.0e-3),
        (-10.0, 3.6e3, 2.5, 0.0, 0.25, 0.25, 5.0e-3),
        (-20.0, 3.6e3, 2.5, 0.0, 0.15, 0.50, 3.0e-3),
        (-10.0, 6.0e2, 2.5, 0.0, 0.30, 0.30, 2.0e-3),
        (-10.0, 3.6e3, 20.0, 0.0, 0.30, 0.30, 2.0e-3),
        (-12.0, 3.6e3, 10.0, 0.0, 0.40, 0.25, 1.5e-3),
        (-5.0, 3.6e3, 2.5, 32.0, 0.30, 0.25, 2.0e-3),
        (-10.0, 7.2e3, 15.0, 32.0, 0.40, 0.30, 2.5e-3),
        (-20.0, 3.6e3, 5.0, 32.0, 0.20, 0.20, 1.2e-3),
    ]
    for T, t, ra, S, ri, rj, pen in combos:
        closure = keel.AdfreezeClosure(contact_time_s=t, roughness_um=ra,
                                       salinity_gkg=S)
        res = keel.bonded_pair_pull(ri, rj, pen, closure, T, pull_velocity=1e-3,
                                    dt=2e-4, n_steps=400000)
        rel = abs(res["f_peak_n"] - res["f_break_analytic_n"]) / res["f_break_analytic_n"]
        sets.append({
            "T_C": T, "contact_time_s": t, "roughness_um": ra, "salinity_gkg": S,
            "r_i_m": ri, "r_j_m": rj, "penetration_m": pen,
            "tau_adfreeze_MPa": res["tau_adfreeze_pa"] / 1e6,
            "area_m2": res["area_m2"],
            "f_break_analytic_N": res["f_break_analytic_n"],
            "f_break_dem_N": res["f_peak_n"],
            "rel_error": rel,
        })
    errs = [s["rel_error"] for s in sets]
    saline = [s for s in sets if s["salinity_gkg"] > 0.0]
    return {"sets": sets, "max_rel_error": max(errs),
            "saline_sets": len(saline),
            "saline_all_bonded": bool(all(s["f_break_dem_N"] > 0.0 for s in saline)),
            "pass": bool(max(errs) <= 0.05)}


def part_negative_controls():
    """Criterion 7: the three negative controls."""
    closure = keel.AdfreezeClosure()
    zero_dT = keel.freeze_bond_spec(closure, 0.0, 0.3, 0.3, 2e-3)
    # h_k = 0: no rubble in the keel envelope -> no resistance.  A zero-depth
    # KeelConfig is rejected by construction (an empty domain is not a keel), so
    # the control is realised as the same box with every block pushed out of the
    # drag envelope: the plate sweeps a keel of zero height.
    empty = keel.KeelRubble(keel.KeelConfig(**CFG_BASE))
    # park every block behind the plate's swept path (x > drag distance, well
    # short of the rigid back face): the plate then sweeps a keel of zero height
    empty.pos[:, 0] = 0.55 * empty.lx
    f_empty = empty.drag(distance_m=0.3, velocity=0.45).f_keel_n
    f_empty_analytic = keel.kerr_keel_resistance(
        depth_m=0.0, width_m=6.0, porosity=0.45, friction_angle_deg=25.0,
        cohesion_pa=0.0, surcharge_pa=0.0)["f_kerr_N"]
    # n -> 1: vanishing solid fraction
    cfg2 = keel.KeelConfig(**{**CFG_BASE, "dt": 6e-4})
    ghost = keel.KeelRubble(cfg2, radius_scale=1.0e-3)
    ghost.buoyant = True
    ghost.floor_z = 0.0
    f_ghost = ghost.drag(distance_m=0.4, velocity=0.45).f_keel_n
    ref_ghost = keel.kerr_keel_resistance(depth_m=4.0, width_m=6.0, porosity=0.999,
                                          friction_angle_deg=25.0, cohesion_pa=0.0,
                                          surcharge_pa=0.0)["f_kerr_N"]
    return {
        "dT_zero_f_break_N": zero_dT.f_break_n,
        "dT_zero_pass": bool(zero_dT.f_break_n == 0.0),
        "h_k_zero_f_keel_N": float(f_empty),
        "h_k_zero_kerr_N": float(f_empty_analytic),
        "h_k_zero_pass": bool(f_empty < 1.0e3 and f_empty_analytic == 0.0),
        "n_to_one_f_keel_N": float(f_ghost),
        "n_to_one_kerr_N": float(ref_ghost),
        "n_to_one_pass": bool(f_ghost < 1.0e3 and ref_ghost < 1.0e3),
    }


def _run_case(h_k, temperature, bonds, phi_macro, phi_lo=15.0, cfg_extra=None):
    cfg = keel.KeelConfig(depth_m=h_k, **{**CFG_BASE, **(cfg_extra or {})})
    t0 = time.time()
    run = keel.drag_keel(cfg, ice_temperature_c=temperature, bonds_enabled=bonds,
                         voxel_m=0.14, consolidation_preload=CONSOLIDATE["preload"],
                         consolidation_steps=CONSOLIDATE["steps"],
                         consolidation_viscosity=CONSOLIDATE["viscosity"],
                         overlap_guard=CONSOLIDATE["overlap_guard"],
                         relax_steps=300)
    ms = run.measurements
    standing = ms["standing"]
    n_stand = standing["porosity"]["porosity"]
    cohesion = ms["bonds"]["cohesion_smeared_Pa"]
    kerr = keel.kerr_keel_resistance(
        depth_m=h_k, width_m=cfg.width_m, porosity=n_stand,
        friction_angle_deg=phi_macro, cohesion_pa=cohesion,
        surcharge_pa=cfg.surcharge_pa,
    )
    kerr_lo = keel.kerr_keel_resistance(
        depth_m=h_k, width_m=cfg.width_m, porosity=n_stand,
        friction_angle_deg=phi_lo, cohesion_pa=cohesion,
        surcharge_pa=cfg.surcharge_pa,
    )
    out = {
        "h_k_m": h_k,
        "temperature_C": temperature,
        "bonds_enabled": bool(bonds),
        "f_keel_N": float(run.f_keel_n),
        "f_keel_std_N": float(run.state["f_keel_std_N"]),
        "f_kerr_N": float(kerr["f_kerr_N"]),
        "f_kerr_N_phi_low": float(kerr_lo["f_kerr_N"]),
        "phi_used_deg": float(phi_macro),
        "phi_lo_deg": float(phi_lo),
        "ratio_dem_kerr": float(run.f_keel_n / kerr["f_kerr_N"]) if kerr["f_kerr_N"] > 0 else None,
        "ratio_dem_kerr_phi_low": float(run.f_keel_n / kerr_lo["f_kerr_N"]) if kerr_lo["f_kerr_N"] > 0 else None,
        "porosity_standing": n_stand,
        "porosity_after_drag": ms["after_drag"]["porosity"]["porosity"],
        "porosity_after_release": ms["after_release"]["porosity"]["porosity"],
        "particles_inside_standing": standing["particles_inside_envelope"],
        "bonds_formed": ms["bonds"]["formed_initial"],
        "bonds_active_standing": ms["bonds"]["active_standing"],
        "bonds_active_after_release": ms["after_release"]["bonds"]["active"],
        "cohesion_smeared_Pa": cohesion,
        "force_chain_standing": standing["force_chain"],
        "force_chain_after_drag": ms["after_drag"]["force_chain"],
        "mass_balance": ms["mass_balance"],
        "kerr_components": {k: kerr[k] for k in ("kp", "surcharge_term_N_m",
                                                 "weight_term_N_m", "cohesion_term_N_m")},
        "wall_time_s": time.time() - t0,
    }
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    study = {"module": "tensordem.ice_ridge_keel_freezebond",
             "campaign": "p49w2-arb7927-n08-ice-ridge-keel-freezebond",
             "quick": QUICK, "parts": {}}

    def record(name, fn):
        t0 = time.time()
        try:
            study["parts"][name] = fn()
            study["parts"][name]["ok"] = True
        except Exception as exc:                     # honest FAIL, keep going
            study["parts"][name] = {"ok": False, "error": repr(exc),
                                    "trace": traceback.format_exc()[-1500:]}
        study["parts"][name]["elapsed_s"] = time.time() - t0
        with open(os.path.join(OUT, "study.json"), "w") as fh:
            json.dump(study, fh, indent=1, default=float)
        print(f"[{name}] ok={study['parts'][name]['ok']} "
              f"t={study['parts'][name]['elapsed_s']:.1f}s", flush=True)

    record("bond_mapping", part_bond_mapping)
    record("negative_controls", part_negative_controls)

    phi = 25.0

    def part_macro():
        out = keel.macroscopic_friction_angle(n_particles=300, size_m=3.0,
                                             confining_pa=5.0e3, axial_rate=0.25,
                                             max_strain=0.15, settle_steps=700)
        out.pop("history", None)
        return out

    record("friction_upscaling", part_macro)
    macro = study["parts"]["friction_upscaling"]
    phi_meas = macro.get("phi_macro_deg")
    phi_contact = macro.get("contact_friction_angle_deg", 26.565)
    # The DEM upscaling is a *diagnostic*.  Unless the measured angle passes the
    # plausibility gate the continuum input is bracketed instead: the physical
    # upper bound is the contact friction angle (spheres, no rolling resistance)
    # and 15 deg is the lower bound reported for rounded, poorly graded rubble.
    phi_hi = float(phi_contact if (phi_meas is None or not macro.get("valid")) else phi_meas)
    phi_lo = 15.0
    study["phi_bracket_deg"] = {"low": phi_lo, "high": phi_hi,
                                "measured": phi_meas,
                                "measured_valid": bool(macro.get("valid", False))}

    def part_bonded():
        return _run_case(4.0, -10.0, True, phi_hi, phi_lo)

    record("keel_bonded_h4", part_bonded)

    def part_unbonded():
        return _run_case(4.0, -10.0, False, phi_hi, phi_lo)

    record("keel_unbonded_h4", part_unbonded)

    for h_k in (2.0, 6.0):
        def part(h=h_k):
            return _run_case(h, -10.0, True, phi_hi, phi_lo)
        record(f"keel_bonded_h{int(h_k)}", part)

    def part_dT():
        base = keel.KeelRubble(keel.KeelConfig(depth_m=4.0, **CFG_BASE))
        base.consolidate(preload=CONSOLIDATE["preload"], steps=CONSOLIDATE["steps"],
                         buoyant=False, viscosity=CONSOLIDATE["viscosity"],
                         overlap_guard=CONSOLIDATE["overlap_guard"],
                         final_relax_steps=CONSOLIDATE["final_relax_steps"])
        cases = []
        for T in TEMPS:
            sim = copy.deepcopy(base)
            sim.initialise_bonds()
            sim.float_free_relax(250)
            sim.close_bottom(0.0)
            sim.ice_temperature_c = T
            # rebuild the bond envelope at this temperature from the same contacts
            sim.initialise_bonds()
            standing = sim.measure(voxel_m=0.16)
            run = sim.drag(distance_m=0.5, velocity=0.45)
            cases.append({
                "T_C": T,
                "tau_adfreeze_MPa": sim.closure.tau_mpa(T),
                "f_break_mean_N": float(np.mean(sim.bond_fn_break[sim.bond_fn_break > 0]))
                if (sim.bond_fn_break > 0).any() else 0.0,
                "bonds_formed": int(sim.bond_formed),
                "bonds_active_standing": int(standing["bonds"]["active"]),
                "porosity_standing": standing["porosity"]["porosity"],
                "f_keel_N": float(run.f_keel_n),
                "f_keel_std_N": float(run.state["f_keel_std_N"]),
            })
        rho = _spearman([c["T_C"] for c in cases], [c["f_keel_N"] for c in cases])
        return {"cases": cases, "spearman_rho": rho, "pass": bool(rho >= 0.9)}

    record("dT_sweep", part_dT)

    def part_adapter():
        """The standalone key adapter: platform parity + a key table."""
        from tensordem import adfreeze_key_adapter as adapter

        parity = adapter.platform_parity()
        table = adapter.adfreeze_key_table(
            [adapter.AdfreezeKeyParams(T, salinity_gkg=32.0) for T in TEMPS],
            0.30, 0.25,
        )
        rows = [{k: v for k, v in row.items() if k != "provenance"} for row in table]
        return {"platform_parity": parity, "key_table": rows,
                "liquidus_note": adapter.LIQUIDUS_NOTE}

    record("adapter_parity", part_adapter)

    with open(os.path.join(OUT, "study.json"), "w") as fh:
        json.dump(study, fh, indent=1, default=float)
    print("written", os.path.join(OUT, "study.json"))


if __name__ == "__main__":
    main()