"""Acceptance verdicts for the ice-ridge keel freeze-bond face.

Reads ``artifacts/ice_ridge_keel/study.json`` and writes
``artifacts/ice_ridge_keel/acceptance.json``: one record per machine-checkable
criterion of campaign node ``p49w2-arb7927-n08`` with the measured number, the
gate, the verdict and the provenance of the number inside the study file.

The verdicts are computed, never typed in -- a criterion that fails says so in
the file that the report quotes.

Usage::

    PYTHONPATH=src python scripts/ice_ridge_keel_acceptance.py
"""

from __future__ import annotations

import json
import os

OUT = os.path.join("artifacts", "ice_ridge_keel")
STUDY = os.path.join(OUT, "study.json")
ACCEPT = os.path.join(OUT, "acceptance.json")


def _ok(cond) -> str:
    return "PASS" if cond else "FAIL"


def _get(study, part, *path, default=None):
    node = study["parts"].get(part, {})
    if not node.get("ok"):
        return default
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def build(study: dict) -> dict:
    parts = study["parts"]
    rows = []

    # --- criterion 1: F_break = tau_adfreeze * A_neck within 5 % -----------
    bond = parts.get("bond_mapping", {})
    err = bond.get("max_rel_error")
    rows.append(dict(
        id="C1_bond_key_consistency",
        gate="DEM F_break vs tau_adfreeze*A_contact, relative error <= 5 % (10 dT/t sets)",
        value=err,
        unit="-",
        detail={"n_sets": len(bond.get("sets", [])),
                "saline_sets": bond.get("saline_sets"),
                "saline_all_bonded": bond.get("saline_all_bonded")},
        verdict=_ok(err is not None and err <= 0.05),
        provenance="study.json:parts.bond_mapping.max_rel_error",
    ))

    # --- criterion 2: porosity of the freeze-bonded heap ------------------
    por = {f"h{int(h)}": _get(study, f"keel_bonded_h{int(h)}", "porosity_standing")
           for h in (2, 4, 6)}
    vals = [v for v in por.values() if v is not None]
    rows.append(dict(
        id="C2_porosity_band",
        gate="freeze-bonded keel porosity n = 0.30 .. 0.45 (h_k = 2 / 4 / 6 m)",
        value=vals,
        unit="-",
        detail={"per_depth_m": por},
        verdict=_ok(bool(vals) and all(0.30 <= v <= 0.45 for v in vals)),
        provenance="study.json:parts.keel_bonded_h*/porosity_standing",
        note=("h_k = 4 m sits 1 % above the upper edge (n = 0.455); the 2 m and 6 m "
              "cases are inside the band"),
    ))

    # --- criterion 3: DEM vs Kerr-type reference --------------------------
    ratio = {f"h{int(h)}": _get(study, f"keel_bonded_h{int(h)}", "ratio_dem_kerr")
             for h in (2, 4, 6)}
    rvals = [v for v in ratio.values() if v is not None]
    rows.append(dict(
        id="C3_kerr_arbitration",
        gate="F_keel / F_Kerr within 15 % (h_k = 2 / 4 / 6 m)",
        value=rvals,
        unit="-",
        detail={"per_depth_m": ratio,
                "ratio_with_phi_low": {f"h{int(h)}": _get(study, f"keel_bonded_h{int(h)}",
                                                          "ratio_dem_kerr_phi_low")
                                       for h in (2, 4, 6)},
                "phi_bracket_deg": study.get("phi_bracket_deg")},
        verdict=_ok(bool(rvals) and all(abs(v - 1.0) <= 0.15 for v in rvals)),
        provenance="study.json:parts.keel_bonded_h*/ratio_dem_kerr",
        note=("the DEM measures 4.7-6.9x the rigid-plastic Rankine reference; the "
              "reference is a free-field passive-wedge idealisation while the DEM "
              "box is closed at the back and lid-loaded, and the DEM drag also "
              "carries the plowing of a jammed bonded skeleton"),
    ))

    # --- criterion 4: sensitivity to freeze-bond strength -----------------
    sweep = parts.get("dT_sweep", {})
    cases = sweep.get("cases", [])
    fs = [c["f_keel_N"] for c in cases]
    taus = [c["tau_adfreeze_MPa"] for c in cases]
    rho = sweep.get("spearman_rho")
    # the criterion is monotone in the *bond strength*: report the Spearman of
    # tau vs F (identical magnitude to T vs F, positive sign)
    rows.append(dict(
        id="C4_freezebond_sensitivity",
        gate="F_keel monotone increasing with freeze-bond strength, Spearman rho >= 0.9",
        value=(-rho if rho is not None else None),
        unit="-",
        detail={"spearman_T_vs_F": rho,
                "tau_range_MPa": [min(taus), max(taus)] if taus else None,
                "f_keel_range_N": [min(fs), max(fs)] if fs else None,
                "f_keel_rel_span": (max(fs) - min(fs)) / min(fs) if fs else None},
        verdict=_ok(rho is not None and -rho >= 0.9),
        provenance="study.json:parts.dT_sweep.spearman_rho",
        note=("bond strength spans 9.4x (0.14 -> 1.32 MPa) but the drag resistance "
              "moves only 4.4 % (1.99 -> 2.08 MN) and is non-monotone at -20 degC: "
              "the resistance of this jammed, closed-box keel is dominated by the "
              "solid packing, not by the bond strength"),
    ))

    # --- criterion 5: no-bond degeneration --------------------------------
    unb = parts.get("keel_unbonded_h4", {})
    n_unb_standing = unb.get("porosity_standing")
    n_unb_release = unb.get("porosity_after_release")
    neg = parts.get("negative_controls", {})
    rows.append(dict(
        id="C5_no_freezebond_degeneration",
        gate="tau_adfreeze = 0 -> F_break = 0 and keel porosity n >= 0.55",
        value={"f_break_N": neg.get("dT_zero_f_break_N"),
               "n_unbonded_standing": n_unb_standing,
               "n_unbonded_after_release": n_unb_release},
        unit="-",
        detail={"dT_zero_pass": neg.get("dT_zero_pass")},
        verdict=_ok(bool(neg.get("dT_zero_pass"))
                    and n_unb_standing is not None and n_unb_standing >= 0.55),
        provenance="study.json:parts.negative_controls + parts.keel_unbonded_h4",
        note=("the bond half of the criterion holds exactly (F_break = 0); the "
              "porosity half does not: the unbonded heap reaches n = 0.45 (0.48 "
              "after release) because the measurement box is confined by the rock "
              "back face and the lid, so it cannot dilate to the free-field 0.55"),
    ))

    # --- criterion 6: force-chain tail fit --------------------------------
    r2 = {f"h{int(h)}": _get(study, f"keel_bonded_h{int(h)}",
                             "force_chain_standing", "r2") for h in (2, 4, 6)}
    r2v = [v for v in r2.values() if v is not None]
    rows.append(dict(
        id="C6_force_chain_tail",
        gate="P(f) strong-force tail fit R^2 >= 0.90",
        value=r2v,
        unit="-",
        detail={"per_depth_m": r2,
                "alpha": {f"h{int(h)}": _get(study, f"keel_bonded_h{int(h)}",
                                             "force_chain_standing", "alpha")
                          for h in (2, 4, 6)}},
        verdict=_ok(bool(r2v) and all(v >= 0.90 for v in r2v)),
        provenance="study.json:parts.keel_bonded_h*/force_chain_standing.r2",
    ))

    # --- criterion 7: negative controls -----------------------------------
    rows.append(dict(
        id="C7_negative_controls",
        gate="dT = 0 -> F_break = 0; h_k = 0 -> F_keel = 0; n -> 1 -> F_keel -> 0",
        value={"dT_zero_F_break_N": neg.get("dT_zero_f_break_N"),
               "h_k_zero_F_keel_N": neg.get("h_k_zero_f_keel_N"),
               "h_k_zero_F_kerr_N": neg.get("h_k_zero_kerr_N"),
               "n_to_one_F_keel_N": neg.get("n_to_one_f_keel_N")},
        unit="N",
        detail={k: v for k, v in neg.items() if k.endswith("_pass")},
        verdict=_ok(all(neg.get(k) for k in ("dT_zero_pass", "h_k_zero_pass",
                                             "n_to_one_pass"))),
        provenance="study.json:parts.negative_controls",
    ))

    # --- criterion 8: mass conservation -----------------------------------
    drift = max((abs(parts[k]["mass_balance"]["relative_drift"])
                 for k in ("keel_bonded_h2", "keel_bonded_h4", "keel_bonded_h6",
                           "keel_unbonded_h4")
                 if parts.get(k, {}).get("ok")), default=None)
    rows.append(dict(
        id="C8_mass_conservation",
        gate="total rubble mass drift <= 1e-6",
        value=drift,
        unit="-",
        detail={"mass_kg": parts.get("keel_bonded_h4", {}).get("mass_balance")},
        verdict=_ok(drift is not None and drift <= 1e-6),
        provenance="study.json:parts.keel_bonded_h*/mass_balance.relative_drift",
    ))

    # --- adfreeze parity with the platform model (support for C1) ---------
    parity = _get(study, "adapter_parity", "platform_parity", default={})
    rows.append(dict(
        id="S1_platform_adfreeze_parity",
        gate="mirrored tau_adfreeze == tensorphysics.ice.adfreeze over the grid (<=1e-12)",
        value=parity.get("max_rel_error"),
        unit="-",
        detail={"available": parity.get("available"),
                "n_points": parity.get("n_points"),
                "liquidus_Tf_32gkg_mirror_C": parity.get("liquidus_Tf_32gkg_mirror_C"),
                "liquidus_Tf_32gkg_platform_C": parity.get("liquidus_Tf_32gkg_platform_C")},
        verdict=_ok(bool(parity.get("parity_ok"))) if parity.get("available")
        else "WITHHELD",
        provenance="study.json:parts.adapter_parity.platform_parity",
    ))

    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    return {
        "module": "tensordem.ice_ridge_keel_freezebond",
        "adapter": "tensordem.adfreeze_key_adapter",
        "campaign": "p49w2-arb7927-n08",
        "criteria": rows,
        "summary": {
            "n_criteria": len(rows),
            "n_pass": n_pass,
            "n_fail": sum(1 for r in rows if r["verdict"] == "FAIL"),
            "n_withheld": sum(1 for r in rows if r["verdict"] == "WITHHELD"),
            "verdicts": {r["id"]: r["verdict"] for r in rows},
        },
        "artifacts": {
            "study": "artifacts/ice_ridge_keel/study.json",
            "figures": [
                "fig1_adfreeze_closure.png", "fig2_bond_key_parity.png",
                "fig3_keel_drag_vs_depth.png", "fig4_dT_sensitivity.png",
                "fig5_porosity_forcechain.png", "fig6_adapter_parity.png",
            ],
            "tests": "tests/test_ice_ridge_keel_freezebond.py",
            "doc": "docs/ice_ridge_keel_freezebond_face.md",
        },
    }


def main() -> dict:
    with open(STUDY) as fh:
        study = json.load(fh)
    report = build(study)
    with open(ACCEPT, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(f"wrote {ACCEPT}")
    for row in report["criteria"]:
        print(f"  {row['id']:32s} {row['verdict']:9s} {row['value']}")
    print("summary:", report["summary"]["verdicts"])
    return report


if __name__ == "__main__":
    main()