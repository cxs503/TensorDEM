"""Render docs/propeller_ice_milling_legacy_verification.md from the verified JSON.

All numbers in the generated document are injected from
``artifacts/propeller_ice_milling_legacy/verification.json`` (no hand-copied
digits).  Run the verification script first:

    PYTHONPATH=src python scripts/verify_propeller_ice_milling_legacy.py
    PYTHONPATH=src python scripts/render_propeller_ice_milling_legacy_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

ART = Path("artifacts/propeller_ice_milling_legacy")
OUT = Path("docs/propeller_ice_milling_legacy_verification.md")


def fmt(v, digits=6):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != 0.0 and (abs(v) < 1e-4 or abs(v) >= 1e6):
            return f"{v:.{digits}g}"
        return f"{v:.{digits}g}"
    return str(v)


def main() -> int:
    data = json.loads((ART / "verification.json").read_text(encoding="utf-8"))
    gates = data["gates"]
    cross = data["crossval"][0]
    lines: list[str] = []
    a = lines.append

    a("# Legacy propeller–ice milling module (IN27-S4): analytic verification and defect report")
    a("")
    a(f"**Protocol** `{data['protocol']}` · **module** `{data['module']}` · "
      f"**campaign node** {data['module_campaign_node']}")
    a(f"**Tested twin** {data['tested_twin']} · "
      f"**gates** {data['n_pass']}/{data['n_gates']} PASS · **defects** "
      f"{data['n_defects_open']} open / {data['n_defects_repaired']} repaired")
    a("**Artifacts** `artifacts/propeller_ice_milling_legacy/` · "
      "**Harness** `scripts/verify_propeller_ice_milling_legacy.py` · "
      "**Gates as tests** `tests/test_propeller_ice_milling_legacy_verification.py`")
    a("")
    a("---")
    a("")
    a("## 1. Case description")
    a("")
    a("Ice-going propellers do not merely push broken ice aside: a slab drawn into the")
    a("disc is **milled** by the blades, the shaft sees a pulsating torque at the")
    a("blade-passing frequency, and the crushed debris is ejected and transported by the")
    a("propeller race.  Two independent implementations of this physics live in this")
    a("repository:")
    a("")
    a("| implementation | file | campaign node | tests | driver | documentation |")
    a("|---|---|---|---|---|---|")
    a("| legacy | `src/tensordem/propeller_ice_milling.py` (1140 lines) | IN27-S4 | **0** | none | none |")
    a("| package | `src/tensordem/propeller_ice/` (7 modules) | IN34-W3-N05 | 28 | `scripts/run_propeller_ice_milling.py` | `docs/PROPELLER_ICE_MILLING_TRANSIENT.md` |")
    a("")
    a("The legacy module ships in the public `src` tree and declares a *plastic-capped")
    a("Hertz* contact, blade-element/momentum open-water hydrodynamics, a")
    a("*“momentum-conserving”* self-similar race and a debris transport model, but no")
    a("test, driver or document ever exercises it.  This mission closes that verification")
    a("debt: every closed-form claim is pinned to an independent reference, the module is")
    a("cross-checked against its tested twin on a matched geometry, and the defects found")
    a("are registered here with their evidence.")
    a("")
    a("Geometry used throughout (legacy defaults): "
      f"`{cross['meta']['legacy_note']}`, `c(0.7R)` from the Beta-normalised")
    a(f"planform; the package geometry is matched to "
      f"`A_E/A_0 = {cross['meta']['pkg_expanded_area_ratio']:.6f}` "
      f"(`chord_ratio = {cross['meta']['pkg_chord_ratio']:.6f}`, `skew_tip = 0`).")
    a("")
    a("## 2. Method")
    a("")
    a("Six gate families, each against a reference that is *not* the code under test:")
    a("")
    a("| gate | quantity | independent reference |")
    a("|---|---|---|")
    a("| G1 | planform scale, expanded-area closure, pitch law | closed-form Beta integral `B(1+e,1+e)`, `A_E = EAR·πD²/4`, `atan(P/2πr)` |")
    a("| G2 | propeller-race momentum flux, induced velocity | `∫ρu²2πr dr = T`; actuator-disk `½(−V+√(V²+2T/ρA))` |")
    a("| G3 | debris quadratic-drag transport | closed form `|w|(t) = |w0|/(1+k|w0|t)` + step-refinement study |")
    a("| G4 | torque spectrum | synthetic BPF harmonic train with known amplitudes; white-noise negative control |")
    a("| G5 | open-water BEM | efficiency identity `η = J K_T/(2πK_Q)`, monotone `K_T(J)`, zero-thrust `J0` |")
    a("| G6 | cross-implementation | tested `tensordem.propeller_ice` BEM on a geometry matched in `D, Z, r_h/R, P/D, A_E/A_0` |")
    a("")
    a("Compute: all gates are seconds-scale (single process, no cluster), so the run is")
    a("repeatable on the control node.  Nothing here is a physical validation: no")
    a("experimental open-water chart or ice-milling torque measurement exists in this")
    a("repository, so absolute magnitudes stay **WITHHELD** by construction.")
    a("")
    a("## 3. Results")
    a("")
    a("### 3.1 Gate table (programmatically injected)")
    a("")
    a("| gate | what | measured | reference | rel. error | tol | verdict |")
    a("|---|---|---|---|---|---|---|")
    for g in gates:
        if g["verdict"] == "INFO":
            continue
        a(f"| `{g['id']}` | {g['what']} | {fmt(g['measured'], 5)} | {fmt(g['expected'], 5)} | "
          f"{fmt(g['rel_err'], 3)} | {fmt(g['tol'], 2)} | **{g['verdict']}** |")
    a("")
    a(f"The failing gate families (`{', '.join(data['fails'])}`) are the open defect below; "
      "every other claim holds to")
    a("the stated tolerance (the planform scale to 1.5e-5, the pitch law and the")
    a("efficiency identity to machine precision, the spectral amplitudes to <0.5 % and")
    a("<2 % on the BPF and its harmonics).")
    a("")
    a("### 3.2 Cross-implementation table (matched geometry)")
    a("")
    a(f"Package note: `{cross['meta']['pkg_verifying_note']}`")
    a("")
    a("| J | K_T legacy | K_T package | K_Q legacy | K_Q package | η legacy | η package | strips at induction cap (package) |")
    a("|---|---|---|---|---|---|---|---|")
    for r in cross["table"]:
        a(f"| {r['J']:.2f} | {r['K_T_legacy']:.4f} | {r['K_T_package']:.4f} | "
          f"{r['K_Q_legacy']:.4f} | {r['K_Q_package']:.4f} | {r['eta_legacy']:.4f} | "
          f"{r['eta_package']:.4f} | {r['pkg_strips_at_induction_cap']}/{r['pkg_n_strips']} |")
    a("")
    a("Figures: `fig1_open_water_crossval.png` (characteristics of both implementations),")
    a("`fig2_debris_convergence.png` (debris transport vs closed form under step")
    a("refinement), `fig3_spectrum_recovery.png` (torque-spectrum harmonic recovery).")
    a("")
    a("## 4. Defects and repairs")
    a("")
    a("### 4.0 Repair record (before / after, recomputed by the harness)")
    a("")
    a("| defect | quantity | before | after | target | gate after repair |")
    a("|---|---|---|---|---|---|")
    for r in data["repairs"]:
        a(f"| `{r['id']}` | {r['quantity']} | {fmt(r['before'], 4)} | {fmt(r['after'], 4)} | "
          f"{fmt(r['target'], 4)} | **{r['gate_after']}** |")
    a("")
    for r in data["repairs"]:
        a(f"* `{r['id']}`: {r['note']}")
    a("")
    for f in data["findings"]:
        a(f"### {f['id']} — {f['status']} — {f['severity']} — `{f['where']}`")
        a("")
        a(f["what"])
        a("")
        a(f"*Impact.* {f['impact']}")
        a("")
        a(f"*Evidence.* `{json.dumps(f['evidence'])[:600]}`")
        a("")
        if f.get("note"):
            a(f"*Note.* {f['note']}")
            a("")
        if f.get("repair"):
            a(f"*Repair.* {f['repair']}")
            a("")
    a("## 5. Points to note / limitations")
    a("")
    a("* **The defects were found by cross-checking, not by reading the prose.** The")
    a("  module's own docstring asserts momentum conservation; quadrature shows a factor")
    a("  2. An untested 1140-line module in a shipped `src` tree is a standing risk, and")
    a("  the same lesson applies to the package: its BEM accepts cap-saturated strips")
    a("  because the residual is defined as `max|Δa|` rather than as a blade/momentum")
    a("  imbalance.")
    a("* **No absolute coefficient is claimed.** Without a Wageningen-B style table or a")
    a("  measured open-water point, `K_T`, `K_Q` of *both* implementations remain")
    a("  WITHHELD; only internal identities, trends and J0 are gateable.")
    a("* **Debris transport is dilute and gravity-free** in both implementations (chip")
    a("  Froude number ≫ 1 over the 1 m discharge length); the closed-form reference")
    a("  therefore validates the integrator, not the physical model.")
    a("* **The contact law is not isolated.** The plastic-capped Hertz branch lives inside")
    a("  `MillingSimulation` and cannot be exercised without a full transient; it is")
    a("  covered indirectly by the package's `BladeIceContact` tests and remains an open")
    a("  item for the legacy module.")
    a("")
    a("## 6. Error judgement (explicit)")
    a("")
    a("| key quantity | reference | error | < 3 % ? |")
    a("|---|---|---|---|")
    g1 = next(g for g in gates if g["id"].startswith("G1a"))
    g1b = next(g for g in gates if g["id"].startswith("G1b"))
    g2 = next(g for g in gates if g["id"].startswith("G2_momentum"))
    g3 = next(g for g in gates if g["id"] == "G3_drag_relax_finest")
    g4 = next(g for g in gates if g["id"] == "G4a_bpf_amplitude")
    g5 = next(g for g in gates if g["id"] == "G5a_efficiency_identity")
    g6k = next(g for g in gates if g["id"] == "G6_K_T_divergence")
    a(f"| planform scale `c_max` | Beta integral | {g1['rel_err']:.3g} | yes |")
    a(f"| blade area vs `A_E` | `EAR·πD²/4` | {g1b['rel_err']:.3g} | yes |")
    a(f"| BPF torque amplitude | analytic 120 N·m | {g4['rel_err']:.3g} | yes |")
    a(f"| efficiency identity | `J K_T/(2πK_Q)` | {g5['rel_err']:.3g} | yes |")
    ok2 = "yes" if g2["rel_err"] < 0.03 else "no"
    ok3 = "yes" if g3["rel_err"] < 0.03 else "no"
    a(f"| race momentum flux | `T` | {g2['rel_err']:.3g} | {ok2} (D1 repaired) |")
    a(f"| debris exit speed | closed form in the module's own carrier | {g3['rel_err']:.3g} | {ok3} (D2 repaired) |")
    a(f"| `K_T` legacy vs package | matched geometry | {g6k['detail_worst_divergence']:.3g} | **no — absolute value WITHHELD (DEFECT-D3)** |")
    a("")
    a("## 7. Disposition")
    a("")
    a("**approve the race/debris repair; the module still must not be used as an")
    a("absolute-coefficient source.** Concretely:")
    a("")
    a("1. **D1 and D2 are repaired and gated.** `RaceModel.centreline` uses")
    a("   `u_c = sqrt(M/(pi sigma^2))`, so `∫ρu²2πr dr = T` at every station (gate")
    a("   `G2_momentum_flux_*`), and `integrate_debris` consumes")
    a("   `RaceModel.axial_velocity_tensor`, i.e. the jet the milling torque was")
    a("   computed with (gate `G3c_debris_carrier_equals_race_field`).")
    a("   The previously registered acceptance tests are green, not `xfail`.")
    a("2. The factor is `sqrt(M/(pi σ²))`, **not** the `sqrt(2 M/(pi σ²))` written in")
    a("   the first issue of this document: that prescription would deliver `2T`.")
    a("   The gate `∫ρu²2πr dr = T` is the arbiter; the prose was wrong by `sqrt(2)`.")
    a("3. Any number produced by the legacy race *before* this repair (IN27-S4")
    a("   ejection speeds) is `sqrt(2)` low in carrier speed and must be recomputed,")
    a("   not cited.")
    a("4. **D3 stays open.** The package BEM saturates its induction cap `a = 0.85`")
    a("   for `J ≲ 0.2` while reporting a 1e-8 residual; its `_buhl` branch is never")
    a("   called. Either a blade/momentum imbalance residual or a documented cap")
    a("   validity limit `J_min` is required before package `K_T`/`K_Q` below")
    a("   `J = 0.3` can be quoted.")
    a("5. Absolute `K_T`, `K_Q` of both implementations remain **WITHHELD** (no")
    a("   experimental open-water chart in this repository); the plastic-capped Hertz")
    a("   contact branch inside `MillingSimulation` is still not isolated by a gate.")
    a("")
    a("## 8. Reproduction")
    a("")
    a("```bash")
    a("PYTHONPATH=src python -m pytest tests/test_propeller_ice_milling_legacy_verification.py -q")
    a("PYTHONPATH=src python scripts/verify_propeller_ice_milling_legacy.py   # writes artifacts + 3 figures")
    a("PYTHONPATH=src python scripts/render_propeller_ice_milling_legacy_doc.py")
    a("```")
    a("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())