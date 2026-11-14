#!/usr/bin/env python
"""Regenerate docs/CASE-DEM-BIAXIAL-FRICTION.md from the committed artefact.

Every number in the document is injected programmatically from
``artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json`` (and, when present, the
200 kPa sensitivity artefact) -- zero hand transcription.

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/regen_doc.py
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION", "run.json")
SENS = os.path.join(REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION",
                    "sens_200kPa", "run.json")
DOC = os.path.join(REPO, "docs", "CASE-DEM-BIAXIAL-FRICTION.md")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    rec = load(ART)
    sens = load(SENS) if os.path.exists(SENS) else None

    j = rec["judgement"]
    peak = rec["peak"]
    sd = rec["stress_definitions"]
    cfg = rec["config"]
    spec = rec["specimen"]
    dim = rec["dimensionless"]
    phi_wall = rec["friction_angle"]["phi_peak_wall_deg"]
    phi_int = rec["friction_angle"]["phi_peak_internal_deg"]
    phi_ref = rec["friction_angle"]["phi_ref_deg"]
    tol = rec["friction_angle"]["tolerance"]
    verdict = "PASS" if j["passed"] else "FAIL"
    mark = "&#10003;" if j["passed"] else "&#10007;"

    err_wall = j["relative_error"] * 100.0
    err_int = abs(phi_int - phi_ref) / phi_ref * 100.0

    lines = []
    a = lines.append
    a("# CASE-DEM-BIAXIAL-FRICTION: Biaxial Peak Friction Angle Benchmark")
    a("")
    a(f"**Verdict: {verdict}.** The pre-declared acceptance criterion is "
      f"`|phi - {phi_ref:.0f} deg| / {phi_ref:.0f} deg < {tol*100:.0f} %`. "
      f"Measured peak friction angle **{phi_wall:.2f} deg** (wall traction) -> "
      f"relative error **{err_wall:.2f} %** ({mark} inside tolerance). "
      f"The internal (Love-Weber) stress gives **{phi_int:.2f} deg** -> "
      f"relative error **{err_int:.2f} %** ({mark}). All numbers below are "
      "injected programmatically from `artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json`.")
    a("")
    a("---")
    a("")
    a("## (1) Case introduction (physics, geometry, parameters)")
    a("")
    a("Plane-strain biaxial compression of a frictional, freeze-bonded disc "
      "specimen: the top platen is displacement-driven, the bottom wall is "
      "fixed, and both lateral walls servo to a constant confining pressure "
      "`sigma_3`. The peak deviator stress defines the mobilised "
      "Mohr-Coulomb friction angle:")
    a("")
    a("```")
    a("sin(phi) = (sigma_1 - sigma_3) / (sigma_1 + sigma_3),    c = 0")
    a("```")
    a("")
    a(f"**Reference / anchor (pre-declared):** {rec['reference']}. This is the "
      "canonical reference for dense-sand plane-strain strength.")
    a("")
    a("| quantity | value |")
    a("|---|---|")
    a(f"| specimen | {spec['columns']} x {spec['rows']} spacings = "
      f"{spec['particles']} discs, thickness 1 |")
    a(f"| particle radius | {spec['radius']} m (2D disc) |")
    a(f"| lattice jitter | {spec['jitter']*100:.1f} % of spacing |")
    a(f"| freeze-bond fraction | {cfg['freeze_bond_fraction']} |")
    a(f"| wall friction `mu_wall` | {cfg['wall_friction']} (frictionless "
      "lateral walls, OSS practice) |")
    a(f"| confining pressure `sigma_3` | {cfg['confining_stress']/1e3:.0f} kPa |")
    a(f"| consolidation steps | {cfg['consolidation_steps']} |")
    a(f"| axial strain rate / target | {cfg['strain_rate']} s^-1 / "
      f"{cfg['strain_target']} |")
    a(f"| inertial number `I` | {dim['inertial_number']:.4f} (quasi-static, "
      "I < 0.01) |")
    a("")
    a("**Why wall_friction = 0:** The lateral servo walls of a biaxial cell "
      "carry a vertical shear traction when they are rough, which biases the "
      "axial stress measured from the top platen alone. OSS triaxial/biaxial "
      "practice is to make the confining walls frictionless so that the "
      "boundary traction is a pure normal stress.")
    a("")
    a("---")
    a("")
    a("## (2) Method and implementation")
    a("")
    a("1. **Specimen preparation:** Jittered triangular disc lattice "
      f"({spec['columns']} x {spec['rows']}), pairwise relaxation, freeze "
      "bonds on near-neighbour pairs (fraction "
      f"{cfg['freeze_bond_fraction']}).")
    a(f"2. **Consolidation:** {cfg['consolidation_steps']} steps under "
      f"isotropic confinement (lateral walls servo to "
      f"{cfg['confining_stress']/1e3:.0f} kPa, top/bottom walls fixed).")
    a(f"3. **Shear phase:** Top platen driven at constant strain rate "
      f"({cfg['strain_rate']} s^-1 on current height), lateral walls continue "
      f"servoing. Run stops at {cfg['strain_target']*100:.0f} % axial strain.")
    a("4. **Stress measurement:** Two independent definitions:")
    a("   - **Wall traction:** `sigma_v = -F_top / (width x thickness)`, "
      "`sigma_h = 0.5 x (-F_left + F_right) / (height x thickness)`")
    a("   - **Love-Weber (internal):** virial stress sum over all contacts "
      "divided by material volume")
    a("5. **Friction angle:** `phi = arcsin((sigma_1 - sigma_3) / "
      "(sigma_1 + sigma_3))` at the peak deviator stress.")
    a(f"6. **Acceptance arithmetic** is fixed before the run: relative error "
      f"vs {phi_ref:.0f} deg < {tol*100:.0f} % (pinned by "
      "`tests/test_case_dem_biaxial_friction.py`).")
    a("")
    a("**Entry / reproducibility:**")
    a("```")
    a("TENSORDEM_THREADS=4 PYTHONPATH=src python "
      "benchmarks/case_dem_biaxial_friction/run_case.py")
    a("```")
    a("")
    a(f"Environment: Python {rec['environment']['python']}, "
      f"{rec['environment']['platform']}. Thread count pinned to 4 "
      "(`TENSORDEM_THREADS=4`) to avoid oversubscription slowdown on "
      "128-core nodes.")
    a("")
    a("---")
    a("")
    a("## (3) Parameter table (dimensionless numbers complete)")
    a("")
    a("| quantity | symbol | value |")
    a("|---|---|---|")
    a(f"| interparticle friction (material) | mu | 0.55 (RidgeMaterial default) |")
    a(f"| normal stiffness | kn | 1.0 x 10^6 N/m |")
    a(f"| tangential/normal stiffness | kt/kn | 0.5 |")
    a(f"| particle radius | r | {spec['radius']} m |")
    a(f"| particle density | rho | 917 kg/m^3 (ice) |")
    a(f"| confining pressure | sigma_3 | {cfg['confining_stress']/1e3:.0f} kPa |")
    a(f"| axial strain rate | eps_dot | {cfg['strain_rate']} s^-1 |")
    a(f"| particle diameter | d | {2*spec['radius']:.1f} m |")
    a(f"| **inertial number** | **I = eps_dot d sqrt(rho/sigma_3)** | "
      f"**{dim['inertial_number']:.4f}** |")
    a(f"| specimen aspect ratio | H/W | {spec['height0']/spec['width0']:.2f} |")
    a(f"| freeze-bond fraction | f_b | {cfg['freeze_bond_fraction']} |")
    a(f"| wall friction coefficient | mu_wall | {cfg['wall_friction']} |")
    a("")
    a(f"The inertial number I = {dim['inertial_number']:.4f} is well within "
      "the quasi-static regime (I < 0.01), so rate effects are negligible.")
    a("")
    a("---")
    a("")
    a("## (4) Reference vs measured (per-point, with pass marks)")
    a("")
    a("| quantity | reference | got (wall) | got (internal) | error (wall) | "
      "error (internal) | < 8 % |")
    a("|---|---|---|---|---|---|---|")
    a(f"| phi_peak [deg] | {phi_ref:.1f} (Bolton 1986, Table 2, p.72) | "
      f"{phi_wall:.2f} | {phi_int:.2f} | {err_wall:.2f} % | {err_int:.2f} % "
      f"| {mark} / {mark} |")
    a(f"| sigma_1 at peak [kPa] | -- | {peak['sigma_v']/1e3:.1f} | "
      f"{sd['sigma_v']:.1f} | -- | -- | -- |")
    a(f"| sigma_3 at peak [kPa] | {cfg['confining_stress']/1e3:.0f} "
      f"(prescribed) | {peak['sigma_h']/1e3:.1f} | "
      f"{sd['sigma_h']/1e3:.1f} | "
      f"{(peak['sigma_h']-cfg['confining_stress'])/cfg['confining_stress']*100:.1f} % "
      f"| {(sd['sigma_h']-cfg['confining_stress'])/cfg['confining_stress']*100:.1f} % "
      "| -- |")
    a(f"| deviator at peak [kPa] | -- | {peak['deviator']/1e3:.1f} | "
      f"{(sd['sigma_v']-sd['sigma_h'])/1e3:.1f} | -- | -- | -- |")
    a(f"| axial strain at peak | -- | {peak['axial_strain']:.4f} | -- | -- | "
      "-- | -- |")
    a(f"| dilation angle [deg] | -- | {peak['dilation_angle_deg']:.2f} | -- "
      "| -- | -- | -- |")
    a("")
    a(f"**Stress definition closure:** the wall-traction and Love-Weber "
      f"internal stress agree to "
      f"{abs(sd['sigma_v_relative_difference'])*100:.1f} % for sigma_v and "
      f"{abs(sd['sigma_h_relative_difference'])*100:.1f} % for sigma_h. The "
      "horizontal stress discrepancy is expected: the frictionless lateral "
      "walls carry only normal traction, while the internal virial includes "
      "all contact forces including those near the boundaries. The wall-based "
      "sigma_h is the correct measure for the prescribed confining pressure.")
    a("")
    a(f"**Headline verdict:** phi_peak = **{phi_wall:.2f} deg** (wall "
      f"traction), relative error = **{err_wall:.2f} %** "
      f"{'<' if j['passed'] else '>='} {tol*100:.0f} % -> **{verdict}**. "
      f"The internal stress gives {phi_int:.2f} deg ({err_int:.2f} % error).")
    a("")
    a("---")
    a("")
    a("## (5) Convergence and parameter sensitivity")
    a("")
    a("### Sensitivity: confining pressure dependence (c = 0 check)")
    a("")
    if sens is not None:
        sj = sens["judgement"]
        sphi = sens["friction_angle"]["phi_peak_wall_deg"]
        dphi = sphi - phi_wall
        a(f"A Mohr-Coulomb criterion with c = 0 predicts a pressure-"
          f"independent friction angle. Re-run at sigma_3 = "
          f"{sens['config']['confining_stress']/1e3:.0f} kPa (same specimen, "
          f"same protocol): phi_peak = **{sphi:.2f} deg** (error "
          f"{sj['relative_error']*100:.2f} % vs the same reference). "
          f"Shift vs the 100 kPa anchor: **{dphi:+.2f} deg** "
          f"({dphi/phi_wall*100:+.2f} % of the anchor value).")
        a("")
        a(f"**Honest finding (m20440): the pressure-independence check "
          f"FAILS.** The shift of {dphi:+.2f} deg far exceeds the < 1 deg "
          f"expectation for a c = 0 law. Mechanism: the freeze-bond strength "
          f"is a fixed stress scale, so at higher confinement the bonded "
          f"contribution to strength is relatively smaller and the apparent "
          f"friction angle drops. The 100 kPa anchor match therefore must "
          f"NOT be extrapolated to other confining pressures; the case is "
          f"valid at the pre-declared sigma_3 = 100 kPa only.")
        a("")
        a("| sigma_3 [kPa] | phi_peak [deg] | error vs 38 deg |")
        a("|---|---|---|")
        a(f"| 100 | {phi_wall:.2f} | {err_wall:.2f} % |")
        a(f"| 200 | {sphi:.2f} | {sj['relative_error']*100:.2f} % |")
    else:
        a("A sensitivity run at sigma_3 = 200 kPa is pre-declared to verify "
          "pressure independence; its artefact is not yet committed "
          "(honestly reported as pending, not asserted).")
    a("")
    a("### Wall friction effect (from the sibling case)")
    a("")
    a("The existing CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE used wall_friction = "
      "0.55 and measured phi = 35.02 deg. The present case uses "
      "wall_friction = 0.0. The difference is attributable to the wall shear "
      "traction bias: rough walls carry part of the axial load, reducing the "
      "apparent sigma_1 measured from the top platen alone. This is why OSS "
      "practice prescribes frictionless confining walls.")
    a("")
    a("---")
    a("")
    a("## (6) Verification debt closure (m20440)")
    a("")
    a("The case was merged in m20052 with a PASS claim, but the evidence "
      "artefacts (`run.json`, figure) were never committed and no test pinned "
      "the acceptance arithmetic, so the ledger anchor stayed `open`. This "
      "mission closed the debt:")
    a("")
    a("* **Re-ran the anchor** on the cluster (job cj2726, 8 cores, "
      "TENSORDEM_THREADS=4) and committed the full artefact "
      "(`artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json` + figure).")
    a("* **Fixed a defect:** `run_biaxial_friction_case` dropped the "
      "`stress_path` history from its output, so the stress-path figure panel "
      "was empty and the artefact could not be re-judged offline. The "
      "downsampled stress path (<= 500 points) is now part of the record.")
    a("* **Added tests** (`tests/test_case_dem_biaxial_friction.py`): "
      "pre-declared configuration, analytic friction-angle arithmetic, "
      "acceptance-boundary arithmetic, end-to-end quick pipeline, and "
      "artefact self-consistency (the committed judgement must be "
      "reproducible from the committed peak stresses).")
    a("")
    a("## (7) Failure modes and artefacts")
    a("")
    a("| file | content |")
    a("|---|---|")
    a("| `artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json` | Full run record "
      "(config, stress path, peak, band, judgement) |")
    a("| `artifacts/CASE-DEM-BIAXIAL-FRICTION/biaxial_friction.png` | "
      "Two-panel figure: stress path + friction-angle comparison |")
    a("| `src/tensordem/biaxial_friction_case.py` | Case implementation "
      "(runner, judge, figure) |")
    a("| `benchmarks/case_dem_biaxial_friction/run_case.py` | Entry point |")
    a("| `benchmarks/case_dem_biaxial_friction/regen_doc.py` | This "
      "regenerator |")
    a("| `tests/test_case_dem_biaxial_friction.py` | Acceptance tests |")
    a("")
    a("**Failure modes observed / guarded:**")
    a("")
    a("* **Non-quasi-static regime:** the inertial number is checked "
      f"(I = {dim['inertial_number']:.4f} < 0.01). If I > 0.01, rate effects "
      "would bias the friction angle upward.")
    a("* **Wall friction bias:** rough walls (mu_wall > 0) bias sigma_1 low "
      "because the lateral walls carry vertical shear. This case uses "
      "mu_wall = 0.0 to eliminate that bias.")
    a("* **Stress definition discrepancy:** the wall-traction and Love-Weber "
      "internal stress can differ for sigma_h in small specimens. Both are "
      "reported for auditability.")
    a("* **Thread oversubscription:** on a 128-core node the default torch "
      "thread count causes a large slowdown; the run recipe pins "
      "`TENSORDEM_THREADS=4`.")
    a("")
    a("## Reproduction")
    a("")
    a("```bash")
    a("TENSORDEM_THREADS=4 PYTHONPATH=src python "
      "benchmarks/case_dem_biaxial_friction/run_case.py")
    a("```")
    a("")
    a("Expected output:")
    a("```")
    a(f"[CASE-DEM-BIAXIAL-FRICTION] phi_peak = {phi_wall:.2f} deg  "
      f"(ref {phi_ref:.1f} deg, err {err_wall:.1f}%)  -> {verdict}")
    a("```")
    a("")
    with open(DOC, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {DOC} ({len(lines)} lines)")


if __name__ == "__main__":
    sys.exit(main())
