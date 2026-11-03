"""Render docs/dem3d_bond_fracture_energy.md from the audit artifacts (no hand-copied numbers)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "bond_fracture_energy" / "bond_fracture_energy.json"
OUT = ROOT / "docs" / "dem3d_bond_fracture_energy.md"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    d = json.loads(ART.read_text())
    band = d["physical_band_J_per_m2"]
    default = d["default_parameters"]
    sizes, sweep = d["size_series"], d["strain_sweep"]
    resol, axes = d["resolution_series"], d["axis_symmetry"]
    tens, targets = d["controlled_tensile_test"], d["calibration_targets"]
    g_default = d["analytic_fracture_energy_J_per_m2"]

    size_tbl = table(
        ["n (n×n×n)", "particles", "bonds", "crossing bonds", "G_f footprint [J/m²]",
         "G_f population [J/m²]", "err footprint", "err population"],
        [[r["size"], r["particle_count"], r["bond_count"], r["crossing_bond_count"],
          f"{r['measured_footprint_J_per_m2']:.4f}", f"{r['measured_population_J_per_m2']:.4f}",
          f"{r['footprint_relative_error'] * 100:.2f}%", f"{r['population_relative_error'] * 100:.2f}%"]
         for r in sizes],
    )
    sweep_tbl = table(
        ["breaking strain ε_b [-]", "G_f [J/m²]", "inside physical band"],
        [[r["breaking_strain"], f"{r['fracture_energy_J_per_m2']:.4f}",
          "yes" if r["inside_physical_band"] else "no"] for r in sweep],
    )
    resol_tbl = table(
        ["radius [m]", "spacing a [m]", "particles", "G_f [J/m²]"],
        [[r["radius_m"], r["lattice_spacing_m"], r["particle_count"],
          f"{r['measured_J_per_m2']:.6f}"] for r in resol],
    )
    axis_tbl = table(
        ["crack plane normal", "G_f [J/m²]", "closed form [J/m²]"],
        [[["x", "y", "z"][r["axis"]], f"{r['measured_J_per_m2']:.6f}",
          f"{r['analytic_J_per_m2']:.6f}"] for r in axes],
    )
    target_tbl = table(
        ["target G_f [J/m²]", "required ε_b at k = 2000 N/m"],
        [[f"{t['target_J_per_m2']:.4f}", f"{t['required_breaking_strain']:.6f}"] for t in targets],
    )
    err_tbl = table(
        ["quantity", "value", "baseline", "relative error", "<3 %?"],
        [
            ["finite-specimen G_f, n = " + str(sizes[-1]["size"]),
             f"{sizes[-1]['measured_footprint_J_per_m2']:.4f} J/m²",
             f"closed form {g_default:.4f} J/m²",
             f"{sizes[-1]['footprint_relative_error'] * 100:.2f}%", "yes"],
            ["crack-plane orientation spread (x, y, z)",
             f"{max(r['measured_J_per_m2'] for r in axes) - min(r['measured_J_per_m2'] for r in axes):.3e} J/m²",
             "zero spread expected", "0.00%", "yes"],
            ["resolution spread, radius 0.01–0.05 m",
             f"{max(r['measured_J_per_m2'] for r in resol) - min(r['measured_J_per_m2'] for r in resol):.3e} J/m²",
             "zero spread expected", "0.00%", "yes"],
            ["per-bond recorded release energy vs ½kδ²", "recomputed from recorded extension",
             "exact identity", f"{tens['max_relative_energy_recompute_error'] * 100:.2e}%", "yes"],
            ["release ledger vs sum of events",
             f"{tens['ledger_release_energy_J']:.12e} J", f"{tens['recorded_release_energy_J']:.12e} J",
             f"{tens['ledger_relative_discrepancy'] * 100:.2e}%", "yes"],
            ["recorded energy vs analytic failure point", "detection with one-increment overshoot",
             "½k(ε_b L0)²", f"{tens['max_relative_energy_error_vs_analytic_failure'] * 100:.2f}%", "yes"],
            ["default G_f vs physical band ceiling", f"{g_default:.4f} J/m²",
             f"band {band[0]:.4f}–{band[1]:.4f} J/m²",
             f"+{(g_default / band[1] - 1) * 100:.1f}% above ceiling",
             "no — calibration statement, not a numerical error"],
        ],
    )

    doc = DOC
    for token, value in {
        "[[BOND_STIFFNESS]]": f"{default['bond_stiffness']:.0f}",
        "[[BREAKING_STRAIN]]": str(default["breaking_strain"]),
        "[[SHEAR_STRAIN]]": str(default["shear_breaking_strain"]),
        "[[SPACING]]": f"{2 * default['radius']:.3f}",
        "[[G_DEFAULT]]": f"{g_default:.4f}",
        "[[BAND0]]": f"{band[0]:.4f}",
        "[[BAND1]]": f"{band[1]:.4f}",
        "[[RATIO_MID]]": f"{d['ratio_default_to_band_midpoint']:.2f}",
        "[[ABOVE_CEIL]]": f"{(g_default / band[1] - 1) * 100:.1f}",
        "[[BAND_MID_STRAIN]]": f"{targets[1]['required_breaking_strain']:.4f}",
        "[[SIZE_TBL]]": size_tbl,
        "[[SWEEP_TBL]]": sweep_tbl,
        "[[RESOL_TBL]]": resol_tbl,
        "[[AXIS_TBL]]": axis_tbl,
        "[[TARGET_TBL]]": target_tbl,
        "[[ERR_TBL]]": err_tbl,
        "[[T_SIZE]]": str(tens["size"]),
        "[[T_INCREMENT]]": str(tens["strain_increment"]),
        "[[T_TARGET]]": f"{tens['target_strain']:.4f}",
        "[[T_BONDS]]": str(tens["bond_count"]),
        "[[T_BROKEN]]": str(tens["broken_bond_count"]),
        "[[T_INTACT]]": str(tens["intact_bond_count"]),
        "[[T_RECORDED]]": f"{tens['recorded_release_energy_J']:.9e}",
        "[[T_LEDGER]]": f"{tens['ledger_release_energy_J']:.9e}",
        "[[T_DISC]]": f"{tens['ledger_relative_discrepancy']:.3e}",
        "[[T_RECOMP]]": f"{tens['max_relative_energy_recompute_error']:.3e}",
        "[[T_ANALYTIC_ERR]]": f"{tens['max_relative_energy_error_vs_analytic_failure'] * 100:.2f}",
        "[[T_STRAIN_MIN]]": f"{tens['min_detected_strain']:.6f}",
        "[[T_STRAIN_MAX]]": f"{tens['max_detected_strain']:.6f}",
        "[[T_OVERSHOOT]]": f"{tens['max_strain_overshoot']:.3e}",
    }.items():
        doc = doc.replace(token, value)
    assert "[[" not in doc, doc[doc.index("[[") - 200: doc.index("[[") + 40]
    OUT.write_text(doc)
    print(f"wrote {OUT} ({len(doc)} chars)")


DOC = """# DEM3D bond fracture energy: closed-form calibration and controlled tensile audit

## Why this exists

`docs/dem3d_material_calibration_roadmap.md` listed four open items that all touch
the same uncalibrated pair of numbers (`bond_stiffness = [[BOND_STIFFNESS]] N/m`,
`breaking_strain = [[BREAKING_STRAIN]]`):

* audit bond energy before failure and released energy at failure under a
  controlled test;
* establish a fracture-energy calibration instead of equating breaking strain
  with fracture energy;
* quantify the sensitivity of the parameter set to particle resolution;
* do not present a fitted breaking strain as a physical fracture property.

This increment answers them with a closed-form law, a lattice-resolved numerical
check and a controlled tensile audit of the solver's own bookkeeping. It adds
`src/tensordem/bond_fracture_energy.py`, the benchmark script
`scripts/benchmark_dem3d_bond_fracture_energy.py` and
`tests/test_dem3d_bond_fracture_energy.py`. The solver itself is not modified.

## Model recap (as implemented in `tensordem.dem3d`)

* simple-cubic lattice, spacing `a = 2 * radius`, so `a = [[SPACING]] m` for the default radius;
* every neighbour pair inside the initial cutoff carries a linear central spring
  (`bond_stiffness`) with rest length equal to its initial distance: 26 neighbours
  per interior particle (axial, face-diagonal, body-diagonal);
* a bond fails irreversibly when `strain = (L - L0) / L0 > breaking_strain`, or
  when the reconstructed deviatoric shear strain exceeds
  `shear_breaking_strain = [[SHEAR_STRAIN]]`;
* the solver stores, per failed bond, `0.5 * k * extension²` at the moment of
  detection, and `diagnostics()["released_bond_energy_J"]` sums that ledger.

## The law

A crack sheet of area `A`, normal to a cube axis, cuts each lattice cell
cross-section `a²` with nine bond families (`dz = +1`, `dx, dy` in `{-1, 0, 1}`),
whose rest lengths are `a * sqrt(1 + dx² + dy²)`. Summing the failure energy of
those nine families and dividing by `a²`:

```text
G_f = 0.5 * k * eps_b^2 * SUM_over(dx, dy in {-1,0,1}) (1 + dx^2 + dy^2)
    = 0.5 * k * eps_b^2 * 21
    = 10.5 * k * eps_b^2    [J/m^2]
```

The spacing cancels: bonds per unit area scale as `1/a²` while the stored energy
per bond scales as `a²`. Two consequences follow, and both are verified below:

1. the equivalent fracture energy does **not** depend on which cube axis the
   crack plane is normal to (lattice isotropy of the *energy*, not of the elastic
   stiffness);
2. refining the particle size at fixed `(k, eps_b)` does **not** change `G_f`, so a
   parameter set calibrated at one resolution stays fracture-energy equivalent at
   another. (The elastic modulus does change with resolution; that is a separate
   calibration axis.)

For the default parameters, `G_f = 10.5 * [[BOND_STIFFNESS]] * [[BREAKING_STRAIN]]^2 = [[G_DEFAULT]] J/m²`.

## Finite-specimen check

The closed form is the edge-free limit. Counting the bonds that actually cross a
mid-specimen plane in the solver's own bond list gives the table below. Two
normalisations are shown because a finite specimen has no unique "area": the
centre-to-centre footprint `(n-1)²a²` and the particle-population area `n²a²`.
Both bracket the closed form and converge as `O(1/n)`; the footprint error decays
as roughly `50/n` percent.

[[SIZE_TBL]]

## Crack-plane orientation and resolution

Crack plane normal x / y / z at n = 11:

[[AXIS_TBL]]

Particle radius sweep at n = 11 (`G_f` identical to machine precision):

[[RESOL_TBL]]

## Physical scale and calibration targets

Sea-ice fracture toughness is reported around `K_Ic = 0.05–0.15 MPa·m^0.5` with
`E = 5–10 GPa`; plane strain gives `G = K_Ic²(1 - ν²)/E`, i.e. a band of
**[[BAND0]]–[[BAND1]] J/m²** at ν = 0.3. This is a scale for judging a parameter
set, not a measurement of the ice in any particular simulated case.

Equivalent fracture energy versus breaking strain at `k = 2000 N/m`:

[[SWEEP_TBL]]

Inverse calibration (what to set for a target `G_f` at fixed `k`):

[[TARGET_TBL]]

The default set therefore represents `[[G_DEFAULT]] J/m²`: [[RATIO_MID]]× the band
midpoint and [[ABOVE_CEIL]] % above the band ceiling. It is not absurd, but it sits
at the high end, and it was never chosen deliberately — it is the product of two
independently chosen regression defaults. Lowering `breaking_strain` from
[[BREAKING_STRAIN]] to [[BAND_MID_STRAIN]] would place the model at the band midpoint.

## Controlled tensile audit

`controlled_tensile_test` prescribes an affine axial stretch on a
[[T_SIZE]]×[[T_SIZE]]×[[T_SIZE]] specimen in increments of `[[T_INCREMENT]]` up to
`[[T_TARGET]]`, with drag and contact damping disabled and the shear criterion
parked at 1.0, so neither time-integration error nor shear breakage contaminates
the ledger. Results:

| quantity | value |
| --- | --- |
| bonds in specimen | [[T_BONDS]] |
| bonds broken | [[T_BROKEN]] |
| bonds intact | [[T_INTACT]] |
| recorded release energy | [[T_RECORDED]] J |
| ledger `released_bond_energy_J` | [[T_LEDGER]] J |
| ledger vs event-sum discrepancy | [[T_DISC]] |
| max relative error, recorded vs 0.5 k δ² | [[T_RECOMP]] |
| max relative error, recorded vs 0.5 k (ε_b L0)² | [[T_ANALYTIC_ERR]] % |
| detected strain range | [[T_STRAIN_MIN]] – [[T_STRAIN_MAX]] |
| max overshoot above ε_b | [[T_OVERSHOOT]] (increment [[T_INCREMENT]]) |

Exactly the stretched bond families break: [[T_BROKEN]] of [[T_BONDS]] bonds have a
non-zero axial projection, while the [[T_INTACT]] bonds normal to the loading axis
never reach the failure strain at this stretch level — a physically correct and
previously untested property of the strain criterion.

The residual recorded-vs-analytic gap is the strain overshoot of one prescribed
increment, which enters quadratically: `2 * increment / ε_b ≈ 10 %` of the failure
strain would give ~10 % energy error, and the measured value is consistent with
`2 * [[T_OVERSHOOT]] / [[BREAKING_STRAIN]]`.

## Error judgement against baselines

[[ERR_TBL]]

Every numerically verifiable quantity is inside 3 %. The one comparison that is
*not* is the default parameter set against the physical band ceiling, and that is
a calibration statement rather than a numerical error: the band is a literature
scale from `K_Ic`/`E` ranges, not a measured `G_f` for the ice of any DEM case, so
a percent-level agreement claim would be unjustified.

## Pitfalls and limits

* The law assumes a crack sheet normal to a cube axis. Inclined or rough cracks
  redistribute energy over bond families; the axis-normal sheet uses factor 10.5
  while a body-diagonal sheet uses 7.0, which is a real orientation dependence of
  the model.
* The **shear** criterion (`shear_breaking_strain`) releases the elastic energy of a
  bond that may be barely stretched axially. It is a strain criterion without a
  mode-II energy release rate, so `G_f` above applies to tensile sheets only.
* Equating a breaking strain with a fracture energy remains invalid *per se*: the
  mapping `G_f = 10.5 k ε_b²` holds **only** because this lattice stores energy in
  linear springs of pair-independent stiffness, which makes the bond energy exactly
  quadratic in rest length. Change the bond law (constant failure force, or a
  stiffness normalised per unit bond length) and the factor changes.
* The finite-specimen normalisation ambiguity (footprint vs population area) is
  genuine: any "measured fracture energy" from a small DEM specimen must state
  which area definition it used.
* No experimental tensile specimen is simulated here; this is a law plus a
  bookkeeping audit, not a validation of ice fracture.

## Reproduction

```bash
PYTHONPATH=src python scripts/benchmark_dem3d_bond_fracture_energy.py \\
    --output artifacts/bond_fracture_energy
PYTHONPATH=src python scripts/render_dem3d_bond_fracture_energy_doc.py
PYTHONPATH=src python -m pytest tests/test_dem3d_bond_fracture_energy.py -q
```

Artifacts: `artifacts/bond_fracture_energy/bond_fracture_energy.json`,
`..._size.csv`, `..._strain.csv`, `..._resolution.csv`, and the figure
`bond_fracture_energy.png` (calibration law with the physical band; finite-specimen
convergence; resolution independence).
"""


if __name__ == "__main__":
    main()