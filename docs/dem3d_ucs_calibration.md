# DEM3D UCS experimental-curve calibration screen

This tool provides a reproducible first-stage parameter screening loop around the
existing 3-D platen-compression solver. It consumes a user-supplied experimental
CSV and ranks a grid of candidate bond stiffness and tensile breaking-strain
values against the measured engineering stress-strain curve.

## Input format

UTF-8 CSV with the following required columns (additional columns are ignored):

\`\`\`csv
axial_strain,compressive_stress_Pa
0.0,0.0
0.002,120000
0.005,260000
0.010,310000
0.015,280000
\`\`\`

The numbers above are illustrative only, not measured ice data. Replace them with
data from a documented test and preserve the original observations. Strain must
start at zero, increase strictly, remain within 0–0.08, and have at least three
points. Compressive stress must be finite and nonnegative, with at least one
positive value. Use engineering stress in Pa and dimensionless engineering strain.

## Run

\`\`\`bash
python scripts/calibrate_dem3d_ucs.py \\
  --experimental-csv data/ice_ucs_curve.csv \\
  --bond-stiffness-scales 0.5 1.0 2.0 \\
  --breaking-strains 0.01 0.015 0.02 \\
  --output results-dem3d-ucs-calibration
\`\`\`

By default, the 3 × 3 grid runs each candidate twice and requires deterministic
history signatures. Each simulation covers the maximum experimental strain.
Only bond stiffness and tensile breaking strain vary; geometry, loading rate,
density, contact and platen parameters remain fixed. The workflow writes:

- \`ucs_calibration_report.json\`: checks, ranked candidates, score definition,
  parameter values, peak metrics, broken-bond fraction, energy diagnostics and
  limitations.
- \`ucs_calibration_summary.csv\`: one row per candidate, ranked by objective.
- \`ucs_calibration_histories.csv\`: full numerical stress-strain histories.

The objective is a transparent baseline:

\`normalized stress-curve RMSE + 0.25 × relative peak-stress error + 0.25 × normalized peak-strain error\`.

Do not change weights after looking at the result to force a preferred candidate.
For a publication or engineering study, define the objective, uncertainty and
acceptance criteria before fitting; include repeat tests and independent
validation curves.

## Interpretation limits

A PASS only confirms finite, repeatable numerical candidates and adequate strain
coverage. A low objective is not proof of unique parameter identification or
physical validity. The present sweep does not fit fracture energy, shear failure,
contact friction, damping, temperature, salinity, or rate effects. The model is
a simple-cubic bonded-sphere prism under infinite frictionless prescribed-speed
platens, not a laboratory-equivalent ice specimen.

Do not use illustrative CSV values as calibration targets. Archive the raw
experimental curve, specimen dimensions, temperature, salinity, loading rate,
instrument uncertainty, and boundary conditions. Fit on one dataset and validate
against independent stress-strain and fracture observations. A physical fracture
energy calibration requires additional fracture tests and a model-aware energy
release audit; breaking strain is not fracture energy.
