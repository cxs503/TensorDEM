# TensorDEM DEM3D benchmark programme: paper-style technical report

> **Manuscript status:** reproducible software-verification report / manuscript scaffold. Numerical tables and plots must be populated from the archived output of the exact run being reported. The repository does not hard-code illustrative numbers as computed results. A regression PASS is not experimental validation.

## Abstract

This report defines a reproducible benchmark programme for TensorDEM, a PyTorch-based bonded-sphere discrete element method (DEM) prototype intended as a foundation for ice-fracture and ship–ice interaction studies. The benchmark programme separates (i) analytic and invariant verification of implemented force and fracture laws, (ii) specimen-scale numerical response fixtures, (iii) time-step and particle-resolution sensitivity, (iv) prescribed moving-wedge resistance studies, and (v) comparisons of published empirical resistance formulae against a traceable reference-data table. Sampled particle trajectories and fracture-event records are post-processed into damage, displacement and crack-location maps; CSV histories are used to produce load–deflection, stress–strain, resistance–time and sensitivity comparison plots.

The current model uses linear central-force bonds and linear penalty normal contact. Its simple-cubic lattice, prescribed tooling and limited contact physics are important restrictions. The programme therefore distinguishes code regression, numerical sensitivity, parameter calibration and physical validation. No claim of full-scale icebreaking prediction follows from a passing software benchmark.

**Keywords:** discrete element method; bonded particles; sea ice; fracture; benchmark; verification; sensitivity analysis; ship–ice resistance.

## 1. Scope and research questions

The benchmark suite is intended to answer four progressively harder questions:

1. **Are the implemented equations and invariants exercised correctly?** Isolated spring/contact tests, force balance, objective rigid rotation, irreversible bond failure and time-step guards address this layer.
2. **Does the solver produce finite, reproducible specimen-scale histories?** Compression and three-point-bending fixtures test load transfer, deformation, fracture bookkeeping and output persistence.
3. **How sensitive are observables to numerical choices?** Time-step, load-ramp and particle-resolution campaigns compare peak force/stress, deflection, first-fracture time and broken-bond count.
4. **Can the present model predict physical icebreaking loads?** Not yet established. That requires traceable experimental geometries and uncertainties, material calibration on one data subset, and independent hold-out validation under matched definitions and conditions.

The last question is deliberately kept separate from the first three. Deterministic signatures and finite output are useful software evidence, but they do not establish physical accuracy.

## 2. Benchmark case catalogue

| Layer | Case / campaign | Primary observables | Current evidence and interpretation |
|---|---|---|---|
| Analytic mechanics | Isolated axial bond | Force versus bond extension | Compare the implemented central-force law with its analytical expression. |
| Analytic mechanics | Linear penalty contact | Normal force versus overlap | Verifies the implemented linear penalty law; it is not Hertz–Mindlin contact. |
| Invariants | Force balance and action–reaction | Net-force residual | Checks internal force balance and tool reaction consistency for a controlled state. |
| Invariants | Tensile / shear failure and rigid rotation | Broken bonds, failure mode, residual force | Checks irreversible failure and objectivity under prescribed deformation states. |
| Numerical specimen response | Affine compression patch | Reaction, stress, secant modulus | Kinematic patch test; not a standard laboratory UCS test. |
| Dynamic specimen response | Platen UCS-style compression | Stress–strain curve, peak stress, fracture timing | Regression fixture using a simple-cubic prism and prescribed frictionless platens. |
| Flexure | Three-point-bending load-transfer fixture | Support reaction, midspan deflection, broken bonds | Fixed-particle supports and distributed nodal load; not a calibrated flexural-strength test. |
| Numerical sensitivity | UCS time-step campaign | Peak stress, strain at peak, first-fracture time | Compares coarser steps with the finest step in the campaign; the finest run is not an exact solution. |
| Numerical sensitivity | Bending load-ramp campaign | Peak support reaction, deflection, damage | Step-count changes also alter ramp duration and effective loading rate; not pure time-step convergence. |
| Numerical sensitivity | Bending particle-resolution campaign | Reaction, deflection, broken bonds | Envelope is approximately preserved, but stiffness is not recalibrated with resolution. Treat as sensitivity, not mesh-objective convergence. |
| Ship–ice contact prototype | Moving wedge / bow | Ice resistance, resistance work, fracture onset | Prescribed dry-contact geometry without fluid dynamics, buoyancy or ship motion. |
| Geometry / parameter sensitivity | Bow angle, speed, width and time step | Peak/mean resistance, work, broken-bond fraction | Controlled parameter sweeps; differences are model sensitivities, not uncertainty bounds. |
| External baseline audit | MT Uikku level-ice resistance formulas | MAPE, signed error, RMSE | Compares published empirical formulas with the traceable experimental reference table. It does **not** validate TensorDEM. |
| 3-D visualization | Named indentation benchmark suite | Particle damage, displacement, fracture locations, force history | Maps are generated only where the relevant sampled trajectory or CSV actually exists. |

The machine-readable case definitions live under `benchmarks/dem3d/`. Each individual case note explains its loading fixture, output fields and limitations. Use those case notes alongside this report; do not detach numerical values from the configuration that generated them.

## 3. Governing model and numerical implementation

TensorDEM's 3-D prototype represents ice as spherical particles connected by an initially prescribed simple-cubic central-force bond network. The current bond force is linear in the change of bond length. Bonds can fail irreversibly when the configured tensile or objective shear-strain criterion is exceeded. After bond failure, particles interact through a linear penalty normal-contact law. The solver uses PyTorch tensors and a semi-implicit explicit-time integration workflow with a conservative time-step guard.

The implementation does not currently represent the full physics of sea ice. In particular, the simple-cubic central-force network is directionally biased; the present contact model has no tangential Mindlin history; particles have no rotational degrees of freedom; and the prescribed wedge is not a dynamic vessel. Hydrodynamics, free-surface effects, buoyancy, floe motion, ship heave/pitch, and a calibrated constitutive law are outside the current model.

All reported dimensional quantities should use SI units. Every plotted quantity should identify its definition, sign convention, coordinate, units and normalization. For example, the bow benchmark defines positive ice resistance as the non-negative component opposing prescribed forward motion; UCS engineering stress uses the mean absolute platen reaction divided by the initial gross cross-sectional area.

## 4. Software installation and execution

Run commands from a clean checkout at the repository root. Use a Python version supported by `pyproject.toml`.

### 4.1 Install

```bash
python -m pip install -e ".[test,viz]"
python -m unittest discover -s tests -v
```

The `viz` extra installs Matplotlib and Pillow for figures and GIFs. The benchmark runners use CPU execution for deterministic signatures where specified.

### 4.2 Run the 3-D indentation suite and generate particle maps

```bash
python scripts/run_dem3d_benchmark_suite.py \
  --output results-dem3d-benchmark-suite
python scripts/visualize_dem3d_benchmarks.py \
  --input results-dem3d-benchmark-suite \
  --output results-dem3d-visualizations
```

The suite writes per-case histories, reports and fracture-event CSV files. Where sampled particle states exist, the visualizer emits `damage_initial.png`, `damage_final.png`, `displacement_final.png` and `damage_evolution.gif`. Where supported fracture-event columns are present, it emits `fracture_crack_map.png`; recognized histories can produce `force_history.png`. The per-case `visualization_manifest.json` is the authoritative list of files actually produced.

### 4.3 Run specimen and ship–ice prototype cases

```bash
python scripts/run_dem3d_mechanics_benchmarks.py --output results-dem3d-mechanics
python scripts/run_dem3d_specimen_response_benchmarks.py --output results-dem3d-specimen-response
python scripts/benchmark_dem3d_ucs.py --output results-dem3d-ucs
python scripts/benchmark_dem3d_three_point_bending.py --output results-dem3d-three-point-bending
python scripts/benchmark_dem3d_moving_wedge_bow.py --output results-dem3d-moving-wedge-bow
```

### 4.4 Run sensitivity and empirical-formula baseline studies

```bash
python scripts/benchmark_dem3d_ucs_timestep_convergence.py --output results-dem3d-ucs-timestep
python scripts/benchmark_dem3d_bending_load_convergence.py --output results-dem3d-bending-load-convergence
python scripts/benchmark_dem3d_bending_resolution.py --output results-dem3d-bending-resolution
python scripts/benchmark_dem3d_wedge_resistance_sensitivity.py --output results-dem3d-wedge-sensitivity
python scripts/benchmark_dem3d_ship_ice_formula_baselines.py --output results-dem3d-mt-uikku-formulas
```

### 4.5 Assemble paper figures from the output folders

After running the desired cases, run the comparison post-processor:

```bash
python scripts/plot_dem3d_benchmark_paper.py \
  --input-root . \
  --output results-dem3d-paper-figures
```

The post-processor is intentionally data-driven: a figure is generated only when its input CSV exists and contains the required finite numeric columns. It writes a JSON manifest with generated figures and skipped plots, and emits curve figures for the UCS stress–strain response, three-point-bending load–deflection response, moving-wedge resistance history, bending-resolution sensitivity, UCS time-step sensitivity and empirical formula/reference comparison where inputs are available. Missing data are reported rather than synthesized.

## 5. Recommended figure set and interpretation

Use the figures generated from the same archived run as the numerical tables. The standard filenames are:

| Figure file | Intended paper figure | What it can establish | What it cannot establish by itself |
|---|---|---|---|
| `damage_initial.png`, `damage_final.png` | Initial/final 3-D particle damage maps | Spatial distribution of the broken-bond fraction at sampled times | Continuum stress, fracture toughness or physical crack width |
| `displacement_final.png` | Final displacement magnitude | Particle displacement relative to the initial configuration | Stress or strain tensor field |
| `damage_evolution.gif` | Sampled damage evolution | Qualitative timing and location of damage development | Unresolved events between saved frames |
| `fracture_crack_map.png` | Broken-bond midpoint cloud grouped by failure mode | Where exported bond failures occurred | A continuous crack surface unless a separate reconstruction is defined |
| `force_history.png` | Force/reaction history for an individual case | Time evolution of the selected reaction or stress history | Agreement with an experiment unless an aligned reference series is overlaid |
| `ucs_stress_strain.png` | Dynamic platen compression curve | Numerical stress–strain response for the configured specimen | Calibrated ice compressive strength |
| `three_point_bending_load_deflection.png` | Support reaction against midspan deflection | Load-transfer and compliance trends in the current fixture | Laboratory flexural strength with the current fixed-particle support |
| `moving_wedge_resistance_time.png` | Prescribed bow resistance history | Transient resistance under a specified numerical configuration | Full-scale ship resistance without matched physics and validation |
| `bending_resolution_sensitivity.png` | Peak reaction/deflection against refinement factor | Sensitivity to the tested particle resolutions | Mesh-independent material response when stiffness is not recalibrated |
| `ucs_timestep_sensitivity.png` | Peak stress against time-step factor | Sensitivity to the selected time-step campaign | Exact convergence from comparison with only a finite set of steps |
| `formula_reference_comparison.png` | Published empirical formulas versus reference values | Accuracy of the selected formula baselines against the included reference table | TensorDEM-vs-experiment accuracy |

### 5.1 Curve comparison protocol

Before comparing two curves, document that they use the same observable definition, coordinate/sign convention, unit, geometry, loading rate, physical time interval and sampling convention. If time grids differ, interpolate only over their common interval and state the interpolation method. Do not extrapolate beyond the reference range. For experimental data, retain the source, digitization method and uncertainty.

For a reference curve (y_i^{\mathrm{ref}}) and prediction (y_i^{\mathrm{pred}}), a useful normalized error is

[
\mathrm{NRMSE}=\frac{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i^{\mathrm{pred}}-y_i^{\mathrm{ref}})^2}}
{y_{\max}^{\mathrm{ref}}-y_{\min}^{\mathrm{ref}}}.
]

If the reference range is zero or not meaningful, report the dimensional RMSE instead. Also report peak-load error, peak location, and the event timing error where relevant. Fracture counts should be compared separately because discrete crack paths can change discontinuously.

### 5.2 Figure captions and provenance

Each paper figure should be accompanied by: case ID; input manifest; code revision; solver configuration; time step and number of steps; particle/bond counts; output CSV used; plotted field definition and units; and the exact script command. Preserve the corresponding JSON report and CSV alongside the figure. The visualizer's manifest records actual outputs; it does not substitute for this metadata.

## 6. Results reporting template

Populate the following table from the generated JSON/CSV outputs for the exact revision under study. Do not fill it with estimates or values copied from a different parameter set.

| Case ID | Revision / configuration | Particles / bonds | Time step (s) | Peak load or resistance | Final broken bonds | Repeatability / sensitivity result | Evidence file |
|---|---|---:|---:|---:|---:|---|---|
| [populate from run] | [commit + manifest] | [from report] | [from report] | [value + units] | [from report] | [PASS/WARN and definition] | [report and CSV path] |

For each case, discuss (1) whether the code-level acceptance checks passed; (2) how the force and damage histories evolve; (3) sensitivity to time step, loading rate, particle resolution and boundary conditions; (4) whether a matched reference exists; and (5) the remaining uncertainty. Separate observed output from interpretation.

## 7. Verification, calibration and validation are distinct

- **Verification:** the code solves its stated discrete equations correctly. Analytic force checks, invariants, objectivity, energy bookkeeping and regression tests contribute evidence here.
- **Numerical sensitivity / convergence:** the reported observables are assessed under time-step, loading-ramp and particle-resolution changes. A small change over a limited set is evidence of insensitivity over that tested range, not a universal convergence proof.
- **Calibration:** material parameters are fitted to a declared calibration dataset with objective functions and uncertainty.
- **Validation:** calibrated parameters are evaluated against independent experiments not used in the fit, under comparable specimen or ship–ice conditions.
- **Engineering qualification:** requires broader evidence, uncertainty bounds, applicability limits and independent review.

A green CI run supports only the checks that CI actually executed. An empirical-formula baseline comparison is not a DEM validation, and a visually plausible fracture cloud is not a substitute for quantitative comparison.

## 8. Limitations and next work

Priorities before using the model for quantitative ship-ice resistance are:

1. Establish source-traceable laboratory compression, flexure, tensile/splitting and fracture-energy data, including geometry, loading rate and uncertainty.
2. Replace or quantify simple-cubic lattice directional bias; assess alternative lattice/particle arrangements and calibrate resolution-dependent parameters.
3. Add physically appropriate tangential contact, frictional history and particle rotation where required by the target phenomenon.
4. Audit a complete energy balance, including damping work, external work and numerical integration error.
5. Validate specimen response on held-out measurements before tuning ship–ice cases.
6. Compare ship–ice predictions with matched published test conditions, hull geometry, ice thickness/strength, speed and resistance definitions; add fluid/free-surface effects only with a validated coupling strategy.
7. Improve visualization with explicit crack-surface reconstruction only when its geometric assumptions are documented; never relabel particle damage as continuum stress.

## References and source traceability

- TensorDEM's own case manifests and case-specific notes under `benchmarks/dem3d/` and `docs/` define the exact software protocols and should be cited with a commit hash.
- Ji, Di and Long (2017), “DEM for sea-ice uniaxial compressive and flexural strength,” DOI: [10.1061/(ASCE)EM.1943-7889.0000996](https://doi.org/10.1061/(ASCE)EM.1943-7889.0000996). The current bending fixture cites this as context only; it does not assert that the fixture reproduces the paper's measured values.
- The MT Uikku formula-baseline protocol records its experimental reference provenance in the generated JSON report and input CSV. Cite that source directly when presenting the formula comparison.

## Reproducibility checklist

- [ ] Record the exact Git commit and Python/PyTorch/Matplotlib/Pillow versions.
- [ ] Archive the complete case manifests and command lines.
- [ ] Retain raw CSV histories, JSON reports, fracture-event CSV and sampled trajectory files.
- [ ] Generate figures from those archived outputs; do not manually edit numerical axes or values.
- [ ] State whether a figure is analytic verification, numerical regression, sensitivity, calibration or independent validation.
- [ ] Label every reference dataset and document any digitization or interpolation.
- [ ] Report missing plots and failed cases; do not silently omit them.
