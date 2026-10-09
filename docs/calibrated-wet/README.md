# Material-consistent wet DEM sensitivity

This opt-in driver executes `CalibratedIceDEM` with actual per-bond coefficients
inside the unchanged TensorLBM `CoupledIce2D.step`. It preserves the previous
uncalibrated driver and published evidence. `E=1000 Pa`, realizable central-force
`nu=1/3`, thickness 0.2 m and mass 12.3795 kg are fixed. This intentionally soft
material and fluid viscosity 0.02 m²/s are numerical fixture parameters, not
measured ice/seawater properties.

The explicit global-fluid embedding translates local y by the particle radius;
all refinements have disk exterior y in [0.5,0.65] m and x in [0.375,0.825] m.
This embedding is reconstructed on restart. Disk outer envelope is 0.45 × 0.15 m for 6×2, 12×4 and 18×6 resolutions. Each
case compensates homogeneous disk density to keep total mass fixed; disk area
is not a nonoverlapping continuum control volume. Fixed first/last center
columns move inward by a radius; the center-spring material span changes.
Contact stiffness 150 N/m and damping 0.1 N·s/m are fixed but contact point count
changes. Therefore the refinement is a **combined geometry/contact/material
sensitivity**, not a clean continuum boundary-value convergence test.

Three wet intact cases, three dry intact cases, a wet timestep-half case
(exchange interval held fixed), and one longer fracture-threshold case execute
actual dynamics. The intact cases run 0.03 s; the fracture case runs 0.15 s.
The per-bond elastic coefficients, populations, held interpolation/spreading
map, force sample clock, impulses, work and full DEM state are saved in each
JSON restart. Subcycling is recomputed from actual mass and the conservative
coefficient upper bound. All eight midway restart trajectories match bitwise.

Results:

| Wet case | Peak tool force N | Tool impulse N·s | Broken bonds |
|---|---:|---:|---:|
| 6×2 | 0.796498 | 0.0183423 | 0 |
| 12×4 | 1.837948 | 0.0323423 | 0 |
| 18×6 | 1.964736 | 0.0374048 | 0 |
| 12×4 timestep half | 1.850730 | 0.0324953 | 0 |
| 12×4 fracture, longer duration | 7.170310 | 0.5929955 | 51 |

12×4 → 18×6 peak changes 6.898%; impulse changes 15.653%, failing the 3%
sensitivity gate. Time halving changes peak 0.6954% and impulse 0.4729%.
Dry refinement also varies materially, so interface errors alone do not explain
the discrepancy. No thresholds are retuned to force a pass. The fracture case
uses inherited uncalibrated strain thresholds and is not a fracture convergence
or material qualification result.

Maximum momentum residual is below 3.9e-12 kg·m/s. Nonzero final DEM energy
residuals are reported, including 6.91e-4 J in the fracture fixture; interface
work error is separate. The point-friction fluid interface is permeable and
contains liquid inside particles. There is no free surface, buoyancy, particle
rotation, bending moment, 3-D geometry or resolved pressure boundary here.
Physical accuracy remains false.

Reproduce from TensorDEM with a Torch environment and adjacent TensorLBM:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src:../TensorLBM/src
python scripts/calibrated_wet_study.py
python scripts/audit_calibrated_wet.py
python -m pytest -q tests/test_calibrated_wet_coupling.py tests/test_calibrated_dem.py
```

The raw audit binds source hashes, full fields and summary reconstruction. Ten
related tests cover coefficient integrity, complete held-exchange restart,
corrupt clocks/populations/subcycling and calibrated backend behavior.

Next: separate prescribed distributed-load elasticity refinement from changing
contact boundaries; introduce fixed continuum support geometry, pressure
resolved finite solid boundaries, contact scaling qualification and independent
ice fracture calibration before claiming a converged wet breaking case.

PR #12 replaces the optional NumPy tensor bridge in coefficient hashing with
explicit little-endian float64 packing. All eight full raw restart JSON hashes
remain identical after an actual replay; only the solver source hash changes.
The compatibility test disables Tensor.numpy and restores the original digest
and snapshot before taking a real dynamics step.
