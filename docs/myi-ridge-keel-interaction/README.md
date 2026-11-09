# Multi-year ice ridge keel interaction (TensorDEM)

Closed chain implemented in this worktree for campaign node **P33-W2-N04**:

| stage | module | what it does |
|---|---|---|
| MYI keel geometry statistics | `tensordem.myi_ridge.geometry` | truncated log-normal keel depth / porosity / block-size sampling with moment matching; plane-strain keel section reconstruction with a measured (never assumed) porosity and keel depth |
| ridge crushing DEM | `tensordem.myi_ridge.crushing` | plane-strain bonded-disc rubble layer crushed by a rigid hull plate; penalty contact + damping, regularised Coulomb friction, freeze bonds with a tensile cut-off and a cohesion that softens with plastic slip; closed energy ledger; quasi-static local damping |
| extreme-load scaling | `tensordem.myi_ridge.scaling` | `F_max = p * h_keel * L_c`, log-space two-regressor fit of `F_max ~ sigma^a h^b`, monotonicity checks, load envelope assembly |
| ice-belt structural check | `tensordem.myi_ridge.panel` | Bogner-Fox-Schmit bicubic Hermite plate element (clamped / simply supported, uniform or patch load), peak bending stress against `sigma_y / 1.15`, required thickness |

Entry points

```python
from tensordem.myi_ridge import geometry, crushing, scaling, panel

stats = geometry.MyiRidgeStatistics()
depths = geometry.sample_keel_depths(stats, 400, seed=1)
section = geometry.build_keel_section(20.0, 0.30, stats, seed=1)

result = crushing.run_crush_case(20.0, ice_strength=7.5e6, porosity=0.30,
                                 hull_velocity=2.0, advance=0.8)
print(result.pressure_peak, result.run["ledger"]["closure_of_exchanged"])

envelope = scaling.load_envelope([...], loaded_length=1.5)
check = panel.check_ice_belt(1.0e6, span_x=2.4, span_y=2.4, thickness=0.030)
```

Campaign drivers

```bash
python scripts/run_myi_ridge_campaign.py --all --jobs 9
python scripts/run_myi_ridge_campaign.py --aggregate
python scripts/myi_ridge_ledger_convergence.py
python scripts/make_myi_ridge_figures.py
python scripts/make_myi_ridge_report.py
```

Tests: `tests/test_myi_ridge_geometry.py`, `tests/test_myi_ridge_crushing.py`,
`tests/test_myi_ridge_scaling.py`, `tests/test_myi_ridge_panel.py`.

The report (physics, tables, figures, criterion verdict, limitations and the
honest failure of the strength-exponent criterion) is in
[`REPORT.md`](REPORT.md).

## Why the strength exponent fails

The plane-strain cell confines the rubble between a rigid floor, a rigid
far-field wall and the advancing plate, so the plate pressure is set by the
geometric constraint of squeezing a nearly incompressible disc assembly rather
than by the freeze-bond strength (measured `p` = 5-14 MPa, i.e. 1-2
`sigma_ice`, against 1-3 MPa local ridge pressures used in ice-belt design).
The DEM therefore orders the cases correctly (monotone in `sigma_ice`,
`h_keel`, `V`) but does not supply a design pressure, and the fitted exponent
`a` comes out near 0.2 instead of 1.  The next iteration needs an escape path
with a servoed confining stress, block comminution, or a calibration of the
pressure level against ridge indentation data.