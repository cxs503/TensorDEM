# 3-D DEM energy audit and reproducible indenter benchmark

This adds an opt-in audit layer without changing the base `IceDEM3D` API or the
existing `tensordem-3d` CLI. It is intended for verification and regression work.

## Run the benchmark

From a checkout with the package dependencies installed:

```bash
python scripts/benchmark_dem3d.py --steps 400 --sample-every 10 --verify-repeat
```

The script writes:

- `history_3d.csv`: sampled reactions, damage, mechanical energy, cumulative work,
  dissipation and the energy-balance residual.
- `summary_3d.json`: fixed configuration, peak absolute vertical reaction,
  final energy ledger and a SHA-256 signature.
- `final_checkpoint_3d.pt`: restart-capable solver state including the audit ledger.
- `fracture_events_3d.csv`: one row per broken bond, with stable bond/particle
  indices, initial and final bond midpoints and lengths, failure mode (tensile,
  shear, or mixed), failure time, extension at detection, and estimated release energy.

The repeat check runs the same CPU case twice and requires matching signatures.
It deliberately avoids `Tensor.numpy()`, so it can run in minimal PyTorch CI
environments without the NumPy bridge.

## Energy ledger

The audit subclass tracks:

- indenter work, using the ice-on-tool reaction and prescribed tool velocity;
- work by user-supplied nodal forces;
- drag, particle-contact damping and tool-contact damping estimates;
- the solver's existing cumulative bond-release energy estimate;
- mechanical energy (kinetic plus surviving-bond, particle-contact and tool-contact
  elastic energy), and a signed balance residual.

The residual is computed as

```text
E_mechanical - E_reference - W_tool - W_external
+ D_drag + D_particle + D_tool + E_fracture_release
```

A nonzero residual is expected: the solver uses semi-implicit Euler, contact damping
and work are integrated discretely, and the fracture term is an estimate based on
the bond extension at detection. This ledger is an engineering diagnostic, not an
exact energy-conservation proof. It excludes gravity, rotational degrees of freedom,
thermal energy, and any constitutive damage energy not represented by the current
bond-release estimate.

## Validation limits

The baseline is a reproducible regression case, not calibrated ice-material
validation or a ship-scale prediction. Before quantitative use, perform time-step
and lattice-sensitivity studies, calibrate bond/contact parameters against material
data, and compare load-displacement and fracture patterns with an independent
experiment or reference solution.

## Fracture event traceability

`fracture_events_3d.csv` is an event table rather than a sampled damage curve.
It contains only bonds that have broken by the end of the run; an intact run still
writes the header so downstream tooling can rely on a stable schema. `bond_id`
is the deterministic index in the solver's sorted bond topology, while
`particle_i` and `particle_j` identify its endpoints. Failure time, extension,
and energy are the values recorded when the irreversible failure was detected;
the final midpoint and length describe the post-run geometry and are not the
failure geometry. The failure mode code is 1=tensile, 2=shear, and 3=mixed.

This output supports crack-front reconstruction and debugging, but a broken-bond
network is not by itself a calibrated macroscopic fracture model.
