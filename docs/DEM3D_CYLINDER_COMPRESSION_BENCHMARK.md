# DEM3D circular-cylinder diametral compression benchmark

## Purpose and model

This benchmark exercises a non-rectangular specimen, compact particle indexing,
irreversible bond failure, moving loading platens, restartable raw output, and a
reproducible force-displacement export in one run. The specimen is a finite-thickness
circular cylinder: its cross-section is circular in the x-z plane and it spans
`ny` particle layers in y. Opposed planar platens load it along z.

This is a **solver integration / regression benchmark**, not yet a validated
Brazilian-disc material test. The present lattice bond law is not calibrated to
an experimentally measured ice constitutive response. Do not interpret the
reported force as a validated ice strength.

## Run

Install the package and optional visualization/test dependencies if needed:

```bash
python -m pip install -e '.[test,viz]'
python scripts/run_dem3d_cylinder_benchmark.py --output results-cylinder
```

A quick smoke run:

```bash
python scripts/run_dem3d_cylinder_benchmark.py --steps 100 --save-every 10 --output results-cylinder-smoke
```

Increase `nx` and `nz` together for a finer cross-section and `ny` for a
thicker specimen. The cost grows with particle/bond count; perform time-step and
resolution studies before interpreting peak load.

## Geometry and numerical checks

- `--geometry cylinder_xz` retains only lattice sites inside a circular x-z mask.
- The retained nodes are compactly re-indexed; no bond may point to a removed node.
- Bonds are created only between retained sites that were neighbors on the original
  26-neighbor lattice.
- `--top-platen` and `--bottom-platen` apply opposed prescribed velocities.
- The solver records force, energy, broken-bond, and time-history diagnostics.

## Outputs

- `history.csv`: solver diagnostics at saved times.
- `load_displacement.csv`: time, imposed platen closure, top/bottom reaction
  magnitudes, and broken-bond count.
- `checkpoint_3d.pt`: restart checkpoint.
- `trajectory_3d.pt`: saved particle/bond trajectory and fracture history.
- `benchmark_summary.json`: parameters, summary metrics, units, and the
  explicit validation limitation.

## Acceptance checks

For an automated regression, verify finite state and reaction histories, compact
valid bond indices, initial force balance, and checkpoint round-trip identity.
For a physical validation study, additionally report at least three resolutions,
a stable time-step refinement, calibrated elastic response, peak load and failure
displacement against a named experiment/reference, and a crack-path comparison.
