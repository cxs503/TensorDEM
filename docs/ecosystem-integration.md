# Tensor ecosystem integration

TensorDEM is the specialist discrete-element layer for bonded particles, irreversible bond failure, fragment contact and DEM diagnostics. Model scope and limitations must remain visible in coupled engineering applications.

## Responsibilities

- Own particle state, bond/contact laws, time integration, fracture events and restart/checkpoint behavior.
- Maintain DEM-native CPU/GPU execution, particle/fragment diagnostics and solver-specific benchmarks.
- Define external-load inputs and document reactions, energies and fracture outputs.
- Distinguish a research prototype from a verified component and a physically validated ice model.

## Boundaries

- **TensorPhysics** coordinates DEM stepping, subcycling, coupling clocks, run manifests and artifact provenance; it must not duplicate DEM kernels.
- **TensorPlatform** owns shared dataset catalog, lineage, access control and AI/model lifecycle.
- **TensorFEM** may represent hull/structure response; define contact ownership and energy/work accounting to prevent duplicated or missing interactions.
- **TensorLBM** or **TensorFVM** may supply fluid loads only through explicit mapping adapters with documented units, coordinate frames and conservation checks.

## External-load contract

External loads must declare shape, units, coordinate frame, physical time and whether values are forces or tractions. State whether loads are instantaneous or accumulated; the solver must not silently cache or double-apply them. Validate resultant force and moment, time-step/subcycling alignment, work/energy accounting and restart continuity.

## Integration path

1. Preserve standalone DEM tests and benchmark cases.
2. Publish a versioned adapter for configuration, stepping, diagnostics, restart and artifact export.
3. Integrate through TensorPhysics with explicit clocks and deterministic run manifests.
4. Register trajectories, fracture summaries and validation metadata with TensorPlatform.
5. Do not interpret prototype ice-breakage or prescribed-hull results as validated ship resistance without calibration, resolution/time-step studies and experimental comparison.

## Repository map

- [TensorPlatform](https://github.com/cxs503/TensorPlatform) — shared data and AI services.
- [TensorPhysics](https://github.com/cxs503/TensorPhysics) — scientific workflow orchestration and multiphysics coordination.
- [TensorLBM](https://github.com/cxs503/TensorLBM) — lattice Boltzmann solver.
- [TensorFVM](https://github.com/cxs503/TensorFVM) — finite-volume solver.
- [TensorFEM](https://github.com/cxs503/TensorFEM) — finite-element solver.
- [TensorDEM](https://github.com/cxs503/TensorDEM) — discrete-element solver.
