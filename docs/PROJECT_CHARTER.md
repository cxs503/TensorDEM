# TensorDEM — discrete-element solver charter

**Canonical ecosystem vision:** [Tensor Ecosystem: Vision for the AI Era](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md)

## Mission
Provide independently testable discrete-element methods, including particle/bond/contact dynamics and supported fracture behavior. TensorDEM is a specialist numerical capability, not the complete icebreaker application or the general-purpose agent platform.

## Owns
- Particle, bond, contact, breakage, integration and DEM-specific physical/numerical models.
- DEM inputs/outputs, units, diagnostics, force/energy checks, parameter provenance and solver-local benchmarks.
- Reproducible standalone examples before coupling to other solvers.

## Does not own
- Overall research planning, generic agent orchestration, or shared platform infrastructure.
- Other solvers' numerical kernels or all coupled-workflow acceptance.
- Claims of validated fracture physics based only on a plausible animation.

## Integration
TensorPhysics may coordinate DEM with fluid/structure solvers after clock, load mapping, units, and conservation contracts are tested. Publish run metadata and artifacts through documented interfaces; TensorPlatform owns shared domain-neutral runtime and provenance services.

## Quality gates
Check limiting cases, force/contact behavior, energy and momentum accounting where applicable, time-step sensitivity, fracture/contact invariants, and comparison with trusted benchmark or experimental evidence. Label calibration and unvalidated coupled models clearly.

## Agent rules
Before implementing a substantial change, an agent must:

1. Read this charter, the README, relevant code/tests, and recent PRs.
2. Explain why the change belongs in TensorDEM and identify affected cross-repository contracts.
3. Prefer stable, versioned public interfaces over private imports or copied logic.
4. Include executable tests and a reproducible example or quantitative benchmark appropriate to the change.
5. Report exactly what was run, what passed or failed, and what was not tested.
6. Distinguish implementation, unit-test, integration, benchmark, validation, and production evidence.
7. Never claim scientific or engineering validation unless relevant domain evidence supports it.
8. Keep general platform features in TensorPlatform and cross-solver research orchestration in TensorPhysics.
9. Keep changes scoped, reviewable, and reversible. PR count and workflow completion are not quality measures.

## Maturity
This charter states intended responsibility; it does not claim that the coupled icebreaking model is validated.
