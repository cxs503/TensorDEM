# Agent instructions

Before making changes, read the [Tensor ecosystem vision](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md) and this repository's [project charter](docs/PROJECT_CHARTER.md). These documents define ownership boundaries; the source code and tests define what is actually implemented.

## Required development behavior

- State the goal, why it belongs in this repository, and which public or cross-repository contracts it affects.
- Preserve the repository boundary: TensorDEM owns discrete-element particle, bond, contact, and supported fracture methods. Read docs/PROJECT_CHARTER.md and the ecosystem vision. Keep coupled workflow coordination in TensorPhysics and generic runtime services in TensorPlatform. Test limiting cases, forces/energy, time-step sensitivity and reference evidence as appropriate; an animation is not validation.
- Inspect relevant code, tests, recent changes, and existing benchmark records before editing.
- Prefer stable, versioned public interfaces; avoid private cross-repository imports and duplicated sources of truth.
- Include runnable tests/examples and quantitative evidence appropriate to the change. Report exact commands, results, failures, untested paths, assumptions, and limitations.
- Distinguish implementation, unit tests, integration tests, benchmarks, domain validation, and production deployment evidence. Never claim more than the evidence establishes.
- Keep changes scoped, reviewable, and reversible. Do not optimize for PR count, code volume, or workflow completion alone.
