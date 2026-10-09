# 2-D indentation benchmark: reproducible verification protocol

This benchmark is a regression case for the current research prototype, not a
calibrated ice-material validation or a ship-scale prediction.

## Fixed baseline

- Model: 2-D bonded circular particles, central-force bonds and post-break contact.
- Grid: nx=7, ny=3.
- Tool speed: 0.2 m/s.
- Axial break strain: 0.008.
- Objective shear threshold: 0.02.
- Integrator: semi-implicit Euler with the solver conservative default time step.
- Duration: 160 steps. Record diagnostics every 10 steps and at the final step.

## Required checks

1. Two identical runs produce bitwise-identical positions, velocities, bond states
   and sampled diagnostics on the same software/device.
2. Sampled time, tool reaction and kinetic energy remain finite.
3. Broken-bond count never decreases (damage is irreversible).
4. Time-step refinement is evaluated separately by the timestep-refinement test;
   deterministic regression alone does not establish convergence or physical validity.

## Interpretation and next validation gates

- reaction_y is the vertical ice-on-tool reaction, positive upward.
- Compare reaction histories only after matching geometry, boundary conditions,
  time step and material parameters.
- Before quantitative use, perform a lattice-sensitivity study, calibrate stiffness
  and failure thresholds against material data, and validate force-displacement
  response against an independent experiment or reference.
- The central-force square lattice has directional bias and no particle rotation,
  explicit shear-bond moments, tangential friction, crushing, gravity, buoyancy or
  fluid coupling. This benchmark cannot validate those missing mechanisms.