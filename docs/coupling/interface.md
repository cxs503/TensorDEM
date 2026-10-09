# Auditable 2-D DEM coupling interface

`IceDEM.exchange_state()` reports SI global x/y positions, velocities, physical
extruded disk masses, fixed-node mask, radius, thickness, time and timestep.
Forces have shape `(N,2)` in N. `step(external_forces=..., load_time=sim.time)`
holds these forces constant over the next DEM step; stale timestamps fail.
Moments use `x*Fy-y*Fx` about the global origin, counterclockwise positive.
There are no particle rotational DOFs or transferred couple stresses.

`last_step` records start/end time, tool reaction (ice on tool), support force
(on ice), external resultant and moment, and external work `F dot (x1-x0)`.
`accounting` accumulates each sampled force impulse and prescribed-tool work.
Damping loss uses actual clipped dashpot force relative to the elastic force;
quadrature is left endpoint over each actual timestep. Fracture release is
spring potential removed on the first failure only, not a calibrated fracture
energy. Calling `forces` retains legacy irreversible failure updates; repeated
calls on the same state never recount release. `mechanical_energy()` and
`exchange_state()` do not update failure. Any initial prestrain failure belongs
in the initial accounting baseline.

Report numerical energy residual as:

```
E_final - E_initial + (damping_final-damping_initial)
 + (release_final-release_initial) - (external_work_final-external_work_initial)
 - (tool_work_final-tool_work_initial)
```

Semiimplicit Euler is first order. This residual is not forced to zero, and
its timestep sensitivity must be measured. Forces applied at fixed nodes
have zero mechanical work and are balanced by recorded supports. Linear drag
is an external momentum sink, recorded separately as `drag_impulse_Ns`.
Sum tool/support/external/drag impulses when auditing linear momentum.

`fragments()` gives connected surviving-bond components, IDs, retained mass,
centroid and mean velocity. This is connectivity, not calibrated ice fracture.
`IceDEM.from_snapshot(sim.snapshot())` reconstructs the complete reference
lattice, physical state, broken bonds, tool/hull motion, timestep, configuration
and accounting. Serialization is JSON compatible; unknown/missing fields,
nonfinite arrays, topology mismatches and inconsistent fixed states fail closed.
The caller must also checkpoint its fluid populations and coupling clock.

Tests cover deterministic restarted trajectories, load force/moment/work,
constraint impulses, nonduplicate failure release and analytic uniform-load
first-order timestep refinement. This is a 2-D central-force prototype;
three-dimensional ice, tangential friction, bending bonds and calibrated sea
ice strength remain outside its verified capabilities.
