# Smooth prescribed plane contact: dynamic refinement

This opt-in experiment addresses the impulsive boundary onset in the
[fixed-plane stress test](../continuum-boundary/README.md). Its old source,
raw evidence and 10.1258% contact-impulse refinement failure remain unchanged.
The new case changes **tool history**, not spring/contact parameters. It is a
separate smooth-motion qualification, not a retroactive pass for the stress test.

## Physical problem and accounting

The control-volume network has exactly the same 1×0.25×0.1 m geometry,
22.5 kg mass, density 900 kg/m³, E=100000 Pa, ν=1/3 axial/diagonal
coefficients, support and transverse constraints, boundary tributary areas,
and penalty coefficient `50E/L` as the fixed-plane experiment. It is still a
linear material and a prescribed rigid plane, without fracture or fluid.

The plane starts at x=1 m with **zero overlap and zero force**. It approaches
1 mm with the C2 quintic `10s³−15s⁴+6s⁵`, `s=t/0.5s`, then holds at
x=0.999 m until t=0.8 s. This one physical history is unchanged on all meshes.
The tool has prescribed motion and no actuator/mass dynamics.

Each Verlet step evaluates contact at both endpoint plane positions. Work
into the network/contact system is `mean(F_on_network) × Δplane`; the opposite
force is the reaction on the tool. The stationary-support work is zero. Raw
files include every step's mean reaction, plane increment and dt, allowing
independent reconstruction of tool work and impulse. The sum of kinetic,
network strain and contact potential energy is compared with this integrated
tool work. The energy error is **not forced to zero**: contact activation and
tool quadrature leave a reported numerical residual. Restart includes the
network state, prescribed-motion clock and complete tool ledgers.

## Actual results

Four actual runs: three spatial grids and the finest grid with half timestep.
No coefficient, penalty, physical boundary, load ramp, or mass was retuned.

| Intervals | Tool impulse (N·s) | Final displacement (m) | Max energy residual (J) |
|---|---:|---:|---:|
| 16×4 | 1.512796607 | −0.000978390514 | 8.45e−11 |
| 32×8 | 1.512777969 | −0.000978389904 | 2.95e−11 |
| 64×16 | 1.512774625 | −0.000978386849 | 9.27e−12 |
| 64×16, dt/2 | 1.512774626 | −0.000978386839 | 2.32e−12 |

The declared dynamic gate uses **impulse and final displacement**, both with
3% tolerance. Finest refinement changes are **0.000221%** and **0.000312%**;
both pass. Previous-grid impulse change is 0.001232%. The displacement change
is not monotone, so no formal spatial convergence order is claimed. Time
halving changes impulse 0.000000107% and displacement 0.000001106%.

Final tool work on the finest grid is 0.00137568247 J. Maximum energy
residual relative to that work is 6.74e−9, and decreases to 1.68e−9 at dt/2.
All four momentum residuals are below 8e−17 N·s. Each run has bitwise-identical
restart continuation. Initial peak contact force is zero; the dynamic gate
therefore does not rely on the earlier overlap-determined peak metric.

## Reproduce and audit

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/smooth_plane_study.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_smooth_plane.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m unittest discover -s tests -p test_moving_plane.py
```

`study.json` binds both backend sources and four raw field files by SHA256.
The audit reconstructs final fields, energy, mass and momentum, validates the
physical tool increments, and independently sums every step's tool work and
impulse. Two focused tests check the C2 motion endpoints, tool-work sign,
energy error and restart equality.

The result qualifies this **smooth prescribed plane and linear network** only.
The old impulsive contact patch remains failed; curved disk contact, wet
particle refinement, calibrated ice failure, moving fluid boundaries and
three-dimensional breaking ice remain unqualified. No generic physical
accuracy badge is issued.
