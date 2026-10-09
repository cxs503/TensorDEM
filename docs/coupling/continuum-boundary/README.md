# Fixed physical boundary dynamic refinement

This study isolates material and contact from the disk radius and clamped-center
changes in the earlier wet DEM refinement. It does **not** repair or supersede
that wet study. The new backend is a linear, two-dimensional central-spring
network with control-volume masses; it has no fracture, rotations, fluid, or
moving tools. Existing `IceDEM` and calibrated disk dynamics are unchanged.

## Common physical problem

All three meshes occupy the same `[0,1] × [0,0.25] m` rectangle, thickness 0.1 m,
density 900 kg/m³ and total mass **22.5 kg**. Boundary coordinates remain fixed
under refinement: 16×4, 32×8 and 64×16 intervals. Vertex control-volume masses
have half/quarter boundary weights. The x=0 support constrains x motion; all
nodes constrain y motion in the dynamic longitudinal cases. Right boundary
forces use tributary surface areas summing to **0.025 m²**.

Axial spring stiffness is `3Et/4` and diagonal stiffness `3Et/8`, with half
weights for axial springs along the outer faces. This gives isotropic plane
stress `ν=1/3`, `E=100000 Pa`. The independent affine patch imposes
`u_x=εx, u_y=−εy/3`, checks recovered E, zero transverse traction, interior
force balance and analytical strain energy. Dynamic y constraints give the
longitudinal modulus `C11=9E/8`; they do not represent unrestrained Poisson
motion.

## Actual dynamic cases and results

Ten runs evolve the actual nodal network for **0.12 s** with velocity Verlet:
three step-traction runs, three fixed-plane contact runs, three separation
negative controls, and one finest contact run with timestep halved.

The step traction is 100 Pa, starting from rest. Endpoint displacement is
compared with the independently evaluated longitudinal-bar Fourier solution,
with wave speed `sqrt(C11/ρ)` and 20,000 terms. The errors are:

| Intervals | Analytic displacement error | Contact impulse (N·s) |
|---|---:|---:|
| 16×4 | 0.076531% | 0.349176 |
| 32×8 | 0.095958% | 0.302519 |
| 64×16 | 0.005608% | 0.271887 |

All load cases pass a declared 3% analytical threshold, but the errors are
**not monotone**. This is a step-load transient and no formal observed spatial
order is claimed.

Contact uses a stationary x=0.999 m plane acting only on reference right-face
vertices, with `F_i=−(50E/L) A_i max(x_i+u_i−plane,0)`. The penalty per area
and the plane coordinate are identical on all meshes. The initial 1 mm overlap
stores exactly **0.0625 J** of contact potential; this is an impulsive contact
onset patch, not a realistic collision trajectory. The separation control
places the plane at x=1.001 m and has zero forces and zero motion.

**Contact is not converged:** finest impulse changes **10.1258%** (previous
pair 13.3619%), failing 3%. Finest endpoint displacement changes about 9.02%.
Peak changes only 0.0853%, but the initial prescribed overlap determines the
peak, so that metric cannot qualify dynamic contact. Timestep halving changes
the finest impulse **0.000882%**. The remaining error is therefore not explained
by this timestep sensitivity. Next work should use a smooth prescribed tool
motion with exact moving-plane work accounting and check impulse and
traction-history convergence, retaining the impulsive onset case as a stress
test.

Maximum contact energy residual decreases from 0.0001971 to 0.00007868 J
(0.315% to 0.126% of the initial potential). External contact work is reported
separately; because the plane is stationary it equals minus the contact
potential change, and energy auditing includes the initial potential.
Dynamic-load energy residual is below 5.83e−8 J. Support and external impulses
are integrated with the same Verlet force quadrature; momentum residuals are
below 9e−16 N·s. All ten runs have bitwise-identical restart continuation.

## Evidence and reproduction

Each raw JSON contains final positions, displacements, velocities, masses,
edges and per-bond coefficients, time history, tool impulse and full restart.
`study.json` binds source and raw files by SHA256. `audit.json` independently
reconstructs final energy, momentum, mass, topology and analytic displacement;
it does not rely on qualification badges. Five focused unit tests check
material, fixed physical geometry, positive/negative contact, input rejection,
conservation and restart.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/continuum_boundary_study.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_continuum_boundary.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m unittest discover -s tests -p test_continuum_boundary.py
```

Physical accuracy remains **unqualified**. A fixed-plane linear network does
not establish curved disk contact convergence, calibrated ice fracture,
moving fluid boundaries, or three-dimensional breaking ice.
