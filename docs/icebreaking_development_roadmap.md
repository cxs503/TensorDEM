# TensorDEM ice-breaking development roadmap

## Selected direction

- Fracture model: bonded-particle ice with tensile/shear failure and post-break contact.
- Coupling strategy: validate DEM in isolation first, then connect to TensorLBM incrementally.
- Current implementation is a 2-D indentation prototype. It is not yet a ship-bow transit solver or an engineering-calibrated ice-strength model.

## Increment 1 — stable DEM-to-coupling boundary

This increment adds a minimal external nodal-load contract without coupling TensorLBM prematurely:

- IceDEM.forces(external_forces=None) accepts an optional (N, 2) PyTorch tensor of instantaneous nodal forces in newtons, in global x/y coordinates.
- IceDEM.step(external_forces=None) applies those loads for one time step. Loads are supplied afresh each step; the solver does not retain or accumulate them.
- Inputs are checked for shape and finite values and moved to the simulation device/dtype.
- diagnostics() reports the net fixed-boundary support reaction in addition to the existing ice-on-tool reaction. The support reaction is the negative of the unconstrained force residual on fixed particles.
- The existing tool reaction sign convention remains unchanged.

TensorLBM adapter code should later map fluid traction to particle nodal forces, make the force transfer conservative, and document the sampling time and coordinate frame. Do not pass pressure directly as a force; integrate traction over the represented surface/area first.

## Verification sequence

1. **Kinematics and equilibrium:** rigid translation/rotation, undeformed lattice and force balance.
2. **Bond law:** axial tension, compression, irreversible failure, then contact-only response after failure.
3. **Shear fracture criterion (implemented in this increment):** reconstruct local deformation gradients from the initial bonded neighborhood, calculate Green-Lagrange strain, and irreversibly break bonds when the equivalent in-plane shear strain exceeds shear_breaking_strain. Tests cover affine shear and rigid-body rotation. This adds a shear failure criterion, not explicit shear-bond forces/torques.
4. **Boundaries:** check fixed-node positions and report support reactions for global force balance.
5. **External loads:** verify load additivity, shape/finiteness validation and one-step response.
6. **Numerical stability:** repeat representative tests with smaller time steps and compare force histories and broken-bond counts.
7. **Scale:** replace all-pairs interaction detection with a neighbor list before increasing ice-field size.
8. **Coupling:** run DEM-only tests first; then one-way fluid loading; then conservative two-way coupling with TensorLBM.

## Acceptance criteria for this increment

- Existing DEM tests remain compatible with the optional argument.
- A zero external-load tensor gives the same forces as omitting the argument.
- A finite nodal load changes the corresponding particle acceleration in the expected direction.
- Invalid shape, non-tensor input and non-finite loads fail clearly.
- Fixed boundary particles remain exactly fixed and support-reaction diagnostics are available.

## Known limits

This increment implements objective shear-triggered bond failure but does **not** implement explicit shear-bond stiffness, particle rotations/torques or contact friction. The existing force law remains central-force; shear failure is inferred from the local Green-Lagrange strain field. The criterion requires calibration against material tests. These limitations, calibrated material parameters and 3-D geometry remain future work.
