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
3. **Shear law (next mechanics increment):** introduce an objective bond shear measure and rotational degrees of freedom/torques, then test pure shear separately from rigid-body rotation. Do not approximate shear failure with a fixed global-axis displacement test; that would incorrectly damage a rigidly rotated lattice.
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

This increment does **not** implement shear bond stiffness/failure yet. The current central-force bonds only fail in tension; the shear model is deliberately the next mechanics increment because it needs an objective formulation consistent with rigid-body rotation, rather than a coordinate-dependent shortcut. Rotational DOFs, shear bond forces/torques, contact friction, calibrated material parameters and 3-D geometry remain future work.
