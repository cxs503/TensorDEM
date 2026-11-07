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

## Increment 2.5 — FEM failure surface to DEM ice field generation (SI-2/DEM)

This increment bridges the FEM cohesive failure analysis (SI-2/FEM leg, TensorFEM) and the DEM fracture simulation. It provides initialization capability to generate bonded-particle ice fields from FEM failure surfaces.

### Position in roadmap

This capability sits between step 3 (shear fracture criterion) and step 8 (coupling) in the verification sequence. It enables the DEM side to accept pre-computed fracture patterns from FEM, rather than relying solely on in-situ bond breakage during simulation.

### Input contract (frozen)

The input format is locked for parallel development with TensorFEM SI-2/FEM leg:

```json
{
  "faces": [
    {
      "id": 0,
      "area": 0.01,
      "normal": [0.0, 0.0, 1.0],
      "centroid": [0.5, 0.5, 0.3],
      "G": 100.0,
      "released_energy": 1.0
    }
  ],
  "units": "SI"
}
```

Fields:
- `id`: integer face identifier
- `area`: face area (m²)
- `normal`: unit normal vector [x, y, z]
- `centroid`: face centroid coordinates [x, y, z] (m)
- `G`: fracture energy release rate (J/m²)
- `released_energy`: total energy released by this face (J)

### Implementation

Module: `src/tensordem/fracture_gen.py`

Key components:
- `load_failure_faces(path)`: Parse failure_faces.json
- `generate_particle_lattice(domain, radius)`: Generate simple cubic lattice
- `build_bond_topology(positions, radius)`: Build nearest-neighbor bond network
- `identify_bonds_crossing_face(bonds, positions, face)`: Detect bonds intersecting failure planes
- `generate_fracture_field(faces, config)`: Complete pipeline from faces to DEM field
- `FractureLedger`: Mass and energy conservation accounting

### Mass mapping decision (frozen in prereg)

**Choice**: Equal-volume spheres

Each DEM particle has mass = density × (4/3 π r³). Total DEM mass = N_particles × particle_mass.

**Rationale**: This mapping preserves physical meaning in DEM where particles are discrete spheres. It is consistent with the existing IceDEM3D mass property and allows direct comparison with FEM continuum mass when the domain is fully discretized.

Alternative considered: unit material volume equal division (each FEM element's mass divided equally among its constituent particles). Rejected because it breaks the spherical particle assumption and complicates energy accounting.

### Acceptance criteria

1. **Mass conservation** (tolerance 1e-12 kg): |DEM_mass - FEM_mass| < 1e-12
2. **Energy accounting** (tolerance 1e-12 J): ∑released_energy maps to broken_bond_count × unit_bond_energy
3. **Bond breakage at faces**: All bonds crossing failure faces are broken; intact region bonds remain unbroken
4. **Analytical enumeration**: Given face count/area, closed-form particle count and total mass match exactly

### Verification

Benchmark script: `benchmarks/fracture_gen_ledger.py`

Generates `fracture_gen_ledger_report.json` with machine-readable results for all criteria
(`validation_report.json` belongs to the ice-floe physics benchmark).

Unit tests: `tests/test_fracture_gen.py`

Covers JSON parsing, lattice generation, bond topology, face intersection detection, mass/energy conservation, and integration with IceDEM3D initial state format.

### Integration with IceDEM3D

The generated `FractureField` can be converted to IceDEM3D initial state via `to_icedem3d_initial_state()`, returning a dict with keys: `positions`, `velocities`, `bond_indices`, `rest_lengths`, `alive`. This allows direct feeding into `IceDEM3D.step(external_forces=...)` for continued simulation with pre-broken bonds.

### Known limits

- Current implementation defaults to a simple cubic lattice; hexagonal close
  packing (HCP) is now supported through ``FractureGenConfig.packing="hcp"``
  (see ``docs/dem3d_fracture_hcp_packing.md``). Random / polydisperse packing
  is still not supported.
- Bond breakage detection uses plane intersection; does not handle curved failure surfaces
- Mass mapping assumes uniform density; does not account for FEM element-level density variations
- Energy accounting is bookkeeping only; does not modify bond stiffness or failure criteria in subsequent simulation

## Increment 3 — rigid multi-sphere (clump) particles, IN14-D3

Status: implemented and verified for spheres; polyhedral particles declared only.

`src/tensordem/clump.py` adds rigid clumps: a set of overlapping spheres glued
into one non-deformable body with union mass properties (overlap counted once),
validation (radius, density, overlap limit, connectivity, triple-overlap
rejection) and an independent voxel cross-check of the union geometry.
`src/tensordem/clump_dynamics.py` adds the six-DOF rigid body with
member-summed clump-wall and clump-sphere contacts. Evidence and numbers:
`docs/clump_multisphere_particles.md`, `artifacts/m4950/`.

Verification baseline (dumbbell, r = 0.05 m, rho = 900 kg/m^3):
union inertia vs the closed-form dumbbell value 2.3e-16 relative; free-fall energy
drift 1.4e-12 relative with zero rotation; inclined-plane rolling acceleration
1.216777535 vs 1.216777588 m/s^2 analytic (4.4e-8); two-member static contact
carries the weight to 0.0 relative error; clump-sphere third-law residual 0.0 N;
`tests/test_clump_multisphere.py` 36 passed.

### Roadmap: polyhedral particles (declared, not implemented)

1. **P1** convex hull mass properties (tetrahedralisation; verified against a
   dense sphere clump as the member count grows).
2. **P2** polyhedron-plane and polyhedron-sphere contact (support function /
   closest point; weight balance and torque-vanishing checks).
3. **P3** polyhedron-polyhedron narrow phase (GJK overlap + SAT/clipping manifold;
   face-face / edge-edge / vertex-face reference cases and A-B symmetry).
4. **P4** polyhedra as clump templates vs true hull contact, to quantify the
   multi-sphere approximation error (target < 5 % peak force / penetration).

Known limits of the implemented clump path: no clump-clump contact, triple member
overlaps rejected, restitution calibrated to 0.014-0.063 absolute, tumbling
geometries can lock in history-based Coulomb friction, no rolling resistance or
torsion friction, uniform density only. See
`docs/clump_multisphere_particles.md` §5 for details.
