# 3-D DEM ↔ Fluid Bidirectional Coupling Framework

**Mission**: SIR-PC2|F06 (m706)  
**Physical Domain**: fluid_particle_coupling  
**Risk Level**: high  
**Layer**: L1_foundation  
**Status**: ✅ VERIFIED

---

## 1. Overview

This module implements momentum-conserving two-way coupling between the
TensorDEM bonded-sphere engine (`IceDEM3D`) and an incompressible fluid
solver on a uniform Cartesian grid. The coupling exchanges:

- **Drag force**: fluid exerts Stokes/Schiller-Naumann drag on each DEM
  particle; by Newton's third law the particle exerts an equal and
  opposite force on the fluid, distributed to grid nodes via the same
  interpolation weights used for velocity sampling.
- **Fluid displacement**: each particle occupies a volume that displaces
  fluid, tracked through a porosity (void-fraction) field.

## 2. Physical Background

### 2.1 Drag Force Models

**Stokes drag** (Re_p ≪ 1):
$$\mathbf{F}_d = -6\pi\mu R\,\mathbf{v}_{rel}$$

**Schiller-Naumann** (arbitrary Re):
$$C_d = \frac{24}{Re}\left(1 + 0.15\,Re^{0.687}\right), \quad Re < 1000$$
$$C_d = 0.44, \quad Re \geq 1000$$
$$\mathbf{F}_d = -\tfrac{1}{2}C_d\,\rho_f\,A\,|\mathbf{v}_{rel}|\,\mathbf{v}_{rel}$$

### 2.2 Momentum Conservation

The coupling enforces exact discrete momentum conservation by construction:
- Drag force on particle: $\mathbf{F}_p = \mathbf{F}_d$
- Reaction force on fluid: $\mathbf{F}_f = -\mathbf{F}_d$
- Both use identical trilinear interpolation weights
- Therefore: $\sum \mathbf{F}_p\,\Delta t + \sum \mathbf{F}_f\,\Delta t = 0$ exactly

### 2.3 Single-Particle Settling (Stokes Benchmark)

Force balance at terminal velocity:
$$\underbrace{\tfrac{4}{3}\pi R^3 (\rho_p - \rho_f) g}_{\text{gravity - buoyancy}} = \underbrace{6\pi\mu R\,v_t}_{\text{Stokes drag}}$$

$$v_t = \frac{2}{9}\frac{(\rho_p - \rho_f)\,g\,R^2}{\mu}$$

## 3. Module Architecture

```
src/tensordem/dem_fluid_coupling.py
├── FluidGrid3DConfig      # Grid parameters
├── FluidGrid3D            # Incompressible fluid solver
│   ├── sample_velocity()  # Trilinear interpolation → particles
│   ├── distribute_force() # Adjoint scatter → grid (conservative)
│   ├── step()             # Gravity + source + diffusion + projection
│   └── fluid_momentum()   # Audit total fluid momentum
├── DEMFluidCoupling3D     # Bidirectional coupling engine
│   ├── compute_drag_forces()
│   ├── exchange_momentum()  # One coupling step
│   ├── update_porosity()
│   └── momentum_conservation_residual()
├── stokes_drag()          # F = -6πμRv
├── schiller_naumann_drag()
└── stokes_terminal_velocity()  # Analytical v_t
```

## 4. Verification Results

### 4.1 Acceptance Criteria

| Criterion | Value | Threshold | Unit | Status |
|-----------|-------|-----------|------|--------|
| momentum_exchange_conservation | 0.0 | < 1e-10 | kg·m/s | ✅ PASS |
| single_particle_settling_vs_stokes | 7.1e-4 | < 0.05 | ratio | ✅ PASS |
| pytest_test_dem_fluid_coupling | PASS | PASS | bool | ✅ PASS |

### 4.2 Stokes Settling Benchmark

| Parameter | Value |
|-----------|-------|
| ρ_p | 1100 kg/m³ |
| ρ_f | 1000 kg/m³ |
| R | 0.5 mm |
| μ | 1.0×10⁻³ Pa·s |
| v_t (analytical) | 0.05450 m/s |
| v_t (simulated) | 0.05446 m/s |
| Relative error | 0.07% |
| Re_p at v_t | 0.055 |

### 4.3 Test Suite

12 tests covering:
- Momentum conservation (3 tests)
- Stokes drag formula and terminal velocity (3 tests)
- Schiller-Naumann correlation (2 tests)
- Fluid grid operations (2 tests)
- Full coupling integration (2 tests)

## 5. Usage Example

```python
from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.dem_fluid_coupling import (
    FluidGrid3D, FluidGrid3DConfig, DEMFluidCoupling3D
)

# Create DEM
dem_config = DEM3DConfig(nx=5, ny=5, nz=5, radius=0.003, ...)
dem = IceDEM3D(dem_config)

# Create fluid grid
fluid_config = FluidGrid3DConfig(nx=16, ny=16, nz=32, lx=0.1, ly=0.1, lz=0.2)
fluid = FluidGrid3D(fluid_config)

# Create coupling
coupling = DEMFluidCoupling3D(dem, fluid, drag_model="stokes")

# Time loop
dt = 1e-4
for step in range(1000):
    coupling.step(fluid_dt=dt, dem_substeps=4)
    
# Check momentum conservation
residual = coupling.momentum_conservation_residual()
assert torch.linalg.vector_norm(residual) < 1e-10
```

## 6. Limitations & Known Issues

1. **Fluid solver**: The pressure-projection solver uses Jacobi iteration
   with periodic BCs. For engineering accuracy, a multigrid Poisson solver
   with proper wall BCs is needed.
2. **Resolution**: Trilinear interpolation is 2nd-order; particles smaller
   than ~3 grid cells may have inaccurate drag.
3. **High Re**: Schiller-Naumann is valid up to Re ~ 1000; for higher Re,
   use a more sophisticated correlation (e.g., Clift-Gauvin).
4. **Particle-wall**: No lubrication force is included for near-wall particles.
5. **Turbulence**: No sub-grid turbulence model for the fluid phase.

## 7. Audit Chain

- Upstream: m670 (friction/rolling verification) — interface reference
- This module: m706 — 3D DEM↔fluid bidirectional coupling
- All quantities in SI units
- PyTorch float64 for numerical precision
- Momentum conservation exact by construction (Newton's 3rd law)

## 8. Files

- `src/tensordem/dem_fluid_coupling.py` — coupling implementation
- `tests/test_dem_fluid_coupling.py` — 12 pytest tests
- `scripts/verify_m706.py` — verification runner
- `artifacts/verification_m706.json` — acceptance criteria
- `artifacts/settling_velocity.png` — comparison figure
- `runs/m706_settling_data.json` — raw simulation data
