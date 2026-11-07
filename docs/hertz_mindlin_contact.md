# Hertz–Mindlin Contact Model

## Overview

The Hertz–Mindlin contact model combines the classical Hertz normal contact law with the Mindlin tangential contact law for elastic frictional spheres. This implementation adds overlap-dependent tangential stiffness to the existing Cundall-Strack Coulomb friction framework in TensorDEM.

## Theory

### Normal Direction (Hertz 1882)

For two elastic spheres in contact:

```
F_n = (4/3) E* √(R*) δ^(3/2)
k_n = F_n/δ = (4/3) E* √(R*) √δ    (secant stiffness)
```

where:
- `E*` = reduced Young's modulus = E / [2(1-ν²)] for equal spheres
- `R*` = effective radius = R/2 for equal spheres
- `δ` = normal overlap

### Tangential Direction (Mindlin 1949)

The Mindlin no-slip tangential stiffness is:

```
k_t = 8 G* √(R* δ)
```

where `G*` is the reduced shear modulus following Johnson (Contact Mechanics, 1985, eq. 3.39):

```
1/G* = (2-ν₁)/G₁ + (2-ν₂)/G₂
```

For two equal spheres of the same material (G = E/[2(1+ν)]):

```
G* = E / [4(2-ν)(1+ν)]
```

### Stiffness Ratio

The ratio of tangential to normal stiffness for equal spheres is:

```
k_t / k_n = 6 G* / E* = 3(1-ν) / (2-ν)
```

| ν   | k_t/k_n |
|-----|---------|
| 0.0 | 1.500   |
| 0.25| 1.286   |
| 0.3 | 1.235   |
| 0.4 | 1.125   |

### Coulomb Saturation

The tangential force is limited by the Coulomb friction criterion:

```
|F_t| ≤ μ |F_n|
```

When sliding occurs, the tangential displacement state is projected onto the friction cone:

```
δ_t ← F_t / k_t
```

This ensures history-consistent sliding (Cundall-Strack projection).

## Implementation

### New Module: `tensordem/hertz_mindlin.py`

- `hertz_reduced_shear_modulus(E, ν, rigid_counterpart=False)`: Computes G*
- `hertz_mindlin_tangential_stiffness(δ, E*, G*, R*)`: Computes k_t
- `hertz_mindlin_tangential_force(...)`: Combined Mindlin stiffness + Coulomb saturation
- `mindlin_stiffness_ratio(ν)`: Returns k_t/k_n for equal spheres

### Modified: `tensordem/dem3d.py`

- Added `_hertz_g_star` property
- Updated tangential friction in `forces()` to use Mindlin stiffness when `contact_model="hertz"`
- State projection uses local k_t for correct cone projection

### Modified: `tensordem/dem3d_energy.py`

- Updated tangential energy calculation to use Mindlin stiffness for Hertz contacts

## Usage

```python
from tensordem.dem3d import DEM3DConfig, IceDEM3D

cfg = DEM3DConfig(
    nx=3, ny=3, nz=3,
    radius=0.01,
    density=1000.0,
    contact_model="hertz",       # Enable Hertz-Mindlin
    young_modulus=1e9,           # Pa
    poisson_ratio=0.3,           # 
    friction_coefficient=0.3,    # Coulomb friction
    restitution_coefficient=1.0,
    dt=1e-6,
)

sim = IceDEM3D(cfg)
# ... run simulation
```

## Verification

### Test Suite: `tests/test_hertz_mindlin.py`

19 tests covering:

1. **Mindlin stiffness ratio** (6 tests): k_t/k_n = 3(1-ν)/(2-ν) for various ν
2. **Tangential force** (3 tests): stick regime, slip regime, state projection
3. **Negative controls** (6 tests): invalid parameters rejected
4. **Integration tests** (4 tests): full DEM simulation with Hertz-Mindlin

### Verification Artifacts

- `artifacts/hertz_mindlin_stiffness_ratio.png`: k_t/k_n vs Poisson's ratio
- `artifacts/hertz_mindlin_force_displacement.png`: Force-displacement curves
- `artifacts/hertz_mindlin_report.json`: Machine-readable verification results

### Running Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_hertz_mindlin.py -v
PYTHONPATH=src python3 scripts/verify_hertz_mindlin.py
```

## Limitations

- **No clump/multi-particle shapes**: This implementation is for spherical particles only
- **No calibration tools**: Material parameter calibration is out of scope
- **Linear tangential spring**: The Mindlin model uses a linear incremental spring; full Mindlin-Deresiewicz history dependence (loading/unloading/reloading) is not implemented
- **No rolling resistance coupling**: Rolling resistance is handled separately

## References

1. Hertz, H. (1882). "Über die Berührung fester elastischer Körper". Journal für die reine und angewandte Mathematik.
2. Mindlin, R.D. (1949). "Compliance of Elastic Bodies in Contact". Journal of Applied Mechanics, 16(3), 259-268.
3. Johnson, K.L. (1985). Contact Mechanics. Cambridge University Press.
4. Cundall, P.A. & Strack, O.D.L. (1979). "A discrete numerical model for granular assemblies". Géotechnique, 29(1), 47-65.
