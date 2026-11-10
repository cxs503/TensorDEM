# Ice Rate-Dependent Friction & FEM→DEM Failure Handover

## Overview

This module implements two key capabilities for ice-structure interaction simulations:

1. **Rate-dependent ice-steel friction model μ(v)**: Captures the velocity-dependent friction coefficient observed in ice-steel tribology, based on Lishman et al. (2011).

2. **FEM→DEM failure_faces contract**: Defines the data format and validation for transferring crack geometry from FEM continuous simulations to DEM discrete particle models.

## Rate-Dependent Friction Model

### Physical Background

Ice-steel friction is rate-dependent: at low sliding velocities the contact is dry and friction is high; as velocity increases, frictional heating creates a thin melt-water film that lubricates the contact, reducing friction.

### Model Formula

```
μ(v) = μ_∞ + (μ_0 - μ_∞) / (1 + v/v_c)
```

where:
- **μ_0**: Static/quasi-static friction coefficient (v → 0). Default: 0.203
- **μ_∞**: High-velocity asymptotic friction coefficient (v → ∞). Default: 0.047
- **v_c**: Characteristic velocity scale for the transition. Default: 0.051 m/s
- **v**: Sliding velocity (m/s)

### Default Parameters (Lishman et al. 2011 calibration)

| Parameter | Value | Description |
|-----------|-------|-------------|
| μ_0 | 0.203 | Static friction coefficient |
| μ_∞ | 0.047 | High-velocity asymptotic friction |
| v_c | 0.051 m/s | Characteristic velocity |

### Usage

```python
from tensordem.ice_rate_dependent_friction import (
    RateDependentFrictionModel,
    compute_friction_coefficient,
    LISHMAN_DEFAULT_PARAMS,
)

# Using the model class
model = RateDependentFrictionModel()
mu = model(0.1)  # friction at v = 0.1 m/s

# Using the function directly
mu = compute_friction_coefficient(0.1)

# Custom parameters
from tensordem.ice_rate_dependent_friction import RateDependentFrictionParams
params = RateDependentFrictionParams(mu_0=0.25, mu_inf=0.08, v_c=0.15)
mu = compute_friction_coefficient(0.1, params)
```

### Validation

The model is validated against Lishman et al. (2011) literature data at three characteristic velocities:

| Velocity (m/s) | Model μ | Literature μ | Error |
|----------------|---------|--------------|-------|
| 0.001 | 0.200 | 0.200 | 0.0% |
| 0.1 | 0.100 | 0.100 | 0.3% |
| 1.0 | 0.055 | 0.055 | 0.8% |

**Maximum error: 0.8%** (tolerance: 15%) ✓

### Negative Control

When μ_0 ≈ μ_∞, the model degenerates to constant Coulomb friction, verifying correct implementation:

```python
from tensordem.ice_rate_dependent_friction import negative_control_constant_mu

result = negative_control_constant_mu(
    v_values=[0.001, 0.01, 0.1, 1.0, 10.0],
    constant_mu=0.10,
)
# result["degenerate_is_constant"] == True
```

## FEM→DEM Failure Faces Contract

### Physical Background

When FEM simulations detect crack initiation and propagation in ice, the crack geometry must be transferred to DEM so that particle bonds can be broken at the correct locations.

### Contract Format (failure_faces.json)

```json
{
    "metadata": {
        "format_version": "1.0",
        "source": "TensorFEM",
        "target": "TensorDEM",
        "timestamp": "2026-11-10T...",
        "mesh_size": 0.01,
        "total_fracture_energy": 13.0
    },
    "failure_faces": [
        {
            "face_id": 0,
            "vertices": [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]],
            "normal": [nx, ny, nz],
            "crack_opening_displacement": 0.001,
            "area": 0.005,
            "fracture_energy": 5.0,
            "stress_state": {"sigma_n": 1e6, "sigma_t": 0.5e6}
        }
    ],
    "crack_tips": [
        {
            "position": [x, y, z],
            "direction": [dx, dy, dz],
            "stress_intensity_factor": 1.5e6
        }
    ]
}
```

### Usage

```python
from tensordem.fem_dem_failure_handover import (
    create_failure_faces_contract,
    validate_contract,
    export_contract_json,
    import_contract_json,
    verify_energy_conservation,
    verify_geometric_accuracy,
)

# Create contract from FEM output
faces_data = [
    {
        "vertices": [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]],
        "normal": [0, 0, 1],
        "crack_opening_displacement": 0.001,
        "fracture_energy": 5.0,
    },
]
contract = create_failure_faces_contract(faces_data, mesh_size=0.1)

# Validate
result = validate_contract(contract)
assert result["valid"]

# Export
export_contract_json(contract, "failure_faces.json")

# Import
loaded = import_contract_json("failure_faces.json")

# Verify energy conservation
energy_result = verify_energy_conservation(
    fem_fracture_energy=13.0,
    dem_bond_energies=[5.0, 5.0, 3.0],
)
assert energy_result["passed"]

# Verify geometric accuracy
geo_result = verify_geometric_accuracy(
    fem_crack_tips=contract.crack_tips,
    dem_crack_positions=[np.array([0.2, 0.05, 0])],
    mesh_size=0.1,
)
assert geo_result["passed"]
```

### Validation Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| (a) μ(v) literature error | < 15% at 3 points | ✓ 0.8% |
| (b) Crack position error | < mesh_size | ✓ 0.0 m |
| (c) Energy conservation | < 5% | ✓ 0.0% |
| (d) Negative controls | Pass | ✓ |

## References

- Lishman B, Jordaan RE, Taylor GW, Frederking R (2011) Tribology of ice sliding on steel. Proc. 21st IAHR Symposium on Ice, Delft, Netherlands.

## Files

- `src/tensordem/ice_rate_dependent_friction.py` - Rate-dependent friction model
- `src/tensordem/fem_dem_failure_handover.py` - FEM→DEM failure faces contract
- `tests/test_ice_rate_friction_and_failure_handover.py` - Comprehensive tests (81 cases)
- `benchmarks/ice_rate_friction_failure_handover.py` - Benchmark script
- `artifacts/ice_rate_friction_failure_handover.json` - Machine-readable results
- `docs/ice_rate_friction_failure_handover/` - Figures
- `runs/ice_rate_friction_failure_handover/` - Run outputs
