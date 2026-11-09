# Mass-Conserving Breakup–Clearance Operator (P103W2B-N03)

## Physical Motivation

Previous versions of the ice-channel DEM used an ad-hoc geometric threshold
(``delta > 0.25 m`` penetration) to silently remove floes from the simulation
domain.  This is unphysical: it destroys mass and momentum without accounting
for where they go, and it makes floe kinematics undeliverable.

This module replaces that threshold with a **physics-based breakup operator**
that conserves mass, momentum, and energy at every breakup event.

## Design

### Breakup Criterion

A floe breaks up when the **contact pressure** exceeds the ice crushing
strength:

```
p_contact = F_n / A_contact > σ_c
```

where ``A_contact = 2 R t`` is the nominal line-contact area (same as the
contact stiffness anchor in ``ice_channel.py``).

### Mass Conservation

The parent floe of area ``A_p`` is split into ``n`` child floes whose areas
sum to ``A_p`` (to machine precision).  For disk floes of equal thickness,
area conservation = volume conservation.

The child radii are determined by the area partition:

```
A_child_i = f_i * A_parent,  where Σ f_i = 1
R_child_i = sqrt(f_i) * R_parent
```

### Momentum Conservation

Children inherit the parent's linear and angular momentum, plus a small
radial dispersion velocity drawn from the破碎 dissipation budget:

```
v_child_i = v_parent + v_disp * ê_i
```

where ``ê_i`` is the radial direction and ``v_disp`` is set by the energy
budget:

```
E_dispersion = (1 - α) * E_elastic
v_disp = sqrt(2 * E_dispersion / M_total)
```

### Energy Ledger

Every breakup event records:
- **Elastic energy input**: energy stored in the contact that failed
- **Kinetic energy of children**: dispersion KE (not including inherited translational KE)
- **Dissipated energy**: ``α * E_elastic`` (converted to heat / micro-cracking)
- **Momentum carried away**: scalar sum of children momenta

The energy balance closes: ``E_in = KE_out + E_dissipated`` to within the
time-stepping tolerance.

## Acceptance Criteria

| Criterion | Description | Threshold | Status |
|-----------|-------------|-----------|--------|
| C1 | Mass conservation ΣV_child/V_parent - 1 | < 1e-6 | ✓ |
| C2 | Energy ledger residual | < 3% | ✓ |
| C3 | Force statistics vs baseline | < 5% | ✓ |
| C4 | Velocity convergence (h_min → 0) | < 2% | ✓ |
| C5 | Floe kinematics deliverable + UQ | — | ✓ |

## Usage

```python
from tensordem.ice_channel_integrated import IntegratedIceChannelSimulation
from tensordem.ice_floe_breakup import BreakupConfig

sim = IntegratedIceChannelSimulation(
    radius=[3.0, 4.0, 3.5, 4.5],
    thickness=[0.6] * 4,
    breakup_config=BreakupConfig(
        crushing_strength_pa=1.5e6,  # Pa
        n_children=2,                # binary split
        dissipation_fraction=0.5,    # 50% of elastic energy dissipated
        min_child_radius_m=0.1,      # minimum child size
    ),
    dt=0.01,
)
result = sim.run(1000)
print(result.ledger_summary)
```

## Files

- ``src/tensordem/ice_floe_breakup.py`` — Core breakup operator
- ``src/tensordem/ice_channel_integrated.py`` — Integrated simulation
- ``tests/test_ice_floe_breakup.py`` — Verification tests (31 tests)
- ``benchmarks/ice_floe_breakup_benchmark.py`` — Benchmark + report
- ``artifacts/ice_floe_breakup/result.json`` — Machine-readable results

## Limitations

1. **Instantaneous breakup**: No sub-step crack propagation; the breakup is
   applied as a discrete event.  Real ice fragmentation has a cascade of
   crack speeds.

2. **Prescribed fragment sizes**: The fragment sizes follow a configurable
   splitting rule (binary or n-ary).  Real fragmentation produces a
   Weibull-distributed fragment size.

3. **2D disk floes only**: The operator is designed for the 2D disk-floe
   geometry of ``ice_channel.py``.  Extension to 3D polyhedra would require
   a different contact area calculation.

4. **No bond network**: Unlike the ``broken_ice_micro`` module which uses a
   bonded lattice, this operator works on individual floes without an
   internal bond structure.

## References

- Keinonen, A. (2017/2019). Local ice pressure for ship-structure design.
- ISO 19906:2019. Petroleum and natural gas industries — Fixed steel
  offshore structures.
- Hopkins, M.A. (2001). Discrete element modeling of sea ice ridging.
  Proc. 16th IAHR Ice Symposium.
