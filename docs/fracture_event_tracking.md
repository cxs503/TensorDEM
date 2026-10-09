# Per-bond fracture event tracking

TensorDEM now keeps a first-failure ledger for each pair in the 2-D bonded-disk lattice. The ledger is intended for reproducible post-processing of crack initiation and fragmentation, not as a substitute for a calibrated fracture law.

## Event fields

`IceDEM.fracture_events()` returns events ordered by pair index. Each event includes:

- particle IDs and pair index;
- failure time in seconds;
- failure mode: `tensile`, `shear`, or `mixed`;
- current midpoint in metres;
- axial extension at the failure evaluation in metres;
- spring energy removed at failure in joules.

The failure mode records which configured threshold(s) were crossed on the force evaluation that first broke the bond. A bond is recorded only once. Repeated force evaluations do not duplicate events or add the same energy twice.

## Restart behavior

The per-pair failure mode, time, extension, and event energy are serialized in the restart snapshot. Restart loading validates array shapes, finite values, allowed modes, and consistency between the failure ledger and the surviving-bond topology. Corrupted or internally inconsistent ledgers are rejected.

## Energy interpretation and limits

The event energy is the actual elastic energy of the central spring removed when that bond failed. For `CalibratedIceDEM`, it uses the actual per-bond spring coefficient. It is **not** a material fracture toughness (G_c), and it does not establish mesh-independent fracture energy. The current model remains translational-only: it has no particle rotation DOFs, bending moments, or calibrated mixed-mode cohesive law. Event positions are reported at the time of query, so after subsequent motion they are current bond midpoints rather than immutable crack-initiation coordinates; the failure time, mode, extension, and energy are retained from the event itself.

## Example

```python
from tensordem import DEMConfig, IceDEM

sim = IceDEM(DEMConfig(nx=9, ny=5))
# Advance the simulation with sim.step(...).
events = sim.fracture_events()
for event in events:
    print(event["time_s"], event["mode"], event["released_spring_energy_J"])
```
