# Prescribed polyline hull contact prototype

TensorDEM currently uses a circular indenter moving vertically. A 2-D icebreaker
prototype needs contact against a prescribed hull outline. This increment adds
a PyTorch penalty-contact kernel for a polyline profile. The kernel is now wired
into the IceDEM Python API through optional constructor arguments; the CLI still
uses the original circular indenter.

The API accepts particle positions and velocities, hull vertices already
translated into global coordinates for the current time, and the prescribed
hull velocity. It returns per-particle contact forces and the equal/opposite
hull reaction. Units are SI. Each disk contacts the nearest point on the whole
polyline, including segment endpoints. The implementation rejects non-finite
inputs and zero-length segments.

This does not yet digitize the USCGC Glacier lines plan, include friction,
hull flexibility, water loads, or claim calibrated resistance. Next steps are
to integrate this kernel into the DEM force balance, add a prescribed horizontal
motion driver and reaction history, then build and verify a traceable bow profile.


## IceDEM integration

Pass a local polyline and a prescribed translation to IceDEM. The vertices are
relative to the hull reference point; hull_start is its initial global position,
and hull_velocity is constant in m/s.

```python
profile = torch.tensor([[0.0, -0.1], [0.0, 0.2]], dtype=torch.float64)
sim = IceDEM(
    DEMConfig(tool_speed=0.2),
    hull_profile=profile,
    hull_start=(-0.3, 0.0),
    hull_velocity=(0.2, 0.0),
)
sim.step()
```

When a hull profile is provided, it replaces the circular indenter contact.
The diagnostics tool position denotes the translating hull reference point.
Choose DEMConfig.tool_speed consistently with the prescribed hull speed because
the conservative time-step estimate currently uses tool_speed. This API does
not yet expose the hull profile in the command-line interface or write an
explicit hull-x column to the CSV history.

## Discretization error (verified)

The kernel is a polygon approximation of the hull, so its accuracy is set by the
chord sagitta `R (1 - cos(pi/M))`. Verified against an exact circular hull in
`docs/polyline_hull_contact_verification.md`: the force-magnitude error equals
`sagitta / penetration` (second order in `M`, contact lost entirely once the
sagitta exceeds the penetration) while the force direction is only first order
(`180/M` degrees). For a 5 m bow radius and 10 mm penetration, less than 50
segments lose contact over most of the bow; 143 segments are needed for a 5 %
force error.
