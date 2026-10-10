# DEM3D moving plane-platen boundaries

TensorDEM now supports optional infinite horizontal top and bottom plane walls in
the 3-D bonded-sphere solver. They use a linear normal penalty law with optional
normal damping and prescribed constant z-velocity. The defaults disable both
platens, preserving prior model behavior.

## CLI example

Compress a specimen with the upper platen moving down and the lower platen
moving up:

```bash
python -m tensordem.dem3d_cli \
  --nx 4 --ny 4 --nz 3 --steps 1000 \
  --top-platen --bottom-platen \
  --platen-stiffness 5000 --platen-damping 0.5 \
  --top-platen-velocity -0.05 \
  --bottom-platen-velocity 0.05 \
  --output results-dem3d-platens
```

Velocity is signed in the global z direction: negative is downward, positive is
upward. Gap values are measured from the initial outer particle surface. The
CSV history reports signed top and bottom platen reactions and platen contact
energy. For the current sign convention, the top platen's reaction is generally
positive z and the bottom platen's reaction generally negative z during
compression.

## Integration smoke benchmark

```bash
python scripts/benchmark_dem3d_platen_compression.py \
  --steps 50 --output results-dem3d-platen-compression
```

The runner checks that both reactions activate and that the recorded state and
energy remain finite. It emits JSON and CSV histories.

## Limitations

The planes are infinite and horizontal; there is no platen friction, tilt,
finite platen geometry, servo-controlled force mode, or automatic lateral
confinement. The current benchmark is a numerical integration smoke test, not a
standard UCS test, strength calibration, experimental validation, or an
engineering-qualified ice model. A material test still needs controlled loading
rate, boundary-condition verification, mesh/particle-resolution studies, and
independent reference data.
