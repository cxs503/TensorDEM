# Prescribed polyline hull contact prototype

TensorDEM currently uses a circular indenter moving vertically. A 2-D icebreaker
prototype needs contact against a prescribed hull outline. This increment adds
a standalone PyTorch penalty-contact kernel for a polyline profile; it is not yet
wired into IceDEM.step or the CLI.

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
