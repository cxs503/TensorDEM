# Polyline hull contact: verification against closed-form geometry

The kernel `tensordem.hull_contact.polyline_contact_forces` (merged with the
`IceDEM` hull-profile API) models a rigid bow as a polyline with penalty contact
against disks of radius `R_p`.  It had smoke tests only.  That left the one
question a user must answer before believing a bow load unanswered: **how many
hull points does a digitized bow need before the polyline reproduces the real
curved hull, and what error does a given digitization commit?**

This note closes that debt with an analytic reference, a measured order of
accuracy, a closed-form digitization rule, and a defect fix found along the way.
It verifies *geometry*. Nothing here validates the physical resistance of a ship
in ice: the penalty stiffness, the ice material and the contact law are not
benchmarked against experiments in this document.

Reproduce with:

```
PYTHONPATH=src python scripts/hull_contact_verification.py
PYTHONPATH=src python -m pytest tests/test_hull_contact_verification.py -q
```

Artifacts: `docs/hull-contact-verification/report.json`,
`discretization_scan.csv`, `convergence.png`, `bow_contact_geometry.png`.

## 1. Reference and case set

The analytic benchmark is a rigid **circular** hull of radius `R = 0.5 m`
with the identical penalty law (`exact_circular_hull_force`), evaluated against an
**inscribed** polygon on the same circle -- an inscribed polygon is exactly what a
lines-plan digitization gives, because the traced points lie on the hull surface
and the chords cut inside it.  The chord sagitta is

```
s(M) = R (1 - cos(pi/M)) ~ R pi^2 / (2 M^2)
```

Disks of radius `R_p = 0.05 m` (penalty stiffness
`k = 1000 N/m`) are placed at penetration depths
`delta = f * R_p` for f in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}.  Probe points are placed
*on the polygon*: one disk above every chord midpoint and one above every vertex,
plus quarter-chord points for the direction metric (4 probes
per segment). A uniform angular grid aliases the chord-midpoint extremum and
under-reports the error by up to a factor of two -- the first version of this
study made exactly that mistake and reported 12 % where the truth is 24 %.

## 2. Analytic result: the magnitude error *is* the sagitta

A disk centred above a chord midpoint sits at distance `R - s` from the polygon
instead of `R` from the circle, so its overlap drops from `delta` to `delta - s`
and

```
max |dF| / F = s(M) / delta      for s(M) < delta,
             = 1 (total contact loss)   otherwise.
```

Measured to machine precision (`test_mid_chord_error_equals_sagitta_over_penetration`,
10 decimal places), and the reported scan reproduces the law exactly:

| M | penetration δ (m) | sagitta s (m) | s/δ | max |ΔF|/F | max direction error (°) | dropout |
|---|---|---|---|---|---|---|
| 16 | 0.0250 | 0.00961 | 0.384 | 0.3843 | 5.625 | 0.00 |
| 16 | 0.0050 | 0.00961 | 1.921 | 1.0000 | 180.000 | 0.75 |
| 16 | 0.0010 | 0.00961 | 9.607 | 1.0000 | 180.000 | 0.75 |
| 32 | 0.0250 | 0.00241 | 0.096 | 0.0963 | 2.813 | 0.00 |
| 32 | 0.0050 | 0.00241 | 0.482 | 0.4815 | 2.813 | 0.00 |
| 32 | 0.0010 | 0.00241 | 2.408 | 1.0000 | 180.000 | 0.75 |
| 64 | 0.0250 | 0.00060 | 0.024 | 0.0241 | 1.406 | 0.00 |
| 64 | 0.0050 | 0.00060 | 0.120 | 0.1205 | 1.406 | 0.00 |
| 64 | 0.0010 | 0.00060 | 0.602 | 0.6023 | 1.406 | 0.00 |
| 128 | 0.0250 | 0.00015 | 0.006 | 0.0060 | 0.703 | 0.00 |
| 128 | 0.0050 | 0.00015 | 0.030 | 0.0301 | 0.703 | 0.00 |
| 128 | 0.0010 | 0.00015 | 0.151 | 0.1506 | 0.703 | 0.00 |
| 256 | 0.0250 | 0.00004 | 0.002 | 0.0015 | 0.352 | 0.00 |
| 256 | 0.0050 | 0.00004 | 0.008 | 0.0075 | 0.352 | 0.00 |
| 256 | 0.0010 | 0.00004 | 0.038 | 0.0376 | 0.352 | 0.00 |

Convergence over the smooth branch (log-log least squares): magnitude order
**-1.999** (second order) and direction order
**-1.000** (first order) for `delta = 0.5 R_p`.
The force *direction* is only first order accurate: the chord normal is tilted by
up to half the chord angle, `pi/M` rad = `180/M` degrees.  For a 20-segment bow
that is 9 deg of load-direction error, which for ship-ice work propagates directly
into the split between surge resistance and transverse load.

![convergence](convergence.png)

## 3. Digitization rule (the engineering result)

Inverting the sagitta law gives the number of segments needed for a target
relative force error `eps` at penetration `delta`:

```
M >= pi / arccos(1 - eps * delta / R),     eps * delta < R
```

and contact is retained at all only when `s(M) < delta`, i.e.
`M > pi / arccos(1 - delta/R)` (`required_segments`).  Predicted vs measured
(first grid level meeting each target):

| penetration | measured order | predicted M (5 %) | measured M (5 %) | predicted M (15 %) | measured M (15 %) | measured M (5 deg) |
|---|---|---|---|---|---|---|
| 0.01 Rp (0.0005 m) | -2.000 | 315 | n/a | 182 | 192 | 96 |
| 0.02 Rp (0.0010 m) | -2.000 | 223 | 256 | 129 | 192 | 64 |
| 0.05 Rp (0.0025 m) | -2.000 | 141 | 192 | 82 | 96 | 32 |
| 0.1 Rp (0.0050 m) | -1.999 | 100 | 128 | 58 | 64 | 24 |
| 0.2 Rp (0.0100 m) | -1.999 | 71 | 96 | 41 | 48 | 24 |
| 0.5 Rp (0.0250 m) | -1.999 | 45 | 48 | 26 | 32 | 24 |

Predicted values agree with the measured grid levels to within one level
(resolution factor 1.33--1.5); the residual is grid spacing, not a model error.

The numbers are unforgiving, and this is the headline caveat for bow-profile
work: keeping contact is *far* weaker than being accurate.  At `delta = 0.02 R_p`
contact survives from `M = 50` segments
(predicted) / 64 (measured on the grid)
but the error there is still ~60 %; a 5 % error
needs `M = 223`.

Translated to real hulls (`real_bow_guidance`):

| bow radius (m) | penetration (m) | hull points /180° | chord sagitta (m) | sagitta/penetration | required segments | contact lost |
|---|---|---|---|---|---|---|
| 5 | 0.01 | 10 | 0.2447 | 24.47 | 50 | yes |
| 5 | 0.01 | 20 | 0.0616 | 6.16 | 50 | yes |
| 5 | 0.01 | 40 | 0.0154 | 1.54 | 50 | yes |
| 5 | 0.05 | 10 | 0.2447 | 4.89 | 23 | yes |
| 5 | 0.05 | 20 | 0.0616 | 1.23 | 23 | yes |
| 5 | 0.05 | 40 | 0.0154 | 0.31 | 23 | no |
| 20 | 0.01 | 10 | 0.9789 | 97.89 | 100 | yes |
| 20 | 0.01 | 20 | 0.2462 | 24.62 | 100 | yes |
| 20 | 0.01 | 40 | 0.0617 | 6.17 | 100 | yes |
| 50 | 0.02 | 10 | 2.4472 | 122.36 | 112 | yes |
| 50 | 0.02 | 20 | 0.6156 | 30.78 | 112 | yes |
| 50 | 0.02 | 40 | 0.1541 | 7.71 | 112 | yes |

A 5 m bow radius with 20 digitized points has a 62 mm chord sagitta; a DEM
penetration of 10 mm is six times smaller, so a polygon bow would report **no
contact at all** over most of the bow length, while the true hull is touching.
The rule above says 50 segments (3.6 deg per chord) are needed at that depth,
and 223 for a 5 % force error.  This matters for the repository's own demo
profiles: the two-point vertical profile in `docs/polyline_hull_contact.md` is a
straight line (zero sagitta, exactly verified), but any curved profile built from
a handful of lines-plan points is not.

![geometry](bow_contact_geometry.png)

## 4. Checks that pass exactly

| check | reference | result |
|---|---|---|
| point-to-polyline distance | independent cross-product/endpoint formula | max deviation 3.12e-17 m over 200 random points (12 contacting) |
| endpoint (corner) contact | circle-to-circle `k (R_p - d)` at a vertex | relative error 3.38e-15 |
| action-reaction | `sum(F) + R = 0` | exact zero, 0.0e+00 |
| no attraction | magnitude >= 0, force along the outward normal | 0 violations in 23 contacts |
| rigid rotation (0.30, 1.10, pi rad) | `F(Rx, Rv, Rvtx) = R F(x, v, vtx)` | max dev 6.0e-14 N |
| mirror `y -> -y` | forces mirror | max dev 0.0e+00 N |
| vertex-order reversal | same geometry, reversed digitization | max dev 1.1e-14 N (non-degenerate cases) |

## 5. Defect found and fixed

**Degenerate on-line contact was digitization-order dependent.**  When a disk
centre lies exactly on the polyline, the radial normal is undefined and the old
fallback used the left normal of the *stored* segment direction, so the same
hull traversed in the opposite order produced the opposite force:
`[[0, +50]] N` versus `[[0, -50]] N` for the identical geometry -- i.e. a 50 N
*attraction* into the hull instead of a 50 N repulsion.  On-line coincidence is
reachable in practice: a square DEM lattice on `y = 0` with a hull line at
`y = 0` is precisely the demo configuration in `docs/polyline_hull_contact.md`.

Fix (`src/tensordem/hull_contact.py`): the fallback normal is built from a
canonical tangent (first non-zero component made positive), which makes it
independent of vertex order, and it is then oriented against the relative
approach velocity `v - v_hull` so that the force never drives the disk further
along its own approach path.  After the fix the reversed digitization gives
max deviation 0.0e+00 N
(`test_degenerate_on_line_contact_is_vertex_order_independent`) and, for a hull
advancing in +x, the on-line disk is pushed in +x
(`test_degenerate_contact_opposes_the_approach_velocity`).

## 6. Error budget and limits of validity

* The verified statements are geometric and exact to round-off (3e-17 m
  distance, 3e-15 relative force): the discretization law is
  analytic, the rest is float64 closure.
* The convergence orders (-2.00, -1.00) are
  asymptotic in `1/M` for a *circular* hull.  For an arbitrary digitized profile the
  local curvature radius replaces `R`, and a profile with a corner (a
  discontinuous tangent) is reproduced exactly at the corner but has no
  second-order behaviour at all -- there the error is set by the neighbouring
  hull form, not by `M`.
* The reference is the same penalty law with exact geometry, so this verifies the
  geometry of the contact, not the contact law itself: no stiffness calibration,
  no restitution, no friction (the kernel has none), no hull flexibility, no
  water loads.  The kernel remains a geometric prototype, not a calibrated
  ice-resistance model.
* Elliptical or non-circular bows are covered only through the local curvature
  argument above; no separate analytic reference is implemented for them.
