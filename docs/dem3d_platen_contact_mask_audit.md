# Plane-platen contact mask audit (DEM3D)

Status: defect found, fixed, regression-tested. Scope: `src/tensordem/dem3d.py`
(infinite top/bottom platens), all platen-driven benchmarks (UCS, platen
compression, three-point bending, Brazilian/diametral compression) and the wedge
and bow cases that enable platens.

## 1. What the code did

`IceDEM3D.forces()` evaluates each platen as a one-sided spring-dashpot:

    top_overlap     = (z + R - z_platen).clamp_min(0)
    top_relative_v  = v_z - v_platen
    top_mag         = (k_p * top_overlap + gamma_p * top_relative_v).clamp_min(0)

The **stiffness** term vanishes outside the contact because `top_overlap` is
clamped at zero, but the **damping** term does not: `gamma_p * top_relative_v`
is evaluated for every particle in the specimen. The spherical tool contact in
the same file does mask its force (`torch.where(tool_overlap > 0, ..., 0)`), so
the platen path was an outlier rather than a deliberate convention.

## 2. Consequence

1. **Fictitious reaction.** With `n` particles and no particle anywhere near the
   plane, the reported platen reaction is `n * gamma_p * |v_platen|`. Measured:
   48-particle specimen, platens 0.5 m away, `gamma_p = 0.5 N s/m`,
   `v_platen = 0.05 m/s` gives a reaction of `1.2000000000000002 N` with zero
   contacting particles - exactly the closed form (figure, panel a).
2. **Spurious body force.** The fictitious term is not symmetric: the top platen
   adds `+gamma_p (v_z - v_top)` and the bottom platen subtracts
   `gamma_p (v_z - v_bottom)`, so any particle whose velocity differs from the
   platen velocity is dragged. For a specimen translating with velocity `u` this
   injects a net force `2 gamma_p u` per particle (top and bottom add), i.e. a
   spurious viscous drag that is not present in the physics and that is not
   accounted for by `drag` in the energy ledger.
3. **Load-level bias.** Every platen benchmark reports a peak load that contains
   the offset `n * gamma_p * |v_p|`. For the UCS smoke case (48 particles,
   `gamma_p = 0.5`, `v_p = 0.05 m/s`) the offset is 1.2 N; for the Brazilian case
   in `benchmarks/dem3d/brazilian_cases.json` (447 particles, `gamma_p = 0.5`,
   `v_p = 0.002 m/s`) it is 0.447 N - about 4x the physical load measured after
   900 steps. Zero damping configurations (the DEM3D CLI default is
   `--platen-damping 0.0`) are unaffected.
4. **Energy accounting is not affected.** `platen_contact_energy_J` uses
   `0.5 k_p overlap^2`, so it was already mask-consistent; only the force and
   reaction were wrong.

## 3. Fix

One line per platen, identical in form to the tool contact:

    top_mag = torch.where(top_overlap > 0, top_mag, torch.zeros_like(top_mag))

and the mirror for the bottom platen. No parameter, default or API changes; no
change for particles that actually intersect a plane, so physically valid
contact behaviour is untouched.

## 4. Verification

`tests/test_dem3d_platen_contact_mask.py` (5 tests, all green):

| Test | Claim |
| --- | --- |
| `test_no_contact_gives_zero_reaction` | reaction and platen energy are exactly 0 with no contacting particle |
| `test_no_contact_does_not_move_the_specimen` | 120 steps with distant platens leave positions bit-identical |
| `test_reaction_equals_contacted_particle_sum` | reaction equals the sum over contacting particles only |
| `test_reaction_switches_on_only_when_the_plane_reaches_the_specimen` | force is off before the plane reaches the surface |
| `test_tool_and_platen_agree_when_nothing_is_in_contact` | tool and platen now share the same no-contact convention |

Measurements before/after on the same configuration:
reaction `1.2000000000000002 N -> 0.0` (top), `-1.2000000000000002 N -> 0.0`
(bottom); closed form `n gamma_p |v_p| = 1.2 N` matches the pre-fix value to
machine precision, which confirms the mechanism rather than a coincidence.

## 5. Effect on previously published numbers

Absolute platen load levels in earlier campaign reports were inflated by the
offset above; ratios, timings and fracture events at nonzero damping are close
but not identical, so re-running a campaign against the fixed engine will shift
peak loads downwards (by the offset) and can change fracture onset slightly.
Reports produced before this fix should be read with that offset in mind, and
any load compared against an analytical line load must be regenerated.

## 6. Brazilian (diametral) apparatus: what the benchmark now says

`scripts/benchmark_dem3d_brazilian.py` + `benchmarks/dem3d/brazilian_cases.json`
build a bonded-sphere cylinder whose axis lies along *y* (`geometry="cylinder_xz"`)
and press it along *z*, which is the classical Brazilian disc configuration: the
platen contact is a diametral line load and the indirect tensile strength is
`sigma_t = 2 P / (pi D t)`.

Protocol invariants (green at 900 steps, `artifacts/dem3d_brazilian/`): reaction
symmetry `1.2e-16`, split-stress definition residual `< 1e-12`, repeatable
digest, contact-arc ratio `0.067` (below the 0.2 limit, so the diametral
line-load premise of the apparatus does hold), `D = 0.300 m`, `t = 0.060 m`,
447 particles.

**Withheld, not passed:** the elastic centre field is *not* equilibrated at 900
steps. The centre virial stress is `sigma_xx = -0.06 Pa` against
`2P/(pi D t) = 4.01 Pa` and the principal ratio is still `+6.1` instead of the
Hondros value `-3`. A wave transit across `D` needs about `3.3e3` steps and the
viscoelastic relaxation of the lattice is longer, so the benchmark's physics
checks must be evaluated after an equilibration campaign of order `2e4` steps
(about 12 min of wall clock on one core). That campaign is **not** part of this
milestone; the physics comparison is therefore WITHHELD rather than reported as
a pass or a failure.

## 7. Interpretation limits

* Central-force (normal-only) bonds mean the lattice is not an isotropic
  continuum, so agreement with a continuum field is an apparatus diagnostic and
  never a material calibration.
* A staircase-digitised crown produces a finite-width contact patch, not a
  mathematical line load.
* Peak split stress is a discrete-lattice output; it is not ice tensile strength.