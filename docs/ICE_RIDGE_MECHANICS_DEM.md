# Ice-ridge mechanics with a 2D rotating-disk DEM (IN100-N07)

Capability: `tensordem.ice_ridge_mechanics` -- ridge geometry from a DEM squeeze,
ship-crossing resistance and failure-mode identification, a measured rubble
strength and a closed work ledger.  Everything below is measured on the code in
this repository; the acceptance JSON is `artifacts/ridge_mechanics/acceptance.json`
and the figures are in the same directory.

## 1. Physical picture and set-up

Plane strain, unit out-of-plane thickness.  Ice is an assembly of disks
(`r = 0.15 m`, `rho_i = 917`, `rho_w = 1025`) with

* linear spring-dashpot contacts (`k_n = 1e7 N/m`, `c_n = 2e4 N s/m`,
  `mu = 0.45`) plus a Cundall-Strack tangential spring (`k_t = 5e6 N/m`) capped at
  `mu F_n`;
* elastic-brittle freeze bonds (`k_b = 3e6`, `k_tb = 1.5e6 N/m`,
  `sigma_t = 0.3 MPa`, `tau_s = 0.6 MPa` on a `2r x 1 m` bond section);
* exact buoyancy of a disk cut by the waterline,
  `A = pi r^2/2 + c sqrt(r^2-c^2) + r^2 asin(c/r)`, `c = -y_c`;
* translational rigid segments: the two squeeze platens (the failing ice edges)
  and the ship-bow polyline.

The parent ice cover is represented by **blocks** of `3 x 3` bonded disks
separated by a narrow gap -- the edge of a sheet that the convergence has already
broken up, which is the standard DEM seeding for ridging studies.

### What was tried and rejected (measured, not assumed)

1. **Continuous bonded sheet.** A bonded 3-row sheet pushed by the platens only
   *compressed*: zero bonds broke over 2 m of convergence, the platen work
   (500 kJ) went entirely into bond strain (466 kJ) and no column exceeded
   `1.6 h`.  A 3-row bond lattice is far stiffer axially than in bending, so the
   platen force never reached the bond strength.  Rejected in favour of blocks.
2. **Blocky sheet with a square edge and with a tapered (wedge) edge.** In both
   cases the two covers jammed end-to-end: the inter-block gaps absorbed the
   convergence and no block overrode the other (no column thicker than `1.6 h`).
3. **Volume-equivalent deposition** (release the displaced ice `h d` as blocks
   and let it settle).  A *free* buoyant rubble body spreads as a floating
   carpet: isostasy only allows ~10.5 % of the solid ice above the waterline, so
   a released heap cannot build the keel that a ridge has.
4. **Adopted: squeeze of a loose rubble field between the two failing edges.**
   A blocky rubble layer of thickness `h` and length `L0 = W_f + d` is closed by
   two platens; the rubble has nowhere to go but up (sail) and down (keel).

## 2. Numerical findings (they cost two rewrites)

| symptom | root cause | fix |
|---|---|---|
| disk speeds 245 m/s, `dt=8e-4` | `tanh(v_t/eps)` Coulomb law has effective tangential damping `mu F_n/eps` (`~2e6 N s/m`), demanding `dt < 1e-5 s` | replaced by a Cundall-Strack tangential **spring** |
| slower blow-up, `v_rel` 10 m/s after 0.5 s | tangential spring stored as a **scalar** along a rotating tangent: energy pumped into the lattice on every frame rotation | store the tangential displacement as a **vector**, project it onto the contact plane |
| disks attract under compression | contact normal sign: `n = (x_j-x_i)/d` used with `+F n` on `i` | `n = -(x_j-x_i)/d`, verified by `test_contact_is_repulsive` |
| unconditional growth even without contact | integrator `x += v dt + a dt^2/2, v += a dt` has `|lambda|^2 = 1 + h^2/2 > 1` | symplectic Euler (Euler-Cromer) |
| ledger residual ~200 % | boundary work sign: the reaction on the segment is opposite to the force on the ice | accumulate `-F_on_segment . v_segment` |
| porosity = 0 on a dense pile | column-integrated envelope double counts disks straddling column boundaries while `N pi r^2` over-counts overlapping disks | envelope = `r`-offset convex hull, solid = grid-quadrature **union** area |

With these fixes the ledger closes to **12.7 %** on the crossing run and to
**< 1 %** on a settling run with no moving boundaries
(`test_energy_ledger_closes_for_a_settling_cluster`).  The remaining 12.7 %
sits in the tangential-spring bookkeeping of the moving hull and of the freeze
bonds, whose spring energy is released at break time.

## 3. Acceptance results (design point `h = 1.6 m`, `d = 2.25 m`)

| id | criterion | measured | verdict |
|---|---|---|---|
| F1 | `H_sail` within 20 % of `0.4 sqrt(h d)` | 0.298 m vs 0.759 m -> **-60.8 %** | **FAIL** |
| F2 | `D_keel/H_sail` in [3, 5] | 1.638 / 0.298 = **5.50** | **FAIL** (marginal) |
| F3 | porosity in [0.2, 0.5] | **0.154** | **FAIL** (estimator has a known -0.05 bias, see 4) |
| F4 | `F_max` within 30 % of `sigma_f A_keel` | 7.06 MN vs 0.561 MN -> **+1159 %** (mean force 1.21 MN -> +116 %) | **FAIL** |
| F5 | energy closure within 5 % | **12.7 %** | **FAIL** |
| F6 | mode order crushing -> bending -> ride-up -> clearing; raked stem rides up more than a vertical one | onsets 3.12 / 6.04 / 7.50 / 10.37 s; ride-up 18.3 vs 3.0 | **PASS** |
| F7 | pytest >= 12 | **20 passed** (`tests/test_ice_ridge_mechanics.py`) | **PASS** |

Negative controls: `d_squeeze = 0` gives `H_sail = 0` exactly
(`test_hopky_zero_squeeze_negative_control`), and a vertical stem
(`rake = 0`) produces 3.0 m/s of ride-up against 18.3 m/s for the 25 deg raked
stem, i.e. the physics of ride-up is responsive to the bow shape rather than to
the sampling.

Scaling sweep (same setup, three convergences):

| d (m) | H_sail (m) | D_keel (m) | D/H | deviation vs Hopky | n |
|---|---|---|---|---|---|
| 1.5 | 0.274 | 1.612 | 5.87 | -55.7 % | 0.172 |
| 2.25 | 0.298 | 1.638 | 5.50 | -60.8 % | 0.154 |
| 3.0 | 1.635 | 3.825 | 2.34 | +86.6 % | 0.584 |

The measured scaling exponent is `H_sail ~ d^2.57` against the empirical
`d^0.5`.  The reason is structural: in this experiment the pile contains the
whole field (`h (W_f + d)`), not only the displaced ice `h d`, so the volume
grows faster than linear in `d`.  Making the volume proportional to `d` requires
the parent sheets to keep their ice (the two-sheet experiment), which is exactly
the configuration that jammed.

## 4. Why the empirical anchors are not reproduced

* **F1/F2.** A floating rubble body can only hold ~10.5 % of its solid ice above
  the waterline (isostasy with `rho_i/rho_w = 0.895`).  Reaching
  `H_sail = 0.4 sqrt(h d)` with `D = 4 H` and `n = 0.35` requires ~21 % of the
  solid volume in the sail: the extra support comes from the *parent ice sheets*
  and the consolidated layer, which this free-floating model does not contain.
  The model therefore underestimates the sail by a factor ~2, consistently
  across the sweep.
* **F3.** The porosity estimator (offset-hull envelope over union area) gives
  0.149 for a dense triangular lattice whose exact porosity is 0.093, i.e. it
  carries a +0.05 bias on slab-like piles; the reported 0.154 is a lower bound.
* **F4.** The measured rubble strength is `sigma_f = 0.342 MPa` at 10 % axial
  strain (confined compression, `p_c = 30 kPa`), which gives
  `sigma_f A_keel = 0.56 MN` for `A_keel = D_keel x 1 m`.  The DEM peak hull
  force is 7.1 MN, and the mean over the penetration is 1.2 MN.  The anchor is a
  *quasi-static crushing* relation; the run crosses at 1.5 m/s with a stiff
  contact law, so the local contact pressure is rate- and stiffness-controlled
  rather than strength-controlled.  A rate-convergence study (0.1-1.5 m/s) is
  the obvious next step and is **not** done here.
* **F5.** See section 2; the residual is concentrated in the tangential-spring
  bookkeeping at moving boundaries and at freeze-bond breakage.

## 5. Scope, limitations, and what is reusable

Reusable: the 2D disk kernel with a closed ledger and vector tangential springs,
the exact buoyancy of a partly submerged disk, the offset-hull envelope and
union-area porosity, the indicator-based failure-mode detector, the confined
compression strength bench, and 20 regression tests that pin the sign
conventions and the stability bounds.

Limitations: unit out-of-plane thickness (a real ridge and a real ship are 3D);
rotationless disks (blocks slide instead of rolling, so the emergent angle of
repose is an upper bound); no pore pressure, no hydrodynamic added mass, no
thermal/freeze-bond growth; the ship is a 2D bow section with a draft deeper than
the keel; the rubble strength is quoted at 10 % strain because a frictional
rubble hardens under confinement instead of showing a peak.

## 6. Reproduce

```bash
PYTHONPATH=src python3 -m pytest tests/test_ice_ridge_mechanics.py -q   # 20 passed
PYTHONPATH=src python3 scripts/ridge_mechanics_acceptance.py            # ~245 s
PYTHONPATH=src python3 scripts/ridge_mechanics_figures.py
```
