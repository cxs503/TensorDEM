# CASE-DEM-PLANE-SHEAR-MUIDI — Round 21: is the low-I cell "not shearing"?

**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池 · 平面剪切 mu(I) 流变律标定)
**Reference:** da Cruz et al. 2005, *Phys. Rev. E* **72**:021309; GDR MiDi 2004,
*Eur. Phys. J. E* **14**:341. **Tolerance:** < 10 % on `mu(I)`, `phi(I)`.
**Round type:** estimator-defect audit + one new diagnostic field. Two points
re-run on the round-20 device.
**Tools:** `tensordem.mu_i_rheology.strain_partition` (new),
`benchmarks/plane_shear_mu_i/low_i_profile_audit.py`,
`benchmarks/plane_shear_mu_i/test_strain_partition.py`.

---

## ① Case introduction

Round 20 landed a single point of this anchor's sweep, at `I = 3e-3`, and
reported `q ≈ -0.01`, where `q` is the measured / imposed shear-rate ratio. It
read that as *"the bed is not shearing; the cell is still in its start-up
transient at strain 0.5"*, and concluded that the remaining work for the anchor
is a **strain-window** study at low `I` rather than more particles. That
conclusion is load-bearing for the anchor's protocol; it also happens to rest on
a single estimator.

`shear_rate_measured` is a weighted least-squares slope fitted to the **central
50 % of the gap** (`z_lo = 0.25 plate_z`, `z_hi = 0.75 plate_z`, six bins, and
`profile_r2` was never reported). A flat interior profile and a static bed are
different physical statements:

* a **static bed** under a moving wall is wall slip, and Coulomb then forces the
  wall traction ratio to sit at `mu_contact = 0.5`;
* a **localised bed** shears, but only inside the two wall layers, so the
  interior window is a plug and its least-squares slope collapses while both
  walls still move.

Round 21 tells them apart by measuring the whole gap.

Device (identical to round 20, `high_n_coverage.py`): plane-shear box
32d×16d×18d (0.16×0.08×0.09 m) of N spheres (d = 10 mm, rho = 2500 kg/m³),
periodic in x, smooth frictional floors/platen (`mu_w = 0.5`), constant-stress
servo at P = 62.5 kPa, linear spring-dashpot (kn = 6.25e4 N/m, kt/kn = 0.5,
`mu_contact` = 0.5, `mu_rolling` = 0.05, e = 0.3, seed 5279), 3000 settle steps,
strain ceiling 0.5 with `max_recover_strain = 0.25`.

| parameter | value |
|---|---|
| I targets re-run | `3e-03`, `1e-02`, `1e-02`, `3e-03` |
| N particles | 1276 / 1276 / 1276 / 1276 |
| imposed rate (1/s) | 1.5 / 5 / 5 / 1.5 |
| p measured (Pa) | 5.725e+04 / 5.813e+04 / 6.069e+04 / 6.744e+04 |
| mu_wall (tau/p) | 0.234 / 0.179 / 0.212 / 0.106 |
| phi (container) | 0.5928 / 0.5916 / 0.5916 / 0.5907 |
| strain achieved | 0.213 / 0.270 / 0.270 / 0.124 |

## ② What was done

1. **New diagnostic** `strain_partition` in `tensordem.mu_i_rheology`: bins the
   *whole* gap (default 9 bins, particles with y inside the bulk band) and
   returns, on the same state, a full-gap weighted least-squares rate, an
   **extreme-bin difference rate** (`(v_top - v_bottom)/(z_top - z_bottom)`,
   immune to profile curvature), the legacy interior rate, the two wall velocity
   ratios `v_top/v_platen` and `v_bot/v_platen`, and the dimensionless
   **interior strain fraction** = interior slope / extreme-bin slope
   (1 = linear Couette, 0 = interior plug).
2. **Dead-field fix.** `shear_rate_profile` advertised `slip_top` in its
   early-return branches and never computed it on the success path, so the key
   was `NaN` in every archived run (rounds 1-20). It is now defined whenever the
   platen kinematics are supplied, and the per-bin profile is archived.
3. `ServoShearCase.profile_detail` gates the extra sampling. It is a
   **diagnostic only** — nothing feeds back into the integrator; the regression
   test runs the same protocol with the flag off and on and requires identical
   stresses, strain and sample count.
4. Four points re-run on the round-20 device with the diagnostic on
   (`I = 3e-3` and `I = 1e-2`, two protocol tiers; see the note in ①). Pre-registered decision rule (`PREREG` in the runner, written
   before the run): `LOCALISED` if `q_interior <= 0.2`, `q_adjacent >= 0.6`,
   interior strain fraction `<= 0.5`; `SLIP` if `q_full <= 0.2` and
   `v_top/v_platen <= 0.5`; `INCONSISTENT` if a flat full-gap profile coexists
   with a transmitted platen velocity (that would violate no-slip);
   `QUOTABLE` if `|q_adjacent - 1| <= 0.2`.

## ③ Results

<!-- table 1 -->
| I target | N | imposed rate 1/s | p Pa | mu_wall | phi (container) | strain requested | strain achieved | strain budget | q interior (legacy) | q full gap | q extreme-bin | v_top / v_platen | v_bot / v_platen | interior strain frac. | r2 interior fit | I imposed | I legacy abscissa | I full-gap | I extreme-bin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3e-03 | 1276 | 1.5 | 5.725e+04 | 0.234 | 0.5928 | 0.20 | 0.213 | False | 0.143 | 0.724 | 1.080 | 0.985 | 0.025 | 0.271 | 0.599 | 3.135e-03 | 4.389e-04 | 2.271e-03 | 3.386e-03 |
| 1e-02 | 1276 | 5 | 5.813e+04 | 0.179 | 0.5916 | 0.20 | 0.270 | False | -0.108 | 0.542 | 1.134 | 1.012 | 0.004 | -0.113 | 0.445 | 1.037e-02 | 1.073e-03 | 5.623e-03 | 1.175e-02 |
| 1e-02 | 1276 | 5 | 6.069e+04 | 0.212 | 0.5916 | 0.05 | 0.270 | False | -0.077 | 0.557 | 1.118 | 1.005 | 0.011 | -0.070 | 0.401 | 1.015e-02 | 7.692e-04 | 5.655e-03 | 1.135e-02 |
| 3e-03 | 1276 | 1.5 | 6.744e+04 | 0.106 | 0.5907 | 0.05 | 0.124 | False | -0.253 | 0.495 | 1.149 | 1.023 | 0.001 | -0.231 | 0.532 | 2.888e-03 | 7.487e-04 | 1.431e-03 | 3.319e-03 |

<!-- table 2 -->
| point | verdict | decision-rule checks |
|---|---|---|
| i0p003 | **LOCALISED** | localised_shear=PASS; wall_slip=fail; internally_inconsistent=fail; abscissa_quotable=PASS |
| i0p01 | **LOCALISED** | localised_shear=PASS; wall_slip=fail; internally_inconsistent=fail; abscissa_quotable=PASS |
| i1e2 | **LOCALISED** | localised_shear=PASS; wall_slip=fail; internally_inconsistent=fail; abscissa_quotable=PASS |
| i3e3 | **LOCALISED** | localised_shear=PASS; wall_slip=fail; internally_inconsistent=fail; abscissa_quotable=PASS |


## ④ Verdict

Verdicts: `i0p003` = **LOCALISED**, `i0p01` = **LOCALISED**, `i1e2` = **LOCALISED**, `i3e3` = **LOCALISED**.

For `i0p003`, `i0p01`, `i1e2`, `i3e3` the bed *is* shearing at (within the rule) the imposed rate, but the legacy central-50 % estimator cannot see it: the interior strain fraction is far below 1 while the extreme-bin rate is close to the imposed one. Round 20's reading of the same device as *"the bed is not shearing"* is therefore an **estimator artefact**, not a material statement, and the protocol conclusion drawn from it ("low-I points need a longer strain window") does not follow.

## ⑤ Figures

| file | caption |
|---|---|
| `runs/low_i_profile_r21/fig_velocity_profile.png` | Final velocity profile over the **whole** gap, normalised by the platen velocity. Dashed = linear no-slip Couette. A curve that hugs the dashed line everywhere is uniform shear; a curve that is flat in the middle and rises inside the wall layers is localised shear. |
| `runs/low_i_profile_r21/fig_estimators.png` | Left: the three rate estimators against accumulated strain (`q` = measured / imposed), grey band = ±20 %. Right: interior strain fraction; the green band (0.7-1.3) is linear Couette. The legacy estimator's verdict and the whole-gap estimator's verdict are visibly different answers to the same question. |

## ⑥ Error verdict (honest)

| quantity | criterion | measured | verdict |
|---|---|---|---|
| kinematic delivery (`q_adjacent`) | within 20 % of 1 | 1.08 / 1.13 / 1.12 / 1.15 | see table 2 |
| legacy abscissa (`q_interior`) | within 20 % of 1 | 0.14 / -0.11 / -0.08 / -0.25 | see table 2 |
| wall slip | `v_top/v_platen >= 0.9` for no-slip | 0.98 / 1.01 / 1.00 / 1.02 | see table 2 |
| strain request honoured | <= 1.05x requested | see the `strain budget` column: **all four points violated it** (0.12-0.27 delivered for 0.05-0.2 requested) | **FAIL** (recovery gate; the round-17 flag catches it, which is what it is for) |
| mu(I) / phi(I) vs reference | < 10 % | not judged here (estimator round) | **WITHHELD** |

**Is any key quantity within 3 % of its reference? No.** This round does not
produce a mu(I)/phi(I) number at all; it decides *which abscissa* those numbers
must be quoted on. Any claim to the contrary would be a category error, and the
anchor's ledger verdict remains **FAIL / WITHHELD** unchanged.

## ⑦ Pitfalls and limitations

* The interior strain fraction can come out **negative**: the extreme-bin
  slope has one sign while the interior slope is a noise-level wobble with the
  other. That is a plug to within the noise, and it is reported as a signed
  number rather than clipped, because clipping would hide exactly the
  degeneracy being diagnosed.
* Round 20's conclusion is re-examined, not its archived numbers: the round-20
  run files were not committed (`runs/high_n_r20/` does not exist in the repo),
  so the comparison here is between *estimators on the same device and
  protocol*, not between round-20 and round-21 point files.
* The interior strain fraction is a coarse discriminator: on a 9-bin grid with
  the profile kink sitting exactly on the window edge it reads ~0.2 for a pure
  interior plug instead of 0 (see the unit test). It separates the two worlds by
  a factor ≈ 4, not by a factor ∞.
* Two points only, one seed each. This is an estimator round; it makes no claim
  about seed spread.
* The extreme-bin difference estimator is immune to curvature but *not* to a
  genuinely non-linear profile: for a convex profile it is an upper bound on the
  mean gradient. It is quoted as a bound, and the full-gap least-squares value
  is carried alongside.
* The reference magnitudes stay **PROVENANCE-UNVERIFIED** (round 13 finding,
  unchanged): nothing in this round touches the bibliographic blocker.

## ⑧ Recommended next steps

1. Re-quote the existing archive on the **whole-gap** abscissa
   (`I_adjacent`) — the low-I rows were discarded as "transient" on the
   strength of an estimator that cannot see localisation.
2. If the whole-gap abscissa passes, the remaining mu(I) deficit is the
   estimator-spread question (wall / full / bulk disagree by up to 89 % in
   round 20), not a strain-window question.
3. Still blocking any ledger closure: an archived reading of da Cruz 2005 /
   GDR MiDi 2004 for the `mu_s`, `mu_2`, `I_0` triple and the phi magnitudes.
