# CASE-DEM-PLANE-SHEAR-MUIDI — Round 11: rough-wall (grafted monolayer) mode

**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池·平面剪切 mu(I) 流变律标定), status `open`.
**Scope of this round:** mechanism isolation, not a new verdict. Rounds 1-10 left
sample identification as the open cause of the −20…−30 % `mu` deficit; the leading
candidate is **wall slip** (round 8 measured `gamma_dot_measured/gamma_dot_imposed`
= 0.38…0.97). da Cruz et al. (2005) drive their cell with *rough* walls for exactly
this reason. This round adds that wall as a switch on the *same* device, so wall
geometry is the only experimental variable.

## ① Case & device

Same device, protocol, contact law and reference law as rounds 1-10
(`benchmarks/plane_shear_mu_i/reference_law.py`); only `wall_mode` changes.

| quantity | value |
|---|---|
| particles | spheres, r = 5 mm, ρ = 2500 kg/m³, ≈378 per run |
| cell | W×D×H = 0.12 × 0.05 × 0.06 m, x periodic, y confined |
| contact | linear spring kn = kt/0.5 = 6.25e4 N/m, Cundall-Strack tangential, μ_p = 0.5, μ_r = 0.05, e = 0.3 |
| control | constant-stress servo, P = 62.5 kPa, shear strain 0.2, settle 2500 steps, ≥400 shear steps |
| reference | μ(I) = 0.36 + 0.48/(0.30/I + 1) (da Cruz 2005 / GDR MiDi 2004; PROVENANCE-UNVERIFIED constants, unchanged) |

`wall_mode="smooth"` = historical flat platen + frictionless floor (baseline,
default, bit-for-bit identical to rounds 1-10).
`wall_mode="rough"` = grafted monolayer of r-spheres on platen **and** floor,
lattice spacing 1.05 d (44 grafts per wall, 58 % areal coverage), moving rigidly
with the wall; normal law identical to the particle-particle law
(`_graft_wall_force`), tangential Cundall-Strack per particle saturated at
`mu_w·Σf_n`; passage of a bulk particle between two grafts is geometrically
closed (spacing 1.05 d < 2 d, pinned by test).

## ② What was done

1. `src/tensordem/mu_i_rheology.py`: new `graft_lattice()` and `_graft_wall_force()`,
   `ServoShearCase.wall_mode` (default `"smooth"`) / `rough_spacing`, guarded so the
   smooth path is untouched; rough branch replaces the floor/platen blocks inside
   `_step_forces` and feeds `fn_sum` / `ftx_sum` to the same servo.
2. `benchmarks/plane_shear_mu_i/rough_wall_sweep.py`: 8-point plan
   (2 wall modes × I ∈ {0.01, 0.03, 0.1, 0.3} × seed 5279), JSONL per point,
   `--index/--total` shard dispatch for the cluster.
3. `tests/test_rough_wall_plane_shear.py`: 10 tests — lattice geometry/blocking,
   unit-cell normal-force analytic check, Coulomb cap, damping sign, validation,
   default-mode regression, rough-run smoke, and a `smooth` vs default equality test.

## ③ Results

Smooth wall (baseline, this round, seed 5279; wall clock 30-44 s/point):

| I_target | μ_ref | μ_wall | err | μ_full (virial) | err | rate meas/imposed | profile R² | servo valid |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.3755 | 0.3813 | **+1.5 %** | 0.5000 | +33.2 % | 0.889 | 0.837 | True |
| 0.03 | 0.4036 | 0.3283 | −18.7 % | 0.3498 | −13.3 % | 1.026 | 0.890 | False |
| 0.1 | 0.4800 | 0.3362 | −30.0 % | 0.3704 | −22.8 % | 0.632 | 0.891 | True |
| 0.3 | — | — | — | — | — | — | — | (run did not complete before this report) |

Rough wall: **FAIL at every attempted point** — `MuIRheologyError: servo did not
reach the target normal stress` (settle budget `max(6·settle_steps, 12000)` exhausted).
A first diagnosis attempt (larger servo step cap `servo_max_step_d = 0.02`) never
executed: the override collided with the protocol's own `settle_steps` argument
(`TypeError`), so the fix is *untested*, not *disproved*. The cluster job that runs
the full 8-point plan is `cj1963` (still queued at report time).

## ④ Figures

None in this report: the decisive series (rough wall) produced no data — the mode
does not reach the target confining stress — and plotting the baseline alone would
repeat round 8's figure. The round-8/round-10 figures remain the visual record for
the smooth wall.

## ⑤ Notes / limitations

1. **What is delivered is a control, not a verdict.** The wall mode is the only
   changed quantity, and its smooth branch is pinned equal to the historical path
   (`test_wall_mode_is_the_only_difference`), so any future rough/smooth difference
   is attributable to wall geometry.
2. **The rough wall does not close the stress servo with the historical gains.**
   Plausible causes, none of them tested here: (i) 44 discrete graft contacts
   (58 % coverage) make the transmitted force noisy and the servo descent cap
   (`0.002 d` per `servo_stride`) too slow to close the initial gap within
   `max_settle`; (ii) the monolayers consume about one diameter of shear gap at
   each wall, so the target pressure is reached at a smaller gap than the flat
   platen would need; (iii) the aggregated per-particle tangential spring
   (`mu_w·Σf_n`) over-stiffens the interface when several grafts touch one particle.
3. **The baseline still misses the law** at I ≥ 0.03 (−13…−30 %) and the low-I
   point is the only one inside the band, reproducing rounds 8-10 within their
   scatter; `phi(I)` remains WITHHELD.
4. **Single seed, no ensemble**, and the I = 0.3 baseline point is missing from
   this report (its run had not finished).
5. Cluster job `cj1963` (`TensorDEM`, stage_dir = this worktree, 8 shards) will
   deliver the full matrix once the queue reaches it; its artefacts land in
   `benchmarks/plane_shear_mu_i/runs/rough_wall/shard_*.jsonl`.

## ⑥ Error judgment (explicit)

| quantity | reference | worst measured | error | within 3 %? |
|---|---|---|---|---|
| μ_wall, smooth, I = 0.01 | 0.3755 | 0.3813 | **+1.5 %** | **yes** |
| μ_wall, smooth, I = 0.03 | 0.4036 | 0.3283 | −18.7 % | **no** (>10 % band) |
| μ_wall, smooth, I = 0.1 | 0.4800 | 0.3362 | −30.0 % | **no** (>10 % band) |
| μ_full (virial), smooth, worst | law | 0.5000 | +33.2 % | **no** |
| graft normal force vs analytic spring law | kn·overlap | exact | ≤1e-12 | **yes** (unit test) |
| graft tangential force vs Coulomb cap | μ_w·f_n | exact | ≤1e-9 | **yes** (unit test) |
| rough-wall μ(I) | law | **not measured** (servo FAIL) | — | **no verdict** |

**Verdict.** Capability: the rough-wall switch is implemented, guarded and
regression-tested; it is a *new control*, not a new number. Physics: **H(wall slip
is the binding constraint) is NOT adjudicated** — the rough mode cannot yet hold
the target stress, so this round neither confirms nor refutes round 10's
identification hypothesis. Ledger row stays `open`; `phi(I)` stays `WITHHELD`.
Honest FAIL, recorded as such: the next round must first make the grafted platen
reach the servo target (wider coverage / smaller spacing, or a settle-step cap that
scales with the gap), then repeat the 4-point sweep with ≥3 seeds.
