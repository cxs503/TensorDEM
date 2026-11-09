# DEM contact physics module (maturation node P121W5-N11)

**Status: see the acceptance table below (machine-generated into
`artifacts/contact_physics/integration_report.json`).**

This page documents the integrated contact-physics module built in the
ice-navigation campaign:

| node | physics | delivered module | maturation adapter |
|---|---|---|---|
| `P121W5-N02` | temperature-dependent freeze bond | `tensordem.ice_bond_viscoelastic`, `tensordem.wet_ice_cohesion.ice_resistance` | `src/tensordem/contact_physics/bond.py` |
| `P121W5-N05` | ice-ridge morphology / contact-network statistics | `tensordem.ridge_dem`, `tensordem.ice_rubble_microstructure` | `src/tensordem/contact_physics/statistics.py` |
| `P121W5-N08` | melt-lubrication <-> contact two-way coupling | `tensordem.interface_thermal.{lubrication,friction_heat,coupling}` | `src/tensordem/contact_physics/lubrication.py` |
| integration | one operating point, one loop | -- | `src/tensordem/contact_physics/march.py` |

## 1. Why a maturation layer instead of a rewrite

Three campaign nodes each produced a verified module with its own conventions:
a bond law on Prony series, a friction/flux law with a Newton-solved coupled
marching solver, and a set of statistical estimators over a lattice realisation.
Every one of them is already covered by its own verification suite, so **no
physics is re-implemented** here.  The layer does exactly four things:

1. fixes a single call signature per capability (one object, one method), so a
   caller does not have to know which module owns which piece;
2. binds the modules to one `OperatingPoint` (工况) so a bond law and a
   lubrication law are evaluated at the *same* interface temperature, contact
   pressure and slip speed;
3. adds the one integration-level rule that was previously only implicit in the
   engine -- Kachanov damage accumulates but is **truncated at rupture `D = 1`**,
   and a broken bond carries no force;
4. freezes the upstream public API with a committed baseline so the adapters can
   never silently drift into a re-implementation.

The linkage is deliberately **directed**: the lubrication module produces the
interface temperature, the bond module consumes it (through `F_ref(T)`), and the
bond forces feed the statistics sampler.  No module's internal numerics are
touched, which is why an integrated run reproduces the three standalone runs to
round-off (criterion a).

## 2. Public API

All names are importable from `tensordem.contact_physics`
(`__all__` = `TemperatureDependentBond`, `LubricationContactCoupling`,
`RidgeContactStatistics`, `RidgeMorphologyCase`, `OperatingPoint`,
`ContactPhysicsMarch`), and each is defined in one of `contact_physics/bond.py`,
`contact_physics/lubrication.py`, `contact_physics/statistics.py`,
`contact_physics/march.py`.

### 2.1 `TemperatureDependentBond` (`bond.py`, node N02)

```python
TemperatureDependentBond(
    law: MaxwellWiechertBond,
    reference_temperature_c: float = -10.0,
    salinity_ppt: float = 5.0,
    reference_force_n: float = 1.0e3,
    damage_time_s: float = 60.0,
    damage_exponent: float = 3.0,
)
```

* `from_unrelaxed_stiffness(unrelaxed_stiffness, *, equilibrium_fraction,
  delayed_elastic_fraction, relaxation_time, **kwargs)` -- build the Prony law
  from a target instantaneous stiffness (N/m).
* `brine_volume(temperature_c) -> float` -- Timco & Weeks (2010) brine fraction;
  refuses `T >= 0` (no brine exists) and `T < -60 degC` (outside the fit) instead
  of extrapolating.
* `strength_factor(temperature_c) -> float` -- `(1 - sqrt(v_b/0.29))**2`
  normalised to 1.0 at `reference_temperature_c`; `0.0` at a melting interface
  (and at any temperature where the brine volume passes the 0.29 pore limit).
* `within_strength_fit(temperature_c) -> bool` -- whether that value is an
  interpolation (`-60 <= T <= -2 degC`) or an extrapolation of the fit.
* `reference_force(temperature_c) -> float` -- Kachanov `F_ref(T)`.
* `advance(branch_offsets, elongation_old, elongation_new, dt) -> dict` --
  `branch_offsets`, `force`, `work`, `dissipation`, `stored_energy`.
* `force_at(branch_offsets, elongation)`, `stored(branch_offsets, elongation)`.
* `damage_increment(force, temperature_c, dt) -> Tensor` -- raw Kachanov rate.
* `accumulate_damage(damage, increment) -> Tensor` -- accumulation truncated at
  `D = 1` (the integration-level rupture rule).
* `relaxation_modulus(time) -> float` -- Prony series `G(t)` in N/m.

### 2.2 `LubricationContactCoupling` (`lubrication.py`, node N08)

```python
LubricationContactCoupling(contact=None, *, max_iterations=40, tolerance=1e-6, allow_melting=True)
```

* `reset()` -- rebuild the coupled solver in its initial state (returns `self`).
* `step(dt) -> dict` -- one coupled step: the solver's record
  (`time`, `interface_temperature_c`, `mu`, `heat_flux_w_m2`,
  `film_thickness_m`, `iterations`, `residual`, `converged`, `mode`) plus
  `film_normal_stiffness_n_m`, `load_bearing_fraction`, `friction_flux_w_m2`.
* `run(duration_s, dt, record_every=1) -> dict` -- march and return the solver
  summary including its energy ledger and mechanical work.
* the per-step record also carries `melt_fraction`, `load_bearing_fraction`,
  `friction_flux_w_m2` and `film_normal_stiffness_n_m`, all derived from the same
  converged state (no second evaluation of the law).
* `mu(T=None)`, `melt_fraction(T=None)`, `load_bearing_fraction(T=None)`,
  `friction_flux(T=None)`, `friction_power_w(T=None)`, `melt_rate(q_net)`,
  `normal_stiffness()`, `normal_force_at_approach(approach_m)` -- the scalar laws
  and the mechanical branch of the loop; all default to the current state.
* properties `time`, `interface_temperature_c`, `film_thickness_m`, `model`,
  `history`, `case`.

### 2.3 `RidgeMorphologyCase` / `RidgeContactStatistics` (`statistics.py`, node N05)

```python
RidgeMorphologyCase(keel: RidgeKeel, material: RidgeMaterial, tip_height: float = 1.0)
RidgeContactStatistics(case: RidgeMorphologyCase)
```

* `assemble() -> dict` -- build (once, seeded) the bonded keel lattice.
* `morphology() -> dict` -- `keel_depth_m`, `keel_width_m`, `keel_angle_deg`,
  `cross_section_area_m2`, `requested_porosity`, `realised_porosity`,
  `n_particles`, `n_bonds`, `bond_density_1_per_m2`, `mean_coordination`,
  `driven_particles`, `particle_radius_m`.
* `coordination_sweep() -> list[dict]` -- `Z` versus the contact-force cutoff.
* `force_network(forces, *, n_bins=20) -> dict` -- `n_contacts`, `mean_force`,
  `coefficient_of_variation`, `strong_fraction`, `strong_load_fraction`.
* `sample(forces, *, n_bins=20) -> dict` -- `morphology` + `force_network` +
  `n_particles` in one record.

### 2.4 `OperatingPoint` / `ContactPhysicsMarch` (`march.py`)

```python
OperatingPoint(contact=IceSteelContact(), bond=..., ridge=..., strain_rate_1_s=1e-3,
               n_bonds=64, gauge_length_range_m=(0.5, 1.5), dt_s=2e-4,
               statistics_stride=10, seed=20261108)
ContactPhysicsMarch(point=None, *, max_iterations=40, tolerance=1e-6,
                    allow_melting=True, statistics=None)
```

`statistics=` accepts a pre-assembled `RidgeContactStatistics`: the ridge lattice
build is state independent, so assembling it once and reusing it across marches is
legitimate (the overhead benchmark does exactly that and excludes it from both
timed paths).

* `OperatingPoint.gauge_lengths() -> ndarray` -- deterministic per-bond gauge
  lengths (seeded), so the force scatter of the bundle is reproducible.
* `ContactPhysicsMarch.step(dt=None) -> dict` -- one integrated increment:
  lubrication (N08) -> bond (N02) at that interface temperature -> statistics
  sampling (N05) every `statistics_stride` steps.
* `ContactPhysicsMarch.run(n_steps) -> dict` -- `steps`, `time_s`, `final`,
  `histories`, `energy` (`work_j`, `dissipation_j`, `stored_energy_j`,
  `ledger_error`), `statistics`, `force_samples`, `invariants`.
* `ContactPhysicsMarch.summary() -> dict` -- the same summary without advancing.
* attributes `point`, `bond`, `lubrication`, `statistics`, `time`, `history`.

## 3. Usage

```bash
PYTHONPATH=src python3 examples/contact_physics_demo.py
```

```python
from tensordem.contact_physics import ContactPhysicsMarch, OperatingPoint
from tensordem.interface_thermal.coupling import IceSteelContact

point = OperatingPoint(contact=IceSteelContact(cells=61), strain_rate_1_s=1e-5)
summary = ContactPhysicsMarch(point).run(200)

summary["final"]["interface_temperature_c"]  # degC
summary["final"]["film_thickness_m"]         # m (0 while no melting)
summary["final"]["damage_max"]               # <= 1 (truncated at rupture)
summary["energy"]["ledger_error"]            # ~1e-17
summary["statistics"]["morphology"]["realised_porosity"]
```

Three regimes ship with the tests and the benchmark:

| regime | setting | what it exercises |
|---|---|---|
| `cold_slow` | `IceSteelContact(cells=61, slip_velocity_m_s=0.02, initial_temperature_c=-20)`, `strain_rate_1_s=1e-5` | sensible heating, `mu(T)` collapse without melting, intact bonds, interface stays inside the fitted band (`T <= -2 degC`) |
| `cold_rupture` | same contact, `strain_rate_1_s=0.06` | progressive creep rupture and force redistribution, still inside the fitted band (so it is creep, not a brine-volume collapse) |
| `melting` | default contact (`v = 0.2 m/s`) with `MeltLubricationModel(mu_wet=0.02)` | Stefan film growth, `k_eff < k_contact`, `F_ref -> 0` at the melting point |

## 4. Verification (criterion a) and acceptance

Reproduce with:

```bash
PYTHONPATH=src python3 scripts/freeze_contact_physics_api.py             # (b) once, pre-integration
PYTHONPATH=src python3 scripts/contact_physics_regression_gate.py         # (d) upstream node suites
PYTHONPATH=src python3 benchmarks/contact_physics_integration.py          # (a)(c)(e) + figures
PYTHONPATH=src python3 scripts/build_contact_physics_doc.py               # refresh the table below
PYTHONPATH=src python3 -m pytest -q tests/test_contact_physics_integration.py
PYTHONPATH=src python3 scripts/post_contact_physics_report.py             # platform report
```

**Measurement protocol for criterion (c)** (matters on a shared machine): torch
pinned to one thread, the ridge realisation pre-assembled outside the timed region
in *both* paths, the two implementations timed alternately, and the **minimum** of
``repeats`` as the estimator (``timeit`` convention).  All raw samples are stored
in the artifact, so the estimate can be re-judged; on a contended box the median
is not a usable estimator (the medians of the two paths cross sign).

The numbers quoted below are written by the benchmark into
`artifacts/contact_physics/integration_report.json`; the table is refreshed from
that file by `scripts/build_contact_physics_doc.py`.

<!-- ACCEPTANCE-TABLE -->

Generated from `artifacts/contact_physics/integration_report.json` (git `c2e8461e4a50`, 2026-11-08T11:24:36.627023+00:00).

| criterion | requirement | measured | pass | artifact |
|---|---|---|---|---|
| (a) integration | integrated run == three standalone runs, max rel. error < 1e-10 | 0.0e+00 | True | `artifacts/contact_physics/integration_report.json` |
| (b) API stability | zero signature changes vs pre-integration baseline `c2e8461e4a50` | 0 changes | True | `artifacts/contact_physics/api_baseline.json` |
| (c) performance | time and Python peak-memory overhead < 5 % | time -8.08 %, memory -11.48 % | True | `artifacts/contact_physics/performance_pytest.json` |
| (d) regression gate | upstream N02/N05/N08 suites green | 150 passed, 0 failed, 0 errors | True | `artifacts/contact_physics/regression_gate.json` |
| (e) documentation | all public API documented, example runs | coverage 100 % | True | `docs/contact_physics_module.md` |

### Per-regime parity (criterion a)

| regime | steps | duration [s] | max rel. error | T_int end [degC] | mu end [-] | film end [m] | damage max | broken bonds |
|---|---|---|---|---|---|---|---|---|
| cold_rupture | 200 | 0.0400 | 0.0e+00 | -5.9257 | 0.10000 | 0.000e+00 | 1.0000 | 9 |
| cold_slow | 200 | 0.0400 | 0.0e+00 | -5.9257 | 0.10000 | 0.000e+00 | 0.0000 | 0 |
| melting | 300 | 0.0600 | 0.0e+00 | 0.0000 | 0.02000 | 3.374e-05 | 1.0000 | 64 |

### Performance (criterion c)

| quantity | integrated | standalone (three drivers) | overhead |
|---|---|---|---|
| wall time (min of 7, 1 thread) [s] | 0.5773 | 0.6280 | -8.08 % |
| wall time (median of 7) [s] | 0.7054 | 0.7327 | -3.73 % |
| Python peak (tracemalloc, min) [bytes] | 160402 | 181196 | -11.48 % |

All five criteria pass: **True**.  Figures: `artifacts/contact_physics/fig_coupling_history.png`, `artifacts/contact_physics/fig_bond_response.png`, `artifacts/contact_physics/fig_parity.png`, `artifacts/contact_physics/fig_statistics.png`.

## 5. Known limitations and scope

1. **The coupling is directed, not simultaneous.** The interface temperature from
   N08 feeds the bond damage clock, but the bond force does *not* feed back into
   the contact pressure.  Closing that loop is a physics change (`p` becomes an
   unknown of the Newton solve), out of scope for a maturation node; the budget
   for it is an explicit follow-up.
2. **Kachanov truncation is an integration rule, not a new law.** The upstream
   rate law is unbounded; the adapter clamps `D` at 1 and masks the force of a
   broken bond.  The raw increment stays available
   (`damage_increment`) for the verification surface.
3. **`T >= 0` in `strength_factor` is a limit, not a fit.** A melting interface
   has no solid bond, so the factor is exactly 0 and the bond ruptures within one
   step.  This is deliberate and pinned by a test; it is *not* a calibrated
   value.
4. **`-60 degC <= T <= -2 degC` is the fitted band of the strength relation.**
   Below `-60 degC` the brine-volume relation is refused rather than
   extrapolated.  Above `-2 degC` it is evaluated (not clamped) because its own
   limit is physically correct -- at 5 ppt the brine volume passes the 0.29 pore
   limit near `-0.7 degC` and the factor becomes exactly 0 -- but
   `within_strength_fit()` flags those values as out-of-fit, and the near-melting
   collapse of the factor, not creep, is what ruptures bonds there.
5. **Statistics are of one realisation.** `RidgeContactStatistics` samples a
   *single* seeded keel lattice; it is not a ridge-severity distribution.  An
   ensemble over seeds is the follow-up if a design value is required.
6. **Performance is measured on one process, one thread.** The overhead figure is
   a Python-dispatch figure for the regime above; it says nothing about GPU or
   multi-threaded scaling.  The measured overhead is *negative* (the integrated
   path is slightly cheaper) for a fair reason worth stating: the standalone
   driver has to keep every per-step lubrication record so it can replay the
   interface temperatures into the bond loop, while the integrated march consumes
   them online.  That is a memory *saving* of the integrated design, not a
   benchmark trick -- both paths run the same solver, the same bond law and the
   same statistics code.
7. **N05's morphological numbers are lattice statistics.** `realised_porosity`
   differs from the requested porosity because the thinning step removes
   particles (e.g. 0.258 versus 0.30 requested); the realised value is the one
   quoted everywhere.

## 6. Files

| path | role |
|---|---|
| `src/tensordem/contact_physics/{__init__,bond,lubrication,statistics,march}.py` | the module |
| `tests/test_contact_physics_integration.py` | criteria (a)-(e) as pytest |
| `examples/contact_physics_demo.py` | runnable example |
| `scripts/freeze_contact_physics_api.py` | frozen API baseline (criterion b) |
| `scripts/contact_physics_regression_gate.py` | upstream node gate (criterion d) |
| `benchmarks/contact_physics_integration.py` | benchmark + figures + report |
| `scripts/build_contact_physics_doc.py` | injects the acceptance table below from the artifacts |
| `scripts/post_contact_physics_report.py` | publishes the platform report (numbers read from artifacts) |
| `artifacts/contact_physics/` | api baseline, integration report, gate result, figures |