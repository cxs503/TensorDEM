# TensorDEM3D contact-law restitution audit
## Why this document exists

`DEM3DConfig.contact_damping` is a raw dashpot coefficient (N s/m) with no stated
physical meaning, and the only restitution mapping in the repository
(`tensordem.contact_tangential.restitution_to_damping`) implements the **bilateral**
mass-spring-dashpot result

    zeta = c_n / (2 sqrt(k_n m_eff)) ,    e = exp(-pi zeta / sqrt(1 - zeta^2))

(Cundall & Strack 1979; Johnson, *Contact Mechanics*, 1985). The law actually
implemented by `IceDEM3D` is **unilateral**: every contact force is clipped,

    F_n = max(0, k_n delta + c_n d(delta)/dt)      (delta = overlap > 0)

so the dashpot cannot act in tension and the contact opens as soon as the spring
force is overcome. For the same `c_n` the clipped law therefore rebounds with a
larger velocity than the bilateral formula predicts, and it still rebounds where
the formula predicts sticking. This audit measures that difference, verifies the
engine against a reference solution of its *own* law, and replaces the mapping
with an inverted (numerically exact) one: `tensordem.restitution`.

`benchmarks/ice_floe_physics.py` anchor 2 checks the algebra of the bilateral
formula and integrates the bilateral ODE in the benchmark itself; it does **not**
exercise `IceDEM3D`. Its pass therefore certifies the formula, not the solver.

## Set-up and method

* Particle: radius 0.025 m, density 917 kg/m^3, mass 0.0600175 kg; contact stiffness
  5000 N/m. Effective masses: sphere/sphere 0.0300088 kg (`m/2`), sphere/platen
  0.0600175 kg (`m`, infinitely massive partner).
* `pair` -- two free spheres collide head-on inside `IceDEM3D` (bond mask
  cleared without releasing bond energy, drag 0, tool/platen out of reach).
* `platen` -- one sphere rebounds off the infinite-mass bottom platen
  (`platen_stiffness`/`platen_damping`), a different code branch with a
  different effective mass.
* Reference: RK4 solution of the clipped-law impact problem with
  detach/exit event localisation (`tensordem.restitution.unilateral_contact_detail`,
  4000 samples per contact period; resolution sensitivity < 1e-5 relative).
* Comparison branch: `tensordem.restitution.classical_restitution` = the
  bilateral formula above.
* Engine restitution is measured as `|v_out| / |v_in|` of the relative normal
  velocity, sampled on the last free-flight step before overlap and the first
  free-flight step after it.

Reproduce: `python scripts/verify_dem3d_restitution.py`
(outputs `outputs/dem3d_restitution/audit.json` + figures).

## Damping sweep (engine vs reference vs textbook formula)

| c_n (N s/m) | pair e (engine) | pair e (reference clip) | pair e (bilateral) | platen e (engine) | platen e (reference clip) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.99987 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| 2 | 0.78333 | 0.78353 | 0.77311 | 0.83892 | 0.83948 |
| 5 | 0.56636 | 0.56594 | 0.51946 | 0.65679 | 0.65972 |
| 10 | 0.35481 | 0.35747 | 0.24544 | 0.46099 | 0.46276 |
| 20 | 0.17495 | 0.17540 | 0.01178 | 0.25903 | 0.25901 |
| 40 | 0.06555 | 0.06626 | 0.00000 | 0.11038 | 0.11113 |

Contact duration and peak overlap were measured as well; the largest relative
duration deviation from the reference is 195.65% and the largest restitution
deviation 0.00% absolute (relative 1.08%). The bilateral formula is off by up
to 0.1632 in `e` -- for `c_n = 40` the engine still rebounds (e = 0.0655) where the
formula predicts full sticking (e = 0).

## Mapping closure (target e -> c_n -> engine e)

Inverting the reference law (`restitution.damping_for_restitution`) and feeding
the result back into the engine:

| target e | pair c_n | pair e measured | rel. error | platen c_n | platen e measured | rel. error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 0.8394 | 0.90009 | 0.010% | 1.187 | 0.89976 | -0.026% |
| 0.7 | 2.994 | 0.69971 | -0.042% | 4.234 | 0.70045 | 0.064% |
| 0.5 | 6.254 | 0.49932 | -0.137% | 8.844 | 0.50083 | 0.167% |
| 0.3 | 12.18 | 0.29981 | -0.064% | 17.23 | 0.29916 | -0.279% |

Using the bilateral mapping instead (`restitution.classical_damping`) leaves a
worst-case closure error of **32.3%** (it under-prescribes `c_n` because the
clip removes dissipation). The clipped-law inversion stays below **0.28%**.

## Practical consequences

* `DEM3DConfig` defaults (`contact_damping = 5`, `contact_stiffness = 5000`,
  `radius = 0.025`, `density = 917`) correspond to `e = 0.56636` for sphere/sphere
  contact and `e = 0.65679` for sphere/platen contact, not to the 0.5195 the bilateral
  formula would give.
* To impose a physical restitution use
  `c_n = tensordem.restitution.damping_for_restitution(e, k_n, m_eff)` with
  `m_eff = effective_mass(m, m)` for two equal spheres and
  `m_eff = effective_mass(m)` for a tool/platen/wall contact.
  `tensordem.contact_tangential.restitution_to_damping` remains correct as the
  bilateral textbook relation but must not be used to set the engine damping.
* Contacts inside a bonded lattice that are still bonded are handled by the bond
  spring, so their effective mass differs from the isolated-pair value; the
  mapping applies to broken-bond/free contacts only.

## Limitations

* Linear (Hookean) contact, no Hertzian non-linearity, no adhesion, no
  tangential/rotational branch and no rate dependence; the ice material itself is
  not calibrated here. The result certifies *implementation fidelity of the
  contact law*, not the physical restitution of ice.
* Single isolated impact per case; simultaneous contacts, confinement and
  neighbouring bonds change the effective mass and the outcome.
* Restitution of the clipped law is velocity independent by construction
  (confirmed to 0.0036 across a 20x impact-speed range, see `impact_speed_study`);
  a velocity-dependent restitution would require a non-linear (Hertzian) model.
* Time-step sensitivity is below the reported error bars in the tested range
  (`timestep_refinement`: spread 0.00255 in `e`), so the residual ~1% is the
  semi-implicit Euler bias of the engine, not under-resolution.

## Figures

* `outputs/dem3d_restitution/fig1_restitution_vs_damping.png` -- restitution vs
  dashpot coefficient: engine markers, clipped-law reference curve, bilateral
  text-book curve (pair and platen).
* `outputs/dem3d_restitution/fig2_closure_and_error.png` -- left: absolute
  restitution error of the engine against both models; right: target-vs-measured
  closure of the two damping mappings (the bilateral mapping falls below the
  diagonal).
