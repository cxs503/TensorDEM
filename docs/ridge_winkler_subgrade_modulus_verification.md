# Winkler subgrade modulus of the ridge-grounding subgrade: Vesic (1961) vs the merged width-only form

Protocol `tensordem-ridge-winkler-subgrade-verification-v1`. Verification of the merged
geotechnical chain in `src/tensordem/ridge_geotech.py` against the Vesic (1961) closed form and
against an independent 3-D elastic half-space beam reference. Scope: code- versus
reference-verification, not a physical validation.

Status of the touched capability: **REAL** for the implementation, **WITHHELD-turned-quantified**
for the modulus model: `docs/ice_ridge_pipeline_grounding.md` §4 item 6 listed the subgrade
modulus as "a chosen elastic model, not a derived value". It is now derived, checked, and the
error it carries is bounded.

## 1. Case, physics and parameters

A first-year ice ridge keel bears on a buried subsea pipeline. The pipe is an Euler-Bernoulli beam
of width `B` on a Winkler bed; the bed stiffness `k_v` converts pipe settlement into soil pressure
`p = k_v w`, so it sets both the embedment and the bending moment that drive the DNV-OS-F101
utilisation. The merged model is

    k_v = 0.65 * E_s / ((1 - nu_s^2) * B)                                 (merged)

Vesic (1961), fitting a Winkler beam to Biot's (1937) beam-on-elastic-half-space solution, gives

    k_v = 0.65 * (E_s / (1 - nu_s^2)) * (E_s B^4 / (E_p I_p))^(1/12) / B  (Vesic)

The coefficient `0.65` is identical, so the merged form is Vesic's relation with the dimensionless
beam/soil **rigidity factor** `rho = (E_s B^4 / (E_p I_p))^(1/12)` replaced by 1. Because `rho`
enters at the 1/12 power the omission is easy to overlook, but for subsea line pipes it is a
20-40 % effect, not a rounding error.

| item | value |
|---|---|
| pipeline | 36 in, `D = 0.914 m`, `t = 0.0159 m`, `E_p = 210 GPa` |
| pipe rigidity | `E_p I_p = 9.507e8 N m^2` (`I_p = 4.527e-3 m^4`) |
| seabed | `E_s = 10 MPa`, `nu_s = 0.30` (`MohrCoulombSoil` defaults) |
| applied load | `P = 867 kN` (`N/m` of ridge) over a `2.0 m` contact patch |
| rigidity ratio | `E_s B^4 / (E_p I_p) = 7.345e-3`, `rho = 0.664` |
| merged `k_v` | `7.815 MN/m^3` (`k_eff = k_v B = 7.143 MN/m^2`) |
| Vesic `k_v` | `5.189 MN/m^3` (`k_eff = 4.743 MN/m^2`) |

## 2. What was done

1. Read the merged path (`MohrCoulombSoil.subgrade_modulus`, `initial_vertical_stiffness`, the
   `pipe_stress_profile` / `ridge_dem` consumers) and identified the missing rigidity factor.
2. Built an **independent 3-D continuum reference** in
   `scripts/verify_ridge_winkler_subgrade_modulus.py`: the beam is coupled to an elastic half-space
   through the Boussinesq settlement kernel

       w(x) = (1 - nu^2)/(pi E) * integral integral p(s, t)/r ds dt

   discretised with piecewise-constant contact pressure on a 1-D element mesh and *exact*
   closed-form rectangle integrals of `1/r` (`quadrant_over_r`, `strip_kernel`; antiderivative
   `u asinh(t/u) + t asinh(u/t)`). The coupled linear system `[K_b Q; -T C](d, p) = (f, 0)` is
   solved after row/column equilibration. Nothing from the Winkler model enters the reference.
3. Self-verified the reference three ways before using it as an anchor:
   * kernel vs 2-D Gauss-Legendre quadrature (relative error `0.0`, i.e. `< 1e-15`);
   * beam half vs the **exact** Hetenyi (1946) patch-load solution derived by superposing the
     point-load solution, `w(0) = (q/k)(1 - e^-ba cos ba)`, `M(0) = (q/(2 beta^2)) e^-ba sin ba`;
   * coupled solution vs global equilibrium, compatibility residual, and length/mesh convergence.
4. Inverted the reference for the Winkler modulus that reproduces its maximum moment, and compared
   that inverted value with Vesic's closed form (a check of the *literature formula*, not of the
   library).
5. Added the opt-in Vesic path to the library (`subgrade_modulus_model="vesic"`, needs
   `beam_ei`) plus the module-level `vesic_subgrade_modulus`; the default path is unchanged.

## 3. Results

Figure: `docs/artifacts/ridge_winkler_subgrade/subgrade_modulus_verification.png`
(JSON: `docs/artifacts/ridge_winkler_subgrade/subgrade_modulus_verification.json`).

Reference-solver self-checks: kernel vs brute-force quadrature `0.0` relative error; reciprocity
`0.0`; beam FE vs exact closed form `-0.035 %` (deflection) and `+0.137 %` (moment); global
equilibrium `1.0e-9`; compatibility residual `5.9e-17 m`.

<!-- AUTOGEN:RESULT_TABLES -->
| model | max deflection [mm] | max moment [kNm/m] | k_eff [MN/m^2] |
|---|---|---|---|
| 3-D half-space (reference) | 16.1534 | 959.654 | — |
| Winkler, merged k (no rigidity factor) | 11.8904 | 840.385 | 7.143 |
| Winkler, Vesic (1961) k | 15.3427 | 951.178 | 4.743 |
| Winkler, k inverted from the reference moment | 15.6200 | 959.654 | 4.604 |

| model | deflection error vs reference | moment error vs reference |
|---|---|---|
| merged k | -26.39 % | -12.43 % |
| Vesic k | -5.02 % | -0.88 % |
| inverted k (floor for any Winkler fit) | -3.30 % | -0.00 % |

| soil E_s [MPa] | rigidity factor | merged k deflection err | Vesic k deflection err | merged k moment err | Vesic k moment err |
|---|---|---|---|---|---|
| 2 | 0.5807 | -25.1 % | -1.9 % | -14.6 % | -0.4 % |
| 5 | 0.6267 | -25.9 % | -3.5 % | -13.4 % | -0.7 % |
| 10 | 0.6640 | -26.4 % | -5.0 % | -12.4 % | -0.9 % |
| 20 | 0.7035 | -26.8 % | -7.0 % | -11.3 % | -1.0 % |
| 50 | 0.7593 | -27.1 % | -10.2 % | -9.7 % | -1.0 % |

| pipe | D [m] | t [mm] | rigidity factor | merged k / Vesic k |
|---|---|---|---|---|
| 8 in | 0.203 | 12.7 | 0.6037 | 1.656 |
| 18 in | 0.457 | 12.7 | 0.6403 | 1.562 |
| 36 in | 0.914 | 15.9 | 0.6640 | 1.506 |

| half-space convergence | L [m] | n_elem | element [m] | w_max [mm] | M_max [kNm/m] | equilibrium err |
|---|---|---|---|---|---|---|
| | 80 | 400 | 0.20 | 15.9307 | 959.654 | 8.7e-10 |
| | 160 | 800 | 0.20 | 16.1534 | 959.654 | 1.0e-09 |
| | 160 | 400 | 0.40 | 16.0485 | 956.173 | 3.6e-12 |
<!-- /AUTOGEN -->

### 3.1 Reference solver convergence (36 in pipe, `E_s = 10 MPa`, patch load)

Moment is the design quantity and it converges: the length refinement (80 -> 160 m) changes it by
`5e-7` relative and the element refinement (0.4 -> 0.2 m) by `0.36 %`. Absolute *settlement* is not
a well-posed metric for a half-space beam - the settlement field carries a logarithmic
length-dependent offset (the far field of the 160 m model settles 20.9 mm while the loaded centre
settles 0.34 mm) - which is why the comparison below uses the **differential** deflection measured
10 m away from the load, and the moment as the primary criterion.

### 3.2 The debt, quantified on the reference case

The merged modulus is 1.55x the modulus that the continuum solution itself demands; Vesic's closed
form lands within 3.0 % of that independently inverted value (`k_vesic/k_inverted = 1.030`).

### 3.3 Sweeps: soil modulus and pipe size (figure panels c and d)

Across `E_s = 2-50 MPa` the merged modulus misplaces the design moment by 9.7-14.6 % and the
differential deflection by 25-27 %; the Vesic modulus stays within 1.0 % on the moment and within
10.3 % on the deflection. The omitted rigidity factor grows for smaller/less rigid pipes: +50.6 %
(36 in), +56.2 % (18 in), +65.6 % (8 in) at `E_s = 10 MPa`.

## 4. Implementation taken into the library

* `vesic_subgrade_modulus(young_modulus, poisson_ratio, width, beam_rigidity, coefficient=0.65)`
  - the closed form, with input validation.
* `MohrCoulombSoil(subgrade_modulus_model="vesic")` + `subgrade_modulus(width, beam_ei=...)` and
  `initial_vertical_stiffness(diameter, beam_ei=...)` - opt-in only. `beam_ei` is mandatory for
  the Vesic model, so a caller cannot silently get a width-only bed.
* Default (`"width_only"`) is bit-for-bit the old behaviour; the existing coupled campaign results
  are therefore reproducible and are **not** silently changed by this verification.

## 5. Notes, pitfalls and scope

* **Sign convention.** In this work `w` is positive downward and the contact pressure positive in
  compression. Reporting `min(w)` as "maximum deflection" silently reports the far-field *uplift*;
  the first run of the reference made exactly that mistake and produced nonsense deflection errors
  of -300 %. Deflections are now taken as `max(w)` and compared differentially.
* **Absolute settlement of a half-space beam diverges logarithmically** with the beam length
  (the net pressure is positive, so the `1/r` tail never cancels). Only curvature-derived
  quantities (moment, differential deflection) are compared; this is also why Vesic fitted the
  *moment*.
* **`0.65` is not the error.** The merged code already uses Vesic's coefficient; the missing item is
  the rigidity factor. A reader who "fixes" the coefficient instead would make the model worse.
* **One Winkler `k` cannot match two responses.** Even the ideal moment-fit modulus leaves a 3.3 %
  differential-deflection error (the Winkler family floor in this configuration); Vesic's 5.0 % is
  close to that floor. This is a property of Winkler idealisation, not of the formula.
* **Surface beam benchmark.** Vesic's fit and this reference are surface-beam problems; the real
  pipe is buried with 0.5 m cover, which stiffens the bed and is *not* modelled here. Embedment is
  a separate, still-unverified effect - this document bounds the modulus model only.
* **Elastic, isotropic, homogeneous half-space.** No stratification, no rate effects, no pore
  pressure; `MohrCoulombSoil` is an elastic-perfectly-plastic (soil) model whose *elastic* branch is
  what this modulus describes.
* **Out of scope:** the pipe-soil friction/gapping, the plastic capacity `p_ult` (verified
  separately in `docs/ridge_capacity_skempton_audit.md`) and the DEM keel itself.

## 6. Error verdict against the platform precision standard

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| Vesic `k` vs the independently inverted reference modulus | half-space (inverted) | 3.01 % | marginally **no** (just outside; well inside 15 %) |
| Vesic `k` maximum moment | half-space | 0.88 % | **yes** |
| merged `k` maximum moment | half-space | 12.43 % | **no** (> 3 %, still < 15 %) |
| merged `k` differential deflection | half-space | 26.39 % | **no** (> 15 %, unacceptable) |
| merged `k` moment over the soil range `E_s = 2-50 MPa` | half-space | 9.65-14.62 % | **no** |
| Vesic `k` moment over the same range | half-space | 0.39-0.99 % | **yes** |
| beam FE vs exact Hetenyi patch closed form | analytic | 0.14 % (moment), 0.035 % (deflection) | **yes** |
| settlement kernel vs 2-D quadrature | numerical | 0.0 % | **yes** |

**Verdict.** The merged width-only modulus fails the 3 % bar on the design moment (12.4 %) and the
15 % bar on deflection (26.4 %); Vesic's full relation passes the 3 % bar on the moment that enters
the DNV utilisation. The recommendation is therefore to switch the coupled campaign to
`subgrade_modulus_model="vesic"` before quoting pipeline bending moments, and to re-run the
utilisation numbers with it. Honest caveat: the comparison is a *surface*-beam benchmark, so the
3 % statement qualifies the modulus model, not the buried-pipe response.

## Test

    PYTHONPATH=src python -m pytest tests/test_ridge_winkler_subgrade_verification.py -q

Covers the kernel (exact vs quadrature, sign folding, symmetry, positivity), the beam FE against
the closed form, equilibrium/compatibility/convergence of the coupled solver, the Vesic scaling
exponents (`E_s^{13/12}`, `(E_p I_p)^{-1/12}`, `B^{-2/3}`, `1/(1-nu^2)`), the reduction of Vesic to
the merged form at unit rigidity ratio, the unchanged default path, the opt-in path, and the two
debt assertions (reference recovery within 5 %, merged-modulus moment error below -3 %).

    PYTHONPATH=src python scripts/verify_ridge_winkler_subgrade_modulus.py

regenerates the JSON, the figure and the tables above.