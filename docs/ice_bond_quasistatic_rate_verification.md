# Quasi-static rate campaign for the rate-dependent DEM ice bond

Verification surface for the merged driver `tensordem.bond_quasistatic`
(mission m6541). The campaign script is `scripts/verify_bond_quasistatic_rate.py`;
this document is rendered from `artifacts/ice_bond_quasistatic/verification.json`
by `scripts/render_ice_bond_quasistatic_doc.py` (no hand-copied numbers).

## 1. Case introduction

The m6464 specimen campaign (`docs/ice_bond_specimen_rate_dependence.md`)
recorded an honest FAIL: a 4x4x3 opposed-platen specimen gave a fitted
strength exponent of 0.244 against the analytic `1/(m+1) = 1/3`, with a
non-flat negative control and a strength 7-9x the rigid-plate bound -- the
specimen was inertia dominated. The gap ledger named the next node:
*isolate the constitutive response with a single-bond or single-chain
driver*. This document reports that campaign.

Geometry and loading: a 2 x 2 bonded column (simple-cubic lattice, spacing
`a = 2r`) loaded in **tension** by a prescribed displacement on the top layer.
With `n_layers = 2` the bottom layer is a hard support and the top layer is
driven kinematically, so there is **no free particle at all**: the bond
extension history is exactly `eps(t) = edot * t` and inertia cannot enter.
Tension is deliberate -- separating spheres carry the load through the bond
law alone, so the particle-particle contact branch provably cannot
contaminate the measurement. What is exercised is the engine's own call
sites (`IceDEM3D._bond_force` + `_commit_fracture` + `_commit_creep_damage`).

Parameters (from the JSON):

| quantity | value |
| --- | --- |
| particle radius r [m] | 0.025 |
| bond stiffness k_u [N/m] | 2000.0 |
| breaking strain eps_b [-] | 0.015 |
| creep time t_ref [s] | 0.005 |
| relaxation times [s] | [0.005, 0.05] |
| equilibrium fraction [-] | 0.35 |
| delayed-elastic fractions [-] | [0.45, 0.2] |
| steps per run [-] | 3000 |
| lattice amplification factor [-] | 9.3333 (net layer force / one axial bond force) |

## 2. How it was done

* `ChainTensionDriver` builds the column in the real `IceDEM3D` engine and
  steps it with `dt = t_f(reference) / 3000` (capped by the engine's
  stability bound), i.e. the integration resolution is tied to the
  *reference* failure time, not to the measured one.
* Closed-form references, independent of the engine's stepping:
  * linear + Kachanov: `D(t) = (edot/eps_b)**m * t**(m+1) / ((m+1) t_ref) = 1`
    gives `eps_f/eps_b = [(m+1) t_ref edot / eps_b] ** (1/(m+1))`;
  * Maxwell-Wiechert: Prony force `F(t) = edot L0 [k_e t + sum k_a tau_a
    (1 - exp(-t/tau_a))]` integrated to high resolution for the damage
    reference (no free parameters).
* Arms: A brittle control (m = 0); B ductile m = 2; C dt convergence
  (three dt, damage at half the reference strain); D Maxwell-Wiechert m = 2;
  E exponent sweep m = 1, 2, 4; F inertia dial via free layers
  (n_layers = 2, 3, 4, 6); G inertia dial via density (x0.5, x1, x2).
* Toolchain: `PYTHONPATH=src python scripts/verify_bond_quasistatic_rate.py`
  (single process, ~30 runs x 3000 steps); figures are matplotlib PNGs in
  `artifacts/ice_bond_quasistatic/`.

## 3. Computed results

### 3.1 Arm summaries (fitted vs analytic exponent)

| arm | m | 1/(m+1) | fitted slope | slope err | max F err | max eps err | KE/SE = 0 | modes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_brittle_control | 0 | 0.0000 | nan | n/a | 0.067% | 0.033% | yes | 1 |
| B_ductile_m2 | 2 | 0.3333 | 0.3333 | 0.000% | 0.033% | 0.000% | yes | 4 |
| D_maxwell_wiechert_m2 | 2 | 0.3333 | 0.3679 | 10.373% | 0.085% | 0.000% | yes | 4 |
| E_exponent_m1 | 1 | 0.5000 | 0.5000 | 0.000% | 0.033% | 0.000% | yes | 4 |
| E_exponent_m4 | 4 | 0.2000 | 0.2000 | 0.000% | 0.033% | 0.000% | yes | 4 |

### 3.2 Arm A -- brittle control (must be rate independent)

| rate [1/s] | regime | eps_f meas | eps_f ref | eps err | F meas [N] | F ref [N] | F err | mode | KE/SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | brittle | 0.01500 | 0.01500 | 0.033% | 1.5010 | 1.5000 | 0.067% | 1 | 0.00e+00 |
| 0.1 | brittle | 0.01500 | 0.01500 | 0.033% | 1.5010 | 1.5000 | 0.067% | 1 | 0.00e+00 |
| 0.03 | brittle | 0.01500 | 0.01500 | 0.009% | 1.5005 | 1.5000 | 0.031% | 1 | 0.00e+00 |

### 3.3 Arm B -- ductile m = 2 (envelope `sigma_f ~ edot**(1/3)`)

| rate [1/s] | regime | eps_f meas | eps_f ref | eps err | F meas [N] | F ref [N] | F err | mode | KE/SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | ductile | 0.01191 | 0.01191 | 0.000% | 1.1909 | 1.1906 | 0.033% | 4 | 0.00e+00 |
| 0.2 | ductile | 0.00877 | 0.00877 | 0.000% | 0.8775 | 0.8772 | 0.033% | 4 | 0.00e+00 |
| 0.1 | ductile | 0.00696 | 0.00696 | 0.000% | 0.6965 | 0.6962 | 0.033% | 4 | 0.00e+00 |
| 0.05 | ductile | 0.00553 | 0.00553 | 0.000% | 0.5528 | 0.5526 | 0.033% | 4 | 0.00e+00 |
| 0.02 | ductile | 0.00407 | 0.00407 | 0.000% | 0.4073 | 0.4072 | 0.033% | 4 | 0.00e+00 |

### 3.4 Arm D -- Maxwell-Wiechert m = 2 (engine vs independent reference)

| rate [1/s] | regime | eps_f meas | eps_f ref | eps err | F meas [N] | F ref [N] | F err | mode | KE/SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | ductile | 0.01477 | 0.01477 | 0.000% | 0.7727 | 0.7722 | 0.064% | 4 | 0.00e+00 |
| 0.1 | ductile | 0.01138 | 0.01138 | 0.000% | 0.5108 | 0.5105 | 0.074% | 4 | 0.00e+00 |
| 0.03 | ductile | 0.00841 | 0.00841 | 0.000% | 0.3311 | 0.3308 | 0.085% | 4 | 0.00e+00 |

### 3.5 Arm E -- exponent sweep (emergent exponent tracks 1/(m+1))

| rate [1/s] | regime | eps_f meas | eps_f ref | eps err | F meas [N] | F ref [N] | F err | mode | KE/SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | ductile | 0.00671 | 0.00671 | 0.000% | 0.6710 | 0.6708 | 0.033% | 4 | 0.00e+00 |
| 0.1 | ductile | 0.00387 | 0.00387 | 0.000% | 0.3874 | 0.3873 | 0.033% | 4 | 0.00e+00 |
| 0.03 | ductile | 0.00212 | 0.00212 | 0.000% | 0.2122 | 0.2121 | 0.033% | 4 | 0.00e+00 |

| rate [1/s] | regime | eps_f meas | eps_f ref | eps err | F meas [N] | F ref [N] | F err | mode | KE/SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | ductile | 0.01306 | 0.01306 | 0.000% | 1.3063 | 1.3058 | 0.033% | 4 | 0.00e+00 |
| 0.1 | ductile | 0.01048 | 0.01048 | 0.000% | 1.0486 | 1.0482 | 0.033% | 4 | 0.00e+00 |
| 0.03 | ductile | 0.00824 | 0.00824 | 0.000% | 0.8242 | 0.8239 | 0.033% | 4 | 0.00e+00 |

### 3.6 Arm C -- dt convergence of the damage clock

| dt [s] | D engine | D ref | rel err |
| --- | --- | --- | --- |
| 7.310e-06 | 0.125063 | 0.125000 | 0.050% |
| 1.462e-05 | 0.124875 | 0.125000 | 0.100% |
| 2.924e-05 | 0.125250 | 0.124003 | 1.006% |

Fitted order in dt: **2.17**
(target: first order, i.e. ~1).

### 3.7 Arms F/G -- inertia dials

| arm | n_layers | rho [kg/m3] | KE/SE | F err |
| --- | --- | --- | --- | --- |
| F_inertia_layers2 | 2 | 917 | 0.000e+00 | 0.033% |
| F_inertia_layers3 | 3 | 917 | 6.503e-03 | 0.320% |
| F_inertia_layers4 | 4 | 917 | 1.729e-02 | 6.312% |
| F_inertia_layers6 | 6 | 917 | 6.162e-02 | 6.555% |
| G_inertia_mass0.3 | 4 | 275 | 4.551e-03 | 6.369% |
| G_inertia_mass0.1 | 4 | 92 | 1.263e-03 | 6.369% |
| G_inertia_mass0.03 | 4 | 28 | 2.847e-04 | 6.369% |

log-log correlation between KE/SE and strength error: **0.824**; strength error at zero inertia: **0.033%**.

| arm | max strain non-uniformity [-] |
| --- | --- |
| F_inertia_layers2 | 0.000 |
| F_inertia_layers3 | 2.000 |
| F_inertia_layers4 | 3.000 |
| F_inertia_layers6 | 4.999 |
| G_inertia_mass0.3 | 3.000 |
| G_inertia_mass0.1 | 3.000 |
| G_inertia_mass0.03 | 3.000 |

Honest reading of the two dials: the layer dial (F) shows the error
growing with both KE/SE and the strain non-uniformity, but the mass dial
(G, fixed `n_layers = 4`) holds the error at a **density-independent
floor of ~6.4 %** while KE/SE varies by a factor of ~16. The dominant
deviation for free-layer runs is therefore the *quasi-static strain
non-uniformity* of the multi-layer column (which scales with
`n_layers - 1`), with inertia a secondary modulation -- the KE/SE index
alone does not collapse all runs (correlation 0.824, not ~1). The
zero-inertia, zero-non-uniformity cell (`n_layers = 2`) is the clean
constitutive measurement and is the one the exponent claims rest on.

## 4. Figures

All figures are generated by the campaign script and stored next to the
JSON; the report embeds them.

![fig_strength_vs_rate](../artifacts/ice_bond_quasistatic/fig_strength_vs_rate.png)

![fig_dt_convergence](../artifacts/ice_bond_quasistatic/fig_dt_convergence.png)

![fig_inertia_dials](../artifacts/ice_bond_quasistatic/fig_inertia_dials.png)

![fig_load_history](../artifacts/ice_bond_quasistatic/fig_load_history.png)

![fig_error_vs_inertia](../artifacts/ice_bond_quasistatic/fig_error_vs_inertia.png)

## 5. Notes and limits

* This is the **single-bond / bonded-column level**: the 2x2x2 cell has one
  axial bond family per layer pair, so load redistribution across many
  bonds (a true specimen) is not exercised. The lattice amplification
  factor 9.3333 (affine network factor) converts the measured layer
  force to the axial bond force; it is checked against the closed form in
  `tests/test_bond_quasistatic.py`.
* The Maxwell-Wiechert arm has no strict closed-form *force*; its reference
  is an independent high-resolution integration of the same Prony law, so
  the comparison validates the engine's branch advance and damage clock,
  not the spectrum itself (which remains WITHHELD against laboratory data).
* `n_layers > 2` arms are *diagnostics*: they attribute the deviation
  from the closed form. The measured attribution (see 3.7) is that the
  dominant term is the quasi-static strain non-uniformity of the
  multi-layer column (scales with `n_layers - 1`), with inertia a
  secondary modulation -- the platen campaign lacked any such
  attribution, which is why its FAIL could not be diagnosed.
* No laboratory data set in this repository calibrates the default
  spectrum; the parameter values are literature-shaped (Sinha 1978).

## 6. Error verdict (key quantities vs reference)

| quantity | measured | reference | error | < 3 %? |
| --- | --- | --- | --- | --- |
| ductile m=2 max strength err | 0.5528 N @ 0.05 1/s | 0.5526 N | 0.033% | yes |
| ductile m=2 max strain err | 0.00696 @ 0.1 1/s | 0.00696 | 0.000% | yes |
| MW m=2 max strength err | 0.3311 N @ 0.03 1/s | 0.3308 N | 0.085% | yes |
| MW m=2 fitted slope vs 1/3 | 0.3679 | ref-fit 0.3680 | 0.009% | yes (engine vs ref) |
| brittle control max strength err | 1.5010 N @ 0.5 1/s | 1.5000 N | 0.067% | yes |
| m=1 slope vs 1/2 | 0.5000 | 0.5000 | 0.000% | yes |
| m=4 slope vs 1/5 | 0.2000 | 0.2000 | 0.000% | yes |
| damage clock order in dt | 2.17 | ~1 (first order) | - | - |

The MW arm's fitted slope (0.3679) sits 10.3 % above `1/3`; the
independent reference fit is 0.3680, i.e. the offset is a *constitutive
effect* of the relaxation spectrum (early-time unrelaxed stiffness adds
rate stiffening on top of the creep damage), not a numerical error --
the engine reproduces the reference to 0.009 %. The pure `1/(m+1)` law
is the long-time (relaxed) asymptote of that arm.

## Reproduction

```bash
PYTHONPATH=src python scripts/verify_bond_quasistatic_rate.py
PYTHONPATH=src python scripts/render_ice_bond_quasistatic_doc.py
PYTHONPATH=src python -m pytest tests/test_bond_quasistatic.py -q
```
