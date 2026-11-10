# Ship-ice collision energy-conservation surface (IN63-N01)

Structural half of an ice-region ship-ice collision energy balance: rigid-plastic
hinge absorption, serial double-hull allocation, penetration depth from energy
conservation, and a Minorsky (1959) empirical cross-check. Everything here is SI
(m, N, J, Pa, kg) and every number in this document is produced by
`scripts/benchmark_hull_collision_energy.py` into
`runs/hull_collision_energy/report.json` (canonical content SHA-256
`4ba00f6b87f1e2739fd4c92f256fe1eb8085bf75f0fa71dcf8d3e34bde0fff2e`).

**Verdict of this run: `PASS`** (failed checks: `[]`).

## 1. What problem this solves

A ship striking ice (or a boulder in ice) must be shown to survive the hit: the
collision kinetic energy has to go somewhere before the hull is breached through
the inner shell, and the resulting damage geometry (length, width, depth) is the
input for SOLAS damage-stability. The platform previously had no verifiable
statement of that energy path. This node provides one in a form a machine can
falsify: a closed-form hinge anchor, a residual that is reported rather than
assumed, a layer-allocation identity checked by an independent quadrature, a
`V = 0` negative control, and a comparison with the classical empirical law.

## 2. Model

Side-shell bay between two transverse frames, span `s`, plating thickness `t`,
`N` longitudinals of plastic modulus `Z`, yield stress `sigma_y`, contact patch
`L_c x W_c`.

Plastic hinge (rigid-perfectly-plastic):

    M_p,plate = sigma_y * t^2 / 4          (per unit plate width)
    M_p,long  = sigma_y * Z                (longitudinal + attached plating)
    theta(d)  = 8 * d / s                  (clamped-clamped three-hinge mechanism)
    E_hinge(d) = (8/s) * ( N*M_p,long + M_p,plate*W_eff ) * d

Membrane (fully plastic stretching of the same bay, mean plastic strain
`(d/s)^2`):

    E_membrane(d) = sigma_y * W_eff * t * d^2 / s

Friction: the normal force is in equilibrium with the structural resistance, so
over the penetration history

    E_friction(d) = mu * E_structural(d)      (work-average normal force N_bar = E_struct/d)

Energy conservation and the layer stack:

    E_k = 0.5 * m_eff * V^2 = E_outer(d) + E_inner(d) + E_friction(d) + E_residual

Outer shell deforms over `0 <= d <= d_1` (`d_1` = distance to the inner shell),
the inner shell engages for `d > d_1` and is declared **breached** once the
indenter has passed its plating thickness. The depth `d` is the root of the energy
balance, found by bracketed bisection (no derivative, no tuning parameter); the
residual is whatever is left over and is reported.

### Design case (scenario data, injected - not library constants)

| quantity | value |
|---|---|
| outer shell | t = 22 mm, sigma_y = 355 MPa |
| inner shell | t = 18 mm, sigma_y = 355 MPa |
| frame spacing | 2.40 m |
| outer surface to inner shell | 2.022 m |
| longitudinal (T 400x11 + 150x15, 600x22 attached plate) | Z = 1.9242e-03 m^3 |
| contact patch | 3.0 m x 3.0 m |
| ice-steel friction | mu = 0.15 |
| striking mass / added mass | 2.00e+06 kg / 0.1 -> m_eff = 2.20e+06 kg |
| outer shell hinge / membrane coefficient | 1.1814e+07 N/m / 9.7625e+06 N/m^2 |
| inner shell hinge / membrane coefficient | 7.8130e+06 N/m / 6.6562e+06 N/m^2 |
| outer / inner / total capacity | 63.80 / 14.47 / 90.01 MJ |

## 3. Acceptance results

| criterion | verdict | numbers |
|---|---|---|
| criterion_a_energy_residual | PASS | max_residual_relative = 5.810e-13, threshold = 2.000e-02 |
| criterion_a_solver_vs_closed_form | PASS | max_relative_error = 4.923e-13, threshold = 1.000e-06 |
| criterion_b_anchor | PASS | max_relative_error = 2.500e-09, threshold = 1.000e-02 |
| criterion_c_negative_control | PASS | depth_at_2ms_m = 2.656e-01, depth_at_zero_velocity_m = 0.000e+00 |
| criterion_d_allocation | PASS | inner_engaged_in_overload = True, inner_intact_flags = [True, False], max_quadrature_relative_error = 2.622e-05, sum_equals_total = True, threshold = 2.000e-02 |
| criterion_e_minorsky | PASS | cases_in_domain = ['sweep_V8', 'moderate_outer_only', 'overload_inner_engaged'], max_relative_error_in_domain = 2.176e-01, threshold = 3.000e-01 |
| criterion_f_monotone_sweep | PASS | depths_m = [0.2655708470581379, 0.7855213211745724, 1.3680756655359643, 1.9711085660682874], strictly_increasing = True |

### (b) analytic anchor - three-point bending, `E = M_p * theta_max`

Section: 600 mm x 22 mm, span 2.4 m, deflection 0.12 m; `M_p` from the fibre
quadrature; `P_u` from virtual work.

| mechanism | theta_max [rad] | P_u [N] | E closed form [J] | E incremental [J] | E = P_u*d [J] | section rel. err | max rel. err |
|---|---|---|---|---|---|---|---|
| simply_supported | 0.200000 | 42955.000 | 5154.600 | 5154.600 | 5154.600 | 2.5e-09 | 2.5e-09 |
| clamped | 0.400000 | 85910.000 | 10309.200 | 10309.200 | 10309.200 | 2.5e-09 | 2.5e-09 |

The closed form is `M_p*theta_max`; the two numerical routes use a fibre-section
integral and a virtual-work collapse load, i.e. they do not reuse `b*t^2/4` or the
kinematic relation they are checking. Agreement is at quadrature precision
(2.5e-09), far inside the 1 % threshold.

### (a)(f) velocity sweep, outer shell only (m_eff = 2.20e+06 kg)

| V [m/s] | E_k [MJ] | d [m] | E_plastic [MJ] | E_friction [MJ] | residual / E_k | E_outer [MJ] | E_inner [MJ] | inner shell | V_damage [m^3] |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.40 | 0.2656 | 3.826 | 0.574 | 5.8e-13 | 3.826 | 0.000 | intact | 0.774 |
| 4 | 17.60 | 0.7855 | 15.304 | 2.296 | 2.6e-13 | 15.304 | 0.000 | intact | 0.774 |
| 6 | 39.60 | 1.3681 | 34.435 | 5.165 | 3.5e-13 | 34.435 | 0.000 | intact | 0.774 |
| 8 | 70.40 | 1.9711 | 61.217 | 9.183 | 7.8e-14 | 61.217 | 0.000 | intact | 0.774 |

Penetration is strictly increasing in `V` and stays below the 2.022 m
double-hull depth, so at these energies the inner shell is intact and the cargo
boundary is not breached. The residual is at bisection precision
(5.8e-13) and is not an
assumed number: the solver output is checked independently against the closed-form
root of `(1+mu)(k1*d + k2*d^2) = E_k`
(max relative difference 4.9e-13).

### (d) double-hull allocation

| case | m [kg] | V [m/s] | E_k [MJ] | d [m] | E_outer [MJ] | E_inner [MJ] | sum [MJ] | Simpson error | inner shell |
|---|---|---|---|---|---|---|---|---|---|
| moderate_outer_only | 2.2e+06 | 8 | 70.4 | 1.9711 | 61.217 | 0.000 | 61.217 | 1.1e-15 | intact |
| overload_inner_engaged | 4.5e+06 | 6 | 81.0 | 2.5931 | 63.803 | 6.632 | 70.435 | 2.6e-05 | BREACHED |

The >= 2 % criterion is met by construction for the *sum* (`E_total = sum_k E_k`
by definition of the model), so it is *not* the interesting number here: the
independent check is the composite-Simpson integral of the resistance curve
`int_0^d (dE/dd') dd'`, computed from the derivative rather than from the energy
expressions (2.6e-05).
The overload case exercises the serial ordering: the outer shell saturates at
63.80 MJ, the inner shell then absorbs
6.632 MJ and is declared breached, which is exactly the
statement needed for damage-stability input.

### (e) Minorsky (1959) cross-check

`V_damage = C * M * V^2 / sigma_y` with `C = sigma_y_ref / (2*47.1 MJ/m^3)`,
`sigma_y_ref = 235 MPa` (C = 2.4947): this reproduces
Minorsky's line exactly at his reference steel, so the comparison is an
independent cross-check, not a fit.

| case | E_k [MJ] | V_model [m^3] | V_Minorsky [m^3] | relative error | fully plastic (d/s >= 0.75) | status |
|---|---|---|---|---|---|---|
| sweep_V2 | 4.4 | 0.774 | 0.062 | +1151.9 % | no | WITHHELD (out of domain) |
| sweep_V4 | 17.6 | 0.774 | 0.247 | +213.0 % | no | WITHHELD (out of domain) |
| sweep_V6 | 39.6 | 0.774 | 0.557 | +39.1 % | no | WITHHELD (out of domain) |
| sweep_V8 | 70.4 | 0.774 | 0.989 | -21.8 % | yes | **in domain** |
| moderate_outer_only | 70.4 | 0.774 | 0.989 | -21.8 % | yes | **in domain** |
| overload_inner_engaged | 81.0 | 1.281 | 1.138 | +12.5 % | yes | **in domain** |

Only cases inside Minorsky's calibration regime (energy in 10-400 MJ **and** a
fully developed mechanism) are judged against the 30 % threshold; the light cases
are reported with their (large) errors and marked WITHHELD. The reason is
physical and predeclared in `prereg/IN63_N01_hull_collision_energy.json`: the
panel-level model pins a whole frame bay as yielded as soon as the mechanism
forms, which at 4 MJ (d/s = 0.33) is a granularity artefact rather than a
statement about damaged steel volume.

## 4. Figures

![figures/fig1_resistance_energy.png](../runs/hull_collision_energy/figures/fig1_resistance_energy.png)
![figures/fig2_penetration_minorsky.png](../runs/hull_collision_energy/figures/fig2_penetration_minorsky.png)
![figures/fig3_energy_partition.png](../runs/hull_collision_energy/figures/fig3_energy_partition.png)
![figures/fig4_hinge_anchor.png](../runs/hull_collision_energy/figures/fig4_hinge_anchor.png)

## 5. Caveats and scope

* Rigid-perfectly-plastic material: no hardening, no fracture/tearing energy, no strain-rate sensitivity (ice collisions are rate sensitive, so absorbed energy is under-estimated).
* The indenter is rigid: ice crushing is not modelled, so the reported penetration is an upper bound for a real ice collision of the same kinetic energy.
* Panel-level granularity: the outer shell bay yields as a whole, so the model cannot resolve partial plasticity of a bay (this is why the Minorsky comparison is restricted to fully plastic cases, see the predeclared domain rule).
* The membrane term is the rigid-plastic limit; for d/span < 0.1 the real response is elastic.
* Friction is taken as mu times the structural dissipation (work-average normal force), a quasi-static idealisation; dynamic friction and the ice contact pressure distribution are not resolved.
* Damage geometry is a mechanism-based equivalent box (length x width x depth), not a measured damage extent; SOLAS input should treat it as a bounding estimate.

## 6. Reproduction

```bash
python -m pytest tests/test_hull_collision_energy.py -q          # 25 tests
python scripts/benchmark_hull_collision_energy.py --output runs/hull_collision_energy
```

Artefacts: `runs/hull_collision_energy/report.json` (all numbers + SHA-256 audit
chain), `sweep.csv`, `figures/fig1..fig4`.
