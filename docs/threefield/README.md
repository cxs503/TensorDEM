# Three-field coupling: ice – hull – propeller (TensorDEM `threefield`)

This directory documents the `tensordem.threefield` package: a DEM (hull–ice) /
LBM (propeller race) / thermodynamic (fragment melting) coupling for the
question *"what does the propeller race do to the ice debris that the hull has
already broken?"*

## Module map

| module | role | evidence |
|---|---|---|
| `resistance.py` | level-ice resistance baselines (Lindqvist-style three-component, Riska-style speed-linear) plus the band they span | REAL (analytic), provenance caveat below |
| `wake_lbm.py` | D3Q19 lattice Boltzmann, Smagorinsky subgrid closure, Guo-forced actuator disk, free-slip lateral walls, time averaging, Gaussian self-similar fit | REAL |
| `fragmentation.py` | turbulent bending-stability criterion + mass-conserving fragmentation cascade + Weibull statistics | REAL (criterion), assumption-limited (kernel) |
| `thermal.py` | Ranz–Marshall convection, Stefan interface condition, closed-form melting solution | REAL |
| `hull_ice.py` | moving-wedge bow DEM (`dem3d_wedge`) with exposed contact-pressure field and connected-component fragment census | REAL solver, MOCK-strength material (see limitations) |
| `coupling.py` | the coupled pass: inventory → fragmentation → melting → mass ledger, plus downstream ship impact | REAL |

## Reproducing the study

```bash
python scripts/run_threefield_coupling.py --stage dem --dem-steps 1500 --dem-thickness 0.30
python scripts/run_threefield_coupling.py --stage dem --dem-steps 1500 --dem-thickness 0.45
python scripts/run_threefield_coupling.py --stage dem --dem-steps 1500 --dem-thickness 0.60
# lattice-Boltzmann race: one chunk per invocation (checkpointed)
python scripts/run_threefield_coupling.py --stage lbm --lbm-chunk 600 --lbm-steps 2100   # x4
python scripts/run_threefield_coupling.py --stage couple
python scripts/plot_threefield_coupling.py
```

Artifacts: `artifacts/threefield/{dem_sweep,wake_lbm,threefield_report}.json` and
`artifacts/threefield/figures/*.png`. Tests: `python -m pytest tests/test_threefield.py`.

## Equations as implemented

**Level-ice baselines** (`resistance.py`), per unit bow width where noted:

```
R_crush = 0.5 * sigma_c * h^2 * (tan(alpha) + mu / cos(alpha))            [N/m]
R_bend  = 0.25 * sigma_f * B * h^2 / L                                   [N/m]
R_sub   = rho_w * g * h^2 * B * (0.9 + 0.05 * mu * L / B)                [N]
Lindqvist-style: R = R_crush + R_bend + R_sub + 0.5 rho_w B h V^2
Riska-style:     R = (1 + 0.021 phi)(R_crush + R_sub) + R_bend + C2 * V
                 C2 = (1 + 0.063 phi)[0.5 mu rho_w B h sqrt(g h)
                                      + sigma_f B h^2 / (3 L sqrt(g h))]
```

**Provenance caveat (explicit):** the component *structure* follows the
published Lindqvist (1989) and Riska (1989) level-ice models, but the primary
papers were not reachable from the offline sandbox in which this module was
written. The coefficient set above is therefore *the implemented and documented
set*, guarded only by structural self-tests (component additivity, speed
linearity, monotonicity). The two baselines are consequently used as a **band**,
and every DEM-versus-formula number in the report is quoted against both
baselines. Do not cite this module as a reproduction of the published
predictions.

**Ranz–Marshall + Stefan** (`thermal.py`):

```
Nu = 2 + 0.6 Re^(1/2) Pr^(1/3),  h_c = Nu k_w / D
dD/dt = -2 h_c (T_w - T_m) / (rho_i L_f)   =  -A/D - B/sqrt(D)
A = 4 k_w dT /(rho_i L_f*),  B = 1.2 k_w (u/nu)^(1/2) Pr^(1/3) dT /(rho_i L_f*)
```

with `L_f* = L_f + c_p (T_m - T_i)` when sensible heating is included. The ODE is
separable; `analytic_ransom_marshall_time` integrates it in closed form
(`t = [2s^3/3B - A s^2/B^2 + 2A^2 s/B^3 - 2A^3 ln(A+Bs)/B^4]` with `s = sqrt(D)`)
and `analytic_diameter_ransom_marshall` inverts it by bisection — that is the
reference used to grade the time integration.

**Turbulent floe breakup** (`fragmentation.py`): drag load `q = 0.5 rho_w C_d u'^2 h`
per unit width on a free-ended floe gives a peak bending stress
`sigma_b = 0.375 rho_w C_d u'^2 L^2 / h`, hence

```
L_crit = sqrt(8 sigma_f h / (3 rho_w C_d u'^2))
```

Floes above `L_crit` are split by a mass-conserving cascade (2–3 children, volume
fractions `~U^3`), so the fragment masses sum exactly to the parent mass.

**LBM closure**: single-relaxation-time D3Q19, Guo forcing for the actuator disk,
Smagorinsky eddy viscosity resolved analytically from the non-equilibrium stress

```
tau_eff = 0.5 (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |Pi| / rho)),
Pi_ab = sum_i c_ia c_ib (f_i - f_i^eq)
```

and the empirical race model (used for the profile gate) is the self-similar
Gaussian whose centreline follows from momentum conservation,

```
u(r) = U + du_c exp(-r^2/(2 sigma^2)),  sigma = sigma_0 + beta (x - x_0)
2 pi rho (U + du_c/2) du_c sigma^2 = T  (thrust)
```

## Known limitations

1. The DEM lattice uses the repository's smoke-test micro-parameters
   (`bond_stiffness = 1200 N/m`), i.e. an effective Young's modulus of order
   10^4 Pa against 5·10^9 Pa for sea ice. Absolute resistance transfer is
   therefore not meaningful (see Fig. 1 and the report's criterion (a)).
   Naively scaling the stiffness by the material ratio (×2·10^5) makes the
   lattice fail in a shock at the first contact and was rejected.
2. The LBM race is under-resolved: 10 lattice cells per propeller diameter,
   `Re = 7200`. The resolved turbulence intensity is ~1 % instead of the ~15 %
   of a physical race, and the resolved spreading is ~0 instead of `beta = 0.085`.
   The profile gate therefore tests the Gaussian functional form, and the
   spreading law is used predictively only as an engineering closure.
3. The DEM fragment census (216–432 particles) is dominated by single-particle
   debris; it cannot carry a floe-size statistic. Criterion (c) fails for that
   reason and is reported as such.