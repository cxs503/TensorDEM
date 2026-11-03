# 3-D spherical-particle buoyancy: kernel + closed-form verification

Status: **REAL** (analytic geometry kernel, closed-form anchors, one-dimensional
dynamics). Scope: force kernel and its verification; the DEM3D time loop is not
modified in this increment.

## 1. Case introduction (why this module exists)

`tensordem.buoyancy_wet` (merged as SI-D, commit `e2baeab`) supplies buoyancy
for the **2-D** extruded-disk model: the submerged *area* fraction comes from
circular-segment geometry and is multiplied by an extrusion thickness. The
**3-D** bonded-sphere engine (`tensordem.dem3d.IceDEM3D`, `DEM3DConfig`) has no
gravity term and no buoyancy at all, so a 3-D particle cannot float. The obvious
shortcut — reuse the 2-D kernel for a sphere — is geometrically wrong, and this
document quantifies by how much.

Physical parameters of the verification case:

| parameter | value | note |
|---|---:|---|
| sphere radius R | 0.05 m | representative ice floater |
| water density rho_w | 1025 kg/m3 | sea water |
| ice density rho_i | 917 kg/m3 | density ratio r = 0.8946 |
| gravity g | 9.81 m/s2 | |

Geometry kernel. With the sphere centre at elevation `h` above the free surface,
the sphere occupies `z in [h-R, h+R]`, the water occupies `z <= 0`, and in
centre-relative coordinates `q = z - h` the submerged cap is `q in [-R, -h]`:

```
V_sub(h) = int_{-R}^{-h} pi (R^2 - q^2) dq = pi (2R^3/3 - R^2 h + h^3/3)
f(h)     = V_sub / (4/3 pi R^3) = (R - h)^2 (2R + h) / (4 R^3)
-dV_sub/dh = pi (R^2 - h^2) = A_wp(h)                      (waterplane identity)
K        = -dF_b/dh = rho_w g A_wp(h)                       (restoring stiffness)
```

Archimedes equilibrium `f(h_eq) = rho_i/rho_w` becomes, with `u = h_eq/R`, the
cubic `u^3 - 3u + 2 - 4r = 0`. Its derivative `3u^2 - 3 <= 0` on `[-1, 1]`, so
the root is unique and a bracketed bisection is exact and branch-free; the
polynomial root from `numpy.roots` is used as the independent cross-check.

## 2. Method

New public module `src/tensordem/buoyancy_wet3d.py`:

| entry point | purpose |
|---|---|
| `submerged_volume_sphere(h, R)` | integrated cap volume, SI m3 |
| `submerged_fraction_sphere(h, R)` | volume fraction (principal kernel) |
| `submerged_fraction_sphere_factored(h, R)` | second algebraic route, cross-check only |
| `waterplane_area_sphere(h, R)` | `pi (R^2 - h^2)`, used by the derivative identity |
| `hydrostatic_stiffness_sphere(h, R, rho_w, g)` | `rho_w g A_wp`, analytic K |
| `equilibrium_centre_height` / `equilibrium_draft` | Archimedes inversion by bisection |
| `submerged_fraction_sphere_tensor` | vectorised kernel for particle batches |
| `compute_buoyancy_force_3d`, `compute_wet_drag_force_3d`, `compute_wet_forces_3d` | `(N,3)` force drops for `IceDEM3D` |

Configuration reuses `WetCouplingConfig` (`enabled`, `water_density`, `gravity`,
`water_level`, `wet_drag_coeff`); the 2-D `thickness` field is deliberately not
used, because a sphere has no extrusion depth.

Reproduction:

```
PYTHONPATH=src python scripts/audit_dem3d_buoyancy_float.py
PYTHONPATH=src python -m pytest tests/test_buoyancy_wet3d.py -q
```

The audit script is analytic/one-dimensional only (401-point geometry grid,
20 001-point Simpson quadrature, two ~1e5-step scalar ODE runs) and finishes in
seconds; it writes `docs/dem3d-buoyancy-float/metrics.json`, four PNG figures and
the machine-generated `REPORT.md`. Every number quoted below is read from that
file — no value is transcribed by hand.

## 3. Results

### 3.1 Geometry kernel vs independent routes (R = 0.05 m, 401 samples)

| check | measured | requirement | verdict |
|---|---:|---:|---|
| max rel. error vs 20 001-point Simpson quadrature | 3.16e-11 | < 1e-8 | pass |
| max abs. difference between two algebraic factorisations | 6.66e-16 | < 1e-12 | pass |
| max rel. error of `-dV/dh` vs waterplane area | 1.51e-08 | < 1e-6 | pass |
| max abs. difference scalar vs tensor route | 6.66e-16 | < 1e-14 | pass |
| monotone decreasing in `h` | true | required | pass |
| `f(0)` / `f(R)` / `f(-R)` | 0.5 / 0 / 1 | exact | pass |

### 3.2 Archimedes equilibrium and force balance

| rho_p [kg/m3] | r | h_eq/R | draft [m] | residual `f(h_eq)-r` | |u diff| vs poly root | `F_b/F_g` rel. err |
|---:|---:|---:|---:|---:|---:|---:|
| 330 | 0.3220 | +0.242130 | 0.037893 | -1.7e-16 | 3.6e-16 | 5.2e-16 |
| 500 | 0.4878 | +0.016262 | 0.049187 | +3.9e-16 | 4.5e-16 | 8.7e-16 |
| 700 | 0.6829 | -0.249052 | 0.062453 | -8.9e-16 | 1.5e-15 | 1.2e-15 |
| 850 | 0.8293 | -0.474675 | 0.073734 | +1.3e-15 | 1.8e-15 | 1.4e-15 |
| 917 (ice) | 0.8946 | -0.597162 | 0.079858 | -5.6e-16 | 1.3e-15 | 7.5e-16 |
| 1000 | 0.9756 | -0.813795 | 0.090690 | +5.6e-16 | 1.9e-15 | 6.9e-16 |

### 3.3 One-dimensional floating dynamics

Damped relaxation from a fully submerged start (`dt` = 2e-4 s, 6 s, `C_wet` =
2 N·s/m) and an undamped small oscillation (`dt` = 2e-5 s, amplitude 0.01 R):

| quantity | value |
|---|---:|
| particle mass | 0.480140 kg |
| analytic `h_eq` | -0.029858 m |
| final `h` | -0.029858 m |
| `|h_final - h_eq| / R` | 4.68e-06 |
| analytic period `2 pi sqrt(m/K)` | 0.610777 s |
| measured period (zero crossings) | 0.610800 s |
| period rel. error | 3.74e-05 |

### 3.4 Cross-dimension trap: the 2-D disk kernel must not be reused

| h/R | f_2D (disk) | f_3D (sphere) | rel. error |
|---:|---:|---:|---:|
| -0.90 | 0.981307 | 0.992750 | 1.15 % |
| -0.50 | 0.804499 | 0.843750 | 4.65 % |
| 0.00 | 0.500000 | 0.500000 | 0.00 % |
| +0.25 | 0.342519 | 0.316406 | 8.25 % |
| +0.50 | 0.195501 | 0.156250 | 25.12 % |
| +0.90 | 0.018693 | 0.007250 | 157.83 % |
| ice equilibrium (-0.597) | 0.856174 | 0.894634 | 4.30 % |

The error is **asymmetric**: both kernels are normalised to their own total
measure, so the discrepancy is negligible for deeply immersed particles and
explodes for shallow ones (25 % at `h = +R/2`, 158 % at `h = +0.9R`). Propagated
to an engineering quantity, substituting the 2-D kernel at the ice equilibrium
gives `h_eq = -0.033777 m` instead of `-0.029858 m`, i.e. a draft **4.91 % too
deep** — below the 15 % acceptability gate but well above a 3 % tolerance, and
useless for shallowly immersed debris where the 15 % gate is violated outright.

## 4. Figures

* `geometry_kernel.png` — 3-D volume fraction vs the 2-D area fraction with the
  ice/water density line, plus the `-dV_f/dh` vs `A_wp/pi R^2` overlay that
  verifies the kernel derivative.
* `buoyancy_equilibrium.png` — `F_b(h)` curve with the three equilibrium points
  (rho_p = 330, 700, 917) where the curve crosses each particle weight.
* `relaxation.png` — 1-DOF relaxation settling onto the analytic draft.
* `cross_dimension_error.png` — 2-D-kernel misuse error per immersion depth
  against the 15 % gate.

## 5. Notes, limitations, applicability

* **The kernel is not yet wired into `IceDEM3D.step`.** `DEM3DConfig` has no
  gravity entry, and adding gravity + buoyancy to the solver changes every
  existing DEM3D benchmark trajectory. That coupling is a separate, heavier
  increment; this delivery provides the verified force drops and the analytic
  anchors they will be scored against.
* The verification is **geometric and hydrodynamic equilibrium only**. No
  experiment or literature curve is compared, because the kernel is closed-form:
  the exact reference is the analytic integral, and the independent routes are
  quadrature, a second factorisation and a polynomial root. Dynamic added mass,
  wave radiation and free-surface deformation are outside the model
  (`F_wet = -C_wet f_sub v` is a linear damping placeholder, not a drag law).
* `water_level` is a flat free surface; sloped/curved surfaces are not supported,
  and no inter-particle buoyancy interaction (e.g. added mass of a cluster) is
  modelled.
* Rolling resistance and rotational buoyancy moments are absent, consistent with
  the current solver's translational-only particle state.
* The bisection is bounded to 200 iterations and stops at 1e-14 relative
  interval width; at `tol = 1e-14` and R = 0.05 m the residual is already at
  machine precision (1e-16), so tightening it buys nothing.

## 6. Error verdict against analytic references

| key quantity | reference | measured deviation | < 3 % ? |
|---|---|---:|---|
| submerged volume fraction `f(h)` | Simpson quadrature of the exact integral | 3.16e-11 rel. | yes |
| derivative `-dV/dh` | waterplane area `pi(R^2-h^2)` | 1.51e-08 rel. | yes |
| equilibrium elevation `h_eq` | polynomial root of the cubic | 1.3e-15 (in `u`) | yes |
| buoyancy force at equilibrium | particle weight `rho_p g V` | 1.4e-15 rel. | yes |
| damped steady state | analytic draft | 4.68e-06 (in `R`) | yes |
| oscillation period | `2 pi sqrt(m/K)` | 3.74e-05 rel. | yes |

All checked quantities are at machine precision, i.e. far inside the 3 % bar.
The only non-trivial number in this document is the **2-D kernel misuse error**
(25.12 % shallow, 4.91 % on the ice draft): that is a modelling trap being
quantified, not a residual of the new kernel.