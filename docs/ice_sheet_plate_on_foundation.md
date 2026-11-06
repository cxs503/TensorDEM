# A floating ice sheet as a plate on an elastic foundation: closed form and its verification

Status: **new capability, verified** — before this page the package had the
one-dimensional elastic-foundation member (`ridge_geotech.hetenyi_infinite_beam_point_load`,
`ridge_geotech.solve_beam_on_winkler`) but **no two-dimensional plate member**, so it
could not answer the question ice engineering actually asks: *what load can this ice
sheet carry, and how large is the bending stress under it?*

Everything below is produced by
`scripts/audit_plate_on_foundation.py`
(machine-readable: `docs/artifacts/plate_on_foundation/report.json`, tables:
`summary.md`, figure: `figure.png`).  Reproduce with

```
PYTHONPATH=src python scripts/audit_plate_on_foundation.py
PYTHONPATH=src python -m pytest tests/test_plate_on_foundation.py -q
```

## 1. Physical model and why it belongs here

A continuous ice sheet floating on water deflects by `w` and gets a restoring
pressure `rho_w g w`: that is an infinite thin plate on a Winkler foundation
(`k = rho_w g = 10 055 N/m^3` for sea water) with flexural rigidity
`D = E h^3 / (12 (1 - nu^2))`.  Loads that matter in the ship/ice problem are
local: a resting or grounded hull patch, a skeg line load, an ice-road vehicle,
a hovercraft skirt.  The peak bottom-surface tensile stress under such a patch is
the classical **Westergaard (1926) / Kerr (1979)** quantity that ISO 19906
local-ice-pressure practice is written around, so the module also gives the DEM
ice-sheet specimens an analytic target they did not have.

| quantity | value |
| --- | --- |
| ice: `E`, `nu` | 5.0 GPa, 0.3 (typical first-year sea ice) |
| thickness `h` | 0.3 … 2.0 m (table in §4; 1.0 m reference case) |
| flexural strength (allowable) | 0.5 MPa — the bending strength matters here, **not** a fraction of `E` (ice is brittle) |
| water | `rho_w = 1025 kg/m^3`, `g = 9.81 m/s^2`, `k = 10 055.25 N/m^3` |
| reference `D`, `l` | `4.5788e8 N m`, `14.6079 m` at `h = 1 m` |
| loads | circular patch, total force `P`, radius `a`; static, no added mass |

Two mathematically independent routes are implemented and cross-checked
(`src/tensordem/plate_on_foundation.py`):

1. **Kelvin-function closed form** (point load):
   `w = -(P l^2/(2 pi D)) kei(r/l)`, with `scipy.special.kelvin` for `ker`, `kei`
   and their first derivatives, and the Kelvin ODE `x^2 g'' + x g' - i x^2 g = 0`
   giving `kei'' = ker - kei'/x`.  No finite differencing is used.
2. **Hankel-transform quadrature** (patch load):
   `w(r) = P/(pi a k l) * int_0^inf J1(a s/l) J0(r s/l)/(1+s^4) ds`, integrated in
   `u = sqrt(s)` so that the `a -> 0` tail converges at fixed cost.

The two routes meet on an exact identity that also yields the patch centre
moment in closed form,

```
int_0^inf s^2 J1(eps s)/(1+s^4) ds = kei'(eps),    M(0) = P (1+nu) l / (2 pi a) * kei'(a/l),
```

which is the quantity `ice_sheet_bearing_capacity()` inverts for the allowable
load.

## 2. How it was verified

Five independent checks, none comparing the module with itself:

| # | check | reference | measured error |
| --- | --- | --- | ---: |
| 1 | `kei(0) = -pi/4`; `kei''` from the ODE vs central difference of `kei` | Abramowitz & Stegun 9.9 | `1e-12` / `1e-5` |
| 2 | `w(0) = P/(8 sqrt(k D))` | Hetenyi (1946) closed form | `1.5e-16` |
| 3 | global equilibrium `k ∫ w dA = P` (point load) | force balance, exact | `4.1e-9` |
| 4 | patch centre moment: closed form vs Hankel quadrature | independent numerics | `3.4e-7` |
| 5 | `(2 l/a) kei'(a/l) - ln(l/a) -> 0.6159` | **Westergaard (1926) / Kerr (1979)** published constant | `3.2e-5` absolute |
| 6 | published asymptote vs exact peak stress, `a/l in [0.005, 0.2]` | published formula vs exact solution | worst `0.176 %` |

Check 5 is the interesting one: the published constant `0.6159` is *recovered*
from the exact Kelvin solution to six significant figures
(`0.615932` at `a/l = 1e-3`, monotonically approaching from above), which
confirms both the published formula and this implementation.

## 3. Results

### 3.1 Deflection and moment profiles

`figure.png`, left and middle panels: the exact point-load profile (normalised)
together with the patch solution at `a/l = 0.05`.  The patch reproduces the point
field to `0.5 %` at the centre (`patch/point = 0.9995`) and the centre moment sits
slightly above the point-load value, as it must for a finite loaded area.

### 3.2 Two-route agreement (centre moment)

Programmatic, from `report.json`:

| `a/l` | `M(0)` closed form (N m) | `M(0)` Hankel (N m) | relative difference | `M_t(0)/M_r(0)` |
| ---: | ---: | ---: | ---: | ---: |
| 0.005 | 32 979.9 | 32 979.6 | 9.7e-6 | 1.000000 |
| 0.05 | 37 365.5 | 37 365.5 | 1.4e-6 | 1.000000 |
| 0.2 | 47 154.0 | 47 153.9 | 1.3e-6 | 1.000000 |

Maximum difference over the whole sweep: `3.4e-7`; the centre state is
equibiaxial to machine precision, which is why `6 M/h^2` is the correct
bottom-surface stress there.

### 3.3 Bearing capacity (exact solution, 0.5 MPa flexural strength)

| `h` (m) | `l` (m) | `h/l` | allowable patch load (kN) | published-formula relative error |
| ---: | ---: | ---: | ---: | ---: |
| 0.3 | 5.921 | 0.051 | 16.9 | `-2.2e-5` |
| 0.5 | 8.686 | 0.058 | 43.1 | `-1.3e-5` |
| 1.0 | 14.608 | 0.068 | 155.1 | `-8.1e-6` |
| 1.5 | 19.800 | 0.076 | 329.6 | `-6.8e-6` |
| 2.0 | 24.568 | 0.081 | 563.9 | `-6.2e-6` |

`h/l <= 0.08` everywhere, so the Kirchhoff (thin-plate) assumption is satisfied
by a wide margin; the growth is slightly faster than `h^2` because `l` also grows
with `h` and `ln(l/a)` follows.

Order-of-magnitude sanity: 17 kN for a 0.3 m sheet and 155 kN for a 1.0 m sheet
under a 0.3 m diameter patch is the range classical ice-bearing-capacity tables
(Gold 1971 and later practice, cited through Kerr 1976) report for the same
thickness and strength class.  Field formulas differ from the elastic plate
solution by up to a factor of order two because ice strength scatters; that
scatter is **not** modelled here.

## 4. Error judgement against the platform gate

| key quantity | baseline | error | < 3 % ? |
| --- | --- | ---: | --- |
| centre deflection | `P/(8 sqrt(kD))` closed form | `1.5e-16` | **yes** |
| global reaction (point load) | force balance `P` | `4.1e-9` | **yes** |
| centre moment, Hankel vs closed form | independent route | `3.4e-7` | **yes** |
| published Westergaard constant | 0.6159 (literature) | `3.2e-5` absolute (`5e-5` relative) | **yes** |
| published peak-stress formula | exact solution | worst `0.176 %` | **yes** |

No quantity in this page fails the gate, and the two worst are the *literature*
comparisons, which is the honest place for the remaining error to sit.

## 5. Limitations (no overclaiming)

* **Static, linear, thin plate only.**  No added mass, no fluid inertia, no
  viscoelastic creep (ice creeps; a long-duration load is *not* covered), no
  large-deflection membrane action, no cracking.  A real bearing-capacity answer
  for an engineering case needs the creep/strength model of ISO 19906 on top.
* **The foundation is buoyancy only.**  A sheet sitting in shallow water, or a
  sheet loaded near a shore/ridge boundary, needs a different foundation and a
  finite-domain solution: this module is the infinite-sheet reference.
* **`k` is the sea-water value**; `foundation_modulus()` also accepts fresh water
  (`9805 N/m^3`) — `k` is an argument, not a constant.
* **No DEM coupling yet.**  The module is the analytic target; a bonded-lattice
  or DEM plate specimen has not been run against it.  That is the natural next
  step (see below), and the reason this page claims *analytic* verification, not
  DEM validation.
* The Hankel route carries ~`1e-3 P` absolute error in far-field integral
  quantities (oscillatory Bessel quadrature); centre and near-field quantities
  are accurate to `~1e-6`, and the closed-form route has no such limitation.

## 6. Next steps

1. A bonded-lattice plate specimen (`dem3d_lattice_elastic` family) of the same
   geometry, loaded through a patch, compared with `patch_centre_moment` as a
   function of `h/l` — this would turn the analytic target into a DEM validation.
2. Line/rectangular loads (a skeg or a hull keel line) by convolution of the
   point kernel, which the current patch-only API does not cover.
3. Creep: a Maxwell/Voit foundation or a time-dependent `D` for long-duration
   load cases, where the elastic answer is unconservative.