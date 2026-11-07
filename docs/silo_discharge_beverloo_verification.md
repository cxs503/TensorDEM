# Beverloo silo-discharge benchmark (`silo_discharge`) — independent verification

**Task**: m6296 · **Scope**: `src/tensordem/silo_discharge.py`, `src/tensordem/granular3d_core.py`
(module imported from merge `30f1715`, PR #399 / m5481) · **Evidence grade after this work**: the
module's own claim ("REAL 3-D DEM, reproduces Beverloo with C ≈ 0.58, k ≈ 1.4") is **NOT
substantiated** → the capability is downgraded from REAL to **WITHHELD**, and three concrete
contact-law defects had to be fixed before the module could run at all.

---

## 1. Case introduction — what was supposed to be verified

`tensordem.silo_discharge` (m5481) advertises a real 3-D DEM silo: a cylindrical silo with a flat
bottom plate and a centred circular orifice drains under gravity, the particles are integrated with
the `granular3d_core` kernel (linear spring-dashpot normal, Cundall–Strack tangential, capped
rolling), and the steady mass flow rate is compared with the Beverloo correlation
(Beverloo, Leniger & van de Velde, 1961)

```
W = C · rho_b · sqrt(g) · (D - k·d)^(5/2)          C ~ 0.58, k ~ 1.4   (glass beads)
```

At the time of this verification the module had **no test, no documentation and no evidence
artefact** anywhere in the repository: a merged "REAL" claim with `MOCK`-grade backing. That is the
verification debt this task pays.

Geometry and parameters used here (all SI; `scripts/m6296_silo_beverloo_sweep.py`):

| quantity | value |
|---|---|
| particle diameter `d` | 0.015 m, density 2500 kg/m³ (glass-like) |
| silo radius / diameter | 0.11 m / 0.22 m |
| fill height `H` | 0.12 m (FILL) and 0.21 m (head case) |
| orifice diameter `D` | 3d, 4d, 5d = 0.045, 0.060, 0.075 m |
| `D_silo/D` | 4.9 / 3.7 / 2.9 |
| friction `mu` / rolling `mu_r` | 0.4 / 0.02 |
| restitution | 0.3 |
| contact stiffness `kn` | 1.0e5 N/m (`dt = 4.669e-5 s`, 20 steps per contact period) |
| wall friction | 0.0 (smooth side wall), plate thickness 0.005 m |
| particles | 1685 (FILL), 2859 (head) — HCP column, jitter 0.002 r |
| measurement | 5 windows after a 0.25 s settle; `W` = mean, ± 1σ across windows |

## 2. Method / toolchain

1. **Read the code first** and drive it with the module's own entry points
   (`SiloCase`, `simulate_silo`, `beverloo_fit`, `beverloo_prediction`) — no formula shortcut, no
   re-implementation of the flow rate.
2. The module **diverged** on the first attempt (non-finite positions / `ValueError: floating point
   overflow` inside `cKDTree.query_pairs` within a few dozen steps). Section 5 lists the three
   defects found and fixed; each fix is pinned by a regression test.
3. `scripts/m6296_silo_beverloo_sweep.py` runs the matrix {D/d = 3, 4, 5} + a gravity case
   (`g/4`) + a fill-height case (`1.75 H`) and writes one JSON per case under `runs/m6296_silo/`.
   Executed on the 702 cluster (node n315, loongarch64) via `run_cluster`; wall clock 686 / 321 /
   255 / 295 / 526 s.
4. `scripts/m6296_beverloo_analysis.py` reads those JSONs, refits `(C, k)` with the module's own
   `beverloo_fit`, and writes `runs/m6296_silo/summary.json` plus the four figures below. Every
   digit in this document is copied from that programmatic output (no hand-copied numbers).
5. `tests/test_silo_discharge_beverloo.py` — 12 tests (plate geometry, tangential sign, spin-energy
   monotonicity, Beverloo inversion round-trip, negative control, end-to-end sanity):

```
$ PYTHONPATH=src python -m pytest tests/test_silo_discharge_beverloo.py -q
12 passed in 12.04s
```

## 3. Results

### 3.1 Flow rate vs the literature law

| case | D/d | H/D | n₀ | discharged | ρ_b (kg/m³) | W (kg/s) | W_lit (kg/s) | rel. err vs lit. |
|---|---|---|---|---|---|---|---|---|
| D3 | 3.00 | 2.67 | 1685 | 13 | 1632 | 0.0383 ± 0.0856 | 0.2645 | **−85.5 %** |
| D4 | 4.00 | 2.00 | 1685 | 19 | 1632 | 0.1399 ± 0.3128 | 0.8905 | **−84.3 %** |
| D5 | 5.00 | 1.60 | 1685 | 47 | 1632 | 0.4614 ± 1.0318 | 2.0088 | **−77.0 %** |

Fitted coefficients (module's `beverloo_fit`, k-grid 1.0–2.0): **C = 0.1569, k = 1.775**
(literature 0.58 ∈ [0.55, 0.60] and 1.4 ∈ [1.2, 1.6] — both **outside** the literature band);
power-law slope of `W` vs `D` = **4.855** (target 2.5).

### 3.2 Where the numbers come from — the flow is not steady

The per-window series exposes the reason (figure 3): discharge happens **only during the first
window**, all later windows are exactly zero in all three orifice cases
(`D3: [0.191, 0, 0, 0, 0]`, `D4: [0.699, 0, 0, 0, 0]`, `D5: [2.307, 0, 0, 0, 0]` kg/s).
The column above the orifice **arches over** after a start-up burst of 13–47 particles (a few percent
of the 1685-particle fill) and the discharge stops. The reported `W` is therefore a
start-up-burst average, *not* a steady Beverloo rate; the large ±σ (larger than the mean itself)
says the same thing.

### 3.3 Auxiliary checks (also non-discriminating, because of 3.2)

| check | measured | expected | verdict |
|---|---|---|---|
| gravity scaling, `g/4` vs `g` (D/d = 4) | W ratio **1.05** | 0.50 (`W ∝ sqrt g`) | FAIL |
| fill height 1.75 H vs H (D/d = 4) | W change **+5.3 %** | < ±10 % (head independence) | pass, but weak (burst-limited) |

### 3.4 Field snapshot

Figure 4 (`docs/figures/m6296_silo_field_snapshot.png`) is a vertical `x–z` cross-section of the D4
run at t = 0.55 s, particles coloured by speed. Only the plug above the orifice is in motion
(|v| ≤ ~1 m/s) while the bulk stays frozen — the visual counterpart of the stall in 3.2.

### Figures

* `docs/figures/m6296_silo_flow_vs_orifice.png` — W(D/d) with the fitted and the literature
  Beverloo curves.
* `docs/figures/m6296_silo_loglog_exponent.png` — log–log W vs (D − k d)/d with the 5/2 reference.
* `docs/figures/m6296_silo_windows.png` — per-window W/W_mean (start-up burst, then zero).
* `docs/figures/m6296_silo_field_snapshot.png` — DEM speed field at t = 0.55 s.

## 4. Error judgement (explicit)

| quantity | DEM | literature / target | relative error | < 3 % ? |
|---|---|---|---|---|
| `C` | 0.1569 | 0.58 (0.55–0.60) | −72.9 % | **No** |
| `k` | 1.775 | 1.4 (1.2–1.6) | +26.8 % | **No** |
| exponent on D | 4.855 | 2.5 | +94 % | **No** |
| W (D/d = 3 / 4 / 5) | 0.0383 / 0.1399 / 0.4614 kg/s | 0.2645 / 0.8905 / 2.0088 kg/s | −85.5 / −84.3 / −77.0 % | **No** |
| W(g/4)/W(g) | 1.05 | 0.50 | +110 % | **No** |

**No key quantity is within 3 %; the deviations are 77–110 %, and the underlying quantity is not
even a steady flow rate.** The verdict is an honest FAIL: `silo_discharge` is a *working, stable
3-D DEM silo* after the fixes below (it integrates 1685–2859 particles for 15 000–37 500 steps
without divergence and discharges mass through the orifice), but it does **not** reproduce the
Beverloo law, and the `C ≈ 0.58, k ≈ 1.4` claim in its docstring must be treated as **WITHHELD**
until an ordered-lattice-independent, steady-discharge configuration exists.

## 5. Defects found and fixed (the real increment)

1. **Rim-cylinder contact on the plate was an infinite cylinder.**
   `edge_hit = (|radial − a| < r) & (−t − r < z < r)` also matched particles resting on the plate
   **top** (`z ≈ +r`) within one radius of the rim: they received `kn·(r − |radial − a|)` up to
   ~250 N sideways on a 1.3 g particle (measured 232 N at step 3), which launched particles at
   4.6 m/s and blew the run up within a few dozen steps. Fixed with a *nearest-feature* plate model:
   top face, underside, edge cylinder (only for `−t ≤ z ≤ 0`) and the two sharp rim circles at
   `(a, 0)` and `(a, −t)`, each gated on the side of the plate its centre is on.
2. **The "underside" face was applied inside the aperture and without a `z` gate.**
   `(radial < a)` is the *hole* — there is no material there — and a particle one radius under the
   plate read a penetration of `r + |z| ≈ 15 mm` → 1.5 kN push. Fixed to
   `(radial ≥ a) & (z ≤ −t)`; the top face is gated to `z ≥ 0`.
3. **Tangential surface-velocity sign inverted** (`granular3d_core.contact_forces`): the rotational
   part used `+(omega_j + omega_i) × (r n)` where the correct relative surface velocity is
   `−(omega_j + omega_i) × (r n)`. The error pumps energy into the spin instead of damping it; it
   is invisible for pure translation, which is why it survived review. Fixed, and pinned by a sign
   test plus a spin-energy monotonicity test.

Regression coverage: `tests/test_silo_discharge_beverloo.py` (12 tests) passes on the fix. Defects
1–2 are *measured* on the published code (first divergence at integration step 3 by a 232 N spurious
side force; the run reaches non-finite coordinates inside 3000 steps), and the tests
`test_plate_top_face_does_not_kick_particles_sideways`,
`test_aperture_is_open_below_the_plate` and `test_particle_far_above_plate_feels_nothing` encode
exactly those numbers. Defect 3 is a sign error proven analytically in the test docstring; restoring
the old expression flips the sign asserted by
`test_tangential_contact_opposes_the_spin_pair` / `test_tangential_contact_does_not_inject_energy_into_spin`
(not re-executed against a restored copy: mutating the tracked sources in place is blocked by the
sandbox policy). Blast radius checked: `granular3d_core` is imported only by
`silo_discharge` (no other module or test in the repository).

`simulate_silo(..., return_history=True)` now also returns `velocities` (needed for the field
figure); the default path is unchanged.

## 6. Pitfalls / limitations / open items

* **Ordered initial lattice arches.** `hcp_positions_in_cylinder` with `packing_gap = 0` and
  `jitter = 0.002 r` starts from a near-perfect crystal; a 3–5 d circular orifice is spanned by an
  arch almost immediately. A disordered (rain-deposited or strongly jittered) initial packing is
  required for a steady discharge. The jitter route is **not** a drop-in fix: with
  `jitter = 0.12 r, packing_gap = 0.02` the run **still diverges** (same overflow), i.e. there is a
  second, unrepaired instability under strong initial overlaps (initial `kn·δ ≈ 120 N` contacts).
  Reproduce with
  `PYTHONPATH=src python -c "... SiloCase(jitter=0.12, packing_gap=0.02) ..."`; the reproducer is
  `/tmp/jit.py` logic in this task's log.
* **Wall friction is kinetic, not Cundall–Strack.** Plate/wall friction is
  `mu·|F_n|·v_t/|v_t|` (no static branch, no spin coupling); irrelevant here (`wall_mu = 0`) but it
  blocks a faithful rough-wall silo.
* **`rho_b` is a geometric bulk density** (`mass·n₀/(πR²H)`): 1632 kg/m³ (packing fraction 0.653).
  It is used only inside the Beverloo prefactor, so a bias there moves `C` directly.
* **Small system.** 1685 particles is at the low end for silo statistics; window counts of 13–47
  discharged particles carry 15–30 % Poisson noise. The deviation measured here (−77…−86 %) is far
  larger than that, so the FAIL conclusion is not noise-limited.
* **Not addressed**: rough walls, non-spherical particles, hopper (non-flat) bottoms, and the
  `mu(I)` direct-shear companion the module docstring mentions.

## 7. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_silo_discharge_beverloo.py -q      # 12 tests
PYTHONPATH=src python scripts/m6296_silo_beverloo_sweep.py                    # 5 cases, ~35 min 1 core
PYTHONPATH=src python scripts/m6296_silo_snapshot.py                          # field snapshot
PYTHONPATH=src python scripts/m6296_beverloo_analysis.py                      # tables + figures
```

Machine-readable evidence: `runs/m6296_silo/{D3,D4,D5,grav,head}.json`, `runs/m6296_silo/snapshot.json`,
`runs/m6296_silo/summary.json` (written by the analysis script; includes git sha, host, wall time,
per-window series and the fitted coefficients).