# Ice-induced whipping: DEM pulses → wet modes → deck stress → fatigue

**Module** `tensordem.whipping` (new) · **Benchmark** `benchmarks/whipping_response.py` ·
**Artifact** `artifacts/whipping.json`, `artifacts/whipping/*.png` · **Tests**
`tests/test_whipping_response.py` (25 items)

This node closes the chain that the DEM ice-load work was still missing: the
dynamic, pulse-by-pulse consequence of an ice-hull contact-force record on the
hull girder, expressed as a whipping contribution to fatigue damage.  It is a
fresh implementation (no reuse of an existing verified case in this repository).

## 1. Case and parameters

| quantity | value | note |
|---|---|---|
| hull girder length `L` | 150 m | free-free beam idealisation |
| waterline breadth / draft | 22 m / 10.5 m | parabolic fore/aft taper |
| block coefficient `Cb` | 0.86 | ice-going bulk-carrier class |
| section inertia `I`, `E` | 30 m⁴, 206 GPa | `EI = 6.18e12 N·m²` |
| deck distance `y` | 7.0 m | deck stress station amidships |
| mass per length | 203.6 t/m | `rho Cb B T` |
| modal damping `zeta` | 1.5 % | all modes |
| full-scale bow load | 15 MN peak | linear rescale of the DEM record |
| DEM lattice | 40 bonded disks, `r`=0.15 m, `t`=0.4 m | reduced prototype |
| DEM hull | V-profile, 1.5 m/s, 2.4 s | 1.23 kHz native, resampled to 1 kHz |

## 2. Method

1. **DEM record** (`dem_load.py`): `IceDEM` with a prescribed bow profile
   indenting a bonded ice sheet; the hull reaction magnitude is sampled at
   1 kHz over 2.4 s (`artifacts/whipping/contact_force_1kHz.csv`, 9 pulse
   events, 120 broken bonds).
2. **Pulse features** (`pulses.py`): events are split at 10 % of the record
   maximum; per event the rise time (10 % level → peak), peak force, FWHM width
   and impulse `I = ∫F dt` are measured, and each event is rebuilt as a
   two-branch `sin^p/cos^p` pulse whose exponent solves `F_peak c_p (t_r+t_d) = I`
   with `c_p = Γ((p+1)/2)/(√π Γ((p+2)/2))` (impulse reproduced to machine
   precision; no quadrature).
3. **Section added mass** (`sections.py`): 2-D constant-panel BEM with the
   high-frequency free-surface condition `φ = 0` imposed *exactly* by the odd
   mirror image of the section; panel integrals are closed-form.  Verified
   against the analytic values for a deep ellipse (`ρπa²`) and a semi-ellipse on
   the waterline (`ρπa²/2`), 0.2 % / 1 % at 80 panels.
4. **Dry and wet modes** (`modes.py`): free-free Hermite beam FEM (60 elements),
   matched to the closed-form free-free frequencies to < 1e-4; added mass
   reduced on the dry basis, wet eigenproblem `(I + ΦᵀM_aΦ)a = ω_dry²a/ω_wet²`.
5. **Modal superposition** (`response.py`): piecewise-exact Duhamel integration
   (exact for a piecewise-linear load, no stability limit); deck stress
   `σ = Σ E y φ_i''(L/2) q_i`.
6. **Rainflow + Miner** (`rainflow.py`, `fatigue.py`): ASTM E1049-85 range-pair
   counting, DNV-RP-C203 curve D in air (`log a1 = 12.164, m = 3` up to
   `N = 1e7`, then `m = 5`; knee 52.6 MPa — asserted in the tests).
7. **Split and controls**: quasi-static `< 1 Hz`, whipping `1–50 Hz`
   (zero-phase Butterworth, 4th order); negative controls are the mode-1-only
   (pseudo-static) model and a `< 0.5 Hz` analysis band.

## 3. Results

| quantity | value | criterion | verdict |
|---|---|---|---|
| dry modes 1–4 | 0.872 / 2.403 / 4.711 / 7.789 Hz | vs closed form | < 1e-4 rel. |
| wet modes 1–4 | 0.688 / 1.834 / 3.569 / 5.854 Hz | — | — |
| wet frequency drop, mode 1 | **21.1 %** | 10–30 % | **PASS** |
| wet frequency drop, mode 2 | **23.7 %** | 10–30 % | **PASS** |
| midship added mass | 269.8 t/m (`m_a/ρB² = 0.544`) | — | — |
| added mass / hull mass per length | 1.32 | — | — |
| pulse rise time, all 9 events | 23 → 186 ms | 10–50 ms | **FAIL** |
| whipping stress share (band split) | **18.3 %** | 20–40 % | **FAIL** |
| whipping stress share (modal split) | 14.8 % | 20–40 % | (cross-check) |
| whipping damage share (removal technique) | **78.6 %** | 15–35 % | **FAIL** |
| whipping damage share (band split) | 2.6 % | 15–35 % | (cross-check) |
| energy ledger residual (2 kHz grid) | **…** < 1e-11 | < 1e-6 | **PASS** |
| negative control: mode-1-only | 0.0 % | < 5 % | **PASS (red)** |
| negative control: `< 0.5 Hz` band | 0.75 % | < 5 % | **PASS (red)** |
| dispersion over the 3 largest pulses | rise 26.6 %, stress 11.1 %, damage 300 % | < 20 % | **FAIL** |
| deck stress | std 42.1 MPa, peak 102 MPa, max rainflow range 150 MPa | plausible | — |

## 4. Honest diagnosis of the four failures

* **(b) rise time.** The reduced DEM indentation produces two families of
  events: fast fracture-driven impacts (23–59 ms, 6 of 9 events) and slow
  crushing build-ups (141–186 ms, the three largest).  Only the first family
  sits inside the 10–50 ms band.  A full-scale continuous-crushing record is
  needed before the band can be judged; the DEM prototype at 1.5 m/s with a
  40-disk lattice does not resolve the full-scale crushing rate.
* **(c) whipping stress share 18 %** (band) / **15 %** (modal).  Just below the
  20–40 % band.  The DEM record is *spiky*: mean/peak = 0.08, whereas
  full-scale ice-crushing records sustain 30–50 % of the peak, which loads the
  quasi-static band far more strongly.  The share is also sensitive to the
  modal truncation (the three retained whipping modes are 1.8/3.6/5.9 Hz).
* **(d) whipping damage share.** The two accepted definitions disagree by a
  factor of 30: the standard *removal technique* (`D_total − D_quasi`) gives
  78.6 %, the direct band split gives 2.6 %.  The reason is that the S-N curve
  is applied with `m = 5` in this stress range (all ranges < 52.6 MPa), so the
  damage is controlled by the largest few ranges, and the two decompositions do
  not preserve the range multiset.  With only 4 rainflow cycles in a 2.4 s
  window the statistic is not converged; a long-term ice-load history is
  required before this criterion is meaningful.
* **(g) dispersion.** Rise time 26.6 % and damage 300 % exceed 20 % because the
  three largest DEM events have both different rise times (141/185/186 ms) and
  different parametric exponents; the damage share is a 5th-power quantity and
  therefore amplifies the pulse-to-pulse variability.

## 5. Limitations

* The DEM lattice is a reduced prototype; the load amplitude is rescaled
  linearly to a documented 15 MN bow load, so absolute stress and damage levels
  are indicative only — all ratio criteria are independent of that factor.
* The added mass is a 2-D strip (high-frequency) value and therefore an upper
  bound; the 3-D relief factor is exposed as a sensitivity (`artifacts/
  whipping.json → relief_sensitivity`, drop 21.1 % → 12.8 % for relief 1.0 →
  0.5), not hidden in a tuned constant.
* Zero-phase filtering of a 2.4 s record has edge transients; the mode-1-only
  negative control shows 8.8 % filter leakage in the 1–50 Hz band, which is why
  the modal split is reported alongside.
* No comparison against Kujala (1991) measurements: the primary source is not
  available in this worktree, so the literature anchor is **WITHHELD** rather
  than approximated.

## 6. Reproduction

```bash
python -m pytest tests/test_whipping_response.py -q     # 25 passed
python benchmarks/whipping_response.py                  # full DEM + chain
python benchmarks/whipping_response.py --reuse          # structural part only
```