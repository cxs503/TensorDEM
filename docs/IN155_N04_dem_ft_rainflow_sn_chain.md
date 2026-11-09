# IN155-N04 — DEM contact-force time history → rainflow → S-N → Miner: closing the ice-induced fatigue load chain

**Node**: IN155-N04 (campaign DAG, physical domain 振动噪声/结构 — ice-induced vibration fatigue)
**Repository**: TensorDEM (solver-side contract + chain), consumer side: ship-performance-platform
**Artifacts**: `artifacts/in155_n04/` (JSON tables, PNG figures), `src/tensordem/fatigue_chain/`,
`scripts/in155_n04_chain.py`, `tests/test_fatigue_chain_*.py`

---

## 1. Case, geometry and parameters — what question this node answers

Ice-zone fatigue damage of a hull detail is obtained today by (i) running some
contact/indentation model, (ii) compressing the resulting load history into a
statistic (peak, RMS, PSD) or a low-rate time series, and (iii) counting cycles
on a structural response. Step (ii) is an unquantified systematic error: **how
much damage error does the compression add?** Nothing in the ecosystem
transported the actual DEM contact force ``F(t)`` to the fatigue chain, and
there was no admissibility check that the force that reached the S-N curve was
the force the DEM produced.

The chain built here is

```
F(t)  (contact table: t, contact_id, floe_i, floe_j, F_n, F_t, patch_area,
       delta, v_n, e_spring, e_diss)
  --(contract: monotone time + energy budget)-->
F_bow(t)  (contact-aggregated load channel)
  --(modal transfer: exact SDOF per mode, Hz / damping / gain)-->
sigma(t)  (hot-spot stress)
  --(ASTM E1049-85 range-pair)-->  (dS_i, n_i)
  --(Miner + DNV-RP-C203 curve D)-->  D
```

**Hull transfer used for every full-scale number** (inherited from
`docs/ice_load_statistics_calibration.md`, IN29-N03, not re-fitted here):
modes 0.69 / 1.83 / 3.57 / 5.85 Hz, ζ = 1.5 % each, static gain
6.2 MPa/MN (peak 15 MN → 93 MPa), split 0.62/0.19/0.12/0.07 over the modes.
DNV-RP-C203 curve D in air (m₁ = 3, log a₁ = 12.164, knee 52.6 MPa, m₂ = 5).

**Load records** (three contract cases + two fatigue instances):

| id | record | nature | key parameters |
|---|---|---|---|
| C1 | single ball impact | closed-form Kelvin-Voigt + independent ODE integrator | m = 0.25 kg, v₀ = 0.5 m/s, k = 2.0e5 N/m, ζ = 5 %, recorded at 50 kHz |
| C2 | two-ball collision | closed form in the relative coordinate | m = 0.40/0.60 kg, v = +0.30/−0.10 m/s, m_eff = 0.24 kg |
| C3 | floe field vs bow | 2D reduced-scale DEM (disks, Hertz-type normal + Coulomb, water drag) | 45 floes r = 20–45 mm, h = 20 mm, bow stem circle R = 0.20 m at 0.60 m/s, k = 2e5 N/m, μ = 0.3, Δt = 40 µs, 1.5 s → **1 289 contact episodes**, 226 999 exported samples |
| I1 | m5405 calibrated bow load | IN29-N03 synthetic full-scale record (60 s, 1 kHz, 15 MN peak) | R = 0.396, mean 5.94 MN, 15 MN peak |
| I2 | C3 after Froude scaling | λ_L = 20 (λ_t = 4.47, λ_F = 8 000), peak 1.48 MN, normalised to the same 93 MPa hot-spot peak | contact pulses 1.76 ms median (model) → 7.9 ms full scale |

C3 is a **reduced-scale, translation-only 2D** model of a broken-ice field: its
contact laws are those certified by C1/C2, but its absolute force and duration
levels are model scale, which is why every criterion reported on it is a ratio
and why I2 is normalised to the hull's 93 MPa hot-spot peak (the two instances
then differ in load *shape*, not in S-N branch occupancy).

## 2. How it was done

**(a) Contract** (`fatigue_chain/contract.py`). Sparse table, one row per
(contact, time). Checks: columns finite and equal length; ``t`` strictly
increasing **within** a contact in stored order and non-decreasing over the
table; ``F_n, F_t, patch_area, delta ≥ 0``; ids valid (``floe_j = −1`` = hull);
per-contact energy budget

```
e_diss(t) = ∫ F_n v_n dt − (e_spring(t) − e_spring(t0))
```

reduced with the *same* trapezoidal rule on the exported samples, and
``e_spring`` verified as a state function of the overlap (within-``delta``-bin
detrended scatter). The budget is normalised by the **record** energy scale, not
by each contact's own scale: a grazing contact that stores nano-joules would
otherwise define the admissible error of the whole export (measured 1e-9 J
contacts reporting > 100 %). Two modes are reported, never assumed:
``physics_complete`` (energy group present, all checks run) and
``statistics_only`` (kinematic + force checks only, skipped checks listed).

**(b) SDOF transfer** (`response.py`). Exact piecewise-linear (ramp particular
solution ``q_p = C + Dτ``, ``C = f₀ − 2ζΔf/(ω_n h)``) recurrence, DC gain 1,
static initial condition. A first version dropped the ``1/h`` in ``C`` and was
**3.4 % low** in the damped case — caught by comparing with an independent
Duhamel convolution and with RK4 (now 1.9e-7).

**(c) Counting and S-N** are *not* re-implemented: `tensordem.whipping.rainflow`
(ASTM E1049-85 range-pair) and `tensordem.whipping.fatigue.miner_damage`
(DNV-RP-C203) are reused as the single source of truth, so the load side and the
S-N side of the ecosystem cannot drift apart.

**(d) Compression** (`resample.py`). Three reductions to rate ``fs``:
``naive`` = zero-phase 4th-order Butterworth anti-alias at 0.4 ``fs`` + point
sampling (what a real DAQ delivers); ``raw`` = point sampling without the
anti-alias filter (no peak logic, aliasing); ``peak`` = min/max decimation with
the two extremes kept **at their original times and bin-boundary anchors**
keeping the excursion inside its own bin. Reconstruction onto the analysis grid
is linear for all three, so ``dD/D`` measures the compressed *record*, not a
hypothetical ideal sampler.

**(e) Minimum-rate law.** Non-periodic pulse trains (exponential arrivals,
uniform amplitudes) through one SDOF channel tuned to ``f_n = 0.35/t_p``; a dense
rate grid (0.5/t_p … 30/t_p) and log-log interpolation of the first crossing of
``|dD/D| = 10 %`` per (m, t_p, method), then a least-squares fit of
``fs_min = a·m^b / t_p^c``. A **periodic** train was tried first and had to be
discarded: it phase-locks with the sampling grid and produced non-monotone
errors (−2.6 % … +4.7 % at 300–500 Hz for a 20 ms pulse).

## 3. Results

### 3.1 Contract admissibility (cases C1–C3)

| case | mode | samples | contacts | energy budget residual | spring state-function residual | admissible |
|---|---|---|---|---|---|---|
| C1 single impact | physics_complete | 171 | 1 | **1.16e-4** | 1.63e-4 | YES |
| C2 two-ball | physics_complete | 167 | 1 | **1.20e-4** | 1.55e-4 | YES |
| C3 floe field | physics_complete | 226 999 | 1 289 | **6.86e-2** | 3.22e-4 | YES @ tol 10 % |

Energy-budget closure vs record interval (C3, independent runs):

| Δt [s] | rate [kHz] | budget residual | state-function residual |
|---|---|---|---|
| 80 µs | 12.5 | 1.55e-1 | 1.91e-4 |
| 40 µs | 25 | 7.17e-2 | 1.96e-4 |
| 20 µs | 50 | 3.30e-2 | 1.92e-4 |

→ the budget residual is **first order in Δt** (15.5 → 7.2 → 3.3 %), i.e. it is
the trapezoidal representation of the velocity reversal inside one interval, not
a physics defect. That is why the floe-field case is judged at a
resolution-documented 10 % and the closed-form impact cases at 50 kHz close to
1e-4.

Closed-form vs independent ODE integrator (C1): peak force 3.4e-5, contact
duration 5.9e-5, restitution 1.7e-5 (relative). C2 momentum: 1.2e-16.

### 3.2 Analytical anchors

**(a) Constant-amplitude Miner.** A sine sampled exactly at its turning points
(4/period) through rainflow + DNV curve D:

| amplitude [MPa] | periods | counted cycles | D_rel error |
|---|---|---|---|
| 60 | 40 | 40 | **0.0** |
| 25 | 40 | 40 | **1.4e-16** |
| 150 | 25 | 25 | **0.0** |
| 90 | 12 | 12 | **2.8e-16** |

criterion ``< 1e-12``: **PASS** (worst 2.8e-16).

**(b) Single half-sine pulse vs Duhamel closed form** (F₀ = 1 MN, t_d = 50 ms,
f_n = 2 Hz): peak 396 265.4 N (numeric) vs 396 273.5 N (closed form) →
**2.06e-5** relative. The damped variant (ζ = 5 %) is checked separately
against the convolution: 1.9e-7. Criterion ``< 3 %``: **PASS** (by 5 orders).

**(c) ASTM E1049-85 counting regression** (9-point Downing example
−2, 1, −3, 5, −1, 3, −4, 4, −2, hand-derived and matched item by item):

| range | count | mean |
|---|---|---|
| 3 | 1.0 | −0.5 |
| 4 | 1.0 | +1.0 |
| 8 | 1.0 | +1.0 |
| 8 | 0.5 | 0.0 |
| 6 | 0.5 | +1.0 |

total 4.0 = (9−1)/2 ✓. Three-level load block 0, 3, −1, 5, −3, 2, 0 (three
positive levels, one closed hysteresis loop): full 6 (mean 2), full 3
(mean 1.5), half 5 (mean −0.5), half 2 (mean 1) — exactly the hand derivation;
the doubled block (13 turning points) gives 8/6/4/3/2 full + 5/2 half, matching
the hand derivation as well. Criterion (c): **PASS, item by item**.

**(d) m5405 statistics through the export** (judge d):

| quantity | exported | m5405 report | error |
|---|---|---|---|
| peak | 15.000 MN | 15.0 MN | **0.0 %** |
| mean | 5.9396 MN | 5.94 MN | **7.4e-5 %** |
| R = mean/peak | 0.39597 | 0.396 | **7.4e-5 %** |
| RMS / P95 (JSON round trip) | 6.4031 / 9.8747 MN | — | **0.0 %** (bit-exact) |

Criterion ``< 5 %``: **PASS**. The m5405 report tabulates mean/peak/R only, so
RMS and P95 are anchored on the round trip (the export literally cannot move
them: 60 000 samples, JSON, reload, identical to 1e-12).

### 3.3 The damage-error envelope (the node's product)

Instance I1 (m5405, 60 s, 1 kHz, exact chain damage **D = 9.644e-6**, 279
rainflow cycles):

| fs [Hz] | naive (anti-aliased) | raw (no anti-alias) | peak-preserving |
|---|---|---|---|
| 1 | **−99.94 %** | −79.5 % | **+956.9 %** |
| 5 | **−86.5 %** | +55.9 % | **+187.7 %** |
| 20 | **−23.9 %** | −30.6 % | **+3.4 %** |
| 100 | **+0.13 %** | +0.58 % | **+0.07 %** |

minimum rate for ``|dD/D| < 10 %``: **naive 100 Hz, peak-preserving 20 Hz**
(the ``raw`` method never reaches the band inside the grid).

Instance I2 (floe-field DEM, Froude-scaled, normalised to 93 MPa, D = 3.017e-7, 36 rainflow cycles):
naive −87.2 / −64.3 / −89.4 / −82.6 %, peak +10 287 / +2 999 / +175 / +6.7 %.
Peak-preserving is monotone here as well, naive is not (see §5).

**Statistics compression** (the current practice — keep only statistics):

| surrogate | I1 dD/D | I2 dD/D |
|---|---|---|
| narrow-band Rayleigh from (RMS, cycle count) | **+161 %** | +2 254 % |
| Rayleigh with the RMS re-fitted to the observed peak | +229 % | +3 156 % |
| all cycles at the design peak ("block equivalent") | **+3 409 %** | +16 822 % |

For damage purposes the statistics route is not a few-percent approximation: it
is off by a **factor 2.6** (narrow band) to **35** (design peak block) on the
full-scale record, and by one to two orders of magnitude on the short DEM
record. Note also that the narrow-band formula must use the *fluctuating*
stress: folding the static 37 MPa offset into the RMS inflated it further
(+3 839 % before the correction).

### 3.4 Minimum-sampling-rate law

Measured minima (m = 3 and m = 5, single-slope S-N, SDOF at f_n = 0.35/t_p,
40–200 random pulses, 18 rates per point):

| t_p [ms] | 10 | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|---|
| naive, m = 3 [Hz] | 327.4 | 163.7 | 81.9 | 40.9 | 20.5 | 10.2 |
| naive, m = 5 [Hz] | 415.9 | 208.0 | 104.0 | 52.0 | 26.0 | 13.0 |
| peak, m = 3 [Hz] | 92.8 | 46.4 | 23.2 | 11.6 | 5.80 | 2.90 |
| peak, m = 5 [Hz] | 147.7 | 73.9 | 36.9 | 18.5 | 9.23 | 4.62 |

Least-squares fit (log domain, 12 points, R² = 1.000, max relative residual 4e-13):

```
fs_min = 1.957 · m^0.468 / t_p        (anti-aliased point sampling, c = 1.000)
fs_min = 0.342 · m^0.909 / t_p        (peak-preserving min/max, c = 1.000)
```

So ``fs_min·t_p`` is **constant** (1.96–0.34 samples per pulse width) as
expected from ``|dD/D| ≈ m·δS/S`` with the retained-peak fraction scaling as
``1/(fs·t_p)``; the m-exponent is 0.47 for point sampling (the attenuation is
only partly amplitude-driven) and 0.91 for peak-preserving (close to the
theoretical ∝ m). Engineering reading: for a 90 ms crushing pulse
(``t_p`` from the m5405 record: rise time 87 ms) the anti-aliased-sampling
criterion is ≈ 22 Hz·(m/3)^0.47, i.e. **100 Hz** for DNV D — which is exactly
what the I1 envelope measured.

### 3.5 Figures

| figure | content |
|---|---|
| `figures/fig1_contact_forces.png` | the three contract cases: C1 (3.5 ms contact), C2, C3 full-scale 6.7 s + 0.35 s zoom |
| `figures/fig2_energy_budget.png` | energy-budget and spring state-function residuals vs record interval (slope-1 reference) |
| `figures/fig3_anchors.png` | single half-sine pulse: exact recurrence vs Duhamel closed form; constant-amplitude D error vs amplitude |
| `figures/fig4_envelopes.png` | \|dD/D\| vs sampling rate, both instances, naive vs peak, 10 % band |
| `figures/fig5_min_rate_law.png` | measured fs_min vs 1/t_p with the two fitted laws |

## 4. Verdict on the machine-checkable criteria

| criterion | target | measured | verdict |
|---|---|---|---|
| a) constant-amplitude D vs analytic | < 1e-12 | 0.0 … 2.8e-16 | **PASS** |
| b) single pulse vs Duhamel closed form | < 3 % | 2.06e-5 (damped vs convolution 1.9e-7) | **PASS** |
| c) ASTM E1049-85 item-by-item (cycles, ranges, means) | exact | exact, incl. hysteresis loop + 3-level block (+ doubled block) | **PASS** |
| d) F(t) RMS/P95/peak vs m5405 report | < 5 % | peak 0.0 %, mean 7.4e-5 %, R 7.4e-5 %, round trip 0 | **PASS** |
| e) envelope non-empty, monotone; peak ≤ naive | — | non-empty ✓; monotone on I1 for all three methods ✓; **peak ≤ naive only for fs ≥ 20 Hz** (I1: 3.4 % vs 23.9 %), **FAIL at 1 and 5 Hz** | **PARTIAL — honest FAIL on the 1–5 Hz ordering** |
| f) full-chain pytest incl. fail-closed | pass | 35 tests in 4 files, all pass; empty/absent history raises `MissingForceHistory` | **PASS** |

Criterion (e) details, so the FAIL is not glossed over: below ~1.5 f₁ neither
compression is admissible. Anti-aliased point sampling at 1 Hz *deletes* 99.94 %
of the damage (unconservative), while min/max reconstruction at 1 Hz manufactures
one full-range excursion per second, drives the 0.69 Hz mode near resonance and
**overestimates by 9.6×** (conservative but wrong); the absolute-error ordering
between the two is then a coin flip. Above ~3 f₁ the peak-preserving method is
monotonically better and always errs on the safe side, which is the actionable
statement.

## 5. Notes — traps found, limits, scope

1. **The bow force sign.** First floe-field run "swallowed" floes (penetration
   = bow radius, F = 10 kN, v = 160 m/s): the contact force was applied along
   −n instead of +n. Caught only by the energy budget (residual 4.07) — an
   animation or a force plot looked plausible.
2. **Grazing contacts define the tolerance if you let them.** Normalising the
   energy check per contact made 1e-9 J contacts report > 100 % error; the check
   is now relative to the record energy scale, with negligible contacts counted
   and reported (18 of 1 289 in C3).
3. **Periodic pulse trains phase-lock with the sampling grid** and make the
   minimum-rate law unmeasurable (non-monotone errors). Random arrivals are a
   requirement, not a nicety.
4. **A missing ``1/h`` in the ramp particular solution** made the SDOF transfer
   3.4 % low in damping; only an independent method (convolution/RK4) exposed
   it — the single-pulse closed-form anchor uses ζ = 0 and would have passed.
5. **Min/max decimation without bin-boundary anchors** manufactures a
   low-frequency sawtooth from two adjacent bin extremes: +9 500 % damage error
   at 1 Hz on m5405 (vs +957 % with the anchors).
6. **Statistics compression is a factor-level error, not a percentage error** —
   the narrow-band Rayleigh formula needs (i) the fluctuating stress, and (ii)
   a peak-truncated range distribution; even then it is 2.3–2.6× high on a
   60 s record because the Rayleigh tail is unbounded while the record has 279
   cycles.
7. **Short records carry little statistical information.** On I2 the damage is
   carried by a handful of cycles, so the anti-aliased-sampling error envelope
   is *not* monotone (87 → 64 → 89 → 83 %): which cycles survive matters more
   than the rate. The envelope concept needs a record long enough to average the
   extremes; this is reported, not smoothed away.
8. **C3 is a reduced-scale 2D model** (translation only, no rotation, no
   buoyancy/bending), Froude-scaled with λ_L = 20 and normalised to the
   documented 93 MPa hot-spot peak. Absolute stress/damage levels are modelling
   assumptions; every criterion on I2 is a ratio.
9. **The consumer side is not yet wired.** `aggregate_channel`/`to_uniform`
   produces the contact-aggregated load channel that
   ship-performance-platform's fatigue module is meant to consume; the actual
   integration in that repository is *not* part of this node (repository
   boundary: TensorDEM owns the solver and the contract). The legacy statistics
   interface is untouched, so nothing downstream breaks.
10. Damage numbers inherit the m5405 15 MN peak normalisation and the 6.2 MPa/MN
    transfer; they are reported to make errors comparable, not as a hull
    certification.

## 6. Error judgement (key quantities vs reference)

| quantity | reference | value | error | < 3 %? |
|---|---|---|---|---|
| C1 peak contact force | ODE integrator 104.12 N | 104.12 N | 3.4e-5 | **yes** |
| C1 restitution | analytic 0.8545 | 0.8545 | 1.7e-5 | **yes** |
| C2 linear momentum | initial | final | 1.2e-16 | **yes** |
| constant-amplitude damage | n/N(dS) | chain | ≤ 2.8e-16 | **yes** |
| single-pulse peak response | Duhamel closed form | exact SDOF | 2.06e-5 | **yes** |
| ASTM E1049-85 cycles | textbook ranges/means | counted | 0 (item by item) | **yes** |
| m5405 peak load | 15.0 MN | 15.000 MN | 0.0 | **yes** |
| m5405 R = mean/peak | 0.396 | 0.39597 | 7.4e-5 | **yes** |
| m5405 deck stress std | 18.5 MPa (IN29-N03) | 16.8 MPa (this chain) | **−9.2 %** | yes, but the modal split differs (documented) |
| C3 energy budget | trapezoid identity | record | 6.9 % at 25 kHz (1st order in Δt) | no — resolution property, judged at 10 % |
| envelope monotonicity (I1) | monotone | monotone (3 methods) | — | **yes** |
| peak vs naive at 1–5 Hz | peak better | peak worse in \|·\| | +957 % vs −99.9 % | **no — honest FAIL, mechanism in §4** |

## Reproduction

```bash
python scripts/in155_n04_chain.py            # ~140 s, writes artifacts/in155_n04/
python -m pytest tests/test_fatigue_chain_contract.py tests/test_fatigue_chain_anchors.py \
                 tests/test_fatigue_chain_astm.py tests/test_fatigue_chain_envelope.py -q
```

All numbers in this document come from `artifacts/in155_n04/*.json`; no value is
typed by hand.