# IN29-N03 - statistical calibration of DEM ice-load time histories

*Paper-level report of campaign node IN29-N03 (mission 5405), repository
TensorDEM.  Generated from ``scripts/in29_n03_report.py``; every number in it is
read from ``artifacts/in29_n03/*.json``.*

# IN29-N03: statistical calibration of the DEM ice-load time history

**Module** `tensordem.whipping.load_stats`, `tensordem.whipping.load_calibration` ·
**Campaign script** `scripts/in29_n03_calibration.py` ·
**Artifacts** `artifacts/in29_n03/` · **Tests** `tests/test_ice_load_stats_calibration.py`
(31 items) · **Regression lock** `tests/test_whipping_response.py`
(25/25 passed)

## 1. Case, geometry and parameters

The question of this node is not "how large is the peak ice load" but "does the
load *history* look like a measured ice-pressure record". A record made of a few
isolated spikes and a record of continuous crushing can share the same peak and
still produce completely different whipping and fatigue content; the campaign
therefore fixes the statistical targets

| symbol | definition | target band |
|---|---|---|
| `R` | mean/peak of the record | 0.30-0.50 |
| `t_r` | 10 %-90 % rise time of an event | 0.05-0.25 s |
| slope | least-squares PSD slope, 0.5-10 Hz | -1.5 … -0.5 decade/decade |
| whipping share | rms of the 1-50 Hz deck stress / total | 0.15-0.45 (centre 0.30) |

The hull is the calibrated 150 m ice-going girder of the whipping chain
(`ShipCase`: L = 150 m, B = 22 m, T = 10.5 m, Cb = 0.86, EI = 6.18e12 N m^2,
deck y = 7 m, zeta = 1.5 %); the wet modes are 0.69 / 1.83 / 3.57 / 5.85 Hz and
the load is applied at 0.06 L with the deck stress taken amidships. The
calibrated four controls are

| control | meaning | calibrated value |
|---|---|---|
| `crushing_scale` | fragment size / contact width | 1.681 |
| `contact_damping` | DEM contact damping ratio | 0.173 |
| `floe_geometry` | floe size & thickness distribution scale | 1.092 |
| `froude` | ship speed / sqrt(gL) | 0.0179 (= 0.69 m/s) |

The record is 60 s at 1 kHz (60 000 samples, peak normalised to the documented
15 MN bow load), i.e. 1953 rainflow cycles and
60 s of physical time.

## 2. Method

**(a) Statistics** (`load_stats.py`). `R`, the rise time, the FWHM hold time,
the 10 % decay time, onset-to-onset intervals, the Welch PSD slope, the rainflow
count and the quasi-static (`< 1 Hz`) energy fraction are computed from one
uniform record. Events are attached to local maxima found by a prominence test
(5 % of the dynamic range) and their baseline is the *left base* returned by the
same test - the load level the pulse started from. That definition survives
dense overlapping crushing, where level crossing merges everything into one
event. Rise/hold times are measured on a 10 Hz-conditioned copy of the record
(the crushing micro-noise floor would otherwise supply a spurious trough a few
samples from every peak); `R`, the spectrum and the rainflow count use the raw
record.

**(b) DEM anchor** (`load_calibration.dem_anchor_table`). Real indentation DEM
runs of this repository at two contact dampings give the *prototype* pulse
statistics. The DEM sheet is a reduced-scale prototype, so its times are Froude
scaled to the hull, `lambda_t = sqrt(L_ship / L_model) = 15.8`
(`L_model = 0.60 m`). The anchor is reported as
a **negative** result: the prototype's crushing build-ups scale to ~1 s at full
scale and show no measurable damping dependence (two-point exponent
-0.06), so the reduced DEM does
not resolve the full-scale crushing timescale. The rise-time reference is
therefore a calibrated constant (`TR_REF`), not a DEM measurement.

| damping [N s/m] | R of DEM record | t_r model [ms] | t_r full scale [ms] | events | broken bonds |
|---|---|---|---|---|---|
| 5 | 0.202 | 74.8 | 1183.1 | 4 | 208 |
| 15 | 0.213 | 69.9 | 1104.8 | 4 | 208 |

The raw DEM record lands at R = 0.202 and its full-scale
rise times at 1.18 s: both outside
the campaign bands. That is the gap this node closes, and it is why the load
timescale is calibrated instead of inherited from the prototype.

**(c) Four-factor Latin hypercube sweep** (`load_calibration.sweep`). 20 stratified
points over the control box (crushing scale 0.4-2.5, contact damping 0.03-0.6,
floe geometry 0.5-2.0, Froude 0.015-0.09); every point is synthesised (60 s,
1 kHz) and described with the statistics above; the whipping share is estimated
from the load spectrum and the modal transfer functions (validated in §3).

**(d) Surrogate calibration.** A quadratic least-squares surrogate maps the four
controls to the four normalised statistic residuals (surrogate R^2 per statistic:
['0.95', '0.77', '0.95', '0.88']); its minimum is re-simulated directly
and the neighbourhood is refined for three rounds, so no surrogate number enters
the reported statistics.

**(e) Closed-loop verification** (`load_calibration.closed_loop`). The calibrated
record is fed through the exact piecewise-Duhamel modal chain (steady-state
initial condition: the static deflection of the mean bow load plus the response
to the zero-mean fluctuation), giving the 1-50 Hz / `< 1 Hz` split, the exact
modal split (modes >= 2 versus mode 1), three damage-share estimators on
DNV-RP-C203 curve D, and the per-segment dispersion of R and of the damage share.

**(f) Identifiability.** The Jacobian of the four normalised statistics with
respect to the four controls (10 % perturbations) gives the Fisher information
with the seed-to-seed noise covariance; its eigenvalues, the marginal
per-parameter information and every two-parameter sub-block answer "which
parameters can the statistics separate".

## 3. Results

### 3.1 Calibrated load statistics (60 s, 1 kHz)

| quantity | value | target | verdict |
|---|---|---|---|
| R = mean/peak | 0.396 | 0.30-0.50 | PASS |
| rise time t_r (median, 10-90 %) | 0.087 s | 0.05-0.25 | PASS |
| hold time t_h (median FWHM) | 0.090 s | — | — |
| PSD slope 0.5-10 Hz | -0.86 dec/dec | -1.5 … -0.5 | PASS |
| rainflow cycles | 1953 | >= 1000 | PASS |
| quasi-static energy fraction (<1 Hz) | 0.117 | — | — |
| inter-arrival mean / CV | 0.049 s / 1.66 | — | — |
| events, duty cycle | 1222 / 1.72 | — | — |
| mean / peak load | 5.94 / 15.0 MN | — | — |

The pulse statistics of the full record: t_r median 0.087 s
(p10 0.032, p90 0.229 s), i.e.
0.60 of the detected events inside the
50-250 ms band.

### 3.2 Closed loop: stress and damage

| quantity | value | target | verdict |
|---|---|---|---|
| whipping stress share (1-50 Hz / total) | 0.295 | 0.15-0.45 | PASS |
| whipping stress share (exact modal split) | 0.247 | cross-check | — |
| deck stress std / peak | 18.5 / 93 MPa | plausible | — |
| whipping damage share, Butterworth band split | 0.036 | — | — |
| whipping damage share, ideal FFT split | 0.032 | — | — |
| whipping damage share, modal split | 0.014 | — | — |
| whipping damage share, removal technique | 0.486 | diagnostic | — |
| gap of the two filter-based definitions | 0.126 | < 0.15 | PASS |
| segment R dispersion (4 x 15 s) | 0.092 | < 0.10 | FAIL |
| segment damage-share dispersion | 2.63 | < 0.10 | FAIL |
| gap, filter pair vs exact modal split (diagnostic) | 1.36 | — | — |

Per-segment detail:

| segment (15 s) | R | whipping damage share (ideal split) | PSD slope |
|---|---|---|---|
| 1 | 0.412 | 0.312 | -1.03 |
| 2 | 0.448 | 0.004 | -0.82 |
| 3 | 0.410 | 0.097 | -0.90 |
| 4 | 0.411 | 0.057 | -1.09 |

### 3.3 Spectral response estimate against the exact chain

The sweep uses a spectral estimate of the whipping share; the exact chain is the
reference. On 4 validation records the spectral estimate is
0.98x the exact value (max error
8 %), which is the bias of a Welch-based
linear estimate when the quasi-static part of the load cannot be resolved. The
reported whipping share is always the exact-chain number.

| case | share (exact) | share (spectral) | bias | std exact [MPa] | std spectral [MPa] |
|---|---|---|---|---|---|
| 0 | 0.256 | 0.248 | 0.971 | 23.48 | 23.27 |
| 1 | 0.269 | 0.282 | 1.049 | 17.96 | 18.02 |
| 2 | 0.386 | 0.354 | 0.917 | 14.15 | 14.57 |
| 3 | 0.251 | 0.249 | 0.991 | 20.74 | 19.66 |

### 3.4 Which parameters control the waveform

| control | dR/dlnp | d t_r/dlnp | d slope/dlnp | d share/dlnp | Fisher (single) |
|---|---|---|---|---|---|
| crushing_scale | -0.81 | -0.30 | -0.28 | -1.0 | 19650596061.2 |
| contact_damping | -0.09 | -0.31 | -0.04 | -0.2 | 33698703251.5 |
| floe_geometry | -1.43 | -1.53 | 0.12 | -0.3 | 785867035060.2 |
| froude | -1.09 | -0.44 | -0.23 | -0.3 | 4725352694.0 |

Fisher eigenvalues of the joint 4-parameter problem:
['0.187', '4.189', '311.178', '843941686751.399'] (condition number
4504480390398.5). The four controls load on the statistics
along a nearly common direction - each of them moves R down, the rise time up,
the slope down and the whipping share down - so the *joint* calibration is
ill-conditioned even though every control carries information on its own. This is
reported as a limitation, not hidden: the calibrated point is a *combination*,
not four independently measured parameters.

### 3.5 Convergence of the damage-share statistics

| T [s] | whipping share | damage share (band) | damage share (ideal) | damage share (modal) | segment dispersion | R dispersion | stress cycles |
|---|---|---|---|---|---|---|---|
| 60 | 0.295 | 0.036 | 0.032 | 0.014 | 2.63 | 0.092 | 170 |
| 120 | 0.207 | 0.009 | 0.007 | 0.003 | 1.22 | 0.046 | 278 |
| 240 | 0.260 | 0.019 | 0.017 | 0.006 | 0.53 | 0.025 | 622 |

### 3.6 LHS sweep (first 20 points)

| # | crushing | damping | floe | Fr | R | t_r [s] | slope | share |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.85 | 0.376 | 0.99 | 0.0169 | 0.426 | 0.087 | -0.86 | 0.252 |
| 1 | 2.44 | 0.030 | 1.67 | 0.0710 | 0.253 | 0.074 | -0.69 | 0.502 |
| 2 | 1.03 | 0.077 | 0.80 | 0.0640 | 0.269 | 0.073 | -0.80 | 0.422 |
| 3 | 0.62 | 0.102 | 1.05 | 0.0857 | 0.263 | 0.072 | -0.72 | 0.500 |
| 4 | 1.84 | 0.092 | 1.86 | 0.0378 | 0.331 | 0.083 | -0.72 | 0.545 |
| 5 | 2.15 | 0.039 | 1.88 | 0.0238 | 0.389 | 0.081 | -0.72 | 0.588 |
| 6 | 0.78 | 0.066 | 0.69 | 0.0541 | 0.280 | 0.082 | -0.78 | 0.281 |
| 7 | 1.60 | 0.055 | 1.51 | 0.0786 | 0.278 | 0.073 | -0.65 | 0.648 |
| 8 | 1.55 | 0.260 | 1.41 | 0.0884 | 0.277 | 0.078 | -0.69 | 0.423 |
| 9 | 1.34 | 0.292 | 0.62 | 0.0310 | 0.320 | 0.089 | -0.93 | 0.257 |
| 10 | 0.47 | 0.136 | 0.61 | 0.0422 | 0.347 | 0.075 | -0.81 | 0.271 |
| 11 | 1.73 | 0.044 | 0.93 | 0.0296 | 0.359 | 0.075 | -0.93 | 0.300 |
| 12 | 1.21 | 0.121 | 1.23 | 0.0196 | 0.405 | 0.086 | -0.84 | 0.424 |
| 13 | 0.97 | 0.052 | 0.54 | 0.0605 | 0.292 | 0.059 | -0.76 | 0.336 |
| 14 | 2.30 | 0.394 | 1.32 | 0.0348 | 0.333 | 0.086 | -0.81 | 0.299 |
| 15 | 2.26 | 0.242 | 0.53 | 0.0472 | 0.286 | 0.090 | -0.91 | 0.189 |
| 16 | 1.93 | 0.165 | 1.13 | 0.0583 | 0.299 | 0.079 | -0.77 | 0.251 |
| 17 | 1.38 | 0.184 | 0.84 | 0.0743 | 0.264 | 0.080 | -0.75 | 0.347 |
| 18 | 0.56 | 0.568 | 1.54 | 0.0509 | 0.328 | 0.081 | -0.77 | 0.409 |
| 19 | 2.07 | 0.511 | 0.75 | 0.0798 | 0.277 | 0.075 | -0.76 | 0.300 |

## 4. Acceptance criteria

| criterion | value | verdict |
|---|---|---|
| (a) 4-segment R and damage-share dispersion < 10 % | R 0.092, damage 2.63 | FAIL |
| (b) whipping stress share in 0.15-0.45 | 0.295 | PASS |
| (c) two damage-share definitions within 15 % | filter pair gap 0.126; modal cross-check 1.36 | PASS |
| (d) R in 0.30-0.50 | 0.396 | PASS |
| (e) slope within 0.5 dec/dec of -1 | -0.86 | PASS |
| (f) >= 2 identifiable parameters | 4 with Fisher > 1 | PASS |
| (g) m5081 25-test regression lock | 25 passed / 0 failed | PASS |
| (h) >= 18 tests including negatives | 31 | PASS |
| (e2) rainflow cycles >= 1000 | 1953 | PASS |
| (e3) physical time >= 60 s | 60.0 s | PASS |

Overall: **HONEST FAIL on the criteria marked FAIL**.

## 5. Notes, limitations and traps

* **Filters are not neutral for a range^5 statistic.** The whipping *damage*
  share is 0.036 with a 4th-order Butterworth
  split and 0.032 with an ideal
  (brick-wall) split of the same record - a
  13 %
  difference from the filter alone. Any damage-share number is therefore only
  meaningful together with its filter, and the acceptance pair is the two
  filter-based estimates; the modal split
  (0.014) is a cross-check that fails
  additivity: D(mode 1) + D(modes >= 2) is only
  0.69
  of D(total), because rainflow pairing is not linear.
* **The removal technique is biased** for a quasi-static-dominated stress:
  (D_total - D_quasi)/D_total = 0.486 while
  the band split gives 0.036. The m5081 node
  reported the same failure with a factor 30; on this long, statistically
  calibrated record it is a factor 13.3,
  which quantifies the improvement but does not make the estimator usable.
* **Segment dispersion of a heavy-tailed statistic (honest FAIL).** The damage
  share is dominated by a handful of the largest ranges, so four 15 s windows do
  not agree to 10 % on a 60 s record: the dispersion is
  2.63 against a required 0.10, while
  the R dispersion already passes (0.092).
  The convergence study (§3.5) shows the trend: 2.63
  -> 1.22 ->
  0.53 for T = 60 / 120 / 240 s,
  i.e. the criterion needs a record of order 10^3 s, not the 60 s that the
  campaign specifies. This is a property of the range^5 statistic, not of the
  calibration, and it is left as an honest FAIL.
* The load synthesiser is not a DEM run: it is a DEM-*anchored* statistical model
  (Froude-scaled pulse times, impulse-defined events, 1/f crushing fluctuation,
  unilateral contact force). Absolute stress and damage levels inherit the 15 MN
  peak normalisation; all reported criteria are ratios or dimensionless.
* The DEM anchor uses a 13x5 lattice, 1.2 s and two dampings because the explicit
  DEM step is bounded by the total dashpot damping (dt = 0.2 m / c_total), which
  makes high contact dampings computationally prohibitive; its times are Froude
  scaled and its damping exponent is a two-point slope
  (-0.06), so it is evidence
  about the prototype, not a calibration input.
* No measured full-scale ice-pressure record is present in this worktree, so the
  target bands are the campaign's engineering bands, not a literature fit:
  the literature anchor is **WITHHELD**, not approximated.

## 6. Reproduction

```bash
python -m pytest tests/test_ice_load_stats_calibration.py -q   # 31 passed
python -m pytest tests/test_whipping_response.py -q            # 25 passed (regression lock)
python scripts/in29_n03_calibration.py                         # full campaign (~10 min, DEM anchor included)
python scripts/in29_n03_calibration.py --quick                 # reuse the cached DEM anchor
```
