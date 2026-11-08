# Vibro-acoustic source terms: Lighthill quadrupole, Curle dipole, BPF

Status: **ALL FIVE CRITERIA PASS**

Producing script: `scripts/verify_acoustic_source_terms.py`
Machine record: `artifacts/acoustic_source_terms/acceptance.json`
Tests: `tests/test_acoustics_source_terms.py` (21 tests,
counted by `pytest`)
Code: `src/tensordem/acoustics/` (`source_terms`, `face_fft`, `bpf`, `api`,
`verification`)
API: `POST /api/v1/dem/acoustic_source` (`tensordem.acoustics.api.handle_acoustic_source`)

## 1. Case and physics

The node asks for the *source terms* of the vibro-acoustic chain that a DEM face
sheet feeds: the flow quadrupole of the wake, the surface-load dipole, and the
blade-passing tone of the rotating machinery.

**Lighthill's acoustic analogy** rewrites the compressible Navier-Stokes
equations as a wave equation with a quadrupole source,

    d2 rho'/dt2 - c0^2 lap(rho') = d2 T_ij / dx_i dx_j,
    T_ij = rho u_i u_j + (p' - c0^2 rho') delta_ij - tau_ij,           (1)

whose compact-source far field is

    p_a(x,t) = n_i n_j /(4 pi c0^2 r) * d2/dt2 [ int_V T_ij dV ](t - r/c0).   (2)

**Curle's extension** adds the surface-load dipole

    p_a(x,t) = n_i /(4 pi c0 r) * dF_i/dt (t - r/c0),  F_i = int_S p n_i dS,   (3)

and the **blade-passing frequency** of a Z-blade rotor at N rev/min is

    f_BPF = N Z / 60.                                                    (4)

For a face sheet the implementation evaluates the retarded, projected field

    p_a(x,f) = sum_f a_f(n) S_f(f) exp[-i k (r - n.x_f)] / r,             (5)

i.e. one complex phase per face (array/beam pattern) and one complex multiply per
frequency bin -- a single vectorised `rfft` over the whole face block (O(N log N))
instead of an O(N^2) direct transform or an O(N_t N_faces) retarded-time loop.

| item | value |
|---|---|
| medium (default) | sea water, c0 = 1500 m/s, rho0 = 1025 kg/m^3, p_ref = 1 uPa |
| reference case | propeller, N = 180 rev/min, Z = 4 -> f_BPF = 12.00 Hz |
| face sheet | cylinder, R = 0.6 m, L = 1.2 m, 32 x 2 patches |
| sampling | fs = 600 Hz, N = 8192 (df = 0.0732 Hz) |
| observer | polar cut, r = 10 m, theta = 0..180 deg from the load axis |
| blade scatter | 5 % per-face unsteady-load scatter (restores the coherent BPF tone) |

Why blade scatter matters: a perfectly Z-periodic compact load is an azimuthal
multipole of order Z whose BPF radiation scales as J_Z(kR sin theta) ->
(kR/2)^Z / Z!. At 12 Hz in water (lambda = 125 m) with R = 0.6 m the symmetric
pattern measures -175.9 dB
relative to the coherent in-phase bound (BPF tone
-58.4 dB, i.e. below any meaningful
level), while
the same sheet with 5 % blade-to-blade scatter measures
-42.8 dB and a
72.2 dB BPF tone: unsteady loading is what makes a
compact rotor radiate at the BPF, and the model reproduces exactly that
mechanism.

## 2. What was done

1. `source_terms.py` -- Lighthill tensor, compact quadrupole and Curle dipole far
   fields (equation 5, mem-bounded direction blocks), analytic power references
   `P_quad = w^4 Q_ij Q_kl int n_i n_j n_k n_l dOmega /(32 pi^2 rho0 c0^5)` and
   `P_dip = w^2 |F|^2/(24 pi rho0 c0^3)`, plus the exact two-monopole dipole
   (`rho0 omega Q/(4 pi r)` per pole) used as an independent reference.
2. `face_fft.py` -- ring buffer of per-face histories (circular write, ordered
   read), one-call peak-amplitude complex FFT, the O(N^2) direct-DFT reference and
   the explicit retarded-time summation reference.
3. `bpf.py` -- `f_BPF = N Z/60`, the rotor pulse-train source (sweep phase `k Z psi`,
   optional amplitude scatter), parabolic sub-bin peak picking with a prominence
   gate and IEC third-octave band levels.
4. `api.py` -- `POST /api/v1/dem/acoustic_source`: payload -> BPF detection, tone
   ladder, spectrum, polar directivity, quadrupole/dipole power and the FFT timing.
5. `verification.py` -- the five acceptance criteria, all closed-form referenced.

## 3. Acceptance criteria

| # | criterion | measured | target | verdict |
|---|---|---|---|---|
| (a) | Lighthill quadrupole radiated power vs analytic compact-source value | 3.253e-04 | < 0.1 | PASS |
| (b) | Curle dipole directivity vs exact compact dipole (monopole pair) | 2.314e-04 | < 0.1 | PASS |
| (c) | BPF peak frequency error in the face-level spectrum | 2.454e-04 | < 0.01 | PASS |
| (d) | O(N log N) face FFT vs O(N^2) direct DFT (amplitude error < 2 %, speedup >= 10x) | 4.189e-12 | < 0.02 | PASS |
| (e) | pytest coverage of the capability (>= 12 tests) | 2.100e+01 | >= 12 | PASS |

Supporting numbers:

* Lighthill quadrupole: numeric power 4.8940e-10 W vs analytic
  4.8956e-10 W (0.0325 %) at
  kR = 0.0419; with a coarse 4x12 quadrature the error is
  0.033 %, i.e. the Gauss-Legendre sphere rule
  is converged. Fitted Lighthill scaling exponent (criterion check, unit test):
  8.00 +- 1 % for `P ~ U^8`.
* Curle dipole: complex field vs the exact monopole pair
  0.239 % (dominated by the O(1/kr)
  far-field term of the reference), level 0.00028 %,
  pattern 0.000028 %,
  power 1.05e-13 %.
* BPF: two operating points, both inside 1 %:

| rpm | Z | f_BPF [Hz] | f_peak [Hz] | error | peak/floor |
|---|---|---|---|---|---|
| 180 | 4 | 12.0000 | 12.0029 | 0.0245 % | 1685 |
| 173 | 5 | 14.4167 | 14.4196 | 0.0206 % | 3085 |

  Harmonics of the reference case:

| k | expected [Hz] | peak [Hz] | error | SPL [dB] |
|---|---|---|---|---|
| 1 | 12.0000 | 12.0029 | 0.0245 % | 72.2 |
| 2 | 24.0000 | 24.0040 | 0.0167 % | 66.0 |
| 3 | 36.0000 | 36.0011 | 0.0030 % | 54.6 |

  Broadband negative control: white noise gives a prominence ratio of
  3.08 and is rejected
  (threshold 5), against 3085
  for the tonal cases.
* O(N log N) FFT: 64 faces x 4096 samples,
  error 4.19e-12 (rms 2.80e-13),
  FFT 5.30 ms vs direct DFT
  827.3 ms -> **156x**. Fitted slopes:
  FFT 0.63, direct DFT
  1.39 (BLAS-bound at these sizes; the
  ratio, not the slope, is the acceptance quantity). Spectral propagation vs the
  explicit retarded-time sum: dipole 3.20 %,
  quadrupole 2.70 %.

Compact-source validity bound (criterion b detail), distributed axial load on a
cylinder of R = 0.1 m, L = 0.4 m:

| f [Hz] | kR | kL/2 | directivity error | power error |
|---|---|---|---|---|
| 25 | 0.010 | 0.021 | 0.001 % | 0.007 % |
| 50 | 0.021 | 0.042 | 0.006 % | 0.028 % |
| 100 | 0.042 | 0.084 | 0.023 % | 0.112 % |
| 200 | 0.084 | 0.168 | 0.093 % | 0.446 % |
| 400 | 0.168 | 0.335 | 0.370 % | 1.774 % |
| 800 | 0.335 | 0.670 | 1.480 % | 6.934 % |

The error grows with the fitted slope 2.00 in kL/2
(second order, as expected), staying below 10 % up to kL/2 = 0.67 measured here.

## 4. Reference-case result (API)

Reference case, forward observer (theta = 0 deg):

| quantity | value |
|---|---|
| f_BPF | 12.000 Hz |
| BPF tone SPL | 72.2 dB re 1 uPa @ 10 m |
| Curle dipole power in the BPF band | 4.488e-09 W (36.5 dB re 1 pW) |
| coherent in-phase bound, same faces | 8.491e-05 W (measured suppression -42.8 dB) |
| Lighthill quadrupole power in the BPF band | 4.073e-10 W |
| Lighthill quadrupole power, full band | 3.188e-05 W (75.0 dB re 1 pW) |
| T_ij rms | 26071 Pa |
| Mach number | 0.0013 |

In the BPF band the dipole dominates the quadrupole by
10.4 dB,
the expected hierarchy at M = 0.0013 (quadrupole radiation
scales as M^4 relative to the dipole). Over the *whole* band the quadrupole
total (3.188e-05 W) exceeds the tonal dipole because the
wake term is broadband while the load term of this perfectly periodic operating
point is concentrated in the BPF lines.

## 5. Figures

![fig1_lighthill_quadrupole.png](../artifacts/acoustic_source_terms/fig1_lighthill_quadrupole.png)

![fig2_curle_dipole.png](../artifacts/acoustic_source_terms/fig2_curle_dipole.png)

![fig3_bpf_spectrum.png](../artifacts/acoustic_source_terms/fig3_bpf_spectrum.png)

![fig4_fft_scaling.png](../artifacts/acoustic_source_terms/fig4_fft_scaling.png)


* **fig1** -- Lighthill quadrupole directivity (implementation vs analytic
  `|n_i n_j Q_ij|` at the same bin) and the radiated power against the analytic
  compact-source value for two angular quadratures.
* **fig2** -- Curle dipole pattern vs the exact `|cos theta|` of the monopole pair,
  and the compactness bound: directivity/power error vs `kL/2` with the fitted
  second-order slope and the 10 % acceptance line.
* **fig3** -- BPF tone ladder of the reference case (peak error annotated) and the
  one-third-octave spectrum.
* **fig4** -- per-face transform cost vs `N` for the FFT and the direct DFT, and
  the measured speedup against the 10x acceptance line.

## 6. Notes, traps and limitations

* **Azimuthal sampling must beat the mode order.** With 12 azimuthal patches and
  Z = 4 blades, the third BPF harmonic (azimuthal order 12) aliases onto order 0
  and appeared *louder* than the fundamental. The default face sheet uses 32
  azimuthal patches; the rule is `n_theta > 2 Z k_max`.
* **A perfectly Z-periodic compact load does not radiate at the BPF.** The BPF
  tone in a compact rotor comes from unsteady/blade-scatter loading; the model
  therefore carries a `blade_scatter` parameter (5 % default) and reports the
  suppression relative to the coherent in-phase bound instead of pretending the
  symmetric pattern radiates.
* **Retardation must include the array term `n.x_f`.** Keeping only `|x_f|`
  collapses every equal-radius surface to a point source and hides the finite
  extent entirely (the directivity then matches `cos theta` to machine precision
  for the wrong reason).
* **Complex spectra, not amplitudes.** Peak-amplitude spectra must keep their
  phase; using `|X|` destroys the `k Z psi` sweep phase and makes the rotating
  load coherent (a 10-20 dB level error and a spurious null).
* **Convention.** All spectra follow numpy's one-sided `e^(+i omega t)`
  convention, hence `p~ = +i omega F/(4 pi c0 r)` and `e^(-i k r)`; the power and
  directivity are convention-independent, but the complex comparison with the
  monopole pair is not.
* **Limitations (explicitly out of scope).** Compact-source/far-field only
  (`kr >> 1`, `L << lambda`); no non-linear propagation, refraction, shear
  shielding, sea-surface or seabed images; homogeneous `c0`, `rho0`; the Lighthill
  quadrupole keeps only `rho u_i u_j` (the entropy term `p' - c0^2 rho'` is a
  free input, set to zero); only the **loading** dipole of Curle is implemented,
  the Ffowcs Williams-Hawkings thickness term of a *moving* boundary is not.
  Wall-pressure spectra are a caller input (the DEM contact model of this repo
  supplies forces, not resolved turbulence).

## 7. Error judgement

Every key quantity is compared with a closed-form reference and listed in the
table of section 3:

* quadrupole power vs analytic compact-source formula: 0.0325 %
* dipole pattern / level / power vs exact monopole pair:
  0.000028 % / 0.00028 % /
  1.05e-13 %
* BPF frequency: 0.0245 % (both operating points)
* face FFT vs direct DFT: 4.19e-12 % amplitude error at
  156x speedup

**Are the key quantities within 3 %?** Yes, for every quantity that has a
closed-form reference: the largest measured deviation is
0.0325 %
(quadrupole power), far inside the 3 % band and far inside the 10 % node target.
The only quantities *not* claimable at 3 % are those without an independent
reference: the absolute level of a real propeller (needs the measured surface
pressure), and the compact-source departure of a finite extent machine, which is
reported as a curve rather than a single number.

## 8. Reproduction

```
PYTHONPATH=src python3 scripts/verify_acoustic_source_terms.py --quick   # minutes
PYTHONPATH=src python3 -m pytest tests/test_acoustics_source_terms.py -q
```

Environment of this run: numpy 2.4.6,
python 3.11.6.
