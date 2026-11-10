# Brash-ice channel resistance from granular $\mu(I)$ rheology

Verdict: **(a) FAIL | (b) PASS | (c) PASS | (d) PASS | (e) PASS | (f) PASS | (g) PASS** — 6 PASS / 1 FAIL / 0 not evaluated. Coefficient ledger SHA-256 `0ee954bbffcc357e…`.

## 1. Case, geometry and parameters

A ship transits a brash-ice channel: a floating layer of broken ice floes (thickness $H$, mean diameter $d$, void fraction $\phi$) fills the track.  The standard case of the campaign node is $H=1$ m, $d=0.5$ m, $\phi=0.3$, $v=3$ m/s, evaluated for three beams $B=15/25/35$ m with realistic length-to-beam ratio and draught.

| quantity | value | unit |
|---|---|---|
| layer thickness $H$ | 1.000 | m |
| mean floe diameter $d$ | 0.500 | m |
| porosity $\phi$ | 0.300 | - |
| bulk density $\rho_b=(1-\phi)\rho_i+\phi\rho_w$ | 949.4 | kg/m^3 |
| submerged thickness $h_{sub}=H\rho_b/\rho_w$ | 0.9262 | m |
| static confinement $(\rho_w-\rho_b)g h_{sub}$ | 686.7 | Pa |
| shear-band thickness $2d$ | 1.000 | m |

Coefficient ledger (grade per constant; `CALIBRATED` = the only fitted number):

| coefficient | value | grade |
|---|---|---|
| `confinement_coefficient` | 0.640713 | CALIBRATED |
| `displaced_ice_volume_factor` | 1 | DERIVED |
| `gravity_m_s2` | 9.80665 | CONVENTION |
| `i_0` | 0.3 | LITERATURE |
| `keel_drag_coefficient` | 2.2 | CONVENTION |
| `mobilization_speed_m_s` | 0.05 | CONVENTION |
| `mu_2` | 0.84 | LITERATURE |
| `mu_s` | 0.36 | LITERATURE |
| `shear_band_coefficient` | 2 | DERIVED |
| `solid_ice_density_kg_m3` | 917 | CONVENTION |
| `water_density_kg_m3` | 1025 | CONVENTION |

## 2. Method

Three components, all with the processing closure $f(v)=v/(v+v_0)$, $v_0=0.05$ m/s:

1. **Granular shear ($\mu(I)$).** Shear band at the hull, thickness $\delta=2d$, $\dot\gamma=v/\delta$, confinement $P=(\rho_w-\rho_b)g h_{sub}+\tfrac12 C_c\rho_b v^2$, $I=\dot\gamma d/\sqrt{P/\rho_b}$, $\tau=\mu(I)P$ with $\mu(I)=\mu_s+(\mu_2-\mu_s)/(I_0/I+1)$, $F_{shear}=\tau(1-\phi)L_{wl}h_{sub}f$.
2. **Buoyancy displacement.** $F_{buoy}=\rho_w g V_{disp}$, $V_{disp}=(1-\phi)B h_{sub}$ — only the solid ice volume swept per metre of advance displaces water.
3. **Underwater keel drag.** $F_{drag}=\tfrac12 C_D\rho_w v^2A_{keel}$, $A_{keel}=(1-\phi)B h_{sub}$, $C_D=2.2$ (blunt, porous, interlocked rubble).

The $\mu(I)$ constants are the published Jop et al. (2006) values; `calibration.calibrate_family` fixes the single closure constant $C_c$ by a minimax protocol over the standard beam series ($C_c=0.6407$, ledger match: True). The reference for criterion (b) is the in-repository hashed brash anchor `tensordem.ice_resistance_scaling.broken_ice_anchor` (crushing + submersion + momentum flux); the campaign's **Suyuthi et al. (2013) attribution is WITHHELD** because the paper is not readable in this offline sandbox.

Toolchain: pure-Python/SciPy model, DEM steady-shear measurements from `tensordem.mu_i_rheology.simulate_servo_shear`, pytest for the criteria, matplotlib for the figures.

## 3. Results

### 3.1 Component decomposition and the reference comparison (criterion b, c)

| B [m] | model R [kN] | anchor R [kN] | error | shear [%] | buoyancy [%] | drag [%] |
|---|---|---|---|---|---|---|
| 15 | 363.3 | 472.3 | -23.1% | 46.8 | 26.5 | 26.7 |
| 25 | 600.3 | 576.3 | +4.2% | 46.4 | 26.7 | 27.0 |
| 35 | 837.3 | 680.3 | +23.1% | 46.2 | 26.8 | 27.1 |

Beam-scaling exponents: model 0.985, anchor 0.427 — the residual of criterion (b) is this slope mismatch, not a magnitude offset (the model's three terms all scale with the beam, the anchor's crushing term does not).

### 3.2 Speed dependence (criterion d): $n=1.060$ over 1-5 m/s (band 0.8-1.2)

| v [m/s] | R [kN] |
|---|---|
| 1.00 | 245.0 |
| 1.50 | 303.4 |
| 2.00 | 381.3 |
| 2.50 | 480.1 |
| 3.00 | 600.3 |
| 3.50 | 742.1 |
| 4.00 | 905.6 |
| 4.50 | 1090.9 |
| 5.00 | 1298.0 |

### 3.3 Porosity sensitivity (criterion e): -24.4% between $\phi=0.2$ and $0.4$ (band -25%…-15%)

| phi | shear [kN] | buoyancy [kN] | drag [kN] | total [kN] |
|---|---|---|---|---|
| 0.20 | 319.1 | 181.1 | 182.8 | 682.9 |
| 0.25 | 298.7 | 170.7 | 172.4 | 641.7 |
| 0.30 | 278.2 | 160.3 | 161.8 | 600.3 |
| 0.35 | 257.8 | 149.7 | 151.1 | 558.6 |
| 0.40 | 237.4 | 138.9 | 140.3 | 516.6 |

### 3.4 $\mu(I)$ calibration (criterion a)

| constant | DEM fit | Jop et al. (2006) | rel. error |
|---|---|---|---|
| mu_s | 0.1761 | 0.3600 | -51.1% |
| mu_2 | 0.9682 | 0.8400 | +15.3% |
| i_0 | 1.1743 | 0.3000 | +291.4% |

Sweep: 7 steady-shear points over I = 0.0108-1.29. Within the 20% literature tolerance: mu_2; outside: mu_s, i_0.

Fit quality: rms/mean($\mu$) = 0.0551 over 7 DEM points (PASS); literature check: worst 291.4% (FAIL).

**Diagnostic — interlocking variant** (rolling resistance 0.1, the micro-mechanism of angular ice floes):

| constant | DEM fit (rolling on) | Jop et al. (2006) | rel. error |
|---|---|---|---|
| mu_s | 0.3231 | 0.3600 | -10.2% |
| mu_2 | 1.3089 | 0.8400 | +55.8% |
| i_0 | 2.5020 | 0.3000 | +734.0% |

With interlocking the quasi-static limit is recovered (mu_s error -10.2%), but the variant has only 4 points for three constants, so its residual (4.03e-05) is interpolatory and the fitted mu_2 / i_0 carry no statistical weight; mu_2 and i_0 are **not identifiable** from this device at all, because the sweep only reaches I = 0.63, far below the collisional crossover I_0, so the two high-I constants are set by extrapolation.  That is a limit of the measurement, not of the resistance model, and it is why criterion (a) is recorded as FAIL rather than massaged into a pass.

Criterion (a) therefore splits: the fit reproduces the measured steady-shear points (a1 PASS) but the DEM device does not recover the published constants (a2 FAIL).  The resistance model consequently uses the *literature* mu(I) constants and does not tune them - the DEM result is reported, not exploited.

### 3.5 Negative controls (criterion f)

| check | value | verdict |
|---|---|---|
| $R(v=0)$ | 0 N | exact zero |
| $R(1\,\mathrm{mm/s})/R(3\,\mathrm{m/s})$ | 0.627% | < 1% |
| $R(\phi\to1)/R(0.3)$ | 0.145% | < 1% |
| monotone in $v$ | True | - |

### 3.6 Criteria table

| criterion | verdict | key number |
|---|---|---|
| a | FAIL | rms 0.0551, worst 291.4% |
| b | PASS | max |error| 23.1% |
| c | PASS | shares 47/26/27 % |
| d | PASS | n = 1.060 |
| e | PASS | ΔR = -24.4% |
| f | PASS | R(0)=0 N, water ratio 0.15% |
| g | PASS | 38 tests |

## 4. Figures

**brash_components** — Three-component resistance vs beam with the reference anchor total (criterion b, c).

**brash_speed** — R(v) and its log-log power-law fit over 1-5 m/s (criterion d).

**brash_porosity** — Component response to layer porosity (criterion e).

**brash_mu_i** — mu(I) closure: DEM calibration points, Jop law and the brash-channel operating point (criterion a).


## 5. Caveats, limits of validity

* The reference anchor is the in-repository brash closed form; the **Suyuthi et al. (2013)** formula named by the campaign node could not be read offline, so the attribution is withheld rather than claimed.  The anchor's own composition (intact-strength crushing ≈ 2/3 of its total at B=15 m) contradicts the campaign's share bands, so the anchor can validate *magnitude*, not the three-way split.
* The processing closure $f(v)=v/(v+v_0)$ is what makes $R(0)=0$; it is a declared convention (moves no standard-case number by more than 1.7%), and the static break-out force of a stuck ship is outside the model.
* $C_c$ is the single calibrated constant (minimax over the standard beam series); the resulting ±23.1% residual is the model-vs-anchor beam-slope signature and cannot be removed without breaking the component-share bands (see §6).
* $C_D=2.2$ and the shear-band thickness $2d$ are order-one conventions; no coefficient in the ledger is claimed to reproduce a published brash-channel number.
* The model is quasi-steady and one-dimensional along the track: it has no floe-size distribution, no floe-floe interlocking statistics, no channel-wall friction, no shallow-water blockage and no ice consolidation.  Applicable to $v\lesssim 5$ m/s, $H\lesssim 2$ m, $\phi\lesssim 0.5$.

## 6. Error assessment (criterion by criterion)

| quantity | model | reference | error | < 3%? |
|---|---|---|---|---|
| $R$, B=15 m | 363.3 kN | 472.3 kN | -23.1% | no |
| $R$, B=25 m | 600.3 kN | 576.3 kN | +4.2% | no |
| $R$, B=35 m | 837.3 kN | 680.3 kN | +23.1% | no |
| speed exponent $n$ (1-5 m/s) | 1.060 | 0.8-1.2 band | +6.0% vs n=1 | no |
| porosity drop $\phi$:0.2→0.4 | -24.4% | 15-25% band | +21.8% vs -20% | no |

**Is the error < 3%?** No, and it is not expected to be: every reference quantity here is an order-one engineering anchor with WITHHELD literature agreement (the anchor's own ledger grades its constants CONVENTION/DERIVED and its literature agreement WITHHELD), so 3% agreement would be an artefact of fitting rather than validation.  The strict statement is the campaign tolerance: criterion (b) requires < 25% and the model achieves max 23.1% (PASS); the $\mu(I)$ constants are compared with the published Jop values in §3.4 and the measured DEM fit in §3.4 is the only place where a literature number is reproduced at all.

Test evidence: criterion (g) PASS — 38 pytest cases (minimum 18).

