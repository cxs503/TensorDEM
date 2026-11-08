# `beverloo_discharge` measurement audit — a fabricated "measurement" replaced by a DEM mass balance

**Mission**: m6457 (self-start, models seat) · **Scope**: `src/tensordem/beverloo_discharge.py`,
`tests/test_beverloo_discharge_measurement.py`, `tests/test_dem_benchmarks.py` (one vacuous test
replaced), `scripts/m6457_beverloo_measurement_sweep.py`, `scripts/m6457_beverloo_analysis.py` ·
**Evidence grade**: wrapper **MOCK → REAL**; the underlying silo physics keeps m6296's honest FAIL and
is now bounded (§4–§5).

---

## 1. Case introduction — what was audited

`beverloo_discharge` packages the classic silo benchmark: a bulk solid drains through a bottom orifice
and the mass flow rate follows Beverloo et al. (1961)

```
W = C · rho_b · sqrt(g) · (D - k·d)^(5/2)        C ~ 0.58, k ~ 1.4...1.6 (glass beads)
```

The module shipped two analytic helpers (`beverloo_prediction`, `beverloo_2d_prediction`) and one
function that *looked* like the simulation: `simulate_discharge(case)`, documented as "Run silo
discharge using the repose_calibration solver". It had no test and no document anywhere in the
repository (grep: only `tests/test_dem_benchmarks.py` and `tests/test_granular_benchmarks.py` import
the module, and they touch only the analytic helpers).

**Defect found.** `simulate_discharge` ran an unrelated *repose* (cylinder-lift) simulation, discarded
the trajectory and returned

| returned field | value | why it is wrong |
|---|---|---|
| `flow_rate_kg_s` | `beverloo_prediction(case)["W_kg_s"]` | the *prediction* returned as the measurement (bit-identical) |
| `n_discharged` | `0` | nothing was discharged — the repose model has no orifice |
| `total_discharged_mass_kg` | `0.0` | idem |
| `r2` | `1.0` (constant) | a perfect "agreement" no data produced |
| `t_settle_s` | `0.0` (constant) | idem |

Reproduced on an 80-particle case (79 s CPU, dt = 1.8186e-05, 4125 steps): the returned rate was
`0.002941253508448396` kg/s, **exactly** `beverloo_prediction(...)["W_kg_s"]`, with `n_discharged = 0`
and `r2 = 1.0`. A consumer of that function would have recorded a 0 % "validation error" against
Beverloo without a single contact integrated through an orifice.

Second finding: `tests/test_dem_benchmarks.py::test_beverloo_C_k_calibration` asserted
`abs(0.58 - 0.58) / 0.58 < 0.10` while its docstring claimed "Calibrated C, k within 10 % of
literature" — a literal compared to itself. Replaced by a real inversion round-trip.

## 2. What was done (method, tool chain)

1. **`simulate_discharge` now measures.** It delegates the integration to the verified
   flat-bottom-orifice solver `tensordem.silo_discharge.simulate_silo` (m5481; three contact defects
   fixed and verified in m6296) and returns the *mass-balance* rate `W = mean(dm_discharged/window)`.
   `rel_err_vs_beverloo`, `recovered_C`, `recovered_C_at_reference_rho` carry the comparison
   separately; `r2` is now the coefficient of determination of cumulative discharged mass vs time
   (*steadiness*, not agreement) with `r2_basis` stating so.
2. **Discharge protocol explicit**: `settle_time`, `window`, `n_windows` (>= 5), `max_time` =
   settle + discharge, `plate_thickness` (default 2 r). A case whose `max_time` cannot cover the
   windows is still constructible for analytic use, but `to_silo_case`/`simulate_discharge` refuse it.
3. **Transparent mass accounting**: `window_span_discharged_mass_kg` and
   `outside_window_discharged_mass_kg` separate mass that leaves during settling from the window
   average, so the rate cannot be inflated by pre-window losses.
4. **Initial-state stability guard**: a jittered HCP lattice can be *born* interpenetrated and the
   solver then diverges. `initial_interpenetration()` measures the worst overlap in units of r and
   `simulate_discharge` refuses above `MAX_INITIAL_OVERLAP_OVER_RADIUS = 0.25` with a clear
   `BeverlooError` instead of dying 300 s later inside `cKDTree.query_pairs` (floating-point overflow).
5. **Campaign**: `scripts/m6457_beverloo_measurement_sweep.py` ran 6 cases on the 702 cluster
   (n315/n330, loongarch64) through the *wrapper* interface — one JSON per case in
   `runs/m6457_beverloo/`, raw stdout `runs/raw_n315.log`, `runs/raw_n330.log`.
   `scripts/m6457_beverloo_analysis.py` builds `artifacts/m6457/summary.json` and the three figures.
   Every digit below is printed by that script from those JSONs.
6. **Tests**: `tests/test_beverloo_discharge_measurement.py` (11 tests, 23 s) pins the tautology
   regression, the mass-balance identity, the r² definition, the overlap guard, the case validation
   and the case->silo mapping.

## 3. Results

### 3.1 Measured vs predicted flow rate (programmatic table)

| case | D/d | n0 | discharged | phi (measured) | W_meas (kg/s) | +/-1 sigma | W_lit (kg/s) | rel. err | steady r2 | windows>0 | CV(windows 2-5) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_base_D5 | 5.00 | 1655 | 73 | 0.641 | 0.3225 | 0.5657 | 1.8356 | -82.4% | 0.771 | 0.80 | 1.052 | 517 |
| F_base_D4 | 4.00 | 1655 | 20 | 0.641 | 0.0884 | 0.1976 | 0.7915 | -88.8% | 0.000 | 0.20 | n/a | 556 |
| B_jitter_D5 | 5.00 | 1000 | 489 | 0.387 | 1.6434 | 0.1445 | 1.8356 | -10.5% | 0.999 | 1.00 | 0.052 | 170 |
| C_lowmu_D5 | 5.00 | 1000 | 552 | 0.387 | 1.8290 | 0.2314 | 1.8356 | -0.4% | 0.998 | 1.00 | 0.100 | 159 |
| E_lowmu_D5_tall | 5.00 | 1750 | 633 | 0.407 | 1.9704 | 0.1244 | 1.8356 | +7.3% | 1.000 | 1.00 | 0.031 | 365 |
| D_lowmu_D4 | 4.00 | 1000 | 293 | 0.387 | 0.9189 | 0.1420 | 0.7915 | +16.1% | 0.995 | 1.00 | 0.155 | 222 |

`A/F` are the **ordered-lattice controls** (jitter 0.002 r, lattice gap 0.02 -> initial overlap 0.0);
`B/C/D/E` are **disordered** packings (gap 0.15, jitter 0.08 r -> overlap 0.042 r). Particle size
d = 0.015 m, rho_p = 2500 kg/m3, silo R = 0.11 m in all cases.

### 3.2 The measurements are real measurements

* mass-balance residual `|dm - N_d*m_p| <= 1.8e-15 kg` in all six runs (float64 round-off);
* every flowing case has 5/5 non-zero windows and `r2 >= 0.995`, while the stalled controls show the
  start-up burst then zeros (`A: [1.33, 0.20, 0.044, 0.044, 0]` kg/s, `F: [0.44, 0, 0, 0, 0]`) with
  `r2 = 0.771 / 0.000`;
* the measured rate is never the correlation value (asserted in the suite).

### 3.3 What the campaign says about the underlying solver (m6296 follow-up)

The ordered-lattice stall that made m6296 an honest FAIL is a **packing artefact, not a contact-law
failure**: changing only the initial packing (A -> B; same mu = 0.4, same D/d = 5) turns a stalled
burst (`W = 0.3225 kg/s`, `-82.4%`) into a steady flow
(`W = 1.6434 kg/s`, `-10.5%`, CV of windows 2-5 = 5.2%).
Mean |error| over the two ordered controls is 85.6% vs
8.6% over the four flowing cases.

### Figures

* `docs/figures/m6457_W_measured_vs_literature.png` — measured W +/- 1 sigma vs the Beverloo prediction.
* `docs/figures/m6457_window_series.png` — per-window rates: burst + arching (A/F) vs steady (B/C/D/E).
* `docs/figures/m6457_recovered_C_vs_packing.png` — C recovered with the measured rho_b vs packing fraction.

## 4. Error judgement (explicit)

| quantity | measured | reference | rel. error | < 3 % ? |
|---|---|---|---|---|
| W, D/d = 5, mu = 0.15, H = 0.12 m (C) | 1.8290 kg/s | 1.8356 kg/s | -0.4% | yes (-0.4 %) |
| W, D/d = 5, mu = 0.15, H = 0.20 m (E) | 1.9704 kg/s | 1.8356 kg/s | +7.3% | no |
| W, D/d = 5, mu = 0.4 (B) | 1.6434 kg/s | 1.8356 kg/s | -10.5% | no |
| W, D/d = 4, mu = 0.15 (D) | 0.9189 kg/s | 0.7915 kg/s | +16.1% | no |
| W, ordered controls (A / F) | 0.3225 / 0.0884 kg/s | 1.8356 / 0.7915 kg/s | -82.4% / -88.8% | no |
| C recovered at reference rho_b (C / E / B / D) | 0.578 / 0.623 / 0.519 / 0.673 | 0.58 | -0% ... +16% | no |

**"Is the key quantity within 3 %": one of six cases is (-0.4 %), and that match must not be read as a
validated C = 0.58.** Part of the agreement is a packing-density compensation: the disordered bed
measures phi = 0.387 (rho_b = 968 kg/m3) while the correlation is
evaluated at the literature reference 0.64 rho_p = 1600 kg/m3. Normalising with the *measured* bulk
density gives C = 0.955 (+65%), so the module's own
"C ~ 0.58" stays **WITHHELD** as a physical claim.

## 5. Notes, limitations, honest FAIL carried forward

* **The wrapper change is a defect fix, not a physics improvement.** The silo physics is m6296's: a
  stable 3-D DEM silo whose steady rate sits 10-16 % from Beverloo for flowing disordered beds and
  which stalls entirely on the ordered lattice.
* **Only two orifice diameters (D/d = 4, 5) flow**, so no independent two-parameter (C, k) fit is
  claimed; `silo_discharge.beverloo_fit` needs >= 3 diameters on the flowing configuration — the next
  node for this trajectory.
* **Head independence is weakly tested**: E (H = 0.20 m) vs C (H = 0.12 m) differs by
  +7.7%, inside the 10 % band but with
  different particle counts (1750 vs 1000) and bulk densities (phi 0.407 vs
  0.387).
* **The overlap guard is empirically calibrated, not derived**: 0.19 r ran, 0.415 r diverged; 0.25 r
  sits in between. The exact failure mode inside `granular3d_core` was not isolated.
* **A -> B moves two knobs together** (`packing_gap` and `jitter`) because no jitter disorders the
  lattice without exceeding the gap; the honest statement is "ordered vs disordered packing".
* **Not addressed**: polydispersity, rough walls (`wall_mu > 0`), hopper bottoms, larger D/d where the
  silo wall dominates.

## 6. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_beverloo_discharge_measurement.py -q      # 11 tests
PYTHONPATH=src python -m pytest tests/test_silo_discharge_beverloo.py -q             # 12 tests
PYTHONPATH=src python -m pytest tests/test_dem_benchmarks.py -k Beverloo -q
PYTHONPATH=src python scripts/m6457_beverloo_measurement_sweep.py --quick           # 80-particle smoke
PYTHONPATH=src python scripts/m6457_beverloo_measurement_sweep.py --case A_base_D5 B_jitter_D5 C_lowmu_D5
PYTHONPATH=src python scripts/m6457_beverloo_measurement_sweep.py --case E_lowmu_D5_tall F_base_D4 D_lowmu_D4
PYTHONPATH=src python scripts/m6457_beverloo_analysis.py                            # summary.json + figures
```
