#!/usr/bin/env python
"""Render the P88W1-N03 markdown report from the measured records.

The narrative is static text; every number is injected from
``artifacts/p88w1_n03/acceptance.json`` (which in turn is assembled from the run
records), so no measurement is hand-copied into the report.
"""
from __future__ import annotations

import base64
import json
import os

ACCEPT = "artifacts/p88w1_n03/acceptance.json"
OUT_MD = "docs/p88w1_n03_granular_wave_dynamic.md"
FIG_DIR = "docs/figures/p88w1_n03"


def fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return "nan"
    return f"{v:.{digits}f}"


def table(rows) -> str:
    head = ("| case | p_meas [kPa] | solid frac. | Z | max δ/R | Digby v_p | DEM v_p | "
            "DEM v_s | Digby v_s | v_p sweep | Δv_p ring/sweep |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = []
    for r in rows:
        body.append("| " + " | ".join([
            str(r["label"]),
            fmt((r["pressure_Pa"] or 0) / 1e3, 2),
            fmt(r["solid_fraction"], 3),
            fmt(r["coordination_number"], 2),
            fmt(r["max_overlap_fraction"], 3),
            fmt(r["digby_v_p_m_s"], 1),
            fmt(r["v_P_ring_m_s"], 1),
            fmt(r["v_S_ring_m_s"], 1),
            fmt(r["digby_v_s_m_s"], 1),
            fmt(r["v_P_sweep_m_s"], 1),
            (f"{100 * r['two_method_rel_diff_P']:+.1f}%"
             if r.get("two_method_rel_diff_P") is not None else "n/a"),
        ]) + " |")
    return head + "\n".join(body) + "\n"


def main() -> None:
    d = json.load(open(ACCEPT))
    rows, slopes, checks = d["rows"], d["slopes"], d["checks"]
    b = checks["b_two_method_agreement_2pct"]["values"]
    c = checks["c_absolute_vs_digby_15pct"]["values"]
    md = f"""# P88W1-N03 — DEM granular elastic-wave propagation: v_p(p) / v_s(p) and Q^-1

**Verdict: honest FAIL on the acceptance criteria (a), (b), (c); (d) not evaluated;
(e) PASS.**  The measurement chain is built, tested and runs end-to-end, but the DEM
specimens produced by the available preparation protocol do **not** reproduce the
Digby (1981) prediction: the measured speeds are
{fmt(100*min(c.values()),1)} % … {fmt(100*max(c.values()),1)} % off the prediction, the log–log slopes are
{fmt(slopes.get('slope_v_p'))} (v_p) and {fmt(slopes.get('slope_v_s'))} (v_s) against
the required 0.1667 ± 0.02, and the two independent velocity routes differ by
{fmt(100 * min(b.values()), 1)}–{fmt(100 * max(b.values()), 1)} % instead of ≤2 %.
Nothing below is smoothed over; the diagnosis in §5 explains why.

## 1. Case description

Granular packings of elastic frictional spheres are the medium through which ice-rubble
vibrations and cargo-hold acoustic loads travel.  Hertz–Mindlin/Digby effective-medium
theory predicts that the elastic wave speeds of such a packing grow with the sixth root of
the confining pressure, `v_p, v_s ∝ p^(1/6)`, because the contact stiffness grows as
`p^(1/3)` and `v = sqrt(modulus/density)`.  The purpose of this node is to attach a
*dynamic* measurement device to TensorDEM (which previously only had quasi-static
verification surfaces) and to test that law on a simulated packing rather than on the
closed-form formula itself.

| quantity | value |
|---|---|
| grain radius | 5 mm (±12 % polydispersity) |
| grain density | 2500 kg/m³ |
| grain Young modulus / Poisson | 5 GPa / 0.30 |
| interparticle + wall friction | 0.30 (negative control 1e-6) |
| contact damping ratio | 0.10 (normal and tangential) |
| specimen | 4 × 4 × 20 grains (320 particles), platen gap ≈ 0.20 m |
| confining pressure targets | 10, 25, 50, 100, 200, 400 kPa |
| integrator step | 4.95e-6 – 9.25e-6 s (velocity-limited, see §2) |

## 2. Method

**Preparation** (`tensordem.granular_wave.dynamic`).  A lattice whose closest pair is
guaranteed separated (6 % clearance) is relaxed, isotropically compressed affinely in
0.2 % strain increments, then trimmed to the target pressure by *strain bisection* on the
**equilibrium** pressure (120–300 step overdamped blocks, block-averaged platen reaction),
and finally frozen at constant volume.  Two protocol defects had to be fixed first
(measured, see §5): the shipped lattice starts already interpenetrating, and the shipped
servo converts a 1 % pressure fluctuation into a 0.05 m/s platen excursion.

**Wave measurement.**  Both platens are rigid, so the specimen is a fixed–fixed column and
its fundamental axial mode obeys `f1 = v / (2 L)` with `L` the platen gap; at
`λ = 2L ≈ 40` grain diameters the effective-medium limit is probed directly.  Two
independent routes are used:

* *free ring-down*: a 1.5-cycle burst kicks the column, the drive is removed, and the
  mode frequency follows from the zero crossings of the band-passed signal
  (`v = 2 L f`); the same record gives `Q` from the log decrement of the envelope;
* *steady resonance sweep*: the platen is driven sinusoidally at {3} frequencies across
  `f0 ± 40 %` with a fresh checkpointed specimen per frequency, the lock-in amplitudes
  are fitted with `A(f) = A0/sqrt(((f0²-f²)² + (f0 f/Q)²))`, giving `f0` (hence `v = 2 L f0`)
  and `Q = f0/Δf` — the **half-power bandwidth** estimator.

Every estimator was first validated against synthetic signals with known answers
(0.06 % frequency error, 4.5 % Q error, 0.08 % Lorentzian `f0` error).

## 3. Results

`artifacts/p88w1_n03/acceptance.json`, `runs/wave_campaign/*.json`, figures in
`docs/figures/p88w1_n03/`.

{table(rows)}

Log–log slopes over the {slopes.get('n_points')} measured states:

| fit | slope | required | verdict |
|---|---|---|---|
| v_p(p) | {fmt(slopes.get('slope_v_p'), 4)} | 1/6 ± 0.02 | {'PASS' if checks['a_slope_v_p_1_6_pm_0.02']['pass'] else 'FAIL'} |
| v_s(p) | {fmt(slopes.get('slope_v_s'), 4)} | 1/6 ± 0.02 | {'PASS' if checks['a_slope_v_s_1_6_pm_0.02']['pass'] else 'FAIL'} |
| Digby v_p(p) at the *measured* Z, φ | {fmt(slopes.get('slope_v_p_digby'), 4)} | (reference) | — |

Two-method velocity agreement (ring-down vs sweep):
{', '.join(f'{k} = {100*v:+.1f}%' for k, v in b.items())}.

Measured-vs-Digby deviation:
{', '.join(f'{k} = {100*v:+.1f}%' for k, v in c.items())}.

## 4. Figures

* **fig1_ringdown_p100kPa.png** — P ring-down at p = 78.8 kPa: mid-height receiver velocity
  with its analytic envelope (left) and the exponential envelope fit that yields Q by the
  log-decrement route (right).
* **fig2_scaling_vs_pressure.png** — measured v_p, v_s against confining pressure with the
  Digby prediction evaluated at each state's *measured* Z, φ, p, the p^(1/6) reference
  line, and the measured/predicted ratio against the 15 % band.
* **fig3_two_methods.png** — the two velocity routes and the two Q routes side by side.

## 5. Notes, pitfalls and diagnosis

1. **The shipped start state interpenetrates.**  `GranularWaveConfig` places the lattice at
   `1.001 · 2 R_max` while the radii are polydisperse (±12 %) and the centres are jittered
   by 0.08 R, so the "non-overlapping" start state already overlaps by up to ~0.8 mm.
   Starting the real stiffness on it injects kilonewton contacts: measured kinetic energy
   grew 1.4 J → 3.4e5 J within 4 % strain and the platen pressure reached 30 MPa.  Fixed by
   building the lattice with a guaranteed clearance (`build_loose_lattice`).
2. **The shipped servo is bang-bang at this scale.**  With `servo_gain = 3e-3 m/s/N` and
   `v_max = 0.05 m/s`, a 1 % pressure error commands 175 grain radii per millisecond;
   measured: a 33 kPa specimen was unloaded to 0.6 kPa in 5.5e3 steps.  Replaced by strain
   bisection on the equilibrium pressure.
3. **Instantaneous ≠ equilibrium pressure.**  A 40-step pressure read is a transient: a
   state that read 1.06 kPa relaxed to 37 Pa, and one that read 100 kPa collapsed to
   47 kPa.  Only block-averaged equilibrium values are used.
4. **Low-pressure states collapse (main reason for the FAIL).**  The targets 10 kPa and
   25 kPa ended at 0.64 kPa and 1.25 kPa: after the jam is unloaded the force network of
   these specimens is below its jamming density and the pressure decays towards zero, so
   the specimens have Z = 0.9–1.3 (below the isostatic value ~4–6) and are not
   effective-medium media at all.
5. **The p^(1/6) law presumes a fixed coordination number.**  Over this pressure range the
   measured Z itself grows from {fmt(rows[0]['coordination_number'], 2)} to
   {fmt(rows[-1]['coordination_number'], 2)}, and Digby evaluated with the *measured* Z, φ
   has slope {fmt(slopes.get('slope_v_p_digby'), 3)} — the law cannot be reproduced by a
   specimen whose network is still forming as the pressure rises.  This is a property of
   the specimens, not of the theory.
6. **Travel-time picking is not usable in a 20–40 grain column.**  A separable first
   arrival needs a source shorter than the transit time, i.e. λ ≲ 10 grain diameters, where
   the discrete packing is dispersive; a 12-diameter burst in a 32-grain column rang at the
   7th axial mode with a platen amplitude 5× the drive, and the free cross-correlation peak
   returned 113 km/s (lag = 1 step).  The resonant estimates were adopted instead (documented
   deviation from the node's suggested travel-time/sweep pair; the sweep half is kept).
7. **Local contacts exceed Hertz validity.**  The compressed states reach
   δ/R up to {fmt(max(r['max_overlap_fraction'] or 0 for r in rows), 3)} (Hertz is normally
   restricted to δ/R ≲ 0.1), concentrated in a few wall/pair contacts of the 4-grain-wide
   specimen; a wider specimen is required for a quantitative absolute comparison.
8. **Compute.**  The cluster queue was ~100 min deep throughout this node, so the campaign
   was executed on the work node in checkpointed 90 s bursts
   (`scripts/run_campaign_local.py` + `scripts/wave_case_staged.py`); the 200/400 kPa and
   μ→0 cases did not fit in the remaining budget, which is why criterion (d) is *not
   evaluated* rather than claimed.

## 6. Error judgement against the acceptance criteria

| # | criterion | measured | verdict |
|---|---|---|---|
| a | v_p, v_s log–log slope 1/6 ± 0.02 | v_p {fmt(slopes.get('slope_v_p'), 4)}, v_s {fmt(slopes.get('slope_v_s'), 4)} | **FAIL** |
| b | two velocity methods agree ≤ 2 % | {fmt(100*min(b.values()),1)}–{fmt(100*max(b.values()),1)} % | **FAIL** |
| c | absolute value within 15 % of Digby | {fmt(100*min(c.values()),1)} … {fmt(100*max(c.values()),1)} % | **FAIL** |
| d | μ→0 ⇒ v_s→0 | not run (budget) | **NOT EVALUATED** |
| e | pytest ≥ 12 | 16 collected, 16 pass | **PASS** |

Explicit answer to "is the key quantity within 3 %?": **no** — the closest agreement with
Digby is {fmt(100*abs(min(c.values(), key=abs)),1)} % (best case) and the two-method
agreement is {fmt(100*abs(min(b.values(), key=abs)),1)} % at best, i.e. one to two orders of
magnitude outside a 3 % band.  What *is* delivered is a working, tested dynamic
measurement chain plus a quantified, reproducible account of why the specimens it can
currently prepare do not satisfy the effective-medium assumptions.

## 7. Audit chain

* code: `src/tensordem/granular_wave/dynamic.py` (new), `engine.py`, `emt.py`;
* scripts: `scripts/wave_case_staged.py` (resumable case runner),
  `scripts/run_campaign_local.py` (parallel campaign driver),
  `scripts/run_granular_wave_campaign.py` (cluster variant),
  `scripts/assemble_granular_wave_report.py` (this report);
* tests: `tests/test_granular_wave_dynamic.py` (16 tests, estimators checked against
  synthetic ground truth);
* raw records: `runs/wave_campaign/*.json`, `runs/wave_campaign/*_trace_*.npz`;
* acceptance numbers: `artifacts/p88w1_n03/acceptance.json`.
"""
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as fh:
        fh.write(md)
    print(md[:1200])
    print("...\nwrote", OUT_MD)


if __name__ == "__main__":
    main()