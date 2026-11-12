# CASE-DEM-PLANE-SHEAR-MUIDI — Round 5: the constant-stress premise is violated (audit + device gate)

> **Anchor**: `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=open)
> **Repository**: TensorDEM · **Mission**: m16671 · **Round**: 5 (after R1 m14960, R2 m16427, R3 m16497/m16568, R4 m16633)
> **Artifacts**: `benchmarks/plane_shear_mu_i/servo_validity_fix.py`,
> `runs/servo_validity_gate.json`, `runs/servo_validity_gate_tables.md`,
> `runs/fig_servo_validity.png`, `test_servo_validity_gate.py`
> **Device change**: `tensordem.mu_i_rheology.simulate_servo_shear` validity gate

## ① Case and why it matters

The case confronts the in-repo servo shear cell (`simulate_servo_shear`) with the
da Cruz et al. (2005) / GDR MiDi (2004) `mu(I)` and `phi(I)` laws for glass
beads. Device: x-periodic box, rough platen at constant normal stress
`P = 62.5 kPa` (`kn = 62.5 kN/m`, `mu_contact = 0.5`, `mu_rolling = 0.05`,
`e = 0.3`, `r = 5 mm`, `rho = 2500 kg/m3`, `gamma = 0.2`, seed 5279), bed
`0.12 x 0.05 x 0.06 m` (5d x 6 layers, N = 294) for the `baseline` arm and
`0.12 x 0.08 x 0.09 m` (8d x 9 layers, N = 924) for the `thick` arm; six
inertial numbers `I = 1e-3 ... 0.3`. The whole verification rests on one
premise: **`mu = tau_wall / p_wall` is a constant-stress measurement**, i.e. the
platen really holds `P`.

## ② What was done

1. Re-ran the stored round-3 points with the current code, gate off. The output
   is **bit-identical** to the stored artifacts for both arms
   (`mu = 0.4915930657...` / `0.5686248840...`), so the discrepancy below is not
   stale data (the failure mode R4 suspected for `phi`).
2. Quantified the premise: the platen pressure error
   `p_meas/P_applied - 1` is reported by the device nowhere and is large:
   baseline `-16.5 %`, thick `+57.8 %` at `I = 0.3`; in the **final 20 % window**
   the drift is `-31.6 %` / `+69.9 %`.
3. Added a constant-stress validity gate to the device: once the requested
   strain budget is spent, shearing continues (same contact law, same imposed
   rate, no physics change) until the servo holds `P` within
   `pressure_control_tol = 5 %` for `pressure_recover_window = 200` steps (cap
   `max_recover_steps = 20000`); the result now carries
   `pressure_control_error(_final)`, `servo_valid`, `recover_steps`,
   `shear_steps_executed`, `mu_final`, `phi_bed_final`.
   `max_recover_steps = 0` reproduces rounds 1-3 exactly.
4. Re-measured the invalid points with the gate on and compared, plus
   reproducible tests (`test_servo_validity_gate.py`, 12 tests).

## ③ Results (programmatic, from `runs/servo_validity_gate_tables.md`)

| arm | I | gate | p err (tail) | p err (final) | servo_valid | recover steps | mu final | mu err vs ref |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.01 | on | -8.3 % | -0.4 % | yes | 702 | 0.3959 | +5.3 % |
| baseline | 0.1 | on | -12.7 % | -0.6 % | yes | 3492 | 0.3234 | -33.5 % |
| baseline | 0.3 | off | -16.5 % | -31.6 % | **no** | 0 | 0.5538 | -9.3 % |
| baseline | 0.3 | on | -3.6 % | -0.8 % | yes | 20000 | 0.3477 | -42.3 % |
| thick | 0.1 | on | +5.2 % | -0.4 % | yes | 521 | 0.3818 | -20.1 % |
| thick | 0.3 | off | +57.8 % | +69.9 % | **no** | 0 | 0.6204 | +8.3 % |
| thick | 0.3 | on | +6.1 % | +3.1 % | yes | 8988 | 0.4652 | -22.0 % |

Elevation of the round-3 mis-measurement: the previously reported tail-mean
`mu` (0.4916 baseline / 0.5686 thick at `I = 0.3`) sits *inside* the 10 % band
(and the thick point even reads +8.3 % high/close), but once the stress is
actually held the same state gives `0.3477` (-42 %) and `0.4652` (-22 %).
**The control violation was worth 30-40 % of `mu`, i.e. more than the model
error it was being used to judge.** The `phi` reference of the case is also
sign-inverted (in-repo law `phi = 0.58 + 0.06/(I0/I+1)` *increases* with `I`,
while every round-3 arm dilates: `phi_bed_steady` falls with `I`, e.g. thick
`0.5769 -> 0.5736`, and gate-on thick `0.5733 -> 0.5363`), so `phi` agreement
(-2 % for thick) is coincidence, not verification.

## ④ Figures

`runs/fig_servo_validity.png`: left `mu` vs `I`, right `phi` vs `I`, reference
law as black curve, gate-off artifacts as grey dashed, gate-on as filled
markers. The gate-on `mu` collapses away from the reference while the
`phi` curve moves the *opposite* way to the reference law.

## ⑤ Caveats / limitations found

* **The bed is not stationary at high I even under valid stress**: baseline
  `I = 0.3` needs the full 20000 recovery steps and the bed keeps compacting, so
  `phi_final` (0.5775) ends up *above* the low-I value (0.5326) — `phi(I)` in
  that range stays WITHHELD rather than "verified".
* Strain achieved during recovery is large (up to 98, vs the requested 0.2), so
  the gate-on high-I point is a long-sheared, compacted state, not the state
  round 1-3 intended to measure. A protocol redesign (pre-consolidate the bed to
  the target stress, N >> 294, >= 10 layers, bulk virial stress instead of wall
  traction) is required before high-I `mu(I)` can be claimed at all.
* The platen is a rough wall (`wall_mu = 1.5`, above `mu_2 = 0.84`), so the
  measured `mu` is not capped by the wall; the invariant `mu <= wall_mu` is
  tested as a sanity guard.

## ⑥ Error adjudication

| quantity | before (as published R3) | after (valid stress) | criterion | verdict |
|---|---|---|---|---|
| mu at I = 0.01 | +2.2 % | +5.3 % | < 10 % | PASS |
| mu at I = 0.1 | -13.2 % | -33.5 % | < 10 % | FAIL |
| mu at I = 0.3 | -19.5 % | -42.3 % | < 10 % | FAIL |
| phi (all I) | -10 % ... -15 % (stale issue) / -2 % thick | -11.9 % at I=0.3 (thick) | < 10 % | FAIL / WITHHELD |
| pressure control | up to +57.8 % (unreported) | <= 5 % (gated, 6/6 runs) | implicit | fixed |

**< 3 % achieved? No.** `mu` at low `I` is within 5.3 %, everything above
`I ~ 0.1` is 20-42 % low; the honest verdict is that `CASE-DEM-PLANE-SHEAR-MUIDI`
is **FAIL at high I** (measurement-validated), and the round-3 "marginal" verdict
was inflated by the control violation. Recommendation for the ledger: keep the
anchor open, split the acceptance into (a) `I in [1e-3, 0.04]`: `mu` and `phi`
within 10 % — achievable with the current device; (b) `I > 0.1`: WITHHELD
pending the protocol redesign; (c) require `servo_valid` for any reported point.
