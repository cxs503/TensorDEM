# TensorDEM3D particle-particle normal dashpot: engine closure (m2626)

**Status:** the pair-contact normal dashpot is now an observable, verified parameter.
Previously `DEM3DConfig.contact_damping` was a *dead parameter* for sphere–sphere
contacts: the pair law derived its dashpot exclusively from
`restitution_coefficient`, so injecting any `contact_damping` value into
`IceDEM3D` left the measured restitution at `e ≈ 1.0`.

## 1. The defect (evidence first)

`tests/test_restitution.py::EngineClosureTests` was red on pristine `HEAD`:

```
FAILED ...test_engine_matches_reference_law
FAILED ...test_pair_impact_matches_target
AssertionError: c=2.994 target=0.7 measured=0.99987   (rel err 42.8 %)
```

Root cause (`src/tensordem/dem3d.py`, pair loop):

```python
if cfg.restitution_coefficient < 1.0:      # only this branch set gamma_n
    gamma_n = 2*zeta*sqrt(k*m_eff)
else:
    gamma_n = 0.0                          # contact_damping never used
```

Three facts make this a contract defect rather than a design choice:

1. `contact_damping` is documented in the dataclass as the normal dashpot;
2. the **tool** branch (`tool_mag = tool_elastic - contact_damping*vn`) and the
   **platen** branch (`platen_damping`) already use the *absolute* convention —
   the platen branch passes the very same closure test at 3 %;
3. `recommended_dt` already includes `contact_damping` in its damping bound, so
   the stability estimate assumed a damper the force law did not have.

## 2. Fix (minimal, opt-in, defaults untouched)

New config field:

```python
contact_damping_model: str = "restitution"   # "restitution" | "absolute"
```

* `"restitution"` (default) — byte-for-byte legacy behaviour: `gamma_n` from
  `restitution_coefficient`, `contact_damping` still not in the pair law.
* `"absolute"` — `gamma_n = contact_damping` (same convention as tool/platen), so
  the pair law is the clipped law `F_n = max(k*δ − c·δ̇, 0)` whose reference
  solution is `tensordem.restitution.unilateral_restitution(c, k, m_eff)`.
* Setting `"absolute"` together with `restitution_coefficient < 1` raises
  `ValueError` — two dashpot conventions at once are ambiguous.

The default path is unchanged (`test_dem3d.py`, `test_dem3d_restitution.py`,
`test_dem3d_hertz_contact.py`, `test_contact_tangential.py`,
`test_dem3d_energy_audit.py` all green), so no regression artefact moves.

## 3. Case set-up

| quantity | value |
|---|---|
| sphere radius / density | 0.025 m / 917 kg·m⁻³ (ice) |
| particle mass | 0.060018 kg |
| contact stiffness (linear) | 5000 N·m⁻¹ |
| pair effective mass (two moving spheres) | 0.0300088 kg |
| platen effective mass (one moving sphere) | 0.0600175 kg |
| impact speed (each body) | 0.5 m·s⁻¹, pair closing rate 1.0 m·s⁻¹ |
| integrator | semi-implicit Euler, `dt = recommended_dt` = 6.002e-5 s (136 steps in contact) |
| reference | RK4 (4000 samples/period) of the same clipped law, `tensordem.restitution` |

## 4. Results — engine vs reference (restitution)

`artifacts/dem3d_pair_damping_closure/audit.json`, figure
`fig1_restitution_vs_damping.png`.

| c_n [N·s/m] | pair: engine | pair: ref (unilateral) | rel err | platen: engine | platen: ref | rel err |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.99987 | 1.00000 | −0.013 % | 1.00000 | 1.00000 | 0.000 % |
| 2 | 0.78333 | 0.78353 | −0.026 % | 0.83892 | 0.83947 | −0.066 % |
| 5 | 0.56636 | 0.56594 | +0.074 % | 0.65679 | 0.65972 | −0.444 % |
| 10 | 0.35481 | 0.35747 | −0.745 % | 0.46099 | 0.46276 | −0.383 % |
| 20 | 0.17495 | 0.17540 | −0.257 % | 0.25903 | 0.25901 | +0.006 % |
| 40 | 0.06555 | 0.06626 | −1.081 % | 0.11038 | 0.11113 | −0.681 % |

Inverted damping → target restitution (closure of `damping_for_restitution`):

| target e | pair c_n | pair measured | rel err | platen c_n | platen measured | rel err |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.8394 | 0.90009 | +0.010 % | 1.187 | 0.89976 | −0.027 % |
| 0.7 | 2.994 | 0.69971 | −0.041 % | 4.234 | 0.70045 | +0.064 % |
| 0.5 | 6.254 | 0.49932 | −0.136 % | 8.844 | 0.50083 | +0.166 % |
| 0.3 | 12.18 | 0.29981 | −0.063 % | 17.23 | 0.29916 | −0.279 % |

Aggregates: `max |e_engine − e_unilateral| = 0.00293` (**1.081 %** worst relative),
`max closure error = 0.279 %`. The bilateral textbook mapping
(`classical_damping`) misses the same targets by **32.3 %** — the clipped law
removes dissipation, so only the unilateral inversion closes it.

Additional consistency:

* **peak overlap** at default configuration: pair 1.838e-3 m vs reference
  1.843e-3 m (−0.27 %); platen 1.280e-3 m vs 1.407e-3 m (−9.0 %).
* **time-step refinement** (c_n = 10): e = 0.354807 at `dt`, 0.357359 at `dt/2`
  (spread 0.26 % against a 0.74 % law error) — the residual is *not*
  time-discretisation.
* **impact-speed study** (c_n = 10): e = 0.357363 / 0.354807 / 0.358404 at
  v = 0.1 / 0.5 / 2.0 m·s⁻¹ (spread 0.36 %), consistent with the speed
  independence of the linear-clipped law.

## 5. Honest limits

* **Duration comparison is not a closed metric.** The reference stops at the
  instant the clipped force reaches zero (detachment at δ > 0); the engine keeps
  stepping while δ > 0 with zero force and reports the exit at δ = 0, so
  `duration_err_vs_reference` reaches 196 % in the strongly damped rows. Both
  describe the same physics but time different events; only `e` (and peak
  overlap) are used as acceptance quantities.
* **No experimental calibration.** These tests prove implementation fidelity to
  the stated clipped law with `k = 5000 N·m⁻¹` (a penalty stiffness, not an ice
  modulus). They do not validate ice restitution against measurements, and the
  legacy default (`"restitution"`, `e = 1`) remains uncalibrated.
* **No angular DOF.** Normal dashpot only; tangential/rolling branches are
  unchanged (friction and rolling coefficients default to 0 in this campaign).
* **Hertz branch** scales `gamma_n` through the local secant stiffness for
  `restitution_coefficient`-mode only; `absolute` mode uses a constant `c_n`
  (the physically standard choice for a linear dashpot) — a nonlinear
  dashpot/stiffness combination is out of scope here.

## 6. Reproduction

```
python -m pytest tests/test_dem3d_pair_contact_damping.py tests/test_restitution.py -q
python scripts/verify_dem3d_restitution.py --output artifacts/dem3d_pair_damping_closure
```

`tests/test_dem3d_pair_contact_damping.py` pins both sides of the contract:
legacy default (`contact_damping` invisible, `e = 1`) and absolute mode
(monotonic in `c_n`, ≤3 % vs the reference law, ≤3 % on inverted targets,
dt-refinement-stable, platen convention consistent).