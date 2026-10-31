# `POST /calculations/channel-resistance` — API contract

Handler: `tensordem.channel_resistance.handle_channel_resistance(payload)`
Pure function of its payload; no state, no I/O. Returns a JSON-serialisable dict.

## Request

```json
{
  "beam_m": 4.0,
  "thickness_m": 1.0,
  "concentration": 0.6,
  "velocity_ms": 0.5,
  "sigma_f_pa": 500000.0,
  "channel_coefficient": 0.0485,
  "concentration_exponent": 0.951,
  "thickness_exponent": 0.034,
  "velocity_coefficient": 0.0,
  "convoy": {
    "ship_length_m": 2.355,
    "n_escort": 2,
    "spacing_m": 2.4,
    "consolidation_gain": 1.067,
    "consolidation_length_m": 7.93,
    "interference_penalty": 2.567,
    "interference_length_m": 0.50
  }
}
```

Required: `beam_m`, `thickness_m`, `concentration`, `channel_coefficient`.
Optional: `velocity_ms` (0.5), `sigma_f_pa` (5.0e5), `concentration_exponent`
(1.0), `thickness_exponent` (1.0), `velocity_coefficient` (0.0), `convoy`.

## Response

```json
{
  "status": "ok",
  "resistance_n": 91234.5,
  "resistance_kn": 91.2345,
  "model": "Riska(1989) brash-ice channel resistance",
  "form": "R = C_h * (B * h * sigma_f) * (C/C_ref)^p * (h/h_ref)^a * (1 + k_v v)",
  "inputs": {"...": "echo of the request"},
  "reference": {"concentration": 0.6, "thickness_m": 1.0, "sigma_f_pa": 500000.0},
  "limitations": ["...", "..."],
  "convoy": {
    "spacing_opt_m": 3.2,
    "spacing_bracket_m": [1.177, 7.064],
    "energy_opt_j_per_m": 1053362.0,
    "energy_at_lower_bound_j_per_m": 1957131.0,
    "energy_at_upper_bound_j_per_m": 1141265.0,
    "interior": true,
    "curvature_j_per_m3": 55085.0,
    "spacing_opt_over_length": 1.36,
    "parameters": {"...": "echo"},
    "energy_at_requested_spacing_j_per_m": 1070000.0
  }
}
```

`interior` is `false` whenever the minimum sits on the bracket, in which case
`curvature_j_per_m3` is `NaN`; consumers must not treat a boundary optimum as an
interior one.

## Errors

* `KeyError` — a required field is missing.
* `ValueError` — payload is not a mapping, a length/coefficient is not positive,
  a value is not finite, or the spacing bracket is empty.

## Validity range

`0.25 <= concentration <= 0.7`, `0.3 m <= thickness_m <= 2.0 m`,
`beam_m` within ±25 % of the 4 m calibration hull, `velocity_ms <= 1.5`.
The coefficient set was calibrated on the 2-D plan-view DEM of this repository;
it is not a substitute for full-scale channel-resistance measurements.