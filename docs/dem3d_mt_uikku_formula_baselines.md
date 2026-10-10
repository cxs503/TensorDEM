# MT Uikku empirical ice-resistance formula baselines

## Purpose

This benchmark scores four published empirical predictions against the experimental
reference values transcribed from Table 8 of Hu and Zhou (2016). It provides a
transparent baseline for later DEM validation; it does **not** compare TensorDEM
outputs with experiments.

Reference dataset:
- `benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv`

Literature source:
- Hu, Z. and Zhou, L. (2016), “Further study on level ice resistance and channel
  resistance for an icebreaking vessel,” *International Journal of Naval
  Architecture and Ocean Engineering*, 8(2), 169–176.
- DOI: [10.1016/j.ijnaoe.2016.01.004](https://doi.org/10.1016/j.ijnaoe.2016.01.004)

## Run

```bash
python scripts/benchmark_dem3d_ship_ice_formula_baselines.py \
  --output results-dem3d-mt-uikku-formulas
python -m unittest tests.test_dem3d_ship_ice_formula_baselines
```

The run writes a JSON report, a per-case CSV and a per-formula summary CSV.

## Metrics

For each formula and all ten reference cases:

- **MAPE**: mean absolute percentage error, in percent.
- **Mean signed error**: prediction minus experiment; positive means overprediction.
- **RMSE**: root mean square error in kN.
- Counts of overpredictions, underpredictions and exact matches.

The experimental reference uses the Table 8 reference column consistently. It is
not silently mixed with the Table 3 mean-force column, where small discrepancies
exist for several tests. Missing Table 3 force statistics for tests 401–403 remain
missing in the source CSV; they are not inferred by this scoring script.

## Interpretation and acceptance

These scores describe the empirical formulas on the cited reference set. Do not
interpret the best formula as proof that TensorDEM is valid, or use the formula
MAPE as the DEM's error. The DEM must first reproduce comparable conditions and
resistance definitions, including ice properties, model/full-scale conversion,
bow geometry, hull–ice friction, ice breaking and submergence effects. Material
parameters should be calibrated on independent specimen tests, then frozen for
the ship-resistance comparison.

No acceptance threshold is imposed here: choosing one requires a declared
engineering use case and uncertainty budget.
