# Experimental reference: MT Uikku level-ice resistance

## Why this case is useful

This reference set is transcribed from Hu and Zhou, **“Further study on level ice resistance and channel resistance for an icebreaking vessel,”** *International Journal of Naval Architecture and Ocean Engineering*, 8(2), 169–176 (2016), DOI: [10.1016/j.ijnaoe.2016.01.004](https://doi.org/10.1016/j.ijnaoe.2016.01.004). The article is open access and reports ice-basin model tests for the icebreaking tanker MT Uikku, including ice properties, measured resistance statistics and comparisons with several established empirical formulas. See also the [ScienceDirect article and tables](https://www.sciencedirect.com/science/article/pii/S2092678215300133).

The ship model scale is 1:31.56. The article reports full-scale ship particulars and resistance values in kN. The CSV preserves two separate reported resistance columns rather than silently reconciling their small differences:
- `measured_mean_resistance_table3_kN`: mean and statistics reported in Table 3.
- `experimental_reference_table8_kN`: experimental comparison values used in Table 8 to calculate formula errors.

These values differ slightly for some cases (for example, Test 103 is 480 kN in Table 3 and 470 kN in Table 8). Treat this as a source-table discrepancy/rounding issue to document, not as a value to silently overwrite.

## Dataset

`mt_uikku_level_ice.csv` contains ten level-ice cases: Tests 103, 104, 205, 206, 301, 302, 303, 401, 402 and 403. The input fields include ice thickness, flexural and compressive strength, ice elastic modulus, and ice drift speed. Where the paper does not report Table 3 force statistics for Tests 401–403 in the retrieved table, those fields are intentionally empty; they are not estimated.

Table 8 also reports published predictions from Lindqvist, Riska, Jeong and Keinonen. These are **baseline model comparisons**, not experimental ground truth. The paper notes that different formulas have materially different errors and that thin-ice/low-speed cases may involve elastic buckling rather than the assumed bending failure.

## Recommended validation protocol for TensorDEM

This is a higher-level target for a ship–ice model, not a direct acceptance test for the current idealized dry moving-wedge smoke benchmark. A fair comparison requires a geometrically and dimensionally consistent ship model, a calibrated ice sheet, and explicit treatment of buoyancy/submergence, hull–ice friction, and the experimental/full-scale conversion.

1. **Material calibration first:** fit bonded-particle parameters against independent sea-ice uniaxial compression and three-point-bending measurements. Do not tune these parameters to the ship-resistance cases and then call those same cases an independent validation.
2. **Freeze the model:** publish the particle spacing, bond law, contact/friction law, time step, domain, hull geometry, and all calibrated parameters before running the ten resistance cases.
3. **Compare like with like:** report predicted mean resistance in kN against `experimental_reference_table8_kN`; compare mean/std/maximum/minimum only when the corresponding experimental statistic is available. State the scale conversion explicitly.
4. **Report metrics:** per-case signed relative error, MAE, RMSE, mean absolute percentage error (MAPE), and bias. Include the published formula columns as context, not as targets to fit.
5. **Demonstrate numerical robustness:** repeat the baseline at two finer time steps and at least two particle resolutions; report convergence separately from experimental error.
6. **Keep channel ice separate:** Tests 304–306 and 404–406 are broken-ice channel cases and must not be mixed into the level-ice dataset.

## Suggested acceptance gates

These are proposed engineering gates, not criteria claimed by the source paper:
- Data-ingestion test: all ten case IDs and units load correctly; missing values remain missing.
- Calibration gate: compressive and flexural strengths are matched on independent specimen cases within a declared tolerance.
- Numerical gate: deterministic reruns on the same CPU configuration and a documented time-step/resolution study.
- Resistance gate: predeclare an error tolerance before examining the ten predictions. A practical first screening could target MAPE <= 25% across the ten cases, while also inspecting each case individually; do not use the aggregate alone to hide poor cases.
- Physical-scope gate: if the current solver does not model ice bending/breakage, buoyancy/submergence, and hull friction, mark this target **not yet comparable** rather than reporting a misleading pass/fail.

## Related bonded-ice material validation references

- Ji, Di and Long, “DEM Simulation of Uniaxial Compressive and Flexural Strength of Sea Ice: Parametric Study,” *Journal of Engineering Mechanics* (2017), [DOI: 10.1061/(ASCE)EM.1943-7889.0000996](https://doi.org/10.1061/(ASCE)EM.1943-7889.0000996). Uses physical sea-ice experimental data to calibrate bonded-particle DEM against uniaxial compression and flexural strength.
- Long, Ji and Wang, “Validation of microparameters in discrete element modeling of sea ice failure process,” *Particulate Science and Technology* (2019), [DOI: 10.1080/02726351.2017.1404515](https://doi.org/10.1080/02726351.2017.1404515). Compares bonded-particle DEM with experimental uniaxial-compression and three-point-bending results.

## Important limitation

The MT Uikku paper provides a published, traceable resistance reference, but this CSV is a transcription of article tables, not a raw downloadable time-history dataset. It supports resistance-summary validation and formula benchmarking. It does not by itself provide the original force time series, particle-scale truth, or a complete geometry file.
