# TensorDEM verification campaigns

This document describes the tensordem.verification workflow for repeatable numerical sensitivity runs. It is deliberately separate from the solver core: the workflow measures what the current model does, but it does not alter the force law, tune parameters automatically, or certify the model against physical experiments.

## Scope and interpretation

The campaign framework currently supports:

- SI-validated case specifications for the existing 2-D bonded-disk model.
- A resolution series with approximately fixed physical width and target height.
- Time histories of tool reaction, boundary reaction, kinetic energy, spring energy, contact energy, and broken-bond count.
- Shared-time force-history comparison with linear interpolation across different sample times.
- Peak force, signed and absolute impulse, force RMS difference, mean bias, peak-force difference, and signal correlation.
- Deterministic history digests and strict JSON reports, plus spreadsheet-friendly CSV summaries.
- A per-case step cap, strict finite-value checks, report schema validation, and atomic output replacement.

The existing model uses disk mass, central axial springs, bond deletion, repulsive contact and a semi-implicit Euler integrator. It does **not** have particle rotations, explicit bending moments, a calibrated fracture-energy law, or an experimental validation database. Therefore, a small error between two resolutions is evidence of numerical sensitivity only; it is not proof that the force is physically correct.

## Run a resolution campaign

Install the project and test dependencies, then run:

    python -m pip install -e ".[test]"
    python scripts/dem_verification_campaign.py \
      --nx-levels 5,7,9 \
      --width 0.3 \
      --height 0.2 \
      --thickness 0.2 \
      --duration 0.002 \
      --speed 0.2 \
      --save-every 1 \
      --output results/verification

Outputs:

- results/verification/campaign.json: complete input specifications, realized geometry, solver metadata, histories, summaries, comparisons, and history SHA-256 digests.
- results/verification/summary.csv: one row per case with input, geometry, summary, and comparison fields.

Use a shorter duration for a smoke test. The default step cap is two million integration steps per case. If a finer resolution causes a smaller stable time step, the run may fail before integration with an explanatory error; reduce the duration or explicitly choose a smaller set of levels. A requested --dt is checked by the solver against its conservative bound.

## Python API

    from tensordem.verification import (
        CampaignSpec,
        run_case,
        run_resolution_campaign,
        read_report,
        write_report,
        write_summary_csv,
    )

    specs = [
        CampaignSpec(nx=5, duration_s=1e-4),
        CampaignSpec(nx=7, duration_s=1e-4),
        CampaignSpec(nx=9, duration_s=1e-4),
    ]
    report = run_resolution_campaign(specs)
    write_report(report, "results/campaign.json")
    write_summary_csv(report, "results/summary.csv")
    checked = read_report("results/campaign.json")

CampaignSpec records both target geometry and the realized discretized dimensions. Particle radius is derived from target width and nx; ny is rounded from the target height, so the realized height can vary slightly by resolution. Always compare the geometry values rather than assuming that both dimensions match the target exactly.

run_case records the initial state and then samples every save_every steps, always including the final state. For a case requiring N integration steps, the output contains approximately ceil(N / save_every) + 1 samples. The simulation duration can exceed the requested horizon by less than one integration step.

compare_histories compares a candidate with a reference over their common time interval. It forms the union of both sample-time sets within that interval and linearly interpolates each force series onto the shared set. This avoids a nearest-neighbour comparison, but the RMS is a sample-based statistic rather than a continuous-time integral. If the sampling schedule is highly irregular, inspect the raw histories as well.

## Report integrity and reproducibility

Each case history is encoded as strict JSON numeric values and hashed with SHA-256. Dictionary key order does not change the digest; changed values or timestamps do. The top-level campaign digest is computed from the ordered list of case digests. Reports include a schema identifier and are rejected if the schema, case references, comparison references, history digest, or strict-JSON constraints are inconsistent.

Output files are written through a temporary file followed by an atomic replace, so a failed write does not leave a partially written target file. This does not provide a transactional guarantee across the JSON and CSV pair: if the second write fails, the first report can still exist.

A matching digest demonstrates that the sampled numeric record is unchanged. It does not guarantee bitwise reproducibility across different PyTorch versions, hardware, devices, or floating-point execution environments.

## Recommended engineering acceptance sequence

1. **Smoke test:** run a very short case and verify finite state values, nonzero time advancement, and a readable report.
2. **Repeatability:** run the exact same case twice in the same environment and compare history digests.
3. **Time-step sensitivity:** for selected small cases, repeat with a smaller valid time step while keeping geometry and other parameters fixed. Compare the force histories on the overlap interval.
4. **Resolution sensitivity:** compare at least three resolutions, and report realized width/height, particle count, timestep, duration, broken bonds, peak force and impulse.
5. **Energy review:** inspect kinetic, bond and contact energy histories alongside external/tool work and damping/fracture accounting available in the solver. Do not assume a total-energy balance closes until every term and sign convention has been independently checked.
6. **Material calibration:** compare the material patch tests and fitted parameters against independent laboratory data. Do not infer Young's modulus or fracture toughness from a single peak-force metric.
7. **Physical validation:** compare pre-registered predictions against experiments not used for calibration, including force-time curves, failure patterns, ice thickness, indentation speed and boundary conditions.

The campaign currently automates steps 1–4 and supplies evidence useful for step 5. It does not automatically pass the remaining steps or label a model as validated.

## Known limitations

- Only the existing two-dimensional central-force IceDEM backend is run.
- The benchmark uses a circular indenter. The current CLI does not accept a hull polyline; ship-hull studies should use the hull-capable solver interface directly and extend the campaign adapter with explicit hull provenance.
- Contact/fracture energy values are diagnostic snapshots. A physically complete energy balance requires checking the solver's discrete work and dissipation conventions, especially at bond deletion and under prescribed tool motion.
- Resolution changes the particle radius and row count. The current scalar spring stiffness is not automatically recalibrated to preserve a continuum modulus across levels.
- The reference case defaults to the largest nx, which is a comparison convention, not proof that the finest case is converged.
- No acceptance thresholds are imposed by default. Thresholds must be selected before inspecting results and justified by the intended engineering use.

## Test

    python -m unittest discover -s tests

The tests check data validation, interpolation, quadrature, force metrics, parameter-grid construction, repeatable short runs, report round-tripping, digest tampering detection and invalid reference handling. Passing these tests verifies the analysis and reporting code paths; it is not a physical validation result.
