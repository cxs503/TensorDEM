# CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT — Benchmark vs Campbell (2002)

> Reference: Campbell, C. S. (2002). *Granular shear flows at the elastic limit*.
> J. Fluid Mech. **465**, 261-291. DOI: [10.1017/S002211200200109X](https://doi.org/10.1017/S002211200200109X)

## Summary

* **κ span**: 361.7 – 578703.7 (3.2 decades)
* **Gated points**: 11 / 12
* **Global fit**: β = 0.9290, n = 0.1419, R² = 0.9984
* **Elastic limit**: β_high = 0.9800, n_high = 0.0399
* **Elastic coefficient**: C = 0.013570 (WITHHELD vs Campbell — paywalled)
* **Overall**: **PASS**

## Benchmark Legs

| Leg | Quantity | Ref | Got | Dev | Tol | Verdict |
|---|---|---|---|---|---|---|
| L1_elastic_quasi_static_exponent | d ln tau / d ln gdot in the elastic-quasi-static l | 0 | 0.0399 | 0.0399 | 0.3 | **PASS** |
| L2_elastic_inertial_branch | beta = d ln tau* / d ln kappa over the gated ladde | 0 < beta < 1 | 0.9290 |  | (0.0, 1.0) | **PASS** |
| L3_direct_rate_exponent_kn1e4 | n = d ln tau / d ln gdot at kn = 1e4 (constant-vol | 0 | 0.2159 | 0.2159 | 0.3 | **PASS** |
| L4_kappa_collapse_identity | tau* is identical for iso-kappa (k_n, gdot) pairs | exact identity | 0.0000 |  | 1e-06 | **PASS** |
| L5_monotone_approach_to_elastic_limit | beta increases monotonically with kappa (approach  | monotone increasing | beta_low_half < beta_high_half |  | — | **PASS** |
| L6_coefficient_elastic_branch | C in tau* = C * kappa (elastic-quasi-static platea | None | 0.0136 |  | — | **WITHHELD** |
| L7_bagnold_inertial_exponent | n = d ln tau / d ln gdot in the inertial limit (ka | 2 | None |  | 0.3 | **NOT_SUPPORTED** |

## Declared Deviations

* **geometry**: 2-D disks vs Campbell's 3-D spheres
* **coefficient**: WITHHELD (Campbell's numerical tables are paywalled)
* **inertial_limit**: NOT_SUPPORTED (linear-spring cell cannot reach kappa << 1)

## Evidence Detail

### L1_elastic_quasi_static_exponent
* **Source**: Campbell (2002) abstract: stresses scale purely elastically => n = 0
* **Evidence**: R3 kappa-collapse, beta_high_half = 0.9800, n_high_half = 0.0399, sigma_n = 0.1039

### L2_elastic_inertial_branch
* **Source**: Campbell (2002) abstract: stresses scale with both elastic and inertial properties
* **Evidence**: R3 kappa-collapse, beta_global = 0.9290, R² = 0.9984, span 361.7–578703.7

### L3_direct_rate_exponent_kn1e4
* **Source**: Same anchor as L1: elastic-quasi-static => n = 0
* **Evidence**: R3 rate sweep at kn=1e4, n = 0.2159, R² = 0.9632

### L4_kappa_collapse_identity
* **Source**: SLLOD + linear-spring + Coulomb is exactly invariant under (kn, gdot) -> (lambda^2 kn, lambda gdot)
* **Evidence**: R3 kappa-collapse, worst spread = 0.0e+00, 4 groups

### L5_monotone_approach_to_elastic_limit
* **Source**: Campbell (2002): elastic-quasi-static is the high-kappa limit
* **Evidence**: R3: beta_low_half = 0.8932, beta_high_half = 0.9800

### L6_coefficient_elastic_branch
* **Source**: Campbell (2002) tables: full text CLOSED (paywalled)
* **Evidence**: Our value: C = 0.013570 at kappa = 578703.7

### L7_bagnold_inertial_exponent
* **Source**: Linear-spring cell cannot reach kappa << 1 without overlap of order d
* **Evidence**: R4 inertial-branch: all gated points at kappa > 9; soft-particle probe at kn=1 fails overlap gate
