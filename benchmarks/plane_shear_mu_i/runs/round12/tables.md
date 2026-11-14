# Round 12 tables - rough vs smooth wall, mu(I) and phi(I)

3 seeds per point (5279, 6113, 7919), single device `ServoShearCase`, only `wall_mode` changes.

## Table 1 - mu(I), platen traction (historical estimator)

| I | mu_ref | smooth mean +- sd (err) |
|---|---|---|
| 0.01 | 0.3755 | 0.3813 +- 0.0000 (+1.5%) |
| 0.03 | 0.4036 | 0.3283 +- 0.0000 (-18.7%) |
| 0.1 | 0.4800 | 0.3362 +- 0.0000 (-29.9%) |
| 0.3 | 0.6000 | 0.3497 +- 0.0000 (-41.7%) |

## Table 1b - mu(I), full-cell virial (p)

| I | mu_ref | smooth |
|---|---|---|
| 0.01 | 0.3755 | 0.5000 +- 0.0000 (+33.2%) |
| 0.03 | 0.4036 | 0.3498 +- 0.0000 (-13.3%) |
| 0.1 | 0.4800 | 0.3704 +- 0.0000 (-22.8%) |
| 0.3 | 0.6000 | 0.4379 +- 0.0000 (-27.0%) |

## Table 1b - mu(I), full-cell virial (sigma_zz)

| I | mu_ref | smooth |
|---|---|---|
| 0.01 | 0.3755 | 0.3784 +- 0.0000 (+0.8%) |
| 0.03 | 0.4036 | 0.3245 +- 0.0000 (-19.6%) |
| 0.1 | 0.4800 | 0.3367 +- 0.0000 (-29.9%) |
| 0.3 | 0.6000 | 0.3515 +- 0.0000 (-41.4%) |

## Table 1b - mu(I), slab virial (both)

| I | mu_ref | smooth |
|---|---|---|
| 0.01 | 0.3755 | 0.3328 +- 0.0000 (-11.4%) |
| 0.03 | 0.4036 | 0.3345 +- 0.0000 (-17.1%) |
| 0.1 | 0.4800 | 0.3214 +- 0.0000 (-33.0%) |
| 0.3 | 0.6000 | 0.3299 +- 0.0000 (-45.0%) |

## Table 1b - mu(I), slab virial (midpoint)

| I | mu_ref | smooth |
|---|---|---|
| 0.01 | 0.3755 | 0.3015 +- 0.0000 (-19.7%) |
| 0.03 | 0.4036 | 0.3394 +- 0.0000 (-15.9%) |
| 0.1 | 0.4800 | 0.3356 +- 0.0000 (-30.1%) |
| 0.3 | 0.6000 | 0.3322 +- 0.0000 (-44.6%) |

## Table 2 - servo and kinematics

| I | mode | seeds | servo valid | P err (final) | rate meas/imposed | profile R^2 | I meas |
|---|---|---|---|---|---|---|---|
| 0.01 | smooth | 1 | 1/1 | -0.4% | 0.889 | 0.837 | 0.0093 |
| 0.03 | smooth | 1 | 0/1 | -8.7% | 1.026 | 0.890 | 0.0328 |
| 0.1 | smooth | 1 | 1/1 | -0.6% | 0.632 | 0.891 | 0.0677 |
| 0.3 | smooth | 1 | 1/1 | -0.8% | 0.916 | 0.969 | 0.2798 |

## Table 3 - phi(I), full-gap bed definition

| I | phi_ref (PROVENANCE-UNVERIFIED) | smooth |
|---|---|---|
| 0.01 | 0.6381 | 0.5314 +- 0.0000 (-16.7%) |
| 0.03 | 0.6345 | 0.5394 +- 0.0000 (-15.0%) |
| 0.1 | 0.6250 | 0.5496 +- 0.0000 (-12.1%) |
| 0.3 | 0.6100 | 0.5727 +- 0.0000 (-6.1%) |

## Table 4 - outcome vs the ledger acceptance (<10 %)

| estimator | mode | points inside 10 % | worst |err| | mean err |
|---|---|---|---|---|
| platen traction | smooth | 1/4 | 41.7% | -22.2% |
| full-cell virial (p) | smooth | 0/4 | 33.2% | -7.5% |
| full-cell virial (sigma_zz) | smooth | 1/4 | 41.4% | -22.5% |
| slab virial (both) | smooth | 0/4 | 45.0% | -26.6% |
| slab virial (midpoint) | smooth | 0/4 | 44.6% | -27.6% |

## Failures (honest FAIL records)

* rough I=0.1 seed=5279: MuIRheologyError: servo did not reach the target normal stress
* rough I=0.3 seed=5279: MuIRheologyError: servo did not reach the target normal stress
