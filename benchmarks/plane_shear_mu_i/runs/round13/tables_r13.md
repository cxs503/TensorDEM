# Round 13 tables - full-range (1e-3..0.3) rough vs smooth wall

3 seeds per point (5279, 6113, 7919), single device `ServoShearCase`, only `wall_mode` changes.  Reference: da Cruz 2005 / GDR MiDi 2004 as quoted in-repo (mu_s, mu_2, I0) = (0.36, 0.84, 0.30); phi magnitudes PROVENANCE-UNVERIFIED (see `reference_law.py`).

## Table 1 - mu(I), platen traction (historical estimator)

| I | mu_ref | rough mean +- sd (err) | smooth mean +- sd (err) |
|---|---|---|---|
| 0.01 | 0.3755 | n/a | 0.3260 +- 0.0782 (-13.2%) |
| 0.03 | 0.4036 | 0.1548 +- 0.0000 (-61.7%) | 0.3283 +- 0.0000 (-18.7%) |
| 0.1 | 0.4800 | n/a | 0.3362 +- 0.0000 (-29.9%) |
| 0.3 | 0.6000 | n/a | 0.3497 +- 0.0000 (-41.7%) |

## Table 1b - mu(I), full-cell virial (p)

| I | mu_ref | rough | smooth |
|---|---|---|---|
| 0.01 | 0.3755 | n/a | 0.4166 +- 0.1180 (+11.0%) |
| 0.03 | 0.4036 | 0.1345 +- 0.0000 (-66.7%) | 0.3498 +- 0.0000 (-13.3%) |
| 0.1 | 0.4800 | n/a | 0.3704 +- 0.0000 (-22.8%) |
| 0.3 | 0.6000 | n/a | 0.4379 +- 0.0000 (-27.0%) |

## Table 1b - mu(I), full-cell virial (sigma_zz)

| I | mu_ref | rough | smooth |
|---|---|---|---|
| 0.01 | 0.3755 | n/a | 0.3224 +- 0.0792 (-14.1%) |
| 0.03 | 0.4036 | 0.0657 +- 0.0000 (-83.7%) | 0.3245 +- 0.0000 (-19.6%) |
| 0.1 | 0.4800 | n/a | 0.3367 +- 0.0000 (-29.9%) |
| 0.3 | 0.6000 | n/a | 0.3515 +- 0.0000 (-41.4%) |

## Table 1b - mu(I), slab virial (both)

| I | mu_ref | rough | smooth |
|---|---|---|---|
| 0.01 | 0.3755 | n/a | 0.3104 +- 0.0317 (-17.3%) |
| 0.03 | 0.4036 | 0.2829 +- 0.0000 (-29.9%) | 0.3345 +- 0.0000 (-17.1%) |
| 0.1 | 0.4800 | n/a | 0.3214 +- 0.0000 (-33.0%) |
| 0.3 | 0.6000 | n/a | 0.3299 +- 0.0000 (-45.0%) |

## Table 1b - mu(I), slab virial (midpoint)

| I | mu_ref | rough | smooth |
|---|---|---|---|
| 0.01 | 0.3755 | n/a | 0.2897 +- 0.0167 (-22.8%) |
| 0.03 | 0.4036 | 0.3253 +- 0.0000 (-19.4%) | 0.3394 +- 0.0000 (-15.9%) |
| 0.1 | 0.4800 | n/a | 0.3356 +- 0.0000 (-30.1%) |
| 0.3 | 0.6000 | n/a | 0.3322 +- 0.0000 (-44.6%) |

## Table 2 - servo and kinematics

| I | mode | seeds | servo valid | P err (final) | rate meas/imposed | profile R^2 | I meas |
|---|---|---|---|---|---|---|---|
| 0.01 | smooth | 2 | 1/2 | -2.8% | 0.771 | 0.785 | 0.0081 |
| 0.03 | rough | 1 | 0/1 | -6.3% | 0.483 | 0.734 | 0.0152 |
| 0.03 | smooth | 1 | 0/1 | -8.7% | 1.026 | 0.890 | 0.0328 |
| 0.1 | smooth | 1 | 1/1 | -0.6% | 0.632 | 0.891 | 0.0677 |
| 0.3 | smooth | 1 | 1/1 | -0.8% | 0.916 | 0.969 | 0.2798 |

## Table 3 - phi(I), full-gap bed definition

| I | phi_ref (PROVENANCE-UNVERIFIED) | rough | smooth |
|---|---|---|---|
| 0.01 | 0.6381 | n/a | 0.5326 +- 0.0016 (-16.5%) |
| 0.03 | 0.6345 | 0.4856 +- 0.0000 (-23.5%) | 0.5394 +- 0.0000 (-15.0%) |
| 0.1 | 0.6250 | n/a | 0.5496 +- 0.0000 (-12.1%) |
| 0.3 | 0.6100 | n/a | 0.5727 +- 0.0000 (-6.1%) |

## Table 4 - outcome vs the ledger acceptance (<10 %)

| estimator | mode | points inside 10 % | worst |err| | mean err |
|---|---|---|---|---|
| platen traction | rough | 0/1 | 61.7% | -61.7% |
| platen traction | smooth | 0/4 | 41.7% | -25.9% |
| full-cell virial (p) | rough | 0/1 | 66.7% | -66.7% |
| full-cell virial (p) | smooth | 0/4 | 27.0% | -13.1% |
| full-cell virial (sigma_zz) | rough | 0/1 | 83.7% | -83.7% |
| full-cell virial (sigma_zz) | smooth | 0/4 | 41.4% | -26.3% |
| slab virial (both) | rough | 0/1 | 29.9% | -29.9% |
| slab virial (both) | smooth | 0/4 | 45.0% | -28.1% |
| slab virial (midpoint) | rough | 0/1 | 19.4% | -19.4% |
| slab virial (midpoint) | smooth | 0/4 | 44.6% | -28.4% |

## Table 5 - seed-to-seed spread (sample identification)

| I | mode | mu_wall seeds | spread (max-min)/ref |
|---|---|---|---|
| 0.01 | smooth | 0.2707, 0.3813 | 29.5% |

## Failures (honest FAIL records)

* rough I=0.1 seed=5279: MuIRheologyError: servo did not reach the target normal stress
* rough I=0.3 seed=5279: MuIRheologyError: servo did not reach the target normal stress
