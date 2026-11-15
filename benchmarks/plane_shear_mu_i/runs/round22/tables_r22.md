### Table A - archived reference law (parsed, not transcribed)

| constant | value | source |
|---|---|---|
| `mu_min` | 0.220 | da Cruz 2005 Eq. (14) |
| `b` | 1.000 | da Cruz 2005 Eq. (14) |
| `nu_max` | 0.810 | da Cruz 2005 Eq. (13) |
| `a` | 0.300 | da Cruz 2005 Eq. (13) |
| `I_sat` | 0.200 | da Cruz 2005 Sec. IV B (fit breaks down above) |

`mu(I) = 0.22 + 1.00 I` (I <= 0.2), `phi(I) = 0.81 - 0.30 I`.

### Table B - mu: measured vs each candidate reference

| I_bulk | in validity | mu (full_zz) | DACRUZ_LINEAR err | PRIMARY_REPO err | TEST_GDR_MIDI err |
|---|---|---|---|---|---|
| 0.0059 | yes | 0.2869 | +27.0% | -22.3% | -25.4% |
| 0.0350 | yes | 0.3998 | +56.8% | -2.5% | -2.1% |
| 0.0736 | yes | 0.3394 | +15.6% | -25.3% | -21.8% |
| 0.3454 | no | 0.3552 | -37.2% | -42.4% | -32.2% |

### Table C - pass/fail per candidate and per stress ordinate

| candidate | ordinate | n | in-validity max abs err | within 10% (validity) | passes |
|---|---|---|---|---|---|
| DACRUZ_LINEAR | `mu_full_zz_final` | 4 | +56.8% | 0/3 | no |
| DACRUZ_LINEAR | `mu_full_final` | 4 | +66.0% | 0/3 | no |
| DACRUZ_LINEAR | `mu_wall_final` | 4 | +61.6% | 0/3 | no |
| DACRUZ_LINEAR | `mu_slab_mid_final` | 4 | +58.2% | 0/3 | no |
| DACRUZ_LINEAR | `mu_slab_both_final` | 4 | +55.2% | 2/3 | no |
| PRIMARY_REPO | `mu_full_zz_final` | 4 | +25.3% | 1/3 | no |
| PRIMARY_REPO | `mu_full_final` | 4 | +17.0% | 2/3 | no |
| PRIMARY_REPO | `mu_wall_final` | 4 | +27.2% | 1/3 | no |
| PRIMARY_REPO | `mu_slab_mid_final` | 4 | +28.1% | 1/3 | no |
| PRIMARY_REPO | `mu_slab_both_final` | 4 | +35.7% | 1/3 | no |
| TEST_GDR_MIDI | `mu_full_zz_final` | 4 | +25.4% | 1/3 | no |
| TEST_GDR_MIDI | `mu_full_final` | 4 | +13.0% | 2/3 | no |
| TEST_GDR_MIDI | `mu_wall_final` | 4 | +25.1% | 1/3 | no |
| TEST_GDR_MIDI | `mu_slab_mid_final` | 4 | +31.0% | 1/3 | no |
| TEST_GDR_MIDI | `mu_slab_both_final` | 4 | +38.2% | 1/3 | no |

### Table D - phi vs the archived dilatancy law

| ordinate | n | in-validity max abs err | within 10% |
|---|---|---|---|
| `phi_slab` | 4 | +21.5% | 0/3 |
| `phi_bed_steady` | 4 | +34.0% | 0/3 |

### Table E - monotonicity of the measured mu

| ordinate | sequence (I, mu) | decreasing steps | largest drop |
|---|---|---|---|
| `mu_full_zz_final` | (0.006, 0.287), (0.035, 0.400), (0.074, 0.339), (0.345, 0.355) | 1 | -15.1% |
| `mu_full_final` | (0.006, 0.375), (0.035, 0.409), (0.074, 0.377), (0.345, 0.433) | 1 | -7.8% |
| `mu_wall_final` | (0.006, 0.288), (0.035, 0.412), (0.074, 0.331), (0.345, 0.355) | 1 | -19.7% |
| `mu_slab_mid_final` | (0.006, 0.265), (0.035, 0.403), (0.074, 0.335), (0.345, 0.326) | 2 | -16.9% |
| `mu_slab_both_final` | (0.006, 0.237), (0.035, 0.396), (0.074, 0.321), (0.345, 0.322) | 1 | -19.0% |

### Table F - can a Jop triple ever represent da Cruz?

Least-squares Jop-form fit to the archived linear law over I = 0.001..0.2: mu_s = 0.215, mu_2 = 1.080, I_0 = 0.678 (max abs err 2.14%).

| in-repo triple | mu_s err | mu_2 err | I_0 err | max mu err vs archived law |
|---|---|---|---|---|
| PRIMARY_REPO | +67.4% | -22.2% | -55.8% | +63.6% |
| TEST_GDR_MIDI | +76.3% | -40.6% | -58.9% | +71.9% |
