### T1. Rayleigh secular equation vs literature

| nu | c_R/c_s (secular equation) | literature | relative error |
| ---: | ---: | ---: | ---: |
| 0 | 0.874032 | 0.874032 | 5.594e-08 |
| 0.25 | 0.919402 | 0.919402 | 3.407e-07 |
| 0.5 | 0.955313 | 0.955313 | 5.223e-07 |

### T2. Engineering fit `0.163 nu + 0.877` vs the exact equation

| nu | c_R/c_s (secular) | c_R/c_s (fit) | relative deviation |
| ---: | ---: | ---: | ---: |
| 0 | 0.874032 | 0.877 | 0.0033957 |
| 0.05 | 0.883695 | 0.88515 | 0.00164677 |
| 0.1 | 0.893106 | 0.8933 | 0.000217214 |
| 0.15 | 0.90222 | 0.90145 | 0.000853178 |
| 0.2 | 0.910996 | 0.9096 | 0.00153227 |
| 0.25 | 0.919402 | 0.91775 | 0.00179648 |
| 0.3 | 0.927413 | 0.9259 | 0.00163111 |
| 0.35 | 0.935013 | 0.93405 | 0.00103007 |
| 0.4 | 0.942195 | 0.9422 | 4.847e-06 |
| 0.45 | 0.94896 | 0.95035 | 0.00146508 |
| 0 | 0.874032 | 0.877 | 0.0033957 |

### T3. Step limits of the engine configurations

| config | N | free DOF | omega_max (rad/s) | dt_crit (s) | dt_R = 2r/c_R (s) | dt_R/dt_crit | recommended dt (s) | recommended/dt_crit | row-sum bound / exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| box4_k2000 | 64 | 96 | 657.476 | 0.00304194 | 0.00460996 | 1.51547 | 7.139e-05 | 0.023467 | 2.00216 |
| box9x5x3_k2000 | 135 | 315 | 680.589 | 0.00293863 | 0.00460996 | 1.56874 | 7.139e-05 | 0.024292 | 1.93416 |
| box6_k20000 | 216 | 432 | 2240.49 | 0.000892661 | 0.0014578 | 1.63309 | 3.569e-05 | 0.0399844 | 1.85795 |

### T4. Wave speeds and isotropic moduli per configuration

| config | rho_lattice (kg/m^3) | G_VRH (Pa) | nu_VRH | c_s (m/s) | c_R (m/s) | c_p (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| box4_k2000 | 480.14 | 66434.8 | 0.266313 | 11.7629 | 10.8461 | 20.8426 |
| box9x5x3_k2000 | 480.14 | 66434.8 | 0.266313 | 11.7629 | 10.8461 | 20.8426 |
| box6_k20000 | 480.14 | 6.643e+05 | 0.266313 | 37.1975 | 34.2983 | 65.9101 |

### T5. Top-mode energy amplification in the engine (config `box4_k2000`)

| dt/dt_crit | d = 2 dt/dt_crit | peak E / E0 (engine) | 1 + d/(2-d) (closed form) | closed-form rel. error | envelope growth | flagged unstable |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.1 | 0.2 | 1.11111 | 1.11111 | 1.656e-07 | 1.00001 | false |
| 0.25 | 0.5 | 1.33333 | 1.33333 | 1.047e-07 | 1.00001 | false |
| 0.5 | 1 | 2 | 2 | 1.253e-06 | 1 | false |
| 0.8 | 1.6 | 5.00001 | 5 | 2.525e-06 | 1.00002 | false |
| 0.9 | 1.8 | 9.99996 | 10 | 3.658e-06 | 1.00004 | false |
| 0.95 | 1.9 | 19.9995 | 20 | 2.641e-05 | 0.999967 | false |
| 1 | 2 | 8.580e+05 | inf | inf | 4.81076 | false |
| 1.02 | 2.04 | 1.419e+12 | inf | inf | 39.0567 | true |
| 1.05 | 2.1 | 1.337e+12 | inf | inf | 31.2913 | true |

### T6. Measured instability threshold of the real engine

| config | predicted dt/dt_crit | measured dt/dt_crit | relative error | measured dt_crit (s) | steps per trial | bisection iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| box4_k2000 | 1 | 1 | 7.629e-07 | 0.00304194 | 600 | 18 |
| box9x5x3_k2000 | 1 | 1 | 1.526e-06 | 0.00293863 | 600 | 18 |

### T7. Accuracy budget implied by the closed form

| tolerated artificial amplification | required dt/dt_crit |
| ---: | ---: |
| 1 % | 0.0049505 |
| 3 % | 0.0145631 |
