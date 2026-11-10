# Predeclared acceptance gates - mission P46W3-N02

Fixed **before** the benchmark numbers were inspected. Changing any threshold
after looking at the results invalidates the comparison; a failure is reported
as an honest FAIL, never by relaxing the gate.

| gate | quantity | threshold | source of the threshold |
|---|---|---|---|
| a1 | Glen exponent fitted from DEM constant-stress creep | \|n_fit - 3\| / 3 <= 0.10 | mission criterion (a) |
| a2 | 95 % CI of the noise Monte-Carlo fit contains 3.0 | required | experimental design (uncertainty must be reported) |
| b1 | max relative error of the DEM strain-time curve vs `eps_0 + eps_1 t^(1/3)` | <= 0.05 | mission criterion (b) |
| b2 | fitted Andrade exponent vs 1/3 | <= 0.10 | mission's literature value, tighter than the 0.05 band of the curve gate |
| b3 | fitted `eps_0` vs `sigma/E` | <= 0.02 | elastic limit identity |
| c1 | Pearson r of log(tau_eff) vs log(fluidity) for the Norton bond sweep | \|r\| >= 0.90 | mission criterion (c) |
| c2 | fitted log-log slope of that sweep vs the analytic -1 | <= 0.05 | Maxwell time-scale identity |
| c3 | Pearson r of log(tau_eff) vs log(eta_alpha), fractional sweep | \|r\| >= 0.90 | mission criterion (c), fractional branch |
| c4 | max relative deviation of the alpha = 1 relaxation from `exp(-t/tau)` | <= 0.01 | exact limit (Newtonian dashpot) |
| c5 | max relative deviation of the alpha = 1/2 relaxation from `exp(x^2) erfc(x)` | <= 0.03 | exact limit (half-order element) |
| d1 | strain with fluidity A -> 0 vs `sigma/E` | <= 0.01 | mission criterion (d), Norton branch |
| d2 | strain with eta_alpha -> inf vs `sigma/E` | <= 0.01 | mission criterion (d), fractional branch |
| d3 | creep strain at `eta_alpha = 1e4 * nominal` vs nominal creep ratio | <= 0.01 | monotone collapse of creep with the creep parameter |
| l1 | `A(-10 C)` model default inside the literature band | inside | literature band, ±1 decade |
| l2 | `eta_eff` at 1 MPa inside 1e11..1e13 Pa s | inside | textbook band |
| l3 | activation energy vs the two-branch Hobbs/Paterson values | <= 0.15 rel | literature-summary band |
| l4 | fitted Andrade exponent inside 0.2..0.4 | inside | literature band |
| l5 | Cole 1998 / Schulson 2001 curve overlay | WITHHELD (not PASS) | no digitized data in tree - must not be faked |

`l5` is deliberately not a pass/fail gate: it is recorded as WITHHELD with the
reason string, and the mission report states it as an open item.