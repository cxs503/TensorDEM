# Round 16 -- strain-converged mu(I)/phi(I) vs the JOP law

Source: cluster jobs cj2950-cj2955 (R15 submission), smooth walls,
seed 5279, N = 294, sheared to strain >= 2 (gamma ~ 99 at I = 0.3).
Errors are vs `reference_law.py` (provenance status in result.json).

| I_target | I_bulk | dI/I [%] | gamma | mu(20%) | mu(25%) | mu_ref(t) | mu_ref(b) | err mu 20%@t [%] | err mu 25%@t [%] | err mu 25%@b [%] | phi(25%) | phi_ref | err phi [%] | mu std(25%) [%] | steady phi/mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.00133 | +33 | 2.00 | 0.2487 | 0.2403 | 0.3616 | 0.3621 | -31.2 | -33.6 | -33.6 | 0.5607 | 0.6398 | -12.4 | 27.8 | Y/N |
| 0.003 | 0.00239 | -20 | 2.01 | 0.2894 | 0.3153 | 0.3648 | 0.3638 | -20.7 | -13.6 | -13.3 | 0.5563 | 0.6394 | -13.0 | 32.2 | N/N |
| 0.01 | 0.00496 | -50 | 2.05 | 0.3059 | 0.3068 | 0.3755 | 0.3678 | -18.5 | -18.3 | -16.6 | 0.5504 | 0.6381 | -13.7 | 5.3 | N/N |
| 0.03 | 0.01490 | -50 | 2.86 | 0.3033 | 0.2925 | 0.4036 | 0.3827 | -24.9 | -27.5 | -23.6 | 0.5523 | 0.6345 | -13.0 | 19.4 | N/N |
| 0.1 | 0.07809 | -22 | 6.26 | 0.3592 | 0.3785 | 0.4800 | 0.4591 | -25.2 | -21.2 | -17.6 | 0.5529 | 0.6250 | -11.5 | 20.7 | N/N |

## Verdict

- mu(I) within 10 % (20 % window, I_target): **FAIL**
- mu(I) within 10 % (25 % window, I_target): **FAIL**
- phi(I) within 10 % (25 % window): **FAIL**
- mu noise (std of final 25 % window) exceeds the 10 % band at every point: **no**
- **Overall: FAIL**
