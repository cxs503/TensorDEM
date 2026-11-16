# R33 — `wall_translate` ablation (generated, do not hand-edit)

artifacts: 5 JSON in `runs/wall_translation_r33/`

## Table A — point-by-point closure diagnostics

| mode | I_target | n_free | p_wall/P | φ | mu_bulk | γ̇_meas/γ̇_nom | slip_bot/h | slip_top/h | r² | source |
|---|---|---|---|---|---|---|---|---|---|---|
| advected | 0.05 | 310 | 0.894 | 0.8302 | 0.282 | 0.193 | +1.69 | -0.01 | 0.598 | `mid_n310_wfixed_I0.05_s0.json` |
| advected | 0.05 | 762 | 0.712 | 0.8405 | 0.315 | 0.457 | +0.14 | -0.00 | 0.545 | `mid_wfixed_I0.05_s0.json` |
| advected | 0.05 | 310 | 0.676 | 0.7997 | 0.312 | 0.631 | +0.17 | -0.22 | 0.979 | `smoke_filled_n310_wfixed_I0.05_s0.json` |
| frozen | 0.05 | 310 | 0.059 | 0.8152 | 0.142 | 0.103 | +4.49 | -4.80 | 0.860 | `smoke_filled_n310_wfrozen_I0.05_s0.json` |
| frozen | 0.05 | 240 | 0.004 | 0.6328 | 0.263 | 0.032 | -14.14 | +15.09 | 0.688 | `smoke_wfrozen_I0.05_s0.json` |


## Table B — judged against the archived 2-D ruler

| mode | I_measured | mu | mu_ref(I_meas) | err % | φ | φ_ref(I_meas) | err % | within 10 % both? |
|---|---|---|---|---|---|---|---|---|
| advected | 0.0120 | 0.282 | 0.232 | +21.3 | 0.8302 | 0.8064 | +3.0 | no |
| advected | 0.0272 | 0.315 | 0.247 | +27.3 | 0.8405 | 0.8018 | +4.8 | no |
| advected | 0.0431 | 0.312 | 0.263 | +18.8 | 0.7997 | 0.7971 | +0.3 | no |
| frozen | 0.0255 | 0.142 | 0.246 | -42.3 | 0.8152 | 0.8023 | +1.6 | no |
| frozen | 0.0293 | 0.263 | 0.249 | +5.5 | 0.6328 | 0.8012 | -21.0 | no |
