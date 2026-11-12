# m16316 board audit — m16243 CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT

**Verdict: L3** (verdict_by=board-audit). Audited branch `agent/m16243` @ 5a4f14140f0a, main @ 181b91cd2b4a; merged into main: **False**.

## 1. Referee — the only judge

`tensor_benchmark_cases WHERE anchor_slug='CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT'` → 4 rows:

- `R1_bagnold_exponent_inertial_limit|n = d ln tau / d ln gdot in the inertial limit (tau ~ rho d^2 gdot^2)|15 % rel.|pending`
- `R2_regime_boundary_kappa_c|kappa_c for the elastic-inertial -> inertial transition|factor 3 (order of magnitude)|pending`
- `R3_elastic_quasistatic_exponent|n for tau* ~ kappa (elastic-quasi-static limit)|absolute 0.3|pending`
- `R4_bagnold_coefficient_plateau|tau* plateau coefficient (WITHHELD: JFM 2002 pp.261-291 CLOSED, no OA copy)|WITHHELD|pending`

## 2. Deliverables (bytes)

| file | bytes | sha256[:16] | commits touching |
|---|---|---|---|
| `src/tensordem/elastic_limit_shear.py` | 35555 | `c835017b69aa5c84` | 3 |
| `src/tensordem/cli.py` | 13010 | `f83c4f2b280f3bb1` | 17 |
| `benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py` | 11044 | `459cdf42d5bc7551` | 3 |
| `tests/test_elastic_limit_shear.py` | 7457 | `26d359dc6cd7c6de` | 3 |
| `docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT.md` | 18465 | `ff013c122447c07c` | 3 |
| `artifacts/case_dem_shear_campbell_elastic_limit/scaling.json` | 5942 | `320dc6060872bea4` | 2 |
| `artifacts/case_dem_shear_campbell_elastic_limit/dt_convergence.json` | 638 | `a151365eb80abbb4` | 1 |
| `artifacts/case_dem_shear_campbell_elastic_limit/fig_tau_star_kappa.png` | 44768 | `2f14ec8ad4d010b9` | 2 |
| `artifacts/case_dem_shear_campbell_elastic_limit/logs/sweep_all.log` | 0 | `e3b0c44298fc1c14` | 1 |

deliverables are committed on the mission branch, but that branch is not an ancestor of main: trunk still ships the round-1 kernel and has no CLI wiring for this case. Trunk (`main`) CLI wiring lines: 0.

## 3. Arithmetic recomputed from each artifact's own case block

Worst relative difference (recomputed vs filed kappa/tau*/mu/I): **0** (all four groups, all 8 points) — filed dimensionless groups are arithmetic-consistent with the artifacts' own case blocks.

## 4. The gate that decides which points count

- artifacts record `energy.gate = [0.1]`; their own `gates.all`: `{'kn1e+00_g1_n400': 0, 'kn1e+01_g1_n400': 0, 'kn1e+02_g1_n400': 0, 'kn1e+03_g1_n400': 0, 'kn1e+04_g1_n400': 0}`
- under that gate: **0 gated points** → refused: fewer than 2 points pass the gate
- under the committed code's gate (0.30): **3 gated points**, β=0.869594, n=0.260812 (matches `scaling.json`: True); local n=0.2761, n=0.2456

every point in the shipped artifacts carries gates.energy=0 and gates.all=0 under the energy gate recorded inside the artifact (0.1); the headline scaling exists only after re-gating at 0.30, and the artifacts were never regenerated to reflect it.

## 5. Referee legs

| leg | reference | tolerance | got | within? |
|---|---|---|---|---|
| R1 | n=2 (Bagnold) | 15 % rel | no gated point; ungated kn=1e0 sweep → n=0.0979, pressure exponent 0.133, overlap up to 0.59 d | **NO** |
| R2 | kappa_c=1e4 | factor 3 | no regime break in 0.5787-5787; the one point inside the factor-3 band (κ=5787) still follows the same law | **NO** |
| R3 | n=0 (elastic limit) | absolute 0.30 | n=0.2608 (β=0.8696), |dev|=0.2608; τ* still grows ×54.9; I=[0.5561, 0.2106, 0.0763] | **YES** (weak) |
| R4 | plateau coefficient | WITHHELD | nothing asserted anywhere | n/a |

Passed ['R3'] / 4 legs; failed ['R1', 'R2']; withheld ['R4'].

## 6. Energy audit

- `kn1e+00_g1_n400`: window residual 0.0474, global 0.0343, gate recorded in the artifact 0.1 (0.5× over)
- `kn1e+01_g1_n400`: window residual 0.1352, global 0.0721, gate recorded in the artifact 0.1 (1.4× over)
- `kn1e+02_g1_n400`: window residual 0.1924, global 0.1534, gate recorded in the artifact 0.1 (1.9× over)
- `kn1e+03_g1_n400`: window residual 0.2547, global 0.2188, gate recorded in the artifact 0.1 (2.5× over)
- `kn1e+04_g1_n400`: window residual 0.2386, global 0.2092, gate recorded in the artifact 0.1 (2.4× over)

The global residual reproduces arithmetically from the artifact's own work/dissipation/stored numbers: **True**.

Probe (`runs/audit_m16316/probe_kn1e3.log`, instrumented copy, same seed): hypothesis = the residual is the tangential-spring energy retained by pairs that have separated → rejected as the (sole) mechanism: retained tangential energy of non-contacting pairs is 0.00% of the window work while the residual stays at 25.5%.

the audit residual is 1.9-2.5x the energy gate recorded in the artifacts; the 'model-level, dt-independent' explanation is plausible but its mechanism is unidentified, so the audit cannot certify the energetics of the gated points.

## 7. Reproduction by the auditor

- filed kn=1e3 point: τ*=11.187963375044525; auditor re-run 11.187963375044525 (rel diff 0.0e+00, bit identical True); params {'kn': 1000.0, 'shear_rate': 1.0, 'n_particles': 400, 'strain_total': 1.8, 'strain_average_from': 0.8, 'dt_over_contact_time': 0.0025, 'seed': 20261112}
- point reproduces bit-exactly from the branch commit at the filed parameters -- the artifact is real output of this code
- documented repro command `PYTHONPATH=src python -m tensordem.cli elastic-limit-shear --kn 1e4 --shear-rate 1.0`: exit 0, stdout 0 B, stderr 0 B, `__main__` guards in cli.py: 0 → the documented command is a silent no-op: cli.py has no `if __name__ == '__main__'` guard, so `python -m tensordem.cli` imports the module, never calls main(), prints nothing and exits 0
- working invocation `PYTHONPATH=src python -m tensordem elastic-limit-shear --kn/--n/--strain ...`: κ=57.87037037037038, τ*=1.5620518782513184, μ=0.46235181203320175, gates_all=True at N=200 (artifact runs/audit_m16316/cli_point_kn1e2.json)

## 8. Tests

`PYTHONPATH=src pytest -q tests/test_elastic_limit_shear.py` → **15 passed, 1 warning in 8.72s** (exit 0); they assert min-image vs brute force for s in {0,.1,.3,.5,-.4}; single-contact stress sign + normalisation; packing solid fraction + overlap-free start; synthetic n=0 / n=2 branch recovery by the fitter; refusal when fewer than 3 gated points.

## 9. Report check

Figure embedded as png_base64: **False** (the figures payload carries only name+caption; the PNG itself lives in the repo: `artifacts/case_dem_shear_campbell_elastic_limit/fig_tau_star_kappa.png`, {'size': [1540, 616], 'mean': 249.5755702900995, 'std': 27.522077693062336, 'nonwhite_fraction': 0.10434622195985832}).

Six required sections present: **6/6** ({'①': True, '②': True, '③': True, '④': True, '⑤': True, '⑥': True}); figures payload 150 B; explicit “<3%” answer: False.

## 10. Disposition

**L3 — capability real but the registered benchmark is only partially met.** L1 met: 15 tests assert invariants (min-image vs brute force, single-contact stress sign+normalisation, packing fraction, synthetic n=0/n=2 recovery, gate-refusal path) -- replayed green by the auditor. L2 met: a self-contained 825-line SLLOD/Lees-Edwards kernel with a real input->compute->output path; one filed point reproduces bit-exactly (tau*=11.187963375044525). L3 met at branch level: `elastic-limit-shear` is wired into src/tensordem/cli.py (+37 lines) and runs end-to-end via `python -m tensordem` (kappa=57.87, tau*=1.562, all gates pass at N=200) -- but the branch is NOT merged into main, so trunk still ships the round-1 kernel with no wiring. L4 NOT met: of the 4 registered referee legs only ['R3'] passes (1/4); R1 and R2 fail, R4 is WITHHELD. That single pass is a cross-over exponent in the dense-inertial regime (I=0.08-0.56, tau* still growing x54.9), it exists only after an energy gate loosened 3x after the runs (the shipped artifacts carry gates.all=0 under their own gate 0.1 and were never regenerated), the energy audit closes only to 19-25% of the shear work with an unidentified mechanism, the documented CLI repro command is a silent no-op, and the finite-size study is not run. Real kernel + tests + CLI wiring + a partially validated case = L3.

To reach L4:

1. run kn=1e5/1e6 rungs + rate sweeps at kn=1e6 (sharded) so R1/R2 have gated points in the inertial band
2. identify or fix the mechanism of the 19-25% energy-audit residual (the retained-tangential-history hypothesis was falsified by the probe)
3. regenerate the artifacts under the final gate so point JSONs and scaling.json stop contradicting each other
4. merge agent/m16243 into main, otherwise the capability is not in trunk
5. N=800 finite-size + an uncertainty on n (the 3-point fit carries no error bar)
6. fix the documented repro command (python -m tensordem, not python -m tensordem.cli)

Figure: `docs/board_audit/m16316/fig_m16316_audit.png` (left: referee R3 recompute + reference slopes; right: audit residual vs the two competing gates).

Overstated claims found: (i) “elastic-limit leg reproduced” while τ* still grows ×54.9 and I=0.08–0.56 (dense-inertial, not quasi-static); (ii) the documented CLI repro line, which exits 0 without running anything; (iii) the case ships point JSONs whose own gates say FAILED while scaling.json counts those same points as gated.

