# P82W2-N09 — multi-rate contact integration (MTS) and the time-step gate

**Deliverable** `TensorDEM/src/tensordem/mts.py` (contract + integrator),
`benchmarks/mts_contact_integration.py` (campaign),
`tests/test_mts_contact_integration.py` (17 tests).
**Evidence** `runs/P82W2_N09/verification.json`, figures in
`docs/figures/P82W2_N09/`.

## 1. Case, physics and parameters

The node asks for a *machine-decidable* time-step contract for DEM: under a
stiffness hierarchy the stiff contacts and the friction/rolling arms must remain
both **computable** and **conserving**, and the friction arm's cost per step must
come back down.

| symbol | meaning | value |
|---|---|---|
| $t_c$ | Hertz contact duration, $C(5m/4k)^{2/5}v^{-1/5}$, $C=2.9427$ | from `tensordem.hertz_impact_time` |
| $dt_c$ | Cundall bound $2\\sqrt{m/k}$ (quasi-static scale) | from `tensordem.contact_timestep_ledger` |
| margin | largest admissible single-rate $dt/dt_c$ | 0.05 (20 samples/contact) |
| tight ratio | $dt/dt_c$ at which the gate refuses the step | **0.5** |
| impactor | mass / Hertz stiffness / speed | $10^{-3}$ kg / $10^{4}$ N/m / 0.05 m/s |
| packing | bidisperse 2-D disks, $N=256$, $r_{mean}=10$ mm, $\\mu=0.5$, $\\mu_r=0.05$ | $k_n=3\\times10^{4}$ N/m, $\\zeta=0.4$ |
| hierarchy | stiff skeleton | 10 % of disks carry contacts 100× stiffer ($t_c$ ratio 10:1) |

The contract (layer 1) is the class `StabilityGate`: `ratio = dt/dt_c`,
`substeps = ceil(ratio/margin)`, and `status in {ok, subcycle, fail}` with
`fail` at `ratio >= 0.5`.  The integrator (layer 2) is `MTSDisks`: the **fast**
arm resolves the normal springs of the active contact set on
`h = Δt/n_sub`, the **slow** arm advances the tangential/rolling internal
variables once per `Δt` with the friction-cone return map, and the friction-cone
projection is re-applied inside every sub-step.  Forces are still exchanged as
equal-and-opposite pairs, and the position update keeps the leapfrog
drift-kick structure, so momentum is exact to round-off and the energy error is
bounded.

## 2. How it was done

1. `hertz_gate / hooke_gate / friction_gate / cundall_gate` build the three
   closed-form scales; the velocity-dependent Hertz gate comes straight out of
   `hertz_impact_time.hertz_contact_time`.
2. Single-pair impacts are stepped by the contract on the shipped
   `contact_timestep_ledger.WallImpact` kernel; the nominal restitution is
   calibrated by bisection against a converged (`dt/t_c = 5\\times10^{-4}$) solve
   (`calibrate_alpha`), so the measured ladder error is *integration* error only.
3. The assembly integrator re-uses the shipped contact laws (linear normal
   spring + dashpot, Cundall–Strack tangential with cone return map,
   `k_r = 0.25 k_t r_{eff}^2` rolling) and adds the rate split; the pressure is
   validated against `tensordem.granular2d.stress.virial_tensor`.
4. The jamming protocol shares one prepared state (compressed with the *shipped*
   m5524-lineage engine) between the two rates, so the comparison isolates the
   integrator.
5. The stage-A matrix is re-evaluated in 1/2/4 separate processes for the shard
   criterion.

## 3. Results

### 3.1 Restitution ladder (criterion a)

| e | dt/t_c | ratio | sub-steps | e measured | e converged | error | gate |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.02 | 0.02 | 1 | 0.50036 | 0.50000 | 0.07 % | ok |
| 0.5 | 0.05 | 0.05 | 1 | 0.50290 | 0.50000 | 0.58 % | ok |
| 0.5 | 0.1 | 0.1 | 2 | 0.50290 | 0.50000 | 0.58 % | subcycle |
| 0.5 | 0.2 | 0.2 | 4 | 0.50290 | 0.50000 | 0.58 % | subcycle |
| 0.9 | 0.02 | 0.02 | 1 | 0.90022 | 0.90000 | 0.02 % | ok |
| 0.9 | 0.05 | 0.05 | 1 | 0.90163 | 0.90000 | 0.18 % | ok |
| 0.9 | 0.1 | 0.1 | 2 | 0.90163 | 0.90000 | 0.18 % | subcycle |
| 0.9 | 0.2 | 0.2 | 4 | 0.90163 | 0.90000 | 0.18 % | subcycle |

### 3.2 Energy / momentum ledger, 1e5 coarse steps (criterion b)

assembly: $N=64$, $\\varphi=0.739$,
$\\mu=0$, $\\zeta=0$, hierarchy 100:1 on 10 %,
outer margin 0.05.

| rate | n_sub | dt [s] | steps | time span [s] | max |dE|/E | |dP|/|P| | wall [s] |
|---|---|---|---|---|---|---|---|
| single | 1 | 6.400e-06 | 100000 | 6.4004e-01 | 7.156e-03 | 2.11e-15 | 82.1 |
| mts | 10 | 6.400e-05 | 100000 | 6.4004e+00 | 7.268e-03 | 1.09e-14 | 401.5 |

### 3.3 Double-periodic shear (scenario 2)

| mu | tau/P single | tau/P multi-rate | rel. diff | Z single | Z multi-rate |
|---|---|---|---|---|---|
| 0.1 | 0.1678 | 0.1711 | 1.98 % | 2.673 | 2.824 |
| 0.3 | 0.1170 | 0.0558 | 52.33 % | 2.746 | 2.728 |
| 0.5 | 0.0777 | 0.0347 | 55.27 % | 3.133 | 2.949 |
| 0.7 | 0.0725 | 0.0974 | 34.47 % | 2.820 | 2.711 |

### 3.4 Friction jamming, N=256 (criterion c)

| case | rate | phi_J | phi(P->1) | n_sub | wall [s] | steps |
|---|---|---|---|---|---|---|
| hierarchical | single | 0.72902 | 0.72848 | 1 | 35.8 | 14958 |
| hierarchical | mts | 0.72641 | 0.72543 | 10 | 30.5 | 1494 |
| uniform | single | 0.72866 | 0.72813 | 1 | 4.5 | 1494 |
| uniform | mts | 0.72866 | 0.72813 | 1 | 4.2 | 1494 |
| shipped | single | 0.70805 | - | 1 | 18.2 | 5982 |

W4-P1 servo anchor (shipped `jamming_density_isotropic`): phi_J = 0.73256
(wall 64.6 s). Multi-rate hierarchical vs anchor:
0.84 %.


### 3.5 Tightness negative control (criterion d)

gate verdict at $dt/dt_c=0.5$: **fail**,
rejected = True. Forced (un-subcycled) runs:

| e | e measured | e converged | error | e^2 (energy factor) | detectable |
|---|---|---|---|---|---|
| 0.5 | 2.0317 | 0.5000 | 306.3 % | 4.128 | True |
| 0.9 | 2.3090 | 0.9000 | 156.6 % | 5.332 | True |

### 3.6 Shard reproducibility (criterion e)

| shards | cases/shard | worst |de| | bitwise | pass |
|---|---|---|---|---|
| 1 | [8] | 0.000e+00 | True | True |
| 2 | [4, 4] | 0.000e+00 | True | True |
| 4 | [2, 2, 2, 2] | 0.000e+00 | True | True |

### 3.7 Trade-off: which arm may be held on the coarse step

| variant | phi_J | wall [s] | bias vs fine-state |
|---|---|---|---|
| internal variables on the fine step (default) | 0.72641 | 28.2 | - |
| internal variables on the coarse step | 0.75688 | 25.3 | 4.20 % |

### 3.8 Ablation: what may be frozen on the coarse step

| variant | n_sub | dt outer | energy drift | wall [s] |
|---|---|---|---|---|
| resolved_contacts | 5 | 6.40e-05 | 1.222e-02 | 7.2 |
| frozen_soft | 5 | 6.40e-05 | 2.848e+00 | 5.1 |

## 4. Figures

* `fig1_ladder.png` — restitution error against the ladder with the gate zones.
* `fig2_ledger.png` — energy drift and contact energy over 1e5 steps, both rates.
* `fig3_shear.png` — shear stress ratio and coordination vs $\\mu$.
* `fig4_jamming.png` — pressure ladder + $\\varphi_J$ and the wall clock of the same physics.
* `fig5_ablation.png` — the frozen-soft-force ablation.

## 5. Caveats and limits

* **The speed-up is not free.**  Sub-cycling the stiff skeleton *while freezing*
  the soft contact force is the cheaper scheme and it destroys the energy ledger
  (2.8e+00 vs
  1.2e-02, section 3.7): the
  soft and stiff springs act on the same pair, so the force classes do not
  separate in frequency.  The shipped integrator therefore sub-cycles the whole
  active set and pays for it.
* **Criterion (c) fails on the wall clock, and this is the honest headline.**
  With the internal variables and the contacts resolved accurately (which is what
  keeps $\varphi_J$), the multi-rate solver costs
  1.17x the single-rate one over the same pressure ladder
  (30.5 s vs 35.8 s) and
  0.60x the shipped engine on the same
  ladder.  In a jammed packing the active set is most of the candidate list, so
  sub-cycling costs what it saves: the ten-fold larger coarse step is paid back by
  ten contact passes.  The cheap variants exist (sections 3.7 and 3.8) and they
  are the ones that buy speed -- at 4.2 % $\varphi_J$ bias or a two-order-of-magnitude
  energy violation.
* **Cross-code agreement is limited.**  The shipped engine restarts the
  tangential/rolling state whenever its neighbour list is rebuilt; the multi-rate
  integrator carries `xi`/`roll` across rebuilds by contact key.  That difference
  biases $\\varphi_J$ between the two implementations and is the reason the
  W4-P1 anchor check is reported separately from the same-code check.
* The deeply overlapped end of a *violent* compression ($\\varphi\\to0.92$) is the
  worst case for a coarse step (there the two rates differ by ~2 % in
  $\\varphi$, see the test suite), while the sane pressure ladder used for
  $\\varphi_J$ agrees to 0.36 %.
* **The shear scenario is not converged.**  The profile-thermostat cell only
  reaches a strain of 0.5 with a 3 % polydispersity-free bidisperse packing and
  reports $\tau/P$ values of 0.03-0.17 that are not monotone in $\mu$; the two
  rates therefore disagree by up to 55 %.  The
  stage is reported as a FAIL, not tuned away: a converged steady state needs a
  longer run (strain $\gtrsim 2$) which the node's compute budget did not carry.

## 6. Error judgement

| check | evidence | verdict |
|---|---|---|
| a  restitution ladder | worst 0.58 % (bound 2 %) | PASS |
| b  energy ledger 1e5 | drift 7.268e-03 / single 7.156e-03 (bound 1 %) | PASS |
| b  linear momentum | max |dP|/|P| 1.09e-14 (bound 1e-12) | PASS |
| c  wall clock >= 2x | multi-rate vs own single-rate 1.17x; vs shipped baseline 0.60x | FAIL |
| c  phi_J unchanged | |dphi_J| 0.36 % (bound 0.5 %) | PASS |
| c  phi_J vs W4-P1 servo | 0.84 % (bound 0.5 %) | FAIL |
| d  tightness control | gate rejects 0.5: True, forced error 306.3 % | PASS |
| e  shard reproducibility | worst diff 0.0e+00 | PASS |
| 2  double-periodic shear | worst |d tau/P| 55.27 %, monotone in mu: False | FAIL |
| c  trade-off | coarse internal-variable rate: 4.20 % phi_J bias, 1.11x wall clock | documented |
| ablation | frozen-soft drift 2.85e+00 vs resolved 1.22e-02 | PASS |

**Is the key quantity within 3 % of the benchmark?**  The restitution (the only
quantity with an independent mathematical anchor) is within
0.58 % of the converged solution — **yes**.  The
jamming density has no closed-form anchor; it is compared to the W4-P1 servo
reference and to the same-code single-rate run, and those two comparisons are
reported separately and honestly (see the table).

## 7. Audit trail

```
module      src/tensordem/mts.py
campaign    benchmarks/mts_contact_integration.py
tests       tests/test_mts_contact_integration.py   (17 passed)
evidence    runs/P82W2_N09/verification.json
figures     docs/figures/P82W2_N09/
prereg      prereg/P82W2_N09.json
```
