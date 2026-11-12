### Table A - stored round-3 points vs constant-stress validity (gate off = rounds 1-3 protocol)

| arm | I_target | p_meas (Pa) | p error | mu | phi_bed_steady | valid (<5%) |
|---|---|---|---|---|---|---|
| baseline | 0.001 | 61639 | -1.4% | 0.3295 | 0.5356 | YES |
| baseline | 0.003 | 58838 | -5.9% | 0.3289 | 0.5342 | **NO** |
| baseline | 0.01 | 53116 | -15.0% | 0.3850 | 0.5287 | **NO** |
| baseline | 0.03 | 47116 | -24.6% | 0.4074 | 0.5241 | **NO** |
| baseline | 0.1 | 44937 | -28.1% | 0.4302 | 0.5231 | **NO** |
| baseline | 0.3 | 52182 | -16.5% | 0.4916 | 0.5197 | **NO** |
| stiff | 0.001 | 61010 | -2.4% | 0.3126 | 0.5183 | YES |
| stiff | 0.003 | 62688 | +0.3% | 0.3309 | 0.5151 | YES |
| stiff | 0.01 | 57391 | -8.2% | 0.2957 | 0.5144 | **NO** |
| stiff | 0.03 | 140161 | +124.3% | 0.3474 | 0.5135 | **NO** |
| stiff | 0.1 | 121273 | +94.0% | 0.3727 | 0.5103 | **NO** |
| stiff | 0.3 | 144274 | +130.8% | 0.4019 | 0.5031 | **NO** |
| thick | 0.001 | 62869 | +0.6% | 0.2778 | 0.5769 | YES |
| thick | 0.003 | 63247 | +1.2% | 0.2430 | 0.5765 | YES |
| thick | 0.01 | 70227 | +12.4% | 0.2773 | 0.5765 | **NO** |
| thick | 0.03 | 74717 | +19.5% | 0.1980 | 0.5763 | **NO** |
| thick | 0.1 | 82214 | +31.5% | 0.3868 | 0.5750 | **NO** |
| thick | 0.3 | 98630 | +57.8% | 0.5686 | 0.5736 | **NO** |

### Table B - gate off vs gate on, same (arm, I) point (mu/phi on the **final 20 % window**, which is the window the gate settles on)

| arm | I_target | gate | p error (tail) | p error (final) | servo_valid | recover steps | mu final | phi final | mu err vs ref | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.01 | off | (not run) | | | | | | | |
| baseline | 0.01 | on | -8.3% | -0.4% | yes | 702 | 0.3959 | 0.5326 | +5.3% | 33 |
| baseline | 0.1 | off | (not run) | | | | | | | |
| baseline | 0.1 | on | -12.7% | -0.6% | yes | 3492 | 0.3234 | 0.5529 | -33.5% | 48 |
| baseline | 0.3 | off | -16.5% | -31.6% | no | 0 | 0.5538 | 0.5197 | -9.3% | 27 |
| baseline | 0.3 | on | -3.6% | -0.8% | yes | 20000 | 0.3477 | 0.5775 | -42.3% | 112 |
| thick | 0.1 | off | (not run) | | | | | | | |
| thick | 0.1 | on | +5.2% | -0.4% | yes | 521 | 0.3818 | 0.5730 | -20.1% | 56 |
| thick | 0.3 | off | +57.8% | +69.9% | no | 0 | 0.6204 | 0.5730 | +8.3% | 39 |
| thick | 0.3 | on | +6.1% | +3.1% | yes | 8988 | 0.4652 | 0.5363 | -22.0% | 159 |

### Table C - phi(I) reference sign check

| I | ref_phi (in-repo law) | d(ref_phi)/dI | measured phi (thick, gate on) |
|---|---|---|---|
| 0.001 | 0.5802 | - | n/a |
| 0.01 | 0.5819 | + | n/a |
| 0.1 | 0.5950 | + | 0.5733 |
| 0.3 | 0.6100 | + | 0.5384 |
