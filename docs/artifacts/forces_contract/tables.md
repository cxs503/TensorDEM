### Per-subclass `forces()` keyword contract (live introspection)

| subclass override | `update_history` accepted | missing base keywords |
|---|---|---|
| `BowContactDEM` | yes | none |
| `EnergyAuditedIceDEM3D` | yes | none |
| `MovingWedgeBowIceDEM3D` | yes | none |
| `RudderIceLoadDEM` | yes | none |

### `diagnostics()` purity probe on `BowContactDEM` (6 steps, then probe)

| state component | max |delta| after probe |
|---|---|
| `positions` | 0.000e+00 |
| `velocities` | 0.000e+00 |
| `rolling_torque` | 0.000e+00 |
| `last_reaction` | 0.000e+00 |
| `alive_changes` | 0.000e+00 |
| `time` | 0.000e+00 |

### Bowl-ice stage (thickness 0.15 m, r = 0.025 m, 40 steps)

| quantity | value |
|---|---|
| simulated time [s] | 0.0034 |
| bow travel [m] | 0.0003 |
| peak ice resistance [N] | 80.044 |
| mean ice resistance [N] | 70.851 |
| max contact pressure [Pa] | 2.883e+04 |
| broken bonds at end | 40 |

### Mass error induced by the 917 kg/m3 hard code

| rho_i [kg/m3] | induced mass error [%] | inside +/-3 % gate |
|---|---|---|
| 880 | +4.205 | **no** |
| 890 | +3.034 | **no** |
| 900 | +1.889 | yes |
| 910 | +0.769 | yes |
| 917 | +0.000 | yes |
| 925 | -0.865 | yes |
| 935 | -1.925 | yes |
| 950 | -3.474 | **no** |
