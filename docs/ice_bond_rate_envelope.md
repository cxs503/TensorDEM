# Rate dependence of the DEM bond: where `sigma_f ~ edot^(1/(m+1))` comes from

Referenced by the docstring of `src/tensordem/ice_bond_viscoelastic.py`; this
file supplies the derivation that was missing while the module was merged, and
points at the measured verification.

## 1. Two failure mechanisms on one bond

| mechanism | law | parameters |
|---|---|---|
| brittle cleavage | fail when `abs(F) >= F_b` | `F_b` Weibull (median, modulus `m_w`) |
| creep damage (Kachanov) | `dD/dt = (1/t_ref) (abs(F)/F_ref)**m`, fail at `D = 1` | `t_ref`, `F_ref`, `m` |

Both act on the same bond force `F(t)`; the `D` field does **not** soften the
force in the current module (see limit 4 below), so failure is the earlier of
the two events.

## 2. Closed-form envelope for a linear-elastic bond

Constant strain rate `edot`, elastic bond `F = K edot t`:

```
D(t) = (1/t_ref) (K edot / F_ref)**m * t**(m+1) / (m+1)
D(t_f) = 1   =>   t_f  = [ (m+1) t_ref (F_ref / (K edot))**m ]**(1/(m+1))
sigma_f = K edot t_f  =>   sigma_f ~ edot**(1/(m+1))
```

`m = 2` gives the classical ductile exponent `1/3` (Golding/Schulson) with no
envelope anywhere in the code — the exponent is an output of the failure-time
race. Measured fitted slopes for `m = 1 / 2 / 4`: see §3.5 of
`docs/ice_bond_viscoelastic_verification.md` (the harness table is authoritative
and is regenerated from `artifacts/ice_bond_viscoelastic/verification.json`;
this note only derives the law).

## 3. Brittle/ductile crossover

Equating `t_b = F_b / (K edot)` with `t_f` above gives a single crossover rate

```
edot* = (F_ref / K) * [ (m+1) t_ref (F_b/F_ref)**m ]**(-1/(m+1)) ... (solved numerically)
```

For `K = 1e6 N/m`, `m = 2`, `t_ref = 50 s`, `F_b = 50 N` the crossover sits near
`edot ~ 1e-3 1/s`. The harness sweeps `1e-5 ... 1e-1 1/s` and observes **both**
modes (`creep` at the slow end, `brittle` at the fast end), which is the
mechanism claim the module makes in its docstring.

## 4. Limits (stated, not implied)

1. **Constitutive only.** No engine call site: `dem3d.py` still uses the linear
   bond, so nothing here predicts a specimen strength yet.
2. **`G(t)` is N/m**, a bond stiffness. Converting to Pa needs the bonded
   cross-section.
3. **No damage softening.** `D` accumulates but does not enter `F`; a
   `F(1-D)` branch would move the post-peak response and is not implemented.
4. **The viscoelastic branch shifts the envelope.** With the default spectrum
   the measured slope is `0.344` against `1/3` (3.2 % high) over
   `1e-3 ... 1e-1 1/s`, because early-time behaviour is governed by the
   unrelaxed stiffness; the pure `1/(m+1)` law is the long-time (relaxed)
   asymptote.
5. **Parameter values are WITHHELD.** The default spectrum is
   literature-shaped (Sinha 1978 transient creep); no laboratory data set in
   this repository calibrates it.