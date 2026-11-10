# DEM Coarse-Graining Scaling Rule Table

## Physical Background

Coarse-graining (CG) replaces N particles of radius R with N/f³ particles of
radius f·R while preserving macroscopic behaviour. Two scaling approaches are
implemented and cross-validated:

### 口径 A: Contact-Time Invariance (k_n ∝ f³)
- Derived from t_c = π·√(m_eff/k_n) = invariant
- Since m ∝ f³, we need k_n ∝ f³
- Preserves contact dynamics exactly
- Sound speed scales as f·c (may affect dynamic loading)

### 口径 B: Sound-Speed Invariance (k_n ∝ f)
- Derived from c = √(k_n/m_eff)·L_char = invariant
- k_n ∝ f preserves macroscopic wave propagation
- Contact time grows linearly: t_c ∝ f
- Force peaks may smear at large f

## Scaling Rule Table

| Quantity | 口径 A (Contact-Time) | 口径 B (Sound-Speed) |
|----------|----------------------|---------------------|
| radius | f·R | f·R |
| mass | f³·m | f³·m |
| inertia | f⁵·I | f⁵·I |
| kn | f³·k_n | f·k_n |
| kt | f³·k_t | f·k_t |
| kr | f⁵·k_r | f³·k_r |
| gamma | f³·γ | f²·γ |
| contact_time | t_c (invariant) | f·t_c |
| sound_speed | f·c | c (invariant) |
| restitution | e (invariant) | e (invariant) |
| mu | μ (invariant) | μ (invariant) |
| mu_rolling | μ_r (invariant) | μ_r (invariant) |
| gravity | g (invariant) | g (invariant) |

## Applicability Domains

### A_contact_time_invariant
- **Best for**: Contact dynamics, force-displacement curves, restitution
- **Valid range**: f ≤ 4 (f=8 may show minor deviations)
- **Failure mode**: Wave speed scales as f·c, may affect dynamic loading

### B_sound_speed_invariant
- **Best for**: Wave propagation, dynamic loading, seismic response
- **Valid range**: f ≤ 4 (f=8 contact time grows, may need larger dt)
- **Failure mode**: Contact time grows as f·t_c, force peaks may smear

## Failure Boundaries

| Factor f | 口径 A Valid? | 口径 B Valid? | Notes |
|----------|--------------|--------------|-------|
| 1 | ✓ | ✓ | Baseline |
| 2 | ✓ | ✓ | Both approaches within tolerance |
| 4 | ✓ (borderline) | ✓ | A: minor wave speed drift; B: contact time 4× |
| 8 | ⚠ (may fail) | ⚠ (may fail) | A: wave speed 8×; B: contact time 8×, force smearing |

## Negative Controls

1. **Mass-only scaling** (m → f³m, k_n unchanged): breaks contact time by f^(3/2)
2. **Wrong k_n scaling** (k_n ∝ f³ for sound-speed intent): breaks wave speed by f

## Recommendations

- Use **口径 A** when contact dynamics, force-displacement curves, and restitution
  are the primary concern (f ≤ 4 recommended).
- Use **口径 B** when wave propagation, dynamic loading, and seismic response
  are the primary concern (f ≤ 4 recommended).
- For f > 4, both approaches show increasing deviations; consider hybrid
  approaches or full-resolution simulation.
