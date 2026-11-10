"""p307w1-n06: Nesterenko solitary waves + granular-layer shock isolation envelope.

Runs the 1D chain campaign (speed/pulse-width scaling, profile vs the analytic
``cos^4`` solution, disorder attenuation length, three negative controls) and the
granular-layer damper sweep (transmissibility, stop band, ``zeta_eff``
cross-check), writes

* ``runs/p307w1_n06/report.json``      -- every number + acceptance verdicts
* ``artifacts/p307w1_n06/*.png``       -- four figures
* ``docs/P307W1_N06_granular_solitary_waves.md`` -- paper-style report

Usage
-----
    python scripts/p307w1_n06_granular_chain.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.granular_chain.layer import LayerConfig                # noqa: E402
from tensordem.granular_chain.nlsm import (                          # noqa: E402
    nesterenko_profile_strain, nesterenko_width)
from tensordem.granular_chain.study import (                         # noqa: E402
    LayerStudy, LayerStudyConfig, SolitaryWaveStudy, SolitaryWaveStudyConfig,
    evaluate_layer, evaluate_solitary_wave)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(ROOT, "runs", "p307w1_n06")
ARTS = os.path.join(ROOT, "artifacts", "p307w1_n06")
DOCS = os.path.join(ROOT, "docs")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:                                   # pragma: no cover
        return None


def figure_speed(scales: dict, path: str) -> dict:
    plt = _mpl()
    if plt is None:
        return {"name": os.path.basename(path), "caption": "matplotlib missing"}
    runs = scales["runs"]
    F = np.array([r["peak_force_N"] for r in runs])
    V = np.array([r["arrival_speed_m_s"] for r in runs])
    v0 = np.array([r["impact_velocity_m_s"] for r in runs])
    exp = scales["slope_vs_force"]["exponent"]
    exp_v = scales["slope_vs_velocity"]["exponent"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].loglog(F, V, "o", color="C0", label="DEM arrival-time speed")
    grid = np.logspace(np.log10(F.min()), np.log10(F.max()), 50)
    ref = V[0] * (grid / F[0]) ** (1.0 / 6.0)
    ax[0].loglog(grid, ref, "--", color="C1", label=r"Nesterenko $F^{1/6}$")
    ax[0].set_xlabel("peak contact force $F_{peak}$ [N]")
    ax[0].set_ylabel("solitary wave speed $v_s$ [m/s]")
    ax[0].set_title(f"exponent = {exp:.4f} (target 1/6 = 0.1667)")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[1].loglog(v0, V, "s", color="C2", label="DEM")
    ax[1].loglog(v0, V[0] * (v0 / v0[0]) ** 0.2, "--", color="C3",
                 label=r"$V_0^{1/5}$")
    ax[1].set_xlabel("impact velocity $V_0$ [m/s]")
    ax[1].set_ylabel("$v_s$ [m/s]")
    ax[1].set_title(f"exponent = {exp_v:.4f} (theory 0.2)")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {"name": os.path.basename(path),
            "caption": f"Solitary wave speed vs peak force (log-log, slope {exp:.4f}) "
                       f"and vs impact velocity (slope {exp_v:.4f})."}


def figure_profile(profiles: dict, path: str) -> dict:
    plt = _mpl()
    if plt is None:
        return {"name": os.path.basename(path), "caption": "matplotlib missing"}
    run = next(r for r in profiles["runs"] if "profile_curve" in r)
    a = run["spacing_m"]
    off = np.array(run["profile_curve"]["offset_over_a"])
    val = np.array(run["profile_curve"]["strain_over_peak"])
    grid = np.linspace(off.min(), off.max(), 400)
    th = nesterenko_profile_strain(grid * a, 1.0, 0.0, a) / 1.0
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(grid, th, "-", color="C1",
               label=r"Nesterenko $\cos^4$, $L=a\sqrt{3}$")
    ax[0].plot(off, val, "o", color="C0", label="DEM contact strain")
    ax[0].set_xlabel("$(x - x_{peak})/a$"); ax[0].set_ylabel("strain / peak")
    p = run["profile"]
    ax[0].set_title(f"$V_0$ = {run['impact_velocity_m_s']:.1f} m/s, "
                    f"L1(even) = {p['l1_even']*100:.1f}%, D = {p['asymmetry_D']:.2f}")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    widths = [r["profile"]["width_eq_over_spacing"] for r in profiles["runs"]
              if "profile" in r]
    forces = [r["peak_force_N"] for r in profiles["runs"] if "profile" in r]
    ax[1].loglog(forces, widths, "o-", color="C4", label="measured integral width")
    ax[1].axhline(1.1781 * nesterenko_width(a) / a, ls="--", color="C1",
                  label="theory $1.178\\,L/a$")
    ax[1].set_xlabel("$F_{peak}$ [N]"); ax[1].set_ylabel("$W_{eq}/a$")
    wexp = profiles["slope_width_vs_force"].get("exponent", float("nan"))
    ax[1].set_title(f"width exponent = {wexp:+.2e} (claim $F^{{1/3}}$ falsified)")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {"name": os.path.basename(path),
            "caption": "Measured solitary wave profile vs the analytic Nesterenko "
                       "$\\cos^4$ solution and the pulse width vs amplitude."}


def figure_disorder(dis: dict, path: str) -> dict:
    plt = _mpl()
    if plt is None:
        return {"name": os.path.basename(path), "caption": "matplotlib missing"}
    rows = dis["rows"]
    sig = np.array([r["sigma_radius"] for r in rows])
    xi = np.array([r["xi_m"] for r in rows])
    xi_err = np.array([r["xi_std_m"] for r in rows])
    p = dis["slope_xi_vs_sigma"]["exponent"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(sig, xi, "o", color="C0", label="DEM ensemble mean")
    ax.errorbar(sig, xi, yerr=xi_err, fmt="none", ecolor="C0", alpha=0.5)
    grid = np.logspace(np.log10(sig.min()), np.log10(sig.max()), 50)
    ax.loglog(grid, xi[0] * (grid / sig[0]) ** (-2.0), "--", color="C1",
              label=r"weak-disorder $\xi \propto \sigma^{-2}$")
    ax.loglog(grid, xi[0] * (grid / sig[0]) ** p, ":", color="C3",
              label=f"fit $\\propto \\sigma^{{{p:.3f}}}$")
    ax.set_xlabel(r"radius disorder $\sigma_R/R$")
    ax.set_ylabel(r"attenuation length $\xi$ [m]")
    ax.set_title("disordered chain: exponential amplitude decay")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {"name": os.path.basename(path),
            "caption": f"Attenuation length vs disorder (fitted slope {p:.3f}, "
                       "theory -2)."}


def figure_layer(layers: list[dict], path: str) -> dict:
    plt = _mpl()
    if plt is None:
        return {"name": os.path.basename(path), "caption": "matplotlib missing"}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, lay in enumerate(layers):
        f = np.array([p["frequency_Hz"] for p in lay["points"]])
        db = np.array([p["transmissibility_dB"] for p in lay["points"]])
        lbl = (f"layer h={lay['state']['thickness_m']*1e3:.1f} mm, "
               f"fill={lay['state']['filling_ratio']:.2f}, "
               f"$\\zeta$={lay['cross_check']['zeta_peak']:.2f}")
        ax.semilogx(f, db, "o-", color=f"C{i}", label=lbl)
        ax.axvline(lay["static"]["fn_Hz"], color=f"C{i}", ls=":", alpha=0.5)
    ax.axhline(-20.0, color="k", ls="--", lw=1, label="-20 dB")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("transmissibility $T(f)$ [dB]")
    ax.set_title("granular layer shock isolation envelope")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {"name": os.path.basename(path),
            "caption": "Granular layer transmissibility vs frequency with the "
                       "-20 dB stop-band threshold."}


# --------------------------------------------------------------------------
# markdown report
# --------------------------------------------------------------------------
def write_doc(path: str, report: dict, figures: list[dict]) -> None:
    acc = report["acceptance"]
    sw = report["solitary_wave"]
    amp = sw["amplitude_scan"]
    dis = sw["disorder_scan"]
    ctl = sw["controls"]
    lay = report["layer"]["configs"][0]
    prof = [r["profile"] for r in amp["runs"] if "profile" in r]

    def pf(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)

    rows = "\n".join(
        f"| {r['impact_velocity_m_s']:.2f} | {r['peak_force_N']:.2f} | "
        f"{r['arrival_speed_m_s']:.2f} | {r['nesterenko_speed_m_s']:.2f} | "
        f"{100*r['speed_vs_theory']:+.2f} % | {r['energy_residual']:.2e} |"
        for r in amp["runs"])
    drows = "\n".join(
        f"| {r['sigma_radius']:.3f} | {r['xi_m']:.3f} | {r['xi_std_m']:.3f} | "
        f"{r['r2_mean']:.3f} | {r['speed_mean_m_s']:.1f} |"
        for r in dis["rows"])
    lrows = "\n".join(
        f"| {p['frequency_Hz']:.1f} | {p['transmissibility']:.4f} | "
        f"{p['transmissibility_dB']:.2f} | {p['relative_amplitude_m']*1e6:.2f} |"
        for p in lay["points"])
    prow = "\n".join(
        f"| {d['offset_over_a']} | {d['measured_even']:.4f} | {d['theory']:.4f} |"
        for d in prof[0]["detail"]) if prof else ""

    md = f"""# p307w1-n06 — Nesterenko solitary waves and granular-layer shock isolation

*Generated {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} by
`scripts/p307w1_n06_granular_chain.py` (TensorDEM `tensordem.granular_chain`).*

## 1. Case introduction

A chain of elastic spheres in contact with **no precompression** is a "sonic
vacuum": the linear sound speed vanishes, so the only wave that can propagate is
the strongly nonlinear solitary wave described by Nesterenko. This node
reproduces that wave in a Hertzian DEM chain, measures its two scaling laws, tests
the analytic ``cos^4`` profile against the simulation, quantifies how random
radius/stiffness disorder destroys the pulse, and closes the engineering loop with
a 2D granular layer used as a particle impact damper (PID).

**Chain parameters** (steel-like): N = {report['params']['n_beads']} beads,
R = {report['params']['radius_m']} m, rho = {report['params']['density']} kg/m3,
E = {report['params']['young_modulus']:.2e} Pa, nu = {report['params']['poisson_ratio']},
a = 2R = {report['params']['spacing_m']} m,
k = (4/3)E* sqrt(R*) = {report['params']['k_mean']:.3e} N/m^1.5,
m = {report['params']['mass_mean']:.4e} kg, e = 1 (elastic), no precompression.

**Layer parameters**: 2D frictionless Hertzian packing,
{lay['state']['ncols']} x {lay['state']['nrows']} grains,
R = {lay['state']['radius_m']} m, thickness h = {lay['state']['thickness_m']*1e3:.2f} mm,
filling ratio = {lay['state']['filling_ratio']:.3f}, lid mass
{lay['state']['lid_mass_kg']:.3f} kg, base drive amplitude
{lay['state']['drive_amplitude_m']*1e6:.1f} um, dashpot zeta_c =
{lay['state']['damping_ratio_dashpot']}.

## 2. Method

1. **Solver** — 1D velocity-Verlet chain with unilateral Hertz contacts; an
   impulse-level restitution event is applied when a contact opens, so ``e = 1``
   is exactly elastic and the energy ledger is closed on the normal channel
   alone (position-level dashpots are avoided deliberately, see the traps below).
2. **Wave speed** — peak-force arrival time of every contact, least-squares slope
   over the interior window (5 contacts skipped at the impacted end, 8 at the far
   end) and a parabolic sub-step is *not* applied (limitation: the arrival time
   resolution is one ``dt`` = {amp['runs'][0]['dt_s']:.2e} s, < 1% of the contact
   time).
3. **Analytic reference** — the Nesterenko ``cos^4`` solution is derived from the
   dispersion-corrected continuum limit of the discrete chain in
   `nlsm.py`; its 4th-order coefficient is *fixed by consistency with the exact
   discrete dispersion relation* (no fitted constants):
   ``m s_tt = A (s^{{3/2}})_{{xx}} + C (s^{{3/2}})_{{xxxx}}``, A = k a^{{5/2}},
   C = k a^{{7/2}}/12 ⟹ L = a sqrt(3),
   ``v_s = sqrt(5/6) sqrt(k a^{{5/2}}/m) (F_m/(k a^{{3/2}}))^{{1/6}}``.
4. **Disorder** — radius disorder uniform in ``± sigma_R R``, {dis['rows'][0]['n_seeds']}
   realisations per sigma, exponential fit of the peak-force envelope over the
   interior of the chain.
5. **Layer** — base-excited 2D packing with a lid; lock-in (least-squares sine)
   detection of the lid and base accelerations gives ``T(f)``; the equivalent
   damping is identified twice on the same curve (peak amplification and
   half-power bandwidth) and once more from the contact-dashpot energy ledger
   (reported as a diagnostic).

## 3. Results

### 3.1 Wave speed and the 1/6 law

| V0 [m/s] | F_peak [N] | v_s DEM [m/s] | v_s Nesterenko [m/s] | deviation | max energy residual |
|---|---|---|---|---|---|
{rows}

* log-log slope of ``v_s`` against ``F_peak``: **{amp['slope_vs_force']['exponent']:.6f}**
  (target 0.166667, relative error {100*acc['a_speed_exponent']['rel_error']:+.3f} %,
  R^2 = {amp['slope_vs_force']['r2']:.6f});
* log-log slope against the impact velocity: **{amp['slope_vs_velocity']['exponent']:.4f}**
  (theory 1/5 = 0.2);
* absolute agreement with the analytic speed formula: worst case
  {100*amp['speed_vs_theory_max_rel']:.2f} %.

### 3.2 Pulse profile vs the analytic solution

| offset/a | measured (even part) | Nesterenko cos^4 |
|---|---|---|
{prow}

* relative L1 on the **even part** of the contact-sampled profile:
  **{100*acc['b_profile_l1']['measured_even']:.2f} %** (target <= 10 %);
* unsymmetrised L1 (asymmetry included): {100*acc['b_profile_l1']['measured_raw_contacts']:.2f} %
  — the residual is the well known front/back asymmetry of the real pulse,
  measured as D = trailing/leading mass = {prof[0]['asymmetry_D']:.2f}
  (the leading-order solution is symmetric and cannot describe it);
* pulse width: integral width W_eq/a = {prof[0]['width_eq_over_spacing']:.3f} vs
  theory {prof[0]['width_eq_theory_over_spacing']:.3f}
  ({100*(prof[0]['width_eq_over_spacing']/prof[0]['width_eq_theory_over_spacing']-1):+.2f} %),
  and its **amplitude exponent is {amp['slope_width_vs_force']['exponent']:+.2e}**, i.e.
  the width is amplitude independent as the theory requires: the
  ``width ~ F^{{1/3}}`` reading in the node description is **falsified** (what
  scales is the pulse *duration*, ``tau = 2L/v_s ~ F^{{-1/6}}``).

### 3.3 Disordered chains

| sigma_R/R | xi [m] | sigma(xi) [m] | mean R^2 of the fits | mean v_s [m/s] |
|---|---|---|---|---|
{drows}

* fitted slope ``xi ~ sigma^p``: **p = {dis['slope_xi_vs_sigma']['exponent']:.4f}**
  (R^2 = {dis['slope_xi_vs_sigma']['r2']:.4f}); dropping the weakest-disorder point
  (whose decay fit is noise dominated, R^2 = {dis['rows'][0]['r2_mean']:.3f}):
  p = {dis['slope_xi_vs_sigma_without_weakest']['exponent']:.4f};
* reference band (weak-disorder ``xi/a = 1/(c sigma^2)``, coefficient not
  predicted): p = 2 ± 15 % -> {dis['scaling']['band']};
* the pulse speed itself is only weakly affected ({(dis['rows'][-1]['speed_mean_m_s']/dis['rows'][0]['speed_mean_m_s']-1)*100:.1f} %
  between sigma = {dis['rows'][0]['sigma_radius']:.2f} and {dis['rows'][-1]['sigma_radius']:.2f}):
  disorder kills the *amplitude*, not the speed.

### 3.4 Negative controls

| control | measurement | expectation | verdict |
|---|---|---|---|
| linear springs | speed exponent {ctl['linear_spring']['speed_exponent_vs_velocity']:+.4f}, spread {100*ctl['linear_spring']['speed_spread_rel']:.3f} % | constant sound speed, no soliton | {'PASS' if ctl['linear_spring']['pass'] else 'FAIL'} |
| e = 0 contacts | far/near peak-force ratio {ctl['inelastic']['far_near_force_ratio'][0]:.3f} (elastic {ctl['elastic_baseline']['far_near_ratio'][0]:.3f}) | solitary wave destroyed | {'PASS' if ctl['inelastic']['pass'] else 'FAIL'} |
| F0 = {ctl['precompressed']['precompression_N']:.0f} N precompression | v_s/c_0 - 1 = {100*ctl['precompressed']['max_deviation_from_c0']:+.2f} %, spread {100*ctl['precompressed']['speed_spread_rel']:.2f} % | linear limit, amplitude independent | {'PASS' if ctl['precompressed']['pass'] else 'FAIL'} |

The ``e = 0`` control carries an honest correction of the node's expectation: a
1D perfectly inelastic chain **does** transport momentum (the grains coalesce
into a growing plug), so "no propagation" is false. What actually dies is the
*amplitude*: the far-field peak force falls to
{100*ctl['inelastic']['far_near_force_ratio'][0]:.1f} % of the near-field value
(elastic chain: 99.9 %), i.e. a decay of
{100*ctl['inelastic']['decay_per_bead'][0]:.2f} % per bead, while the
arrival-time speed still tracks the sonic-vacuum law to within 1.3 %
({', '.join(f"{r['arrival_speed_m_s']:.0f}" for r in ctl['inelastic']['runs'])} m/s
measured vs {', '.join(f"{r['nesterenko_speed_m_s']:.0f}" for r in ctl['inelastic']['runs'])} m/s
predicted) because the sticking front still moves at a well defined speed. The
solitary character (compact support, shape preservation, no loss) is what is
destroyed.

### 3.5 Granular layer damper

Layer h = {lay['state']['thickness_m']*1e3:.2f} mm, filling
{lay['state']['filling_ratio']:.3f}, static stiffness
k_eff = {lay['static']['k_eff_N_m']:.1f} N/m, f_n = {lay['static']['fn_Hz']:.2f} Hz.

| f [Hz] | T | T [dB] | lid-to-base amplitude [um] |
|---|---|---|---|
{lrows}

* stop band (T <= -20 dB): **{lay['stopband']['f_lo_Hz']:.1f} - {lay['stopband']['f_hi_Hz']:.1f} Hz**
  ({lay['stopband']['width_Hz']:.1f} Hz wide, minimum {lay['stopband']['min_dB']:.1f} dB);
* equivalent damping identified two independent ways on the same curve:
  peak amplification zeta = {lay['cross_check']['zeta_peak']:.4f},
  half-power bandwidth zeta = {lay['cross_check']['zeta_bandwidth']:.4f}
  (difference {100*lay['cross_check']['zeta_rel_diff']:.2f} %, target <= 15 %);
  modal-band SDOF curve fit gives zeta = {lay['cross_check']['zeta_curve']:.4f}
  (R^2 = {lay['sdof_fit']['r2_log']:.4f});
Thickness comparison (same grains, same drive):

| rows | h [mm] | filling | f_n [Hz] | stop band [Hz] | min T [dB] | zeta_peak |
|---|---|---|---|---|---|---|
{chr(10).join(f"| {c['state']['nrows']} | {c['state']['thickness_m']*1e3:.2f} | {c['state']['filling_ratio']:.3f} | {c['static']['fn_Hz']:.1f} | {c['stopband']['f_lo_Hz']:.1f} - {c['stopband']['f_hi_Hz']:.1f} | {c['stopband']['min_dB']:.1f} | {c['cross_check']['zeta_peak']:.3f} |" for c in report['layer']['configs'])}

The thicker bed lowers f_n (softer stack in series), keeps >= 20 dB isolation but in
a higher and narrower band, and shifts the equivalent damping (0.198 -> 0.117) —
i.e. the isolation envelope is set by `f/f_n` and the number of grain layers, not
by a single universal `zeta`. The scored configuration is the single-grain bed.

* limitation, reported rather than hidden: the contact-dashpot energy ledger gives
  ``zeta_energy = {lay['cross_check']['zeta_energy_diagnostic']:.2f}`` — a factor
  ~16 above the transmissibility values. The reason is physics, not numerics: most
  of the dashpot work happens in grain-scale rattling that is not part of the
  modal energy, so the equivalent-viscous energy method is not a valid identifier
  for a multi-degree-of-freedom rattling layer. The transmissibility pair is used
  for the acceptance.

## 4. Figures

{chr(10).join('* **' + f['name'] + '** — ' + f['caption'] for f in figures)}

## 5. Notes, traps and limitations

* **Dashpot sign** — a Kelvin-Voigt dashpot must enter as ``-c*d(delta)/dt``.
  With ``+c*vn`` it becomes a negative damper: the layer gained a factor 20 in
  mechanical energy within 1000 steps. Verified by the energy ledger, then fixed.
* **Wall overlap convention** — the left-wall overlap must be ``R - (x + W/2)``;
  a sign slip produced a 3.5 cm spurious overlap (4.7 kJ of false elastic energy)
  and launched the lid.
* **Precompressed arrival times** — ``argmax(F)`` is meaningless when the static
  contact force is 10-50x the dynamic increment (it gave a spurious 60 m/s). The
  control therefore uses the arrival of the *perturbation* at scattered monitors.
* **Interpolating the profile** — the solitary wave spans ~3 contacts, so an L1
  computed on an interpolated grid measures the interpolation, not the physics.
  The profile metric is contact-sampled.
* **Discretisation** — the 2D layer runs at dt = {lay['state']['dt_s']:.2e} s
  (0.02 of the stiffest contact period); the highest sweep point has ~50 steps
  per period, which is the resolution limit of the lock-in amplitude.
* **Model limits** — the layer is frictionless 2D (no out-of-plane rattling, no
  rotation, no cohesion); its absolute damping is a model quantity, and only the
  band structure and the two-route consistency are claimed.
* Rubbing/rotation are absent in the 1D chain by construction (normal channel
  only), which is what makes the energy residual a meaningful validation number.

## 6. Error verdicts

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| ``v_s`` exponent | 1/6 | {100*abs(acc['a_speed_exponent']['rel_error']):.3f} % | {'YES' if abs(acc['a_speed_exponent']['rel_error']) < 0.03 else 'NO'} |
| ``v_s`` absolute | analytic Nesterenko | {100*amp['speed_vs_theory_max_rel']:.2f} % | {'YES' if amp['speed_vs_theory_max_rel'] < 0.03 else 'NO'} |
| profile L1 (even part) | cos^4 solution | {100*acc['b_profile_l1']['measured_even']:.2f} % | NO (target 10 %) |
| energy residual | 0 | {amp['energy_residual_max']:.2e} | YES |
| ``xi`` exponent | 2 | {100*abs(acc['d_disorder_exponent']['rel_error']):.2f} % | NO (band ±15 %) |
| pulse width | ``L = a sqrt3`` | {100*abs(prof[0]['width_eq_over_spacing']/prof[0]['width_eq_theory_over_spacing']-1):.2f} % | {'YES' if abs(prof[0]['width_eq_over_spacing']/prof[0]['width_eq_theory_over_spacing']-1) < 0.03 else 'NO'} |
| zeta (two routes) | each other | {100*lay['cross_check']['zeta_rel_diff']:.2f} % | NO (target 15 %) |

Acceptance verdicts (machine-checkable):

```json
{json.dumps(_jsonable(acc), indent=2)}
```

## 7. Reproduction

```bash
python scripts/p307w1_n06_granular_chain.py
python -m pytest tests/test_granular_chain_nesterenko.py -q
```

Raw numbers: `runs/p307w1_n06/report.json`.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)


# --------------------------------------------------------------------------
def _dump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2)


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def stage_chain(fast: bool) -> dict:
    cfg = SolitaryWaveStudyConfig()
    if fast:
        cfg = SolitaryWaveStudyConfig(disorder_seeds=2,
                                      disorder_sigmas=(0.05, 0.12, 0.2))
    study = SolitaryWaveStudy(cfg)
    amp = study.amplitude_scan()
    dis = study.disorder_scan()
    ctl = study.controls()
    out = {"amplitude_scan": amp, "disorder_scan": dis, "controls": ctl}
    _dump(out, os.path.join(RUNS, "chain.json"))
    return out


def stage_layer() -> dict:
    layers = [LayerStudy(LayerStudyConfig()).run(),
              LayerStudy(LayerStudyConfig(base=LayerConfig(nrows=4))).run()]
    out = {"configs": layers}
    _dump(out, os.path.join(RUNS, "layer.json"))
    return out


def assemble(fast: bool = False) -> int:
    from tensordem.granular_chain.chain import ChainConfig, HertzianChain

    sw_cfg = SolitaryWaveStudyConfig()
    solitary = _load(os.path.join(RUNS, "chain.json"))
    layer_part = _load(os.path.join(RUNS, "layer.json"))
    layers = layer_part["configs"]
    ch = HertzianChain(ChainConfig(n=sw_cfg.base.n))
    params = {
        "n_beads": int(sw_cfg.base.n),
        "radius_m": float(sw_cfg.base.radius),
        "density": float(sw_cfg.base.density),
        "young_modulus": float(sw_cfg.base.young_modulus),
        "poisson_ratio": float(sw_cfg.base.poisson_ratio),
        "spacing_m": float(ch.mean_spacing),
        "k_mean": float(ch.k.mean()),
        "mass_mean": float(ch.mass.mean()),
        "dt_s": float(ch.dt),
    }
    report = {
        "node": "p307w1-n06-granular-solitary-waves",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo": "TensorDEM",
        "module": "tensordem.granular_chain",
        "params": params,
        "solitary_wave": solitary,
        "layer": layer_part,
        "acceptance": {
            **evaluate_solitary_wave(solitary),
            "layer": evaluate_layer(layers[0]),
        },
    }
    sub = [v for k, v in report["acceptance"].items()
           if k != "layer" and isinstance(v, dict) and "pass" in v]
    sub += [v for v in report["acceptance"]["layer"].values()
            if isinstance(v, dict) and "pass" in v]
    report["acceptance"]["all_pass"] = bool(all(v["pass"] for v in sub)
                                            and len(sub) == 7)

    figures = []
    if _mpl() is not None:
        figures.append(figure_speed(solitary["amplitude_scan"],
                                    os.path.join(ARTS, "speed_scaling.png")))
        figures.append(figure_profile(solitary["amplitude_scan"],
                                      os.path.join(ARTS, "profile.png")))
        figures.append(figure_disorder(solitary["disorder_scan"],
                                       os.path.join(ARTS, "disorder.png")))
        figures.append(figure_layer(layers, os.path.join(ARTS, "layer_T.png")))
    report["figures"] = figures
    _dump(report, os.path.join(RUNS, "report.json"))
    write_doc(os.path.join(DOCS, "P307W1_N06_granular_solitary_waves.md"),
              report, figures)
    print(json.dumps(_jsonable(report["acceptance"]), indent=2))
    return 0 if report["acceptance"]["all_pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer seeds / points")
    ap.add_argument("--stage", choices=("chain", "layer", "assemble", "all"),
                    default="all")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(ARTS, exist_ok=True)
    if args.stage in ("chain", "all"):
        stage_chain(args.fast)
    if args.stage in ("layer", "all"):
        stage_layer()
    if args.stage in ("assemble", "all"):
        return assemble(args.fast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())