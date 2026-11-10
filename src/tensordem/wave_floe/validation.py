"""C1..C7 criterion runner for the wave-ice coupling verification campaign.

Every criterion returns a dict with the measured numbers, the reference it was
measured against, its tolerance and a ``status`` in
``{"PASS", "FAIL", "WITHHELD"}``.  ``WITHHELD`` is used where the tree holds no
valid reference (an external potential-flow RAO, digitised transmission data):
the honest statement is that the criterion *cannot be adjudicated here*, not
that it passed.

Criteria
--------
C1  single floe RAO: heave/pitch peak position and amplitude versus the
    frequency-domain continuum transfer function, plus the exact long-wave limit
    ``RAO -> 1``;
C2  array transmission ``T``: transfer-matrix model, energy closure, reciprocity
    and the single-interface Fresnel limit; the experimental comparison is
    WITHHELD (no data in the tree, and the present coupled solver is one-way);
C3  wave-induced breakup: ``H_crit`` from the Winkler-foundation beam versus the
    closed-form floating-plate result;
C4  JONSWAP spectral input: ensemble PSD versus the target spectrum (``R^2``);
C5  energy: work/energy audit of the time-domain floe, and the exact
    ``E c_g`` wave energy flux versus a ``<p u>`` depth quadrature;
C6  refinement: floe lattice refined x2, RAO change; beam mesh refined x2,
    stress change;
C7  honest-FAIL sourcing: every non-PASS criterion with its root cause.
"""
from __future__ import annotations

import json
import math
import platform
from typing import Any, Sequence

import numpy as np

from .array import (FloeArrayTransmission, single_interface_transmission,
                    slab_transmission_analytic, transmission_vs_kh)
from .breakup import (WinklerBeam, characteristic_wavenumber, critical_wave_height,
                      finite_length_correction, theory_bending_stress)
from .response import (FloeLattice, FloeResponseModel, continuum_coefficients,
                       discrete_coefficients, frequency_response,
                       rao_from_time_series)
from .spectra import Spectrum, discretize, ensemble_psd, jonswap, pierson_moskowitz, \
    spectral_fidelity, spectrum_grid
from .waves import RegularWave, wave_energy_flux

# ---------------------------------------------------------------------------
# Benchmark cases (all values are inputs of the validation, not fits)
# ---------------------------------------------------------------------------

FLOE_CASE = dict(length=20.0, thickness=1.0, n_x=40, n_y=2)
FLOE_CASE_FINE = dict(length=20.0, thickness=1.0, n_x=80, n_y=4)
WAVE_CASE = dict(height=0.4, period=8.0)
RAO_PERIODS = [1.5, 1.7, 1.9, 2.05, 2.2, 2.32, 2.45, 2.6, 2.8, 3.1, 3.5, 4.0, 5.0,
               6.0, 8.0, 10.0, 13.0, 17.0, 23.0, 32.0, 45.0]
RAO_TD_PERIODS = [2.32, 3.1, 5.0, 8.0, 13.0, 23.0, 45.0]
SPEC_CASE = dict(significant_height=2.0, peak_period=8.0, gamma=3.3)
ARRAY_CASE = dict(depth=1.0, floe_length=5.0, gap=5.0, n_floes=12, thickness=0.2)
BREAKUP_CASE = dict(youngs_modulus=9.0e9, thickness=1.0, poisson=0.3,
                    sigma_f=0.5e6, length=200.0, n_elements=80)


def _status(err: float, tol: float) -> str:
    return "PASS" if err <= tol else "FAIL"


# ---------------------------------------------------------------------------
# C1: single floe RAO
# ---------------------------------------------------------------------------

def _crossing(series, level: float, key: str = "rao_heave") -> float | None:
    """First kL (ascending) at which ``key`` falls below ``level`` (linear interp)."""
    pts = [(r["kL"], r[key]) for r in sorted(series, key=lambda r: r["kL"])]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - level) * (y1 - level) <= 0.0 and y0 != y1:
            return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
    return None


def criterion_c1(verbose: bool = True) -> dict:
    lat = FloeLattice(**FLOE_CASE)
    z_eq = lat.equilibrium_height()
    waves = [RegularWave(height=WAVE_CASE["height"], period=T) for T in RAO_PERIODS]
    continuum = [frequency_response(w, lat, z_eq, reference="continuum") for w in waves]
    discrete = [frequency_response(w, lat, z_eq, reference="discrete") for w in waves]

    # Time-domain points (the solver under test).  Small amplitude so that the
    # comparison is against the *linear* reference (nonlinearity ~ A/h).
    td = []
    for T in RAO_TD_PERIODS:
        amp = 0.02
        w = RegularWave(height=amp, period=T)
        model = FloeResponseModel(lat, w, z_eq=z_eq)
        n_per, n_steady = 240, 11
        tr = model.run(dt=T / n_per, n_steps=int(n_per * n_steady))
        f3 = rao_from_time_series(tr.time, tr.heave - z_eq, w.omega, skip=0.5)
        f5 = rao_from_time_series(tr.time, tr.pitch, w.omega, skip=0.5)
        td.append({
            "period": T,
            "amplitude": amp,
            "rao_heave": f3["amplitude"] / w.amplitude,
            "rao_pitch": f5["amplitude"] / (w.wavenumber * w.amplitude),
            "residual_heave": f3["residual_fraction"],
            "residual_pitch": f5["residual_fraction"],
            "energy": tr.energy,
            "kL": w.wavenumber * lat.length,
        })

    cont_by_T = {round(r.period, 6): r for r in continuum}
    comparisons = []
    for row in td:
        ref = cont_by_T[round(row["period"], 6)]
        comparisons.append({
            "period": row["period"],
            "kL": row["kL"],
            "rao_heave_td": row["rao_heave"],
            "rao_heave_continuum": ref.rao_heave,
            "err_heave": abs(row["rao_heave"] - ref.rao_heave) / ref.rao_heave,
            "rao_pitch_td": row["rao_pitch"],
            "rao_pitch_continuum": ref.rao_pitch,
            "err_pitch": abs(row["rao_pitch"] - ref.rao_pitch) / ref.rao_pitch,
        })

    # The RAO of a floe is monotone (it rides the surface in long waves, is
    # filtered out in short waves), so the discriminating features are the
    # maximum (saturation level) and the wavenumber at which it is filtered out.
    def extremes(series, key):
        get = (lambda r: r[key]) if isinstance(series[0], dict) else \
              (lambda r: getattr(r, key))
        best = max(series, key=get)
        period = best["period"] if isinstance(best, dict) else best.period
        kL = best["kL"] if isinstance(best, dict) else best.kL
        return period, get(best), kL

    td_peak_T, td_peak_v, _ = extremes(td, "rao_heave")
    cont_peak_T, cont_peak_v, _ = extremes(continuum, "rao_heave")
    peak_amp_err = abs(td_peak_v - cont_peak_v) / cont_peak_v
    peak_pos_err = abs(td_peak_T - cont_peak_T) / cont_peak_T

    # Crossings must be evaluated on the *same* period grid on both sides,
    # otherwise the interpolation bias of the coarse grid is mistaken for a model
    # difference (it was 29% before this was fixed).
    cont_on_td_grid = [{"kL": c["kL"], "rao_heave": c["rao_heave_continuum"]}
                       for c in comparisons]
    k90_td = _crossing(td, 0.9)
    k90_cont = _crossing(cont_on_td_grid, 0.9)
    k50_td = _crossing(td, 0.5)
    k50_cont = _crossing(cont_on_td_grid, 0.5)
    band = [c for c in comparisons if 0.1 <= c["kL"] <= 2.0]
    curve_err = max(c["err_heave"] for c in band)
    curve_err_pitch = max(c["err_pitch"] for c in band)

    long_wave = [c for c in comparisons if c["period"] == max(RAO_TD_PERIODS)][0]
    long_wave_err = abs(long_wave["rao_heave_td"] - 1.0)
    disc_err = max(abs(d.rao_heave - c.rao_heave) / c.rao_heave
                   for d, c in zip(discrete, continuum))
    # Restrict the discretisation metric to the resolved band: below kL ~ 0.1 and
    # above kL ~ 2.5 the response itself is a few percent of the wave amplitude and
    # the excitation is a near-perfect cancellation, so a relative error there says
    # nothing about the model (reported separately, not hidden).
    band_disc = [(abs(d.rao_heave - c.rao_heave) / c.rao_heave)
                 for d, c in zip(discrete, continuum) if 0.1 <= d.kL <= 2.5]

    result = {
        "geometry": lat.to_dict(),
        "coefficients_continuum": continuum_coefficients(lat).to_dict(),
        "coefficients_discrete": discrete_coefficients(lat, z_eq).to_dict(),
        "continuum_curve": [r.to_dict() for r in continuum],
        "discrete_curve": [r.to_dict() for r in discrete],
        "td_points": td,
        "comparisons": comparisons,
        "peak_period_continuum": cont_peak_T,
        "peak_period_td": td_peak_T,
        "peak_rao_heave_continuum": cont_peak_v,
        "peak_rao_heave_td": td_peak_v,
        "peak_rao_pitch_continuum": max(r.rao_pitch for r in continuum),
        "peak_position_error": peak_pos_err,
        "peak_amplitude_error": peak_amp_err,
        "kL_rao_90_td": k90_td,
        "kL_rao_90_continuum": k90_cont,
        "kL_rao_50_td": k50_td,
        "kL_rao_50_continuum": k50_cont,
        "kL_filter_90_error": (abs(k90_td - k90_cont) / k90_cont
                               if k90_td and k90_cont else None),
        "kL_filter_50_error": (abs(k50_td - k50_cont) / k50_cont
                               if k50_td and k50_cont else None),
        "max_curve_error_heave": curve_err,
        "max_curve_error_pitch": curve_err_pitch,
        "long_wave_rao_td": long_wave["rao_heave_td"],
        "long_wave_error": long_wave_err,
        "discretisation_error_max": disc_err,
        "discretisation_error_band": max(band_disc),
        "max_td_fit_residual": max(max(r["residual_heave"], r["residual_pitch"])
                                   for r in td),
        "tolerance": {"peak_position": 0.10, "peak_amplitude": 0.15,
                      "curve_band": 0.15, "long_wave": 0.05},
        "status_peak_position": _status(peak_pos_err, 0.10),
        "status_peak_amplitude": _status(peak_amp_err, 0.15),
        "status_curve_band": _status(curve_err, 0.15),
        "status_long_wave": _status(long_wave_err, 0.05),
        "reference": "frequency-domain continuum Morison/FK transfer function "
                     "(self-derived, exact within the model); the long-wave "
                     "limit RAO->1 is independent physics",
        "external_reference": "WITHHELD: no WAMIT/AQWA or experimental RAO data "
                              "exists in this tree",
    }
    result["status"] = ("PASS" if all(result[s] == "PASS" for s in (
        "status_peak_position", "status_peak_amplitude", "status_curve_band",
        "status_long_wave")) else "FAIL")
    return result


# ---------------------------------------------------------------------------
# C2: array transmission
# ---------------------------------------------------------------------------

def criterion_c2() -> dict:
    sweep = transmission_vs_kh([4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0,
                                26.0, 34.0, 46.0], **ARRAY_CASE)
    closure = max(abs(r["energy_closure"] - 1.0) for r in sweep)
    reciprocity = max(r["reciprocity_error"] for r in sweep)
    empty = transmission_vs_kh([6.0, 12.0, 20.0], depth=ARRAY_CASE["depth"],
                               floe_length=ARRAY_CASE["floe_length"],
                               gap=ARRAY_CASE["gap"], n_floes=0,
                               thickness=ARRAY_CASE["thickness"])
    empty_err = max(abs(r["transmission"] - 1.0) for r in empty)
    # Independent closed form: one homogeneous slab, two interfaces + propagation.
    slab_rows = []
    for T in (4.0, 6.0, 8.0, 12.0, 20.0):
        arr = FloeArrayTransmission(period=T, depth=ARRAY_CASE["depth"],
                                    floe_length=ARRAY_CASE["floe_length"], gap=0.0,
                                    n_floes=1, thickness=ARRAY_CASE["thickness"])
        t_tmm = abs(arr.coefficients()[0])
        ana = slab_transmission_analytic(T, ARRAY_CASE["depth"],
                                         ARRAY_CASE["floe_length"],
                                         ARRAY_CASE["thickness"])
        slab_rows.append({"period": T, "tmm": t_tmm,
                          "analytic": ana["transmission"],
                          "rel_error": abs(t_tmm - ana["transmission"]) / ana["transmission"]})
    slab_err = max(r["rel_error"] for r in slab_rows)
    iface = single_interface_transmission(period=8.0, depth=ARRAY_CASE["depth"],
                                          thickness=ARRAY_CASE["thickness"])
    tmin = min(sweep, key=lambda r: r["transmission"])
    return {
        "case": ARRAY_CASE,
        "sweep": sweep,
        "bragg_min_transmission": tmin["transmission"],
        "bragg_period": tmin["period"],
        "bragg_k0_floe_length": tmin["k0_floe_length"],
        "max_energy_closure_error": closure,
        "max_reciprocity_error": reciprocity,
        "empty_channel_transmission_error": empty_err,
        "slab_closed_form_check": slab_rows,
        "slab_closed_form_error": slab_err,
        "single_interface_fresnel": iface,
        "status_energy_closure": _status(closure, 1e-9),
        "status_reciprocity": _status(reciprocity, 1e-9),
        "status_empty_channel": _status(empty_err, 1e-9),
        "status_slab_closed_form": _status(slab_err, 1e-9),
        "tolerance": {"energy_closure": 1e-9, "reciprocity": 1e-9,
                      "slab": 1e-9, "experiment": 0.20},
        "status_experiment": "WITHHELD",
        "reason_experiment": (
            "No digitised transmission data (Kohout et al. 2014 style) exists in "
            "this tree, and the coupled floe solver is driven by a prescribed "
            "incident wave (no free-surface unknown), so it cannot produce a "
            "measured T. The long-wave slab model is also outside the dispersive "
            "regime (kd ~ 1) of those experiments."),
        "reference": "transfer-matrix long-wave slab model checked against the "
                     "closed-form finite-slab solution, the empty channel, "
                     "reciprocity and exact energy closure",
        "status": "WITHHELD",
    }


# ---------------------------------------------------------------------------
# C3: wave-induced breakup threshold
# ---------------------------------------------------------------------------

def criterion_c3() -> dict:
    E = BREAKUP_CASE["youngs_modulus"]
    h = BREAKUP_CASE["thickness"]
    nu = BREAKUP_CASE["poisson"]
    sigma_f = BREAKUP_CASE["sigma_f"]
    k_c = characteristic_wavenumber(E, h, nu)
    theory = critical_wave_height(sigma_f, E, h, k_c, nu)
    beam = WinklerBeam(length=BREAKUP_CASE["length"], thickness=h,
                       youngs_modulus=E, poisson=nu,
                       n_elements=BREAKUP_CASE["n_elements"])
    fem = beam.critical_wave_height(sigma_f, k_c)
    fem_fine = WinklerBeam(length=BREAKUP_CASE["length"], thickness=h,
                           youngs_modulus=E, poisson=nu,
                           n_elements=BREAKUP_CASE["n_elements"] * 2
                           ).critical_wave_height(sigma_f, k_c)
    err = abs(fem["critical_wave_height"] - theory["critical_wave_height"]) \
        / theory["critical_wave_height"]
    mesh_err = abs(fem_fine["critical_wave_height"] - fem["critical_wave_height"]) \
        / fem["critical_wave_height"]
    # Stress-vs-wavelength curve (the physical content: a maximum at k_c).
    ks = np.logspace(math.log10(0.2 * k_c), math.log10(5.0 * k_c), 25)
    curve = []
    for k in ks:
        th = theory_bending_stress(E, h, float(k), 0.1, nu)
        curve.append({"k": float(k), "k_over_kc": float(k / k_c),
                      "sigma": th["sigma"], "w_over_a": th["w_over_a"]})
    # Finite-length edge correction (a deliverable, not a criterion).
    lengths = [1.0 / k_c * f for f in (1.0, 2.0, 4.0, 8.0, 12.0)]
    correction = finite_length_correction(lengths, thickness=h,
                                          youngs_modulus=E, nu=nu,
                                          n_elements=BREAKUP_CASE["n_elements"])
    return {
        "case": BREAKUP_CASE,
        "characteristic_wavenumber": k_c,
        "characteristic_length": 1.0 / k_c,
        "characteristic_wavelength": 2.0 * math.pi / k_c,
        "theory_H_crit": theory["critical_wave_height"],
        "fem_H_crit": fem["critical_wave_height"],
        "fem_check_stress": fem["check_stress"],
        "relative_error": err,
        "mesh_refinement_error": mesh_err,
        "fem_details": {k: v for k, v in fem.items() if k != "check_amplitude"},
        "stress_vs_k": curve,
        "finite_length_correction": correction,
        "tolerance": {"H_crit": 0.25, "mesh": 0.05},
        "status_H_crit": _status(err, 0.25),
        "status_mesh": _status(mesh_err, 0.05),
        "reference": "closed-form floating-plate (Winkler foundation) bending "
                     "stress at k_c",
        "scope": "first-stress criterion; no crack growth, no DEM bond fracture",
        "status": "PASS" if err <= 0.25 else "FAIL",
    }


# ---------------------------------------------------------------------------
# C4: spectral input fidelity
# ---------------------------------------------------------------------------

def criterion_c4(n_realisations: int = 24, seed0: int = 1000) -> dict:
    grid = spectrum_grid(SPEC_CASE["peak_period"], n=240)
    jsw = jonswap(grid, **SPEC_CASE)
    pm = pierson_moskowitz(grid, SPEC_CASE["significant_height"],
                           SPEC_CASE["peak_period"])
    w, amp, _ = discretize(jsw, phase_seed=0)
    Tp = SPEC_CASE["peak_period"]
    dt = Tp / 32.0
    duration = 300 * Tp
    # The Welch segment must resolve the spectral peak: 8 peak periods.
    seg = int(16 * Tp / dt)  # 16 peak periods per Welch segment
    om_est, s_est = ensemble_psd(w, amp, np.array([0.0]), dt=dt,
                                 duration=duration,
                                 n_realisations=n_realisations,
                                 segment=min(seg, int(duration / dt / 4)),
                                 seed0=seed0)
    fid = spectral_fidelity(jsw, om_est, s_est)
    # Convergence curve: R^2 versus number of realisations (a figure for the report).
    conv = []
    for n in (1, 2, 4, 8, 16, 24, 32):
        om_n, s_n = ensemble_psd(w, amp, np.array([0.0]), dt=dt,
                                 duration=duration, n_realisations=n,
                                 segment=min(seg, int(duration / dt / 4)),
                                 seed0=seed0)
        conv.append({"n_realisations": n,
                     "r2": spectral_fidelity(jsw, om_n, s_n)["r2"]})
    # Band-resolution study: the estimator can only reproduce a peak as narrow as
    # its own bandwidth, so R^2 must be reported together with the segment length.
    resolution = []
    for seg_Tp in (4, 8, 16, 32):
        om_r, s_r = ensemble_psd(w, amp, np.array([0.0]), dt=dt,
                                 duration=max(duration, 600 * Tp),
                                 n_realisations=8,
                                 segment=int(seg_Tp * Tp / dt), seed0=seed0)
        f_r = spectral_fidelity(jsw, om_r, s_r)
        resolution.append({"segment_periods": seg_Tp,
                           "delta_omega": float(om_r[1] - om_r[0]),
                           "r2": f_r["r2"], "hs_ratio": f_r["hs_ratio"]})
    single = []
    t_axis = np.arange(0.0, duration, dt)
    from .spectra import synthesize_eta, welch_psd
    ph = np.random.default_rng(seed0).uniform(0.0, 2 * math.pi, w.size)
    eta = synthesize_eta(w, amp, ph, t_axis)
    om1, s1 = welch_psd(eta, dt, segment=min(seg, int(duration / dt / 4)))
    single = spectral_fidelity(jsw, om1, s1)
    return {
        "case": SPEC_CASE,
        "target_moments": jsw.to_dict(),
        "pm_moments": pm.to_dict(),
        "pm_hs_check": pm.significant_height / SPEC_CASE["significant_height"],
        "jonswap_hs_check": jsw.significant_height / SPEC_CASE["significant_height"],
        "fidelity_ensemble": fid,
        "fidelity_single_realisation": single,
        "convergence": conv,
        "resolution_study": resolution,
        "n_realisations": n_realisations,
        "dt": dt,
        "duration": duration,
        "segment_samples": min(seg, int(duration / dt / 4)),
        "estimate_omega": om_est.tolist(),
        "estimate_psd": s_est.tolist(),
        "target_omega": jsw.omega.tolist(),
        "target_psd": jsw.S.tolist(),
        "tolerance": {"r2": 0.95},
        "status": _status(1.0 - fid["r2"], 1.0 - 0.95),
        "reference": "the analytic JONSWAP spectrum used to build the sea state",
    }


# ---------------------------------------------------------------------------
# C5: energy
# ---------------------------------------------------------------------------

def criterion_c5() -> dict:
    lat = FloeLattice(**FLOE_CASE)
    z_eq = lat.equilibrium_height()
    T = 8.0
    w = RegularWave(height=0.1, period=T)
    model = FloeResponseModel(lat, w, z_eq=z_eq)
    n_per = 600
    tr = model.run(dt=T / n_per, n_steps=int(n_per * 6))
    # Still-water power balance in the frequency domain: <P_in> = <P_damp>.
    fd = frequency_response(w, lat, z_eq, reference="discrete")
    p_in = 0.5 * (fd.excitation_heave * np.conj(-1j * fd.omega * fd.heave_phasor)).real
    p_damp = 0.5 * fd.damping_heave * fd.omega ** 2 * abs(fd.heave_phasor) ** 2
    # Free decay: no wave, no damping -> energy conserved.
    quiet = FloeResponseModel(lat, RegularWave(height=0.0, period=T), z_eq=z_eq,
                              damping_ratio_heave=0.0, damping_ratio_pitch=0.0)
    drift = quiet.run(dt=T / 300, n_steps=300 * 10, z0=0.05, record_every=1)
    e_drift = abs(drift.energy["delta_energy"]) / abs(drift.energy["energy_start"])
    flux = wave_energy_flux(w)
    return {
        "wave_case": {"period": T, "height": w.height},
        "td_energy_ledger": tr.energy,
        "td_residual_fraction": tr.energy["residual_fraction"],
        "frequency_domain_power_in": float(p_in),
        "frequency_domain_power_damped": float(p_damp),
        "frequency_domain_balance_error": float(
            abs(p_in - p_damp) / max(abs(p_in), 1e-30)),
        "free_decay_energy_drift": float(e_drift),
        "wave_energy_flux": flux,
        "tolerance": {"td_residual": 0.05, "power_balance": 0.02,
                      "free_decay_drift": 1e-6, "flux_quadrature": 1e-3},
        "status_td_ledger": _status(tr.energy["residual_fraction"], 0.05),
        "status_power_balance": _status(
            float(abs(p_in - p_damp) / max(abs(p_in), 1e-30)), 0.02),
        "status_free_decay": _status(float(e_drift), 1e-6),
        "status_flux": _status(flux["rel_error"], 1e-3),
        "status": "PASS" if (tr.energy["residual_fraction"] <= 0.05
                             and float(abs(p_in - p_damp) / max(abs(p_in), 1e-30)) <= 0.02
                             and float(e_drift) <= 1e-6
                             and flux["rel_error"] <= 1e-3) else "FAIL",
        "reference": "work-energy identity of the linear system (exact); "
                     "E c_g versus <p u> (exact linear result)",
    }


# ---------------------------------------------------------------------------
# C6: refinement
# ---------------------------------------------------------------------------

def criterion_c6() -> dict:
    coarse = FloeLattice(**FLOE_CASE)
    fine = FloeLattice(**FLOE_CASE_FINE)
    z_c = coarse.equilibrium_height()
    z_f = fine.equilibrium_height()
    rows = []
    for T in (2.32, 5.0, 8.0):
        w = RegularWave(height=WAVE_CASE["height"], period=T)
        c = frequency_response(w, coarse, z_c, reference="discrete")
        f = frequency_response(w, fine, z_f, reference="discrete")
        cc = frequency_response(w, coarse, z_c, reference="continuum")
        rows.append({
            "period": T,
            "rao_heave_coarse": c.rao_heave,
            "rao_heave_fine": f.rao_heave,
            "rao_heave_continuum": cc.rao_heave,
            "change": abs(f.rao_heave - c.rao_heave) / c.rao_heave,
            "err_coarse_vs_continuum": abs(c.rao_heave - cc.rao_heave) / cc.rao_heave,
            "err_fine_vs_continuum": abs(f.rao_heave - cc.rao_heave) / cc.rao_heave,
        })
    worst = max(r["change"] for r in rows)
    beam_c = WinklerBeam(length=BREAKUP_CASE["length"],
                         thickness=BREAKUP_CASE["thickness"],
                         youngs_modulus=BREAKUP_CASE["youngs_modulus"],
                         n_elements=BREAKUP_CASE["n_elements"])
    beam_f = WinklerBeam(length=BREAKUP_CASE["length"],
                         thickness=BREAKUP_CASE["thickness"],
                         youngs_modulus=BREAKUP_CASE["youngs_modulus"],
                         n_elements=BREAKUP_CASE["n_elements"] * 2)
    k_c = characteristic_wavenumber(BREAKUP_CASE["youngs_modulus"],
                                    BREAKUP_CASE["thickness"])
    s_c = beam_c.solve(0.1, k_c)["sigma_max"]
    s_f = beam_f.solve(0.1, k_c)["sigma_max"]
    beam_change = abs(s_f - s_c) / s_c
    return {
        "lattice_coarse": coarse.to_dict(),
        "lattice_fine": fine.to_dict(),
        "rao_rows": rows,
        "max_rao_change": worst,
        "beam_stress_change": beam_change,
        "tolerance": {"rao_change": 0.05, "beam_change": 0.02},
        "status_rao": _status(worst, 0.05),
        "status_beam": _status(beam_change, 0.02),
        "status": "PASS" if worst <= 0.05 and beam_change <= 0.02 else "FAIL",
        "reference": "self-convergence (x2 refinement in each discretisation)",
    }


# ---------------------------------------------------------------------------
# C7: honest-FAIL sourcing
# ---------------------------------------------------------------------------

def criterion_c7(criteria: dict) -> dict:
    sources = []
    for name, rec in criteria.items():
        if rec.get("status") in ("FAIL", "WITHHELD"):
            reason = rec.get("reason_experiment") or rec.get("external_reference") \
                or rec.get("scope") or "tolerance not met"
            sources.append({"criterion": name, "status": rec["status"],
                            "root_cause": reason,
                            "measured": {k: rec.get(k) for k in (
                                "relative_error", "peak_position_error",
                                "peak_amplitude_error", "max_rao_change",
                                "td_residual_fraction") if k in rec}})
    return {
        "n_criteria": len(criteria),
        "n_pass": sum(1 for r in criteria.values() if r.get("status") == "PASS"),
        "n_fail": sum(1 for r in criteria.values() if r.get("status") == "FAIL"),
        "n_withheld": sum(1 for r in criteria.values()
                          if r.get("status") == "WITHHELD"),
        "sources": sources,
        "unverified_scope": [
            "external potential-flow RAO (WAMIT/AQWA) — no data in the tree",
            "dispersive multi-floe transmission versus experiment — needs a "
            "free-surface (two-way) solver",
            "DEM bonded-particle fracture under wave load — the C3 route is a "
            "continuum FEM beam, not particle bond rupture",
            "radiation damping: a frequency-independent added mass is used; no "
            "potential-flow coefficient table exists",
        ],
        "status": "PASS" if all(r.get("status") != "FAIL" for r in criteria.values())
        else "PARTIAL",
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def run_validation(include_c4: bool = True, c4_realisations: int = 24,
                   verbose: bool = False) -> dict:
    """Run every criterion and return the report dict (JSON-serialisable)."""
    criteria: dict[str, Any] = {}
    criteria["C1_single_floe_rao"] = criterion_c1()
    if verbose:
        print("C1 done:", criteria["C1_single_floe_rao"]["status"])
    criteria["C2_array_transmission"] = criterion_c2()
    if verbose:
        print("C2 done:", criteria["C2_array_transmission"]["status"])
    criteria["C3_wave_breakup"] = criterion_c3()
    if verbose:
        print("C3 done:", criteria["C3_wave_breakup"]["status"])
    if include_c4:
        criteria["C4_spectral_input"] = criterion_c4(n_realisations=c4_realisations)
        if verbose:
            print("C4 done:", criteria["C4_spectral_input"]["status"])
    criteria["C5_energy"] = criterion_c5()
    if verbose:
        print("C5 done:", criteria["C5_energy"]["status"])
    criteria["C6_refinement"] = criterion_c6()
    if verbose:
        print("C6 done:", criteria["C6_refinement"]["status"])
    criteria["C7_honest_fail"] = criterion_c7(criteria)
    return {
        "campaign": "wave-ice coupling verification (p348w2-n07)",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "conventions": {
            "units": "SI",
            "floe": "2-D beam-seas section, unit width",
            "rao_pitch_normalisation": "|theta|/(k A) (surface slope)",
            "hs": "4 sqrt(m0)",
        },
        "criteria": criteria,
    }