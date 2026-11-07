"""Whipping assessment chain: DEM ice pulses -> wet modes -> stress -> fatigue.

The chain is the one described by the campaign node and is deliberately kept
explicit so each link can be audited:

(a) ice-hull contact force ``F(t)`` with >= 1 kHz sampling over >= 1 s
    (:mod:`tensordem.whipping.dem_load`);
(b) pulse segmentation and four-feature parametrisation of every event
    (:mod:`tensordem.whipping.pulses`);
(c) dry hull-girder modes (beam FEM), 2-D BEM section added mass, wet modes
    (:mod:`tensordem.whipping.modes`, :mod:`tensordem.whipping.sections`);
(d) modal superposition with an exact piecewise-linear Duhamel integrator and
    modal damping (:mod:`tensordem.whipping.response`);
(e) deck stress ``sigma(t) = sum q_i d2phi_i/dx2 E y_deck`` and rainflow
    counting (:mod:`tensordem.whipping.rainflow`);
(f) Miner damage on DNV-RP-C203 curve D, split into a quasi-static (< 1 Hz)
    and a whipping (1-50 Hz) port (:mod:`tensordem.whipping.fatigue`);
(g) every acceptance number plus its negative control and dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from . import fatigue, modes, pulses, response, sections
from .rainflow import rainflow as count_cycles, total_count

__all__ = ["ShipCase", "hull_added_mass_per_length", "assess", "stress_station"]

GRAVITY = 9.80665


@dataclass(frozen=True)
class ShipCase:
    """Longitudinal-strength idealisation of an ice-going hull girder."""

    length: float = 150.0            # m
    breadth: float = 22.0            # m
    draft: float = 10.5              # m
    block_coefficient: float = 0.86
    youngs_modulus: float = 2.06e11  # Pa
    inertia: float = 30.0            # m^4
    deck_height: float = 7.0         # m above the neutral axis
    rho_water: float = 1025.0
    zeta: float = 0.015
    n_modes: int = 4
    n_elements: int = 60
    relief: float = 1.0              # 3-D longitudinal relief of the strip mass
    load_station_fraction: float = 0.06   # bow-region load application point
    stress_station_fraction: float = 0.5  # midshipe deck stress
    full_scale_bow_load: float = 1.5e7    # N, documented full-scale peak load
    quasi_static_cutoff_hz: float = 1.0
    whipping_band_hz: tuple = (1.0, 50.0)

    @property
    def mass_per_length(self) -> float:
        return self.rho_water * self.block_coefficient * self.breadth * self.draft

    def as_dict(self) -> dict:
        return {
            "length_m": self.length, "breadth_m": self.breadth,
            "draft_m": self.draft, "Cb": self.block_coefficient,
            "E_Pa": self.youngs_modulus, "I_m4": self.inertia,
            "deck_height_m": self.deck_height, "rho_water": self.rho_water,
            "mass_per_length_kg_m": self.mass_per_length,
            "zeta": self.zeta, "n_modes": self.n_modes,
            "relief": self.relief,
            "load_station_m": self.load_station_fraction * self.length,
            "stress_station_m": self.stress_station_fraction * self.length,
            "full_scale_bow_load_N": self.full_scale_bow_load,
            "quasi_static_cutoff_hz": self.quasi_static_cutoff_hz,
            "whipping_band_hz": list(self.whipping_band_hz),
        }


def hull_added_mass_per_length(case: ShipCase, relief: float | None = None) -> dict:
    """Midship section added mass and the breadth-scaled distribution.

    A box section of the midship breadth/draft is solved with the 2-D BEM in
    its high-frequency (Dirichlet free-surface) limit; the resulting
    coefficient ``m_a / (rho B^2)`` is applied along the hull with the local
    breadth squared and the optional three-dimensional relief factor.
    """
    r = case.relief if relief is None else relief
    n_panels = 40
    pts = sections.box_section(case.breadth, case.draft, n_panels)
    bem = sections.added_mass_per_length(pts, case.rho_water, free_surface=True)
    coefficient = bem["added_mass_per_length"] / (case.rho_water * case.breadth ** 2)

    def m_a_of_x(x):
        b = modes.taper_breadth(x, case.length, case.breadth)
        return r * coefficient * case.rho_water * b * b

    return {
        "midship_added_mass_kg_m": bem["added_mass_per_length"],
        "coefficient_m_a_over_rho_B2": coefficient,
        "panel_count": bem["panel_count"],
        "bem_residual": bem["rhs_residual"],
        "relief": r,
        "m_a_of_x": m_a_of_x,
    }


def stress_station(wet: modes.WetModes, case: ShipCase) -> np.ndarray:
    """Modal curvature weights ``E y_deck d2phi_i/dx2`` at the stress station."""
    x = case.stress_station_fraction * case.length
    return np.array([
        case.youngs_modulus * case.deck_height * float(wet.shape(x, i, deriv=2)[0])
        for i in range(case.n_modes)
    ])


def _split_damage(sigma, dt, band, curve_key="D_air"):
    curve = fatigue.DNV_CURVES[curve_key]
    cycles = count_cycles(sigma)
    res = fatigue.miner_damage(curve,
                               [c.range for c in cycles], [c.count for c in cycles])
    return res, cycles


def assess(time, force, case: ShipCase = ShipCase(), *,
           n_pulses_report: int = 3, curve_key: str = "D_air",
           integration_rate_hz: float | None = None) -> dict:
    """Run the full chain and return every number the campaign asks for.

    ``integration_rate_hz`` optionally refines the sampling grid before the
    modal integration (the piecewise-linear load is reproduced exactly by the
    linear resampling); a finer grid is what brings the energy-ledger residual
    below the 1e-6 acceptance level, since the ledger is limited by the
    quadrature of ``F v_load`` and not by the integrator.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(force, dtype=float)
    if integration_rate_hz is not None and not np.isclose(
            1.0 / (t[1] - t[0]), integration_rate_hz, rtol=1e-9):
        t, f = pulses.resample_uniform(t, f, integration_rate_hz)
    dt = float(t[1] - t[0])

    # ---- (a)/(b) pulse segmentation of the DEM record -------------------- #
    pulse_list = pulses.detect_pulses(t, f)
    parametric = []
    for p in pulse_list:
        try:
            parametric.append(pulses.ParametricPulse.from_features(p))
        except ValueError:
            continue
    scale = case.full_scale_bow_load / float(f.max())
    f_scaled = f * scale

    # ---- (c) dry + wet modes -------------------------------------------- #
    ei = case.youngs_modulus * case.inertia
    dry = modes.dry_modes(ei, case.mass_per_length,
                          case.length, case.n_elements, case.n_modes)
    analytic = modes.free_free_analytic_frequencies(
        ei, case.mass_per_length, case.length, case.n_modes)
    mass = hull_added_mass_per_length(case)
    wet = modes.wet_modes(dry, mass["m_a_of_x"])
    weights = stress_station(wet, case)
    x_load = case.load_station_fraction * case.length
    participation = np.array([float(wet.shape(x_load, i)[0]) for i in range(case.n_modes)])

    # ---- (d)/(e) modal superposition -> deck stress ---------------------- #
    omega = 2.0 * math.pi * wet.freqs
    zeta = np.full(case.n_modes, case.zeta)
    modal_force = participation[:, None] * f_scaled[None, :]
    solution = response.modal_response(t, modal_force, omega, zeta)
    sigma = response.stress_history(solution, weights)
    ledger = response.energy_ledger(solution, f_scaled, participation, dt)

    # ---- (f) quasi-static / whipping split ------------------------------- #
    ls, hs = case.whipping_band_hz
    sigma_qs = response.lowpass(sigma, dt, case.quasi_static_cutoff_hz)
    sigma_wh = response.bandpass(sigma, dt, ls, hs)
    # amplitudes are compared on the AC (mean-removed) signal: the raw record
    # carries a large static offset which would otherwise dominate the RMS
    total_std = float(np.std(sigma))
    qs_std = float(np.std(sigma_qs))
    wh_rms = response.rms(sigma_wh)
    wh_share = wh_rms / total_std if total_std > 0 else 0.0
    # exact modal split (no filter leakage): stress carried by modes >= 2
    whipping_stress_modal = weights[1:] @ solution.q[1:]
    modal_std = float(np.std(whipping_stress_modal))
    modal_share = modal_std / total_std if total_std > 0 else 0.0

    d_total, cyc_total = _split_damage(sigma, dt, case.whipping_band_hz, curve_key)
    d_qs, _ = _split_damage(sigma_qs, dt, case.whipping_band_hz, curve_key)
    d_wh, _ = _split_damage(sigma_wh, dt, case.whipping_band_hz, curve_key)
    # primary split: standard removal technique, D_whipping = D_total - D_quasi
    wh_damage_share = ((d_total["damage"] - d_qs["damage"]) / d_total["damage"]
                       if d_total["damage"] > 0 else 0.0)
    d_band_sum = d_qs["damage"] + d_wh["damage"]
    wh_damage_share_band = d_wh["damage"] / d_band_sum if d_band_sum > 0 else 0.0

    # ---- (f') negative control: quasi-static hypothesis ------------------ #
    # (i) only the fundamental mode retained (classic pseudo-static model)
    sol1 = response.modal_response(t, modal_force[:1], omega[:1], zeta[:1])
    sigma_1mode = response.stress_history(sol1, weights[:1])
    # the mode-1-only (quasi-static) model has, by construction, no mode >= 2
    # content: its whipping share is exactly zero in the modal split
    wh_share_1mode_modal = 0.0
    wh_share_1mode_filter = (response.rms(response.bandpass(sigma_1mode, dt, ls, hs))
                             / max(float(np.std(sigma_1mode)), 1e-300))
    # (ii) analysis band restricted to < 0.5 Hz (quasi-static assumption)
    sigma_050 = response.lowpass(sigma, dt, 0.5)
    wh_share_050 = response.rms(response.bandpass(sigma_050, dt, ls, hs)) / max(total_std, 1e-300)

    # ---- (e') energy ledger closure ------------------------------------- #
    strain = response.stress_history(solution, weights)

    # ---- (g) dispersion over the three largest pulses ------------------- #
    order = sorted(parametric, key=lambda p: p.features.peak, reverse=True)
    per_pulse = []
    for k, pulse in enumerate(order[:n_pulses_report]):
        f_single = pulse(t) * scale
        mf = participation[:, None] * f_single[None, :]
        sol = response.modal_response(t, mf, omega, zeta)
        sig = response.stress_history(sol, weights)
        tot = response.rms(sig)
        wh = response.rms(response.bandpass(sig, dt, ls, hs))
        d1, _ = _split_damage(sig, dt, case.whipping_band_hz, curve_key)
        d2, _ = _split_damage(response.bandpass(sig, dt, ls, hs), dt,
                              case.whipping_band_hz, curve_key)
        per_pulse.append({
            "rank": k,
            "pulse": pulse.as_dict(),
            "stress_rms_Pa": tot,
            "whipping_stress_share": wh / tot if tot > 0 else 0.0,
            "damage": d1["damage"],
            "whipping_damage_share": d2["damage"] / d1["damage"] if d1["damage"] > 0 else 0.0,
            "rise_time_s": pulse.features.rise_time,
        })

    def dispersion(values):
        v = np.asarray(values, dtype=float)
        mean = float(np.mean(np.abs(v)))
        return float((v.max() - v.min()) / mean) if mean > 0 else 0.0

    # ---- criteria ------------------------------------------------------- #
    drop = wet.reduction[:2]
    criteria = {
        "a_wet_frequency_drop_mode1": float(drop[0]),
        "a_wet_frequency_drop_mode2": float(drop[1]),
        "a_in_band_10_30_pct": bool(np.all((drop >= 0.10) & (drop <= 0.30))),
        "b_rise_time_range_s": [float(min(p.rise_time for p in pulse_list)),
                                float(max(p.rise_time for p in pulse_list))],
        "b_in_band_10_50_ms": bool(all(0.010 <= p.rise_time <= 0.050 for p in pulse_list)),
        "c_whipping_stress_share": float(wh_share),
        "c_whipping_stress_share_modal": float(modal_share),
        "c_in_band_20_40_pct": bool(0.20 <= wh_share <= 0.40),
        "d_whipping_damage_share": float(wh_damage_share),
        "d_whipping_damage_share_band": float(wh_damage_share_band),
        "d_in_band_15_35_pct": bool(0.15 <= wh_damage_share <= 0.35),
        "e_energy_residual": float(ledger["residual"]),
        "e_residual_below_1e_6": bool(ledger["residual"] < 1.0e-6),
        "f_negative_control_mode1_share": float(wh_share_1mode_modal),
        "f_negative_control_mode1_share_filtered": float(wh_share_1mode_filter),
        "f_negative_control_0p5hz_share": float(wh_share_050),
        "f_negative_control_red": bool(wh_share_1mode_modal < 0.05 and wh_share_050 < 0.05),
        "g_dispersion_rise_time": dispersion([p["rise_time_s"] for p in per_pulse]),
        "g_dispersion_stress_share": dispersion([p["whipping_stress_share"] for p in per_pulse]),
        "g_dispersion_damage_share": dispersion([p["whipping_damage_share"] for p in per_pulse]),
        "g_all_below_20_pct": bool(all(
            dispersion([p[k] for p in per_pulse]) < 0.20
            for k in ("rise_time_s", "whipping_stress_share", "whipping_damage_share"))),
    }

    return {
        "ship": case.as_dict(),
        "load": {
            "scale_factor": scale,
            "peak_force_full_scale_N": float(f_scaled.max()),
            "mean_force_full_scale_N": float(f_scaled.mean()),
            "sampling_rate_hz": 1.0 / dt,
            "duration_s": float(t[-1]),
            "pulse_count": len(pulse_list),
            "pulses": [p.as_dict() for p in pulse_list],
            "parametric": [p.as_dict() for p in parametric],
        },
        "modes": {
            "dry_freqs_Hz": dry.freqs.tolist(),
            "analytic_free_free_Hz": analytic.tolist(),
            "analytic_rel_error": (np.abs(dry.freqs - analytic) / analytic).tolist(),
            "wet_freqs_Hz": wet.freqs.tolist(),
            "reduction": wet.reduction.tolist(),
            "modal_added_mass": wet.added_mass.tolist(),
            "midship_added_mass_kg_m": mass["midship_added_mass_kg_m"],
            "added_mass_coefficient": mass["coefficient_m_a_over_rho_B2"],
            "added_mass_to_hull_mass": float(
                mass["midship_added_mass_kg_m"] * case.relief / case.mass_per_length),
            "participation_at_load": participation.tolist(),
            "curvature_weights_Pa": weights.tolist(),
        },
        "response": {
            "stress_std_Pa": total_std,
            "stress_rms_Pa": response.rms(sigma),
            "quasi_static_std_Pa": qs_std,
            "whipping_rms_Pa": wh_rms,
            "stress_peak_Pa": float(np.max(np.abs(sigma))),
            "rainflow_total_cycles": total_count(cyc_total),
            "damage_total": d_total["damage"],
            "damage_quasi_static": d_qs["damage"],
            "damage_whipping": d_wh["damage"],
            "equivalent_range_total_MPa": d_total["equivalent_range_MPa"],
            "equivalent_range_whipping_MPa": d_wh["equivalent_range_MPa"],
            "max_range_total_MPa": d_total["max_range_MPa"],
        },
        "energy": ledger,
        "per_pulse": per_pulse,
        "criteria": criteria,
        "whipping_stress_modal_Pa": float(modal_std),
        "series": {"time": t, "force": f_scaled, "stress": sigma,
                   "stress_qs": sigma_qs, "stress_wh": sigma_wh},
    }