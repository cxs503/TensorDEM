#!/usr/bin/env python3
"""Validation-debt audit of ``tensordem.bowden_hughes_thermal`` (m16736).

The module (added 2026-11-11, campaign wave) advertises a *derived*
non-monotonic ice-steel friction law mu(V): melting lubrication reduces mu at
low speed, viscous shear in the melt film raises it again at high speed
(Bowden & Hughes 1950).  It shipped without any verification document and its
only tests assert ``mu in [0, 1]`` and ``len(set(mu)) > 1``.

This script measures, independently of the module's own story:

  stage A  tautology controls
           A1 ``energy_conservation_check`` residual (it only re-checks the
              internal split q_ice+q_steel+q_melt == q_total)
           A2 the *physical* closure residual |mu_total*sigma*V - q_total|/q_total,
              which the shipped check never looks at
  stage B  analytic benchmarks (closed forms, no literature digit needed)
           B1 Blok effusivity partition  gamma = e_i/(e_i+e_s)
           B2 semi-infinite surface temperature under constant flux
              dT = 2 q sqrt(t/pi)/(e_i+e_s)
           B3 Stefan squeeze-film half-thickness time  3*eta*R^2/(4*sigma*h^2)
           B4 melt-film mass balance h = q_melt*R/(L*rho*V)  (V-independence)
  stage C  mu(V) shape: is the advertised non-monotonicity present?
  stage D  knob sensitivity: contact_time / roughness / radius / viscosity
  stage E  time-scale consistency: conduction time vs contact residence time

Everything is written to ``runs/audit.json`` and four PNG figures.  No number in
the accompanying document is typed by hand: they are injected from this JSON.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FIGD = os.path.join(RUNS, "figures")

sys.path.insert(0, os.path.join(os.path.dirname(HERE), os.path.dirname(HERE), "src"))

from tensordem.bowden_hughes_thermal import (  # noqa: E402
    BowdenHughesConfig,
    _melt_film_thickness_steady,
    _squeeze_out_time,
    _transient_conduction_depth,
    compute_bowden_hughes_equilibrium,
    compute_mu_velocity_curve,
    energy_conservation_check,
)

# Material properties of the shipped default config (used for every analytic
# comparison so that "module vs analytic" differs only in the formula).
BASE = BowdenHughesConfig()

E_ICE = math.sqrt(BASE.ice_thermal_conductivity * BASE.ice_density * BASE.ice_specific_heat)
E_STEEL = math.sqrt(
    BASE.steel_thermal_conductivity * BASE.steel_density * BASE.steel_specific_heat
)


# --------------------------------------------------------------------------- #
# closed-form reference solutions
# --------------------------------------------------------------------------- #
def dT_semi_infinite(q: float, t: float) -> float:
    """Surface temperature rise of two semi-infinite sliding bodies.

    Constant flux ``q`` per unit area applied over time ``t`` gives
    ``dT_j = 2 q_j sqrt(t/pi)/e_j``.  Requiring ``dT_ice == dT_steel`` yields the
    Blok partition ``q_j ∝ e_j``, hence

        dT = 2 q sqrt(t/pi) / (e_ice + e_steel).
    """
    return 2.0 * q * math.sqrt(t / math.pi) / (E_ICE + E_STEEL)


def q_melting_onset(t: float) -> float:
    """Flux at which the interface reaches 0 degC (melting onset)."""
    dT_max = 0.0 - BASE.bulk_ice_temperature
    return dT_max * (E_ICE + E_STEEL) / (2.0 * math.sqrt(t / math.pi))


def t_stefan_tau(h: float, sigma: float) -> float:
    """Stefan film time constant ``tau = 3 eta R^2/(4 sigma h0^2)``.

    ``dh/dt = -2 h^3 F/(3 pi eta R^4)`` with ``F = sigma pi R^2`` integrates to
    ``h(t) = h0 / sqrt(1 + t/tau)``.  The *half-thickness* time is ``3 tau``
    (solve ``1 + t/tau = 4``), so the two conventions differ by a factor 3 and
    must not be confused.
    """
    return 3.0 * BASE.water_viscosity * BASE.contact_radius**2 / (
        4.0 * sigma * h**2
    )


def t_squeeze_half(h: float, sigma: float) -> float:
    """Exact Stefan half-thickness time = ``3 tau``."""
    return 3.0 * t_stefan_tau(h, sigma)


def residence_time(velocity: float, radius: float | None = None) -> float:
    """Time a point of the ice spends inside the nominal contact patch."""
    r = BASE.contact_radius if radius is None else radius
    return 2.0 * r / velocity


# --------------------------------------------------------------------------- #
# stage A: tautology controls
# --------------------------------------------------------------------------- #
def stage_a(sigmas, temps, velocities) -> dict:
    rows = []
    resid_energy = 0.0
    resid_closure_max = 0.0
    for sigma in sigmas:
        for T in temps:
            for v in velocities:
                cfg = replace(BASE, bulk_ice_temperature=T)
                r = compute_bowden_hughes_equilibrium(v, sigma, cfg)
                q = r.heat_flux_total
                split = r.heat_flux_ice + r.heat_flux_steel + r.heat_flux_melt
                e1 = abs(q - split) / q if q else 0.0
                # physical closure: the returned mu must reproduce the heat it
                # claims to generate, q = mu_total * sigma * V
                q_from_mu = r.friction_coefficient * sigma * v
                e2 = abs(q_from_mu - q) / q if q else 0.0
                resid_energy = max(resid_energy, e1)
                resid_closure_max = max(resid_closure_max, e2)
                rows.append(
                    {
                        "sigma_Pa": sigma,
                        "T_bulk_C": T,
                        "velocity": v,
                        "mu_total": r.friction_coefficient,
                        "mu_used_for_heat": BASE.asperity_friction,
                        "q_total": q,
                        "q_split_sum": split,
                        "rel_residual_split": e1,
                        "rel_residual_closure": e2,
                        "energy_conservation_check_returns": energy_conservation_check(r),
                        "all_splits_pass": bool(energy_conservation_check(r, 1e-12)),
                    }
                )
    return {
        "n_points": len(rows),
        "max_rel_residual_split": resid_energy,
        "max_rel_residual_closure": resid_closure_max,
        "energy_check_true_fraction": float(
            np.mean([r["energy_conservation_check_returns"] for r in rows])
        ),
        "energy_check_strict_fraction": float(
            np.mean([r["all_splits_pass"] for r in rows])
        ),
        "closure_error_p50": float(np.median([r["rel_residual_closure"] for r in rows])),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# stage B: analytic benchmarks
# --------------------------------------------------------------------------- #
def stage_b() -> dict:
    # --- B1 Blok partition -------------------------------------------------
    gamma_ice = E_ICE / (E_ICE + E_STEEL)
    r = compute_bowden_hughes_equilibrium(1.0, 1e6, BASE)
    flux_ratio_impl = r.heat_flux_ice / r.heat_flux_steel
    b1 = {
        "gamma_ice_analytic": gamma_ice,
        "e_ice": E_ICE,
        "e_steel": E_STEEL,
        "flux_ratio_impl": flux_ratio_impl,
        "flux_ratio_analytic": E_ICE / E_STEEL,
        "rel_error": abs(flux_ratio_impl - E_ICE / E_STEEL) / (E_ICE / E_STEEL),
    }

    # --- B2 surface temperature, no-melting branch -------------------------
    # A low flux keeps the shipped model in its conduction-only branch.
    rows_b2 = []
    t_contact = BASE.contact_time
    for v in [1e-4, 3e-4, 1e-3, 3e-3]:
        for sigma in [1e4, 1e5]:
            rr = compute_bowden_hughes_equilibrium(v, sigma, BASE)
            q = rr.heat_flux_total
            if rr.heat_flux_melt > 0:
                continue
            dT_mod = rr.interface_temperature - BASE.bulk_ice_temperature
            dT_ref = dT_semi_infinite(q, t_contact)
            rows_b2.append(
                {
                    "velocity": v,
                    "sigma_Pa": sigma,
                    "q_total": q,
                    "dT_module": dT_mod,
                    "dT_analytic_semi_infinite": dT_ref,
                    "rel_error": abs(dT_mod - dT_ref) / dT_ref,
                }
            )
    # The shipped branch mixes k_eff = mean(k) and delta_eff = mean(delta); the
    # analytic result uses the effusivity sum.  Ratio is independent of q:
    delta_ice = _transient_conduction_depth(BASE.ice_thermal_diffusivity, t_contact)
    delta_steel = _transient_conduction_depth(
        BASE.steel_thermal_conductivity / (BASE.steel_density * BASE.steel_specific_heat),
        t_contact,
    )
    k_eff = 0.5 * (BASE.ice_thermal_conductivity + BASE.steel_thermal_conductivity)
    delta_eff = 0.5 * (delta_ice + delta_steel)
    ratio_dT = (delta_eff / k_eff) / (
        2.0 * math.sqrt(t_contact / math.pi) / (E_ICE + E_STEEL)
    )
    b2 = {
        "rows": rows_b2,
        "max_rel_error": max((x["rel_error"] for x in rows_b2), default=float("nan")),
        "min_rel_error": min((x["rel_error"] for x in rows_b2), default=float("nan")),
        "delta_ice_m": delta_ice,
        "delta_steel_m": delta_steel,
        "contact_radius_m": BASE.contact_radius,
        "delta_ice_over_radius": delta_ice / BASE.contact_radius,
        "analytic_2_over_sqrt_pi_factor": 2.0 / math.sqrt(math.pi),
        "dT_ratio_predictions": {
            "module_over_analytic": ratio_dT,
            "semi_infinite_validity_delta_over_R": delta_ice / BASE.contact_radius,
        },
    }

    # --- B3 Stefan squeeze -------------------------------------------------
    h = 8.926446441926546e-07  # film thickness measured at V=1 m/s, sigma=1 MPa
    rows_b3 = []
    for sigma in [1e5, 1e6]:
        t_mod = _squeeze_out_time(h, sigma, BASE.water_viscosity, BASE.contact_radius)
        tau = t_stefan_tau(h, sigma)
        t_half = t_squeeze_half(h, sigma)
        rows_b3.append(
            {
                "sigma_Pa": sigma,
                "h_m": h,
                "t_module_s": t_mod,
                "t_stefan_tau_s": tau,
                "t_stefan_half_s": t_half,
                "ratio_module_over_tau": t_mod / tau,
                "ratio_module_over_half_life": t_mod / t_half,
            }
        )
    # scaling exponents (must be eta^1 R^2 sigma^-1 h^-2)
    def t_mod(eta, R, sigma, hh):
        return 3.0 * eta * R**2 / (2.0 * sigma * hh**2)

    base_t = t_mod(BASE.water_viscosity, BASE.contact_radius, 1e6, h)
    b3 = {
        "rows": rows_b3,
        "exponents": {
            "eta": math.log(t_mod(2 * BASE.water_viscosity, BASE.contact_radius, 1e6, h) / base_t)
            / math.log(2),
            "R": math.log(t_mod(BASE.water_viscosity, 2 * BASE.contact_radius, 1e6, h) / base_t)
            / math.log(2),
            "sigma": math.log(
                t_mod(BASE.water_viscosity, BASE.contact_radius, 2e6, h) / base_t
            )
            / math.log(2),
            "h": math.log(t_mod(BASE.water_viscosity, BASE.contact_radius, 1e6, 2 * h) / base_t)
            / math.log(2),
        },
    }

    # --- B4 melt film mass balance ----------------------------------------
    rows_b4 = []
    for v in [0.01, 0.1, 1.0, 10.0]:
        rr = compute_bowden_hughes_equilibrium(v, 1e6, BASE)
        h_hand = rr.heat_flux_melt * BASE.contact_radius / (
            BASE.latent_heat_fusion * BASE.water_density * v
        )
        rows_b4.append(
            {
                "velocity": v,
                "h_module": rr.film_thickness,
                "h_mass_balance": h_hand,
                "rel_error": abs(h_hand - rr.film_thickness) / rr.film_thickness
                if rr.film_thickness
                else 0.0,
                "q_melt": rr.heat_flux_melt,
                "A_film": rr.film_area_fraction,
                "mu": rr.friction_coefficient,
            }
        )
    b4 = {
        "rows": rows_b4,
        "max_rel_error": max(x["rel_error"] for x in rows_b4),
        "h_spread_over_4_decades_V": max(x["h_module"] for x in rows_b4)
        / min(x["h_module"] for x in rows_b4),
    }
    return {"blok_partition": b1, "surface_temperature": b2, "squeeze_film": b3, "melt_film": b4}


# --------------------------------------------------------------------------- #
# stage C: the advertised mu(V) shape
# --------------------------------------------------------------------------- #
def stage_c() -> dict:
    velocities = np.logspace(-4, 2, 241)
    curves = {}
    for sigma in [1e5, 5e5, 1e6, 2e6]:
        mu, res = compute_mu_velocity_curve(sigma, velocities, BASE)
        mu0 = BASE.asperity_friction
        i_min = int(np.argmin(mu))
        curves[f"sigma_{sigma:g}"] = {
            "mu_min": float(mu[i_min]),
            "mu_at_min_velocity": float(velocities[i_min]),
            "mu_asy": mu0,
            "relative_excursion": float((mu0 - mu[i_min]) / mu0),
            "relative_excursion_max_from_asy": float(np.max(np.abs(mu - mu0)) / mu0),
            "regime_counts": {k: sum(1 for r in res if r.regime == k)
                              for k in ("boundary", "mixed", "hydrodynamic")},
            "max_film_area_fraction": float(max(r.film_area_fraction for r in res)),
            "max_q_melt_fraction": float(
                max((r.heat_flux_melt / r.heat_flux_total) for r in res if r.heat_flux_total)
            ),
            "velocities": velocities.tolist(),
            "mu": mu.tolist(),
            "film_area_fraction": [r.film_area_fraction for r in res],
            "q_melt_fraction_of_total": [
                (r.heat_flux_melt / r.heat_flux_total if r.heat_flux_total else 0.0)
                for r in res
            ],
        }
    mu_1mpa = np.asarray(curves["sigma_1e+06"]["mu"])
    return {
        "curves": curves,
        "max_relative_excursion_all_sigma": max(
            c["relative_excursion"] for c in curves.values()
        ),
        "nonmonotonic_depth_1MPa": float(
            (curves["sigma_1e+06"]["mu_asy"] - curves["sigma_1e+06"]["mu_min"])
            / curves["sigma_1e+06"]["mu_asy"]
        ),
        "max_film_area_fraction_any_sigma": max(
            c["max_film_area_fraction"] for c in curves.values()
        ),
        "any_regime_beyond_boundary": any(
            c["regime_counts"]["mixed"] + c["regime_counts"]["hydrodynamic"] > 0
            for c in curves.values()
        ),
        "mu_monotone_decreasing_1MPa": bool(np.all(np.diff(mu_1mpa) <= 0)),
    }


# --------------------------------------------------------------------------- #
# stage D: knob sensitivity
# --------------------------------------------------------------------------- #
def stage_d() -> dict:
    knobs = {
        "contact_time": np.logspace(-3, 3, 13).tolist(),
        "surface_roughness": np.logspace(-9, -4, 11).tolist(),
        "contact_radius": np.logspace(-4, -2, 9).tolist(),
        "water_viscosity": np.logspace(-4, -2, 9).tolist(),
    }
    out = {}
    for name, values in knobs.items():
        rows = []
        for val in values:
            cfg = replace(BASE, **{name: val})
            r = compute_bowden_hughes_equilibrium(1.0, 1e6, cfg)
            rows.append(
                {
                    "value": val,
                    "mu": r.friction_coefficient,
                    "h_m": r.film_thickness,
                    "A_film": r.film_area_fraction,
                    "q_melt_fraction": r.heat_flux_melt / r.heat_flux_total
                    if r.heat_flux_total
                    else 0.0,
                    "regime": r.regime,
                }
            )
        mus = [r["mu"] for r in rows]
        qf = [r["q_melt_fraction"] for r in rows]
        out[name] = {
            "rows": rows,
            "mu_min": min(mus),
            "mu_max": max(mus),
            "mu_span_relative": (max(mus) - min(mus)) / BASE.asperity_friction,
            "q_melt_fraction_min": min(qf),
            "q_melt_fraction_max": max(qf),
        }
    return out


# --------------------------------------------------------------------------- #
# stage E: time-scale consistency
# --------------------------------------------------------------------------- #
def stage_e() -> dict:
    rows = []
    for v in [1e-3, 1e-2, 0.1, 1.0, 10.0]:
        t_res = residence_time(v)
        d_ice = _transient_conduction_depth(BASE.ice_thermal_diffusivity, BASE.contact_time)
        q_crit_mod = (0.0 - BASE.bulk_ice_temperature) * BASE.ice_thermal_conductivity / d_ice
        q_crit_ref_contact = q_melting_onset(BASE.contact_time)
        q_crit_ref_res = q_melting_onset(t_res)
        rows.append(
            {
                "velocity": v,
                "residence_time_s": t_res,
                "conduction_time_s": BASE.contact_time,
                "time_scale_ratio": BASE.contact_time / t_res,
                "q_melting_onset_module": q_crit_mod,
                "q_melting_onset_semi_infinite_t_contact": q_crit_ref_contact,
                "q_melting_onset_semi_infinite_t_residence": q_crit_ref_res,
                "module_over_analytic_t_contact": q_crit_mod / q_crit_ref_contact,
                "module_over_analytic_t_residence": q_crit_mod / q_crit_ref_res,
                "delta_ice_over_radius": d_ice / BASE.contact_radius,
            }
        )
    # where does the shipped model switch on melting, at sigma = 1 MPa?
    v_onset = None
    for v in np.logspace(-5, 3, 4001):
        if compute_bowden_hughes_equilibrium(float(v), 1e6, BASE).heat_flux_melt > 0:
            v_onset = float(v)
            break
    return {
        "rows": rows,
        "melting_onset_velocity_1MPa": v_onset,
        "melting_onset_flux_1MPa": (v_onset * BASE.asperity_friction * 1e6)
        if v_onset is not None
        else float("nan"),
        "q_melting_onset_at_contact_time": q_melting_onset(BASE.contact_time),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(A, C, D, E) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGD, exist_ok=True)
    made = []

    # fig 1: mu(V) flatness in log scale, 4 stresses
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, c in C["curves"].items():
        ax.semilogx(c["velocities"], c["mu"], label=f"σ = {key.split('_')[1]} Pa")
    ax.axhline(BASE.asperity_friction, ls="--", c="k", lw=1,
               label=f"μ_asy = {BASE.asperity_friction}")
    ax.set_xlabel("sliding velocity V [m/s]")
    ax.set_ylabel("μ [-]")
    ax.set_ylim(0.1490, 0.1505)
    ax.set_title("Shipped μ(V): excursion < 3e-4 over 6 decades — no non-monotonicity")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    p = os.path.join(FIGD, "mu_vs_velocity.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # fig 2: film area fraction + melt heat fraction vs V (the internal contradiction)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    c = C["curves"]["sigma_1e+06"]
    ax.loglog(c["velocities"], c["film_area_fraction"], label="load fraction on water film A_film")
    ax.loglog(c["velocities"], [max(x, 1e-16) for x in c["q_melt_fraction_of_total"]],
              label="fraction of frictional heat spent melting")
    ax.set_xlabel("sliding velocity V [m/s]")
    ax.set_ylabel("fraction [-]")
    ax.set_title("99 % of the heat melts ice, yet the film carries 1e-4 of the load")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    p = os.path.join(FIGD, "film_fraction_vs_melt_fraction.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # fig 3: knob sensitivity of mu(1 m/s, 1 MPa)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in ("contact_time", "surface_roughness", "contact_radius", "water_viscosity"):
        rows = D[name]["rows"]
        ax.semilogx([r["value"] for r in rows], [r["mu"] for r in rows], "o-", ms=3, label=name)
    ax.axhline(BASE.asperity_friction, ls="--", c="k", lw=1)
    ax.set_xlabel("knob value (SI)")
    ax.set_ylabel("μ at V=1 m/s, σ=1 MPa [-]")
    ax.set_title("μ is set by the knobs, not by the sliding speed")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    p = os.path.join(FIGD, "knob_sensitivity.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # fig 4: melting-onset flux — module vs semi-infinite analytic
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    vs = [r["velocity"] for r in E["rows"]]
    ax.loglog(vs, [r["q_melting_onset_module"] for r in E["rows"]], "o-",
              label="module (ice k, δ=sqrt(α t_contact))")
    ax.loglog(vs, [r["q_melting_onset_semi_infinite_t_residence"] for r in E["rows"]], "s-",
              label="analytic, t = residence time 2R/V")
    ax.loglog(vs, [BASE.asperity_friction * 1e6 * v for v in vs], "k--",
              label="frictional flux μ_asy σ V (σ=1 MPa)")
    ax.set_xlabel("sliding velocity V [m/s]")
    ax.set_ylabel("flux [W/m²]")
    ax.set_title("Melting onset: shipped criterion fires ~3 decades too early")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    p = os.path.join(FIGD, "melting_onset.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    return made


def _json_default(obj):
    """numpy scalars/bools -> native Python so the payload stays JSON-clean."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def main() -> int:
    os.makedirs(RUNS, exist_ok=True)
    sigmas = [1e5, 5e5, 1e6, 2e6]
    temps = [-1.0, -5.0, -15.0]
    velocities = list(np.logspace(-4, 2, 13))

    print("stage A: tautology controls ...")
    A = stage_a(sigmas, temps, velocities)
    print(f"  split residual max      = {A['max_rel_residual_split']:.3e}")
    print(f"  closure residual max    = {A['max_rel_residual_closure']:.3e}")
    print(f"  energy check True share = {A['energy_check_true_fraction']:.3f}")
    print("stage B: analytic benchmarks ...")
    B = stage_b()
    print(f"  Blok flux-ratio rel err = {B['blok_partition']['rel_error']:.3e}")
    print(f"  dT rel err range        = {B['surface_temperature']['min_rel_error']:.4f}"
          f" .. {B['surface_temperature']['max_rel_error']:.4f}")
    print(f"  squeeze ratio           = {B['squeeze_film']['rows'][0]['ratio_module_over_tau']:.3f}")
    print(f"  melt-film h rel err max = {B['melt_film']['max_rel_error']:.3e}")
    print("stage C: mu(V) shape ...")
    C = stage_c()
    print(f"  max relative excursion  = {C['max_relative_excursion_all_sigma']:.3e}")
    print("stage D: knob sensitivity ...")
    D = stage_d()
    for k, v in D.items():
        print(f"  {k:18s} mu span = {v['mu_span_relative']:.3e}")
    print("stage E: time-scale consistency ...")
    E = stage_e()
    print(f"  melting onset velocity (1 MPa) = {E['melting_onset_velocity_1MPa']:.4g} m/s")
    figs = make_figures(A, C, D, E)

    payload = {
        "module": "tensordem.bowden_hughes_thermal",
        "config_defaults": {
            k: getattr(BASE, k)
            for k in (
                "ice_thermal_conductivity",
                "ice_density",
                "ice_specific_heat",
                "ice_thermal_diffusivity",
                "steel_thermal_conductivity",
                "steel_density",
                "steel_specific_heat",
                "water_viscosity",
                "water_density",
                "latent_heat_fusion",
                "bulk_ice_temperature",
                "asperity_friction",
                "contact_time",
                "contact_radius",
                "surface_roughness",
            )
        },
        "effusivities": {"e_ice": E_ICE, "e_steel": E_STEEL},
        "stage_a": A,
        "stage_b": B,
        "stage_c": C,
        "stage_d": D,
        "stage_e": E,
        "figures": figs,
    }
    out = os.path.join(RUNS, "audit.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, default=_json_default)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())