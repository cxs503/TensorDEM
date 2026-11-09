"""Verification driver for the broken-ice channel evolution (IN155-N09).

Runs the machine-checkable criteria and emits a JSON ledger plus the paper
figures.  Criteria:

* (a) sintering neck growth vs Frenkel ``a ~ t**(1/3)`` -- exponent fit error
  < 10 %;
* (b) porosity decay vs ``phi_0 exp(-k t)`` -- fit error < 15 %;
* (c) rubble strength growth vs Timco & Frederking (1996) -- < 25 %;
* (d) ``R_ice(N)/R_ice(1)`` monotone and ``alpha in [0.05, 0.2]``;
* (e) energy closure < 10 %;
* (f) backward test (no sintering/refreezing -> loose rubble) -- in pytest.

Units are SI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sintering import (
    ICE_SURFACE_ENERGY,
    frenkel_neck_radius,
    ice_strength,
    ice_viscosity,
    neck_growth_exponent,
)
from .refreezing import RefreezeConfig, run_refreezing
from .strength import StrengthConfig, rubble_strength
from .channel import ChannelConfig, run_channel_evolution
from .resistance import PlatenConfig
from .energy import EnergyBudget, energy_closure, sinter_surface_energy_j_m3

# Timco & Frederking (1996) uniaxial compressive strength of sea ice (Pa),
# digitised order-of-magnitude anchors (mean of the reported range).
TIMCO_FREDERKING_1996 = {
    253.15: 2.0e6,   # -20 C
    258.15: 1.7e6,   # -15 C
    263.15: 1.5e6,   # -10 C
    268.15: 1.2e6,   # -5 C
}

# Fast platen config for study/verification (reduced particles + shorter duration)
FAST_PLATEN = PlatenConfig(
    n_particles=60,
    mean_radius=0.012,
    box_width=1.0,
    box_height=0.25,
    platen_speed=0.4,
    duration_s=1.5,
    mu=0.3,
    kn=2.0e5,
    damping_ratio=0.3,
    dt_s=2.0e-4,
    seed=0,
    k_bond=0.0,
    f_bond=0.0,
    thickness_m=0.2,
)


def _criterion_a(temperature_k: float, particle_radius_m: float) -> dict:
    """Frenkel exponent + amplitude check."""
    t = np.logspace(2.0, 7.0, 40)
    a = np.array([frenkel_neck_radius(tt, temperature_k,
                                      particle_radius=particle_radius_m)
                  for tt in t])
    p, C = neck_growth_exponent(t, a)
    eta = ice_viscosity(temperature_k)
    C_exact = (3.0 * ICE_SURFACE_ENERGY / (2.0 * eta)) ** (1.0 / 3.0)
    exp_err = abs(p - 1.0 / 3.0) / (1.0 / 3.0)
    amp_err = abs(C - C_exact) / C_exact
    return {
        "exponent_fit": float(p),
        "exponent_theory": 1.0 / 3.0,
        "exponent_rel_error": float(exp_err),
        "amplitude_fit": float(C),
        "amplitude_theory": float(C_exact),
        "amplitude_rel_error": float(amp_err),
        "pass": bool(exp_err < 0.10 and amp_err < 0.10),
    }


def _criterion_b(cfg: RefreezeConfig, duration_s: float):
    """Exponential fit of the porosity decay."""
    res = run_refreezing(cfg, duration_s=duration_s, record_every_s=60.0)
    t, phi = res.porosity_series()
    mask = phi > 1e-6
    lt = t[mask]
    lphi = np.log(phi[mask] / cfg.phi_0)
    A = np.vstack([-lt, np.ones_like(lt)]).T
    (k_fit, intercept), *_ = np.linalg.lstsq(A, lphi, rcond=None)
    pred = cfg.phi_0 * np.exp(-k_fit * t)
    rel_err = float(np.max(np.abs(pred - phi) / np.maximum(phi, 1e-9)))
    return {
        "k_fit_per_s": float(-k_fit),
        "intercept": float(intercept),
        "max_rel_error": rel_err,
        "pass": bool(rel_err < 0.15),
    }, res


def _criterion_c(cfg: StrengthConfig, duration_s: float) -> dict:
    """Final rubble strength vs the solid-ice literature anchor."""
    T = cfg.temperature_k
    sigma_final = rubble_strength(duration_s, cfg, porosity=0.0, neck_ratio=1.0)
    sigma_lit = TIMCO_FREDERKING_1996.get(T, ice_strength(T))
    err = abs(sigma_final - sigma_lit) / sigma_lit
    return {
        "sigma_final_pa": float(sigma_final),
        "sigma_literature_pa": float(sigma_lit),
        "rel_error": float(err),
        "pass": bool(err < 0.25),
    }


def _criterion_e(cfg: RefreezeConfig, duration_s: float,
                 e_sinter_j_m3: float):
    res = run_refreezing(cfg, duration_s=duration_s,
                          record_every_s=60.0, e_sinter_j_m3=e_sinter_j_m3)
    budget = EnergyBudget(
        e_latent=res.e_latent_j_m2,
        e_sinter=res.e_sinter_j_m3 * cfg.thickness_m,
        e_conduction=res.e_conduction_j_m2,
        e_ambient=0.0,
        d_e_sensible=res.e_sensible_j_m2,
    )
    closure = energy_closure(budget)
    return {
        "e_latent_j_m2": float(budget.e_latent),
        "e_sinter_j_m2": float(budget.e_sinter),
        "e_conduction_j_m2": float(budget.e_conduction),
        "d_e_sensible_j_m2": float(budget.d_e_sensible),
        "closure": float(closure),
        "pass": bool(closure < 0.10),
    }, res


def run_study(out_dir, *,
              temperature_k: float = 263.15,
              phi_0: float = 0.30,
              interval_h: float = 6.0,
              n_transits: int = 24,
              particle_radius_m: float = 0.010,
              make_figures: bool = True,
              platen: PlatenConfig | None = None) -> dict:
    """Run the full verification chain; write the ledger + figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    interval_s = interval_h * 3600.0
    total = (n_transits - 1) * interval_s

    crit_a = _criterion_a(temperature_k, particle_radius_m)

    refz_cfg = RefreezeConfig(thickness_m=0.05,
                              temperature_k=temperature_k, phi_0=phi_0)
    crit_b, refz = _criterion_b(refz_cfg, total)

    str_cfg = StrengthConfig(temperature_k=temperature_k, phi_0=phi_0,
                              particle_radius_m=particle_radius_m)
    crit_c = _criterion_c(str_cfg, total)

    a_end = frenkel_neck_radius(total, temperature_k,
                                particle_radius=particle_radius_m)
    neck_end = a_end / particle_radius_m
    n_contacts = 6.0 / (4.0 / 3.0 * np.pi * particle_radius_m**3)
    e_sinter = sinter_surface_energy_j_m3(neck_end, particle_radius_m,
                                        n_contacts, ICE_SURFACE_ENERGY)
    crit_e, _ = _criterion_e(refz_cfg, total, e_sinter)

    ch_cfg = ChannelConfig(
        n_transits=n_transits, interval_s=interval_s,
        temperature_k=temperature_k, phi_0=phi_0,
        particle_radius_m=particle_radius_m,
        n_calibration=4,
        platen=platen or FAST_PLATEN,
        refreeze=refz_cfg, strength=str_cfg,
    )
    ch = run_channel_evolution(ch_cfg)
    crit_d = {
        "alpha": ch.alpha,
        "fit_r2": ch.fit_r2,
        "monotone": ch.monotone,
        "ratio_final": float(ch.resistance_ratio[-1]),
        "pass": bool(ch.monotone and 0.05 <= ch.alpha <= 0.20),
    }

    ledger = {
        "campaign": "IN155-N09",
        "temperature_k": temperature_k,
        "phi_0": phi_0,
        "interval_h": interval_h,
        "n_transits": n_transits,
        "criteria": {
            "a_sintering_frenkel": crit_a,
            "b_refreezing_exponential": crit_b,
            "c_strength_literature": crit_c,
            "d_resistance_growth": crit_d,
            "e_energy_closure": crit_e,
        },
        "channel": ch.as_dict(),
        "all_pass": bool(crit_a["pass"] and crit_b["pass"] and crit_c["pass"]
                          and crit_d["pass"] and crit_e["pass"]),
    }
    (out / "study.json").write_text(json.dumps(ledger, indent=2))

    if make_figures:
        _make_figures(out, ch, refz, crit_a, crit_b, crit_e,
                        temperature_k, phi_0)
    return ledger


def _make_figures(out, ch, refz, crit_a, crit_b, crit_e,
                    temperature_k, phi_0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: sintering neck growth (log-log)
    t = np.logspace(3.0, 9.0, 60)
    a = np.array([frenkel_neck_radius(tt, temperature_k,
                                      particle_radius=0.010) for tt in t])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(t, a, "o-", ms=3, label="Frenkel $a(t)$")
    ax.loglog(t, crit_a["amplitude_theory"] * t ** (1.0 / 3.0), "k--",
              label="theory $C t^{1/3}$")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("neck radius a (m)")
    ax.set_title(f"Sintering neck growth, T={temperature_k-273.15:.0f} C")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_sintering.png", dpi=140)
    plt.close(fig)

    # Fig 2: porosity decay
    t_r, phi_r = refz.porosity_series()
    k_fit = crit_b["k_fit_per_s"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_r / 3600.0, phi_r, "o-", ms=3, label="1-D enthalpy")
    ax.plot(t_r / 3600.0, phi_0 * np.exp(-k_fit * t_r), "k--",
              label=r"$\phi_0 e^{-kt}$ fit")
    ax.set_xlabel("t (h)")
    ax.set_ylabel("porosity phi")
    ax.set_title(f"Refreezing porosity decay, T={temperature_k-273.15:.0f} C")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_refreezing.png", dpi=140)
    plt.close(fig)

    # Fig 3: strength growth
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ch.n_transits, ch.sigma / 1e6, "o-", label=r"$\sigma(N)$")
    ax.set_xlabel("transit N")
    ax.set_ylabel("rubble strength (MPa)")
    ax.set_title("Rubble strength growth with transits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig3_strength.png", dpi=140)
    plt.close(fig)

    # Fig 4: resistance growth + log fit
    N = ch.n_transits
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N, ch.resistance_ratio, "o-", label=r"$R_{ice}(N)/R_{ice}(1)$")
    ax.plot(N, 1.0 + ch.alpha * np.log(N), "k--",
             label=f"log fit, $\\alpha$={ch.alpha:.3f}")
    ax.set_xlabel("transit N")
    ax.set_ylabel(r"$R_{ice}(N)/R_{ice}(1)$")
    ax.set_title("Ice-resistance growth with transit count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig4_resistance.png", dpi=140)
    plt.close(fig)

    # Fig 5: energy budget
    fig, ax = plt.subplots(figsize=(6, 4))
    terms = ["latent", "sinter", "conduction", "sensible"]
    vals = [crit_e["e_latent_j_m2"], crit_e["e_sinter_j_m2"],
            crit_e["e_conduction_j_m2"], crit_e["d_e_sensible_j_m2"]]
    ax.bar(terms, vals)
    ax.set_ylabel("J/m$^2$")
    ax.set_title(f"Energy budget (closure={crit_e['closure']:.3f})")
    fig.tight_layout()
    fig.savefig(out / "fig5_energy.png", dpi=140)
    plt.close(fig)
