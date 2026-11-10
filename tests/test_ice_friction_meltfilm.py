"""Unit + acceptance tests for the melt-film ice/steel friction contact.

Run: ``python -m pytest tests/test_ice_friction_meltfilm.py -q``
"""

from __future__ import annotations

import math

import pytest

from tensordem.ice_friction_meltfilm.physics import (
    MaterialPair,
    effusivity,
    film_shear_friction,
    gw_area_fraction,
    halfspace_conductance,
    reynolds_drainage_velocity,
    water_viscosity,
)
from tensordem.ice_friction_meltfilm.contact import (
    ContactState,
    MeltFilmContact,
)
from tensordem.ice_friction_meltfilm.sweep import SPEEDS, _min_location, _curve


MAT = MaterialPair()


# ---------------------------------------------------------------------------
# closures
# ---------------------------------------------------------------------------
def test_effusivity_values():
    # handbook: ice Ih e ~ 2.0e3, steel e ~ 1.3e4 J/(m^2 K sqrt(s))
    assert effusivity(2.2, 917.0, 2050.0) == pytest.approx(2030.0, rel=0.02)
    assert effusivity(45.0, 7850.0, 470.0) == pytest.approx(1.29e4, rel=0.02)
    # series (Blok) combination is dominated by the ice side
    assert 1600.0 < MAT.effusivity_eff < 1900.0


def test_water_viscosity_reference_point():
    # 1.75e-3 Pa s at 0 degC is the textbook value the VFT fit reproduces
    assert water_viscosity(0.0) == pytest.approx(1.75e-3, rel=0.05)
    # and it rises on cooling
    assert water_viscosity(-10.0) > water_viscosity(0.0)


def test_greenwood_williamson_limits():
    # dry limit must be exactly 1 so that mu -> mu_asp (criterion C1)
    assert gw_area_fraction(0.0) == pytest.approx(1.0, abs=1e-12)
    # monotone decreasing, bounded, -> 0 in the separated limit
    lam = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]
    f = [gw_area_fraction(x) for x in lam]
    assert all(f[i] > f[i + 1] for i in range(len(f) - 1))
    assert f[0] == 1.0 and f[-1] < 0.02
    # the shape is GW's Gaussian tail: at Lambda = 1 the normalised area
    # fraction is 2[phi(1) - (1-Phi(1))]/0.7979 = 0.208
    assert gw_area_fraction(1.0) == pytest.approx(0.208, abs=0.005)


def test_halfspace_conductance_scalings():
    g1 = halfspace_conductance(MAT, 1e-3)
    g2 = halfspace_conductance(MAT, 1e-2)
    # G ~ sqrt(V): a 10x faster slider raises the conductance by sqrt(10)
    assert g2 / g1 == pytest.approx(math.sqrt(10.0), rel=1e-9)
    assert halfspace_conductance(MAT, 0.0) == 0.0


def test_reynolds_drainage_scales_with_h_cubed():
    u1 = reynolds_drainage_velocity(1e-8, 1e6, MAT, 0.0)
    u2 = reynolds_drainage_velocity(2e-8, 1e6, MAT, 0.0)
    assert u2 / u1 == pytest.approx(8.0, rel=1e-9)
    assert reynolds_drainage_velocity(0.0, 1e6, MAT, 0.0) == 0.0


def test_film_shear_is_couette():
    eta = water_viscosity(0.0)
    v, h, sigma = 0.1, 1e-7, 2e5
    assert film_shear_friction(h, v, sigma, 0.0) == pytest.approx(
        eta * v / (h * sigma), rel=1e-12)


# ---------------------------------------------------------------------------
# criteria C1, C2, C7 on the steady solver
# ---------------------------------------------------------------------------
def test_c1_cold_and_slow_limits_are_dry():
    contact = MeltFilmContact(MAT)
    for t_bulk in (-40.0, -15.0, -5.0):
        for v in (1e-5, 1e-4):
            r = contact.solve_steady(v, 0.5e6, t_bulk)
            assert not r.melting
            assert r.h_film == 0.0
            assert abs(r.mu - MAT.mu_asp) / MAT.mu_asp <= 0.02
            assert r.mu_film == 0.0


def test_c2_energy_balance_closes():
    contact = MeltFilmContact(MAT)
    for v in (1e-3, 1e-2, 1e-1, 1.0):
        for sigma in (0.1e6, 0.5e6, 2e6):
            r = contact.solve_steady(v, sigma, -1.0)
            assert r.residual_energy_rel <= 1e-9
            assert r.residual_energy == pytest.approx(0.0, abs=1e-8 * max(r.q_in, 1.0))
            if r.melting:
                # the root is admissible only if the interface is above melting
                assert r.t_interface > MAT.t_melt


def test_c7_negative_control_removes_velocity_dependence():
    hot = MeltFilmContact(MAT, thermal_coupling=False)
    mus = [hot.solve_steady(v, 0.5e6, -1.0).mu for v in SPEEDS]
    spread = (max(mus) - min(mus)) / mus[0]
    assert spread <= 0.01
    assert all(abs(m - MAT.mu_asp) <= 0.02 * MAT.mu_asp for m in mus)


# ---------------------------------------------------------------------------
# mass ledger (C3) and ODE/steady-state consistency
# ---------------------------------------------------------------------------
def test_c3_mass_ledger_closes_and_matches_steady_state():
    contact = MeltFilmContact(MAT)
    v, sigma, t_bulk = 0.1, 0.5e6, -1.0
    steady = contact.solve_steady(v, sigma, t_bulk)
    assert steady.melting and steady.h_film > 0.0
    # march from a dry interface to steady state
    state, ledger = contact.march(ContactState(t_interface=t_bulk), v, sigma,
                                  t_bulk, dt=2e-5, n_steps=4000)
    assert ledger["mass_residual_rel"] <= 0.05
    # the marched film must land on the closed-form steady thickness
    assert state.h_film == pytest.approx(steady.h_film, rel=0.15)
    # and the drainage outflow cannot be ignored without breaking the ledger
    assert ledger["storage_only_residual_rel"] > ledger["mass_residual_rel"]


def test_steady_root_is_unique():
    """Independent check: at the converged h a thicker film over-drains.

    Recomputes drainage and latent supply from the public closures (not from
    the solver's internal residual) so the root is verified from outside.
    """
    contact = MeltFilmContact(MAT)
    for v in (3e-3, 3e-2, 0.3):
        r = contact.solve_steady(v, 0.5e6, -1.0)
        if not r.melting:
            continue
        assert r.iterations > 0
        h = r.h_film * 1.2
        p_real = 0.5e6 / r.real_area_fraction
        drain = MAT.latent_rho * reynolds_drainage_velocity(h, p_real, MAT, 0.0)
        q_supply = r.q_latent
        assert drain > q_supply          # over-drained -> residual positive
        h_thin = r.h_film * 0.8
        drain_thin = MAT.latent_rho * reynolds_drainage_velocity(
            h_thin, 0.5e6 / contact._mu(h_thin, v, 0.5e6, 0.0)[1], MAT, 0.0)
        assert drain_thin < drain        # monotone in h


# ---------------------------------------------------------------------------
# shape criteria (C4/C5/C6) on the campaign curves
# ---------------------------------------------------------------------------
def test_c6_temperature_ordering():
    curves = {(s, t): _curve(MAT, s, t) for s in (0.1e6, 0.5e6, 2e6)
              for t in (-1.0, -5.0, -15.0)}
    for s in (0.1e6, 0.5e6, 2e6):
        for i, v in enumerate(SPEEDS):
            cold = curves[(s, -15.0)][i]["mu"]
            mid = curves[(s, -5.0)][i]["mu"]
            warm = curves[(s, -1.0)][i]["mu"]
            assert cold >= mid - 1e-12 and mid >= warm - 1e-12


def test_melting_onset_lowers_mu():
    """The melt film must *reduce* friction above onset (mechanism sanity)."""
    contact = MeltFilmContact(MAT)
    dry = contact.solve_steady(1e-5, 0.5e6, -1.0)
    wet = contact.solve_steady(1.0, 0.5e6, -1.0)
    assert wet.melting and wet.h_film > 0.0
    assert wet.mu < dry.mu
    assert wet.real_area_fraction < 0.9


def test_min_location_never_extrapolates():
    curve = _curve(MAT, 0.5e6, -1.0)
    loc = _min_location(curve)
    lo = curve[loc["index"] - 1]["V"] if loc["index"] > 0 else curve[0]["V"]
    hi = (curve[loc["index"] + 1]["V"]
          if loc["index"] < len(curve) - 1 else curve[-1]["V"])
    assert lo <= loc["V_star_refined"] <= hi


def test_reported_minimum_outside_campaign_band():
    """Honest negative result: V* sits above 1e-2 m/s for every pressure.

    This is the machine-checked statement behind the C4 honest FAIL; if a
    future closure moves the minimum into [1e-4, 1e-2] this test must be
    updated deliberately (regression guard on the reported conclusion).
    """
    for sigma in (0.1e6, 0.5e6, 2e6):
        loc = _min_location(_curve(MAT, sigma, -1.0))
        assert loc["V_star_refined"] > 1e-2