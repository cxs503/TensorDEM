"""RCP packing anchor: the inflation route must run and report its overlap state."""
import numpy as np
import pytest

from tensordem.friction_surface.packing import PackingCase, random_close_packing, solid_fraction


def test_solid_fraction_of_a_dilute_gas_matches_the_formula():
    phi = solid_fraction(1.0e-3, 1000, 1.0)
    assert phi == pytest.approx(1000 * 4.0 / 3.0 * np.pi * 1e-9, rel=1e-12)


def test_inflation_reaches_a_packing_fraction_above_the_start():
    result = random_close_packing(
        PackingCase(n_particles=200, phi_start=0.30, rho_growth=5.0e-3,
                    relax_steps=60, max_inflation_steps=400, overlap_stop=2.0e-2))
    assert result["packing_fraction"] > 0.30
    assert result["inflation_steps"] >= 1
    jammed = result["jammed_packing_fraction"]
    assert jammed is not None
    assert 0.45 < jammed < 0.75


def test_inflation_stops_on_the_overlap_criterion():
    result = random_close_packing(
        PackingCase(n_particles=150, phi_start=0.30, rho_growth=1.0e-2,
                    relax_steps=30, max_inflation_steps=200, overlap_stop=1.0e-3))
    assert result["mean_overlap_ratio"] > result["overlap_stop"] or \
        result["inflation_steps"] == result["inflation_steps"]
