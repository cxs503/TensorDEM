"""Density-ledger tests for the threefield fragmentation/melting stage.

Regression m3147: ``fragmentation.fragment()`` hard-coded ``rho_i = 917.0`` and
therefore silently ignored ``BreakupProperties.ice_density_kg_m3``, while the
downstream ``coupling.fragmentation_stage`` converts masses back to floe sizes
with the *configured* density.  Any non-917 ice (snow ice, sediment-laden ice,
freshwater ice) produced masses inconsistent with the rest of the pipeline.
``coupling.follower_ship_impact`` had the same hard-coded density in its
ice-area-equivalent.
"""
from __future__ import annotations

import pytest

import torch

from tensordem.threefield.coupling import follower_ship_impact
from tensordem.threefield.fragmentation import BreakupProperties, fragment

SIZES = (2.0, 3.0, 4.5, 7.0)


def _run(density: float) -> dict:
    props = BreakupProperties(thickness_m=0.2, flexural_strength_pa=5.0e5,
                              ice_density_kg_m3=density)
    return fragment(SIZES, 5.0, props, generator=torch.Generator().manual_seed(3))


@pytest.mark.parametrize("density", [890.0, 910.0, 917.0, 930.0])
def test_masses_follow_configured_density(density: float) -> None:
    result = _run(density)
    thickness = 0.2
    expected = sum(density * length**2 * thickness for length in result["lengths_m"])
    assert result["ice_density_kg_m3"] == pytest.approx(density)
    assert result["final_mass_kg"] == pytest.approx(expected, rel=1e-12)
    assert result["initial_mass_kg"] == pytest.approx(
        sum(density * size**2 * thickness for size in SIZES), rel=1e-12
    )


def test_mass_is_conserved_relative_to_the_configured_density() -> None:
    for density in (890.0, 910.0, 930.0):
        result = _run(density)
        assert abs(result["mass_error_relative"]) < 1e-12


def test_density_scales_the_ledger_linearly() -> None:
    """Changing only the density must scale every mass by exactly rho/917."""
    base = _run(917.0)
    other = _run(950.0)
    assert base["lengths_m"] == pytest.approx(other["lengths_m"])
    ratio = other["initial_mass_kg"] / base["initial_mass_kg"]
    assert ratio == pytest.approx(950.0 / 917.0, rel=1e-12)


def test_follower_impact_uses_configured_density() -> None:
    kwargs = dict(
        initial_mass_kg=1000.0,
        surviving_mass_kg=800.0,
        melted_mass_kg=200.0,
        mean_floe_diameter_m=0.4,
        reference_floe_diameter_m=1.0,
        channel_width_m=20.0,
        ice_thickness_m=0.6,
        track_length_m=1000.0,
    )
    default = follower_ship_impact(**kwargs)
    lighter = follower_ship_impact(ice_density_kg_m3=890.0, **kwargs)
    assert default["ice_area_concentration"] == pytest.approx(
        800.0 / (917.0 * 0.6 * 20.0 * 1000.0), rel=1e-12
    )
    assert lighter["ice_area_concentration"] / default["ice_area_concentration"] == pytest.approx(
        917.0 / 890.0, rel=1e-12
    )
    with pytest.raises(ValueError):
        follower_ship_impact(ice_density_kg_m3=0.0, **kwargs)