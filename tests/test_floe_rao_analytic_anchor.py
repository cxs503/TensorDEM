"""CASE-DEM-FLOE-RAO-ANCHOR: analytic gates for ``wave_ice/floe_rao.py``.

Closes the verification debt recorded in ``audit_report.md`` (module graded L2:
"单测只断言峰值存在/有限/<100/单调").  The gates below are *external* to the
module's own coefficient choices:

* thin-disc heave added mass -> (8/3) rho R^3   (Lamb 1932, exact limit),
* sphere heave added mass -> (1/2) displaced mass (exact, same formula),
* the closed form equals an independent adaptive quadrature of the same
  Lamb integral,
* heave natural period of a thin floe scales as R^(1/2) at fixed thickness
  (legacy displaced-volume law gives slope 0 -> falsified),
* long-wave heave RAO -> 1 for every added-mass law,
* the legacy default is bit-for-bit unchanged.

Numbers are re-derived here; the benchmark
``benchmarks/case_floe_rao_thin_disc_anchor.py`` writes the full evidence sheet.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.wave_ice.floe_rao import (
    THIN_DISC_ADDED_MASS_FACTOR,
    FloeRAOConfig,
    _hydrostatic_restoring_heave,
    compute_floe_rao_analytic,
    heave_added_mass,
    oblate_spheroid_axial_added_mass,
    thin_disc_added_mass,
)

RHO_W = 1025.0
RHO_I = 917.0
G = 9.81


def _lamb_quadrature(radius: float, half_thickness: float, rho: float = RHO_W) -> float:
    """Independent route to the Lamb §114 axial coefficient (adaptive quad)."""
    from scipy.integrate import quad

    R, h = float(radius), float(half_thickness)

    def integrand(lam: float) -> float:
        return 1.0 / ((h * h + lam) * math.sqrt((R * R + lam) ** 2 * (h * h + lam)))

    def subst(u: float) -> float:
        if u <= 0.0:
            return 0.0
        lam = h * h * (1.0 / (u * u) - 1.0)
        return integrand(lam) * (-2.0 * h * h / (u * u * u))

    alpha3, _ = quad(subst, 1e-12, 1.0, limit=400)
    vol = (4.0 / 3.0) * math.pi * R * R * h
    return vol * rho * alpha3 / (2.0 - alpha3)


# ---------------------------------------------------------------------------
# A1: thin-disc limit (Lamb 1932)
# ---------------------------------------------------------------------------

#: the limit is approached as O(h/R); measured errors are 2.9 % (h/R=1e-1),
#: 0.59 % (2e-2) and 0.06 % (5e-3) -- see the benchmark evidence sheet.
@pytest.mark.parametrize("aspect,tol", [(0.1, 3.5e-2), (0.02, 1e-2), (0.005, 2e-3)])
def test_thin_disc_limit_reached(aspect: float, tol: float) -> None:
    """A33/(rho R^3) -> 8/3 as h/R -> 0."""
    R, h = 1.0, aspect
    combined = oblate_spheroid_axial_added_mass(R, h, RHO_W) / (RHO_W * R**3)
    target = THIN_DISC_ADDED_MASS_FACTOR
    assert abs(combined - target) / target < tol


def test_thin_disc_helper_matches_documented_constant() -> None:
    assert THIN_DISC_ADDED_MASS_FACTOR == pytest.approx(8.0 / 3.0)
    assert thin_disc_added_mass(20.0, RHO_W) == pytest.approx(
        (8.0 / 3.0) * RHO_W * 20.0**3, rel=1e-15)


def test_thin_limit_continuity() -> None:
    """The spheroid closed form must not jump at the h/R -> 0 switch."""
    R = 1.0
    tiny = oblate_spheroid_axial_added_mass(R, 1e-6 * R, RHO_W)
    analytic = thin_disc_added_mass(R, RHO_W)
    assert abs(tiny - analytic) / analytic < 1e-6


# ---------------------------------------------------------------------------
# A2: closed form vs independent quadrature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aspect", [0.5, 0.2, 0.05, 0.01])
def test_closed_form_matches_quadrature(aspect: float) -> None:
    R, h = 1.0, aspect
    closed = oblate_spheroid_axial_added_mass(R, h, RHO_W)
    quad = _lamb_quadrature(R, h, RHO_W)
    assert abs(closed - quad) / quad < 1e-3


def test_sphere_endpoint_is_half_displaced_mass() -> None:
    """h = R is a sphere: A33 = (1/2) * displaced mass (classical exact value)."""
    R = 1.0
    vol = (4.0 / 3.0) * math.pi * R**3
    assert oblate_spheroid_axial_added_mass(R, R, RHO_W) == pytest.approx(
        0.5 * RHO_W * vol, rel=1e-10)


def test_prolate_rejected() -> None:
    with pytest.raises(ValueError):
        oblate_spheroid_axial_added_mass(1.0, 1.5, RHO_W)


# ---------------------------------------------------------------------------
# A3: added-mass law selection and backward compatibility
# ---------------------------------------------------------------------------

def test_default_is_legacy_displaced_volume() -> None:
    cfg = FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0)
    assert cfg.added_mass_model == "displaced"
    R, T = cfg.floe_radius, cfg.floe_thickness
    assert heave_added_mass(cfg, R, T, RHO_W) == pytest.approx(
        RHO_W * math.pi * R**2 * T, rel=1e-15)


def test_lamb_disc_law() -> None:
    cfg = FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0,
                        added_mass_model="lamb_disc")
    R = cfg.floe_radius
    assert heave_added_mass(cfg, R, 1.0, RHO_W) == pytest.approx(
        (8.0 / 3.0) * RHO_W * R**3, rel=1e-14)


def test_coefficient_scales_every_law() -> None:
    for model in ("displaced", "lamb_disc", "spheroid"):
        base = heave_added_mass(FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0,
                                              added_mass_model=model), 20.0, 1.0, RHO_W)
        scaled = heave_added_mass(FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0,
                                                added_mass_model=model,
                                                added_mass_coeff=2.0),
                                  20.0, 1.0, RHO_W)
        assert scaled == pytest.approx(2.0 * base, rel=1e-14)


def test_unknown_law_rejected() -> None:
    with pytest.raises(ValueError):
        heave_added_mass(FloeRAOConfig(added_mass_model="bogus"), 20.0, 1.0, RHO_W)


def test_legacy_law_underpredicts_thin_floes() -> None:
    """For R/T = 20 the classical coefficient is ~17x the displaced-volume one."""
    R, T = 20.0, 1.0
    legacy = heave_added_mass(FloeRAOConfig(floe_radius=R, floe_thickness=T), R, T, RHO_W)
    thin = heave_added_mass(FloeRAOConfig(floe_radius=R, floe_thickness=T,
                                          added_mass_model="spheroid"), R, T, RHO_W)
    assert thin / legacy > 10.0


# ---------------------------------------------------------------------------
# A4/A5: natural period scaling and long-wave RAO limit
# ---------------------------------------------------------------------------

def _natural_period(model: str, R: float, T: float) -> float:
    cfg = FloeRAOConfig(floe_radius=R, floe_thickness=T, added_mass_model=model)
    mass = RHO_I * math.pi * R**2 * T
    A33 = heave_added_mass(cfg, R, T, RHO_W)
    C33 = _hydrostatic_restoring_heave(R, T, RHO_W, G)
    return 2.0 * math.pi * math.sqrt((mass + A33) / C33)


def test_natural_period_scales_with_radius_thin_disc() -> None:
    radii = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    T = 1.0
    y = np.array([_natural_period("spheroid", R, T) for R in radii])
    slope = float(np.polyfit(np.log(radii), np.log(y), 1)[0])
    assert 0.40 < slope < 0.52, slope


def test_natural_period_legacy_is_radius_independent() -> None:
    """Falsified legacy behaviour: with T fixed, Tn does not depend on R."""
    radii = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    T = 1.0
    y = np.array([_natural_period("displaced", R, T) for R in radii])
    slope = float(np.polyfit(np.log(radii), np.log(y), 1)[0])
    assert abs(slope) < 0.02, slope


@pytest.mark.parametrize("model", ["displaced", "lamb_disc", "spheroid"])
def test_long_wave_heave_rao_tends_to_unity(model: str) -> None:
    cfg = FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0,
                        wave_periods=[80.0, 160.0, 320.0, 640.0],
                        added_mass_model=model)
    res = compute_floe_rao_analytic(cfg)
    assert abs(res.rao_heave[-1] - 1.0) < 1e-3
    assert np.all(np.abs(res.rao_heave - 1.0) < 2e-2)


def test_added_mass_law_changes_rao_but_not_default_path() -> None:
    base = FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0,
                         wave_periods=[3.0, 4.0, 6.0, 10.0])
    alt = FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0,
                        wave_periods=[3.0, 4.0, 6.0, 10.0],
                        added_mass_model="spheroid")
    a = compute_floe_rao_analytic(base)
    b = compute_floe_rao_analytic(alt)
    # different added mass -> materially different response at short periods
    assert np.max(np.abs(a.rao_heave - b.rao_heave)) > 1e-2
    # the legacy config reproduces the documented analytic numbers
    assert a.rao_heave[-1] == pytest.approx(0.997143, abs=5e-6)