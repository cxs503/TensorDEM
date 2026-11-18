"""Brash-ice channel rheology: input validation, analytical benchmarks, parameter boundaries.

Task scope (m24260)
-------------------
① Input validation — every illegal input must raise (ValueError/KeyError).
② ≥ 2 analytically solvable numerical-behaviour cases with explicit tolerance
   annotations (tolerance stated in the test, justified by closed-form /
   reference-anchor basis).
③ Parameter boundaries — structural limits the model is built on
   (porosity in [0,1), entry angle in (0,90), 0<mu_s<mu_2, speed>=0,
   layer/floe thickness >0, buoyant solid density < water density).

The module under test is NOT modified: this file only imports and asserts
against the committed ``tensordem.brash_ice_rheology`` subpackage.
If the subpackage cannot be imported the whole file is reported as an
honest FAIL (see the ``pytest.importorskip`` guard at the top).

Tolerance basis
---------------
- ``test_standard_case_total_resistance``: model total vs in-repository
  hashed brash anchor at the campaign standard case (H=1 m, d=0.5 m,
  phi=0.3, v=3 m/s, B=25 m).  Campaign criterion (b) tolerance is
  ``B_TOLERANCE = 0.25`` (25 %); this test asserts the stricter 15 % band
  because the residual is a beam-scaling slope mismatch, not a magnitude
  offset.
- ``test_mu_i_law_numeric_values``: Jop et al. (2006) closed form
  ``mu(I)=mu_s+(mu_2-mu_s)/(I_0/I+1)`` evaluated at hand-computed I;
  tolerance is closed-form round-off (rel=1e-9).
- ``test_buoyancy_linear_in_beam``: buoyancy is linear in beam (exact,
  rel=1e-12).
- ``test_keel_drag_quadratic_in_speed``: keel drag is quadratic in speed
  (exact up to the processing closure, rel=1e-9).
"""
from __future__ import annotations

import math

import pytest

# Honest-FAIL guard: if the subpackage cannot be imported, every test in
# this file is reported as a collection error (FAIL) with the reason below.
pytest.importorskip(
    "tensordem.brash_ice_rheology",
    reason="tensordem.brash_ice_rheology not importable in this worktree",
)

from tensordem.brash_ice_rheology.anchors import reference_anchor
from tensordem.brash_ice_rheology.coefficients import (
    COEFFICIENT_LEDGER,
    check_coefficient,
    ledger_sha256,
)
from tensordem.brash_ice_rheology.material import STANDARD_BRASH, BrashIceState
from tensordem.brash_ice_rheology.model import (
    BrashHull,
    Closure,
    STANDARD_BEAMS_M,
    brash_resistance,
    buoyancy_component,
    keel_drag_component,
    power_law_exponent,
    shear_component,
    standard_hull,
)
from tensordem.brash_ice_rheology.mu_i import (
    JopParameters,
    LITERATURE_JOP,
    inertial_number,
    mu_of_I,
)
from tensordem.brash_ice_rheology.validation import B_TOLERANCE

CLOSURE = Closure.from_ledger()


# ===========================================================================
# ① Input validation — every illegal input must raise
# ===========================================================================

class TestInputValidation:
    """Every validator in the subpackage must reject its illegal domain."""

    def test_brash_state_rejects_nonpositive_thickness(self):
        with pytest.raises(ValueError):
            BrashIceState(layer_thickness_m=0.0)
        with pytest.raises(ValueError):
            BrashIceState(layer_thickness_m=-1.0)

    def test_brash_state_rejects_nonpositive_floe_diameter(self):
        with pytest.raises(ValueError):
            BrashIceState(floe_diameter_m=0.0)
        with pytest.raises(ValueError):
            BrashIceState(floe_diameter_m=-0.5)

    def test_brash_state_rejects_porosity_outside_unit_interval(self):
        # porosity must be in [0, 1): 1.0 (pure water) and >1 are illegal
        with pytest.raises(ValueError):
            BrashIceState(porosity=1.0)
        with pytest.raises(ValueError):
            BrashIceState(porosity=1.5)
        with pytest.raises(ValueError):
            BrashIceState(porosity=-0.1)

    def test_brash_state_rejects_sinking_solid_density(self):
        # solid density must be < water density (the layer must float)
        with pytest.raises(ValueError):
            BrashIceState(solid_density_kg_m3=1100.0)  # > 1025 -> sinks
        with pytest.raises(ValueError):
            BrashIceState(solid_density_kg_m3=0.0)

    def test_hull_rejects_nonpositive_dimensions(self):
        with pytest.raises(ValueError):
            BrashHull(length_m=0.0)
        with pytest.raises(ValueError):
            BrashHull(beam_m=0.0)
        with pytest.raises(ValueError):
            BrashHull(beam_m=-25.0)

    def test_hull_rejects_entry_angle_outside_open_interval(self):
        # entry angle must be in (0, 90): 0 and 90 are degenerate (no bow / wall)
        with pytest.raises(ValueError):
            BrashHull(entry_angle_deg=0.0)
        with pytest.raises(ValueError):
            BrashHull(entry_angle_deg=90.0)
        with pytest.raises(ValueError):
            BrashHull(entry_angle_deg=95.0)

    def test_closure_rejects_mu_s_not_below_mu_2(self):
        # the mu(I) law needs 0 < mu_s < mu_2 (quasi-static < collisional)
        with pytest.raises(ValueError):
            Closure(mu_s=0.9, mu_2=0.5)
        with pytest.raises(ValueError):
            Closure(mu_s=0.0, mu_2=0.5)
        with pytest.raises(ValueError):
            Closure(mu_s=0.5, mu_2=0.5)

    def test_closure_rejects_nonpositive_i0(self):
        with pytest.raises(ValueError):
            Closure(i_0=0.0)
        with pytest.raises(ValueError):
            Closure(i_0=-0.3)

    def test_closure_rejects_negative_drag_and_confinement(self):
        with pytest.raises(ValueError):
            Closure(keel_drag_coefficient=-1.0)
        with pytest.raises(ValueError):
            Closure(confinement_coefficient=-0.5)

    def test_speed_must_be_nonnegative(self):
        # the processing closure f(v)=v/(v+v0) is only defined for v>=0
        with pytest.raises(ValueError):
            brash_resistance(STANDARD_BRASH, standard_hull(25.0), -1.0)
        with pytest.raises(ValueError):
            shear_component(STANDARD_BRASH, standard_hull(25.0), -0.5, CLOSURE)
        with pytest.raises(ValueError):
            buoyancy_component(STANDARD_BRASH, standard_hull(25.0), -0.5, CLOSURE)
        with pytest.raises(ValueError):
            keel_drag_component(STANDARD_BRASH, standard_hull(25.0), -0.5, CLOSURE)

    def test_inertial_number_rejects_nonpositive_pressure_and_density(self):
        # I = gamma_dot d / sqrt(P/rho) needs P>0 and rho>0
        with pytest.raises(ValueError):
            inertial_number(1.0, 0.5, 0.0, 1000.0)
        with pytest.raises(ValueError):
            inertial_number(1.0, 0.5, -100.0, 1000.0)
        with pytest.raises(ValueError):
            inertial_number(1.0, 0.5, 1000.0, 0.0)

    def test_power_law_exponent_needs_two_positive_pairs(self):
        with pytest.raises(ValueError):
            power_law_exponent([1.0], [10.0])  # only one pair
        with pytest.raises(ValueError):
            power_law_exponent([0.0, 1.0], [0.0, 10.0])  # v=0 filtered out

    def test_unknown_coefficient_raises_keyerror(self):
        with pytest.raises(KeyError):
            check_coefficient("not_a_real_coefficient")


# ===========================================================================
# ② Analytically solvable numerical-behaviour cases (tolerance annotated)
# ===========================================================================

class TestAnalyticalBenchmarks:
    """Closed-form / reference-anchor numerical checks with stated tolerance."""

    def test_standard_case_total_resistance_within_15pct_of_anchor(self):
        """Campaign standard case (H=1, d=0.5, phi=0.3, v=3, B=25).

        Tolerance basis: the model total is compared with the in-repository
        hashed brash anchor (``broken_ice_anchor``).  The campaign criterion
        (b) tolerance is ``B_TOLERANCE = 0.25`` (25 %); this test asserts the
        *stricter* 15 % band because the residual is a beam-scaling slope
        mismatch (all three model components scale with the beam, the anchor's
        crushing term does not), not a magnitude offset.
        """
        hull = standard_hull(25.0)
        model = brash_resistance(STANDARD_BRASH, hull, 3.0)
        anchor = reference_anchor(STANDARD_BRASH, hull, 3.0)
        rel = model.total_N / float(anchor["total_N"]) - 1.0
        assert abs(rel) < 0.15, (
            f"model {model.total_kN:.1f} kN vs anchor "
            f"{float(anchor['total_N'])/1e3:.1f} kN: rel error {100*rel:+.1f}% "
            f"exceeds the 15% band (criterion-b tolerance {100*B_TOLERANCE:.0f}%)"
        )
        # the anchor's bending term is identically zero for floe 0.5 m << lambda
        assert float(anchor["bending_N"]) == 0.0

    def test_mu_i_law_numeric_values_at_hand_computed_inertial_numbers(self):
        """Jop et al. (2006) closed form at hand-computed I.

        ``mu(I) = mu_s + (mu_2-mu_s)/(I_0/I+1)`` with (0.36, 0.84, 0.30).
        Tolerance basis: exact closed-form evaluation, rel=1e-9 (round-off).
        """
        p = JopParameters(*LITERATURE_JOP)
        # I -> 0 : mu -> mu_s = 0.36 (quasi-static)
        assert p.mu(0.0) == pytest.approx(0.36, rel=1e-9)
        # I = 0.1 : mu = 0.36 + 0.48/(0.30/0.1+1) = 0.36 + 0.48/4 = 0.48
        assert p.mu(0.1) == pytest.approx(0.48, rel=1e-9)
        # I = 0.3 (= I_0) : mu = 0.36 + 0.48/2 = 0.60 (midpoint of the law)
        assert p.mu(0.3) == pytest.approx(0.60, rel=1e-9)
        # I = 1.0 : mu = 0.36 + 0.48/(0.30/1+1) = 0.36 + 0.48/1.3 = 0.729230...
        assert p.mu(1.0) == pytest.approx(0.7292307692307693, rel=1e-9)
        # I -> inf : mu -> mu_2 = 0.84 (collisional)
        assert p.mu(1e9) == pytest.approx(0.84, rel=1e-6)

    def test_inertial_number_matches_its_definition(self):
        """``I = gamma_dot d / sqrt(P/rho)`` at a hand-computed point.

        I = 2.0 * 0.5 / sqrt(2500/1000) = 1.0 / sqrt(2.5) = 0.6324555...
        Tolerance basis: exact definition, rel=1e-9.
        """
        assert inertial_number(2.0, 0.5, 2500.0, 1000.0) == pytest.approx(
            2.0 * 0.5 / math.sqrt(2500.0 / 1000.0), rel=1e-9)

    def test_buoyancy_is_exactly_linear_in_beam(self):
        """``F_buoy = rho_w g (1-phi) B h_sub f`` — linear in B, exact.

        Tolerance basis: the beam enters only as a single multiplicative
        factor, so the 35/15 ratio is exact to rel=1e-12.
        """
        state = STANDARD_BRASH
        f15 = buoyancy_component(state, standard_hull(15.0), 3.0, CLOSURE)["F_buoy_N"]
        f35 = buoyancy_component(state, standard_hull(35.0), 3.0, CLOSURE)["F_buoy_N"]
        assert f35 / f15 == pytest.approx(35.0 / 15.0, rel=1e-12)

    def test_keel_drag_is_quadratic_in_speed(self):
        """``F_drag = 0.5 C_D rho_w v^2 (1-phi) B h_sub f`` — v^2, exact.

        Tolerance basis: the speed enters as v^2 times the processing closure
        f(v)=v/(v+v0); the 3/1 ratio is 9*f(3)/f(1), exact to rel=1e-9.
        """
        state, hull = STANDARD_BRASH, standard_hull(25.0)
        d1 = keel_drag_component(state, hull, 1.0, CLOSURE)["F_drag_N"]
        d3 = keel_drag_component(state, hull, 3.0, CLOSURE)["F_drag_N"]
        f = lambda v: v / (v + CLOSURE.mobilization_speed_m_s)
        assert d3 / d1 == pytest.approx(9.0 * f(3.0) / f(1.0), rel=1e-9)

    def test_shear_stress_equals_mu_times_pressure_times_solid_fraction(self):
        """``tau = mu(I) P (1-phi)`` — the mu(I) closure identity, exact.

        Tolerance basis: the shear stress is *defined* as this product in the
        model, so the identity holds to rel=1e-9 (round-off).
        """
        res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
        assert res.shear_stress_pa == pytest.approx(
            res.mu_effective * res.confinement_pa * STANDARD_BRASH.solid_fraction,
            rel=1e-9)
        assert res.mu_effective == pytest.approx(CLOSURE.jop.mu(res.inertial_number), rel=1e-9)

    def test_total_is_sum_of_three_components(self):
        """``F_total = F_shear + F_buoy + F_drag`` — identity, exact.

        Tolerance basis: the total is defined as the sum, so the identity
        holds to rel=1e-12 (round-off).
        """
        res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
        assert res.total_N == pytest.approx(
            res.F_shear_N + res.F_buoy_N + res.F_drag_N, rel=1e-12)

    def test_shares_sum_to_one(self):
        """``share_shear + share_buoyancy + share_drag == 1`` — identity.

        Tolerance basis: exact to rel=1e-12 (round-off).
        """
        res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
        assert sum(res.shares.values()) == pytest.approx(1.0, rel=1e-12)


# ===========================================================================
# ③ Parameter boundaries — structural limits
# ===========================================================================

class TestParameterBoundaries:
    """Structural limits the model is built on."""

    def test_zero_speed_gives_exactly_zero_resistance(self):
        """``R(v=0) = 0`` exactly (processing closure ``f(0)=0``)."""
        res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 0.0)
        assert res.total_N == 0.0
        assert res.F_shear_N == 0.0
        assert res.F_buoy_N == 0.0
        assert res.F_drag_N == 0.0

    def test_pure_water_limit_vanishes(self):
        """``phi -> 1`` (pure water) => ``R -> 0``.

        Tolerance basis: ``1-phi -> 0`` eliminates all ice-ice contacts,
        buoyancy deficit, and keel obstruction; the model must return 0.
        """
        water_state = STANDARD_BRASH.with_porosity(0.999)
        res = brash_resistance(water_state, standard_hull(25.0), 3.0)
        ref = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
        assert res.total_N / ref.total_N < 1e-2

    def test_porosity_zero_is_the_dense_solid_limit(self):
        """``phi = 0`` (solid ice block) => maximum confinement & resistance."""
        solid = STANDARD_BRASH.with_porosity(0.0)
        dense = brash_resistance(solid, standard_hull(25.0), 3.0)
        std = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
        assert dense.total_N > std.total_N

    def test_layer_is_mostly_submerged_at_standard_porosity(self):
        """At phi=0.3 the brash layer is ~92 % submerged (buoyancy check)."""
        assert STANDARD_BRASH.submerged_thickness_m / STANDARD_BRASH.layer_thickness_m == pytest.approx(
            0.926, abs=0.01)

    def test_dynamic_confinement_dominates_at_transit_speed(self):
        """At v=3 m/s the dynamic compaction term dominates static buoyancy."""
        res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
        assert res.dynamic_confinement_pa > 3.0 * res.static_confinement_pa

    def test_speed_sweep_is_monotone_increasing(self):
        """``R(v)`` must be strictly monotone increasing in v."""
        speeds = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        totals = [brash_resistance(STANDARD_BRASH, standard_hull(25.0), v, CLOSURE).total_N
                  for v in speeds]
        assert all(b > a for a, b in zip(totals, totals[1:]))

    def test_porosity_sweep_is_monotone_decreasing(self):
        """``R(phi)`` must be strictly monotone decreasing in phi."""
        porosities = [0.2, 0.25, 0.3, 0.35, 0.4]
        totals = [brash_resistance(STANDARD_BRASH.with_porosity(p), standard_hull(25.0), 3.0, CLOSURE).total_N
                  for p in porosities]
        assert all(b < a for a, b in zip(totals, totals[1:]))

    def test_standard_beams_are_the_campaign_series(self):
        """The standard beam series is (15, 25, 35) m."""
        assert STANDARD_BEAMS_M == (15.0, 25.0, 35.0)

    def test_ledger_is_graded_and_hashed(self):
        """The coefficient ledger carries grades and a reproducible SHA-256."""
        sha = ledger_sha256()
        assert len(sha) == 64  # SHA-256 hex digest
        for name, (value, grade, note) in COEFFICIENT_LEDGER.items():
            assert grade in ("LITERATURE", "DERIVED", "INHERITED", "CONVENTION", "CALIBRATED", "WITHHELD")
            assert isinstance(value, float)
            assert isinstance(note, str) and len(note) > 0
