"""Verification debt closure for the level-ice resistance baselines.

Protocol id: ``tensordem-threefield-resistance-audit-v1``

``src/tensordem/threefield/resistance.py`` states in its own provenance caveat
that the coefficient set is "guarded by the structural self-tests in
``tests/test_threefield_resistance.py``".  That module did not exist, so the
caveat pointed at nothing and ``docs/threefield/FORMULAS.md`` (also cited) was
missing.  This file is that guard.

What can and cannot be checked here is stated up front, because the distinction
matters for how the numbers may be cited:

* CHECKED (closed form, no external data): dimensional/scaling exponents of every
  component, the triangular contact-pressure quadrature behind
  ``CRUSHING_EFFICIENCY``, resolution of the friction component on the inclined
  bow face, the internal identities linking the Riska ``C1``/``C2`` coefficients
  to the Lindqvist components at the reference speed ``sqrt(g h)``, the quadratic
  added-drag term, exact partial derivatives, monotonicity, parameter validation,
  and a pinned reproduction table for the three ice thicknesses used by the
  coupling study.
* NOT CHECKED (and therefore graded WITHHELD in the docs): agreement of the
  coefficient set with Lindqvist (1989) / Riska (1989) as published.  The primary
  papers are not available in this offline sandbox.  Nothing in this file
  demonstrates that the baselines reproduce the published predictions; it
  demonstrates that the implementation is self-consistent with the documented
  equations and with stated physical assumptions.

Run with ``python -m pytest tests/test_threefield_resistance.py``.
"""

from __future__ import annotations

import math
import unittest

from tensordem.threefield.resistance import (
    BENDING_EFFICIENCY,
    CRUSHING_EFFICIENCY,
    SUBMERSION_BUOYANCY_COEFFICIENT,
    SUBMERSION_FRICTION_COEFFICIENT,
    Environment,
    HullGeometry,
    IceProperties,
    bending_resistance,
    crushing_resistance,
    lindqvist_resistance,
    relative_error_pct,
    resistance_band,
    riska_resistance,
    submersion_resistance,
)


def quadrature_mean_pressure_factor(n: int = 200001) -> float:
    """Mean (1/sigma_c h) * int_0^h p(y) dy for a triangular pressure profile.

    p(y) = sigma_c * y / h, i.e. zero at the top edge of the ice sheet and
    sigma_c at the bottom edge.  The trapezoidal rule is second-order exact for a
    linear integrand, so this returns 1/2 to machine precision and gives the
    ``CRUSHING_EFFICIENCY`` coefficient an independent derivation.
    """
    h = 1.0
    ys = [h * i / (n - 1) for i in range(n)]
    p = [y / h for y in ys]  # sigma_c factored out
    integral = 0.0
    for i in range(n - 1):
        integral += 0.5 * (p[i] + p[i + 1]) * (ys[i + 1] - ys[i])
    return integral / h  # sigma_c * h / h


def log_log_exponent(fn, base_value: float, factor: float) -> float:
    """Exponent ``a`` in ``fn(x) ~ x**a`` measured by two-point scaling."""
    return math.log(fn(base_value * factor) / fn(base_value)) / math.log(factor)


class ResistanceFixtures(unittest.TestCase):
    def setUp(self) -> None:
        self.ice = IceProperties()
        self.hull = HullGeometry()
        self.env = Environment()
        self.g = self.env.gravity_m_s2
        self.rho = self.env.water_density_kg_m3
        self.mu = self.hull.friction_coefficient

    # ------------------------------------------------------------------ units
    def test_crushing_component_scales_as_stress_times_thickness_squared(self):
        """R_crush is a force: exponent 1 in sigma_c, 2 in h, 0 in B and L."""
        def f(ice: IceProperties = None, hull: HullGeometry = None) -> float:
            return crushing_resistance(ice or self.ice, hull or self.hull)

        sigma_exp = log_log_exponent(
            lambda s: f(IceProperties(compressive_strength_pa=s)), 1.5e6, 4.0
        )
        h_exp = log_log_exponent(lambda h: f(IceProperties(thickness_m=h)), 0.6, 3.0)
        b_exp = log_log_exponent(lambda b: f(hull=HullGeometry(beam_m=b)), 20.0, 2.5)
        l_exp = log_log_exponent(lambda l: f(hull=HullGeometry(length_m=l)), 120.0, 2.0)

        self.assertAlmostEqual(sigma_exp, 1.0, places=12)
        self.assertAlmostEqual(h_exp, 2.0, places=12)
        self.assertAlmostEqual(b_exp, 0.0, places=12)
        self.assertAlmostEqual(l_exp, 0.0, places=12)

    def test_bending_component_scales_as_sigma_flexural_B_h2_over_L(self):
        """R_bend is a force: exponent 1 in sigma_f, 1 in B, 2 in h, -1 in L."""
        def f(sigma=0.5e6, b=20.0, h=0.6, l=120.0) -> float:
            return bending_resistance(
                IceProperties(thickness_m=h, flexural_strength_pa=sigma),
                HullGeometry(beam_m=b, length_m=l),
            )

        self.assertAlmostEqual(log_log_exponent(lambda s: f(sigma=s), 0.5e6, 4.0), 1.0, places=12)
        self.assertAlmostEqual(log_log_exponent(lambda b: f(b=b), 20.0, 2.5), 1.0, places=12)
        self.assertAlmostEqual(log_log_exponent(lambda h: f(h=h), 0.6, 3.0), 2.0, places=12)
        self.assertAlmostEqual(log_log_exponent(lambda l: f(l=l), 120.0, 2.0), -1.0, places=12)

    def test_submersion_component_scales_as_buoyant_volume(self):
        """R_sub is a force: exponent 1 in rho and g, 2 in h, 1 in B.

        The beam exponent is measured with hull friction set to zero, because the
        friction correction (c_f mu L / B) makes the bracket weakly beam-dependent;
        the general B-dependence is pinned separately by
        ``test_submersion_bracket_and_exact_length_derivative``.
        """
        def f(rho=1025.0, g=9.80665, h=0.6, b=20.0) -> float:
            return submersion_resistance(
                IceProperties(thickness_m=h),
                HullGeometry(beam_m=b, friction_coefficient=0.0),
                Environment(water_density_kg_m3=rho, gravity_m_s2=g),
            )

        self.assertAlmostEqual(log_log_exponent(lambda r: f(rho=r), 1025.0, 4.0), 1.0, places=12)
        self.assertAlmostEqual(log_log_exponent(lambda gg: f(g=gg), 9.80665, 4.0), 1.0, places=12)
        self.assertAlmostEqual(log_log_exponent(lambda h: f(h=h), 0.6, 3.0), 2.0, places=12)
        self.assertAlmostEqual(log_log_exponent(lambda b: f(b=b), 20.0, 2.5), 1.0, places=12)

    def test_all_three_components_are_dimensionally_additive(self):
        """The three components are summed directly, so all must be in newtons.

        Guards against reintroducing a per-metre/per-width component into a sum
        that is otherwise a total force.  At the reference geometry each
        component must land in a plausible newtons band for a 120 m ship, and the
        total must equal their exact sum.
        """
        r_c = crushing_resistance(self.ice, self.hull)
        r_b = bending_resistance(self.ice, self.hull)
        r_s = submersion_resistance(self.ice, self.hull, self.env)
        for name, value in (("crushing", r_c), ("bending", r_b), ("submersion", r_s)):
            self.assertGreater(value, 1.0e3, f"{name} implausibly small")
            self.assertLess(value, 1.0e8, f"{name} implausibly large")
        total = lindqvist_resistance(self.ice, self.hull, 0.0, self.env)["total_N"]
        self.assertAlmostEqual(total, r_c + r_b + r_s, delta=1e-9 * total)

    # ------------------------------------------------- crushing assumption
    def test_crushing_efficiency_is_the_triangular_pressure_mean(self):
        """CRUSHING_EFFICIENCY must equal the mean of a triangular pressure field.

        With p linearly growing from zero at the upper ice surface to sigma_c at
        the lower surface, the mean pressure is sigma_c/2, which is exactly the
        implemented coefficient.  A uniform-pressure assumption would give 1.0
        and double the crushing force; this test pins which assumption the code
        encodes.
        """
        self.assertAlmostEqual(quadrature_mean_pressure_factor(), 0.5, places=9)
        self.assertAlmostEqual(CRUSHING_EFFICIENCY, quadrature_mean_pressure_factor(), places=9)

        alpha = math.radians(self.hull.waterline_entry_angle_deg)
        bracket = math.tan(alpha) + self.mu / math.cos(alpha)
        expected = quadrature_mean_pressure_factor() * (
            self.ice.compressive_strength_pa * self.ice.thickness_m**2
        ) * bracket
        self.assertAlmostEqual(
            crushing_resistance(self.ice, self.hull), expected, delta=1e-12 * expected
        )

    def test_crushing_friction_is_resolved_normal_to_the_inclined_face(self):
        """The friction contribution is mu/cos(alpha), not mu*cos(alpha).

        Resolving friction along the inclined bow face gives a geometric factor
        1/cos(alpha); a derivation that resolves it the other way (mu*cos(alpha))
        changes the crushing term by >1 % at a 22 deg entry angle, so this test
        discriminates the two conventions instead of just re-evaluating the code.
        """
        alpha = math.radians(self.hull.waterline_entry_angle_deg)
        scale = CRUSHING_EFFICIENCY * self.ice.compressive_strength_pa * self.ice.thickness_m**2
        implemented = crushing_resistance(self.ice, self.hull) / scale
        normal_convention = math.tan(alpha) + self.mu / math.cos(alpha)
        shallow_convention = math.tan(alpha) + self.mu * math.cos(alpha)

        self.assertAlmostEqual(implemented, normal_convention, places=12)
        self.assertGreater(abs(implemented - shallow_convention) / implemented, 0.01)

    # --------------------------------------------------- Riska C1 / C2 links
    def test_riska_c1_reduces_to_lindqvist_plus_stem_correction(self):
        """C1 - R_lindqvist(V=0) == 0.021 * phi_deg * (R_crush + R_sub) exactly."""
        r_c = crushing_resistance(self.ice, self.hull)
        r_sub = submersion_resistance(self.ice, self.hull, self.env)
        c1 = riska_resistance(self.ice, self.hull, 0.0, self.env)["c1_N"]
        base = lindqvist_resistance(self.ice, self.hull, 0.0, self.env)["total_N"]
        expected = 0.021 * self.hull.stem_angle_deg * (r_c + r_sub)
        self.assertAlmostEqual(c1 - base, expected, delta=1e-12 * abs(expected))

    def test_stem_angle_corrections_use_degrees_not_radians(self):
        """dC1/dphi == 0.021 (R_crush + R_sub) per degree.

        A unit slip to radians would inflate this derivative by 180/pi ~ 57.3, so
        the test also asserts the two conventions are separated by orders of
        magnitude rather than by a rounding margin.
        """
        r_c = crushing_resistance(self.ice, self.hull)
        r_sub = submersion_resistance(self.ice, self.hull, self.env)
        dphi = 1.0
        c1_0 = riska_resistance(self.ice, self.hull, 0.0, self.env)["c1_N"]
        c1_1 = riska_resistance(
            self.ice, HullGeometry(stem_angle_deg=self.hull.stem_angle_deg + dphi), 0.0, self.env
        )["c1_N"]
        measured = c1_1 - c1_0
        expected_degrees = 0.021 * dphi * (r_c + r_sub)
        expected_radians = 0.021 * math.radians(dphi) * (r_c + r_sub)
        self.assertAlmostEqual(measured, expected_degrees, delta=1e-9 * abs(expected_degrees))
        self.assertGreater(measured / expected_radians, 50.0)

    def test_riska_c2_at_reference_speed_recovers_half_the_hydrostatic_friction(self):
        """At V = sqrt(g h) the C2 friction part equals 0.5 mu rho_w g B h^2.

        This is the independent derivation behind the C2 friction coefficient:
        the normal load on the displaced ice is the hydrostatic force of the
mu rho_w g B h^2, and at the shallow-water celerity sqrt(g h) the
        implemented friction term evaluates to ``0.5 (1 + 0.063 phi)`` times that
        load.  The stem-angle correction is part of C2, so the identity is stated
        with it rather than as a bare 0.5.
        """
        for h in (0.30, 0.45, 0.60, 1.00):
            ice = IceProperties(thickness_m=h)
            v_ref = math.sqrt(self.g * h)
            c2 = riska_resistance(ice, self.hull, 0.0, self.env)["c2_N_per_m_s"]
            # friction part of C2, isolated by setting flexural strength to zero
            c2_friction = riska_resistance(
                IceProperties(thickness_m=h, flexural_strength_pa=1e-6),
                self.hull, 0.0, self.env,
            )["c2_N_per_m_s"]
            hydrostatic = self.mu * self.rho * self.g * self.hull.beam_m * h**2
            factor = 0.5 * (1.0 + 0.063 * self.hull.stem_angle_deg)
            self.assertAlmostEqual(c2_friction * v_ref / hydrostatic, factor, places=6)
            self.assertGreater(c2, c2_friction)

    def test_riska_c2_at_reference_speed_recovers_four_thirds_of_static_bending(self):
        """At V = sqrt(g h) the C2 bending part equals (4/3) * R_bend.

        The bending contribution to C2 is sigma_f B h^2 / (3 L sqrt(g h)); at the
reference speed it becomes ``(1 + 0.063 phi) sigma_f B h^2 / (3 L)``, i.e.
        ``(4/3)(1 + 0.063 phi)`` times R_bend.  A coefficient slip on either side
        """
        for h in (0.30, 0.60, 1.00):
            ice = IceProperties(thickness_m=h)
            v_ref = math.sqrt(self.g * h)
            c2_bending = riska_resistance(
                IceProperties(thickness_m=h, compressive_strength_pa=1e-6),
                HullGeometry(friction_coefficient=0.0),
                0.0, self.env,
            )["c2_N_per_m_s"]
            r_bend = bending_resistance(ice, self.hull)
            factor = (4.0 / 3.0) * (1.0 + 0.063 * self.hull.stem_angle_deg)
            self.assertAlmostEqual(c2_bending * v_ref / r_bend, factor, places=6)

    def test_riska_speed_linearity_and_intercept(self):
        """Riska is strictly affine in speed with intercept C1 and slope C2."""
        for h in (0.30, 0.60, 1.00):
            ice = IceProperties(thickness_m=h)
            at_0 = riska_resistance(ice, self.hull, 0.0, self.env)
            at_1 = riska_resistance(ice, self.hull, 1.0, self.env)
            at_3 = riska_resistance(ice, self.hull, 3.0, self.env)
            self.assertAlmostEqual(at_0["total_N"], at_0["c1_N"], delta=1e-9)
            self.assertAlmostEqual(
                (at_3["total_N"] - at_0["total_N"]) / 3.0,
                at_1["total_N"] - at_0["total_N"],
                delta=1e-9 * (at_1["total_N"] - at_0["total_N"]),
            )
            self.assertAlmostEqual(
                at_1["total_N"] - at_0["total_N"], at_1["c2_N_per_m_s"], delta=1e-9 * at_1["total_N"]
            )

    # ------------------------------------------------- Lindqvist speed term
    def test_lindqvist_speed_term_is_pure_quadratic_added_drag(self):
        """total(V) - total(0) == 0.5 rho_w B h V^2 with no linear component."""
        for h in (0.30, 0.60):
            ice = IceProperties(thickness_m=h)
            base = lindqvist_resistance(ice, self.hull, 0.0, self.env)["total_N"]
            v = 1.0
            d1 = lindqvist_resistance(ice, self.hull, v, self.env)["total_N"] - base
            d2 = lindqvist_resistance(ice, self.hull, 2.0 * v, self.env)["total_N"] - base
            expected = 0.5 * self.rho * self.hull.beam_m * h * v**2
            self.assertAlmostEqual(d1, expected, delta=1e-9 * expected)
            self.assertAlmostEqual(d2, 4.0 * expected, delta=1e-9 * expected)

    def test_lindqvist_report_documents_its_speed_dependent_component(self):
        """The reported component list must explain the whole total."""
        result = lindqvist_resistance(self.ice, self.hull, 2.0, self.env)
        parts = (
            result["crushing_N"]
            + result["bending_N"]
            + result["submersion_N"]
            + result["speed_dependent_N"]
        )
        self.assertAlmostEqual(result["total_N"], parts, delta=1e-9 * parts)
        # At 2 m/s the added drag is a non-negligible fraction of the static part,
        # so "nearly speed independent" must not be read as "speed independent".
        static = result["total_N"] - result["speed_dependent_N"]
        self.assertGreater(result["speed_dependent_N"] / static, 0.05)

    # ----------------------------------------------------- submersion details
    def test_submersion_bracket_and_exact_length_derivative(self):
        """R_sub = rho g h^2 B (c_b + c_f mu L / B); dR_sub/dL = c_f mu rho g h^2."""
        h = self.ice.thickness_m
        bracket = SUBMERSION_BUOYANCY_COEFFICIENT + (
            SUBMERSION_FRICTION_COEFFICIENT * self.mu * self.hull.length_m / self.hull.beam_m
        )
        expected = self.rho * self.g * h * h * self.hull.beam_m * bracket
        self.assertAlmostEqual(
            submersion_resistance(self.ice, self.hull, self.env), expected, delta=1e-12 * expected
        )

        d_l = 5.0
        longer = HullGeometry(length_m=self.hull.length_m + d_l)
        measured = (
            submersion_resistance(self.ice, longer, self.env)
            - submersion_resistance(self.ice, self.hull, self.env)
        )
        analytic = SUBMERSION_FRICTION_COEFFICIENT * self.mu * self.rho * self.g * h * h * d_l
        self.assertAlmostEqual(measured, analytic, delta=1e-12 * analytic)

    def test_submersion_without_friction_is_the_buoyant_depression(self):
        """At mu = 0 only the channel-opening buoyancy term survives."""
        h = self.ice.thickness_m
        value = submersion_resistance(
            self.ice, HullGeometry(friction_coefficient=0.0), self.env
        )
        expected = (
            SUBMERSION_BUOYANCY_COEFFICIENT * self.rho * self.g * h * h * self.hull.beam_m
        )
        self.assertAlmostEqual(value, expected, delta=1e-12 * expected)

    # -------------------------------------------------- physical behaviour
    def test_monotonicity_in_every_active_parameter(self):
        """Both baselines must respond in the expected direction to each input."""
        v = 1.0

        def totals(ice=None, hull=None) -> tuple[float, float]:
            ice = ice or self.ice
            hull = hull or self.hull
            return (
                lindqvist_resistance(ice, hull, v, self.env)["total_N"],
                riska_resistance(ice, hull, v, self.env)["total_N"],
            )

        reference = totals()
        increasing = {
            "thickness": totals(IceProperties(thickness_m=0.9)),
            "compressive strength": totals(IceProperties(compressive_strength_pa=3.0e6)),
            "flexural strength": totals(IceProperties(flexural_strength_pa=1.0e6)),
            "friction": totals(hull=HullGeometry(friction_coefficient=0.15)),
            "entry angle": totals(hull=HullGeometry(waterline_entry_angle_deg=35.0)),
            "beam": totals(hull=HullGeometry(beam_m=28.0)),
        }
        for name, values in increasing.items():
            for i, (base, bump) in enumerate(zip(reference, values)):
                self.assertGreater(bump, base, f"model {i} not increasing in {name}")

# The stem angle (phi) enters the Riska C1/C2 correction only; the
        # Lindqvist-style sum as implemented has no phi dependence at all.
        steeper = totals(hull=HullGeometry(stem_angle_deg=45.0))
        self.assertGreater(steeper[1], reference[1], "riska not increasing in stem angle")
        self.assertEqual(steeper[0], reference[0], "lindqvist unexpectedly depends on stem angle")

        shorter = totals(hull=HullGeometry(length_m=60.0))
        for i, (base, bump) in enumerate(zip(reference, shorter)):
            self.assertGreater(bump, base, f"model {i} not decreasing in length")

        slower = (
            lindqvist_resistance(self.ice, self.hull, 0.0, self.env)["total_N"],
            riska_resistance(self.ice, self.hull, 0.0, self.env)["total_N"],
        )
        for base, bump in zip(slower, reference):
            self.assertGreater(bump, base, "not increasing in speed")

    def test_inert_parameters_are_declared_inert(self):
        """Buttock angle and ice elastic modulus do not enter either baseline.

        They are validated by the dataclasses but never used, so they are recorded
        here as inert.  If a future revision makes them active this test fails and
        forces the documentation update.
        """
        v = 1.0
        base = (
            lindqvist_resistance(self.ice, self.hull, v, self.env)["total_N"],
            riska_resistance(self.ice, self.hull, v, self.env)["total_N"],
        )
        active_buttock = HullGeometry(buttock_angle_deg=35.0)
        stiff_ice = IceProperties(elastic_modulus_pa=9.0e9)
        for ice, hull in ((self.ice, active_buttock), (stiff_ice, self.hull)):
            probe = (
                lindqvist_resistance(ice, hull, v, self.env)["total_N"],
                riska_resistance(ice, hull, v, self.env)["total_N"],
            )
            self.assertEqual(probe, base)

    # ------------------------------------------------------------- the band
    def test_band_consistency_and_reported_spread(self):
        """Band bounds/centre/spread must agree, and the spread is wide (recorded).

        The recorded spread near 47 % is a finding, not a target: it is three
        times the 15 % engineering tolerance used by the wider platform, so the
        band cannot serve as a precision gate -- errors must be quoted against
        each baseline separately, which the coupling report does.
        """
        for h, expected_spread in ((0.30, 46.6), (0.45, 47.4), (0.60, 47.7)):
            band = resistance_band(IceProperties(thickness_m=h), self.hull, 1.0, self.env)
            low, high = band["band_low_N"], band["band_high_N"]
            self.assertLess(low, high)
            self.assertAlmostEqual(band["band_centre_N"], 0.5 * (low + high), delta=1e-9 * high)
            recomputed = 100.0 * (high - low) / band["band_centre_N"]
            self.assertAlmostEqual(band["band_spread_pct"], recomputed, places=9)
            self.assertAlmostEqual(round(band["band_spread_pct"], 1), expected_spread, places=6)
            self.assertGreater(band["band_spread_pct"], 15.0)

    def test_lindqvist_is_the_lower_baseline_at_moderate_speed(self):
        """Below the crossing speed the Lindqvist baseline is the lower bound."""
        for h in (0.30, 0.60, 1.00):
            band = resistance_band(IceProperties(thickness_m=h), self.hull, 1.0, self.env)
            self.assertAlmostEqual(band["band_low_N"], band["lindqvist"]["total_N"])
            self.assertAlmostEqual(band["band_high_N"], band["riska"]["total_N"])

    def test_reproduction_table_for_the_coupling_study(self):
        """Pinned component breakdown for the three thicknesses of the study.

        These are the numbers quoted in docs/threefield/FORMULAS.md and in the
        mission report, so they are asserted here to six significant digits and
        are emitted programmatically rather than transcribed.
        """
        expected = {
            0.30: (30911.8250, 1875.0000, 16555.3414, 3075.0000, 52417.1664),
            0.45: (69551.6063, 4218.7500, 37249.5181, 4612.5000, 115632.3743),
            0.60: (123647.3000, 7500.0000, 66221.3655, 6150.0000, 203518.6655),
        }
        keys = (
            "crushing_N",
            "bending_N",
            "submersion_N",
            "speed_dependent_N",
            "total_N",
        )
        for h, row in expected.items():
            out = lindqvist_resistance(IceProperties(thickness_m=h), self.hull, 1.0, self.env)
            for key, pin in zip(keys, row):
                self.assertAlmostEqual(out[key], pin, delta=1e-6 * max(abs(pin), 1.0))

        riska_expected = {0.30: 84220.9, 0.45: 187443.2, 0.60: 331055.7}
        for h, pin in riska_expected.items():
            total = riska_resistance(IceProperties(thickness_m=h), self.hull, 1.0, self.env)["total_N"]
            self.assertAlmostEqual(total, pin, delta=1e-5 * pin)

    # ------------------------------------------------- parameter validation
    def test_invalid_parameters_are_rejected(self):
        for kwargs in (
            {"thickness_m": 0.0},
            {"thickness_m": -1.0},
            {"thickness_m": float("nan")},
            {"compressive_strength_pa": 0.0},
            {"flexural_strength_pa": float("inf")},
        ):
            with self.assertRaises((ValueError, TypeError)):
                IceProperties(**kwargs)
        for kwargs in (
            {"beam_m": 0.0},
            {"waterline_entry_angle_deg": 90.0},
            {"stem_angle_deg": -5.0},
            {"friction_coefficient": -0.1},
        ):
            with self.assertRaises((ValueError, TypeError)):
                HullGeometry(**kwargs)
        with self.assertRaises(ValueError):
            lindqvist_resistance(self.ice, self.hull, -0.5, self.env)
        with self.assertRaises(ValueError):
            riska_resistance(self.ice, self.hull, -0.5, self.env)
        with self.assertRaises(ValueError):
            Environment(water_density_kg_m3=0.0)

    def test_relative_error_helper(self):
        self.assertAlmostEqual(relative_error_pct(110.0, 100.0), 10.0, places=12)
        self.assertAlmostEqual(relative_error_pct(90.0, 100.0), 10.0, places=12)
        with self.assertRaises(ValueError):
            relative_error_pct(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()