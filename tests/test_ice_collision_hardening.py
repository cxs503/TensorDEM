"""Acceptance tests for p256w1-N04: strain-hardening hinges + ISO 19906 ice pressure.

Every number here is checked against an *independent* route or a published
relation - never the module compared with itself:

* closed-form hardening moment      vs fibre quadrature of the same section
* closed-form hardening energy      vs Simpson integration of the fibre moment
* closed-form hardening energy      vs the rigid-hardening upper bound
* hardening energy/resistance       vs finite difference of ``HardenedLayer.energy``
* ``E_h = 0`` surface               vs the N01 ``solve_penetration`` result (exact)
* rigid indenter + no hardening     vs the N01 damage geometry (negative control)
* pressure-area closed form         vs Simpson integration of the ice load
* ``C`` from the energy share       vs ``C`` inverted from a damaged volume
* least-squares ``C`` fit           vs synthetic cases generated with a known C
"""
import math

import pytest

from tensordem.hull_collision import (
    DATASET_STATUS_WITHHELD,
    BilinearHinge,
    HardenedLayer,
    PressureAreaLaw,
    PowerLawHinge,
    SideShellLayer,
    coefficient_uncertainty,
    collision_coefficient_from_share,
    collision_coefficient_from_volume,
    damage_geometry,
    damage_geometry_pressure_driven,
    energy_increment_numeric,
    energy_ledger,
    force_history,
    iso19906_scaling_exponents,
    kinetic_energy,
    load_collision_dataset,
    minorsky_coefficient,
    moment_increment_fibre,
    recalibrate_coefficient,
    solve_indentation,
    solve_penetration,
    structural_coefficients_per_length,
)
from tensordem.hull_collision import strain_hardening
from tensordem.hull_collision.scenario import (
    FRAME_SPAN,
    HARDENING_RATIO,
    ICE_CONTACT_HEIGHT,
    ICE_CONTACT_LENGTH,
    ICE_PRESSURE_EXPONENT,
    ICE_REFERENCE_PRESSURE,
    SCRATCH_SIGMA_Y,
    design_case,
    effective_mass,
    hardened_design_case,
    ice_pressure_law,
)

VELOCITIES = (2.0, 4.0, 6.0, 8.0)
EFFECTIVE = effective_mass()
LAW = ice_pressure_law()


def _indent(structure, velocity, *, side=None, law=LAW):
    side = side if side is not None else hardened_design_case()
    return solve_indentation(
        kinetic_energy(EFFECTIVE, velocity),
        side,
        law=law,
        contact_height=ICE_CONTACT_HEIGHT,
        contact_length=ICE_CONTACT_LENGTH,
    )


# --------------------------------------------------------------- hinge section
def test_bilinear_moment_increment_matches_fibre_quadrature():
    hinge = BilinearHinge(sigma_y=SCRATCH_SIGMA_Y, thickness=0.022, hardening_modulus=0.01 * 210e9)
    for kappa in (0.05, 0.2, 0.5, 2.0, 18.18, 60.0):
        fibre = moment_increment_fibre(
            kappa,
            sigma_y=hinge.sigma_y,
            thickness=hinge.thickness,
            hardening_modulus=hinge.hardening_modulus,
            n_fibres=12001,
        )
        closed = strain_hardening.bilinear_moment_increment_closed_form(
            kappa,
            sigma_y=hinge.sigma_y,
            thickness=hinge.thickness,
            hardening_modulus=hinge.hardening_modulus,
        )
        assert closed == pytest.approx(fibre, rel=5e-4)


def test_criterion_b_hardening_energy_increment_matches_simpson_route():
    hinge = BilinearHinge(sigma_y=SCRATCH_SIGMA_Y, thickness=0.022, hardening_modulus=0.01 * 210e9)
    for theta in (0.1, 0.25, 0.4, 0.885, 1.2, 1.6):
        closed = hinge.energy_increment(theta)
        numeric = energy_increment_numeric(hinge, theta, n_steps=96, n_fibres=2001)
        assert closed > 0.0
        assert abs(numeric - closed) / closed <= 0.02


def test_criterion_b_increment_stays_below_rigid_hardening_bound():
    hinge = BilinearHinge(sigma_y=SCRATCH_SIGMA_Y, thickness=0.022, hardening_modulus=0.01 * 210e9)
    for theta in (0.2, 0.4, 0.885, 1.6):
        closed = hinge.energy_increment(theta)
        bound = hinge.energy_increment_upper_bound(theta)
        assert 0.0 < closed <= bound
        # for well-developed plasticity the bound is tight (elastic-core effect)
        if theta >= 0.885:
            assert closed / bound >= 0.98


def test_hinge_rotation_capacity_caps_peak_fibre_strain():
    hinge = BilinearHinge(
        sigma_y=SCRATCH_SIGMA_Y, thickness=0.022, hardening_modulus=1e9, rotation_capacity=0.2
    )
    assert hinge.peak_fibre_strain(0.1) == pytest.approx(0.1 / 2.0)
    for theta in (0.4, 0.885, 2.0):
        assert hinge.peak_fibre_strain(theta) == pytest.approx(0.2)
    # curvature is constant once the hinge spreads -> constant moment increment
    assert hinge.moment_increment(0.885) == pytest.approx(hinge.moment_increment(2.0))


def test_power_law_hardening_reduces_to_bilinear_at_unit_exponent():
    bilinear = BilinearHinge(sigma_y=SCRATCH_SIGMA_Y, thickness=0.022, hardening_modulus=2e9)
    power = PowerLawHinge(
        sigma_y=SCRATCH_SIGMA_Y, thickness=0.022, hardening_coefficient=2e9, exponent=1.0
    )
    for theta in (0.2, 0.6):
        assert power.moment_increment(theta, n_fibres=6001) == pytest.approx(
            bilinear.moment_increment(theta), rel=2e-3
        )


# ------------------------------------------------------------------- hardening
def test_hardening_zero_is_an_exact_identity_with_n01():
    structure = design_case()
    plastic = strain_hardening.build_hardened_side(structure, hardening_ratio=0.0)
    assert plastic.total_capacity == pytest.approx(structure.total_capacity, rel=1e-15)
    for velocity in VELOCITIES:
        e_k = kinetic_energy(EFFECTIVE, velocity)
        n01 = solve_penetration(e_k, structure)
        new = solve_indentation(e_k, plastic)
        assert new.penetration_depth == n01.penetration_depth
        assert new.structural_energy == pytest.approx(n01.structural_energy, rel=1e-15)
        for layer in plastic.layers:
            u = layer.base.crush_depth
            assert layer.hinge.energy_increment(layer.hinge_rotation(u)) == 0.0
            assert layer.hinge.moment_increment(layer.hinge_rotation(u)) == 0.0


def test_criterion_a_index_swap_within_one_per_mille():
    structure = design_case()
    plastic = strain_hardening.build_hardened_side(structure, hardening_ratio=0.0)
    for velocity in VELOCITIES:
        e_k = kinetic_energy(EFFECTIVE, velocity)
        n01 = solve_penetration(e_k, structure)
        new = solve_indentation(e_k, plastic)
        assert abs(new.penetration_depth - n01.penetration_depth) / n01.penetration_depth <= 1e-3
        assert abs(new.structural_energy - n01.structural_energy) / n01.structural_energy <= 1e-3
        assert (
            abs(new.structural_resistance_N - n01.structural_resistance_N)
            / n01.structural_resistance_N
            <= 1e-3
        )


def test_hardened_layer_resistance_is_energy_derivative():
    side = hardened_design_case()
    layer = side.layers[0]
    step = 1e-6
    for u in (0.05, 0.2, 0.5):
        numeric = (layer.energy(u + step) - layer.energy(u - step)) / (2.0 * step)
        assert layer.resistance(u) == pytest.approx(numeric, rel=1e-6)


def test_hardening_increases_absorbed_energy_at_equal_depth():
    plastic = strain_hardening.build_hardened_side(design_case(), hardening_ratio=0.0)
    hardened = hardened_design_case()
    for depth in (0.2, 0.5):
        assert hardened.structural_energy(depth) > plastic.structural_energy(depth)
    ratio = hardened.structural_energy(0.5) / plastic.structural_energy(0.5)
    assert 1.2 < ratio < 2.0  # 1 % tangent modulus: +48 % at theta ~ 1.7 rad


def test_hardened_layer_rejects_mismatched_hinge():
    base = design_case().layers[0]
    with pytest.raises(ValueError):
        HardenedLayer(
            base=base,
            hinge=BilinearHinge(sigma_y=base.sigma_y, thickness=base.thickness * 2, hardening_modulus=1e9),
        )


# -------------------------------------------------------------- ice pressure
@pytest.mark.parametrize("alpha", (0.0, 0.4, 0.5, 0.6))
def test_pressure_area_closed_form_matches_simpson(alpha):
    law = PressureAreaLaw(reference_pressure=3.0e6, exponent=alpha, reference_area=1.0)
    for depth in (0.05, 0.25, 1.0):
        closed = law.ice_energy(ICE_CONTACT_HEIGHT, depth)
        numeric = law.ice_energy_numeric(ICE_CONTACT_HEIGHT, depth, n_steps=8000)
        assert closed == pytest.approx(numeric, rel=1e-6)
        assert law.ice_load(ICE_CONTACT_HEIGHT, depth) == pytest.approx(
            law.pressure(law.contact_area(ICE_CONTACT_HEIGHT, depth))
            * law.contact_area(ICE_CONTACT_HEIGHT, depth),
            rel=1e-15,
        )


def test_zero_penetration_absorbs_no_ice_energy():
    assert LAW.ice_energy(ICE_CONTACT_HEIGHT, 0.0) == 0.0
    assert LAW.ice_load(ICE_CONTACT_HEIGHT, 0.0) == 0.0
    solution = solve_indentation(0.0, hardened_design_case(), law=LAW,
                                 contact_height=ICE_CONTACT_HEIGHT,
                                 contact_length=ICE_CONTACT_LENGTH)
    assert solution.penetration_depth == 0.0 and solution.ice_energy == 0.0


def test_constant_pressure_gives_linear_energy_law():
    law = PressureAreaLaw(reference_pressure=3.0e6, exponent=0.0, reference_area=1.0)
    assert law.ice_energy(ICE_CONTACT_HEIGHT, 0.25) == pytest.approx(
        3.0e6 * ICE_CONTACT_HEIGHT * 0.25 ** 2 / 2.0, rel=1e-15
    )


def test_pressure_law_rejects_unphysical_exponent():
    with pytest.raises(ValueError):
        PressureAreaLaw(reference_pressure=1e6, exponent=1.0)


def test_criterion_e_scaling_exponents_match_iso_family():
    for alpha in (0.40, 0.45, 0.50, 0.55, 0.60):
        law = PressureAreaLaw(reference_pressure=3.0e6, exponent=alpha)
        analytic = iso19906_scaling_exponents(alpha)
        # numerical derivation: invert the closed-form energy by bisection and fit
        energies = [8.0e6, 12.0e6, 20.0e6, 30.0e6, 45.0e6]
        depths = []
        for energy in energies:
            lo, hi = 1e-9, 1.0
            while law.ice_energy(ICE_CONTACT_HEIGHT, hi) < energy:
                hi *= 2.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if law.ice_energy(ICE_CONTACT_HEIGHT, mid) < energy:
                    lo = mid
                else:
                    hi = mid
            depths.append(0.5 * (lo + hi))
        xs = [math.log(e) for e in energies]
        ys = [math.log(d) for d in depths]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
            (x - mx) ** 2 for x in xs
        )
        assert slope == pytest.approx(analytic["penetration_vs_energy"], rel=1e-9)
        low, high = sorted([1.0 / (2.0 - 0.60), 1.0 / (2.0 - 0.40)])
        assert low - 1e-12 <= slope <= high + 1e-12


# ------------------------------------------------------------ coupled ledger
def test_criterion_c_energy_ledger_closes():
    side = hardened_design_case()
    for velocity in VELOCITIES:
        solution = _indent(design_case(), velocity, side=side)
        ledger = energy_ledger(solution)
        assert ledger["residual_relative"] <= 1e-6
        assert ledger["steel_share"] + ledger["friction_share"] + ledger["ice_share"] == pytest.approx(
            1.0, abs=1e-9
        )
        assert solution.ice_energy > 0.0 and solution.ice_load_N > 0.0


def test_ice_side_reduces_penetration_relative_to_rigid_indenter():
    side = hardened_design_case()
    for velocity in VELOCITIES:
        with_ice = _indent(design_case(), velocity, side=side)
        without_ice = solve_indentation(kinetic_energy(EFFECTIVE, velocity), side)
        assert with_ice.penetration_depth < without_ice.penetration_depth


def test_pressure_integral_damage_length_is_max_of_contact_and_yield_length():
    side = hardened_design_case()
    for velocity in VELOCITIES:
        solution = _indent(design_case(), velocity, side=side)
        assert solution.damage_length_m == pytest.approx(
            max(ICE_CONTACT_LENGTH, solution.yield_length_m)
        )
        assert solution.yield_length_m > 0.0


def test_structural_coefficients_reproduce_n01_layer_resistance():
    layer = design_case().layers[0]
    kappa1, kappa2 = structural_coefficients_per_length(layer)
    for depth in (0.1, 0.4):
        strip_resistance = layer.effective_width * (kappa1 + 2.0 * kappa2 * depth)
        assert strip_resistance == pytest.approx(layer.resistance(depth), rel=1e-15)


def test_criterion_f_negative_control_reproduces_n01():
    structure = design_case()
    plastic = strain_hardening.build_hardened_side(structure, hardening_ratio=0.0)
    box_length = ICE_CONTACT_LENGTH + 2.0 * FRAME_SPAN
    for velocity in VELOCITIES:
        e_k = kinetic_energy(EFFECTIVE, velocity)
        n01 = solve_penetration(e_k, structure)
        n01_geometry = damage_geometry(structure, n01, contact_length=ICE_CONTACT_LENGTH)
        new = solve_indentation(e_k, plastic)  # law=None: rigid indenter, constant pressure
        geometry, _ = damage_geometry_pressure_driven(
            plastic, new, contact_length=ICE_CONTACT_LENGTH, length=box_length
        )
        assert new.penetration_depth == pytest.approx(n01.penetration_depth, rel=1e-12)
        assert new.structural_energy == pytest.approx(n01.structural_energy, rel=1e-12)
        assert new.friction_energy == pytest.approx(n01.friction_energy, rel=1e-12)
        assert geometry.volume_m3 == pytest.approx(n01_geometry.volume_m3, rel=1e-12)
        assert geometry.length_m == n01_geometry.length_m
        assert new.ice_energy == 0.0


def test_force_history_is_monotone_and_energy_consistent():
    side = hardened_design_case()
    history = force_history(
        side,
        kinetic_energy=kinetic_energy(EFFECTIVE, 4.0),
        effective_mass=EFFECTIVE,
        law=LAW,
        contact_height=ICE_CONTACT_HEIGHT,
        contact_length=ICE_CONTACT_LENGTH,
        n_points=25,
    )
    assert history["time_s"][0] == 0.0
    assert all(b >= a for a, b in zip(history["time_s"], history["time_s"][1:]))
    assert all(v <= 4.0 + 1e-9 for v in history["velocity_m_s"])
    assert all(b >= a for a, b in zip(history["load_N"], history["load_N"][1:]))
    assert history["history_depth_m"] <= history["solved_depth_m"]


# -------------------------------------------------------------- recalibration
def test_coefficient_identity_between_share_and_volume_routes():
    for share, velocity in ((0.5, 4.0), (0.83, 8.0)):
        e_k = kinetic_energy(EFFECTIVE, velocity)
        e_steel = share * e_k
        c_share = collision_coefficient_from_share(share, SCRATCH_SIGMA_Y)
        c_volume = collision_coefficient_from_volume(
            e_steel / (47.1e6), EFFECTIVE, velocity, SCRATCH_SIGMA_Y
        )
        assert c_volume == pytest.approx(c_share, rel=1e-12)


def test_recalibration_recovers_a_known_coefficient_on_synthetic_cases():
    """Synthetic cases (clearly not a data set) to test the fitting code only."""
    truth = 3.0
    cases = []
    for index, (mass, velocity, sigma_y) in enumerate(
        ((1.0e7, 4.0, 235e6), (2.0e7, 6.0, 235e6), (8.0e6, 3.0, 315e6))
    ):
        cases.append(
            {
                "case": f"synthetic_{index}",
                "regime": "ship_ship",
                "mass_kg": mass,
                "velocity_m_s": velocity,
                "sigma_y_Pa": sigma_y,
                "volume_m3": truth * mass * velocity ** 2 / sigma_y,
            }
        )
    fit = recalibrate_coefficient(cases, regime="ship_ship")
    assert fit["coefficient"] == pytest.approx(truth, rel=1e-12)
    assert fit["median_relative_residual"] == pytest.approx(0.0, abs=1e-12)
    assert fit["meets_threshold"] is True
    # an ice-regime fit on the same list has no samples and is withheld
    assert recalibrate_coefficient(cases, regime="ice")["status"] == DATASET_STATUS_WITHHELD


def test_recalibration_never_pools_regimes():
    cases = [
        {
            "case": "ship",
            "regime": "ship_ship",
            "mass_kg": 1.0e7,
            "velocity_m_s": 5.0,
            "sigma_y_Pa": 235e6,
            "volume_m3": 20.0,
        },
        {
            "case": "ice",
            "regime": "ice",
            "mass_kg": 2.2e6,
            "velocity_m_s": 4.0,
            "sigma_y_Pa": 355e6,
            "volume_m3": 3.0,
        },
    ]
    ice_fit = recalibrate_coefficient(cases, regime="ice")
    ship_fit = recalibrate_coefficient(cases, regime="ship_ship")
    assert ice_fit["n"] == 1 and ship_fit["n"] == 1
    assert ice_fit["coefficient"] != ship_fit["coefficient"]


def test_missing_dataset_is_withheld_not_invented():
    dataset = load_collision_dataset("/nonexistent/collision_cases.csv")
    assert dataset["status"] == DATASET_STATUS_WITHHELD
    assert dataset["n"] == 0
    assert dataset["cases"] == []
    assert dataset["reason"]


def test_coefficient_uncertainty_envelope():
    envelope = coefficient_uncertainty([2.0, 3.0, 4.0])
    assert envelope["min"] == 2.0 and envelope["max"] == 4.0
    assert envelope["median"] == 3.0
    assert envelope["n"] == 3


def test_iso_reference_minorsky_coefficient_is_retained():
    assert minorsky_coefficient() == pytest.approx(235e6 / (2 * 47.1e6))
    assert collision_coefficient_from_share(1.0, 235e6) == pytest.approx(minorsky_coefficient())


# ------------------------------------------------------------------- guards
def test_invalid_inputs_are_rejected():
    side = hardened_design_case()
    with pytest.raises(ValueError):
        solve_indentation(-1.0, side)
    with pytest.raises(ValueError):
        solve_indentation(1e12, side, law=LAW, contact_height=ICE_CONTACT_HEIGHT)
    with pytest.raises(TypeError):
        solve_indentation(1.0e6, design_case())
    with pytest.raises(ValueError):
        BilinearHinge(sigma_y=1.0, thickness=1.0, hardening_modulus=-1.0)
    layer = design_case().layers[0]
    with pytest.raises(TypeError):
        structural_coefficients_per_length(layer.thickness)
    with pytest.raises(ValueError):
        force_history(side, kinetic_energy=1.0e6, effective_mass=1.0, n_points=2)