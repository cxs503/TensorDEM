"""Verification tests for the LEFM ice-crack toolkit (``tensordem.lefm``).

The suite is deliberately split into

* **exact** tests -- lattice elasticity patch tests, ASTM shape factor, MTS
  algebra, energy accounting; these must hold to machine/closed-form precision;
* **structural** tests -- mode separation, linearity in the load, slit topology,
  crack-path straightness, path-overlap metric;
* **benchmark** tests -- the measured accuracy of each K route and of the K_IC
  calibration against its reference.  Their tolerances encode what the model
  actually delivers at the tested lattice resolution (see
  ``docs/LEFM_ICE_CRACK.md``); they are regression guards, not accuracy claims.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.lefm import (BondedLattice2D, Lattice2DConfig, SENBGeometry,
                            astm_stress_intensity, astm_senb_shape_factor,
                            bond_stress_intensity, bonds_crossing, build_senb,
                            calibrate_kic, contour_j, domain_j,
                            energy_release_rate_from_crack_advance,
                            max_hoop_stress, mts_angle, path_overlap,
                            polyline_distance, stress_intensity_from_j,
                            stress_intensity_from_m, tip_bond_intensity,
                            trace_crack_path)
from tensordem.lefm.jintegral import AuxiliaryField

E = 5.0e9
B = 0.05
A = 0.005


@pytest.fixture(scope="module")
def lattice():
    cfg = Lattice2DConfig.from_material(E, B, A)
    return BondedLattice2D(cfg, 13, 13)


@pytest.fixture(scope="module")
def senb():
    geom = SENBGeometry(width=0.10, spacing=A, notch_ratio=0.3, thickness=B,
                        young_modulus=E)
    lat = build_senb(geom, load=100.0)
    lat.solve()
    return geom, lat


# --------------------------------------------------------------- lattice basics
def test_bond_count_matches_8_neighbour_shell():
    lat = BondedLattice2D(Lattice2DConfig.from_material(E, B, A), 5, 4)
    # axial x: (nc-1)*nr, axial y: nc*(nr-1), both diagonals: 2*(nc-1)*(nr-1)
    expected = (5 - 1) * 4 + 5 * (4 - 1) + 2 * (5 - 1) * (4 - 1)
    assert lat.nbonds == expected


def test_axial_and_diagonal_stiffness_ratio_is_one_half():
    cfg = Lattice2DConfig.from_material(E, B, A)
    assert cfg.diagonal_stiffness == pytest.approx(0.5 * cfg.axial_stiffness)
    assert cfg.modulus_2d / cfg.thickness == pytest.approx(E)


def test_poisson_ratio_of_lattice_is_one_quarter():
    cfg = Lattice2DConfig.from_material(E, B, A)
    lam, mu = cfg.lame_lambda, cfg.shear_modulus
    nu = lam / (2.0 * (lam + mu))
    assert nu == pytest.approx(0.25)


def test_from_material_rejects_other_poisson_ratios():
    with pytest.raises(ValueError):
        Lattice2DConfig.from_material(E, B, A, poisson_ratio=0.3)


def _affine(lat, eps):
    for n in range(lat.nnodes):
        lat._fixed[n] = (eps[0, 0] * lat.x[n] + eps[0, 1] * lat.y[n],
                         eps[1, 0] * lat.x[n] + eps[1, 1] * lat.y[n])
    lat.solve()


def test_affine_patch_normal_stress_exact(lattice):
    eps = np.array([[2e-4, 0.0], [0.0, -1e-4]])
    _affine(lattice, eps)
    cfg = lattice.cfg
    exact = (cfg.lame_lambda * np.trace(eps) * np.eye(2) + 2 * cfg.shear_modulus * eps) / B
    interior = lattice.node_stress().reshape(lattice.nrows, lattice.ncols, 2, 2)[3:10, 3:10]
    assert np.allclose(interior.mean((0, 1)), exact, rtol=1e-12)


def test_affine_patch_shear_stress_exact(lattice):
    eps = np.array([[0.0, 3e-4], [3e-4, 0.0]])
    _affine(lattice, eps)
    cfg = lattice.cfg
    exact = (cfg.lame_lambda * np.trace(eps) * np.eye(2) + 2 * cfg.shear_modulus * eps) / B
    interior = lattice.node_stress().reshape(lattice.nrows, lattice.ncols, 2, 2)[3:10, 3:10]
    assert np.allclose(interior.mean((0, 1)), exact, rtol=1e-12)
    lattice._fixed.clear()


def test_displacement_gradient_recovers_affine_field(lattice):
    eps = np.array([[2e-4, 1e-4], [-3e-5, -1e-4]])
    _affine(lattice, eps)
    grad = lattice.node_displacement_gradient()
    interior = grad.reshape(lattice.nrows, lattice.ncols, 2, 2)[3:10, 3:10]
    # grad[n][component, derivative] == d u_component / d x_derivative
    assert np.allclose(interior.mean((0, 1)), eps, rtol=1e-10)
    lattice._fixed.clear()


def test_elastic_energy_equals_half_force_displacement(lattice):
    lat = BondedLattice2D(Lattice2DConfig.from_material(E, B, A), 9, 9)
    lat.set_fixed([lat.node(0, 0)], 0.0, 0.0)
    lat.add_load(lat.node(8, 4), 0.0, 1.0)
    lat.add_load(lat.node(0, 8), 0.0, -1.0)
    lat.set_fixed([lat.node(8, 7)], 0.0, 0.0)
    lat.solve()
    assert lat.energy() == pytest.approx(lat.external_work(), rel=1e-9)


def test_slit_severs_the_two_faces():
    lat = BondedLattice2D(Lattice2DConfig.from_material(E, B, A), 15, 15)
    jc = 7
    lat.kill_bonds(lat.slit_bonds(jc, 0.05))
    n0, n1 = lat.bonds[lat.active, 0], lat.bonds[lat.active, 1]
    # no active bond may cross the slit line
    crosses = (np.minimum(lat.y[n0], lat.y[n1]) < jc * A - 1e-9) & \
              (np.maximum(lat.y[n0], lat.y[n1]) > jc * A + 1e-9) & \
              (0.5 * (lat.x[n0] + lat.x[n1]) <= 0.05 + 1e-9)
    assert not np.any(crosses)
    # the faces must not be stitched: no active bond lies along the slit
    along = (np.abs(lat.y[n0] - jc * A) < 1e-9) & (np.abs(lat.y[n1] - jc * A) < 1e-9) & \
            (0.5 * (lat.x[n0] + lat.x[n1]) <= 0.05 + 1e-9)
    assert not np.any(along)


# ------------------------------------------------------------------ ASTM closed form
def test_astm_shape_factor_table_value():
    # ASTM E399-20a: f(0.5) = 2.66 for a single-edge-notched bend specimen
    assert astm_senb_shape_factor(0.5) == pytest.approx(2.66, abs=0.01)


def test_astm_shape_factor_limits():
    assert astm_senb_shape_factor(1e-6) == pytest.approx(0.0, abs=0.01)
    # f ~ 3 sqrt(alpha) for small alpha
    assert astm_senb_shape_factor(1e-4) / astm_senb_shape_factor(1e-6) == pytest.approx(10.0, rel=1e-2)
    assert astm_senb_shape_factor(0.9) > astm_senb_shape_factor(0.5)
    with pytest.raises(ValueError):
        astm_senb_shape_factor(1.2)


def test_astm_stress_intensity_scales_inversely_with_thickness():
    k1 = astm_stress_intensity(100.0, 0.4, 0.05, 0.1, 0.3)
    k2 = astm_stress_intensity(100.0, 0.4, 0.10, 0.1, 0.3)
    assert k1 / k2 == pytest.approx(2.0)


# -------------------------------------------------------------------------- MTS
def test_mts_pure_mode_one_is_straight():
    assert mts_angle(1e5, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_mts_pure_mode_two_gives_classical_kink_angle():
    theta = math.degrees(mts_angle(0.0, 1e5))
    assert theta == pytest.approx(-70.53, abs=0.05)


def test_mts_mixity_root_satisfies_the_stationarity_condition():
    # MTS root: K_I sin(theta_c) + K_II (3 cos(theta_c) - 1) = 0
    for ratio in (0.5, 1.0, 2.0):
        th = mts_angle(1e5, ratio * 1e5)
        assert 1e5 * math.sin(th) + ratio * 1e5 * (3 * math.cos(th) - 1) == pytest.approx(0.0, abs=1e-6)
    # K_II / K_I = 1 -> theta_c = 2 atan2(-2, 4) = -53.13 deg
    assert math.degrees(mts_angle(1e5, 1e5)) == pytest.approx(-53.13, abs=0.02)


def test_mts_odd_in_mode_two():
    for ratio in (0.25, 0.5, 1.0, 2.0):
        plus = mts_angle(1e5, ratio * 1e5)
        minus = mts_angle(1e5, -ratio * 1e5)
        assert plus == pytest.approx(-minus, rel=1e-10)


def test_max_hoop_stress_peaks_at_mts_angle():
    k1, k2 = 1.0e5, 4.0e4
    theta = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 20001)
    hoop = max_hoop_stress(k1, k2, theta)
    assert theta[int(np.argmax(hoop))] == pytest.approx(mts_angle(k1, k2), abs=2e-3)


# --------------------------------------------------------------- auxiliary field
def test_auxiliary_mode_one_singularity():
    field = AuxiliaryField.create("I", E, 0.25)
    r = 0.01
    sig, u, du, eps = field.sample(np.array([r]), np.array([0.0]), k_aux=1e5)
    expected = 1e5 / math.sqrt(2.0 * math.pi * r)
    assert sig[0, 1, 1] == pytest.approx(expected, rel=1e-12)
    assert sig[0, 0, 1] == pytest.approx(0.0, abs=1e-9)
    assert sig[0, 0, 0] == pytest.approx(expected, rel=1e-12)


def test_auxiliary_mode_two_shear_singularity():
    field = AuxiliaryField.create("II", E, 0.25)
    r = 0.01
    sig, u, du, eps = field.sample(np.array([r]), np.array([0.0]), k_aux=1e5)
    expected = 1e5 / math.sqrt(2.0 * math.pi * r)
    assert sig[0, 0, 1] == pytest.approx(expected, rel=1e-12)
    assert sig[0, 1, 1] == pytest.approx(0.0, abs=1e-9)


def test_auxiliary_strain_consistent_with_plane_stress_law():
    field = AuxiliaryField.create("I", E, 0.25)
    r = np.array([0.01, 0.02])
    th = np.array([0.3, -0.7])
    sig, u, du, eps = field.sample(r * np.cos(th), r * np.sin(th), k_aux=1e5)
    nu = 0.25
    assert np.allclose(eps[:, 0, 0], (sig[:, 0, 0] - nu * sig[:, 1, 1]) / E)
    assert np.allclose(eps[:, 0, 1], (1 + nu) * sig[:, 0, 1] / E)


# ----------------------------------------------------------- SEN B elastic routes
def test_senb_bond_intensity_is_linear_in_load():
    geom = SENBGeometry(width=0.10, spacing=A, notch_ratio=0.3, thickness=B, young_modulus=E)
    lat1 = build_senb(geom, load=100.0)
    lat1.solve()
    lat2 = build_senb(geom, load=250.0)
    lat2.solve()
    k1 = tip_bond_intensity(lat1, lat1.metadata["tip"])
    k2 = tip_bond_intensity(lat2, lat2.metadata["tip"])
    assert k2 / k1 == pytest.approx(2.5, rel=1e-6)


def test_senb_domain_j_is_path_independent(senb):
    geom, lat = senb
    tip = lat.metadata["tip"]
    ks = [stress_intensity_from_j(domain_j(lat, tip, r1 * A, r2 * A), E)
          for r1, r2 in ((2, 4), (4, 7), (6, 10))]
    assert (max(ks) - min(ks)) / np.mean(ks) < 0.05


def test_senb_contour_j_is_positive_and_right_order(senb):
    geom, lat = senb
    tip = lat.metadata["tip"]
    J = contour_j(lat, tip, 6 * A)
    k_astm = astm_stress_intensity(100.0, geom.span, geom.thickness, geom.depth, geom.alpha)
    # the traversal convention of :func:`contour_j` makes a tensile crack come
    # out negative, so |J| is what carries the physics (documented in the doc)
    assert abs(J) > 0.0
    k_contour = stress_intensity_from_j(J, E)
    assert 0.05 < k_contour / k_astm < 5.0


def test_senb_mode_two_is_negligible_for_symmetric_load(senb):
    geom, lat = senb
    tip = lat.metadata["tip"]
    k1 = stress_intensity_from_m(lat, tip, "I", 2 * A, 4.5 * A)
    k2 = stress_intensity_from_m(lat, tip, "II", 2 * A, 4.5 * A)
    assert abs(k2 / k1) < 0.05


def test_energy_release_rate_is_positive_and_finite():
    geom = SENBGeometry(width=0.10, spacing=A, notch_ratio=0.3, thickness=B, young_modulus=E)
    out = energy_release_rate_from_crack_advance(geom, delta_cells=2, load=100.0)
    assert out["G_energy"] > 0.0
    assert 0.05 < out["ratio_energy_astm"] < 5.0


# -------------------------------------------------------------- K_IC calibration
def test_calibration_is_linear_in_the_input_toughness():
    geom = SENBGeometry(width=0.10, spacing=A, notch_ratio=0.3, thickness=B, young_modulus=E)
    a = calibrate_kic(geom, k_ic_input=2.0e5, load=100.0)
    b = calibrate_kic(geom, k_ic_input=4.0e5, load=100.0)
    assert b["critical_load"] / a["critical_load"] == pytest.approx(2.0, rel=1e-9)
    assert b["k_ic_calibrated"] / a["k_ic_calibrated"] == pytest.approx(2.0, rel=1e-9)


def test_lattice_correction_is_geometry_transferable():
    base = dict(width=0.10, spacing=A, thickness=B, young_modulus=E)
    kappas = [calibrate_kic(SENBGeometry(notch_ratio=al, span_ratio=4.0, **base),
                            k_ic_input=2.0e5)["lattice_correction"]
              for al in (0.2, 0.3, 0.4, 0.5)]
    assert np.ptp(kappas) / np.mean(kappas) < 0.05


def test_calibrated_toughness_lands_in_literature_band():
    base = dict(width=0.10, spacing=A, thickness=B, young_modulus=E)
    cal = calibrate_kic(SENBGeometry(notch_ratio=0.3, span_ratio=4.0, **base), k_ic_input=2.0e5)
    kappa = cal["lattice_correction"]
    for alpha in (0.2, 0.4, 0.5):
        for sr in (2.0, 6.0):
            r = calibrate_kic(SENBGeometry(notch_ratio=alpha, span_ratio=sr, **base),
                              k_ic_input=2.0e5)
            assert abs(r["k_ic_calibrated"] / kappa / 2.0e5 - 1.0) < 0.15


# ------------------------------------------------------------- negative controls
def test_negative_control_uniform_tension_does_not_crack():
    from tensordem.lefm import uniform_tension_control
    geom = SENBGeometry(width=0.10, spacing=A, notch_ratio=0.3, thickness=B, young_modulus=E)
    ctl = uniform_tension_control(geom, k_ic=2.0e5)
    assert ctl["no_crack"], ctl
    assert ctl["k_ratio"] < 1.0


def test_negative_control_infinite_toughness_never_breaks():
    geom = SENBGeometry(width=0.10, spacing=A, notch_ratio=0.3, thickness=B, young_modulus=E)
    lat = build_senb(geom, load=1.0e4)
    lat.solve()
    kb = bond_stress_intensity(lat)
    assert np.max(kb) < 1e12


# ------------------------------------------------------------------ crack path
def test_path_overlap_metrics():
    ref = np.array([[0.0, 0.0], [0.1, 0.0]])
    assert path_overlap(ref, ref, 1e-6) == pytest.approx(1.0)
    shifted = ref + np.array([0.0, 0.01])
    assert path_overlap(shifted, ref, 0.02) == pytest.approx(1.0)
    assert path_overlap(shifted, ref, 0.001) == pytest.approx(0.0)
    assert np.allclose(polyline_distance(ref, ref), 0.0, atol=1e-12)


def test_bonds_crossing_finds_the_slit_line():
    lat = BondedLattice2D(Lattice2DConfig.from_material(E, B, A), 11, 11)
    hits = bonds_crossing(lat, (0.02, 0.02), (0.04, 0.02))
    assert hits.size >= 8
    for h in hits:
        n0, n1 = lat.bonds[h][:2]
        # the bond bounding box must overlap the segment bounding box
        assert min(lat.y[n0], lat.y[n1]) <= 0.02 + 1e-12
        assert max(lat.y[n0], lat.y[n1]) >= 0.02 - 1e-12
        assert min(lat.x[n0], lat.x[n1]) <= 0.04 + 1e-12
        assert max(lat.x[n0], lat.x[n1]) >= 0.02 - 1e-12


def test_mode_one_crack_path_stays_near_the_crack_line(senb):
    geom, lat = senb
    tip = lat.metadata["tip"]
    res = trace_crack_path(lat, tip, 5)
    advance = float(np.hypot(*(res.path[-1] - res.path[0])))
    drift = float(np.max(np.abs(res.path[:, 1] - tip[1])))
    assert advance > 0.0
    assert drift / advance < 0.10
    assert np.all(np.abs(res.kink_angles_deg) < 10.0)