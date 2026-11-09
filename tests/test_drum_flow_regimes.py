"""IN32-D01 rotating-drum granular flow: regime map, free surface, power ledger.

The tests are layered:

* *analytic* tests pin the physics constants (Froude bands, release angle,
  cataract onset, chord angle, centrifugal surface curvature);
* *kernel* tests pin the numerics (symplectic energy conservation, dashpot
  dissipation == energy loss, Coulomb slip dissipation == energy loss, the
  friction *couple* sign, torque-omega power sign);
* *integration* tests run short drums and check the power-ledger closure and the
  regime labels.
"""
import math

import numpy as np
import pytest

from tensordem.drum_flow.kernel import (
    DrumConfig,
    DrumState,
    _elastic_energy,
    _mechanical_energy,
    make_drum_state,
    run_drum,
)
from tensordem.drum_flow.regimes import (
    FR_BANDS,
    REGIMES,
    analytic_surface_curvature,
    cataract_onset_froude,
    classify_froude,
    classify_kinematic,
    detachment_angle_deg,
    fill_to_chord_angle,
    fit_free_surface,
    free_surface_fit_quality,
    froude,
    omega_for_froude,
    simulate_point,
)


# ---------------------------------------------------------------------------
# analytic layer
# ---------------------------------------------------------------------------
def test_froude_roundtrip_and_critical_speed():
    R, g = 0.15, 9.81
    for fr in (0.1, 0.35, 1.0, 1.5):
        om = omega_for_froude(fr, R, g)
        assert froude(om, R, g) == pytest.approx(fr, rel=1e-12)
    # Fr = 1 is the critical speed omega_c = sqrt(g/R)
    assert omega_for_froude(1.0, R, g) == pytest.approx(math.sqrt(g / R), rel=1e-12)


def test_froude_bands_are_complete_and_ordered():
    assert [b[2] for b in FR_BANDS] == list(REGIMES)
    for (lo, hi, _), (lo2, _, _) in zip(FR_BANDS, FR_BANDS[1:]):
        assert hi == lo2


@pytest.mark.parametrize("fr,label", [
    (0.10, "slipping"), (0.29, "slipping"),
    (0.30, "rolling"), (0.69, "rolling"),
    (0.70, "cataracting"), (0.99, "cataracting"),
    (1.00, "centrifuging"), (1.50, "centrifuging"),
])
def test_classify_froude(fr, label):
    assert classify_froude(fr) == label


def test_classify_froude_rejects_negative():
    with pytest.raises(ValueError):
        classify_froude(-0.1)


def test_detachment_angle_matches_analytic():
    # theta_d = arccos(-Fr), measured from the 6 o'clock position
    assert detachment_angle_deg(0.0) == pytest.approx(90.0, abs=1e-9)
    assert detachment_angle_deg(0.5) == pytest.approx(120.0, abs=1e-9)
    assert detachment_angle_deg(0.7071067811865476) == pytest.approx(135.0, abs=1e-6)
    assert detachment_angle_deg(1.0) == pytest.approx(180.0, abs=1e-9)
    assert detachment_angle_deg(1.4) == pytest.approx(180.0, abs=1e-9)


def test_cataract_onset_matches_band_edge_within_10pct():
    """Criterion (d) anchor: 45 deg-above-horizontal release -> Fr = cos(135 deg)."""
    fr_c = cataract_onset_froude(135.0)
    assert fr_c == pytest.approx(0.7071067811865476, rel=1e-12)
    # prescribed cataracting lower edge is 0.7 -> deviation must be < 10 %
    assert abs(fr_c - 0.7) / 0.7 < 0.10


def test_fill_to_chord_angle_limit_and_half_fill():
    assert fill_to_chord_angle(0.5) == pytest.approx(math.pi / 2.0, rel=1e-6)
    assert fill_to_chord_angle(1e-6) < 0.02
    # area identity: segment area fraction reproduced from the angle
    for fill in (0.1, 0.3, 0.45):
        a = fill_to_chord_angle(fill)
        assert (2.0 * a - math.sin(2.0 * a)) / (2.0 * math.pi) == pytest.approx(fill, rel=1e-8)


def test_fit_free_surface_recovers_synthetic_parabola():
    rng = np.random.default_rng(7)
    x = rng.uniform(-0.12, 0.12, 4000)
    y_surface = 0.02 + 0.35 * x + 1.2 * x ** 2
    # fill the bed below the surface with scattered particles
    depth = rng.uniform(0.0, 0.08, x.size)
    y = y_surface - depth
    fit = fit_free_surface(np.column_stack([x, y]), n_bins=16)
    assert fit["r2"] > 0.999
    assert fit["a"] == pytest.approx(0.02, abs=2e-3)
    assert fit["b"] == pytest.approx(0.35, abs=2e-2)
    assert fit["c"] == pytest.approx(1.2, rel=0.10)


def test_fit_free_surface_straight_incline_has_zero_curvature():
    x = np.linspace(-0.1, 0.1, 3000)
    y = -0.02 + 0.6 * x - np.linspace(0.0, 0.05, x.size)
    fit = fit_free_surface(np.column_stack([x, y]), n_bins=16)
    assert abs(fit["c"]) < 0.5
    assert fit["r2"] > 0.99


def test_analytic_surface_curvature_limits_and_sign():
    g = 9.81
    assert analytic_surface_curvature(0.0, g, -0.05) == 0.0
    c = analytic_surface_curvature(4.0, g, -0.05)
    assert c > 0.0
    # matches the exact circular-arc expansion: the arc has curvature
    # 1/(g/w^2 - y_s) and y = y_s + c x^2 needs 2c to equal it
    w2 = 16.0
    assert c == pytest.approx(0.5 / (g / w2 - (-0.05)), rel=1e-12)
    # free_surface_fit_quality reports a relative deviation
    q = free_surface_fit_quality({"c": c * 1.05}, c)
    assert q["rel_deviation"] == pytest.approx(0.05, abs=1e-9)


def test_classify_kinematic_rules():
    th = None
    assert classify_kinematic({"airborne_fraction": 0.2, "r_p05_over_R": 0.3,
                               "wall_velocity_ratio": 0.9}, th) == "cataracting"
    assert classify_kinematic({"airborne_fraction": 0.0, "r_p05_over_R": 0.85,
                               "wall_velocity_ratio": 0.95}, th) == "centrifuging"
    assert classify_kinematic({"airborne_fraction": 0.0, "r_p05_over_R": 0.2,
                               "wall_velocity_ratio": 0.1}, th) == "slipping"
    assert classify_kinematic({"airborne_fraction": 0.0, "r_p05_over_R": 0.2,
                               "wall_velocity_ratio": 0.8}, th) == "rolling"


# ---------------------------------------------------------------------------
# kernel numerics
# ---------------------------------------------------------------------------
def _two_disk_state(cfg, vel, overlap=1e-5, sep_axis="x"):
    r = cfg.particle_radius
    d = 2.0 * r - overlap
    pos = np.array([[0.0, 0.0], [d, 0.0]]) if sep_axis == "x" else np.array([[0.0, 0.0], [0.0, d]])
    return DrumState(pos=pos, vel=np.array(vel), spin=np.zeros(2), omega=0.0, config=cfg)


def test_elastic_energy_is_zero_for_separated_particles():
    cfg = DrumConfig()
    pos = np.array([[0.0, 0.0], [0.05, 0.0]])
    u = _elastic_energy(cfg, pos, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                        np.zeros((0, 2)), np.zeros((2, 2)), cfg.particle_radius)
    assert u == 0.0


def test_normal_spring_conserves_energy_without_damping():
    """e -> 1, no gravity, no friction: K + U_el must be conserved by the integrator."""
    cfg = DrumConfig(gravity=1e-9, restitution=1.0, mu=0.0, mu_wall=0.0, mu_rolling=0.0, dt=2e-5)
    st = _two_disk_state(cfg, [[0.25, 0.0], [-0.25, 0.0]])
    pi = np.array([0], dtype=np.int64)
    pj = np.array([1], dtype=np.int64)
    u_start = _elastic_energy(cfg, st.pos, pi, pj, np.zeros((1, 2)), np.zeros((2, 2)),
                              cfg.particle_radius)
    k_start = _mechanical_energy(cfg, st.pos, st.vel, st.spin)
    out = run_drum(st, revolutions=0.0, measure_last=0.0, sample_every=100, settle_time=0.006)
    e1 = out["mean"]["ke"]
    # total mechanical energy (kinetic + stored spring) is conserved while the
    # undamped pair separates and coasts
    assert abs(e1 - (k_start + u_start)) / (k_start + u_start) < 0.05
    assert e1 > 0.0                      # the pair separated and carries kinetic energy
    assert out["ledger"]["window_s"] > 0.0



def test_normal_dashpot_dissipation_matches_energy_loss():
    cfg = DrumConfig(gravity=1e-9, restitution=0.5, mu=0.0, mu_wall=0.0, mu_rolling=0.0, dt=2e-5)
    st = _two_disk_state(cfg, [[0.2, 0.0], [-0.2, 0.0]])
    out = run_drum(st, revolutions=0.0, measure_last=0.0, sample_every=50, settle_time=0.01)
    led = out["ledger"]
    assert led["phi_n_w"] > 0.0
    # dE/dt must equal -Phi_diss for a purely normal-damped pair
    assert abs(led["dE_w"] + led["phi_n_w"]) / led["phi_n_w"] < 0.05


def test_coulomb_slip_dissipation_matches_energy_loss():
    cfg = DrumConfig(gravity=1e-9, restitution=1.0, mu=0.5, mu_wall=0.0, mu_rolling=0.0, dt=2e-5)
    st = _two_disk_state(cfg, [[0.0, 0.3], [0.0, -0.3]])
    out = run_drum(st, revolutions=0.0, measure_last=0.0, sample_every=50, settle_time=0.01)
    led = out["ledger"]
    assert led["phi_t_w"] > 0.0
    assert abs(led["dE_w"] + led["phi_t_w"]) / led["phi_t_w"] < 0.10


def test_friction_couple_spins_both_disks_the_same_way():
    """Regression: the tangential couple must give tau_i = tau_j, not opposites.

    A sign flip here pumps angular momentum (the bed "boils" and the shell no
    longer grips) - it was the root cause of the first dilute-bed runs.
    """
    cfg = DrumConfig(gravity=1e-9, restitution=1.0, mu=0.5, mu_wall=0.0, mu_rolling=0.0, dt=2e-5)
    st = _two_disk_state(cfg, [[0.0, 0.3], [0.0, -0.3]])
    out = run_drum(st, revolutions=0.0, measure_last=0.0, sample_every=200, settle_time=0.002)
    spin = out["snapshot"]["spin"]
    assert spin[0] * spin[1] > 0.0          # same sign => a couple, not a spin pump
    assert abs(spin[0] - spin[1]) < 1e-6 * max(1.0, abs(spin[0]))


def test_rotating_shell_injects_positive_power_and_closes_ledger():
    cfg = DrumConfig(particle_radius=0.01, dt=6e-5)
    st = make_drum_state(cfg, 0.3, omega_for_froude(0.4, cfg.radius, cfg.gravity))
    out = run_drum(st, revolutions=1.6, measure_last=1.0, sample_every=200, settle_time=0.1)
    led = out["ledger"]
    assert led["p_in_w"] > 0.0
    assert led["phi_diss_w"] > 0.0
    assert led["tau_drum_nm"] < 0.0          # shell gives torque, load resists
    # a 1.6-revolution window is not yet statistically steady, so only the
    # order of magnitude is asserted here; the 2 % acceptance criterion is a
    # steady-state claim evaluated over the whole phase map (see
    # artifacts/drum_flow/verdict.json, criterion (c)).
    assert led["closure_rel"] < 0.20         # |P_in - Phi_diss| / P_in
    assert led["energy_balance_rel"] < 0.25  # dE/dt == P_in - Phi_diss
    # the local (micro-benchmark) accounting is exact to 5-10 %, see the
    # dashpot / Coulomb tests above


def test_make_drum_state_wallows_fill_and_returns_n_particles():
    cfg = DrumConfig(particle_radius=0.01)
    st = make_drum_state(cfg, 0.3, 3.0)
    r_norm = np.linalg.norm(st.pos, axis=1) / cfg.radius
    assert r_norm.max() <= 1.0
    assert st.pos.shape[0] > 40
    with pytest.raises(ValueError):
        make_drum_state(cfg, 1.5, 3.0)


def test_simulate_point_reports_all_criteria_fields():
    cfg = DrumConfig(particle_radius=0.01, dt=6e-5)
    res = simulate_point(fill=0.3, fr=0.5, cfg=cfg, revolutions=1.4, measure_last=1.0,
                         settle_time=0.1, sample_every=200)
    assert res["regime_froude"] in REGIMES
    assert res["regime_kinematic"] in REGIMES
    assert 0.0 <= res["observables"]["airborne_fraction"] <= 1.0
    for key in ("p_in_w", "phi_diss_w", "closure_rel", "tau_drum_nm"):
        assert key in res["ledger"]
    assert math.isfinite(res["detachment_angle_deg"])
    assert res["surface"]["r2_all_particles"] > 0.50   # a surface must be resolvable
    assert res["surface"]["analytic"]["c_analytic"] >= 0.0