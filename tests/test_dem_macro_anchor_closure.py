"""IN29-B1 closure tests: Mindlin history contact, single rolling law, sandpile protocol.

The tests are grouped by the campaign's machine criteria:

* C8 -- prior guard / no calibration (frozen config, fail-closed whitelist);
* C6 -- energy ledger (undamped drift, damped closure);
* contact anchors -- Mindlin loading/unloading backbone, Coulomb cap, stick-slip
  loop dissipation, and the equivalence with Cundall-Strack;
* single rolling implementation -- `dem3d` and `repose_calibration` migrate onto
  `tensordem.rolling_resistance` without changing a single number;
* measurement operators -- synthetic 30 deg cone and analytic packings;
* protocol smoke -- a small pour produces finite anchors and conserves grains.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem import rolling_resistance as rr
from tensordem.contact_tangential import (
    MindlinTangentialConfig,
    advance_tangential_contact,
    mindlin_backbone,
    mindlin_tangential_step,
    rolling_resistance_si3,
)
from tensordem.protocols.sandpile import (
    PRIORS,
    SandpileConfig,
    SandpileProtocolError,
    SandpileRun,
    assert_priors,
    bulk_packing_fraction,
    contact_fabric,
    core_packing_fraction,
    free_surface_profile,
    pair_energy_ledger,
    probe_packing_fraction,
    repose_angle_from_profile,
)

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# C8 -- the priors cannot be calibrated
# ---------------------------------------------------------------------------

def test_config_is_frozen_and_priors_match_the_file():
    cfg = SandpileConfig(n_particles=64)
    assert cfg.mu_s == PRIORS["mu_s"]
    assert cfg.mu_r == PRIORS["mu_r"]
    with pytest.raises(Exception):
        cfg.mu_r = 0.30  # frozen dataclass: runtime write protection
    record = cfg.as_record()
    assert record["calibration_disabled"] is True
    assert record["priors"] == PRIORS


@pytest.mark.parametrize("mu_s,mu_r", [(0.5, 0.30), (0.6, 0.05), (0.5, 0.0)])
def test_baseline_run_refuses_non_prior_coefficients(mu_s, mu_r):
    cfg = SandpileConfig(n_particles=64, mu_s=mu_s, mu_r=mu_r)
    with pytest.raises(SandpileProtocolError):
        assert_priors(cfg)
    with pytest.raises(SandpileProtocolError):
        SandpileRun(cfg)  # the guard runs before the first step


def test_controls_are_whitelisted_and_tagged():
    for control, expected in (("mu_s_zero", (0.0, PRIORS["mu_r"])),
                              ("mu_r_zero", (PRIORS["mu_s"], 0.0))):
        kwargs = dict(n_particles=64, control=control)
        if control == "mu_s_zero":
            kwargs["mu_s"] = 0.0
        else:
            kwargs["mu_r"] = 0.0
        cfg = SandpileConfig(**kwargs)
        assert_priors(cfg)
        assert (cfg.mu_s, cfg.mu_r) == expected
        assert cfg.kind == "negative_control"
        record = cfg.as_record()
        assert record["kind"] == "negative_control"
        # an undeclared mutation is still refused
        mutated = SandpileConfig(**{**kwargs, "mu_r": 0.2 if control == "mu_s_zero" else 0.2})
        with pytest.raises(SandpileProtocolError):
            assert_priors(mutated)


def test_instant_fill_requires_its_control_key():
    with pytest.raises(SandpileProtocolError):
        SandpileConfig(n_particles=64, pour_rate=math.inf)


# ---------------------------------------------------------------------------
# C6 -- energy ledger
# ---------------------------------------------------------------------------

def test_energy_ledger_undamped_drift_and_damped_closure():
    undamped = SandpileConfig(n_particles=64, control="undamped", restitution=1.0,
                              mu_s=0.0, mu_r=0.0, dt=2.0e-5)
    res = pair_energy_ledger(undamped, steps=100_000, closing_speed=0.05)
    assert abs(res["drift_relative"]) < 1e-3, res  # < 0.1 %

    damped = SandpileConfig(n_particles=64, dt=1.0e-6)
    res = pair_energy_ledger(damped, steps=100_000, closing_speed=0.05)
    assert res["dissipated_total"] > 0.0
    assert abs(res["ledger_residual"]) < 1.0e-6, res


# ---------------------------------------------------------------------------
# Mindlin history contact model
# ---------------------------------------------------------------------------

def _drive(steps, dt, v, f_n, config, state=None):
    """Run a constant-slip driver and return (forces, state)."""
    if state is None:
        zeros = torch.zeros((1, 3), dtype=DTYPE)
        state = (zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone())
    forces = []
    for _ in range(steps):
        out = mindlin_tangential_step(*state, v, f_n, config, dt)
        forces.append(float(out[0][0, 0]))
        state = out[1:5]
    return forces, state


def test_linear_backbone_reduces_to_cundall_strack_without_dashpot():
    k_t0, mu, dt, f_n = 5.0e3, 0.5, 1.0e-4, 2.0
    cfg = MindlinTangentialConfig(k_t0=k_t0, mu_s=mu, backbone="linear")
    v = torch.tensor([[0.01, 0.0, 0.0]], dtype=DTYPE)
    force_n = torch.tensor([f_n], dtype=DTYPE)
    mindlin, state = _drive(30, dt, v, force_n, cfg)
    # Cundall-Strack in this module stores +k_t * delta (the force the spring
    # exerts on the pair), the Mindlin model stores the force opposing the slip;
    # the two agree in magnitude and differ by the sign convention.
    disp = torch.zeros((1, 3), dtype=DTYPE)
    cs = []
    for _ in range(30):
        f_t, disp, _sl = advance_tangential_contact(
            disp, v, torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE), force_n,
            k_t0, mu, dt)
        cs.append(float(f_t[0, 0]))
    assert np.allclose(mindlin, [-c for c in cs], rtol=1e-12, atol=1e-12), (mindlin, cs)


def test_virgin_loading_follows_the_mindlin_backbone():
    k_t0, mu, f_n = 5.0e3, 0.5, 2.0
    cfg = MindlinTangentialConfig(k_t0=k_t0, mu_s=mu, backbone="mindlin")
    delta = torch.tensor([0.0, 1.0e-5, 5.0e-5, 2.0e-4, 1.0e-3], dtype=DTYPE)
    env = mindlin_backbone(delta, torch.full_like(delta, f_n), cfg)
    cap = mu * f_n
    d_star = 3.0 * cap / (2.0 * k_t0)
    analytic = cap * (1.0 - (1.0 - torch.clamp(delta / d_star, max=1.0)) ** 1.5)
    assert torch.allclose(env, analytic, rtol=1e-12)
    assert env[0] == 0.0 and env[-1] == pytest.approx(cap)


def test_unloading_branch_slope_is_the_no_slip_stiffness():
    """Unloading must be elastic with slope k_t0 -- no ratcheting."""
    k_t0, mu, dt, f_n = 5.0e3, 0.5, 1.0e-5, 2.0
    cfg = MindlinTangentialConfig(k_t0=k_t0, mu_s=mu, backbone="linear")
    v_up = torch.tensor([[1.0e-2, 0.0, 0.0]], dtype=DTYPE)
    force_n = torch.tensor([f_n], dtype=DTYPE)
    # load until sliding, then reverse
    _f, state = _drive(3000, dt, v_up, force_n, cfg)   # sliding at mu f_n
    cap = mu * f_n
    assert abs(abs(float(state[0][0, 0])) - cap) < 1e-9
    v_dn = -v_up
    forces, state = _drive(1, dt, v_dn, force_n, cfg, state)
    # F = F_r - k (delta - delta_r)  =>  dF/d(delta) = -k_t0 on both branches
    slope = (forces[0] - (-cap)) / (float(v_dn[0, 0]) * dt)
    assert abs(slope) == pytest.approx(k_t0, rel=1e-9)


def test_stick_slip_loop_closes_and_dissipates_the_analytic_amount():
    """Steady loop area = 4 mu F_n (a - mu F_n / k_t0), the Coulomb anchor."""
    k_t0, mu, f_n = 4.0e3, 0.5, 1.0
    dt, speed = 2.0e-6, 1.0e-2
    amp = 1.0e-3                       # displacement amplitude >> delta_s
    cfg = MindlinTangentialConfig(k_t0=k_t0, mu_s=mu, backbone="linear")
    force_n = torch.tensor([f_n], dtype=DTYPE)
    zeros = torch.zeros((1, 3), dtype=DTYPE)
    state = (zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone())
    n_step = int(amp / (speed * dt))
    work = 0.0
    # closed triangular loop: +a, then -2a, then +a (back to the start)
    for sign, n in ((1.0, n_step), (-1.0, 2 * n_step), (1.0, n_step)):
        v = torch.tensor([[sign * speed, 0.0, 0.0]], dtype=DTYPE)
        for _ in range(n):
            out = mindlin_tangential_step(*state, v, force_n, cfg, dt)
            work += float(out[0][0, 0]) * sign * speed * dt
            state = out[1:5]
    analytic = 4.0 * mu * f_n * (amp - mu * f_n / k_t0)
    # the discrete triangular driver starts from the virgin state, so the loop
    # carries a one-off transient: 10 % tolerance, and the sign is what matters
    assert work < 0.0
    assert abs(work) == pytest.approx(analytic, rel=0.10), (abs(work), analytic)


def test_tangential_dashpot_dissipates_the_viscous_work():
    k_t0, mu, dt = 4.0e3, 1.0e9, 1.0e-5   # huge mu -> pure stick
    gamma_t = 2.0
    cfg = MindlinTangentialConfig(k_t0=k_t0, mu_s=0.0, backbone="linear", gamma_t=gamma_t)
    v = torch.tensor([[0.02, 0.0, 0.0]], dtype=DTYPE)
    force_n = torch.tensor([1.0], dtype=DTYPE)
    steps = 100
    forces, _ = _drive(steps, dt, v, force_n, cfg)
    visc = abs(sum(forces)) * float(v[0, 0]) * dt
    expected = gamma_t * float(v[0, 0]) ** 2 * dt * steps
    assert visc == pytest.approx(expected, rel=1e-12)


def test_separated_contact_forgets_history():
    cfg = MindlinTangentialConfig(k_t0=1.0e4, mu_s=0.5, backbone="mindlin")
    zeros = torch.zeros((1, 3), dtype=DTYPE)
    v = torch.tensor([[0.1, 0.0, 0.0]], dtype=DTYPE)
    force = torch.tensor([1.0], dtype=DTYPE)
    out = mindlin_tangential_step(zeros, zeros, zeros, zeros, v, force, cfg, 1e-4)
    assert float(out[0].abs().max()) > 0.0
    # zero normal force -> no force and no state
    out2 = mindlin_tangential_step(*(t.clone() for t in out[1:5]), v,
                                   torch.tensor([0.0], dtype=DTYPE), cfg, 1e-4)
    assert float(out2[0].abs().max()) == 0.0
    assert float(out2[1].abs().max()) == 0.0
    assert float(out2[2].abs().max()) == 0.0


def test_config_rejects_bad_parameters():
    with pytest.raises(ValueError):
        MindlinTangentialConfig(k_t0=0.0)
    with pytest.raises(ValueError):
        MindlinTangentialConfig(k_t0=1.0, mu_s=-0.1)
    with pytest.raises(ValueError):
        MindlinTangentialConfig(k_t0=1.0, backbone="masing")
    with pytest.raises(ValueError):
        MindlinTangentialConfig(k_t0=1.0, unloading_ratio=0.5)


# ---------------------------------------------------------------------------
# Single rolling implementation (migration tests)
# ---------------------------------------------------------------------------

def _legacy_dem3d_couple(relative_v, normal, contact_mag, mu_r, R):
    """Frozen copy of the pre-migration ``dem3d._rolling_resistance_couple``."""
    vn = (relative_v * normal).sum(dim=1, keepdim=True)
    v_tangent = relative_v - vn * normal
    v_t_mag = torch.linalg.vector_norm(v_tangent, dim=1, keepdim=True)
    slip_hat = torch.where(
        v_t_mag > 1e-15,
        v_tangent / v_t_mag.clamp_min(torch.finfo(relative_v.dtype).eps),
        torch.zeros_like(v_tangent))
    f_mag = (mu_r * contact_mag)[:, None]
    force_j = -f_mag * slip_hat
    force_i = -force_j
    torque_i = R * torch.linalg.cross(normal, force_i)
    torque_j = -torque_i
    return force_i, force_j, torque_i, torque_j


def test_dem3d_rolling_couple_uses_the_shared_law():
    torch.manual_seed(0)
    n = 32
    v_rel = torch.randn(n, 3, dtype=DTYPE)
    normal = torch.nn.functional.normalize(torch.randn(n, 3, dtype=DTYPE), dim=1)
    f_n = torch.rand(n, dtype=DTYPE) * 10.0
    mu_r, R = 0.05, 0.005
    legacy = _legacy_dem3d_couple(v_rel, normal, f_n, mu_r, R)
    shared = rr.rolling_resistance_couple(v_rel, normal, f_n, R, mu_r=mu_r)
    for a, b in zip(legacy, shared):
        assert torch.allclose(a, b, rtol=1e-13, atol=1e-15)


def _legacy_epsd(spin, f_n, mu_r, k_r, r, roll, dt):
    """Frozen copy of the pre-migration inline EPSD block of repose_calibration."""
    roll_new = roll + spin * dt
    roll_norm = torch.linalg.vector_norm(roll_new, dim=1)
    cap = (mu_r * r * f_n) / k_r
    scale = torch.where(roll_norm > cap, cap / roll_norm.clamp_min(1e-30),
                        torch.ones_like(roll_norm))
    roll_new = roll_new * scale[:, None]
    return -k_r * roll_new, roll_new


def test_repose_calibration_epsd_matches_the_legacy_formula():
    from tensordem.repose_calibration import _epsd_rolling_torque
    torch.manual_seed(1)
    n = 64
    spin = torch.randn(n, 3, dtype=DTYPE)
    f_n = torch.rand(n, dtype=DTYPE) * 5.0
    roll = torch.randn(n, 3, dtype=DTYPE) * 1e-3
    mu_r, k_r, r, dt = 0.2, 0.05, 0.005, 1e-4
    torque_legacy, roll_legacy = _legacy_epsd(spin, f_n, mu_r, k_r, r, roll, dt)
    torque_new, roll_new = _epsd_rolling_torque(spin, f_n, mu_r, k_r, r, roll, dt)
    assert torch.allclose(torque_new, torque_legacy, rtol=1e-12, atol=1e-18)
    assert torch.allclose(roll_new, roll_legacy, rtol=1e-12, atol=1e-18)


def test_si3_mount_respects_the_rolling_limit_and_is_dissipative():
    n = 16
    spin = torch.tensor([[50.0, 0.0, 0.0]] * n, dtype=DTYPE)
    normal = torch.tensor([[0.0, 0.0, 1.0]] * n, dtype=DTYPE)
    f_n = torch.linspace(1.0, 5.0, n, dtype=DTYPE)
    r_eff = torch.full((n,), 0.0025, dtype=DTYPE)
    mu_r, k_r, dt = 0.05, 0.01, 1e-4
    state = torch.zeros((n, 3), dtype=DTYPE)
    for _ in range(50):
        torque, state, power = rolling_resistance_si3(
            spin, normal, f_n, r_eff, mu_r=mu_r, k_r=k_r, dt=dt, torque_state=state)
        limit = mu_r * f_n * r_eff
        assert torch.all(torch.linalg.vector_norm(torque, dim=1) <= limit + 1e-15)
        assert torch.all(power <= 1e-18)


# ---------------------------------------------------------------------------
# Measurement operators
# ---------------------------------------------------------------------------

def _synthetic_cone(angle_deg: float, radius: float = 0.30, h_max: float = None,
                    grain: float = 0.005, layers: float = 0.0085):
    tan = math.tan(math.radians(angle_deg))
    h_max = tan * radius if h_max is None else h_max
    pts = []
    for rad in np.arange(0.02, radius, 0.008):
        h = h_max - rad * tan
        n = max(2, int(2.0 * math.pi * rad / (2.0 * grain) * 0.6))
        for k in range(n):
            th = 2.0 * math.pi * k / n
            for z in np.arange(grain, h, layers):
                pts.append((rad * math.cos(th), rad * math.sin(th), z))
    return np.asarray(pts)


@pytest.mark.parametrize("angle", [20.0, 30.0, 35.0])
def test_repose_operator_recovers_a_synthetic_cone(angle):
    pts = _synthetic_cone(angle)
    profile = free_surface_profile(pts, 0.005)
    fit = repose_angle_from_profile(profile)
    assert fit["theta_deg"] == pytest.approx(angle, abs=0.9)
    assert fit["r2"] > 0.98


def _cubic_packing(n_cells: int, r: float = 0.005):
    pts = [(i, j, k) for i in range(n_cells) for j in range(n_cells)
           for k in range(n_cells)]
    return np.asarray(pts, dtype=np.float64) * 2.0 * r


def _hcp_packing(n_cells: int, r: float = 0.005):
    """Ideal hcp: triangular layers of pitch 2r, stacked with the B-site shift."""
    a = 2.0 * r
    dz = a * math.sqrt(2.0 / 3.0)
    pts = []
    for layer in range(int(round(n_cells * 1.15))):
        z = layer * dz
        shift_y = (a / math.sqrt(3.0)) if layer % 2 else 0.0
        for i in range(n_cells):
            for j in range(n_cells):
                pts.append((i * a + (a / 2.0) * (j % 2), j * a * math.sqrt(3.0) / 2.0 + shift_y, z))
    return np.asarray(pts)


def test_packing_operators_recover_analytic_packings():
    """Voronoi operator is unbiased; the probe cylinder carries a boundary bias."""
    from tensordem.protocols.sandpile import voronoi_packing_fraction
    cases = ((_cubic_packing(10), math.pi / 6.0),
             (_hcp_packing(7), math.pi / (3.0 * math.sqrt(2.0))))
    for pts, expected in cases:
        vor = voronoi_packing_fraction(pts, 0.005, sample_radius=0.05)
        assert vor["packing_fraction"] == pytest.approx(expected, rel=0.04), (expected, vor)
        probe = probe_packing_fraction(pts, 0.005, probe_radius=0.05,
                                       probe_height=0.08)
        # the probe's surface layer is a known O(r/R) bias: bounded but present
        assert abs(probe["packing_fraction"] / expected - 1.0) < 0.45


def test_core_probe_and_voronoi_operators_agree_on_a_thick_deposit():
    from tensordem.protocols.sandpile import voronoi_packing_fraction
    pts = _synthetic_cone(30.0)
    profile = free_surface_profile(pts, 0.005)
    probe = probe_packing_fraction(pts, 0.005)["packing_fraction"]
    core = core_packing_fraction(pts, 0.005, profile=profile)["packing_fraction"]
    vor = voronoi_packing_fraction(pts, 0.005, sample_radius=0.06)["packing_fraction"]
    assert 0.20 < vor < 0.80
    assert abs(core - vor) < 0.30 and abs(probe - vor) < 0.30


def test_fabric_operator_is_zero_for_isotropic_normals():
    gen = torch.Generator().manual_seed(7)
    normals = torch.nn.functional.normalize(
        torch.randn(4000, 3, dtype=DTYPE, generator=gen), dim=1)
    fabric = contact_fabric(normals, n_particles=1000)
    assert fabric["fabric_anisotropy"] < 0.05
    assert fabric["coordination_number"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Protocol smoke test
# ---------------------------------------------------------------------------

def test_small_pour_produces_finite_anchors():
    cfg = SandpileConfig(n_particles=450, pour_rate=900.0, settle_time=0.15,
                         restitution=0.3, nozzle_radius=0.03, drop_height=0.04)
    record = SandpileRun(cfg).run()
    assert record["poured"] == 450
    assert math.isfinite(record["repose_deg"])
    assert -1.0 < record["repose_deg"] < 90.0
    assert record["repose_fit"]["r2"] > 0.5
    assert record["packing"]["core"]["packing_fraction"] > 0.1
    assert record["priors_used"] == PRIORS
    assert record["max_overlap_ratio"] < 0.5   # no interpenetration blow-up
    assert record["max_speed_m_s"] < 5.0
    assert record["energy"]["dissipated_J"] > 0.0
    # anchors are consistent with the analytic rolling bound theta <= atan(mu_r)
    assert record["repose_deg"] < math.degrees(math.atan(PRIORS["mu_r"])) + 12.0


def test_sandpile_rejects_a_narrow_nozzle():
    with pytest.raises(SandpileProtocolError):
        SandpileConfig(n_particles=64, nozzle_radius=0.006)