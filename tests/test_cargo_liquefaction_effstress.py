"""IN60-N09 acceptance tests: pendular bridge -> r_u -> phi_eff -> shift -> GZ.

Criteria (campaign node IN60-N09):

G1  single-bridge force matches the analytic/Rouault capillary law (< 5 %)
G2  dry-sand repose angle against the 32 deg experiment (|delta| < 3 deg)
G3  phi_eff(MR) rises then falls, peak inside MR/FMP in [0.80, 0.95],
    negative control r_u -> 1 gives phi_eff -> 0
G4  force-energy consistency residual < 5 %
G5  threshold curve monotone decreasing on the hazard window and timestep
    halving changes the threshold by < 10 %
G6  comparability statement against the failed predecessor version

G2 is reported honestly: the absolute dry repose angle is *not* reproduced by
either particle engine (see docs/IN60_N09_cargo_liquefaction_effstress.md §4.2),
so the test asserts the physically meaningful part that the DEM can establish —
that the pendular bridges raise the measured flank angle monotonically in the
cohesion number — and the campaign records the criterion as an honest FAIL.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import cargo_liquefaction_effstress as cl
from tensordem.pendular_pile2d import DiskPile

R = 1.0e-3
GAMMA = cl.WATER_SURFACE_TENSION
F0 = 2.0 * math.pi * GAMMA * R


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------

def test_g1_derjaguin_limit_reproduces_two_pi_gamma_R():
    for v_ratio in (1e-4, 1e-2, 1.0):
        f = cl.rouault_pendular_force(R, GAMMA, v_ratio * R ** 3, 0.0)
        assert f == pytest.approx(F0, rel=5e-3)


def test_g1_kernel_matches_the_hand_written_rouault_formula():
    """Independent transliteration of the documented closed form (<1e-12)."""
    worst = 0.0
    for v_ratio in (0.005, 0.02, 0.1):
        for d_ratio in (0.0, 0.01, 0.05, 0.2):
            v = v_ratio * R ** 3
            gap = d_ratio * R
            a_eq = (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)
            s = gap / (2.0 * a_eq)
            hand = F0 / (1.0 + 1.05 * s + 2.5 * s * s)
            got = cl.rouault_pendular_force(R, GAMMA, v, gap)
            worst = max(worst, abs(got - hand) / hand)
    assert worst < 1e-12


def test_g1_contact_angle_scaling_and_non_wetting_limit():
    for theta in (0.0, 30.0, 45.0):
        f = cl.rouault_pendular_force(R, GAMMA, 0.02 * R ** 3, 0.0, theta)
        assert f == pytest.approx(F0 * math.cos(math.radians(theta)), rel=1e-12)
    assert abs(cl.rouault_pendular_force(R, GAMMA, 0.02 * R ** 3, 0.0, 90.0)) < 1e-18
    assert cl.rouault_pendular_force(R, GAMMA, 0.02 * R ** 3, 0.0, 120.0) == 0.0


def test_g1_monotone_in_gap_and_volume():
    base = cl.rouault_pendular_force(R, GAMMA, 0.02 * R ** 3, 0.0)
    assert base > cl.rouault_pendular_force(R, GAMMA, 0.02 * R ** 3, 0.05 * R)
    assert cl.rouault_pendular_force(R, GAMMA, 0.02 * R ** 3, 0.05 * R) > \
        cl.rouault_pendular_force(R, GAMMA, 0.02 * R ** 3, 0.1 * R)
    assert cl.rouault_pendular_force(R, GAMMA, 0.1 * R ** 3, 0.02 * R) > \
        cl.rouault_pendular_force(R, GAMMA, 0.01 * R ** 3, 0.02 * R)


def test_g1_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        cl.rouault_pendular_force(-1.0, GAMMA, 1e-12, 0.0)
    with pytest.raises(ValueError):
        cl.rouault_pendular_force(R, GAMMA, -1e-12, 0.0)
    with pytest.raises(ValueError):
        cl.rouault_pendular_force(R, GAMMA, 1e-12, -1e-6)


# ---------------------------------------------------------------------------
# G4
# ---------------------------------------------------------------------------

def test_g4_force_is_the_negative_gradient_of_the_potential():
    worst = max(
        cl.force_energy_derivative_residual(R, GAMMA, 0.03 * R ** 3, d * R)
        for d in (1e-3, 0.01, 0.05, 0.2, 0.5)
    )
    assert worst < 1e-6


def test_g4_separation_energy_ledger_closes_within_five_percent():
    mass = 4.0 / 3.0 * math.pi * R ** 3 * 2650.0
    out = cl.separate_pair_energy_budget(
        mass=mass, radius=R, gamma=GAMMA, volume=0.03 * R ** 3,
        gap0=0.02 * R, velocity0=0.5, dt=2e-7, n_steps=400,
    )
    assert out["residual_energy"] < 0.05
    assert out["residual_work_vs_potential"] < 0.05
    assert out["work_by_capillary"] > 0.0


def test_g4_potential_is_positive_and_decreasing_outwards():
    u0 = cl.rouault_pendular_energy(R, GAMMA, 0.03 * R ** 3, 0.0)
    u1 = cl.rouault_pendular_energy(R, GAMMA, 0.03 * R ** 3, 0.1 * R)
    assert u0 > u1 > 0.0


# ---------------------------------------------------------------------------
# effective stress / G3
# ---------------------------------------------------------------------------

def test_effective_stress_reduction_limits():
    assert cl.apply_effective_stress(10.0, 0.0) == 10.0
    assert cl.apply_effective_stress(10.0, 0.25) == pytest.approx(7.5)
    assert cl.apply_effective_stress(10.0, 1.0) == 0.0
    assert cl.apply_effective_stress(10.0, 2.0) == 0.0


def test_g3_phi_eff_rises_then_falls_with_peak_in_the_required_window():
    matrix = cl.CargoMatrix()
    rows = cl.moisture_scan(matrix, np.arange(0.0, 1.0001, 0.01))
    phi = np.array([r["phi_eff_deg"] for r in rows])
    sat = np.array([r["saturation"] for r in rows])
    i = int(np.argmax(phi))
    assert 0.80 <= sat[i] <= 0.95
    assert phi[i] > phi[0] + 3.0            # a real strengthening branch
    assert np.all(np.diff(phi[: i + 1]) >= -1e-9)
    assert np.all(np.diff(phi[i:]) <= 1e-9)
    assert phi[-1] == pytest.approx(0.0, abs=1e-9)


def test_g3_negative_control_ru_to_one_kills_the_strength():
    matrix = cl.CargoMatrix()
    for sr in (0.5, 0.85, 0.95):
        out = matrix.effective_friction_angle(sr * matrix.full_moisture_point, 1.0)
        assert out["phi_eff_deg"] == pytest.approx(0.0, abs=1e-9)
    # and r_u -> 0 gives the dry value at zero moisture
    out = matrix.effective_friction_angle(0.0, 0.0)
    assert out["phi_eff_deg"] == pytest.approx(matrix.phi_dry_deg, rel=1e-9)


def test_full_moisture_point_consistency():
    matrix = cl.CargoMatrix()
    assert matrix.full_moisture_point == pytest.approx(
        matrix.porosity * matrix.rho_w / ((1.0 - matrix.porosity) * matrix.rho_s))
    assert matrix.saturation(matrix.full_moisture_point) == pytest.approx(1.0)
    assert matrix.saturation(2.0 * matrix.full_moisture_point) == 1.0
    assert matrix.capillary_cohesion(0.0) == 0.0


def test_cyclic_pore_pressure_is_monotone_in_cycles_and_saturation():
    matrix = cl.CargoMatrix()
    for sr in (0.9, 0.95, 1.0):
        vals = [cl.cyclic_pore_pressure_ratio(n, 0.1, sr) for n in (0, 10, 50, 200, 2000)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
        assert 0.0 <= vals[-1] <= 1.0
    low = cl.cyclic_pore_pressure_ratio(200, 0.1, 0.8)
    high = cl.cyclic_pore_pressure_ratio(200, 0.1, 0.99)
    assert low == 0.0 and high > 0.0


# ---------------------------------------------------------------------------
# bed relaxation / G5
# ---------------------------------------------------------------------------

def test_bed_relaxation_conserves_volume_and_is_symmetric():
    bed = cl.GranularBed.rectangular(12.0, 3.0, 48)
    v0 = bed.volume
    bed.relax(-18.0, 6.0)
    assert bed.volume == pytest.approx(v0, rel=1e-12)
    assert bed.heights.min() >= 0.0
    assert bed.shift() < 0.0    # psi < 0 (heeling to port) -> cargo moves to -x
    mirror = cl.GranularBed.rectangular(12.0, 3.0, 48)
    mirror.relax(18.0, 6.0)
    assert mirror.shift() == pytest.approx(-bed.shift(), rel=1e-9)


def test_bed_is_stable_below_the_friction_angle():
    bed = cl.GranularBed.rectangular(12.0, 3.0, 48)
    moved = bed.relax(-20.0, 26.0)     # |psi| < phi_eff
    assert moved == 0.0
    assert bed.shift() == pytest.approx(0.0, abs=1e-12)


def test_shift_threshold_falls_with_saturation_on_the_hazard_window():
    matrix = cl.CargoMatrix()
    kwargs = dict(shift_limit_m=0.05, cycles=20, dt=2.0, nx=24,
                  amplitude_bounds_deg=(6.0, 34.0), tol_deg=2.0)
    a = cl.critical_roll_amplitude(matrix, 0.90 * matrix.full_moisture_point, **kwargs)
    b = cl.critical_roll_amplitude(matrix, 1.00 * matrix.full_moisture_point, **kwargs)
    assert a["critical_amplitude_deg"] == a["critical_amplitude_deg"]   # not NaN
    assert b["critical_amplitude_deg"] == b["critical_amplitude_deg"]
    assert a["critical_amplitude_deg"] >= b["critical_amplitude_deg"] - 1e-9


def test_timestep_halving_changes_the_shift_by_less_than_ten_percent():
    matrix = cl.CargoMatrix()
    load = cl.RollExcitation(amplitude_deg=18.0)
    kw = dict(cycles=20, nx=32, record=False, moisture_ratio=0.99 * matrix.full_moisture_point)
    coarse = cl.simulate_roll_shift(matrix, load, dt=2.0, **kw)
    fine = cl.simulate_roll_shift(matrix, load, dt=1.0, **kw)
    a, b = abs(coarse.shift_cycle_mean_m), abs(fine.shift_cycle_mean_m)
    if max(a, b) > 1e-9:
        assert abs(a - b) / max(a, b) < 0.10


# ---------------------------------------------------------------------------
# GZ / stability
# ---------------------------------------------------------------------------

def test_gz_area_loss_limits():
    zero = cl.gz_area_loss(1.5, 0.0, 25000.0, 50000.0)
    assert zero["gz_area_loss_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert zero["residual_list_deg"] == pytest.approx(0.0, abs=1e-12)
    small = cl.gz_area_loss(1.5, 0.1, 25000.0, 50000.0)
    big = cl.gz_area_loss(1.5, 0.5, 25000.0, 50000.0)
    assert 0.0 < small["gz_area_loss_fraction"] < big["gz_area_loss_fraction"] <= 1.0
    assert big["transverse_cog_shift_m"] == pytest.approx(0.5 * 25000.0 / 50000.0)
    assert small["gz_at_30deg_m"] < zero["gz_at_30deg_m"]
    with pytest.raises(ValueError):
        cl.gz_area_loss(0.0, 0.1, 25000.0, 50000.0)


# ---------------------------------------------------------------------------
# G2: particle DEM repose bench (honest: trend, not the absolute angle)
# ---------------------------------------------------------------------------

def _flank_angle_after(steps: int, gamma: float) -> float:
    pile = DiskPile(mu=0.5, mu_rolling=0.2, gamma=gamma, bridge_volume_ratio=0.05,
                    box_width=8.0e-3, box_height=4.0e-3, n_particles=420)
    for _ in range(steps):
        pile.step()
    return pile.repose_angle_deg().get("theta_r_deg", float("nan"))


def test_g2_pendular_bridges_steepen_the_flank_monotonically():
    dry = _flank_angle_after(7000, 0.0)
    wet = _flank_angle_after(7000, 0.36)
    assert dry == dry and wet == wet          # both measured
    assert wet > dry + 1.0                    # cohesion raises the flank angle


def test_g2_cohesion_number_reported_for_the_bench():
    pile = DiskPile(mu=0.5, mu_rolling=0.2, gamma=0.072, bridge_volume_ratio=0.05)
    assert pile.cohesion_number > 0.0
    assert pile.bridge_volume == pytest.approx(0.05 * pile.radius ** 3)


# ---------------------------------------------------------------------------
# G6
# ---------------------------------------------------------------------------

def test_g6_comparability_statement_is_machine_readable():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "campaign", str(__import__("pathlib").Path(__file__).resolve().parents[1]
                        / "scripts" / "in60_n09_cargo_liquefaction.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stmt = mod.g6_comparability()
    assert stmt["failed_node"].startswith("IN39-N08")
    assert stmt["artifacts_available_in_worktree"] is False
    assert stmt["directly_comparable"] and stmt["not_comparable"]
    assert "qualitative" in stmt["status"]