"""XVAL rolling-resistance cross-validation: law identity, lever factor, torsion.

Anchor ``XVAL-DEM-ROUND2-METHOD-SELECTION`` (CASE-COLUMN-COLLAPSE round-2 method
selection vs LIGGGHTS-PUBLIC).  Every assertion here is one line of the
difference list in ``docs/XVAL_DEM_ROUND2_METHOD_SELECTION.md``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular3d_core import Material3D
from tensordem.granular_column.axisym import AxisymCase, AxisymError
from tensordem.granular_column.kernel import (PairSpringState, WallSpringState,
                                             pair_contact_forces,
                                             wall_contact_forces)
from tensordem.xval.oss_rolling import (cdt_torque, hertz_equivalent_shear_modulus,
                                        oss_epsd2, rayleigh_timestep,
                                        tensordem_kernel)
from tensordem.xval.scenes import pair_spin_relax, roll_on_plane

R = 0.03
DT = 5.0e-5


def _mat(**kw) -> Material3D:
    base = dict(radius=R, density=2500.0, mu=0.5, mu_rolling=0.5,
                restitution=0.3, kn=6.0e4, kt_ratio=0.5, gravity=9.81)
    base.update(kw)
    return Material3D(**base)


# ---------------------------------------------------------------------------
# D1 — the production kernel wall law IS the OSS EPSD2 law (same lever)
# ---------------------------------------------------------------------------
def test_kernel_wall_matches_oss_epsd2_transcription():
    mat = _mat()
    m = mat.mass
    n_load = m * 9.81
    kt = mat.kt
    inertia = mat.inertia

    def run_oss():
        model = oss_epsd2()
        model.reset(1)
        w, out = 10.0, []
        t = 0.0
        while t < 0.01:
            o = model.step(np.array([[0.0, w, 0.0]]), np.array([n_load]), R, kt,
                           0.5, DT, normal=np.array([[0.0, 0.0, 1.0]]))
            w += DT * float(o["torque"][0, 1]) / inertia
            t += DT
            out.append(w)
        return np.asarray(out)

    state = WallSpringState(1)
    pos = np.array([[0.0, 0.0, R - n_load / mat.kn]])
    omega = np.array([[0.0, 10.0, 0.0]])
    w = 10.0
    t = 0.0
    meas = []
    while t < 0.01:
        omega[0, 1] = w
        out = wall_contact_forces(pos, np.zeros((1, 3)), omega, mat, DT,
                                 mu_wall=0.0, mu_rolling_wall=0.5, state=state)
        w += DT * float(out["torque"][0, 1]) / inertia
        t += DT
        meas.append(w)
    meas = np.asarray(meas)
    ref = run_oss()
    assert meas.shape == ref.shape
    assert np.max(np.abs(meas - ref)) < 1e-9, np.max(np.abs(meas - ref))


# ---------------------------------------------------------------------------
# D2 — the lever arm: kernel uses r on a pair, LIGGGHTS uses radi*radj/(radi+radj)
# ---------------------------------------------------------------------------
def test_pair_lever_factor_is_two():
    kw = dict(radius=R, density=2500.0, kn=6.0e4, kt_ratio=0.5, mu_r=0.5,
              w0=20.0, fn=1.0, dt=DT, t_max=0.5)
    ours = pair_spin_relax(tensordem_kernel(), reff=R, **kw)
    oss = pair_spin_relax(oss_epsd2(), reff=0.5 * R, **kw)
    kw_mapped = dict(kw); kw_mapped["mu_r"] = 0.25
    mapped = pair_spin_relax(tensordem_kernel(), reff=R, **kw_mapped)
    ratio = oss["t_half"] / ours["t_half"]
    assert 1.95 < ratio < 2.05, ratio
    # the mapping mu_r -> mu_r / 2 recovers the OSS relaxation time
    assert abs(mapped["t_half"] - oss["t_half"]) / oss["t_half"] < 0.01


def test_pair_contact_forces_use_the_convention():
    """A pair contact's rolling torque follows ``mu_r * reff * f_n``."""
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 * R - 1.0e-4]])
    vel = np.zeros((2, 3))
    omega = np.array([[0.0, 500.0, 0.0], [0.0, 0.0, 0.0]])   # saturated cap
    iu = np.array([0])
    ju = np.array([1])

    def torque(lever: str) -> float:
        mat = _mat(rolling_lever=lever, mu=0.0)      # rolling torque only
        state = PairSpringState()
        out = pair_contact_forces(pos, vel, omega, iu, ju, state, mat, DT)
        return float(np.linalg.norm(out["torque"][0]))

    legacy = torque("particle_radius")
    reduced = torque("reduced_radius")
    assert legacy > 0.0 and reduced > 0.0
    assert 1.9 < legacy / reduced < 2.1, (legacy, reduced)


def test_defaults_are_legacy():
    mat = _mat()
    assert mat.rolling_lever == "particle_radius"
    assert mat.rolling_torsion is True
    assert mat.pair_rolling_reff == R
    assert mat.wall_rolling_reff == R
    assert mat.pair_rolling_kr == pytest.approx(mat.kr)
    assert mat.wall_rolling_kr == pytest.approx(mat.kr)
    assert _mat(rolling_lever="reduced_radius").pair_rolling_reff == 0.5 * R
    assert _mat(rolling_lever="reduced_radius").wall_rolling_reff == R


# ---------------------------------------------------------------------------
# D3 — torsion (drilling) spin: the kernel lumps it into the rolling spring
# ---------------------------------------------------------------------------
def test_torsion_switch():
    mat = _mat()
    pos = np.array([[0.0, 0.0, R - 1.0e-4]])
    vel = np.zeros((1, 3))
    omega = np.array([[0.0, 0.0, 20.0]])
    tor_on = wall_contact_forces(pos, vel, omega, _mat(rolling_torsion=True), DT,
                                 mu_wall=0.0, mu_rolling_wall=0.5)
    tor_off = wall_contact_forces(pos, vel, omega, _mat(rolling_torsion=False), DT,
                                  mu_wall=0.0, mu_rolling_wall=0.5)
    assert abs(float(tor_on["torque"][0, 2])) > 0.0
    assert abs(float(tor_off["torque"][0, 2])) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# D4 — OSS references are transcribed, not paraphrased
# ---------------------------------------------------------------------------
def test_cdt_torque_matches_source_definition():
    spin = np.array([[3.0, 4.0, 0.0]])
    out = cdt_torque(spin, fn=2.0, reff=0.5, mu_r=0.3)
    assert np.linalg.norm(out) == pytest.approx(0.3 * 2.0 * 0.5)
    # direction opposes the spin
    assert float(np.dot(out[0], spin[0])) < 0.0
    # zero spin -> zero torque (rolling_model_cdt.h:110/138)
    assert np.allclose(cdt_torque(np.zeros((1, 3)), 2.0, 0.5, 0.3), 0.0)
    # torsion part removed unless requested (rolling_model_cdt.h:148-156); the
    # removal needs the contact normal, exactly as the OSS model has it
    axial = np.array([[0.0, 0.0, 7.0]])
    n = np.array([[0.0, 0.0, 1.0]])
    assert np.allclose(cdt_torque(axial, 2.0, 0.5, 0.3, normal=n), 0.0)
    assert np.linalg.norm(
        cdt_torque(axial, 2.0, 0.5, 0.3, normal=n, torsion_torque=True)) \
        == pytest.approx(0.3)


def test_rayleigh_timestep_matches_source_formula():
    r, rho, nu = 0.03, 2500.0, 0.25
    young = 3.0e7
    shear = young / (2.0 * (1.0 + nu))
    ref = math.pi * r * math.sqrt(rho / shear) / (0.1631 * nu + 0.8766)
    assert rayleigh_timestep(r, rho, shear, nu) == pytest.approx(ref)
    # the Hertz-equivalent modulus of a linear soft spring is much softer than glass
    g_eq = hertz_equivalent_shear_modulus(2.78, 6.0e4, r, nu)
    assert 1.0e7 < g_eq < 1.0e8


# ---------------------------------------------------------------------------
# D5 — the scenes reproduce the analytic saturating-couple law
# ---------------------------------------------------------------------------
def test_spin_down_matches_analytic_deceleration():
    mu_r = 0.5
    out = roll_on_plane(tensordem_kernel(), radius=R, density=2500.0, kn=6.0e4,
                        kt_ratio=0.5, mu_s=0.5, mu_r=mu_r, gravity=9.81,
                        dt=DT, v0=0.30, t_max=0.4, reff=R, sample_every=50)
    theory = 0.30 / ((5.0 / 7.0) * mu_r * 9.81)
    assert out["t_stop"] == pytest.approx(theory, rel=0.02)


def test_static_hold_is_not_exact_in_either_law():
    """Characterisation, not a pass/fail of physics: see the E4 table.

    Below ``tan(theta_c) = mu_r`` neither law holds a sphere *exactly* (Coulomb
    hysteresis on the rolling spring, no history at all in CDT), so both drift
    at a creep speed; above the threshold both roll at a speed set by the
    gravity component.  The measured creep speeds are in
    ``artifacts/xval_dem_round2_method_selection/table_main.md``.
    """
    base = dict(radius=R, density=2500.0, kn=6.0e4, kt_ratio=0.5, mu_s=0.5,
                mu_r=0.5, gravity=9.81, dt=5.0e-5, v0=0.0, t_max=0.2,
                reff=R, sample_every=200)
    below = roll_on_plane(tensordem_kernel(), slope_deg=15.0, **base)
    above = roll_on_plane(tensordem_kernel(), slope_deg=30.0, **base)
    v_below = abs(float(np.mean(below["history"]["v"][-5:])))
    v_above = abs(float(np.mean(above["history"]["v"][-5:])))
    free_roll = 9.81 * math.sin(math.radians(30.0))
    assert v_below < 0.005 * abs(free_roll)     # creep (1e-4 m/s), not rolling
    assert v_above > 0.01 * abs(free_roll)      # slope above arctan(mu_r): rolls
    cdt_below = roll_on_plane(_CDT(), slope_deg=15.0, **base)
    assert abs(float(np.mean(cdt_below["history"]["v"][-5:]))) > 0.0


class _CDT:
    name = "oss_cdt"

    def reset(self, n: int) -> None:
        pass

    def step(self, spin_rel, fn, reff, kt, mu_r, dt, normal=None):
        torque = cdt_torque(spin_rel, fn, reff, mu_r, normal=normal)
        spin = np.atleast_2d(np.asarray(spin_rel, dtype=float))
        return {"torque": torque, "roll": np.zeros_like(spin),
                "slip": spin * float(dt),
                "dissipation": float(-np.einsum("ij,ij->i", torque,
                                                spin * float(dt)).sum()),
                "energy": 0.0}


# ---------------------------------------------------------------------------
# D6 — the case-level switch is validated and off by default
# ---------------------------------------------------------------------------
def test_axisym_case_forwards_and_validates_the_convention():
    case = AxisymCase()
    assert case.rolling_lever == "particle_radius"
    assert case.material().pair_rolling_reff == case.radius
    reduced = AxisymCase(rolling_lever="reduced_radius", rolling_torsion=False)
    mat = reduced.material()
    assert mat.pair_rolling_reff == 0.5 * reduced.radius
    assert mat.rolling_torsion is False
    assert reduced.signature()["rolling_lever"] == "reduced_radius"
    with pytest.raises(AxisymError):
        AxisymCase(rolling_lever="diameter")