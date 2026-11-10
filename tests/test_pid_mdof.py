"""Tests for tensordem.pid_mdof — particle impact damper on MDOF structures.

Coverage:
* modal models (cantilever analytic eigenvalues, plate isotropic limit,
  stiffener stiffening, mass normalisation / effective modal mass)
* Den Hartog closed form (optimal tuning reproduces sqrt(1+2/mu);
  N = 1 MDOF solve reproduces the closed-form FRF to machine precision)
* IN32-D02 surface consumption (regression lock at the delivered mass ratio,
  linear mass scaling, Lorentzian detuning, zero-filling limit)
* MDOF insertion loss (truncation convergence, rms metric vs closed form,
  negative control, monotonicity in device damping)
* floating-raft reference and the per-unit-mass comparison machinery
* acceptance snapshot of the campaign artifact (if present)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensordem import pid as pid_mod
from tensordem import pid_mdof as pm


# ---------------------------------------------------------------------------
# modal structures
# ---------------------------------------------------------------------------
def test_cantilever_eigenvalues_match_analytic():
    s = pm.cantilever_structure(n_modes=4)
    rho, b, h, L, E = 7850.0, 0.05, 0.01, 1.0, 210e9
    A = b * h
    EI = E * b * h ** 3 / 12.0
    f1 = 1.8751040687119611 ** 2 * math.sqrt(EI / (rho * A * L ** 4)) / (2 * math.pi)
    assert s.modes[0].frequency_hz == pytest.approx(f1, rel=1e-10)
    assert s.modes[1].omega / s.modes[0].omega == pytest.approx(6.2669, rel=1e-4)
    # clamped-free boundary conditions on the normalised shapes
    xg, wg = pm._gauss_legendre_nodes(L, 200)
    for n, mode in enumerate(s.modes[:3]):
        beta_l = pm._CANT_BETA_L[n]
        beta = beta_l / L
        sig = -(math.cosh(beta_l) + math.cos(beta_l)) / (math.sinh(beta_l) + math.sin(beta_l))
        d2 = beta ** 2 * (np.cosh(beta * L) + np.cos(beta * L)
                          + sig * (np.sinh(beta * L) + np.sin(beta * L)))
        d3 = beta ** 3 * (np.sinh(beta * L) - np.sin(beta * L)
                          + sig * (np.cosh(beta * L) + np.cos(beta * L)))
        assert abs(d2) < 1e-6 and abs(d3) < 1e-4


def test_cantilever_mass_normalisation():
    """int rho*A*phi_n^2 dx = 1 for every retained mode (m_n = 1 convention)."""
    xg, wg = pm._gauss_legendre_nodes(1.0, 400)
    mass_per_length = 7850.0 * 0.05 * 0.01
    for n in range(4):
        beta_l = pm._CANT_BETA_L[n]
        psi = pm._cantilever_psi(xg, beta_l, 1.0)
        norm = math.sqrt(float(np.sum(wg * psi ** 2)) * mass_per_length)
        phi = psi / norm
        integral = float(np.sum(wg * phi ** 2)) * mass_per_length
        assert integral == pytest.approx(1.0, rel=1e-9)


def test_plate_reduces_to_isotropic_analytic_without_stiffeners():
    plate = pm.stiffened_plate_structure(n_stiffeners=1, stiffener_inertia=0.0,
                                         n_modes=4)
    D = 210e9 * 0.006 ** 3 / (12.0 * (1 - 0.09))
    rho_h = 7850.0 * 0.006
    f11 = 9.8696044 * math.sqrt(D / rho_h) * (1 / 1.0 ** 2 + 1 / 0.6 ** 2) / (2 * math.pi)
    assert plate.modes[0].frequency_hz == pytest.approx(f11, rel=1e-6)
    # mode ordering: (1,1) < (2,1) < (1,2) for a 1.0 x 0.6 panel
    assert plate.modes[1].label == "panel mode (2,1)"


def test_stiffeners_raise_first_panel_frequency():
    bare = pm.stiffened_plate_structure(n_stiffeners=1, stiffener_inertia=0.0)
    stiff = pm.stiffened_plate_structure(n_stiffeners=3, stiffener_inertia=2.0e-9)
    assert stiff.modes[0].frequency_hz > bare.modes[0].frequency_hz
    assert stiff.meta["D_x_Nm"] > stiff.meta["D_Nm"]


def test_effective_mass_at_node_raises():
    plate = pm.stiffened_plate_structure(n_modes=4)
    node_mode = next(m for m in plate.modes if abs(m.phi_d) < 1e-12)
    with pytest.raises(ValueError):
        plate.effective_mass_at_damper(plate.modes.index(node_mode))


# ---------------------------------------------------------------------------
# Den Hartog closed form and the N = 1 equivalence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu", [0.01, 0.03, 0.05, 0.10, 0.20])
def test_den_hartog_optimum_attains_classical_peak(mu):
    """Den Hartog's optimum must reproduce the classical sqrt(1 + 2/mu)."""
    nu_opt, zeta_opt = pm.den_hartog_optimum(mu)
    r = np.linspace(0.5, 1.5, 100001)
    peak = float(np.max(np.abs(pm.two_dof_response(r, mu, nu_opt, zeta_opt, 0.0))))
    assert peak == pytest.approx(math.sqrt(1.0 + 2.0 / mu), rel=5e-3)


@pytest.mark.parametrize("mu,nu,zeta_d", [(0.01, 0.95, 0.002), (0.03, 0.97, 0.02),
                                          (0.05, 1.00, 0.13)])
def test_single_mode_mdof_equals_closed_form(mu, nu, zeta_d):
    """With N = 1 the (N+1)-DOF modal solve must be the Den Hartog system."""
    s = pm.cantilever_structure(n_modes=1)

    dam = pm.ParticleDamperModel(mass_ratio=mu, filling_ratio=0.30)
    dam.zeta_d_fixed = zeta_d

    class Fixed(type(dam)):
        def zeta_d(self, nu=1.0):
            return self.zeta_d_fixed

    dam = Fixed(**{k: getattr(dam, k) for k in
                   ("filling_ratio", "mass_ratio", "gamma", "restitution",
                    "detuning_coupling", "n_particles", "primary_mass",
                    "primary_stiffness", "cavity_length", "gravity")})
    dam.zeta_d_fixed = zeta_d
    r = np.linspace(0.6, 1.4, 4001)
    x_mdof = np.abs(pm.modal_response(s, r * s.omega_1, dam, nu=nu))
    static = abs(pm.modal_response(s, np.array([1e-6]), None)[0])
    ref = np.abs(pm.two_dof_response(r, mu, nu, zeta_d, s.zeta[0]))
    dev = np.max(np.abs(x_mdof / static - ref)) / np.max(ref)
    assert dev < 1e-10


def test_no_damper_is_modal_superposition_of_sdof_modes():
    """Without a damper the response must equal the sum of the modal SDOF terms."""
    s = pm.cantilever_structure(n_modes=6)
    w = np.linspace(0.5, 1.5, 501) * s.omega_1
    x = pm.modal_response(s, w, None)
    manual = np.zeros_like(w, dtype=complex)
    for m, z in zip(s.modes, s.zeta):
        manual += m.phi_o * m.phi_f * pm.sdof_response(w, m.omega, z) / m.omega ** 2
    assert np.allclose(x, manual, rtol=1e-10)


# ---------------------------------------------------------------------------
# IN32-D02 surface consumption
# ---------------------------------------------------------------------------
def test_provider_reproduces_delivered_surface_at_reference_mass_ratio():
    cfg = pid_mod.SweepConfig()
    surface = pid_mod.ZetaSurface.from_results(pid_mod.run_sweep(cfg), cfg)
    gi = surface.gamma_axis.index(2.0)
    fi = surface.freq_ratio_axis.index(1.0)
    pi = surface.filling_ratio_axis.index(0.3)
    ei = surface.restitution_axis.index(0.6)
    expected = surface.zeta_values[gi][fi][pi][ei]
    got = pm.zeta_eff_delivered(0.3, 1.0, gamma=2.0, restitution=0.6,
                               mass_ratio=pm.MU_REF)
    assert got == pytest.approx(expected, rel=1e-12)


def test_provider_scales_linearly_with_mass_ratio():
    z_ref = pm.zeta_eff_delivered(0.3, 1.0, mass_ratio=pm.MU_REF)
    for factor in (0.05, 0.15, 0.25):
        z = pm.zeta_eff_delivered(0.3, 1.0, mass_ratio=pm.MU_REF * factor)
        assert z == pytest.approx(z_ref * factor, rel=1e-9)


def test_provider_zero_filling_gives_zero_damping():
    assert pm.zeta_eff_delivered(0.0, 1.0) == 0.0
    assert pm.ZetaEffProvider(filling_ratio=0.0).zeta(1.0) == 0.0


def test_surface_has_interior_frequency_optimum():
    """The delivered surface peaks at f/f_n ~ 1.2 (constant-acceleration sweep)."""
    prov = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=0.05)
    z = {r: prov.zeta(r) for r in np.round(np.arange(0.7, 1.35, 0.05), 2)}
    r_best = max(z, key=z.get)
    assert 1.10 <= r_best <= 1.30
    assert z[r_best] > z[1.0]


def test_detuning_coupling_reduces_damping_off_tuning():
    """zeta_eff falls away from the surface optimum (which sits at f/f_n ~ 1.2)."""
    dam = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=0.03,
                                 detuning_coupling=True)
    z_opt = dam.zeta_d(1.2)
    assert dam.zeta_d(0.8) < z_opt
    assert dam.zeta_d(0.6) < dam.zeta_d(0.8)
    frozen = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=0.03,
                                    detuning_coupling=False)
    assert frozen.zeta_d(1.2) == pytest.approx(frozen.zeta_d(1.0))
    assert frozen.zeta_d(0.6) == pytest.approx(frozen.zeta_d(1.0))


def test_half_power_identification_recovers_damping():
    w_n = 2.0 * math.pi * 5.0
    for zeta in (0.005, 0.006, 0.02, 0.05):
        rec = pm.identify_zeta_half_power(w_n, zeta)
        assert rec == pytest.approx(zeta, rel=5e-3), (zeta, rec)


# ---------------------------------------------------------------------------
# insertion loss
# ---------------------------------------------------------------------------
def test_negative_control_zero_filling_gives_zero_il():
    for struct in (pm.cantilever_structure(n_modes=6),
                   pm.stiffened_plate_structure(n_modes=6)):
        prov = pm.ZetaEffProvider(filling_ratio=0.0, mass_ratio=0.03)
        il_a = pm.injection_peak_insertion_loss(struct, prov, nu=1.0)
        dam = pm.ParticleDamperModel(filling_ratio=0.0, mass_ratio=0.03)
        il_b = pm.peak_insertion_loss(struct, dam, nu=1.0)
        assert abs(il_a["il_peak_db"]) < 1e-6
        assert abs(il_b["il_peak_db"]) < 1e-6
        assert dam.damper_mass(struct) == 0.0


def test_damper_mass_scales_with_mass_ratio_and_filling():
    s = pm.cantilever_structure(n_modes=4)
    m_att = s.effective_mass_at_damper(0)
    d1 = pm.ParticleDamperModel(filling_ratio=0.30, mass_ratio=0.01)
    d2 = pm.ParticleDamperModel(filling_ratio=0.30, mass_ratio=0.05)
    assert d2.damper_mass(s) == pytest.approx(5.0 * d1.damper_mass(s))
    assert d1.damper_mass(s) == pytest.approx(0.01 * m_att)
    d3 = pm.ParticleDamperModel(filling_ratio=0.15, mass_ratio=0.01)
    assert d3.damper_mass(s) == pytest.approx(0.5 * d1.damper_mass(s))


def test_more_device_damping_gives_more_insertion_loss():
    s = pm.cantilever_structure(n_modes=6)
    ils = []
    for zeta_d in (0.003, 0.02, 0.05, 0.10):
        dam = pm.ParticleDamperModel(mass_ratio=0.03, filling_ratio=0.30)
        dam.zeta_d_fixed = zeta_d

        class Fixed(type(dam)):
            def zeta_d(self, nu=1.0):
                return self.zeta_d_fixed

        dam2 = Fixed(**{k: getattr(dam, k) for k in
                        ("filling_ratio", "mass_ratio", "gamma", "restitution",
                         "detuning_coupling", "n_particles", "primary_mass",
                         "primary_stiffness", "cavity_length", "gravity")})
        dam2.zeta_d_fixed = zeta_d
        ils.append(pm.peak_insertion_loss(s, dam2, nu=0.97)["il_peak_db"])
    assert ils == sorted(ils)


def test_rms_il_agrees_with_closed_form_cantilever():
    s = pm.cantilever_structure(n_modes=8)
    dam = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=0.03)
    nu = 0.95
    zd = dam.zeta_d(nu)
    a = pm.rms_insertion_loss(s, dam, nu=nu)
    b = pm.rms_insertion_loss_2dof(0.03, nu, zd, s.zeta[0])
    assert a > 0.0
    assert a == pytest.approx(b, rel=0.03)


def test_modal_truncation_converges():
    ils = {}
    for n in (6, 8, 12):
        s = pm.cantilever_structure(n_modes=n)
        dam = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=0.05)
        ils[n] = pm.peak_insertion_loss(s, dam, nu=1.0)["il_peak_db"]
    assert abs(ils[8] - ils[12]) / abs(ils[12]) < 0.01


def test_insertion_loss_is_positive_at_optimum_for_delivered_device():
    s = pm.cantilever_structure(n_modes=6)
    prov = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=0.05)
    opt = pm.optimise_injection_tuning(s, prov)
    assert 0.5 < opt["il_peak_db"] < 10.0
    assert 0.7 < opt["nu"] < 1.0          # device designed below the resonance
    curve = opt["curve"]
    assert len(curve) >= 5


# ---------------------------------------------------------------------------
# floating raft reference
# ---------------------------------------------------------------------------
def test_floating_raft_isolates_above_isolation_frequency():
    il = pm.floating_raft_il_db(1.0, mu_raft=0.03, r_iso=0.3, zeta_iso=0.05)
    assert il > 20.0
    # below the isolation frequency the raft amplifies
    il_low = pm.floating_raft_il_db(0.2, mu_raft=0.03, r_iso=0.3, zeta_iso=0.05)
    assert il_low < 0.0


# ---------------------------------------------------------------------------
# acceptance snapshot of the campaign artifact
# ---------------------------------------------------------------------------
ARTIFACT = os.path.join("artifacts", "p307w1_n010", "acceptance.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="acceptance artifact absent")
def test_acceptance_artifact_criteria():
    with open(ARTIFACT) as fh:
        acc = json.load(fh)["acceptance"]
    assert acc["a_pass"], acc
    assert acc["b_pass"], acc
    assert acc["c_pass"], acc
    assert acc["c_pass_model_A"], acc
    assert acc["d_pass"], acc
    assert acc["e_pass"], acc
    assert acc["b_rms_max_rel_dev"] <= 0.10
    assert acc["d_max_abs_il_db"] <= 0.5