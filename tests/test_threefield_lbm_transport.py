"""Verification of the D3Q19 BGK + Smagorinsky core (m2949).

Each test grades one claim against an independent reference; none of them
re-tunes the solver.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.threefield.lbm_transport import (
    double_symmetric_contraction,
    periodic_stream,
    reduced_symmetric_contraction,
    reference_stream,
    sgs_fixed_point_tau,
    shear_wave_decay,
    shear_wave_rate,
    strain_invariants,
    synthetic_strain_field,
    target_eddy_viscosity,
)
from tensordem.threefield.wake_lbm import LBMConfig, WakeLBM

GRADIENTS = {
    "pure_shear": [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "jet_shear": [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "uniaxial": [[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]],
    "plane_strain": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
    "mixed": [[0.3, 0.6, 0.0], [0.0, -0.3, 0.0], [0.0, 0.0, 0.0]],
}


def make(tau0=0.6, cs=0.1, n=12):
    return WakeLBM(
        LBMConfig(nx=n, ny=n, nz=n, tau0=tau0, smagorinsky_cs=cs,
                  disk_radius=1.0, disk_x=8, disk_thickness=1)
    )


def test_streaming_matches_independent_periodic_roll():
    inst = make()
    torch.manual_seed(2949)
    f = torch.rand((inst.nx, inst.ny, inst.nz, 19), dtype=torch.float64)
    assert torch.equal(periodic_stream(inst, f), reference_stream(inst, f))


def test_streaming_is_mass_conserving():
    inst = make()
    torch.manual_seed(1)
    f = torch.rand((inst.nx, inst.ny, inst.nz, 19), dtype=torch.float64)
    total = f.sum()
    streamed = periodic_stream(inst, f).sum()
    # nx*ny*nz*19 = 32832 float64 populations summing to ~1.6e4, so ONE ulp of the
    # total is already 3.6e-12: an absolute 1e-12 bound sits below the resolution of
    # the quantity it measures, and only held while the (thread-dependent) reduction
    # order happened to round exactly.  Assert the conservation law relative to the
    # mass it conserves; 1e-13 ~ 450 ulp is still far tighter than any physical bound.
    assert float((streamed - total).abs()) <= 1e-13 * float(total.abs())


def test_cs_zero_is_exactly_laminar():
    inst = make(cs=0.0)
    rho, u = inst._macroscopic(inst.f)
    tau = inst._effective_tau(inst.f, rho, u)
    assert float((tau - inst.config.tau0).abs().max()) == 0.0


def test_vanishing_strain_leaves_tau0():
    inst = make(cs=0.16)
    f = synthetic_strain_field(inst, [[0.0, 1e-12, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], tau=0.6)
    rho, u = inst._macroscopic(f)
    tau = inst._effective_tau(f, rho, u)
    assert float((tau - inst.config.tau0).abs().max()) < 1e-12


@pytest.mark.parametrize("cs", [0.05, 0.1, 0.16])
@pytest.mark.parametrize("field", sorted(GRADIENTS))
def test_realised_eddy_viscosity_equals_cs2_times_strain(cs, field):
    """nu_t must equal Cs^2 |S| at the closure fixed point, for any strain type."""
    inst = make(cs=cs)
    tau0 = inst.config.tau0
    strain = strain_invariants(GRADIENTS[field])["strain"]
    f = synthetic_strain_field(inst, GRADIENTS[field], tau=sgs_fixed_point_tau(tau0, cs, strain))
    rho, u = inst._macroscopic(f)
    tau_eff = float(inst._effective_tau(f, rho, u)[0, 0, 0])
    nu_t = (tau_eff - tau0) / 3.0
    assert nu_t == pytest.approx(target_eddy_viscosity(cs, strain), rel=1e-12)


def test_reduced_contraction_is_the_documented_negative_control():
    """Summing the six slots unweighted loses a factor sqrt(2) in pure shear.

    This is the pre-m2949 behaviour; the 18 sqrt(2) constant is calibrated for
    the full contraction, so the reduced sum under-predicted nu_t by 29.3 % in a
    shear layer and was exact for normal strain.
    """
    inst = make(cs=0.1)
    for field, expected in (("pure_shear", 1.0 / math.sqrt(2.0)), ("uniaxial", 1.0)):
        f = synthetic_strain_field(inst, GRADIENTS[field], tau=0.6)
        rho, u = inst._macroscopic(f)
        f_neq = f - inst._equilibrium(rho.unsqueeze(-1), u)
        pi = torch.einsum("ik,...i->...k", inst._outer, f_neq)
        full = float(double_symmetric_contraction(pi)[0, 0, 0])
        reduced = float(reduced_symmetric_contraction(pi)[0, 0, 0])
        assert reduced / full == pytest.approx(expected, rel=1e-12)


def test_strain_invariants_decompose_known_fields():
    shear = strain_invariants(GRADIENTS["pure_shear"])
    assert shear["strain"] == pytest.approx(1.0)
    assert shear["divergence"] == pytest.approx(0.0)
    uniaxial = strain_invariants(GRADIENTS["uniaxial"])
    assert uniaxial["strain"] == pytest.approx(math.sqrt(3.0))
    assert uniaxial["shear"] == pytest.approx(0.0)


@pytest.mark.parametrize("tau0", [0.6, 0.8])
def test_shear_wave_decays_at_the_analytic_laminar_rate(tau0):
    """u_x = A cos(k y) is an exact Navier-Stokes solution; it decays as nu k^2."""
    result = shear_wave_decay(tau0=tau0, size=16, mode=1, steps=120)
    assert result.rate_measured == pytest.approx(result.rate_continuum, rel=0.03)
    assert len(result.history) == len(result.times) > 50


def test_shear_wave_rate_matches_nominal_viscosity():
    rate_cont, rate_disc = shear_wave_rate(0.6, 2.0 * math.pi / 16)
    assert rate_cont == pytest.approx((0.6 - 0.5) / 3.0 * (2.0 * math.pi / 16) ** 2)
    assert rate_disc < rate_cont  # discrete Laplacian is smaller than k^2
