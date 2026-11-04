"""Analytic verification of the D3Q19 propeller-race LBM core (wake_lbm).

Binds the shipped lattice kernels (``_equilibrium``, ``_macroscopic``,
``_effective_tau``, ``_stream`` and the free-slip reflection tables) to
closed-form references: equilibrium moments, lattice isotropy, the shear-wave
viscosity ``nu = (tau0 - 1/2)/3``, the ``k^2`` diffusive scaling, the
Smagorinsky closure of Hou et al. (1996) and the lattice sound speed
``cs = 1/sqrt(3)``.

Before this file the solver's evidence was conservation-only; nothing checked
the viscosity or the sound speed, on which every lattice-to-SI conversion in the
three-field coupling depends.

Verification report: ``docs/wake_lbm_analytic_verification.md``.
Producing script: ``scripts/verify_wake_lbm_analytic.py``.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_wake_lbm_analytic.py"
_spec = importlib.util.spec_from_file_location("verify_wake_lbm_analytic", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vwl = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("verify_wake_lbm_analytic", vwl)
_spec.loader.exec_module(vwl)

CS = math.sqrt(1.0 / 3.0)


def test_equilibrium_reproduces_mass_momentum_and_stress() -> None:
    """eq must return rho, rho*u and rho*(cs^2 I + u u) to round-off."""
    out = vwl.equilibrium_moment_check()
    assert out["mass_error"] < 1e-13
    assert out["momentum_error"] < 1e-13
    assert out["moment_flux_error"] < 1e-13


def test_d3q19_weights_are_isotropic_to_fourth_order() -> None:
    """A wrong weight/velocity set would break the viscosity derivation."""
    out = vwl.isotropy_check()
    assert out["zeroth_error"] < 1e-14
    assert out["second_error"] < 1e-14
    assert out["fourth_error"] < 1e-13


@pytest.mark.parametrize("tau0", [0.8, 1.2])
def test_shear_wave_decay_gives_the_closed_form_viscosity(tau0: float) -> None:
    """Free-slip shear mode decays as exp(-nu K^2 t) with nu = (tau0-1/2)/3."""
    k = math.pi / 15.0
    steps = vwl._steps_for((tau0 - 0.5) / 3.0, k, spans=2.0)
    out = vwl.run_shear(vwl.ShearRun(tau0, 0.0, 0.01, 1, max(80, steps // 2)))
    assert abs(out["relative_error"]) < 0.02, out
    assert out["decay_fit_r2"] > 0.999, out
    assert out["mass_drift"] < 1e-12
    assert out["x_momentum_drift"] < 1e-10
    assert abs(out["mean_effective_tau"] - tau0) < 1e-9


def test_second_mode_confirms_diffusive_scaling() -> None:
    """Doubling the wavenumber must quadruple the decay rate, not change nu."""
    k2 = 2.0 * math.pi / 15.0
    out = vwl.run_shear(vwl.ShearRun(1.0, 0.0, 0.01, 2, 120))
    assert out["wavenumber"] == pytest.approx(k2, rel=1e-12)
    assert abs(out["relative_error"]) < 0.02, out


def test_smagorinsky_closure_matches_independent_evaluation() -> None:
    """Closure algebra: zero-stress limit exact, closed form reproduced."""
    out = vwl.smagorinsky_closure_check()
    assert out["zero_stress_tau_error"] < 1e-15
    assert out["max_relative_error"] < 1e-12, out
    # convention drift guard: the pre-m2949 six-slot reference under-estimates
    # |Pi| by sqrt(2) in pure shear, which used to make this comparison read as
    # "1e-3 agreement" instead of "the reference is stale"
    assert out["legacy_pi_ratio"] == pytest.approx(math.sqrt(2.0), rel=2e-3), out
    assert out["legacy_max_relative_error"] > 1e-3, out
    # |Pi| is proportional to the strain rate, i.e. linear in the amplitude
    assert out["fitted_exponent"] == pytest.approx(1.0, abs=0.02)
    # the subgrid term is active at the solver's operating parameters
    assert out["eddy_to_molecular"][-1] > 1.0


def test_sound_speed_matches_one_over_sqrt_three() -> None:
    """Acoustic branch speed fixes every lattice-to-SI conversion."""
    out = vwl.run_acoustic(steps=250)
    assert out["sound_speed_measured"] == pytest.approx(CS, rel=0.01), out
    assert out["omega_measured"] == pytest.approx(CS * out["wavenumber"], rel=0.01)

# ---------------------------------------------------------------------------
# C5b: analytic (Chapman-Enskog) anchor for the (Pi, |S|) pair
# ---------------------------------------------------------------------------
def test_smagorinsky_strain_anchor_matches_chapman_enskog() -> None:
    """|Pi| on the analytic shear mode equals (2 rho/3) sqrt(S_ab S_ab)."""
    out = vwl.smagorinsky_strain_rate_anchor()
    # at the shipped operating wavenumber the CE closed form closes to <1%
    assert out["shipped_wavenumber_pi_relative_error"] < 0.01, out["rows"][0]
    for row in out["rows"]:
        assert row["relative_deviation"] < 0.0, row        # smoothing, not noise
        # leading-order model c_s^2K^2/2; the residual at m=3, A=0.1 reaches
        # 3.3% (higher-order K^4 / u^2 terms), so 5% is the honest band
        assert abs(row["deviation_over_smoothing"] - 1.0) < 0.05, row


def test_smagorinsky_residual_is_o_k_squared_not_an_index_slip() -> None:
    """The residual must scale as K^2 (streaming shift smoothing), not be flat."""
    out = vwl.smagorinsky_strain_rate_anchor()
    assert 3.5 < out["k2_growth_ratio"] < 4.5, out["k2_growth_ratio"]
    assert 7.5 < out["k3_growth_ratio"] < 9.5, out["k3_growth_ratio"]
    # ... and it is a finite-K artefact, not an O(u^2) term: amplitude independent
    assert out["amplitude_ratio"] == pytest.approx(1.0, abs=0.05)


def test_smagorinsky_strain_anchor_negative_controls() -> None:
    """Cs=0 leaves tau_eff = tau0 exactly; zero strain gives zero eddy viscosity."""
    out = vwl.smagorinsky_strain_rate_anchor()
    assert out["negative_control_zero_cs"] == 0.0
    assert out["negative_control_zero_amplitude"] == 0.0
    # documented convention boundary: the one-step transient flux carries no
    # relaxation-time factor, so implied nu_t = Cs^2|S|/tau_eff, not Cs^2|S|
    assert out["implied_nu_ratio"] == pytest.approx(out["implied_nu_ratio_analytic"], rel=0.01)
    assert out["implied_nu_ratio"] > 1.9
