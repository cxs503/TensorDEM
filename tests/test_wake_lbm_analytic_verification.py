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
    # |Pi| is proportional to the strain rate, i.e. linear in the amplitude
    assert out["fitted_exponent"] == pytest.approx(1.0, abs=0.02)
    # the subgrid term is active at the solver's operating parameters
    assert out["eddy_to_molecular"][-1] > 1.0


def test_sound_speed_matches_one_over_sqrt_three() -> None:
    """Acoustic branch speed fixes every lattice-to-SI conversion."""
    out = vwl.run_acoustic(steps=250)
    assert out["sound_speed_measured"] == pytest.approx(CS, rel=0.01), out
    assert out["omega_measured"] == pytest.approx(CS * out["wavenumber"], rel=0.01)