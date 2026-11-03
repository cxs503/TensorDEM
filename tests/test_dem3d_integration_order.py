"""Gate for the DEM3D integrator-order verification (docs/dem3d_integration_order_verification.md).

The heavy study lives in ``scripts/verify_dem3d_integration_order.py``; these
tests re-run the two cheapest, most decisive pieces so the published claims
cannot rot silently.
"""
import importlib.util
import math
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_dem3d_integration_order.py"


def _load():
    spec = importlib.util.spec_from_file_location("vio_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vio_script"] = module
    spec.loader.exec_module(module)
    return module


VIO = _load()


def test_ballistic_follows_semi_implicit_closed_form():
    """step() is semi-implicit Euler: x_N = x_0 + v_0 t + a t^2/2 + a t dt/2."""
    torch.set_num_threads(1)
    run = VIO.ballistic(VIO._build(2000.0).config.recommended_dt)
    assert abs(run["rel_err_vs_closed_form"]) < 1e-8
    # ... and therefore carries exactly the O(dt) offset against the continuum
    assert math.isclose(abs(run["rel_err_vs_continuum"]), run["dt"] / run["t"], rel_tol=1e-6)


def test_ballistic_global_order_is_one():
    torch.set_num_threads(1)
    base = VIO._build(2000.0).config.recommended_dt
    runs = [VIO.ballistic(base / 2 ** k, t_end=0.01) for k in range(3)]
    order = VIO._slope([r["dt"] for r in runs], [abs(r["rel_err_vs_continuum"]) for r in runs])
    assert 0.85 < order < 1.15, order


def test_oscillator_energy_error_is_first_order_and_bounded():
    """Symplectic Euler: energy error O(dt) and not secular."""
    torch.set_num_threads(1)
    period = 2.0 * math.pi / math.sqrt(2000.0 / VIO._build(2000.0).config.mass)
    coarse = VIO.oscillator(VIO.Case(dt=period / 40, periods=4))
    fine = VIO.oscillator(VIO.Case(dt=period / 160, periods=4))
    ratio = coarse["energy_error_last_periods"] / fine["energy_error_last_periods"]
    assert 3.0 < ratio < 5.5, ratio  # O(dt): 4x refinement -> ~4x smaller error
    assert fine["energy_error_last_periods"] < 0.05
    assert 0.8 < coarse["energy_drift_ratio"] < 1.3  # bounded, not growing


def test_shipped_recommended_dt_is_accurate_for_the_bond_mode():
    torch.set_num_threads(1)
    sim = VIO._build(2000.0, contact_stiffness=VIO.SHIPPED_CONTACT_STIFFNESS)
    period = 2.0 * math.pi / math.sqrt(2000.0 / sim.config.mass)
    spp = max(2, int(round(period / sim.config.recommended_dt)))
    run = VIO.oscillator(VIO.Case(dt=period / spp, periods=4))
    assert run["energy_error_last_periods"] < 0.03, run["energy_error_last_periods"]
    assert run["max_position_error"] < 0.03, run["max_position_error"]
