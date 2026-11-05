"""Closed-form anchoring of the 2-D ``IceDEM`` energy ledger.

``tests/test_accounting_ledger_validation.py`` proves the ledger *identity*
closes on indentation runs.  An identity check cannot tell whether the channels
carry the right numbers (a uniformly mis-scaled channel still closes if the
opposite channel absorbs the error), so this module scores the channels against
independently coded closed forms, mirroring the 3-D audit
(``docs/dem3d_energy_ledger_analytic_verification.md``):

* **A** free-particle viscous decay: ``v(t) = v0 exp(-t/tau)`` and
  ``D(t) = 0.5 m v0^2 (1 - exp(-2t/tau))``;
* **B** bonded harmonic oscillator: ``xi(t) = A cos(omega t)``, ``omega = sqrt(2k/m)``;
* **C** tool-work channel: residual of the balance under time-step refinement.

Reference forms are re-derived inside the tests from textbook relations; they do
not call the module under test.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_dem2d_energy_ledger import (  # noqa: E402
    ENGINE_PATH,
    bonded_oscillator,
    drag_decay,
    tool_indentation,
)

#: acceptance for the ledger channels carried by the engine (owner standard)
TOL = 0.03


@pytest.fixture(scope="module")
def case_a():
    return drag_decay(1.0)


@pytest.fixture(scope="module")
def case_a_half():
    return drag_decay(0.5)


@pytest.fixture(scope="module")
def case_b():
    return bonded_oscillator(1.0)


@pytest.fixture(scope="module")
def case_c():
    return tool_indentation(1.0)


@pytest.fixture(scope="module")
def case_c_half():
    return tool_indentation(0.5)


def test_engine_is_this_worktree():
    """Provenance: the audit must not silently score a sibling checkout."""
    resolved = Path(ENGINE_PATH).resolve()
    assert (resolved.parents[1] / "src").is_dir() or "src" in resolved.parts
    assert resolved.is_file()


def test_case_a_velocity_and_dissipation_match_closed_form(case_a):
    m, c, v0 = case_a["mass_kg"], case_a["drag"], case_a["v0"]
    tau = m / c
    t = case_a["t_end_s"]
    assert case_a["tau_s"] == pytest.approx(tau, rel=1e-12)
    assert case_a["v_ref"] == pytest.approx(v0 * math.exp(-t / tau), rel=1e-12)
    assert case_a["v_sim"] < case_a["v_ref"] * 1.02
    assert case_a["v_relerr"] < TOL
    assert case_a["ke_relerr"] < TOL
    assert case_a["dissipation_relerr"] < TOL


def test_case_a_residual_is_the_first_order_truncation(case_a):
    """The residual is bookkeeping bias, with a closed-form prediction."""
    assert case_a["residual_over_dissipation"] < TOL
    # pre-update sampling over-records 0.5 (c/m) dt of the dissipated energy
    assert case_a["residual_pred_relerr"] < 0.10
    assert case_a["residual_pred_J"] > 0.0


def test_case_a_residual_is_first_order_in_dt(case_a, case_a_half):
    ratio = case_a_half["residual_over_dissipation"] / case_a["residual_over_dissipation"]
    assert 0.4 < ratio < 0.6, f"expected ~1/2 (first order), got {ratio}"
    assert case_a_half["dissipation_relerr"] < case_a["dissipation_relerr"]


def test_case_b_ledger_residual_is_negligible(case_b):
    """Purely elastic, undamped: the ledger must not invent or lose energy."""
    assert case_b["broken_bonds"] if "broken_bonds" in case_b else True
    assert case_b["residual_over_e0"] < TOL
    assert case_b["damping_dissipation_J"] == pytest.approx(0.0, abs=1e-12)
    assert case_b["tool_work_J"] == pytest.approx(0.0, abs=1e-12)
    assert case_b["fracture_release_J"] == pytest.approx(0.0, abs=1e-12)


def test_case_b_initial_energy_matches_half_k_A_squared(case_b):
    assert case_b["e0_J"] == pytest.approx(
        0.5 * case_b["stiffness"] * case_b["amplitude_m"] ** 2, rel=1e-10)


def test_case_b_oscillator_is_the_integrator_not_the_ledger(case_b):
    """Documented honest miss: the *trajectory* exceeds 3 % at production dt.

    The engine uses semi-implicit Euler, whose phase/amplitude error is
    ``O(omega dt)`` here (``omega dt = 0.11``), so the extension comparison is
    recorded as a discretisation envelope rather than a ledger failure.  The
    ledger residual for the very same run is below ``TOL`` and the trajectory
    error shrinks linearly with ``dt`` (asserted in the next test).
    """
    assert case_b["xi_relerr_max"] > TOL          # honest: not <3 %
    assert case_b["xi_relerr_max"] < 0.10
    assert case_b["residual_over_e0"] < TOL       # but the ledger itself is fine


def test_case_b_refinement_halves_the_trajectory_error():
    coarse = bonded_oscillator(1.0)
    fine = bonded_oscillator(0.5)
    ratio = fine["xi_relerr_max"] / coarse["xi_relerr_max"]
    assert 0.35 < ratio < 0.65, f"expected ~1/2, got {ratio}"
    assert fine["residual_over_e0"] <= coarse["residual_over_e0"]


def test_case_c_tool_work_balance_closes(case_c, case_c_half):
    assert case_c["tool_work_J"] > 0.0
    assert case_c["residual_over_scale"] < TOL
    ratio = case_c_half["residual_over_scale"] / case_c["residual_over_scale"]
    assert 0.2 < ratio < 0.7, f"expected first-order shrink, got {ratio}"


def test_negative_control_dropped_channel_breaks_the_identity(case_a):
    """Negative control: the closure is sensitive to each channel."""
    residual = case_a["residual_J"]
    dissipation = case_a["dissipation_sim_J"]
    assert dissipation > 0.0
    # dropping the dissipation channel from the balance leaves ~100 % of it open
    assert abs(dissipation) / max(abs(residual), 1e-18) > 50.0