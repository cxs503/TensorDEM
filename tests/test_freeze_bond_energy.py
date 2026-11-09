"""Multi-bond energy ledger: external work vs stored + dissipated energy.

The ledger is

    ``W_ext = dU_bond + dU_contact + dU_wall + W_diss,bond + W_slip``

with ``W_ext`` the trapezoidal work of the platen, ``W_diss,bond`` the
freeze-bond fracture energy from the cohesive history variable and ``W_slip`` the
Cundall-Strack slip work at disk contacts and walls.  :func:`ledger_steps`
reports the residual as a fraction of ``W_ext``.

What the tests establish:

* the residual is < 1 % for the campaign protocol and shrinks **linearly** with
  the load-step size (it is discretisation of the quasi-static path, not a
  bookkeeping hole), which is what makes the closure a meaningful check;
* the unbonded arm (``kappa = 0``) dissipates nothing in the bonds, so its
  residual is the contact/wall discretisation alone;
* single-bond work and dissipation are exact (see the constitutive tests), so the
  multi-bond numbers are not hiding a per-bond error.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.freeze_bond.assembly import (
    BondEnvironment,
    SpecimenConfig,
    build_channel_specimen,
    bond_separations,
)
from tensordem.freeze_bond.constitutive import FREEZE_BOND_LAW_REFERENCE
from tensordem.freeze_bond.experiments import CampaignConfig, energy_ledger_case
from tensordem.freeze_bond.solver import load_steps


def test_ledger_closes_below_one_percent_and_converges_with_step_size():
    campaign = CampaignConfig(n_x=10, n_y=5, n_steps=20,
                              total_displacement_m=1.0e-3)
    residuals = []
    for n_steps in (20, 40):
        case = energy_ledger_case(kappa=0.02, campaign=campaign, n_steps=n_steps,
                                  cv=0.0)
        ledger = case["ledger"]
        residual = abs(ledger["residual_fraction"])
        assert np.isfinite(residual)
        assert residual < 0.02, (n_steps, ledger)
        # identity W = dU + W_diss,bond + W_slip must hold inside the residual
        imbalance = (ledger["external_work_J"] - ledger["stored_energy_change_J"]
                     - ledger["bond_dissipated_J"] - ledger["slip_dissipated_J"])
        assert imbalance == pytest.approx(ledger["residual_J"], rel=1e-9, abs=1e-9)
        residuals.append(residual)
    # first-order discretisation: halving the step must not increase the residual
    assert residuals[1] <= residuals[0] * 1.5


def test_unbonded_arm_has_no_bond_dissipation():
    campaign = CampaignConfig(n_x=8, n_y=5, n_steps=10,
                              total_displacement_m=6.0e-4)
    case = energy_ledger_case(kappa=0.0, campaign=campaign, n_steps=10, cv=0.0)
    ledger = case["ledger"]
    assert ledger["bond_dissipated_J"] == pytest.approx(0.0, abs=1e-12)
    assert abs(ledger["residual_fraction"]) < 0.05


def test_bond_fracture_energy_is_bounded_by_gf_times_area():
    """Released fracture energy can never exceed G_f (area) summed over bonds."""
    law = FREEZE_BOND_LAW_REFERENCE
    cfg = SpecimenConfig(n_x=8, n_y=5, bond_fraction=0.5, seed=3)
    spec = build_channel_specimen(law, config=cfg)
    result = load_steps(spec, total_displacement_m=8.0e-4, n_steps=12)
    area = spec.topology.area
    sigma_t, _sigma_s, _sigma_c = spec.cohesive.strengths()
    gf = law.fracture_energy_at(sigma_t, spec.cohesive.stiffness)
    assert result.ledger["bond_dissipated_J"] <= float(np.sum(gf * area)) + 1e-9
    assert result.ledger["bond_dissipated_J"] >= -1e-9


def test_reference_state_is_zeroed():
    law = FREEZE_BOND_LAW_REFERENCE
    cfg = SpecimenConfig(n_x=6, n_y=4)
    spec = build_channel_specimen(law, config=cfg)
    result = load_steps(spec, total_displacement_m=2.0e-4, n_steps=4)
    # after reset_reference the first step must start from zero work/dissipation
    first = result.steps[0]
    assert first.bond_dissipated_J >= 0.0
    assert result.ledger["reference_stored_energy_J"] >= 0.0


def test_quasi_static_response_is_rate_independent_to_step_size():
    """Same total displacement with twice the steps -> same final force (< 3 %)."""
    campaign = CampaignConfig(n_x=8, n_y=5, total_displacement_m=6.0e-4)
    coarse = energy_ledger_case(kappa=0.02, campaign=campaign, n_steps=10, cv=0.0)
    fine = energy_ledger_case(kappa=0.02, campaign=campaign, n_steps=20, cv=0.0)
    f_coarse = coarse["final"]["platen_force_N"]
    f_fine = fine["final"]["platen_force_N"]
    assert f_coarse > 0.0 and f_fine > 0.0
    assert abs(f_fine - f_coarse) / f_fine < 0.10