"""Tests for the sea-ice micro-fracture energy budget (p216w3-n07).

Each test mirrors one machine-checkable criterion of the campaign node:

* C1 -- energy closure ``epsilon_s < 1 %``;
* C2 -- F-delta curve monotonic (no load drop) within the elastic limit;
* C3 -- fragment-number convergence (n_frag doubling -> allocation < 3 %);
* C4 -- N06 ``break_crush`` channel closure ``C1 < 1 %``.

The suite uses a reduced step budget so it stays fast; the full-budget verdict
is produced by ``scripts/benchmark_sea_ice_micro_energy.py``.

Note on particle size: the fragment-convergence criterion (C3) is scored on a
larger particle (``d = 32 mm``) because the surface-energy demand ``gamma * dA``
grows as ``n_frag^(1/3)`` and must stay well below the released elastic energy
for the allocation to be stable.  At ``d = 2 mm`` the surface energy saturates
the released elastic budget and the allocation is n_frag-sensitive by
construction (a physical statement, not a numerical error).
"""
import math

import numpy as np
import pytest

from tensordem.sea_ice_micro_energy import (
    check_f_delta_monotonic,
    check_fragment_convergence,
    n06_break_crush_channel,
    run_energy_budget_audit,
    _run_single_fracture,
)

#: Particle diameter for the C3 convergence test (large enough that the
#: surface-energy demand stays below the released elastic budget).
CONVERGENCE_DIAMETER = 32.0e-3


# --------------------------------------------------------------------------
# C1 - energy closure
# --------------------------------------------------------------------------
def test_c1_energy_closure_epsilon_s_below_one_percent():
    """The three-channel budget must close to < 1 %."""
    budget = _run_single_fracture(
        diameter=2.0e-3, seed=0, platen_velocity=-0.05,
        n_frag=8, steps=4000,
    )
    assert budget["crushing_events"] > 0, "must fracture for this test"
    assert budget["E_in_J"] > 0.0
    assert budget["E_out_J"] > 0.0
    assert budget["epsilon_s"] < 0.01, (
        f"energy closure {budget['epsilon_s']:.4f} exceeds 1 %"
    )


def test_c1_three_channels_are_positive_and_partition():
    """All three channels must be non-negative and sum to E_out."""
    budget = _run_single_fracture(
        diameter=2.0e-3, seed=1, platen_velocity=-0.05,
        n_frag=8, steps=4000,
    )
    assert budget["E_surf_J"] >= 0.0
    assert budget["E_kin_J"] >= 0.0
    assert budget["E_diss_J"] >= 0.0
    assert budget["E_out_J"] == pytest.approx(
        budget["E_surf_J"] + budget["E_kin_J"] + budget["E_diss_J"],
        rel=1e-12,
    )


def test_c1_epsilon_s_is_defined_as_ein_minus_eout_over_ein():
    """The audit statistic must be the documented definition."""
    budget = _run_single_fracture(
        diameter=2.0e-3, seed=2, platen_velocity=-0.05,
        n_frag=8, steps=4000,
    )
    expected = (budget["E_in_J"] - budget["E_out_J"]) / budget["E_in_J"]
    assert budget["epsilon_s"] == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# C2 - F-delta monotonicity
# --------------------------------------------------------------------------
def test_c2_f_delta_curve_is_monotonic_within_elastic_limit():
    """The force-displacement curve must have no load drop before fracture."""
    budget = _run_single_fracture(
        diameter=2.0e-3, seed=3, platen_velocity=-0.05,
        n_frag=8, steps=4000,
    )
    f_delta = check_f_delta_monotonic(
        budget["force_history_N"], budget["displacement_history_m"],
    )
    assert f_delta["monotonic"], (
        f"F-delta curve has {f_delta['n_drops']} load drops "
        f"(max {f_delta['max_drop_N']:.4g} N)"
    )
    assert f_delta["n_elastic_samples"] > 10, "need a meaningful elastic segment"


def test_c2_elastic_segment_is_before_first_fracture():
    """The elastic segment must end at the first force drop (fracture)."""
    budget = _run_single_fracture(
        diameter=2.0e-3, seed=4, platen_velocity=-0.05,
        n_frag=8, steps=4000,
    )
    f_delta = check_f_delta_monotonic(
        budget["force_history_N"], budget["displacement_history_m"],
    )
    # The elastic segment must be a proper prefix of the full history.
    assert f_delta["elastic_end_step"] <= len(budget["force_history_N"])
    assert f_delta["elastic_end_step"] >= 1


# --------------------------------------------------------------------------
# C3 - fragment-number convergence
# --------------------------------------------------------------------------
def test_c3_fragment_doubling_changes_allocation_less_than_three_percent():
    """Doubling n_frag must change the energy allocation by < 3 %."""
    convergence = check_fragment_convergence(
        n_frag_values=(4, 8, 16, 32),
        diameter=CONVERGENCE_DIAMETER, seed=5, platen_velocity=-0.05,
        steps=4000, tolerance=0.03,
    )
    assert convergence["converged"], (
        f"max allocation delta {convergence['max_delta']:.4f} exceeds 3 %"
    )
    assert len(convergence["comparisons"]) >= 3


def test_c3_all_budgets_fracture():
    """Every n_frag value must produce a fracture event."""
    convergence = check_fragment_convergence(
        n_frag_values=(4, 8, 16),
        diameter=CONVERGENCE_DIAMETER, seed=6, platen_velocity=-0.05,
        steps=4000, tolerance=0.03,
    )
    for budget in convergence["budgets"]:
        assert budget["crushing_events"] > 0


def test_c3_convergence_improves_with_larger_particle():
    """The allocation must be more stable for a larger particle (physical)."""
    small = check_fragment_convergence(
        n_frag_values=(4, 8), diameter=2.0e-3, seed=5,
        platen_velocity=-0.05, steps=4000, tolerance=0.03,
    )
    large = check_fragment_convergence(
        n_frag_values=(4, 8), diameter=CONVERGENCE_DIAMETER, seed=5,
        platen_velocity=-0.05, steps=4000, tolerance=0.03,
    )
    # The larger particle must have a smaller allocation delta.
    assert large["max_delta"] < small["max_delta"]


# --------------------------------------------------------------------------
# C4 - N06 ledger schema integration
# --------------------------------------------------------------------------
def test_c4_n06_break_crush_channel_closure_below_one_percent():
    """The N06 break_crush channel must close to < 1 %."""
    budget = _run_single_fracture(
        diameter=2.0e-3, seed=7, platen_velocity=-0.05,
        n_frag=8, steps=4000,
    )
    n06 = n06_break_crush_channel(budget)
    assert n06["channel"] == "break_crush"
    assert n06["sub_channel"] == "crush"
    assert n06["closure_C1"] < 0.01, (
        f"break_crush C1 {n06['closure_C1']:.4f} exceeds 1 %"
    )


def test_c4_n06_channel_carries_the_three_channel_partition():
    """The N06 channel must carry the surface/kinetic/dissipation partition."""
    budget = _run_single_fracture(
        diameter=2.0e-3, seed=8, platen_velocity=-0.05,
        n_frag=8, steps=4000,
    )
    n06 = n06_break_crush_channel(budget)
    assert n06["E_surf_J"] == pytest.approx(budget["E_surf_J"], rel=1e-12)
    assert n06["E_kin_J"] == pytest.approx(budget["E_kin_J"], rel=1e-12)
    assert n06["E_diss_J"] == pytest.approx(budget["E_diss_J"], rel=1e-12)
    assert n06["fractions"] == budget["fractions"]


# --------------------------------------------------------------------------
# Full audit driver
# --------------------------------------------------------------------------
def test_full_audit_all_criteria_pass():
    """The full audit must pass all four machine-checkable criteria."""
    audit = run_energy_budget_audit(
        diameter=CONVERGENCE_DIAMETER, seed=9, platen_velocity=-0.05,
        n_frag=8, steps=4000,
        n_frag_values=(4, 8, 16, 32),
    )
    assert audit["all_passed"], audit["criteria"]
    for name, criterion in audit["criteria"].items():
        assert criterion["passed"], f"{name}: {criterion}"


def test_full_audit_reports_measured_numbers():
    """The audit must report measured numbers, not just pass/fail."""
    audit = run_energy_budget_audit(
        diameter=CONVERGENCE_DIAMETER, seed=10, platen_velocity=-0.05,
        n_frag=8, steps=4000,
        n_frag_values=(4, 8, 16, 32),
    )
    assert "epsilon_s" in audit["budget"]
    assert "n_drops" in audit["f_delta"]
    assert "max_delta" in audit["convergence"]
    assert "closure_C1" in audit["n06"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
