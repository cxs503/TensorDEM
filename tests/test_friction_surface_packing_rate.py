"""The inflation-quench RCP route must report measurable, non-tautological quantities.

These tests pin the two things the audits of ``CASE-DEM-RCP-PACKING-FRACTION``
demanded:

1. ``phi`` must be a *derived* quantity (a function of the relaxed structure), never
   an echo of the input -- asserted by the negative control in
   ``test_packing_fraction_is_not_an_echo_of_the_start_density``;
2. protocol parameters must either work or be rejected loudly -- asserted by the
   ``mu`` and ``PackingCase`` validation tests (a phantom ``mu`` used to be echoed
   into the result dict while the contact law ignored it).
"""
import math

import numpy as np
import pytest

from tensordem.friction_surface.packing import PackingCase, random_close_packing, solid_fraction

CHEAP = dict(n_particles=200, phi_start=0.30, rho_growth=5.0e-3, relax_steps=30,
             max_inflation_steps=600, overlap_stop=2.0e-2)


def _cheap(**kw):
    cfg = dict(CHEAP)
    cfg.update(kw)
    return PackingCase(**cfg)


def test_rate_per_step_is_the_logarithmic_density_growth():
    case = _cheap(rho_growth=1.0e-4)
    assert case.rate_per_step == pytest.approx(3.0 * math.log1p(1.0e-4), rel=1e-12)


def test_nonzero_friction_is_rejected_instead_of_silently_ignored():
    """`mu` used to be recorded in the result while the contact law ignored it."""
    with pytest.raises(NotImplementedError, match="frictional inflation"):
        _cheap(mu=0.3)


def test_frictionless_is_accepted_and_reported():
    out = random_close_packing(_cheap(mu=0.0))
    assert out["mu"] == 0.0


@pytest.mark.parametrize("field,value", [
    ("n_particles", 1),
    ("rho_growth", 0.0),
    ("relax_steps", 0),
    ("overlap_stop", 0.0),
    ("cell", -1.0),
])
def test_invalid_protocol_parameters_are_rejected(field, value):
    with pytest.raises(ValueError):
        _cheap(**{field: value})


def test_result_reports_the_protocol_it_used():
    out = random_close_packing(_cheap())
    for key in ("relax_steps", "rho_growth", "rate_per_step", "overlap_stop",
                "phi_max_reached", "jammed_by_criterion", "protocol"):
        assert key in out
    assert out["relax_steps"] == CHEAP["relax_steps"]
    assert out["rho_growth"] == CHEAP["rho_growth"]
    assert out["phi_max_reached"] == pytest.approx(out["history"][-1][0])


def test_packing_fraction_is_not_an_echo_of_the_start_density():
    """Negative control: different starts must give *different* jamming densities.

    The geometric route's failure mode was ``phi == 1 - target_porosity`` to 1e-16.
    Here the start density is an initial condition; the measured jamming density
    must move with the seed instead of being pinned by the input.
    """
    a = random_close_packing(_cheap(phi_start=0.20))
    b = random_close_packing(_cheap(phi_start=0.40))
    assert a["jammed_packing_fraction"] != pytest.approx(b["jammed_packing_fraction"],
                                                         abs=1e-12)
    # and neither may coincide with the start density
    assert abs(a["jammed_packing_fraction"] - a["phi_start"]) > 1e-3
    assert abs(b["jammed_packing_fraction"] - b["phi_start"]) > 1e-3


def test_seed_changes_the_realisation_and_the_history_is_monotone_in_phi():
    a = random_close_packing(_cheap(seed=1))
    b = random_close_packing(_cheap(seed=2))
    assert a["jammed_packing_fraction"] != pytest.approx(b["jammed_packing_fraction"],
                                                         abs=1e-12)
    phi = np.array([row[0] for row in a["history"]])
    assert np.all(np.diff(phi) > 0)          # inflation only grows the spheres


def test_solid_fraction_scales_with_inverse_cell_volume():
    base = solid_fraction(1.0e-3, 1000, 1.0)
    assert solid_fraction(1.0e-3, 1000, 2.0) == pytest.approx(base / 8.0, rel=1e-12)


def _rate_run(rate, seed=1):
    return random_close_packing(PackingCase(
        n_particles=200, phi_start=0.30, rho_growth=rate, relax_steps=30,
        max_inflation_steps=4000, overlap_stop=2.0e-2, seed=seed))


def test_a_slower_inflation_rate_does_not_lower_the_jamming_density():
    """Kinetic trapping is the failure mode: faster inflation must not jam *denser*.

    The scan in ``benchmarks/rcp_ls_extrapolation`` quantifies the direction; this
    is the cheap one-sided guard so that a future regression reversing the sign
    cannot pass unnoticed.
    """
    fast = _rate_run(4.0e-3)["jammed_packing_fraction"]
    slow = _rate_run(1.0e-3)["jammed_packing_fraction"]
    assert slow >= fast - 0.02