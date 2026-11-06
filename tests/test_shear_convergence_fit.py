"""Coverage for :mod:`tensordem.shear_convergence`.

``shear_convergence`` (the cross-section convergence study of the DEM3D
direct-shear apparent modulus) had no dedicated test file, so the three
pieces of arithmetic that the convergence claim rests on were unguarded:

* :func:`boundary_layer_fit` -- the least-squares boundary-layer series on
  ``x = 1/n``, the observed convergence-order estimate and the two-point
  Richardson limit.  Tested here by synthetic recovery: a deficit generated
  from a known ``c1/n + c2/n**2`` must be inverted to the same coefficients,
  model ``C`` must report a zero asymptotic bias, and a deliberately injected
  constant offset must be found by model ``C`` (the falsification branch).
* :func:`cross_section_series` -- argument guards run *before* any DEM solve,
  plus one 4-size series cross-checked against a standalone
  :func:`tensordem.shear_static.static_direct_shear` call.
* :func:`surface_lag_profile` -- the near-surface displacement lag on a small
  5x5x3 specimen, and its square/odd-``nz`` preconditions.

Negative cases (``pytest.raises``) cover the input guards; if a guard were
dropped or its message/logic inverted, the corresponding test fails.

Note that ``tensordem.shear_static`` carries a sibling series/fit pair
(``cross_section_series``, ``fit_boundary_layer``) which already has coverage in
``tests/test_dem3d_shear_cross_section_convergence.py``; this file targets the
``shear_convergence`` implementation, which was not imported by any test.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tensordem import shear_convergence as sc
from tensordem.dem3d import DEM3DConfig
from tensordem.shear_specimen import shear_modulus_closed_form
from tensordem.shear_static import static_direct_shear

SIZES = (5, 7, 9, 11, 13)
C1 = 1.2
C2 = 2.0


def _synthetic_deficits(offset: float = 0.0) -> list[float]:
    """Deficit series of a pure boundary layer ``c1/n + c2/n**2`` (plus offset)."""
    return [offset + C1 / n + C2 / n ** 2 for n in SIZES]


def test_boundary_layer_fit_recovers_the_generating_coefficients() -> None:
    fit = sc.boundary_layer_fit(SIZES, _synthetic_deficits())

    assert fit["models"]["A"]["powers"] == [1]
    assert fit["models"]["B"]["powers"] == [1, 2]
    assert fit["models"]["C"]["powers"] == [0, 1, 2]

    coefficients = fit["models"]["B"]["coefficients"]
    assert coefficients["c1"] == pytest.approx(C1, rel=1e-9)
    assert coefficients["c2"] == pytest.approx(C2, rel=1e-9)
    assert fit["models"]["B"]["rms_residual"] < 1e-12
    assert fit["models"]["B"]["dof"] == len(SIZES) - 2

    # a pure boundary layer leaves no asymptotic bias behind
    assert fit["models"]["C"]["coefficients"]["c0"] == pytest.approx(0.0, abs=1e-12)
    assert fit["asymptotic_deficit_model_C"] == pytest.approx(0.0, abs=1e-12)
    assert fit["extrapolated_G_over_C44_model_C"] == pytest.approx(1.0, abs=1e-12)

    # the observed order is the local slope at the three finest sizes: it must
    # reproduce the documented estimator applied to the synthetic series
    deficits = _synthetic_deficits()
    ratio = (deficits[-3] - deficits[-2]) / (deficits[-2] - deficits[-1])
    expected_order = np.log(ratio) / np.log(SIZES[-2] / SIZES[-3])
    assert fit["observed_order"] == pytest.approx(expected_order, rel=1e-12)

    # extrapolation must reduce (never inflate) the finest measured deficit
    assert 0.0 < fit["richardson_asymptotic_deficit"] < fit["deficit_at_largest_n"]


def test_observed_order_and_limit_are_exact_on_a_geometric_size_grid() -> None:
    """A pure 1/n boundary layer on a geometric grid has order 1 and limit 1."""
    sizes = (8, 12, 18, 27)
    deficits = [C1 / n for n in sizes]
    fit = sc.boundary_layer_fit(sizes, deficits, gate=0.03)

    assert fit["observed_order"] == pytest.approx(1.0, rel=1e-12)
    assert fit["richardson_asymptotic_deficit"] == pytest.approx(0.0, abs=1e-12)
    assert fit["richardson_extrapolated_G_over_C44"] == pytest.approx(1.0, rel=1e-12)
    # gate crossing of a pure 1/n layer is exactly c1 / gate
    assert fit["predicted_n_for_gate"] == pytest.approx(C1 / 0.03, rel=1e-12)


def test_gate_prediction_is_self_consistent() -> None:
    gate = 0.03
    fit = sc.boundary_layer_fit(SIZES, _synthetic_deficits(), gate=gate)

    n_gate = fit["predicted_n_for_gate"]
    c1 = fit["models"]["B"]["coefficients"]["c1"]
    c2 = fit["models"]["B"]["coefficients"]["c2"]
    assert c1 / n_gate + c2 / n_gate ** 2 == pytest.approx(gate, rel=1e-9)
    assert n_gate > SIZES[-1]

    assert fit["gate_met_by_measurement"] is False
    assert fit["deficit_at_largest_n"] == pytest.approx(
        1.0 - (1.0 - (C1 / SIZES[-1] + C2 / SIZES[-1] ** 2)), rel=1e-12)


def test_model_c_exposes_a_constant_measurement_bias() -> None:
    """Falsification branch: an ``O(1)`` bias must not be absorbed by the 1/n terms."""
    offset = 0.02
    fit = sc.boundary_layer_fit(SIZES, _synthetic_deficits(offset=offset))

    assert fit["asymptotic_deficit_model_C"] == pytest.approx(offset, abs=1e-12)
    assert fit["extrapolated_G_over_C44_model_C"] == pytest.approx(1.0 - offset, abs=1e-12)
    # with the bias included the residual collapses to round-off
    assert fit["models"]["C"]["rms_residual"] < 1e-12
    # ... and the bias-free model cannot explain the data
    assert fit["models"]["B"]["rms_residual"] > 1e-3


@pytest.mark.parametrize(
    ("sizes", "deficits", "match"),
    [
        (SIZES[:3], _synthetic_deficits()[:3], "need >= 4"),
        (SIZES, _synthetic_deficits()[:-1], "need >= 4"),
        ((5, 7, 7, 11), (0.3, 0.2, 0.15, 0.1), "must increase"),
        (SIZES, (0.3, 0.3, 0.3, 0.3, 0.3), "must increase"),
        (SIZES, (0.3, 0.2, -0.1, 0.05, 0.04), "finite and non-negative"),
        (SIZES, (0.3, 0.2, float("nan"), 0.05, 0.04), "finite and non-negative"),
    ],
)
def test_boundary_layer_fit_rejects_invalid_series(sizes, deficits, match) -> None:
    with pytest.raises(ValueError, match=match):
        sc.boundary_layer_fit(sizes, deficits)


def test_cross_section_series_rejects_degenerate_size_sets() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sc.cross_section_series([5, 7, 9])
    with pytest.raises(ValueError, match="strictly increasing"):
        sc.cross_section_series([5, 7, 7, 11])


def test_cross_section_series_matches_the_standalone_static_solve() -> None:
    series = sc.cross_section_series([5, 7, 9, 11])
    rows = series["rows"]

    assert [row["n"] for row in rows] == [5, 7, 9, 11]
    assert series["fit"]["largest_n_measured"] == 11
    assert series["fit"]["gate_met_by_measurement"] is False

    for row in rows:
        n = row["n"]
        assert row["particle_count"] == n * n * series["nz"]
        assert row["height_over_spacing"] == pytest.approx(series["nz"] - 1)
        assert row["lateral_boundary_fraction"] == pytest.approx(1.0 - (n - 2.0) ** 2 / n ** 2)
        assert row["deficit"] == pytest.approx(1.0 - row["G_effective_over_C44"], abs=1e-15)
        assert row["reaction_relative_error"] < 1e-5

    deficits = [row["deficit"] for row in rows]
    assert all(later < earlier for earlier, later in zip(deficits, deficits[1:]))

    # the coarsest row must reproduce a direct static solve of the same specimen
    cfg = replace(DEM3DConfig(), nx=5, ny=5, nz=3, drag=0.0, fix_x_edges=False)
    direct = static_direct_shear(cfg, gamma=series["gamma"])
    assert rows[0]["G_effective_over_C44"] == pytest.approx(
        direct["G_effective_over_C44"], rel=1e-12)
    assert rows[0]["C44_Pa"] == pytest.approx(shear_modulus_closed_form(cfg), rel=1e-12)


def test_surface_lag_profile_reports_a_lag_behind_the_affine_field() -> None:
    profile = sc.surface_lag_profile(n=5)
    assert profile["n"] == 5
    assert profile["nz"] == 3

    bins = profile["bins"]
    # every node of the mid layer is binned exactly once (5 x 5 nodes)
    assert sum(entry["node_count"] for entry in bins) == 25
    assert [entry["distance_over_spacing"] for entry in bins] == sorted(
        entry["distance_over_spacing"] for entry in bins)
    # the solved field lags the affine field u_x = gamma z
    assert all(entry["mean_lag"] < 0.0 for entry in bins)
    assert profile["mean_lag_overall"] < 0.0
    # the lag is largest at the centre and smallest at the free surface
    assert abs(profile["lag_at_center"]) > abs(bins[0]["mean_lag"])
    assert profile["G_effective_over_C44"] < 1.0


def test_surface_lag_profile_rejects_non_square_and_even_nz() -> None:
    with pytest.raises(ValueError, match="square cross-section"):
        sc.surface_lag_profile(replace(DEM3DConfig(), nx=5, ny=7, drag=0.0))
    with pytest.raises(ValueError, match="odd nz"):
        sc.surface_lag_profile(replace(DEM3DConfig(), nx=5, ny=5, nz=4, drag=0.0))