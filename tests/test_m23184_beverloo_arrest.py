"""m23184 — arrest / arch detection for the Beverloo discharge measurement.

The defect these tests pin: `run_discharge` returned a *well-formed* flow rate
for a bed that had arched over the outlet.  The archived campaign record
`runs/m23114_siloflow_ladder/dt_nom20_s1.json` is such a run — window particle
counts ``[2, 0, 0, 0, 0, 4, 0, 15]``, W = 0.003358 kg/s, i.e. ``-88.5 %`` of the
Beverloo reference — and it was filed as a measurement rather than as an arch.
`discharge_status` now labels it, without changing `flow_rate_kg_s`.
"""
from __future__ import annotations

import math

import pytest

from tensordem.silo_flow.beverloo import (
    ARREST_EMPTY_WINDOW_FRACTION,
    ARREST_RATE_FRACTION,
    beverloo_prediction,
    discharge_status,
)

# quoted verbatim from the two archived records of the *same* frozen case
# (d = 3 mm, D/d = 5, kn = 5e3 N/m, seed 1) — see docs/CASE-DEM-SILO-BEVERLOO_m23184_reproducibility_audit.md
ARCHIVED_ARRESTED = {
    "window_particles": [2, 0, 0, 0, 0, 4, 0, 15],
    "window_time_s": 0.05,
    "flow_rate_kg_s": 0.0033577361261474805,
    "rho_bulk": 1323.7847222222226,
}
ARCHIVED_FLOWING = {
    "window_particles": [28, 42, 41, 40, 38, 47, 41, 48],
    "window_time_s": 0.05,
    "flow_rate_kg_s": 0.030749793997350797,
    "rho_bulk": 1323.7847222222226,
}
CASE = {"orifice_diameter": 0.015, "particle_diameter": 0.003}


def _rates(record: dict) -> list[float]:
    """Window rates implied by the particle counts (m equal per particle)."""
    counts = record["window_particles"]
    mass = 2500.0 * 4.0 / 3.0 * math.pi * (0.5 * CASE["particle_diameter"]) ** 3
    return [c * mass / record["window_time_s"] for c in counts]


def test_archived_arrested_record_is_labelled_arrested() -> None:
    status = discharge_status(
        ARCHIVED_ARRESTED["window_particles"],
        _rates(ARCHIVED_ARRESTED),
        reference_rate=beverloo_prediction(
            CASE["orifice_diameter"], CASE["particle_diameter"], ARCHIVED_ARRESTED["rho_bulk"]),
    )
    assert status["status"] == "ARRESTED"
    assert status["arrested"] is True
    assert status["reasons"], "an arrest must always say why"
    assert status["empty_window_fraction"] >= ARREST_EMPTY_WINDOW_FRACTION
    assert status["rate_over_reference"] < ARREST_RATE_FRACTION


def test_archived_flowing_record_is_labelled_flowing() -> None:
    status = discharge_status(
        ARCHIVED_FLOWING["window_particles"],
        _rates(ARCHIVED_FLOWING),
        reference_rate=beverloo_prediction(
            CASE["orifice_diameter"], CASE["particle_diameter"], ARCHIVED_FLOWING["rho_bulk"]),
    )
    assert status["status"] == "FLOWING"
    assert status["arrested"] is False
    assert status["reasons"] == []
    assert abs(status["rate_over_reference"] - 1.0) < 0.15


def test_empty_windows_alone_arrest_a_record() -> None:
    # five of eight windows empty but a positive mean: burst discharge
    counts = [0, 60, 0, 0, 0, 55, 0, 0]
    rates = _rates({"window_particles": counts, "window_time_s": 0.05})
    status = discharge_status(counts, rates)
    assert status["arrested"] is True
    assert status["empty_window_fraction"] == pytest.approx(6 / 8)


def test_no_particle_at_all_is_arrested() -> None:
    counts = [0] * 6
    status = discharge_status(counts, [0.0] * 6, reference_rate=1.0)
    assert status["arrested"] is True
    assert status["counting_noise_rel"] == math.inf
    assert any("no particle exited" in r for r in status["reasons"])


def test_rate_below_reference_fraction_is_arrested_even_if_steady() -> None:
    # perfectly steady but at 10 % of the correlation -> an arch, not an offset
    counts = [4] * 8
    rates = _rates({"window_particles": counts, "window_time_s": 0.05})
    ref = 10.0 * rates[-1]
    status = discharge_status(counts, rates, reference_rate=ref)
    assert status["arrested"] is True
    assert status["rate_over_reference"] == pytest.approx(0.1, rel=1e-9)


def test_collapse_inside_the_measurement_is_arrested() -> None:
    counts = [40, 40, 40, 40, 2, 1, 1, 1]
    rates = _rates({"window_particles": counts, "window_time_s": 0.05})
    status = discharge_status(counts, rates, reference_rate=rates[0])
    assert status["arrested"] is True
    assert status["last_over_first_half_rate"] < ARREST_RATE_FRACTION * 1.0


def test_no_reference_means_the_reference_rule_is_skipped() -> None:
    counts = [4] * 8
    rates = _rates({"window_particles": counts, "window_time_s": 0.05})
    status = discharge_status(counts, rates)          # reference_rate=None
    assert status["reference_rate_kg_s"] is None
    assert status["rate_over_reference"] is None
    assert status["arrested"] is False


def test_zero_reference_is_skipped_not_divided() -> None:
    # D < k d: the correlation is zero, the ratio must not be formed
    counts = [10] * 6
    rates = _rates({"window_particles": counts, "window_time_s": 0.05})
    status = discharge_status(counts, rates, reference_rate=0.0)
    assert status["rate_over_reference"] is None
    assert status["arrested"] is False


def test_counting_noise_floor_is_inverse_root_n() -> None:
    counts = [10, 20, 30, 40, 50, 60, 70, 80]
    rates = _rates({"window_particles": counts, "window_time_s": 0.05})
    status = discharge_status(counts, rates)
    n_steady = sum(counts[-4:])
    assert status["particles_in_steady_windows"] == n_steady
    assert status["counting_noise_rel"] == pytest.approx(1.0 / math.sqrt(n_steady))


def test_surface_drop_diagnostics_are_reported_not_decided() -> None:
    counts = [40] * 8
    rates = _rates({"window_particles": counts, "window_time_s": 0.05})
    status = discharge_status(counts, rates, surface_drop_m=0.003, particle_diameter=0.003)
    assert status["surface_drop_m"] == pytest.approx(0.003)
    assert status["surface_drop_over_particle"] == pytest.approx(1.0)
    assert status["arrested"] is False          # diagnostics never decide


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        discharge_status([1, 2, 3], [1.0, 2.0])


def test_empty_series_raise() -> None:
    with pytest.raises(ValueError):
        discharge_status([], [])


def test_run_discharge_record_carries_the_diagnosis() -> None:
    """A record must expose ``arrested`` and the full status block."""
    import inspect

    from tensordem.silo_flow import beverloo

    src = inspect.getsource(beverloo.run_discharge)
    assert '"discharge_status": status' in src
    assert '"arrested": bool(status["arrested"])' in src
    assert "reference_rate=beverloo_prediction(" in src