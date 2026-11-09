"""Scaling law fitting, monotonicity and load-envelope assembly."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.myi_ridge import scaling


def test_power_law_fit_recovers_exponent():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = 3.5 * x ** 1.37
    fit = scaling.fit_power_law(x, y)
    assert fit["exponent"] == pytest.approx(1.37, rel=1e-9)
    assert fit["prefactor"] == pytest.approx(3.5, rel=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)


def test_power_law_fit_reports_uncertainty():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x ** 1.0 * np.array([1.0, 1.02, 0.98, 1.01, 0.99])
    fit = scaling.fit_power_law(x, y)
    assert fit["exponent"] == pytest.approx(1.0, abs=0.02)
    assert fit["exponent_stderr"] > 0.0
    assert fit["rms_log_residual"] > 0.0


def test_two_regressor_exponents_recovered():
    points = []
    for sigma in (5.0e6, 7.5e6, 10.0e6):
        for depth in (10.0, 20.0, 30.0):
            load = 4.0e-3 * sigma ** 1.0 * depth ** 1.0
            points.append({"sigma_ice": sigma, "h_keel": depth, "F_max": load})
    fit = scaling.fit_exponents(points)
    assert fit["exponent_sigma"] == pytest.approx(1.0, rel=1e-9)
    assert fit["exponent_depth"] == pytest.approx(1.0, rel=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)


def test_monotone_increasing_flags_violation():
    assert scaling.monotone_increasing([1.0, 2.0, 3.0])
    assert not scaling.monotone_increasing([1.0, 3.0, 2.0])
    assert scaling.monotone_increasing([1.0, 2.0, 1.99], tolerance=0.02)
    report = scaling.monotonicity_report({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    assert report["a"]["monotone_increasing"] is True
    assert report["b"]["monotone_increasing"] is False
    assert report["all_monotone"] is False


def test_extreme_load_scales_with_contact_area():
    assert scaling.extreme_load(2.0e6, 20.0, 1.5) == pytest.approx(6.0e7)
    double_depth = scaling.extreme_load(2.0e6, 40.0, 1.5)
    assert double_depth == pytest.approx(2.0 * 6.0e7)
    with pytest.raises(ValueError):
        scaling.extreme_load(-1.0, 20.0, 1.5)


def test_load_envelope_assembles_cases_and_sweeps():
    cases = []
    for sigma in (5.0e6, 7.5e6, 10.0e6):
        for depth in (10.0, 20.0, 30.0):
            cases.append({
                "sigma_ice": sigma,
                "h_keel": depth,
                "hull_velocity": 2.0,
                "porosity": 0.30,
                "pressure_steady": 1.2e6 * (sigma / 7.5e6) ** 1.0 * (depth / 20.0) ** 0.1,
                "closure_of_exchanged": 0.01,
            })
    envelope = scaling.load_envelope(cases, loaded_length=1.5)
    assert envelope["case_count"] == 9
    assert envelope["exponents"]["exponent_sigma"] == pytest.approx(1.0, abs=0.02)
    assert envelope["monotonicity"]["sigma_ice"]["monotone_increasing"] is True
    assert envelope["load_max"] > envelope["load_min"]
    with pytest.raises(ValueError):
        scaling.load_envelope([], loaded_length=1.5)


def test_fit_requires_three_points_and_positive_data():
    with pytest.raises(ValueError):
        scaling.fit_power_law(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        scaling.fit_power_law(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 3.0]))