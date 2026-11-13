"""Fast regression tests for the finite-size-before-controller switch.

The physics being pinned down:

* ``FixedSphereSpec.correction_timing`` must change the *seed* force of the
  constant-flow-rate pump by exactly the periodic-array ratio lambda(phi) and
  nothing else (the converged force is controller-pinned, so no other field may
  depend on the flag).
* ``SettlingSpec.finite_size_in_drive`` must scale the effective weight by
  lambda(phi) *before* the run, and the post-hoc multiplication must be
  switchable independently, because doing both double-counts the correction.

Every case here is a 24^3--32^3 lattice and <=100 steps, so the file stays in a
normal CI budget; the production sweep lives in
``benchmarks/case_dem_finite_size_before_controller``.
"""

from __future__ import annotations

import math

import pytest

from tensordem.lbmdem import (
    FixedSphereSpec,
    SettlingSpec,
    array_correction,
    run_fixed_sphere,
    run_settling,
)
from tensordem.lbmdem.calibration import _solid_fraction_of_box


def _phi_for(diameter: float, domain_ratio: float) -> float:
    length = diameter * domain_ratio
    return math.pi * diameter**3 / 6.0 / length**3


def test_array_ratio_geometry() -> None:
    for d, ld in ((4.0, 4.0), (6.0, 8.0), (8.0, 16.0)):
        phi = _phi_for(d, ld)
        assert phi == pytest.approx(math.pi / 6.0 / ld**3)
        lam = array_correction(phi)
        assert lam == pytest.approx(1.0 + 1.7601 * phi ** (1.0 / 3.0) + phi)
    assert array_correction(0.0) == 1.0


def test_correction_timing_validation() -> None:
    with pytest.raises(ValueError):
        FixedSphereSpec(correction_timing="during")


def test_fixed_sphere_seed_scales_with_correction() -> None:
    """'after' seeds the unbounded force; 'before' seeds lambda times that."""
    common = dict(diameter=6.0, domain_ratio=4.0, re_target=0.05, max_steps=40,
                  min_steps=10**9, sample_every=40, scheme="mem", dtype="float32")
    before = run_fixed_sphere(FixedSphereSpec(correction_timing="before", **common))
    after = run_fixed_sphere(FixedSphereSpec(correction_timing="after", **common))
    lam = before["array_correction"]
    assert lam > 1.0
    assert before["seed_correction"] == pytest.approx(lam)
    assert after["seed_correction"] == 1.0
    assert before["force_estimated"] / after["force_estimated"] == pytest.approx(lam, rel=1e-12)
    assert before["force_estimated_unbounded"] == pytest.approx(after["force_estimated"])
    # the reported post-hoc ratio is defined identically for both timings
    assert before["stokes_ratio_corrected"] == pytest.approx(
        before["stokes_ratio"] / lam, rel=1e-12)
    # controller seed only: the flag must not leak into the measured quantities
    assert before["momentum_balance_ratio"] > 0.0
    assert after["momentum_balance_ratio"] > 0.0


def test_settling_drive_scales_with_correction() -> None:
    common = dict(diameter=8.0, domain_ratio=4.0, re_target=1.0, steps=40,
                  sample_every=5, window=10, dtype="float32")
    m0 = run_settling(SettlingSpec(finite_size_in_drive=False,
                                   apply_array_correction_posthoc=True, **common))
    m1 = run_settling(SettlingSpec(finite_size_in_drive=True,
                                   apply_array_correction_posthoc=True, **common))
    m2 = run_settling(SettlingSpec(finite_size_in_drive=True,
                                   apply_array_correction_posthoc=False, **common))
    lam = m0["array_correction"]
    assert lam > 1.0
    assert m1["gravity"] / m0["gravity"] == pytest.approx(lam, rel=1e-12)
    assert m2["gravity"] == pytest.approx(m1["gravity"], rel=1e-12)
    assert m0["double_counts_correction"] is False
    assert m1["double_counts_correction"] is True
    assert m2["double_counts_correction"] is False
    # the post-hoc switch only rescales the reported corrected velocity
    assert m2["u_relative_terminal_corrected"] == pytest.approx(
        m2["u_relative_terminal"], rel=1e-12)
    assert m1["u_relative_terminal_corrected"] == pytest.approx(
        m1["u_relative_terminal"] * lam, rel=1e-12)
    assert m1["u_relative_terminal"] == pytest.approx(m2["u_relative_terminal"], rel=1e-9)


def test_settling_operating_point_bias_prediction() -> None:
    """The M0 drive is built for unbounded Stokes drag, so the achieved
    velocity is predicted to sit at 1/lambda of the target (26 % low at L/d=4).
    After 40 steps the sphere is still accelerating, so only the drive ratio is
    checked here; the settled values are measured in the benchmark."""
    spec = dict(diameter=8.0, domain_ratio=4.0, re_target=1.0, steps=20,
                sample_every=5, window=5, dtype="float32")
    m0 = run_settling(SettlingSpec(**spec))
    lam = m0["array_correction"]
    # weight set for unbounded Stokes balance: 3 pi mu d u_guess = (m - V) g
    assert m0["drive_force"] == pytest.approx(
        3.0 * math.pi * m0["nu"] * 8.0 * m0["u_guess"], rel=1e-9)
    m2 = run_settling(SettlingSpec(finite_size_in_drive=True,
                                   apply_array_correction_posthoc=False, **spec))
    assert m2["drive_force"] / m0["drive_force"] == pytest.approx(lam, rel=1e-9)