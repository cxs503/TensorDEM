"""Tests for the Hertz/Tsuji restitution mapping (``tensordem.hertz_restitution``).

Grading references:

* ``alpha = 0`` reproduces the textbook elastic Hertz impact contact time
  ``2.94 (5m/(4k_H))^(2/5) v^(-1/5)`` -- an *external* constant, so the
  reference quadrature is verified before it is used as a reference itself;
* the inverse map ``e -> alpha`` round-trips to machine-level accuracy;
* the DEM3D engine, driven through ``DEM3DConfig(contact_model="hertz")``,
  realises the requested restitution and is impact-speed independent;
* the legacy linear-law mapping bias is pinned so the historical numbers cannot
  silently return.
"""

from __future__ import annotations

import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.hertz_restitution import (
    ELASTIC_CONTACT_TIME_CONSTANT,
    alpha_for_restitution,
    dimensionless_contact_time,
    elastic_hertz_contact_time,
    hertz_contact_time,
    linear_law_alpha,
    restitution_for_alpha,
    tsuji_damping_alpha,
)

YOUNG = 1.0e9
NU = 0.3
RADIUS = 0.05
DENSITY = 900.0


def _pair(*, restitution: float, mapping: str = "exact", dt_scale: float = 0.5):
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY, drag=0.0,
        bond_stiffness=1.0, breaking_strain=1.0e-12, shear_breaking_strain=1.0e-12,
        fix_x_edges=False, fix_bottom=False, tool_gap=1.0e3, tool_speed=1.0e-9,
        contact_model="hertz", young_modulus=YOUNG, poisson_ratio=NU,
        restitution_coefficient=restitution, hertz_damping_mapping=mapping,
    )
    dem = IceDEM3D(cfg)
    n = len(dem.positions)
    pos = torch.zeros_like(dem.positions)
    for i in range(n):
        pos[i] = torch.tensor([10.0 * i, 500.0, 500.0], dtype=dem.dtype)
    pos[0] = torch.tensor([0.0, 0.0, -(RADIUS + 1.0e-6)], dtype=dem.dtype)
    pos[1] = torch.tensor([0.0, 0.0, RADIUS + 1.0e-6], dtype=dem.dtype)
    dem.positions = pos.clone()
    dem.initial_positions = pos.clone()
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    mask[1] = False
    dem.fixed = mask
    dem.forces()
    if int(dem.alive.sum().item()) != 0:
        raise RuntimeError("pair harness assumption violated")
    dem.dt = dem.dt * dt_scale
    dem.velocities = torch.zeros_like(dem.positions)
    return dem


def _impact(*, restitution: float, mapping: str = "exact", v0: float = 0.5) -> dict:
    dem = _pair(restitution=restitution, mapping=mapping)
    dem.velocities[0] = torch.tensor([0.0, 0.0, v0], dtype=dem.dtype)
    dem.velocities[1] = torch.tensor([0.0, 0.0, -v0], dtype=dem.dtype)
    contact_steps = 0
    force_steps = 0
    peak = 0.0
    for _ in range(200_000):
        v_before = float(dem.velocities[0, 2])
        dem.step()
        if float(dem.velocities[0, 2]) != v_before:
            force_steps += 1
        overlap = 2.0 * RADIUS - float(dem.positions[1, 2] - dem.positions[0, 2])
        if overlap > 0.0:
            contact_steps += 1
            peak = max(peak, overlap)
        elif contact_steps:
            break
    else:  # pragma: no cover
        raise RuntimeError("pair did not separate")
    return {
        "e": -float(dem.velocities[0, 2]) / v0,
        "contact_time_force": force_steps * dem.dt,
        "peak_overlap": peak,
        "dt": dem.dt,
    }


class ReferenceQuadratureTest(unittest.TestCase):
    def test_elastic_limit_restitution_is_unity(self) -> None:
        self.assertAlmostEqual(restitution_for_alpha(0.0), 1.0, places=8)

    def test_elastic_contact_time_matches_textbook_constant(self) -> None:
        anchor = dimensionless_contact_time(0.0) * 2.0 / math.sqrt(5.0)
        self.assertLess(abs(anchor / ELASTIC_CONTACT_TIME_CONSTANT - 1.0), 0.01,
                        msg=f"quadrature anchor {anchor:.6f} vs 2.94")

    def test_dimensional_contact_time_reduces_to_textbook_at_zero_damping(self) -> None:
        m_eff, k_h, v0 = 2.0e-3, 3.0e6, 1.0
        # agreement is limited only by the rounding of the 2.94 constant
        self.assertLess(abs(hertz_contact_time(m_eff, k_h, v0, 0.0)
                            / elastic_hertz_contact_time(m_eff, k_h, v0) - 1.0), 2e-3)

    def test_map_is_monotone(self) -> None:
        alphas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
        values = [restitution_for_alpha(a) for a in alphas]
        for lo, hi in zip(values, values[1:]):
            self.assertGreater(lo, hi)

    def test_inverse_map_round_trips(self) -> None:
        for target in (0.30, 0.50, 0.85, 0.95):
            alpha = alpha_for_restitution(target)
            self.assertLess(abs(restitution_for_alpha(alpha) / target - 1.0), 1e-6)

    def test_legacy_linear_mapping_bias_is_pinned(self) -> None:
        # The historical linear-law mapping is exact for the linear dashpot but
        # over-predicts restitution for a Hertz contact; the bias is the reason
        # this module exists and must not silently return.
        bias_030 = restitution_for_alpha(linear_law_alpha(0.30)) / 0.30 - 1.0
        bias_050 = restitution_for_alpha(linear_law_alpha(0.50)) / 0.50 - 1.0
        self.assertGreater(bias_030, 0.40)
        self.assertGreater(bias_050, 0.15)
        # ... and the corrected map removes it.
        for target in (0.30, 0.50):
            self.assertLess(abs(restitution_for_alpha(alpha_for_restitution(target))
                                / target - 1.0), 1e-6)

    def test_invalid_arguments_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tsuji_damping_alpha(0.5, mapping="nope")
        with self.assertRaises(ValueError):
            alpha_for_restitution(1.5)
        with self.assertRaises(ValueError):
            restitution_for_alpha(-0.1)


class EngineMappingTest(unittest.TestCase):
    def test_engine_realises_requested_restitution(self) -> None:
        for target in (0.50, 0.85):
            measured = _impact(restitution=target)["e"]
            self.assertLess(abs(measured / target - 1.0), 0.02,
                            msg=f"e_target={target} measured={measured:.5f}")

    def test_engine_restitution_is_speed_independent(self) -> None:
        values = [_impact(restitution=0.50, v0=v)["e"] for v in (0.2, 2.0)]
        self.assertLess(max(values) / min(values) - 1.0, 0.02, msg=str(values))

    def test_engine_contact_time_matches_reference(self) -> None:
        m_eff = 0.5 * (DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3)
        k_h = (4.0 / 3.0) * (YOUNG / (2.0 * (1.0 - NU ** 2))) * math.sqrt(0.5 * RADIUS)
        for target in (0.50, 0.85):
            result = _impact(restitution=target)
            reference = hertz_contact_time(
                m_eff, k_h, 1.0, alpha_for_restitution(target))
            self.assertLess(abs(result["contact_time_force"] / reference - 1.0), 0.05)

    def test_legacy_option_reproduces_linear_law_mapping(self) -> None:
        target = 0.30
        legacy = _impact(restitution=target, mapping="linear_formula")["e"]
        expected = restitution_for_alpha(linear_law_alpha(target))
        self.assertLess(abs(legacy / expected - 1.0), 0.02)
        self.assertGreater(legacy / target - 1.0, 0.30)

    def test_elastic_limit_leaves_dashpot_off(self) -> None:
        dem = _pair(restitution=1.0)
        self.assertEqual(float(dem._hertz_damping_alpha), 0.0)
        self.assertAlmostEqual(float(dem._hertz_damping_alpha), 0.0, places=12)

    def test_default_mapping_and_validation(self) -> None:
        self.assertEqual(DEM3DConfig().hertz_damping_mapping, "exact")
        with self.assertRaises(ValueError):
            DEM3DConfig(hertz_damping_mapping="bogus")
        # The linear model keeps the legacy (exact for it) mapping untouched.
        self.assertEqual(
            DEM3DConfig(restitution_coefficient=0.5).contact_model, "linear")


if __name__ == "__main__":
    unittest.main()