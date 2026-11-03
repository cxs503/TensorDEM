"""Exact circular-arc bow on the production ``IceDEM`` path (opt-in ``hull_arc``).

The kernel ``tensordem.hull_geometry.arc_hull_contact_forces`` was previously
only reachable from audit scripts; these tests pin the integration in
``tensordem.dem.IceDEM``:

* the arc bow agrees with the closed-form ``k * overlap`` penalty to machine
  precision (``docs/dem2d_hull_polyline_geometry_audit.md`` established the
  polyline bias, this pins the cure),
* the arc solution is the limit of the polyline family, so refining the chord
  count is an *independent* reference for the whole time-integrated run,
* restart, energy and validation contracts survive the new hull type.
"""
import math
import unittest

import torch

from tensordem import DEMConfig, IceDEM
from tensordem.hull_geometry import (
    CircularArc,
    analytic_arc_penetration,
    arc_polyline,
    sagitta,
)

BOW_RADIUS = 0.25
BOW_SPAN = math.pi
BOW_CENTER = (0.0, -0.30)
HULL_SPEED = 0.2


def bow_arc():
    return CircularArc(center=BOW_CENTER, radius=BOW_RADIUS,
                       start_angle=0.0, span=BOW_SPAN)


def bow_profile(n_chords):
    return arc_polyline(BOW_RADIUS, BOW_SPAN, n_chords,
                        start_angle=0.0, center=BOW_CENTER)


def sheet_sim(chords=None, *, damping=5.0):
    cfg = DEMConfig(nx=11, ny=3, contact_damping=damping, tool_speed=HULL_SPEED)
    kw = dict(hull_start=(0.0, 0.0), hull_velocity=(0.0, HULL_SPEED))
    if chords is None:
        return IceDEM(cfg, hull_arc=bow_arc(), **kw)
    return IceDEM(cfg, hull_profile=bow_profile(chords), **kw)


class ArcBowGeometryTests(unittest.TestCase):
    """Single-disk force against the closed-form arc, on the production object."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _one_disk(self, chords, delta):
        cfg = DEMConfig(nx=2, ny=2, contact_damping=0.0, tool_speed=HULL_SPEED)
        angle = 0.5 * math.pi if chords is None else 0.5 * math.pi - 0.5 * BOW_SPAN / chords
        radial = BOW_RADIUS + cfg.tool_radius + cfg.radius - delta
        point = [BOW_CENTER[0] + radial * math.cos(angle),
                 BOW_CENTER[1] + radial * math.sin(angle)]
        kwargs = dict(hull_start=(0.0, 0.0), hull_velocity=(0.0, 0.0))
        sim = (IceDEM(cfg, hull_arc=bow_arc(), **kwargs) if chords is None
               else IceDEM(cfg, hull_profile=bow_profile(chords), **kwargs))
        sim.positions = torch.tensor(
            [point, [10.0, 10.0], [10.0, 20.0], [20.0, 10.0]], dtype=torch.float64)
        sim.velocities = torch.zeros_like(sim.positions)
        sim.alive = torch.zeros_like(sim.alive)  # isolate the hull contact
        return cfg, sim

    def test_arc_bow_matches_closed_form_penalty(self):
        for ratio in (0.002, 0.05, 0.2):
            cfg, sim = self._one_disk(None, ratio * 0.025)
            force, reaction = sim.forces()
            magnitude = float(torch.linalg.vector_norm(force[0]))
            self.assertAlmostEqual(magnitude, cfg.contact_stiffness * ratio * 0.025,
                                   delta=1e-10)
            torch.testing.assert_close(force.sum(dim=0) + reaction,
                                       torch.zeros(2, dtype=torch.float64),
                                       rtol=0, atol=1e-12)

    def test_polyline_midchord_bias_reproduced_by_closed_form(self):
        # Worst case of the secant discretisation: chord penetration is exactly
        # delta - sagitta, so the under-load is -sagitta/delta.
        delta = 0.05 * 0.025
        cfg, sim = self._one_disk(32, delta)
        force, _ = sim.forces()
        measured = float(torch.linalg.vector_norm(force[0]))
        s = sagitta(BOW_RADIUS, BOW_SPAN, 32)
        predicted = cfg.contact_stiffness * max(0.0, delta - s)
        self.assertAlmostEqual(measured, predicted, delta=1e-10)
        self.assertLess((measured - cfg.contact_stiffness * delta)
                        / (cfg.contact_stiffness * delta), -0.20)


class ArcBowIndentationTests(unittest.TestCase):
    """Time-integrated indentation: the arc run *is* the polyline limit."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_arc_equals_fine_polyline_peak_load(self):
        arc = sheet_sim(None)
        fine = sheet_sim(1024)
        peaks = []
        for sim in (arc, fine):
            peak = 0.0
            for _ in range(120):
                peak = max(peak, -float(sim.forces()[1][1]))
                sim.step()
            peaks.append(peak)
        self.assertGreater(peaks[0], 0.0)
        self.assertLess(abs(peaks[0] - peaks[1]) / peaks[0], 5e-3)

    def test_coarse_chords_under_load_the_peak(self):
        arc = sheet_sim(None)
        coarse = sheet_sim(8)
        peaks = []
        for sim in (arc, coarse):
            peak = 0.0
            for _ in range(120):
                peak = max(peak, -float(sim.forces()[1][1]))
                sim.step()
            peaks.append(peak)
        self.assertLess(peaks[1], peaks[0])  # secant bias is one-signed
        self.assertGreater((peaks[0] - peaks[1]) / peaks[0], 1e-3)

    def test_contact_energy_and_balance_on_the_arc_path(self):
        sim = sheet_sim(None)
        for _ in range(60):
            sim.step()
        force, reaction = sim.forces()
        # Bond/pair and hull contact forces are equal-and-opposite; the drag
        # term is external to the ice+hull system and has no reaction partner.
        drag = -sim.config.drag * sim.velocities.sum(dim=0)
        torch.testing.assert_close(force.sum(dim=0) - drag + reaction,
                                   torch.zeros(2, dtype=torch.float64),
                                   rtol=0, atol=1e-10)
        # Exact arc distance -> closed-form contact potential.
        center = torch.tensor(BOW_CENTER, dtype=torch.float64) + sim.tool_position
        radial = torch.linalg.vector_norm(sim.positions - center, dim=1)
        distance = (radial - BOW_RADIUS).abs()
        overlap = (sim.config.radius + sim.config.tool_radius - distance).clamp_min(0.0)
        expected = 0.5 * sim.config.contact_stiffness * overlap.square().sum()
        self.assertAlmostEqual(sim.mechanical_energy()["tool_contact_J"],
                               float(expected), places=8)


class ArcBowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_restart_round_trip_preserves_arc_bow(self):
        sim = sheet_sim(None)
        for _ in range(30):
            sim.step()
        state = sim.snapshot()
        self.assertEqual(state["hull_arc"]["radius"], BOW_RADIUS)
        restored = IceDEM.from_snapshot(state)
        self.assertEqual(restored.hull_arc, sim.hull_arc)
        torch.testing.assert_close(restored.positions, sim.positions, rtol=0, atol=0)
        torch.testing.assert_close(restored.forces()[0], sim.forces()[0], rtol=0, atol=1e-12)
        for _ in range(10):
            sim.step()
            restored.step()
        torch.testing.assert_close(restored.positions, sim.positions, rtol=0, atol=1e-12)

    def test_legacy_snapshot_without_arc_field_restores(self):
        sim = sheet_sim(64)
        state = sim.snapshot()
        del state["hull_arc"]
        restored = IceDEM.from_snapshot(state)
        self.assertIsNone(restored.hull_arc)
        torch.testing.assert_close(restored.hull_profile, sim.hull_profile,
                                   rtol=0, atol=0)

    def test_unknown_restart_field_still_rejected(self):
        state = sheet_sim(None).snapshot()
        state["bogus_field"] = 1
        with self.assertRaisesRegex(ValueError, "fields missing or unknown"):
            IceDEM.from_snapshot(state)

    def test_profile_and_arc_are_mutually_exclusive(self):
        cfg = DEMConfig(nx=5, ny=3)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            IceDEM(cfg, hull_profile=bow_profile(8), hull_arc=bow_arc())

    def test_arc_type_is_checked(self):
        cfg = DEMConfig(nx=5, ny=3)
        with self.assertRaises(TypeError):
            IceDEM(cfg, hull_arc=1.0)


if __name__ == "__main__":
    unittest.main()