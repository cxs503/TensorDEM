"""Regression tests for two contact-kernel defects found by CI clearance (m4637/m4696).

Both defects were *crashes*, not accuracy drifts, and both were invisible to the
per-feature suites until the full suite ran them together:

1. ``IceDEM3D._alive_bond_for_pairs`` gathered into an empty bond-key array.
   A subsystem with **no bonds at all** (``pairs`` empty, e.g. two unbonded
   particles in overlap, which is exactly how the energy-ledger closed-form
   cases are built) leaves ``_bond_keys`` of length zero; the guard
   ``safe = where.clamp(max=max(len(self._bond_keys) - 1, 0))`` then indexed
   element 0 of an empty array -> ``IndexError``.  The correct answer for an
   empty bond set is "none of these pairs is a live bond", i.e. an all-``False``
   mask of the pair-shaped length.

2. ``DEM3DConfig.__post_init__`` demanded ``bond_stiffness > 0`` and
   ``contact_stiffness > 0`` unconditionally.  Under ``contact_model="hertz"``
   neither enters a force expression -- the secant stiffness comes from
   ``young_modulus``/``poisson_ratio`` via :func:`hertz_secant_stiffness` -- so
   ``0.0`` there is the documented "take it from the Hertz model" sentinel, not
   a missing input.  The strict-positivity rule must stay in force on the linear
   path, which really does multiply by both fields.

The tests below pin the fixed behaviour and the two neighbouring behaviours that
a too-broad fix would break: a bonded pair must still map to ``True``, and a
linear configuration must still reject zero stiffness.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
    hertz_secant_stiffness,
)

RADIUS = 0.025
DENSITY = 917.0


def _config(**overrides) -> DEM3DConfig:
    params: dict = dict(
        nx=2,
        ny=2,
        nz=2,
        radius=RADIUS,
        density=DENSITY,
        drag=0.0,
        contact_damping=0.0,
        tool_speed=1e-3,
        tool_gap=1.0,
        fix_x_edges=False,
    )
    params.update(overrides)
    return DEM3DConfig(**params)


def _unbonded_pair_model(**overrides) -> IceDEM3D:
    """Two unbonded particles in sustained overlap, i.e. a bond-free subsystem.

    This reproduces the state the analytic energy-ledger cases install through
    ``scripts.verify_dem3d_energy_ledger_analytic.build_subsystem`` (production
    ``step``/``forces`` path, state arrays overwritten, no bonds).
    """
    model = IceDEM3D(_config(dt=1e-5, **overrides))
    overlap = 2.0e-3
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0 * RADIUS - overlap, 0.0, 0.0]],
                             dtype=model.dtype)
    n = positions.shape[0]
    model.positions = positions.clone()
    model.initial_positions = positions.clone()
    model.velocities = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=model.dtype)
    model.fixed = torch.zeros(n, dtype=torch.bool)
    empty_pairs = torch.zeros((0, 2), dtype=torch.long)
    model.pairs = empty_pairs
    model.bond_indices = empty_pairs
    model.bonded = torch.zeros((0,), dtype=torch.bool)
    model.alive = torch.zeros((0,), dtype=torch.bool)
    model.rest_lengths = torch.zeros((0,), dtype=model.dtype)
    model.reference_bond_vectors = torch.zeros((0, 3), dtype=model.dtype)
    model.initial_normals = torch.zeros((0, 3), dtype=model.dtype)
    model.failure_mode = torch.zeros((0,), dtype=torch.uint8)
    model.failure_time_s = torch.zeros((0,), dtype=model.dtype)
    model.failure_extension_m = torch.zeros((0,), dtype=model.dtype)
    model.failure_energy_J = torch.zeros((0,), dtype=model.dtype)
    model._bond_keys = empty_pairs[:, 0] * n + empty_pairs[:, 1]
    model.time = 0.0
    model.step_count = 0
    return model


class EmptyBondKeyTests(unittest.TestCase):
    """``_alive_bond_for_pairs`` on a lattice without a single bond key."""

    def test_empty_bond_keys_yield_all_false_mask(self) -> None:
        model = _unbonded_pair_model()
        self.assertEqual(len(model._bond_keys), 0)

        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        mask = model._alive_bond_for_pairs(pairs)
        self.assertEqual(tuple(mask.shape), (1,))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(bool(mask.any()), "an empty bond set cannot mark a live bond")

    def test_empty_bond_keys_mask_length_follows_the_query(self) -> None:
        model = _unbonded_pair_model()
        pairs = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.long)
        mask = model._alive_bond_for_pairs(pairs)
        self.assertEqual(tuple(mask.shape), (3,))
        self.assertFalse(bool(mask.any()))

    def test_unbonded_pair_subsystem_steps_and_repels(self) -> None:
        """The pre-fix crash fired from ``step()``; the overlap must now repel."""
        model = _unbonded_pair_model()
        start = float(model.positions[1, 0])
        for _ in range(50):
            model.step()
        self.assertTrue(torch.isfinite(model.positions).all())
        # The pair is in overlap, so the normal contact must separate the
        # particles along x (no bonds exist, so nothing else acts).
        self.assertGreater(float(model.positions[1, 0]), start)

    def test_live_bond_still_maps_to_true(self) -> None:
        """The fix must not turn the non-empty path into a no-op."""
        model = IceDEM3D(_config(dt=1e-5))
        self.assertGreater(len(model._bond_keys), 0)
        pairs = torch.tensor([model.pairs[0].tolist()], dtype=torch.long)
        mask = model._alive_bond_for_pairs(pairs)
        self.assertEqual(tuple(mask.shape), (1,))
        self.assertTrue(bool(mask.all()), "a bonded, alive pair must be reported as such")

        model.alive = torch.zeros_like(model.alive)
        self.assertFalse(bool(model._alive_bond_for_pairs(pairs).any()),
                         "a broken bond must not be reported as alive")


class HertzZeroStiffnessSentinelTests(unittest.TestCase):
    """``bond_stiffness``/``contact_stiffness`` == 0.0 under ``contact_model``."""

    def test_linear_model_still_rejects_zero_bond_stiffness(self) -> None:
        for field in ("bond_stiffness", "contact_stiffness"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    _config(contact_model="linear", **{field: 0.0})

    def test_hertz_model_accepts_zero_stiffness_sentinels(self) -> None:
        cfg = _config(contact_model="hertz", young_modulus=1e9, poisson_ratio=0.3,
                      bond_stiffness=0.0, contact_stiffness=0.0)
        self.assertEqual(cfg.bond_stiffness, 0.0)
        self.assertEqual(cfg.contact_stiffness, 0.0)
        # The Hertz branch supplies the stiffness itself.
        e_star = hertz_reduced_modulus(cfg.young_modulus, cfg.poisson_ratio)
        r_eff = hertz_effective_radius(cfg.radius)
        delta_ref = cfg.hertz_stability_overlap_fraction * cfg.radius
        expected = float(hertz_secant_stiffness(delta_ref, e_star, r_eff))
        self.assertGreater(expected, 0.0)
        self.assertAlmostEqual(cfg._bound_stiffness(0.0, rigid_counterpart=False),
                               expected, places=9)
        self.assertTrue(math.isfinite(cfg.recommended_dt) and cfg.recommended_dt > 0.0)

    def test_negative_stiffness_rejected_in_both_models(self) -> None:
        for model_name, extra in (("linear", {}),
                                  ("hertz", {"young_modulus": 1e9, "poisson_ratio": 0.3})):
            for field in ("bond_stiffness", "contact_stiffness"):
                with self.subTest(model=model_name, field=field):
                    with self.assertRaises(ValueError):
                        _config(contact_model=model_name, **{field: -1.0}, **extra)

    def test_hertz_zero_sentinels_do_not_change_the_linear_bound(self) -> None:
        """The linear model's stiffness bound must come from the given value."""
        cfg = _config(contact_model="linear", bond_stiffness=5.0e4, contact_stiffness=5.0e4)
        self.assertEqual(cfg._bound_stiffness(5.0e4, rigid_counterpart=False), 5.0e4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()