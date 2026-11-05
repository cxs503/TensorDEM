"""Regression pin for the ``MovingWedgeBowIceDEM3D.forces`` kwargs contract.

Background (m3296): ``IceDEM3D.forces`` grew a keyword-only ``update_history``
flag so that probes (diagnostics, energy audits, invariance checks) can evaluate
the constitutive forces *without* advancing the Cundall-Strack tangential slip
history.  ``MovingWedgeBowIceDEM3D`` (``tensordem.dem3d_wedge``) overrides
``forces`` and had kept the older two-argument signature, so the subclass
silently narrowed the base-class contract and **every probe raised
``TypeError: MovingWedgeBowIceDEM3D.forces() got an unexpected keyword argument
'update_history'``** — the immediate cause of the red tests in
``tests/test_dem3d_moving_wedge_bow.py``,
``tests/test_dem3d_wedge_resistance_sensitivity.py`` and
``tests/test_threefield.py``.

These tests pin the contract itself rather than any particular number:

* the signature accepts the same keyword-only flags as the base class;
* a probe is side-effect free: two ``update_history=False`` calls return the
  same force field and leave the stored tangential state untouched, so a probe
  can never be silently promoted into a step.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D


def _bow_model(**config_overrides) -> MovingWedgeBowIceDEM3D:
    cfg = dict(
        nx=3, ny=2, nz=3, radius=0.02, tool_speed=0.1,
        bond_stiffness=1000.0, contact_stiffness=3000.0,
        breaking_strain=0.5, shear_breaking_strain=0.5,
        drag=0.1, contact_damping=0.5,
        friction_coefficient=0.4,
        fix_x_edges=False, fix_bottom=False,
    )
    cfg.update(config_overrides)
    return MovingWedgeBowIceDEM3D(
        DEM3DConfig(**cfg), bow_speed=0.1, wedge_angle_deg=45.0,
    )


class TestForcesSignatureContract:
    """The subclass must accept the base-class probe flags."""

    def test_signature_exposes_base_kwargs(self):
        params = inspect.signature(MovingWedgeBowIceDEM3D.forces).parameters
        assert "update_fracture" in params, params
        assert "update_history" in params, params
        assert params["update_history"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["update_fracture"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_probe_call_does_not_raise(self):
        model = _bow_model()
        force, reaction = model.forces(update_history=False)
        assert force.shape == (int(model.positions.shape[0]), 3)
        assert torch.isfinite(force).all()
        assert torch.isfinite(reaction).all()

    def test_stepping_call_still_works(self):
        model = _bow_model()
        force, reaction = model.forces(update_history=True)
        assert torch.isfinite(force).all()
        assert torch.isfinite(reaction).all()
        bow = float(model.diagnostics()["bow_reaction_x_N"])
        assert bow == bow  # not NaN


class TestProbeSideEffectFreedom:
    """``update_history=False`` must be a pure observer of the friction state."""

    def test_repeated_probe_is_idempotent(self):
        model = _bow_model()
        # ``update_fracture=False`` isolates the friction-history channel from
        # bond-topology commits, which are a separate (also one-shot) side effect.
        force_a, reaction_a = model.forces(update_fracture=False, update_history=False)
        force_b, reaction_b = model.forces(update_fracture=False, update_history=False)
        assert torch.equal(force_a, force_b)
        assert torch.equal(reaction_a, reaction_b)

    def test_probe_does_not_advance_slip_history(self):
        model = _bow_model()
        model.forces(update_history=False)
        before = {k: v[0].clone() for k, v in model._tangential_state.items()}
        for _ in range(3):
            model.forces(update_history=False)
        after = {k: v[0].clone() for k, v in model._tangential_state.items()}
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key]), key

    def test_stepping_call_advances_slip_history(self):
        model = _bow_model()
        model.velocities[:, 0] = 0.05  # transverse kick so the tangent channel slips
        model.forces(update_fracture=False, update_history=False)
        before = {k: (v[0].clone(), v[1].clone()) for k, v in model._tangential_state.items()}
        model.forces(update_fracture=False, update_history=True)
        after = model._tangential_state
        moved = [
            key for key, (disp, _) in ((k, v) for k, v in after.items())
            if key in before and not torch.equal(before[key][0], disp)
        ]
        if not before:
            pytest.skip("no active contact pair in this lattice, nothing to advance")
        assert moved, "a stepping call must advance the tangential displacement"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))