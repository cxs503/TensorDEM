"""Contract tests for ``IceDEM3D.forces()`` in solver subclasses.

Background (regression m3147): the base solver evaluates its diagnostic probe as
``self.forces(external_forces, update_fracture=False, update_history=False)``.
Every subclass that overrides ``forces`` must therefore accept *both* keywords;
two overrides (``MovingWedgeBowIceDEM3D``, ``RudderIceLoadDEM``) only accepted
``update_fracture``, so *any* ``diagnostics()`` call on those solvers raised
``TypeError``.  The threefield hull-ice stage (``run_hull_ice_case``) calls
``diagnostics()`` unconditionally, i.e. the merged DEM stage of the coupled
ice/hull/propeller study was dead on the branch tip.

These tests fail loudly if the keyword contract drifts again.
"""
from __future__ import annotations

import inspect

import pytest

# Importing the modules registers the subclasses on ``IceDEM3D.__subclasses__``.
import tensordem.dem3d as dem3d_module
import tensordem.dem3d_energy  # noqa: F401
import tensordem.dem3d_wedge  # noqa: F401
import tensordem.rudder_ice_load  # noqa: F401
import tensordem.threefield.hull_ice  # noqa: F401
from tensordem.dem3d import IceDEM3D
from tensordem.threefield.hull_ice import HullIceCase, run_hull_ice_case

BASE_FORCE_KEYWORDS = ("external_forces", "update_fracture", "update_history")


def _dem3d_subclasses() -> list[type[IceDEM3D]]:
    found: list[type[IceDEM3D]] = []
    stack = [IceDEM3D]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            found.append(sub)
            stack.append(sub)
    return found


def test_subclass_tree_is_visible() -> None:
    names = {cls.__name__ for cls in _dem3d_subclasses()}
    assert {"MovingWedgeBowIceDEM3D", "RudderIceLoadDEM", "BowContactDEM"} <= names


@pytest.mark.parametrize("cls", _dem3d_subclasses(), ids=lambda c: c.__name__)
def test_override_accepts_base_force_keywords(cls: type[IceDEM3D]) -> None:
    """Every ``forces`` override must keep the base keyword contract (Liskov)."""
    params = inspect.signature(cls.forces).parameters
    missing = [name for name in BASE_FORCE_KEYWORDS if name not in params]
    assert not missing, f"{cls.__name__}.forces() drops base keyword(s) {missing}"
    for name in ("update_fracture", "update_history"):
        assert params[name].default is True, f"{cls.__name__}.forces({name}=) default changed"


def test_base_diagnostics_still_uses_both_keywords() -> None:
    """Guard the premise of the contract test: the base probe passes both flags."""
    source = inspect.getsource(IceDEM3D.diagnostics)
    assert "update_fracture=False" in source
    assert "update_history=False" in source


def test_bow_contact_stage_produces_diagnostics() -> None:
    """The threefield hull-ice stage must complete (it raised ``TypeError`` pre-fix)."""
    case = HullIceCase(thickness_m=0.15, particle_radius_m=0.025, n_length=5,
                       n_width=2, steps=40)
    result = run_hull_ice_case(case)
    history = result["history"]
    assert len(history) == case.steps + 1
    for key in ("bow_tip_x_m", "ice_resistance_N", "contact_particles"):
        assert key in history[0], f"missing history field {key}"
    assert history[-1]["bow_tip_x_m"] > history[0]["bow_tip_x_m"]
    assert result["peak_ice_resistance_N"] >= result["mean_ice_resistance_N"] >= 0.0
    assert result["case"]["particle_count"] > 0
    assert result["fragments"]["total_mass_kg"] > 0.0


def test_module_exposes_moving_wedge_class() -> None:
    """The wedge solver is importable from its documented module path."""
    assert hasattr(dem3d_module, "IceDEM3D")