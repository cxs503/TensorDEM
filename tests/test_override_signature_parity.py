"""Override-signature guard for the DEM solver hierarchy.

Motivation
----------
``IceDEM3D.forces`` is an overridable solver hook whose keyword contract grew
over time (``update_fracture``, then ``update_history``).  A subclass that
overrides the hook with an older, narrower signature keeps importing fine and
keeps passing its own constructor tests -- it only explodes at run time, on the
first call the *base* class makes through the polymorphic hook::

    TypeError: MovingWedgeBowIceDEM3D.forces() got an unexpected keyword
               argument 'update_history'

That is a hard failure of every monitoring path (:meth:`diagnostics`, energy
audits, invariance probes) in the subclass family, but it is invisible to any
test that never calls ``diagnostics`` on that subclass.  This module pins the
contract structurally instead of case by case:

* every class in the package that overrides one of the polymorphic hook methods
  must accept at least the keyword arguments its nearest base declares,
* with the same kinds and the same defaults.

Adding a keyword to the base hook therefore turns any un-migrated override red
immediately, with a message naming the offending class.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Polymorphic hooks that callers reach through the base-class contract.
HOOKS = ("forces", "diagnostics", "step")


#: Modules that execute a CLI on import; they are not importable by design.
NOT_IMPORTABLE = {"__main__"}


def _package_modules() -> list[str]:
    names: list[str] = []
    for path in sorted((SRC / "tensordem").rglob("*.py")):
        if "__pycache__" in path.parts or path.stem in NOT_IMPORTABLE:
            continue
        rel = path.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        names.append(".".join(parts))
    return names


def _overrides() -> list[tuple[str, str, str]]:
    """(module, class, hook) for every override declared inside the package."""
    found: list[tuple[str, str, str]] = []
    for dotted in _package_modules():
        try:
            module = importlib.import_module(dotted)
        except (Exception, SystemExit):  # pragma: no cover - optional dep on this box
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != dotted:
                continue
            for hook in HOOKS:
                if hook in obj.__dict__ and any(hasattr(b, hook) for b in obj.__mro__[1:]):
                    found.append((dotted, obj.__name__, hook))
    return found


OVERRIDES = _overrides()


def _keyword_contract(func) -> dict[str, tuple[inspect._ParameterKind, object]]:
    """Keyword-reachable parameters of ``func`` mapped to (kind, default)."""
    contract: dict[str, tuple[inspect._ParameterKind, object]] = {}
    for name, param in inspect.signature(func).parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            contract[name] = (param.kind, param.default)
    return contract


def test_guard_actually_finds_overrides() -> None:
    """Negative control: the discovery step must not be vacuous."""
    triples = {(cls, hook) for _, cls, hook in OVERRIDES}
    # The two real drift sites this guard was written for.
    assert ("MovingWedgeBowIceDEM3D", "forces") in triples
    assert ("RudderIceLoadDEM", "forces") in triples
    assert len(triples) >= 4


@pytest.mark.parametrize("module,cls,hook", OVERRIDES, ids=lambda v: str(v))
def test_override_accepts_the_base_hook_keywords(module: str, cls: str, hook: str) -> None:
    klass = getattr(importlib.import_module(module), cls)
    override = inspect.getattr_static(klass, hook)
    base = next(
        inspect.getattr_static(b, hook) for b in klass.__mro__[1:] if hasattr(b, hook)
    )
    base_contract = _keyword_contract(base)
    over_contract = _keyword_contract(override)

    missing = sorted(set(base_contract) - set(over_contract))
    assert not missing, (
        f"{module}.{cls}.{hook} overrides {base.__qualname__} but does not accept "
        f"keyword(s) {missing}; callers using the base contract will raise TypeError."
    )
    for name in sorted(base_contract):
        base_kind, base_default = base_contract[name]
        over_kind, over_default = over_contract[name]
        assert over_kind is base_kind, (
            f"{module}.{cls}.{hook}({name}) changes the parameter kind "
            f"({base_kind} -> {over_kind})."
        )
        assert over_default == base_default, (
            f"{module}.{cls}.{hook}({name}) changes the default "
            f"({base_default!r} -> {over_default!r})."
        )


def test_override_guard_catches_a_narrowed_signature() -> None:
    """Negative control: a narrowed override must fail the comparison above."""

    class Base:
        def forces(self, external_forces=None, *, update_fracture=True, update_history=True):
            return external_forces

    class Drifted(Base):
        def forces(self, external_forces=None, *, update_fracture=True):  # noqa: ARG002
            return external_forces

    base_contract = _keyword_contract(Base.forces)
    over_contract = _keyword_contract(Drifted.forces)
    assert sorted(set(base_contract) - set(over_contract)) == ["update_history"]