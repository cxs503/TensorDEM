"""Import-surface guard: every ``tensordem.*`` name an importer asks for must exist.

Motivation
----------
A module family that is *moved or renamed* while its importers (library, scripts,
tests) are not migrated turns into a pytest **collection error** rather than a
red test: the failing file dies at import time, so its assertions never run and
the suite silently loses coverage of that whole file.  That failure mode is slow
to bisect because ``git log -- tests/...`` looks clean -- nothing in ``tests/``
changed, only the module path did.

This guard makes the contract explicit and cheap:

* it scans every Python file in the repository for ``import tensordem.x`` and
  ``from tensordem.x import y`` statements (AST, so comments/strings are ignored),
* imports each referenced module and checks that every imported *name* exists
  either as an attribute of that module or as a submodule,
* is parametrised per referencing file, so a future rename reports *which*
  importer was left behind instead of a bare collection error.

It deliberately does not assert anything about behaviour; it is the structural
floor under the behavioural suites.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Directories that contain first-party importers of the package.
SEARCH_ROOTS = ("src", "tests", "scripts", "examples")
SKIP_DIRS = {"__pycache__", ".git", "outputs", "runs", "results-dem3d-scale", ".venv"}

#: Modules that execute a CLI at import time; they cannot be imported by design.
NOT_IMPORTABLE = {"_main__", "__main__"}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for name in SEARCH_ROOTS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            files.append(path)
    return sorted(files)


def _imports_of(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (absolute module imports, (module, imported-name) pairs)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    names: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tensordem" or alias.name.startswith("tensordem."):
                    modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolved by its own package
                continue
            if node.module and (node.module == "tensordem" or node.module.startswith("tensordem.")):
                modules.add(node.module)
                for alias in node.names:
                    if alias.name != "*":
                        names.add((node.module, alias.name))
    return modules, names


def _module_exists(dotted: str) -> bool:
    if dotted.rsplit(".", 1)[-1] in NOT_IMPORTABLE:
        return (SRC / (dotted.replace(".", "/") + ".py")).is_file()
    try:
        importlib.import_module(dotted)
    except ModuleNotFoundError:
        return False
    except (Exception, SystemExit):  # importable but needs a runtime (CLI/GPU)  # pragma: no cover
        return True
    return True


def _name_exists(module: str, name: str) -> bool:
    mod = importlib.import_module(module)
    if hasattr(mod, name):
        return True
    return _module_exists(f"{module}.{name}")


IMPORTERS = [p for p in _python_files() if _imports_of(p)[0] or _imports_of(p)[1]]


@pytest.mark.parametrize("path", IMPORTERS, ids=lambda p: str(p.relative_to(ROOT)))
def test_importer_module_paths_resolve(path: Path) -> None:
    """No importer may reference a ``tensordem`` module that does not exist."""
    modules, _ = _imports_of(path)
    missing = sorted(m for m in modules if not _module_exists(m))
    assert not missing, (
        f"{path.relative_to(ROOT)} imports module(s) that no longer exist: {missing}. "
        "Either restore the module or migrate the importer to the canonical path."
    )


@pytest.mark.parametrize("path", IMPORTERS, ids=lambda p: str(p.relative_to(ROOT)))
def test_importer_imported_names_exist(path: Path) -> None:
    """No importer may import a symbol that its module does not export."""
    _, names = _imports_of(path)
    missing = sorted(f"{mod}:{name}" for mod, name in names if not _name_exists(mod, name))
    assert not missing, (
        f"{path.relative_to(ROOT)} imports name(s) that no longer exist: {missing}. "
        "Either restore the symbol or migrate the importer to the canonical name."
    )


def test_guard_sees_the_package_at_all() -> None:
    """Negative control: the scanner must actually find first-party importers.

    Without this, a bug in the scan (wrong roots, silent exception) would turn
    both guards above into vacuous passes.
    """
    targets = {p.relative_to(ROOT).as_posix() for p in IMPORTERS}
    assert "tests/test_dem3d_lattice_elasticity3d.py" in targets
    assert "tests/test_lattice_elasticity3d_modulus_factor.py" in targets
    assert "src/tensordem/contact_invariants.py" in targets
    assert len(targets) >= 60


def test_guard_reports_a_planted_missing_module(tmp_path: Path) -> None:
    """Negative control: a deliberately stale import must be detected."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from tensordem.lattice_elasticity3d_renamed import HALF_SPACE_OFFSETS\n"
        "import tensordem.moved_family\n",
        encoding="utf-8",
    )
    modules, names = _imports_of(probe)
    assert "tensordem.moved_family" in modules
    assert ("tensordem.lattice_elasticity3d_renamed", "HALF_SPACE_OFFSETS") in names
    assert not _module_exists("tensordem.moved_family")


def test_canonical_lattice_and_shear_families_import() -> None:
    """The families named in the CI triage must import from the canonical path.

    These are exactly the modules whose six test files were reported as failing
    to collect; pinning them by name means a rename cannot pass CI silently.
    """
    from tensordem import born_huang  # noqa: F401
    from tensordem import dem3d_lattice_elastic  # noqa: F401
    from tensordem import lattice_elasticity2d, lattice_elasticity3d  # noqa: F401
    from tensordem import shear_convergence, shear_specimen, shear_static  # noqa: F401

    required = {
        "lattice_elasticity3d": (
            "HALF_SPACE_OFFSETS",
            "bond_stiffness_for_target_modulus",
            "derived_moduli",
            "lattice_bonds",
            "lattice_positions",
            "lattice_spacing",
            "periodic_elastic_constants",
            "relaxed_uniaxial_modulus",
        ),
        "shear_static": (
            "affine_effective_area",
            "bond_census",
            "shear_stiffness_matrix",
            "static_direct_shear",
        ),
        "shear_convergence": ("ACCURACY_GATE", "boundary_layer_fit", "cross_section_series"),
        "shear_specimen": (
            "direct_shear_specimen_area",
            "shear_modulus_closed_form",
        ),
    }
    modules = {
        "lattice_elasticity3d": lattice_elasticity3d,
        "shear_static": shear_static,
        "shear_convergence": shear_convergence,
        "shear_specimen": shear_specimen,
    }
    for module_name, symbols in required.items():
        module = modules[module_name]
        absent = [s for s in symbols if not hasattr(module, s)]
        assert not absent, f"tensordem.{module_name} lost public symbol(s): {absent}"