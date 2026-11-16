"""Tests for the implementation-equivalence certificate (m23050).

The certificate decides whether two source digests of a simulation module may be
compared numerically without rerunning.  A decision procedure that always says
"equivalent" would be worse than useless, so these tests exercise both
directions: inert edits must pass and behaviour-changing edits must be caught.
Everything here is source-text analysis -- no simulation is run.
"""
from __future__ import annotations

import ast

import pytest

from tensordem import prov_equivalence as pe


BASE = '''
import math

TOL = 1e-9


def helper(x):
    """docstring that may change"""
    return x * x


def run_deposition(spec):
    """entry point"""
    total = helper(spec)
    for i in range(3):
        total = total + i
    return total


def _unused(x):
    return x - 1
'''


def test_identical_source_is_identical():
    cmp_ = pe.certificate(BASE, BASE, ["run_deposition"])
    assert cmp_.verdict == pe.IDENTICAL_SOURCE
    assert cmp_.equivalent


def test_docstring_only_edit_is_inert_but_not_identical_text():
    edited = BASE.replace('"""docstring that may change"""',
                          '"""an entirely different description"""')
    cmp_ = pe.certificate(BASE, edited, ["run_deposition"])
    assert cmp_.text_sha_old != cmp_.text_sha_new
    assert cmp_.verdict == pe.EQUIVALENT_PATH
    assert cmp_.equivalent


def test_appended_helper_outside_path_is_equivalent():
    appended = BASE + '''

def postprocess(points):
    """appended after the integrator"""
    return [p * 2 for p in points]
'''
    cmp_ = pe.certificate(BASE, appended, ["run_deposition"])
    assert cmp_.verdict == pe.EQUIVALENT_PATH
    assert cmp_.equivalent
    assert any(q.startswith("postprocess") for q in cmp_.added)
    assert cmp_.reachable_added == []


def test_modified_reachable_function_needs_runtime():
    edited = BASE.replace("return x * x", "return x * x * 1.0000001")
    cmp_ = pe.certificate(BASE, edited, ["run_deposition"])
    assert cmp_.verdict == pe.NEEDS_RUNTIME
    assert not cmp_.equivalent
    assert cmp_.reachable_modified == ["helper"]


def test_removed_reachable_function_needs_runtime():
    edited = BASE.replace("    total = helper(spec)\n", "    total = spec\n")
    cmp_ = pe.certificate(BASE, edited, ["run_deposition"])
    # helper is now unreachable from the root, so the reachable set shrank:
    # that is not a silent pass -- the verdict must demand a rerun
    assert not cmp_.equivalent
    assert cmp_.verdict in (pe.NEEDS_RUNTIME,)


def test_module_constant_change_is_shared_state():
    edited = BASE.replace("TOL = 1e-9", "TOL = 1e-6")
    edited = edited.replace("total = helper(spec)", "total = helper(spec) + TOL")
    cmp_ = pe.certificate(BASE, edited, ["run_deposition"])
    # the body changed too, so this is a rerun case; the shared-state verdict is
    # reserved for edits that touch only module level
    assert cmp_.verdict == pe.NEEDS_RUNTIME


def test_module_level_only_change_is_equiv_shared_state():
    edited = BASE.replace("TOL = 1e-9", "TOL = 1e-6")  # TOL is never read on path
    cmp_ = pe.certificate(BASE, edited, ["run_deposition"])
    assert cmp_.verdict == pe.EQUIVALENT_PATH_SHARED_STATE
    assert cmp_.equivalent
    assert cmp_.module_level_changed


def test_missing_source_is_not_a_pass():
    cmp_ = pe.certificate(None, BASE, ["run_deposition"])
    assert cmp_.verdict == pe.MISSING_SOURCE
    assert not cmp_.equivalent


def test_unparsed_source_is_not_a_pass():
    broken = "def run_deposition(:\n    pass\n"
    cmp_ = pe.certificate(broken, BASE, ["run_deposition"])
    assert cmp_.verdict == pe.UNPARSED
    assert not cmp_.equivalent


def test_missing_root_is_unparsed_not_silent_pass():
    cmp_ = pe.certificate(BASE, BASE, ["does_not_exist"])
    assert cmp_.verdict == pe.UNPARSED
    assert not cmp_.equivalent


def test_nonlocal_construct_parses():
    """Per-fragment compilation would fail here; the module-level compile must not."""
    src = '''
def run_deposition(spec):
    acc = 0.0

    def inner():
        nonlocal acc
        acc += 1.0

    inner()
    return acc
'''
    cmp_ = pe.certificate(src, src, ["run_deposition"])
    assert cmp_.verdict == pe.IDENTICAL_SOURCE


def test_reachability_follows_method_calls():
    src = '''
class Cell:
    def update(self, dt):
        return self.step(dt)

    def step(self, dt):
        return dt * 2.0


def run_deposition(spec):
    return Cell().update(spec)
'''
    reachable = pe.reachable_from(src, ["run_deposition"])
    assert "Cell.update" in reachable
    assert "Cell.step" in reachable


def test_reachability_is_transitive():
    reachable = pe.reachable_from(BASE, ["run_deposition"])
    assert "helper" in reachable
    assert "_unused" not in reachable


def test_local_imports_resolve_relative_forms():
    src = '''
from .granular3d_core import damping_ratio, Material3D
from ..other.mod import thing
import numpy as np
import tensordem.extra as extra
'''
    got = pe.local_imports(src, "tensordem.deposition_packing".rsplit(".", 1)[0])
    assert "tensordem.granular3d_core" in got
    assert "damping_ratio" in got["tensordem.granular3d_core"]
    # module tensordem.deposition_packing: ".." climbs out of the tensordem
    # package, so the resolved name is the top-level "other.mod"
    assert "other.mod" in got
    assert "numpy" not in got


def test_dependency_closure_follows_edges():
    sources = {
        "tensordem.a": "from .b import f\n",
        "tensordem.b": "from .c import g\n",
        "tensordem.c": "pass\n",
        "tensordem.unused": "x = 1\n",
    }
    closure = pe.dependency_closure(sources, ["tensordem.a"], "tensordem")
    assert set(closure) == {"tensordem.a", "tensordem.b", "tensordem.c"}


def test_code_fingerprint_ignores_layout():
    moved = "\n\n\n" + BASE
    assert pe.code_fingerprint(moved) == pe.code_fingerprint(BASE)


def test_code_fingerprint_tracks_behaviour():
    edited = BASE.replace("total = total + i", "total = total - i")
    assert pe.code_fingerprint(edited) != pe.code_fingerprint(BASE)


def test_function_code_map_covers_nested_scopes():
    fmap = pe.function_code_map(BASE)
    assert "run_deposition" in fmap and "helper" in fmap
    src = "def run_deposition(s):\n    return [x for x in range(3)]\n"
    fmap2 = pe.function_code_map(src)
    assert any("<listcomp>" in q for q in fmap2)


def test_real_module_history_is_equivalent():
    """The actual certificate claim, asserted on the repository's own history.

    Skipped when git is unavailable (e.g. an exported tarball).
    """
    import hashlib
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = "src/tensordem/deposition_packing.py"

    def show(rev):
        res = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=root,
                             capture_output=True, text=True)
        return res.stdout if res.returncode == 0 else None

    new = show("HEAD") or (root / path).read_text()
    for rev in ("b111a050", "e69b7098"):
        old = show(rev)
        if old is None:
            pytest.skip(f"history revision {rev} not available")
        cmp_ = pe.certificate(old, new, ["run_deposition"],
                              module="tensordem.deposition_packing")
        assert cmp_.verdict in pe.EQUIVALENT_VERDICTS, cmp_.notes
        assert cmp_.reachable_modified == []
        assert len(hashlib.sha256(old.encode()).hexdigest()) == 64