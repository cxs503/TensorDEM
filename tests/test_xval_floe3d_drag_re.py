"""Tests for XVAL-DEM-REUSE-POLYHEDRON-FLOE3D-PLUS-DRAG-AND-RE validation."""
from __future__ import annotations

import json
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_PATH = os.path.join(ROOT, "artifacts", "xval_floe3d_drag_re", "results.json")


@pytest.fixture(scope="module")
def results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def test_drag_decay_passes(results):
    r = results["A_drag_decay"]
    assert r["passed"], f"Drag decay failed: rel_error={r['rel_error']:.4e}"
    assert r["rel_error"] < 0.03


def test_restitution_passes(results):
    r = results["B_restitution"]
    assert r["passed"], f"Restitution failed: rel_error={r['rel_error']:.4e}"
    assert r["rel_error"] < 0.10


def test_buoyancy_equilibrium_passes(results):
    r = results["C_buoyancy_equilibrium"]
    assert r["passed"], f"Buoyancy failed: rel_error={r['rel_error']:.4e}"
    assert r["rel_error"] < 0.05


def test_combined_drag_restitution_passes(results):
    r = results["D_combined_drag_restitution"]
    assert r["passed"], f"Combined case failed: rel_error_e={r['rel_error_e']:.4e}"
    assert r["rel_error_e"] < 0.15


def test_convergence(results):
    r = results["E_convergence"]
    assert r["converging"], f"Drag decay not converging: errors={r['errors']}"


def test_nc_zero_drag(results):
    r = results["NC_zero_drag"]
    assert r["passed"], f"Zero drag NC failed: rel_error={r['rel_error']:.4e}"


def test_nc_zero_damping(results):
    r = results["NC_zero_damping"]
    assert r["passed"], f"Zero damping NC failed: e={r['e_measured']:.4f}"


def test_all_pass(results):
    assert results["summary"]["all_pass"], "Not all validation cases passed"
