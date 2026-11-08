"""Curved-hull (faceted cylinder) validation of ``hull_mesh_contact``.

Closes the GAP-DEM-G3 open row "curved-hull validation on a real hull not
re-audited" (``docs/dem_calibration_gap_ledger.md``): the 3-D mesh kernel is
scored against the exact circular-cylinder distance in closed form, the 3-D
analogue of ``docs/polyline_hull_contact_verification.md``.  The audit script
``scripts/m6423_curved_hull_audit.py`` emits every number; these tests pin the
invariants so a future regression is caught by CI.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
_SPEC = importlib.util.spec_from_file_location(
    "m6423_curved_hull_audit", _HERE.parents[1] / "scripts" / "m6423_curved_hull_audit.py")
_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_audit)

R_C, L, R_P, K = 0.5, 2.0, 0.05, 1.0e6


@pytest.fixture(scope="module")
def fine_mesh():
    return _audit.cylinder_mesh(R_C, L, 128, 8)


def _force(center, verts, faces):
    f, _ = _audit.mesh_contact_forces(
        center.unsqueeze(0), torch.zeros(1, 3, dtype=torch.float64),
        verts, faces, torch.zeros(3, dtype=torch.float64),
        particle_radius=R_P, stiffness=K)
    return f[0]


def test_fine_mesh_force_matches_analytic(fine_mesh):
    """Faceted cylinder at n_circ=128: force within the sagitta/overlap bound."""
    verts, faces = fine_mesh
    worst = 0.0
    for az in (0, 47, 133, 270, 311):
        t = math.radians(az)
        for pen in (0.04, 0.02, 0.01):
            rho = R_C - pen
            c = torch.tensor([rho * math.cos(t), rho * math.sin(t), 0.0], dtype=torch.float64)
            f_num = float(_force(c, verts, faces).norm())
            f_ana, _ = _audit.analytic_cylinder_force(c, R_C, L, R_P, K)
            worst = max(worst, abs(f_num - f_ana) / f_ana)
    # sagitta(128) = 1.5e-4 m; worst overlap in the sweep is 10 mm -> 1.5 %
    assert worst < 0.02


def test_direction_within_facet_chord_angle(fine_mesh):
    """Force direction deviates from the true radial by at most the half-chord angle."""
    verts, faces = fine_mesh
    t = math.radians(47.0)
    c = torch.tensor([(R_C - 0.02) * math.cos(t), (R_C - 0.02) * math.sin(t), 0.0],
                     dtype=torch.float64)
    f = _force(c, verts, faces)
    f_ana, n_ana = _audit.analytic_cylinder_force(c, R_C, L, R_P, K)
    cosang = torch.clamp((f / f.norm()) @ n_ana, -1.0, 1.0)
    err_deg = math.degrees(math.acos(cosang))
    assert err_deg <= math.degrees(math.pi / 128) + 1e-9


def test_discretisation_error_converges():
    """Error shrinks with resolution once the sagitta drops below the overlap."""
    def max_err(n_circ):
        verts, faces = _audit.cylinder_mesh(R_C, L, n_circ, 4)
        worst = 0.0
        for az in range(0, 360, 5):
            t = math.radians(az)
            c = torch.tensor([(R_C - 0.01) * math.cos(t), (R_C - 0.01) * math.sin(t), 0.0],
                            dtype=torch.float64)
            f_num = float(_force(c, verts, faces).norm())
            f_ana, _ = _audit.analytic_cylinder_force(c, R_C, L, R_P, K)
            worst = max(worst, abs(f_num - f_ana) / f_ana)
        return worst
    e32, e64, e128 = max_err(32), max_err(64), max_err(128)
    assert e64 < e32
    assert e128 < e64
    assert e128 < 0.02  # inside the 3 % gate at 10 mm overlap


def test_coarse_mesh_error_is_honest_not_silent():
    """At n_circ=16 the sagitta (9.6 mm) is comparable to the 10 mm overlap:
    the error must be *measured*, not hidden -- pin the measured band."""
    verts, faces = _audit.cylinder_mesh(R_C, L, 16, 4)
    t = math.radians(11.25)  # facet midpoint
    c = torch.tensor([(R_C - 0.01) * math.cos(t), (R_C - 0.01) * math.sin(t), 0.0],
                    dtype=torch.float64)
    f_num = float(_force(c, verts, faces).norm())
    f_ana, _ = _audit.analytic_cylinder_force(c, R_C, L, R_P, K)
    rel = (f_num - f_ana) / f_ana
    # the faceted surface sits inside the true circle: force is over-predicted
    assert 0.0 < rel < 1.0


def test_rim_region_matches_analytic():
    """End-rim approach: mesh force within the sagitta bound of the rim circle."""
    verts, faces = _audit.cylinder_mesh(R_C, L, 64, 8)
    z_end = 0.5 * L
    worst = 0.0
    for dz in (-0.02, 0.0, 0.02):
        for dr in (-0.01, 0.0, 0.01):
            c = torch.tensor([(R_C + dr) * math.cos(0.6), (R_C + dr) * math.sin(0.6),
                              z_end + dz], dtype=torch.float64)
            f_num = float(_force(c, verts, faces).norm())
            f_ana, _ = _audit.analytic_cylinder_force(c, R_C, L, R_P, K)
            worst = max(worst, abs(f_num - f_ana) / f_ana)
    assert worst < 0.02


def test_broad_phase_complete_on_curved_mesh(fine_mesh):
    verts, faces = fine_mesh
    res = _audit.broad_phase_completeness(verts, faces, R_C, L, R_P, 100, 7)
    assert res["missing_queries"] == 0


def test_action_reaction_on_curved_mesh(fine_mesh):
    verts, faces = fine_mesh
    c = torch.tensor([[R_C - 0.01, 0.0, 0.0]], dtype=torch.float64)
    f, reaction = _audit.mesh_contact_forces(
        c, torch.zeros(1, 3, dtype=torch.float64), verts, faces,
        torch.zeros(3, dtype=torch.float64), particle_radius=R_P, stiffness=K)
    assert float((reaction + f.sum(dim=0)).norm()) < 1e-9


def test_digitisation_rule_guarantees_sagitta_below_overlap():
    """n_min = ceil(pi / acos(1 - delta/R)) guarantees sagitta < delta."""
    for delta in (0.02, 0.01, 0.005):
        n_min = math.ceil(math.pi / math.acos(1.0 - delta / R_C))
        assert _audit.sagitta(R_C, n_min) < delta
        assert _audit.sagitta(R_C, n_min - 1) >= delta
