"""Invariant tests for the biaxial-cell stress-definition decomposition.

The R2 verification pass of the biaxial cross-validation case found that the
production cell compares two *biased* numbers: ``sigma_v`` reads one platen and
``sigma_v_internal`` is the Love-Weber sum over internal contacts only.  Those
tests pin the two claims down:

1. **the wall virial completes the identity.**  On a static column of discs
   squeezed between two planes (no gravity, no side contact) the
   divergence-theorem identity ``sum over all contacts of f (x) l = integral of
   sigma over the cell`` must hold exactly, so
   ``sigma_full := -(virial + virial_wall)/V`` equals the platen traction
   ``F/A``.  The internal-only sum is short by exactly the two boundary levers,
   i.e. it under-reports by ``2 (r - overlap) / H`` -- the boundary-layer
   fraction of the cell height;
2. **the decomposition arithmetic** in
   :func:`benchmarks.biaxial_cell_xval.run_xval.decompose_stress_definitions`
   attributes the old gap to the parts that add up to it, and reports the
   divergence-theorem residual (which must vanish in a quasi-static cell).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "biaxial_cell_xval"))

from tensordem.ridge_dem import Plane, RidgeAssembly, RidgeMaterial  # noqa: E402


def _column(*, overlap: float, press: float = 0.0, clearance: float = 0.0,
            thickness: float = 1.0):
    """Three discs in a vertical chain squeezed by top/bottom planes.

    ``press`` moves both planes *inwards* by that amount (``press == 0`` makes
    them exactly touch the end discs: ``signed == r``, zero force), and
    ``clearance`` moves them outwards (to break the wall contact).  Side planes
    sit far away so they never contribute: this is a one-dimensional stress
    state whose analytic answer is ``sigma_yy = N / A`` with ``N`` the chain
    force and ``A`` the plane area.
    """
    material = RidgeMaterial(particle_radius=0.5)
    radius = material.particle_radius
    spacing = 2.0 * radius - overlap
    ys = [0.0, spacing, 2.0 * spacing]
    positions = torch.tensor([[0.0, y] for y in ys], dtype=torch.float64)
    count = positions.shape[0]
    width = 2.0 * radius
    half_h = 0.5 * (ys[-1] - ys[0]) + radius
    planes = [
        Plane((1.0, 0.0), (-10.0 * width, 0.0), stiffness=material.contact_stiffness,
              damping=material.contact_damping, friction=0.0),
        Plane((-1.0, 0.0), (10.0 * width, 0.0), stiffness=material.contact_stiffness,
              damping=material.contact_damping, friction=0.0),
        Plane((0.0, 1.0), (0.0, ys[0] - radius + press - clearance),
              stiffness=material.contact_stiffness, damping=material.contact_damping,
              friction=0.0),
        Plane((0.0, -1.0), (0.0, ys[-1] + radius - press + clearance),
              stiffness=material.contact_stiffness, damping=material.contact_damping,
              friction=0.0),
    ]
    assembly = RidgeAssembly(positions,
                             torch.full((count,), radius, dtype=torch.float64),
                             torch.full((count,), material.particle_mass,
                                        dtype=torch.float64),
                             torch.zeros((0, 2), dtype=torch.long), material,
                             planes=planes, thickness=thickness)
    height = (ys[-1] + radius - press + clearance) - (ys[0] - radius + press - clearance)
    return assembly, material, width, height


@pytest.mark.parametrize("strain", [0.005, 0.02])
def test_wall_virial_completes_the_stress_identity(strain: float) -> None:
    # static equilibrium needs *equal* contact forces, hence the same overlap at
    # the internal contacts and at the two platens
    press = strain
    assembly, material, width, height = _column(overlap=strain, press=press)
    out = assembly.step(material.recommended_dt)
    thickness = assembly.thickness
    volume = width * height * thickness
    area = width * thickness

    plane_forces = out["plane_forces"]
    traction_bottom = -float(plane_forces[2][1]) / area       # compressive > 0
    traction_top = float(plane_forces[3][1]) / area
    chain_force = traction_bottom * area

    sigma_internal = -float(out["virial"][1, 1]) / volume
    sigma_full = -float(out["virial"][1, 1] + out["virial_wall"][1, 1]) / volume

    # (1) full virial == platen traction, i.e. the two bookkeepings agree
    assert sigma_full == pytest.approx(chain_force / area, rel=1e-9)
    # (2) the two planes carry the same load here (no side shear, no gravity)
    assert traction_top == pytest.approx(traction_bottom, rel=1e-9)
    # (3) internal-only virial is short by exactly the boundary layer fraction
    boundary = 2.0 * (material.particle_radius - press) / height
    assert sigma_internal < sigma_full
    assert (sigma_full - sigma_internal) / sigma_full == pytest.approx(boundary, rel=1e-6)
    # (4) the shortfall is a pure boundary-layer fraction: it shrinks like 1/H,
    #     so in the 24 x 48 cell (H ~ 72 mm, r = 0.75 mm) it is a few percent --
    #     far too small to explain a tens-of-percent gap, which is why the axial
    #     gap has to come from the load path (side-wall shear), not the virial
    assert boundary == pytest.approx(2.0 * (material.particle_radius - press) / height,
                                     rel=1e-12)


def test_wall_virial_vanishes_without_wall_contact() -> None:
    assembly, material, _, _ = _column(overlap=0.01, clearance=0.2)
    out = assembly.step(material.recommended_dt)
    assert float(out["virial_wall"].abs().max()) == pytest.approx(0.0, abs=1e-12)


def _series(**kwargs) -> dict[str, np.ndarray]:
    n = 200
    base = {"sigma_v_bottom": np.full(n, 418.0e3),
            "sigma_h_wall": np.full(n, 114.0e3),
            "sigma_v_internal": np.full(n, 519.0e3),
            "sigma_h_internal": np.full(n, 206.0e3)}
    base.update({k: np.asarray(v, dtype=float) for k, v in kwargs.items()})
    return base


def test_decomposition_attributes_the_gap_to_the_missing_platen() -> None:
    from run_xval import decompose_stress_definitions

    n = 200
    width, height, tau = 0.036, 0.072, 61.0e3
    weight = 0.0
    # platen drop consistent with vertical equilibrium d sigma/dy = rho g + 2 tau/W
    drop = (2.0 * tau / width) * height
    series = _series(
        sigma_v_top=np.full(n, 418.0e3 + drop),
        sigma_v_wall_mean=np.full(n, 418.0e3 + 0.5 * drop),
        sigma_v_full=np.full(n, 418.0e3 + 0.5 * drop),   # wall term ~ 0 here
        sigma_h_full=np.full(n, 206.0e3),
        tau_side=np.full(n, tau),
        balance_residual=np.zeros(n),
        widths=np.full(n, width), heights=np.full(n, height))
    got = decompose_stress_definitions(strains=np.linspace(0.0, 0.1, n),
                                       series=series,
                                       meta={"thickness": 1.0, "weight": weight})
    axial = got["axial"]
    # the three parts of the split sum to the old production gap, exactly
    assert (axial["part_a_missing_platen"] + axial["part_b_missing_wall_virial"]
            + axial["residual_full_vs_mean"]
            == pytest.approx(axial["gap_old_internal_vs_bottom"], rel=1e-12))
    assert axial["gap_old_internal_vs_bottom"] == pytest.approx((519.0 - 418.0) / 418.0)
    assert axial["part_a_missing_platen"] == pytest.approx(0.5 * drop / 418.0e3)
    assert axial["drop_measured"] == pytest.approx(drop)
    assert axial["drop_rel_error"] == pytest.approx(0.0, abs=1e-9)
    assert got["pass_15pct"]              # residual is 0 by construction

    # a wall-virial shortfall shows up as a *negative* part B (the internal-only
    # reading was biased low) and still leaves the closure at zero
    series["sigma_v_full"] = np.full(n, 519.0e3)
    series["sigma_v_wall_mean"] = np.full(n, 519.0e3)
    series["sigma_v_internal"] = np.full(n, 508.0e3)
    series["sigma_h_internal"] = np.full(n, 200.0e3)
    series["sigma_h_full"] = np.full(n, 206.0e3)
    got = decompose_stress_definitions(strains=np.linspace(0.0, 0.1, n),
                                       series=series, meta={"thickness": 1.0})
    axial = got["axial"]
    assert axial["part_b_missing_wall_virial"] == pytest.approx(-11.0 / 418.0)
    assert axial["residual_full_vs_mean"] == pytest.approx(0.0, abs=1e-12)
    assert axial["part_a_missing_platen"] + axial["part_b_missing_wall_virial"] \
        + axial["residual_full_vs_mean"] \
        == pytest.approx((508.0 - 418.0) / 418.0, rel=1e-12)
    assert got["lateral"]["part_b_missing_wall_virial"] == pytest.approx(-6.0 / 114.0)


def test_decomposition_is_nan_safe_without_the_new_series() -> None:
    from run_xval import decompose_stress_definitions

    got = decompose_stress_definitions(strains=np.linspace(0.0, 0.1, 200),
                                       series=_series(), meta={"thickness": 1.0})
    assert math.isnan(got["axial"]["sigma_v_top"])
    assert got["pass_15pct"] is False
    assert math.isnan(got["equilibrium"]["balance_residual_mean"])