"""SALVAGE 收编 pytest for ``tensordem.fem_dem_fracture`` (mission m24228).

Three mandated slices, kept deliberately small and fast so the module can be
re-run in CI without the full 40x40 production mesh:

1. **Import & public-API snapshot** -- every name in the package ``__all__``
   (and in each submodule's ``__all__``) must resolve to a real object; the
   snapshot is a *contract* that a refactor cannot silently break.
2. **Small-scale numerical behaviour** -- a ``<= 1e3``-cell mesh (here 20x20 =
   400 cells, well under the limit) is solved against the four closed forms the
   module documents (Navier SSSS, clamped square, cylindrical strip, Hetenyi
   Winkler point load) and the FEM error is bounded; the Hetenyi point-load
   case is shown to *converge* toward the closed form as the mesh refines
   within the 1e3-cell budget.
3. **Degenerate / boundary inputs** -- the validation guards on materials,
   meshes, patches, boundary conditions and the Winkler mechanism are exercised
   so a bad input fails loudly instead of producing a silent wrong number.

The module under test is *not* modified; this file only reads its public surface.
"""
from __future__ import annotations

import numpy as np
import pytest

import tensordem.fem_dem_fracture as fdf
from tensordem.fem_dem_fracture import (
    FOUNDATION,
    GRAVITY,
    WATER_DENSITY,
    IcePlateCase,
    PlateMaterial,
    PlateMesh,
    PressurePatch,
    clamped_square_deflection,
    cylindrical_bending_deflection,
    cylindrical_bending_moment,
    dem_initial_conditions,
    hetenyi_point_load_deflection,
    navier_ssss_square_deflection,
    navier_ssss_square_moment,
    solve_plate,
)
from tensordem.fem_dem_fracture.plate_fem import (
    PlateSolution,
    _db_matrix,
    acm_inverse,
    patch_pressure,
)


# --------------------------------------------------------------------------
# 1. import & public-API snapshot
# --------------------------------------------------------------------------
def test_package_all_names_resolve():
    """Every ``__all__`` name of the package must be a real attribute."""
    missing = [name for name in fdf.__all__ if not hasattr(fdf, name)]
    assert not missing, f"missing from package __all__: {missing}"
    # spot-check the four submodule surfaces are re-exported
    for name in ("PlateMaterial", "PlateMesh", "PlateSolution", "solve_plate",
                 "Crack", "CrackNetwork", "Floe", "trace_crack_network",
                 "ConservationLedger", "FailureFaces", "build_failure_faces",
                 "IcePlateCase", "run_handoff", "mesh_convergence",
                 "VelocityField", "MomentumLedger", "build_dynamic_failure_faces"):
        assert hasattr(fdf, name), f"{name} not re-exported"


def test_plate_fem_all_names_resolve():
    from tensordem.fem_dem_fracture import plate_fem as pf
    missing = [name for name in pf.__all__ if not hasattr(pf, name)]
    assert not missing, f"missing from plate_fem __all__: {missing}"


def test_physical_constants_snapshot():
    """The two physical constants the whole chain is built on."""
    assert WATER_DENSITY == 1025.0
    assert GRAVITY == 9.80665
    # rho_w g = the Winkler foundation modulus used by the ice case
    assert abs(FOUNDATION - WATER_DENSITY * GRAVITY) < 1e-12


def test_default_material_snapshot():
    """First-year sea-ice defaults (Timco & Weeks 2010 band)."""
    mat = PlateMaterial()
    assert mat.young_modulus == 5.0e9
    assert mat.poisson_ratio == 0.3
    assert mat.thickness == 0.5
    assert mat.density == 917.0
    assert mat.sigma_flexure == 0.7e6
    # D = E t^3 / (12 (1 - nu^2))
    D = 5.0e9 * 0.5 ** 3 / (12.0 * (1.0 - 0.3 ** 2))
    assert abs(mat.flexural_rigidity - D) < 1e-9
    # area density = rho t
    assert abs(mat.area_density - 917.0 * 0.5) < 1e-9
    # failure moment = sigma_f t^2 / 6
    assert abs(mat.failure_moment() - 0.7e6 * 0.5 ** 2 / 6.0) < 1e-9


# --------------------------------------------------------------------------
# 2. small-scale numerical behaviour (<= 1e3 cells)
# --------------------------------------------------------------------------
def test_small_mesh_is_under_limit():
    """The chosen small mesh is genuinely <= 1e3 cells (the mandate)."""
    mesh = PlateMesh(nx=20, ny=20, length_x=10.0, length_y=10.0)
    assert mesh.n_cells == 400
    assert mesh.n_cells <= 1000


def test_small_ssss_square_deflection_matches_navier():
    """20x20 (400 cells) simply-supported square vs Navier double series."""
    material = PlateMaterial()
    side, q = 10.0, 1000.0
    mesh = PlateMesh(nx=20, ny=20, length_x=side, length_y=side)
    sol = solve_plate(mesh, material,
                      [PressurePatch((side / 2, side / 2), (side, side), q)],
                      boundary="simply_supported")
    ref = navier_ssss_square_deflection(q, side, material)
    # 20x20 is coarser than the 40x40 acceptance mesh; allow 2% here
    assert abs(sol.max_deflection - ref) / ref < 0.02


def test_small_ssss_square_moment_matches_navier():
    """Centre bending moment vs the 0.0479 q a^2 textbook coefficient."""
    material = PlateMaterial()
    side, q = 10.0, 1000.0
    mesh = PlateMesh(nx=20, ny=20, length_x=side, length_y=side)
    sol = solve_plate(mesh, material,
                      [PressurePatch((side / 2, side / 2), (side, side), q)],
                      boundary="simply_supported")
    centre = (mesh.ny // 2) * mesh.nx + mesh.nx // 2
    ref = navier_ssss_square_moment(q, side, material)
    assert abs(abs(sol.moment_x[centre]) - ref) / ref < 0.02
    assert abs(ref / (q * side ** 2) - 0.0479) < 1e-3


def test_small_clamped_square_deflection():
    """Clamped square vs the 0.00126 q a^4 / D closed form."""
    material = PlateMaterial()
    side, q = 10.0, 1000.0
    mesh = PlateMesh(nx=20, ny=20, length_x=side, length_y=side)
    sol = solve_plate(mesh, material,
                      [PressurePatch((side / 2, side / 2), (side, side), q)],
                      boundary="clamped")
    ref = 0.00126 * q * side ** 4 / material.flexural_rigidity
    assert abs(sol.max_deflection - ref) / ref < 0.05


def test_small_strip_bending_matches_beam_limit():
    """Narrow strip -> beam limit ``w = 5 q a^4 / (384 EI)``."""
    material = PlateMaterial()
    q, span, width = 1000.0, 5.0, 1.0
    ei = material.young_modulus * material.thickness ** 3 / 12.0
    ref = 5.0 * q * span ** 4 / (384.0 * ei)
    mesh = PlateMesh(nx=20, ny=4, length_x=span, length_y=width)
    sol = solve_plate(mesh, material,
                      [PressurePatch((span / 2, width / 2), (span, width), q)],
                      boundary="simply_supported", constrained_edges=("x0", "x1"))
    assert abs(sol.max_deflection - ref) / ref < 0.05


def test_small_winkler_plate_converges_to_hetenyi():
    """Winkler point-load case converges toward Hetenyi as the mesh refines.

    The Hetenyi closed form ``w = P l_c^2 / (8 D)`` is a *point* load on an
    infinite plate, so it needs a fine mesh on a large domain to be resolved.
    Within the 1e3-cell budget we show the error is bounded (< 10 %) and
    *monotonically decreasing* as the mesh refines -- a genuine numerical
    behaviour, not a hard-coded number.
    """
    material = PlateMaterial()
    lc = material.characteristic_length(FOUNDATION)
    domain = 10.0 * lc
    load, side = 1.0e5, 0.5
    ref = hetenyi_point_load_deflection(load, material, FOUNDATION)
    errors = []
    for n in (25, 31):                       # 625 and 961 cells, both <= 1e3
        mesh = PlateMesh(nx=n, ny=n, length_x=domain, length_y=domain)
        sol = solve_plate(mesh, material,
                          [PressurePatch((domain / 2, domain / 2), (side, side),
                                         load / side ** 2)],
                          foundation_modulus=FOUNDATION, boundary="free")
        errors.append(abs(sol.max_deflection - ref) / ref)
    # both bounded, and the finer mesh is closer to the closed form
    assert all(e < 0.10 for e in errors), f"errors {errors} not < 10%"
    assert errors[1] < errors[0], f"not converging: {errors}"


def test_small_solution_field_is_finite_and_symmetric():
    """The small solution is finite, and the SSSS square is 4-fold symmetric."""
    material = PlateMaterial()
    side, q = 10.0, 1000.0
    mesh = PlateMesh(nx=20, ny=20, length_x=side, length_y=side)
    sol = solve_plate(mesh, material,
                      [PressurePatch((side / 2, side / 2), (side, side), q)],
                      boundary="simply_supported")
    assert np.all(np.isfinite(sol.w))
    assert np.all(np.isfinite(sol.sigma_max))
    assert sol.max_sigma > 0.0
    # the deflection field is symmetric about the plate centre (x=5, y=5)
    w = sol.w.reshape(mesh.ny + 1, mesh.nx + 1)
    np.testing.assert_allclose(w, w[::-1, ::-1], rtol=1e-6, atol=1e-9)


def test_small_scale_to_failure_is_exact():
    """``scale_to_failure`` puts max sigma exactly on sigma_flexure."""
    material = PlateMaterial()
    side, q = 10.0, 1000.0
    mesh = PlateMesh(nx=20, ny=20, length_x=side, length_y=side)
    sol = solve_plate(mesh, material,
                      [PressurePatch((side / 2, side / 2), (side, side), q)],
                      boundary="simply_supported")
    amp = sol.scale_to_failure()
    failed = sol.scaled(amp)
    assert abs(failed.max_sigma - material.sigma_flexure) / material.sigma_flexure < 1e-9


# --------------------------------------------------------------------------
# 3. degenerate / boundary inputs
# --------------------------------------------------------------------------
def test_material_rejects_nonpositive_young_modulus():
    with pytest.raises(ValueError, match="young_modulus"):
        PlateMaterial(young_modulus=0.0)
    with pytest.raises(ValueError, match="young_modulus"):
        PlateMaterial(young_modulus=-1.0)


def test_material_rejects_out_of_range_poisson():
    with pytest.raises(ValueError, match="poisson_ratio"):
        PlateMaterial(poisson_ratio=-1.0)
    with pytest.raises(ValueError, match="poisson_ratio"):
        PlateMaterial(poisson_ratio=0.5)


def test_material_rejects_nonpositive_thickness():
    with pytest.raises(ValueError, match="thickness"):
        PlateMaterial(thickness=0.0)


def test_material_rejects_nonpositive_strength():
    with pytest.raises(ValueError, match="sigma_flexure"):
        PlateMaterial(sigma_flexure=0.0)


def test_mesh_rejects_too_few_cells():
    with pytest.raises(ValueError, match="2x2"):
        PlateMesh(nx=1, ny=1)
    with pytest.raises(ValueError, match="2x2"):
        PlateMesh(nx=2, ny=1)


def test_mesh_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        PlateMesh(length_x=0.0)
    with pytest.raises(ValueError, match="dimensions"):
        PlateMesh(length_y=-1.0)


def test_patch_rejects_nonpositive_size():
    """A zero-size patch is rejected when its bounds are queried."""
    with pytest.raises(ValueError, match="size"):
        PressurePatch((0.0, 0.0), (0.0, 1.0), 1000.0).bounds()


def test_characteristic_length_rejects_nonpositive_foundation():
    with pytest.raises(ValueError, match="foundation_modulus"):
        PlateMaterial().characteristic_length(0.0)


def test_solve_rejects_unknown_boundary():
    mesh = PlateMesh(nx=8, ny=8, length_x=5.0, length_y=5.0)
    with pytest.raises(ValueError, match="boundary"):
        solve_plate(mesh, PlateMaterial(),
                    [PressurePatch((2.5, 2.5), (1.0, 1.0), 1000.0)],
                    foundation_modulus=FOUNDATION, boundary="bogus")


def test_solve_rejects_negative_foundation():
    mesh = PlateMesh(nx=8, ny=8, length_x=5.0, length_y=5.0)
    with pytest.raises(ValueError, match="foundation_modulus"):
        solve_plate(mesh, PlateMaterial(),
                    [PressurePatch((2.5, 2.5), (1.0, 1.0), 1000.0)],
                    foundation_modulus=-1.0, boundary="free")


def test_solve_rejects_free_plate_without_foundation():
    """A free plate with no Winkler foundation is a rigid-body mechanism."""
    mesh = PlateMesh(nx=8, ny=8, length_x=5.0, length_y=5.0)
    with pytest.raises(ValueError, match="mechanism"):
        solve_plate(mesh, PlateMaterial(),
                    [PressurePatch((2.5, 2.5), (1.0, 1.0), 1000.0)],
                    foundation_modulus=0.0, boundary="free")


def test_solve_rejects_unknown_constrained_edge():
    mesh = PlateMesh(nx=8, ny=8, length_x=5.0, length_y=5.0)
    with pytest.raises(ValueError, match="edge"):
        solve_plate(mesh, PlateMaterial(),
                    [PressurePatch((2.5, 2.5), (1.0, 1.0), 1000.0)],
                    foundation_modulus=FOUNDATION, boundary="free",
                    constrained_edges=("x0", "diag"))


def test_mesh_edge_mask_rejects_unknown_edge():
    mesh = PlateMesh(nx=8, ny=8)
    with pytest.raises(ValueError, match="edge"):
        mesh.edge_node_mask("diag")


def test_patch_pressure_is_zero_outside_patch():
    """patch_pressure must be exactly 0 outside the loaded rectangle."""
    patch = PressurePatch((2.5, 2.5), (1.0, 1.0), 1000.0)
    assert patch_pressure([patch], 0.0, 0.0) == 0.0
    assert patch_pressure([patch], 5.0, 5.0) == 0.0
    # inside the patch it is the full pressure
    assert patch_pressure([patch], 2.5, 2.5) == 1000.0


def test_acm_inverse_is_invertible():
    """The ACM inverse map must be a valid 12x12 invertible matrix."""
    cinv = acm_inverse(1.0, 1.0)
    assert cinv.shape == (12, 12)
    assert np.all(np.isfinite(cinv))
    # the monomial basis at the four corners is non-singular
    assert abs(np.linalg.det(cinv)) > 1e-6


def test_db_matrix_matches_definition():
    """``Db = D [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]``."""
    mat = PlateMaterial()
    D = mat.flexural_rigidity
    db = _db_matrix(mat)
    expected = D * np.array([[1.0, 0.3, 0.0],
                             [0.3, 1.0, 0.0],
                             [0.0, 0.0, 0.35]])
    np.testing.assert_allclose(db, expected, rtol=1e-12)


def test_small_handoff_roundtrip(tmp_path):
    """A small IcePlateCase runs end-to-end and conserves mass/area."""
    case = IcePlateCase(nx=20, ny=20)
    summary = fdf.run_handoff(case, str(tmp_path))
    crit = summary["criteria"]
    assert crit["C1_fracture_declared"] is True
    assert crit["C2_pass"] is True
    assert crit["C3_pass"] is True
    assert crit["C4_pass"] is True
    assert crit["C5_pass"] is True
    # the DEM bodies rebuild the plate mass exactly
    plate_mass = (case.length_x * case.length_y
                  * case.material.thickness * case.material.density)
    assert abs(crit["C5_roundtrip_mass_kg"] - plate_mass) < 1e-6
