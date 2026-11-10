"""Tests for rate-dependent ice-steel friction model and FEM→DEM failure_faces handover.

Covers:
- Rate-dependent friction μ(v) model (Lishman et al. 2011)
- Parameter fitting
- Literature validation
- Negative controls (constant μ degeneration)
- Energy dissipation
- failure_faces.json contract creation/validation
- Geometric utilities (triangle area, normal, intersection)
- Bond breaking at failure faces
- Energy conservation verification
- Geometric accuracy verification
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensordem.ice_rate_dependent_friction import (
    LISHMAN_DEFAULT_PARAMS,
    RateDependentFrictionModel,
    RateDependentFrictionParams,
    compute_friction_coefficient,
    compute_friction_force,
    energy_dissipation_rate,
    export_params_json,
    fit_rate_dependent_params,
    friction_curve_data,
    import_params_json,
    negative_control_constant_mu,
    validate_against_literature,
)

from tensordem.fem_dem_failure_handover import (
    ContractMetadata,
    CrackTip,
    FailureFace,
    FailureFacesContract,
    FEM_DEM_HANDOVER_VERSION,
    angle_between_normals,
    break_bonds_at_faces,
    compute_bond_break_energy,
    create_failure_faces_contract,
    export_contract_json,
    face_area_triangle,
    find_intersected_bonds,
    import_contract_json,
    point_in_triangle,
    segment_triangle_intersection,
    triangle_normal,
    validate_contract,
    verify_energy_conservation,
    verify_geometric_accuracy,
)


# ============================================================================
# Rate-dependent friction tests
# ============================================================================

class TestRateDependentFrictionParams:
    """Tests for RateDependentFrictionParams dataclass."""

    def test_default_params_valid(self):
        """Default parameters should be valid."""
        p = LISHMAN_DEFAULT_PARAMS
        assert p.mu_0 > 0
        assert p.mu_inf >= 0
        assert p.mu_inf < p.mu_0
        assert p.v_c > 0

    def test_custom_params(self):
        """Custom parameters should work."""
        p = RateDependentFrictionParams(mu_0=0.3, mu_inf=0.1, v_c=0.05)
        assert p.mu_0 == 0.3
        assert p.mu_inf == 0.1
        assert p.v_c == 0.05

    def test_invalid_mu_0_negative(self):
        """Negative mu_0 should raise."""
        with pytest.raises(ValueError, match="mu_0 must be positive"):
            RateDependentFrictionParams(mu_0=-0.1, mu_inf=0.05, v_c=0.1)

    def test_invalid_mu_inf_negative(self):
        """Negative mu_inf should raise."""
        with pytest.raises(ValueError, match="mu_inf must be non-negative"):
            RateDependentFrictionParams(mu_0=0.2, mu_inf=-0.01, v_c=0.1)

    def test_invalid_mu_inf_ge_mu_0(self):
        """mu_inf >= mu_0 should raise."""
        with pytest.raises(ValueError, match="mu_inf.*must be less than mu_0"):
            RateDependentFrictionParams(mu_0=0.1, mu_inf=0.1, v_c=0.1)

    def test_invalid_v_c(self):
        """Non-positive v_c should raise."""
        with pytest.raises(ValueError, match="v_c must be positive"):
            RateDependentFrictionParams(mu_0=0.2, mu_inf=0.05, v_c=0)

    def test_frozen(self):
        """Params should be frozen (immutable)."""
        p = LISHMAN_DEFAULT_PARAMS
        with pytest.raises(AttributeError):
            p.mu_0 = 0.5


class TestComputeFrictionCoefficient:
    """Tests for compute_friction_coefficient function."""

    def test_zero_velocity_returns_mu_0(self):
        """v=0 should return mu_0 (static friction)."""
        mu = compute_friction_coefficient(0.0)
        assert mu == pytest.approx(LISHMAN_DEFAULT_PARAMS.mu_0, abs=1e-10)

    def test_very_high_velocity_returns_mu_inf(self):
        """v >> v_c should approach mu_inf."""
        mu = compute_friction_coefficient(100.0)
        assert mu == pytest.approx(LISHMAN_DEFAULT_PARAMS.mu_inf, abs=1e-3)

    def test_characteristic_velocity(self):
        """At v = v_c, mu should be midpoint between mu_0 and mu_inf."""
        p = LISHMAN_DEFAULT_PARAMS
        mu = compute_friction_coefficient(p.v_c)
        expected = p.mu_0 + (p.mu_inf - p.mu_0) / 2.0
        assert mu == pytest.approx(expected, rel=1e-10)

    def test_monotone_decreasing(self):
        """mu(v) should be monotonically decreasing."""
        p = LISHMAN_DEFAULT_PARAMS
        velocities = np.logspace(-4, 0.9, 50)
        mu_values = [compute_friction_coefficient(v, p) for v in velocities]
        for i in range(len(mu_values) - 1):
            assert mu_values[i] >= mu_values[i + 1] - 1e-14

    def test_bounded(self):
        """mu should be bounded in [mu_inf, mu_0]."""
        p = LISHMAN_DEFAULT_PARAMS
        for v in np.logspace(-6, 2, 100):
            mu = compute_friction_coefficient(v, p)
            assert p.mu_inf - 1e-14 <= mu <= p.mu_0 + 1e-14

    def test_negative_velocity_raises(self):
        """Negative velocity should raise."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_friction_coefficient(-0.1)

    def test_formula_correctness(self):
        """Verify the formula μ(v) = μ_0 + (μ_∞ - μ_0)/(1 + v/v_c)."""
        p = RateDependentFrictionParams(mu_0=0.25, mu_inf=0.08, v_c=0.2)
        v = 0.5
        expected = 0.08 + (0.25 - 0.08) / (1.0 + 0.5 / 0.2)
        mu = compute_friction_coefficient(v, p)
        assert mu == pytest.approx(expected, rel=1e-12)


class TestRateDependentFrictionModel:
    """Tests for RateDependentFrictionModel class."""

    def test_callable(self):
        """Model should be callable."""
        model = RateDependentFrictionModel()
        mu = model(0.1)
        assert isinstance(mu, float)
        assert mu > 0

    def test_default_params(self):
        """Default model should use Lishman params."""
        model = RateDependentFrictionModel()
        assert model.params.mu_0 == LISHMAN_DEFAULT_PARAMS.mu_0

    def test_custom_params(self):
        """Custom params should be used."""
        p = RateDependentFrictionParams(mu_0=0.3, mu_inf=0.1, v_c=0.05)
        model = RateDependentFrictionModel(p)
        mu = model(0.05)
        expected = 0.3 + (0.1 - 0.3) / (1.0 + 0.05 / 0.05)
        assert mu == pytest.approx(expected, rel=1e-10)

    def test_repr(self):
        """Repr should be informative."""
        model = RateDependentFrictionModel()
        r = repr(model)
        assert "RateDependentFrictionModel" in r
        assert "mu_0" in r


class TestComputeFrictionForce:
    """Tests for compute_friction_force."""

    def test_basic(self):
        """F_f = mu * N."""
        v = 0.1
        N = 1000.0
        mu = compute_friction_coefficient(v)
        F = compute_friction_force(v, N)
        assert F == pytest.approx(mu * N, rel=1e-10)

    def test_zero_normal_force(self):
        """Zero normal force → zero friction."""
        F = compute_friction_force(0.1, 0.0)
        assert F == 0.0

    def test_negative_normal_force_raises(self):
        """Negative normal force should raise."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_friction_force(0.1, -100.0)


class TestFitRateDependentParams:
    """Tests for parameter fitting."""

    def test_fit_synthetic_data(self):
        """Fit should recover parameters from synthetic data."""
        true_params = RateDependentFrictionParams(mu_0=0.25, mu_inf=0.08, v_c=0.15)
        v_data = np.logspace(-3, 1, 30)
        mu_data = np.array([compute_friction_coefficient(v, true_params) for v in v_data])
        
        fitted = fit_rate_dependent_params(v_data, mu_data)
        
        assert fitted.mu_0 == pytest.approx(true_params.mu_0, rel=0.1)
        assert fitted.mu_inf == pytest.approx(true_params.mu_inf, rel=0.2)
        assert fitted.v_c == pytest.approx(true_params.v_c, rel=0.3)

    def test_fit_requires_3_points(self):
        """Need at least 3 data points."""
        with pytest.raises(ValueError, match="at least 3"):
            fit_rate_dependent_params([0.01, 0.1], [0.2, 0.1])

    def test_fit_length_mismatch(self):
        """Mismatched lengths should raise."""
        with pytest.raises(ValueError, match="same length"):
            fit_rate_dependent_params([0.01, 0.1, 1.0], [0.2, 0.1])


class TestValidateAgainstLiterature:
    """Tests for literature validation."""

    def test_default_params_pass(self):
        """Default params should pass 15% tolerance."""
        result = validate_against_literature()
        assert result["passed"]
        assert result["max_error"] < 0.15

    def test_three_test_points(self):
        """Should have 3 test points."""
        result = validate_against_literature()
        assert len(result["test_points"]) == 3

    def test_tolerance_field(self):
        """Result should include tolerance."""
        result = validate_against_literature()
        assert "tolerance" in result
        assert result["tolerance"] == 0.15


class TestNegativeControl:
    """Tests for negative control (constant μ degeneration)."""

    def test_degenerate_is_constant(self):
        """When mu_0 ≈ mu_inf, model should be nearly constant."""
        result = negative_control_constant_mu(
            v_values=[0.001, 0.01, 0.1, 1.0, 10.0],
            constant_mu=0.10,
        )
        assert result["degenerate_is_constant"]

    def test_realistic_differs_from_constant(self):
        """Realistic params should differ from constant."""
        result = negative_control_constant_mu(
            v_values=[0.001, 0.01, 0.1, 1.0, 10.0],
            constant_mu=0.10,
        )
        assert result["realistic_differs_from_constant"]


class TestEnergyDissipation:
    """Tests for energy dissipation rate."""

    def test_basic(self):
        """P = mu * N * v."""
        v = 0.5
        N = 1000.0
        mu = compute_friction_coefficient(v)
        P = energy_dissipation_rate(v, N)
        assert P == pytest.approx(mu * N * v, rel=1e-10)

    def test_zero_velocity(self):
        """Zero velocity → zero dissipation."""
        P = energy_dissipation_rate(0.0, 1000.0)
        assert P == 0.0


class TestFrictionCurveData:
    """Tests for friction curve data generation."""

    def test_output_shapes(self):
        """Should return arrays of correct shape."""
        v, mu = friction_curve_data(n_points=50)
        assert len(v) == 50
        assert len(mu) == 50

    def test_log_spaced(self):
        """Velocities should be log-spaced."""
        v, mu = friction_curve_data(v_min=0.001, v_max=10.0, n_points=100)
        assert v[0] == pytest.approx(0.001, rel=1e-6)
        assert v[-1] == pytest.approx(10.0, rel=1e-6)


class TestParamsSerialization:
    """Tests for JSON export/import."""

    def test_roundtrip(self):
        """Export then import should preserve params."""
        p = RateDependentFrictionParams(mu_0=0.22, mu_inf=0.06, v_c=0.08)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        
        try:
            export_params_json(p, filepath)
            loaded = import_params_json(filepath)
            
            assert loaded.mu_0 == pytest.approx(p.mu_0)
            assert loaded.mu_inf == pytest.approx(p.mu_inf)
            assert loaded.v_c == pytest.approx(p.v_c)
        finally:
            os.unlink(filepath)

    def test_export_returns_json_string(self):
        """Export without filepath should return JSON string."""
        p = LISHMAN_DEFAULT_PARAMS
        json_str = export_params_json(p)
        data = json.loads(json_str)
        assert "parameters" in data
        assert data["parameters"]["mu_0"] == p.mu_0


# ============================================================================
# FEM→DEM failure_faces contract tests
# ============================================================================

class TestFailureFace:
    """Tests for FailureFace dataclass."""

    def test_basic_creation(self):
        """Should create a valid face."""
        face = FailureFace(
            face_id=0,
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            normal=[0, 0, 1],
            crack_opening_displacement=0.001,
            area=0.5,
            fracture_energy=10.0,
        )
        assert face.face_id == 0
        assert face.area == 0.5

    def test_normal_normalized(self):
        """Normal should be normalized."""
        face = FailureFace(
            face_id=0,
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            normal=[0, 0, 5],  # Not unit
            crack_opening_displacement=0.001,
            area=0.5,
            fracture_energy=10.0,
        )
        assert np.linalg.norm(face.normal) == pytest.approx(1.0, abs=1e-10)

    def test_invalid_vertices_shape(self):
        """Wrong vertex shape should raise."""
        with pytest.raises(ValueError, match="vertices must be"):
            FailureFace(
                face_id=0,
                vertices=[[0, 0], [1, 0], [0, 1]],  # 2D
                normal=[0, 0, 1],
                crack_opening_displacement=0.001,
                area=0.5,
                fracture_energy=10.0,
            )

    def test_negative_cod_raises(self):
        """Negative COD should raise."""
        with pytest.raises(ValueError, match="COD must be non-negative"):
            FailureFace(
                face_id=0,
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                normal=[0, 0, 1],
                crack_opening_displacement=-0.001,
                area=0.5,
                fracture_energy=10.0,
            )


class TestCrackTip:
    """Tests for CrackTip dataclass."""

    def test_basic_creation(self):
        """Should create a valid crack tip."""
        tip = CrackTip(
            position=[1.0, 2.0, 3.0],
            direction=[1.0, 0.0, 0.0],
            stress_intensity_factor=1e6,
        )
        assert tip.position[0] == 1.0
        assert tip.stress_intensity_factor == 1e6

    def test_direction_normalized(self):
        """Direction should be normalized."""
        tip = CrackTip(
            position=[0, 0, 0],
            direction=[3.0, 4.0, 0.0],
            stress_intensity_factor=1e6,
        )
        assert np.linalg.norm(tip.direction) == pytest.approx(1.0, abs=1e-10)


class TestContractMetadata:
    """Tests for ContractMetadata."""

    def test_auto_timestamp(self):
        """Should auto-generate timestamp."""
        meta = ContractMetadata()
        assert len(meta.timestamp) > 0

    def test_invalid_mesh_size(self):
        """Non-positive mesh_size should raise."""
        with pytest.raises(ValueError, match="mesh_size must be positive"):
            ContractMetadata(mesh_size=0)


class TestFailureFacesContract:
    """Tests for FailureFacesContract."""

    def test_to_dict_roundtrip(self):
        """to_dict → from_dict should preserve data."""
        faces = [
            FailureFace(
                face_id=0,
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                normal=[0, 0, 1],
                crack_opening_displacement=0.001,
                area=0.5,
                fracture_energy=10.0,
            )
        ]
        tips = [
            CrackTip(
                position=[0.5, 0.5, 0],
                direction=[1, 0, 0],
                stress_intensity_factor=1e6,
            )
        ]
        contract = FailureFacesContract(
            metadata=ContractMetadata(mesh_size=0.1, total_fracture_energy=10.0),
            failure_faces=faces,
            crack_tips=tips,
        )
        
        data = contract.to_dict()
        loaded = FailureFacesContract.from_dict(data)
        
        assert len(loaded.failure_faces) == 1
        assert len(loaded.crack_tips) == 1
        assert loaded.metadata.mesh_size == 0.1


class TestCreateContract:
    """Tests for create_failure_faces_contract."""

    def test_basic(self):
        """Should create contract from face data."""
        faces = [
            {
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "normal": [0, 0, 1],
                "crack_opening_displacement": 0.001,
                "fracture_energy": 10.0,
            }
        ]
        contract = create_failure_faces_contract(faces, mesh_size=0.1)
        
        assert len(contract.failure_faces) == 1
        assert contract.metadata.mesh_size == 0.1
        assert contract.metadata.total_fracture_energy == 10.0

    def test_auto_area(self):
        """Should auto-compute area if not provided."""
        faces = [
            {
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "crack_opening_displacement": 0.001,
                "fracture_energy": 10.0,
            }
        ]
        contract = create_failure_faces_contract(faces)
        assert contract.failure_faces[0].area == pytest.approx(0.5, rel=1e-6)


class TestValidateContract:
    """Tests for validate_contract."""

    def test_valid_contract(self):
        """Valid contract should pass."""
        faces = [
            FailureFace(
                face_id=0,
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                normal=[0, 0, 1],
                crack_opening_displacement=0.001,
                area=0.5,
                fracture_energy=10.0,
            )
        ]
        contract = FailureFacesContract(
            metadata=ContractMetadata(total_fracture_energy=10.0),
            failure_faces=faces,
        )
        result = validate_contract(contract)
        assert result["valid"]
        assert len(result["errors"]) == 0

    def test_n_faces_counted(self):
        """Should count faces."""
        faces = [
            FailureFace(
                face_id=i,
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                normal=[0, 0, 1],
                crack_opening_displacement=0.001,
                area=0.5,
                fracture_energy=5.0,
            )
            for i in range(3)
        ]
        contract = FailureFacesContract(
            metadata=ContractMetadata(total_fracture_energy=15.0),
            failure_faces=faces,
        )
        result = validate_contract(contract)
        assert result["n_faces"] == 3


class TestContractSerialization:
    """Tests for JSON export/import of contracts."""

    def test_roundtrip(self):
        """Export then import should preserve contract."""
        faces = [
            {
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "normal": [0, 0, 1],
                "crack_opening_displacement": 0.001,
                "fracture_energy": 10.0,
            }
        ]
        contract = create_failure_faces_contract(faces, mesh_size=0.1)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        
        try:
            export_contract_json(contract, filepath)
            loaded = import_contract_json(filepath)
            
            assert len(loaded.failure_faces) == 1
            assert loaded.metadata.mesh_size == 0.1
        finally:
            os.unlink(filepath)


# ============================================================================
# Geometric utility tests
# ============================================================================

class TestFaceAreaTriangle:
    """Tests for face_area_triangle."""

    def test_unit_right_triangle(self):
        """Unit right triangle has area 0.5."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        area = face_area_triangle(vertices)
        assert area == pytest.approx(0.5, rel=1e-10)

    def test_equilateral_triangle(self):
        """Equilateral triangle with side 1 has area sqrt(3)/4."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, math.sqrt(3) / 2, 0],
        ])
        area = face_area_triangle(vertices)
        assert area == pytest.approx(math.sqrt(3) / 4, rel=1e-10)

    def test_3d_triangle(self):
        """3D triangle area."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
        area = face_area_triangle(vertices)
        assert area == pytest.approx(0.5, rel=1e-10)


class TestTriangleNormal:
    """Tests for triangle_normal."""

    def test_xy_plane(self):
        """Triangle in xy-plane has normal in z."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        normal = triangle_normal(vertices)
        assert normal[2] == pytest.approx(1.0, abs=1e-10)
        assert abs(normal[0]) < 1e-10
        assert abs(normal[1]) < 1e-10

    def test_unit_vector(self):
        """Normal should be unit vector."""
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]])
        normal = triangle_normal(vertices)
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-10)


class TestPointInTriangle:
    """Tests for point_in_triangle."""

    def test_center_point(self):
        """Center of triangle should be inside."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        center = np.array([1.0 / 3, 1.0 / 3, 0])
        assert point_in_triangle(center, vertices)

    def test_outside_point(self):
        """Point outside should return False."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        outside = np.array([2.0, 2.0, 0])
        assert not point_in_triangle(outside, vertices)

    def test_vertex(self):
        """Vertex should be inside (boundary)."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert point_in_triangle(np.array([0, 0, 0]), vertices)


class TestSegmentTriangleIntersection:
    """Tests for segment_triangle_intersection."""

    def test_intersecting_segment(self):
        """Segment crossing triangle should intersect."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        p0 = np.array([0.2, 0.2, -1.0])
        p1 = np.array([0.2, 0.2, 1.0])
        
        intersects, point = segment_triangle_intersection(p0, p1, vertices)
        assert intersects
        assert point is not None
        assert point[2] == pytest.approx(0.0, abs=1e-6)

    def test_parallel_segment(self):
        """Segment parallel to triangle should not intersect."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        p0 = np.array([0, 0, 1.0])
        p1 = np.array([1, 0, 1.0])
        
        intersects, point = segment_triangle_intersection(p0, p1, vertices)
        assert not intersects

    def test_missing_segment(self):
        """Segment that misses triangle should not intersect."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        p0 = np.array([5.0, 5.0, -1.0])
        p1 = np.array([5.0, 5.0, 1.0])
        
        intersects, point = segment_triangle_intersection(p0, p1, vertices)
        assert not intersects


class TestAngleBetweenNormals:
    """Tests for angle_between_normals."""

    def test_same_direction(self):
        """Same direction → 0°."""
        angle = angle_between_normals([0, 0, 1], [0, 0, 1])
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_opposite_direction(self):
        """Opposite direction → 180°."""
        angle = angle_between_normals([0, 0, 1], [0, 0, -1])
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_perpendicular(self):
        """Perpendicular → 90°."""
        angle = angle_between_normals([0, 0, 1], [1, 0, 0])
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_45_degrees(self):
        """45° angle."""
        angle = angle_between_normals([0, 0, 1], [1, 0, 1])
        assert angle == pytest.approx(45.0, abs=1e-4)


# ============================================================================
# DEM bond operations tests
# ============================================================================

class TestFindIntersectedBonds:
    """Tests for find_intersected_bonds."""

    def test_intersecting_bond(self):
        """Bond crossing face should be found."""
        bond_positions = np.array([[0.2, 0.2, 0.0]])
        bond_directions = np.array([[0, 0, 1.0]])  # Vertical bond
        
        face = FailureFace(
            face_id=0,
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            normal=[0, 0, 1],
            crack_opening_displacement=0.001,
            area=0.5,
            fracture_energy=10.0,
        )
        
        result = find_intersected_bonds(bond_positions, bond_directions, [face])
        assert 0 in result

    def test_non_intersecting_bond(self):
        """Bond not crossing face should not be found."""
        bond_positions = np.array([[5.0, 5.0, 0.0]])
        bond_directions = np.array([[0, 0, 1.0]])
        
        face = FailureFace(
            face_id=0,
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            normal=[0, 0, 1],
            crack_opening_displacement=0.001,
            area=0.5,
            fracture_energy=10.0,
        )
        
        result = find_intersected_bonds(bond_positions, bond_directions, [face])
        assert len(result) == 0


class TestBreakBondsAtFaces:
    """Tests for break_bonds_at_faces."""

    def test_breaks_intersecting_bonds(self):
        """Should break bonds that intersect faces."""
        bond_status = np.array([1, 1, 1])
        bond_positions = np.array([
            [0.2, 0.2, 0.0],  # Intersects
            [5.0, 5.0, 0.0],  # Does not intersect
            [0.3, 0.3, 0.0],  # Intersects
        ])
        bond_directions = np.array([
            [0, 0, 1.0],
            [0, 0, 1.0],
            [0, 0, 1.0],
        ])
        
        face = FailureFace(
            face_id=0,
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            normal=[0, 0, 1],
            crack_opening_displacement=0.001,
            area=0.5,
            fracture_energy=10.0,
        )
        
        new_status, broken = break_bonds_at_faces(
            bond_status, bond_positions, bond_directions, [face]
        )
        
        assert new_status[0] == 0  # Broken
        assert new_status[1] == 1  # Still intact
        assert new_status[2] == 0  # Broken
        assert 0 in broken
        assert 2 in broken

    def test_already_broken_not_counted(self):
        """Already broken bonds should not be in broken list."""
        bond_status = np.array([0, 1])  # First already broken
        bond_positions = np.array([
            [0.2, 0.2, 0.0],
            [0.3, 0.3, 0.0],
        ])
        bond_directions = np.array([
            [0, 0, 1.0],
            [0, 0, 1.0],
        ])
        
        face = FailureFace(
            face_id=0,
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            normal=[0, 0, 1],
            crack_opening_displacement=0.001,
            area=0.5,
            fracture_energy=10.0,
        )
        
        new_status, broken = break_bonds_at_faces(
            bond_status, bond_positions, bond_directions, [face]
        )
        
        assert 0 not in broken  # Was already broken


class TestComputeBondBreakEnergy:
    """Tests for compute_bond_break_energy."""

    def test_basic(self):
        """E = area * G."""
        E = compute_bond_break_energy(0.01, 100.0)
        assert E == pytest.approx(1.0, rel=1e-10)


class TestVerifyEnergyConservation:
    """Tests for verify_energy_conservation."""

    def test_perfect_conservation(self):
        """Perfect energy match should pass."""
        result = verify_energy_conservation(
            fem_fracture_energy=100.0,
            dem_bond_energies=[30.0, 30.0, 40.0],
            friction_dissipation=0.0,
        )
        assert result["passed"]
        assert result["relative_error"] == pytest.approx(0.0, abs=1e-10)

    def test_with_friction(self):
        """Energy with friction dissipation should balance."""
        result = verify_energy_conservation(
            fem_fracture_energy=100.0,
            dem_bond_energies=[40.0, 40.0],
            friction_dissipation=20.0,
        )
        assert result["passed"]
        assert result["relative_error"] == pytest.approx(0.0, abs=1e-10)

    def test_large_mismatch_fails(self):
        """Large mismatch should fail."""
        result = verify_energy_conservation(
            fem_fracture_energy=100.0,
            dem_bond_energies=[10.0],
            tolerance=0.05,
        )
        assert not result["passed"]
        assert result["relative_error"] > 0.05

    def test_zero_energy(self):
        """Zero FEM energy with zero DEM should pass."""
        result = verify_energy_conservation(
            fem_fracture_energy=0.0,
            dem_bond_energies=[],
        )
        assert result["passed"]


class TestVerifyGeometricAccuracy:
    """Tests for verify_geometric_accuracy."""

    def test_perfect_match(self):
        """Perfect position match should pass."""
        tips = [
            CrackTip(position=[1.0, 2.0, 3.0], direction=[1, 0, 0], stress_intensity_factor=1e6),
        ]
        dem_positions = [np.array([1.0, 2.0, 3.0])]
        
        result = verify_geometric_accuracy(tips, dem_positions, mesh_size=0.1)
        assert result["passed"]
        assert result["max_position_error"] == pytest.approx(0.0, abs=1e-10)

    def test_within_tolerance(self):
        """Position within mesh_size should pass."""
        tips = [
            CrackTip(position=[1.0, 0.0, 0.0], direction=[1, 0, 0], stress_intensity_factor=1e6),
        ]
        dem_positions = [np.array([1.05, 0.0, 0.0])]  # 0.05m off, mesh_size=0.1
        
        result = verify_geometric_accuracy(tips, dem_positions, mesh_size=0.1)
        assert result["position_passed"]

    def test_exceeds_tolerance(self):
        """Position exceeding mesh_size should fail."""
        tips = [
            CrackTip(position=[1.0, 0.0, 0.0], direction=[1, 0, 0], stress_intensity_factor=1e6),
        ]
        dem_positions = [np.array([1.5, 0.0, 0.0])]  # 0.5m off, mesh_size=0.1
        
        result = verify_geometric_accuracy(tips, dem_positions, mesh_size=0.1)
        assert not result["position_passed"]

    def test_normal_angle_check(self):
        """Normal angle check should work."""
        tips = [
            CrackTip(position=[0, 0, 0], direction=[1, 0, 0], stress_intensity_factor=1e6),
        ]
        dem_positions = [np.array([0, 0, 0])]
        fem_normals = [np.array([0, 0, 1])]
        dem_normals = [np.array([0, 0.1, 1])]  # ~5.7° off
        
        result = verify_geometric_accuracy(
            tips, dem_positions, mesh_size=0.1,
            fem_normals=fem_normals, dem_normals=dem_normals,
            angle_tolerance=5.0,
        )
        assert not result["angle_passed"]
        assert result["max_angle_error"] > 5.0

    def test_normal_angle_within_tolerance(self):
        """Normal angle within tolerance should pass."""
        tips = [
            CrackTip(position=[0, 0, 0], direction=[1, 0, 0], stress_intensity_factor=1e6),
        ]
        dem_positions = [np.array([0, 0, 0])]
        fem_normals = [np.array([0, 0, 1])]
        dem_normals = [np.array([0, 0.01, 1])]  # ~0.57° off
        
        result = verify_geometric_accuracy(
            tips, dem_positions, mesh_size=0.1,
            fem_normals=fem_normals, dem_normals=dem_normals,
            angle_tolerance=5.0,
        )
        assert result["angle_passed"]


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests combining friction and failure handover."""

    def test_friction_energy_in_handover(self):
        """Friction dissipation should be trackable in energy balance."""
        # Create a simple scenario
        params = LISHMAN_DEFAULT_PARAMS
        v = 0.1  # m/s
        N = 1000.0  # N
        dt = 1.0  # s
        
        # Friction force and energy
        mu = compute_friction_coefficient(v, params)
        F_f = mu * N
        E_friction = F_f * v * dt
        
        # Energy conservation check
        result = verify_energy_conservation(
            fem_fracture_energy=50.0 + E_friction,
            dem_bond_energies=[50.0],
            friction_dissipation=E_friction,
        )
        assert result["passed"]

    def test_full_handover_workflow(self):
        """Test complete FEM→DEM handover workflow."""
        # 1. Create failure faces (simulating FEM output)
        faces_data = [
            {
                "vertices": [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]],
                "normal": [0, 0, 1],
                "crack_opening_displacement": 0.001,
                "fracture_energy": 5.0,
                "stress_state": {"sigma_n": 1e6, "sigma_t": 0.5e6},
            },
            {
                "vertices": [[0.1, 0, 0], [0.2, 0, 0], [0.1, 0.1, 0]],
                "normal": [0, 0, 1],
                "crack_opening_displacement": 0.002,
                "fracture_energy": 5.0,
                "stress_state": {"sigma_n": 1e6, "sigma_t": 0.5e6},
            },
        ]
        
        crack_tips_data = [
            {
                "position": [0.2, 0.05, 0],
                "direction": [1, 0, 0],
                "stress_intensity_factor": 1.5e6,
            }
        ]
        
        # 2. Create contract
        contract = create_failure_faces_contract(
            faces_data, crack_tips_data, mesh_size=0.1
        )
        
        # 3. Validate contract
        validation = validate_contract(contract)
        assert validation["valid"]
        
        # 4. Export and reimport
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        
        try:
            export_contract_json(contract, filepath)
            loaded = import_contract_json(filepath)
            
            assert len(loaded.failure_faces) == 2
            assert len(loaded.crack_tips) == 1
            assert loaded.metadata.total_fracture_energy == 10.0
        finally:
            os.unlink(filepath)
        
        # 5. Verify energy conservation
        energy_result = verify_energy_conservation(
            fem_fracture_energy=10.0,
            dem_bond_energies=[5.0, 5.0],
        )
        assert energy_result["passed"]
        
        # 6. Verify geometric accuracy
        geo_result = verify_geometric_accuracy(
            fem_crack_tips=loaded.crack_tips,
            dem_crack_positions=[np.array([0.2, 0.05, 0])],
            mesh_size=0.1,
        )
        assert geo_result["passed"]

    def test_no_crack_faces_no_bond_breaking(self):
        """Negative control: no crack faces → no bonds broken."""
        bond_status = np.array([1, 1, 1])
        bond_positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        bond_directions = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float)
        
        new_status, broken = break_bonds_at_faces(
            bond_status, bond_positions, bond_directions, []  # No faces
        )
        
        assert all(new_status == 1)
        assert len(broken) == 0
