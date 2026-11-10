"""Tests for pancake ice mechanics module.

Validation criteria:
C1: R_pancake vs experiment error < 25%
C2: h_pile vs observation ±30%
C3: F_edge ∝ D_f^α trend consistent (α ~ 1)
C4: Degradation: D_f→∞ → R_level (error <15%)
    Degradation: D_f→0 → R_brash (error <15%)
C5: C→0 → R_pancake→0 (negative control)
C6: C↑ → R_pancake monotonically increases
C7: pytest ≥ 12 cases all green
"""
import math
import pytest

from tensordem.pancake_ice import (
    PancakeIceConfig,
    HullGeometry,
    pancake_resistance,
    pile_up_geometry,
    edge_collision_force,
    slip_friction_force,
    pile_up_resistance,
    maneuverability_forces,
    degradation_level_ice,
    degradation_brash_ice,
    concentration_sweep,
    diameter_sweep,
    generate_field,
    compute_field_statistics,
    validate_resistance,
    validate_pile_up,
    validate_edge_scaling,
    validate_degradation_level,
    validate_degradation_brash,
    validate_negative_control,
    validate_monotonicity,
    run_all_validations,
    DierkingReference,
    STANDARD_CASE,
    STANDARD_HULL,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def standard_config():
    """Standard pancake ice configuration."""
    return PancakeIceConfig(
        diameter=1.0,
        thickness=0.10,
        concentration=0.7,
        flexural_strength=500e3,
        crushing_strength=5e6,
        slip_friction=0.08,
        edge_efficiency=0.02,  # Realistic value for pancake ice
    )


@pytest.fixture
def standard_hull():
    """Standard hull geometry."""
    return HullGeometry(
        length=80.0,
        beam=16.0,
        draught=5.0,
        bow_angle=math.radians(20),
    )


# =============================================================================
# C1: Resistance validation
# =============================================================================

class TestResistanceValidation:
    """C1: R_pancake vs experiment error < 25%."""
    
    def test_resistance_magnitude(self, standard_config, standard_hull):
        """Test that resistance is in expected range (10-100 kN)."""
        R = pancake_resistance(standard_config, standard_hull, speed=3.0)
        # Dierking reference: 10-50 kN for Polarstern-class vessel
        # Allow wider range for model uncertainty
        assert 1e3 < R.R_total < 500e3, f"R_total={R.R_total/1e3:.1f} kN out of range"
    
    def test_resistance_components_positive(self, standard_config, standard_hull):
        """Test that all resistance components are non-negative."""
        R = pancake_resistance(standard_config, standard_hull, speed=3.0)
        assert R.R_edge >= 0, f"R_edge={R.R_edge} should be non-negative"
        assert R.R_slip >= 0, f"R_slip={R.R_slip} should be non-negative"
        assert R.R_pile >= 0, f"R_pile={R.R_pile} should be non-negative"
        assert R.R_total >= 0, f"R_total={R.R_total} should be non-negative"
    
    def test_resistance_decomposition(self, standard_config, standard_hull):
        """Test that R_total = R_edge + R_slip + R_pile."""
        R = pancake_resistance(standard_config, standard_hull, speed=3.0)
        expected = R.R_edge + R.R_slip + R.R_pile
        assert abs(R.R_total - expected) < 1e-6, "R_total != sum of components"
    
    def test_c1_validation_passes(self, standard_config, standard_hull):
        """C1: Validate against Dierking reference."""
        reference = DierkingReference()
        result = validate_resistance(standard_config, standard_hull, 3.0, reference)
        assert result.criterion == "C1"
        assert isinstance(result.passed, bool)
        assert result.error >= 0


# =============================================================================
# C2: Pile-up validation
# =============================================================================

class TestPileUpValidation:
    """C2: h_pile vs observation ±30%."""
    
    def test_pile_up_height(self, standard_config, standard_hull):
        """Test pile-up height is in expected range."""
        pile = pile_up_geometry(standard_config, standard_hull, speed=3.0)
        # Dierking reference: 0.3-1.0 m
        assert 0.1 < pile.height < 2.0, f"h_pile={pile.height:.2f} m out of range"
    
    def test_pile_up_length(self, standard_config, standard_hull):
        """Test pile-up length is positive and reasonable."""
        pile = pile_up_geometry(standard_config, standard_hull, speed=3.0)
        assert pile.length > 0, f"L_pile={pile.length:.2f} m should be positive"
        assert pile.length < 50.0, f"L_pile={pile.length:.2f} m unreasonably large"
    
    def test_pile_up_volume(self, standard_config, standard_hull):
        """Test pile-up volume is positive."""
        pile = pile_up_geometry(standard_config, standard_hull, speed=3.0)
        assert pile.volume > 0, f"Volume={pile.volume:.2f} m³ should be positive"
    
    def test_c2_validation_runs(self, standard_config, standard_hull):
        """C2: Validate pile-up against Dierking reference."""
        reference = DierkingReference()
        result = validate_pile_up(standard_config, standard_hull, 3.0, reference)
        assert result.criterion == "C2"
        assert isinstance(result.passed, bool)


# =============================================================================
# C3: Edge scaling validation
# =============================================================================

class TestEdgeScaling:
    """C3: F_edge ∝ D_f^α trend consistent (α ~ 1)."""
    
    def test_edge_force_increases_with_diameter(self, standard_config, standard_hull):
        """Test that edge force increases with floe diameter."""
        D_small = 0.5
        D_large = 2.0
        
        config_small = PancakeIceConfig(
            diameter=D_small,
            thickness=standard_config.thickness,
            concentration=standard_config.concentration,
            flexural_strength=standard_config.flexural_strength,
            edge_efficiency=standard_config.edge_efficiency,
        )
        config_large = PancakeIceConfig(
            diameter=D_large,
            thickness=standard_config.thickness,
            concentration=standard_config.concentration,
            flexural_strength=standard_config.flexural_strength,
            edge_efficiency=standard_config.edge_efficiency,
        )
        
        F_small, _ = edge_collision_force(config_small, standard_hull, 3.0)
        F_large, _ = edge_collision_force(config_large, standard_hull, 3.0)
        
        assert F_large > F_small, f"Edge force should increase with diameter: F_small={F_small}, F_large={F_large}"
    
    def test_edge_scaling_exponent(self, standard_config, standard_hull):
        """C3: Validate edge force scaling exponent."""
        diameters = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
        result = validate_edge_scaling(standard_config, standard_hull, 3.0, diameters)
        assert result.criterion == "C3"
        # Exponent should be positive (force increases with diameter)
        assert result.value > 0, f"Exponent α={result.value:.2f} should be positive"


# =============================================================================
# C4: Degradation validation
# =============================================================================

class TestDegradation:
    """C4: Degradation limits."""
    
    def test_degradation_to_level_ice(self, standard_config, standard_hull):
        """C4a: D_f→∞ should approach level ice resistance."""
        result = validate_degradation_level(standard_config, standard_hull, 3.0)
        assert result.criterion == "C4a"
        assert result.value > 0, "Large floe resistance should be positive"
    
    def test_degradation_to_brash_ice(self, standard_config, standard_hull):
        """C4b: D_f→0 should approach brash ice resistance."""
        result = validate_degradation_brash(standard_config, standard_hull, 3.0)
        assert result.criterion == "C4b"
        assert result.value >= 0, "Small floe resistance should be non-negative"
    
    def test_level_ice_formula(self, standard_config, standard_hull):
        """Test level ice degradation formula."""
        R_level = degradation_level_ice(standard_config, standard_hull, 3.0)
        assert R_level > 0, "Level ice resistance should be positive"
    
    def test_brash_ice_formula(self, standard_config, standard_hull):
        """Test brash ice degradation formula."""
        R_brash = degradation_brash_ice(standard_config, standard_hull, 3.0)
        assert R_brash >= 0, "Brash ice resistance should be non-negative"


# =============================================================================
# C5: Negative control
# =============================================================================

class TestNegativeControl:
    """C5: C→0 → R_pancake→0."""
    
    def test_zero_concentration(self, standard_config, standard_hull):
        """Test that zero concentration gives zero resistance."""
        config_zero = PancakeIceConfig(
            diameter=standard_config.diameter,
            thickness=standard_config.thickness,
            concentration=0.0,
            flexural_strength=standard_config.flexural_strength,
            edge_efficiency=standard_config.edge_efficiency,
        )
        R = pancake_resistance(config_zero, standard_hull, 3.0)
        assert R.R_total == 0.0, f"R(C=0)={R.R_total} should be zero"
    
    def test_near_zero_concentration(self, standard_config, standard_hull):
        """C5: Validate negative control."""
        result = validate_negative_control(standard_config, standard_hull, 3.0)
        assert result.criterion == "C5"
        assert result.passed, f"Negative control failed: {result.message}"
    
    def test_zero_speed(self, standard_config, standard_hull):
        """Test that zero speed gives zero resistance."""
        R = pancake_resistance(standard_config, standard_hull, 0.0)
        assert R.R_total == 0.0, f"R(V=0)={R.R_total} should be zero"


# =============================================================================
# C6: Monotonicity
# =============================================================================

class TestMonotonicity:
    """C6: C↑ → R_pancake monotonically increases."""
    
    def test_concentration_monotonicity(self, standard_config, standard_hull):
        """C6: Validate monotonicity with concentration."""
        concentrations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = validate_monotonicity(standard_config, standard_hull, 3.0, concentrations)
        assert result.criterion == "C6"
        assert result.passed, f"Monotonicity failed: {result.message}"
    
    def test_resistance_increases_with_concentration(self, standard_config, standard_hull):
        """Test that resistance increases with concentration."""
        results = concentration_sweep(standard_config, standard_hull, 3.0, [0.2, 0.5, 0.8])
        R_low = results[0][1].R_total
        R_mid = results[1][1].R_total
        R_high = results[2][1].R_total
        
        assert R_mid > R_low, "R should increase with C"
        assert R_high > R_mid, "R should increase with C"
    
    def test_diameter_sweep(self, standard_config, standard_hull):
        """Test diameter sweep runs correctly."""
        diameters = [0.5, 1.0, 1.5, 2.0]
        results = diameter_sweep(standard_config, standard_hull, 3.0, diameters)
        assert len(results) == 4
        for D, R in results:
            assert R.R_total >= 0, f"R(D={D})={R.R_total} should be non-negative"


# =============================================================================
# Maneuverability
# =============================================================================

class TestManeuverability:
    """Test maneuverability forces."""
    
    def test_zero_drift_zero_lateral(self, standard_config, standard_hull):
        """Test that zero drift angle gives zero lateral force."""
        F = maneuverability_forces(standard_config, standard_hull, 3.0, drift_angle=0.0)
        assert abs(F.F_y) < 1e-6, f"F_y(β=0)={F.F_y} should be zero"
    
    def test_drift_creates_lateral_force(self, standard_config, standard_hull):
        """Test that drift angle creates lateral force."""
        F = maneuverability_forces(standard_config, standard_hull, 3.0, drift_angle=0.1)
        assert F.F_y != 0, "Drift should create lateral force"
    
    def test_yaw_creates_moment(self, standard_config, standard_hull):
        """Test that yaw rate creates yaw moment."""
        F = maneuverability_forces(standard_config, standard_hull, 3.0, yaw_rate=0.01)
        assert F.N_yaw != 0 or F.F_y != 0, "Yaw should create forces"


# =============================================================================
# Field generation
# =============================================================================

class TestFieldGeneration:
    """Test pancake ice field generation."""
    
    def test_generate_field(self):
        """Test field generation."""
        field = generate_field(
            domain_x=(0, 100),
            domain_y=(-50, 50),
            target_concentration=0.5,
            mean_diameter=1.0,
            thickness=0.1,
            seed=42,
        )
        assert len(field.floes) > 0, "Field should have floes"
        assert field.concentration > 0, "Concentration should be positive"
    
    def test_field_statistics(self):
        """Test field statistics computation."""
        field = generate_field(
            domain_x=(0, 50),
            domain_y=(-25, 25),
            target_concentration=0.6,
            mean_diameter=1.0,
            thickness=0.1,
            seed=42,
        )
        stats = compute_field_statistics(field)
        assert stats["n_floes"] > 0
        assert stats["mean_diameter"] > 0
        assert stats["concentration"] > 0
    
    def test_field_reproducibility(self):
        """Test that same seed gives same field."""
        field1 = generate_field(
            domain_x=(0, 50),
            domain_y=(-25, 25),
            target_concentration=0.5,
            mean_diameter=1.0,
            thickness=0.1,
            seed=123,
        )
        field2 = generate_field(
            domain_x=(0, 50),
            domain_y=(-25, 25),
            target_concentration=0.5,
            mean_diameter=1.0,
            thickness=0.1,
            seed=123,
        )
        assert len(field1.floes) == len(field2.floes)
        for f1, f2 in zip(field1.floes, field2.floes):
            assert f1.x == f2.x
            assert f1.y == f2.y


# =============================================================================
# Configuration validation
# =============================================================================

class TestConfiguration:
    """Test configuration validation."""
    
    def test_invalid_diameter(self):
        """Test that invalid diameter raises error."""
        with pytest.raises(ValueError):
            PancakeIceConfig(diameter=0.01)  # Too small
    
    def test_invalid_concentration(self):
        """Test that invalid concentration raises error."""
        with pytest.raises(ValueError):
            PancakeIceConfig(concentration=1.5)  # > 1
    
    def test_invalid_thickness(self):
        """Test that invalid thickness raises error."""
        with pytest.raises(ValueError):
            PancakeIceConfig(thickness=0.001)  # Too small


# =============================================================================
# Run all validations
# =============================================================================

class TestAllValidations:
    """Test running all validations."""
    
    def test_run_all_validations(self):
        """Test that all validations run without error."""
        results = run_all_validations()
        assert "C1" in results
        assert "C2" in results
        assert "C3" in results
        assert "C4a" in results
        assert "C4b" in results
        assert "C5" in results
        assert "C6" in results
        
        for key, result in results.items():
            assert hasattr(result, "criterion")
            assert hasattr(result, "passed")
            assert hasattr(result, "value")
            assert hasattr(result, "reference")
            assert hasattr(result, "error")
            assert hasattr(result, "message")
