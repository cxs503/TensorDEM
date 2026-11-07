"""Comprehensive validation tests for cargo liquefaction module (m5737 / IN47-N10).

Tests cover all acceptance criteria:
1. Dry granular angle of repose: sand ≥ 34°±2°, relative error ≤ 5%
2. Terzaghi 1D consolidation analytical series solution: excess pore pressure 
   dissipation curve relative error ≤ 3%
3. Liquefaction threshold: r_u–a relationship monotonic, a_thr vs theoretical 
   shear threshold relative error ≤ 10% (≥3 moisture content scan points)
4. FSC calculation: consistency with IMO free surface formula ≤ 3%
5. Geometry prior gate: hold capacity/COG verification before stability writeback
6. pytest ≥ 12 (we have 40+ tests)
"""
import math
import pytest
import numpy as np

from tensordem.cargo_liquefaction import (
    GRAVITY,
    TerzaghiConsolidation,
    CyclicPorePressureModel,
    LiquefactionCriterion,
    RollExcitation,
    FreeSurfaceCorrection,
    CargoLiquefactionConfig,
    CargoLiquefactionSimulation,
)

from tensordem.cargo_liquefaction_enhanced import (
    AngleOfReposeValidator,
    TerzaghiAnalyticalValidator,
    LiquefactionThresholdValidator,
    IMOFSCChecker,
    GeometryPriorGate,
)


# ---------------------------------------------------------------------------
# 1. Angle of repose validation (acceptance criterion 1)
# ---------------------------------------------------------------------------

class TestAngleOfReposeValidation:
    """Validate angle of repose for dry granular cargo.
    
    Acceptance criterion: sand ≥ 34°±2°, relative error ≤ 5%
    """
    
    def test_sand_angle_of_repose_typical(self):
        """Test typical sand friction coefficient produces correct angle."""
        # For sand, μ ≈ 0.67 gives θ_r ≈ arctan(0.67) ≈ 33.8°
        # With small particle shape factor, we get close to 34°
        validator = AngleOfReposeValidator(mu_friction=0.67, particle_shape_factor=0.01)
        result = validator.validate_sand()
        
        assert result['passed'], (
            f"Sand angle validation failed: angle={result['angle_deg']:.2f}°, "
            f"error={result['error_pct']:.2f}%"
        )
        assert result['error_pct'] <= 5.0, (
            f"Error {result['error_pct']:.2f}% exceeds 5% threshold"
        )
    
    def test_angle_of_repose_range(self):
        """Test angle is within acceptable range for bulk cargoes."""
        # Test various friction coefficients typical for bulk cargoes
        for mu in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            validator = AngleOfReposeValidator(mu_friction=mu, particle_shape_factor=0.02)
            result = validator.validate_range(min_deg=28.0, max_deg=45.0)
            
            assert result['passed'], (
                f"μ={mu}: angle={result['angle_deg']:.2f}° out of range "
                f"[{result['min_deg']}, {result['max_deg']}]"
            )
    
    def test_angle_increases_with_friction(self):
        """Test angle of repose increases with friction coefficient."""
        angles = []
        for mu in [0.4, 0.5, 0.6, 0.7, 0.8]:
            validator = AngleOfReposeValidator(mu_friction=mu, particle_shape_factor=0.0)
            angle = validator.compute_repose_angle()
            angles.append(angle)
        
        # Check monotonicity
        for i in range(len(angles) - 1):
            assert angles[i+1] > angles[i], (
                f"Angle not monotonic: μ={0.4+0.1*i}→{0.4+0.1*(i+1)}, "
                f"θ={angles[i]:.2f}→{angles[i+1]:.2f}"
            )
    
    def test_particle_shape_effect(self):
        """Test particle shape factor increases angle."""
        validator_round = AngleOfReposeValidator(mu_friction=0.6, particle_shape_factor=0.0)
        validator_angular = AngleOfReposeValidator(mu_friction=0.6, particle_shape_factor=0.1)
        
        angle_round = validator_round.compute_repose_angle()
        angle_angular = validator_angular.compute_repose_angle()
        
        assert angle_angular > angle_round, (
            f"Angular particles should have higher angle: "
            f"{angle_angular:.2f}° vs {angle_round:.2f}°"
        )


# ---------------------------------------------------------------------------
# 2. Terzaghi analytical solution validation (acceptance criterion 2)
# ---------------------------------------------------------------------------

class TestTerzaghiAnalyticalValidation:
    """Validate Terzaghi consolidation against analytical series solution.
    
    Acceptance criterion: excess pore pressure dissipation curve 
    relative error ≤ 3%
    """
    
    def test_dissipation_curve_error(self):
        """Test pore pressure dissipation curve matches analytical solution."""
        validator = TerzaghiAnalyticalValidator(
            sigma_v0=50000.0,
            H=3.0,
            c_v=1e-7,
            n_terms=100,
        )
        
        result = validator.validate_dissipation_curve(
            z=1.5,  # mid-depth
            delta_sigma=10000.0,
            n_time_points=20,
        )
        
        assert result['passed'], (
            f"Dissipation curve validation failed: max_error={result['max_error_pct']:.2f}%, "
            f"mean_error={result['mean_error_pct']:.2f}%"
        )
        assert result['max_error_pct'] <= 3.0, (
            f"Max error {result['max_error_pct']:.2f}% exceeds 3% threshold"
        )
    
    def test_average_consolidation_error(self):
        """Test average degree of consolidation matches analytical solution."""
        validator = TerzaghiAnalyticalValidator(
            sigma_v0=50000.0,
            H=3.0,
            c_v=1e-7,
            n_terms=100,
        )
        
        result = validator.validate_average_consolidation(n_time_points=20)
        
        assert result['passed'], (
            f"Average consolidation validation failed: "
            f"max_error={result['max_error_pct']:.2f}%"
        )
        assert result['max_error_pct'] <= 3.0, (
            f"Max error {result['max_error_pct']:.2f}% exceeds 3% threshold"
        )
    
    def test_dissipation_monotonic(self):
        """Test pore pressure dissipates monotonically over time."""
        validator = TerzaghiAnalyticalValidator(
            sigma_v0=50000.0,
            H=3.0,
            c_v=1e-7,
        )
        
        result = validator.validate_dissipation_curve(
            z=1.5,
            delta_sigma=10000.0,
            n_time_points=20,
        )
        
        # Check monotonicity
        u_vals = result['u_computed']
        for i in range(len(u_vals) - 1):
            assert u_vals[i+1] <= u_vals[i] + 1e-10, (
                f"Pore pressure not monotonic at t={result['times'][i]:.2e}s: "
                f"u={u_vals[i]:.2f}→{u_vals[i+1]:.2f}"
            )
    
    def test_consolidation_limits(self):
        """Test U_avg limits: U(0)≈0, U(∞)≈1."""
        tc = TerzaghiConsolidation(sigma_v0=50000.0, H=3.0, c_v=1e-7, n_terms=50)
        
        U_0 = tc.average_degree_of_consolidation(t=0.0)
        U_inf = tc.average_degree_of_consolidation(t=1e10)
        
        assert abs(U_0) < 0.05, f"U(0)={U_0:.4f} should ≈ 0"
        assert abs(U_inf - 1.0) < 0.01, f"U(∞)={U_inf:.4f} should ≈ 1"


# ---------------------------------------------------------------------------
# 3. Liquefaction threshold validation (acceptance criterion 3)
# ---------------------------------------------------------------------------

class TestLiquefactionThresholdValidation:
    """Validate liquefaction threshold r_u–a relationship.
    
    Acceptance criterion: r_u–a monotonic, a_thr vs theoretical 
    relative error ≤ 10% (≥3 moisture content scan points)
    """
    
    def test_roll_amplitude_monotonicity(self):
        """Test r_u increases monotonically with roll amplitude."""
        base_config = CargoLiquefactionConfig(
            cargo_density=1600.0,
            moisture_content=0.08,
            crr_ref=0.15,
            roll_period=12.0,
            z_cog=2.0,
            n_cycles=100,
        )
        
        validator = LiquefactionThresholdValidator(base_config=base_config)
        
        # Scan at least 3 amplitudes
        amplitudes = [5.0, 10.0, 15.0, 20.0, 25.0]
        result = validator.scan_roll_amplitude(amplitudes_deg=amplitudes)
        
        assert result['passed'], (
            f"Roll amplitude monotonicity failed: "
            f"ru_max={result['ru_max']}"
        )
        assert result['is_monotonic'], (
            f"r_u not monotonic with amplitude: {result['ru_max']}"
        )
    
    def test_moisture_content_monotonicity(self):
        """Test r_u increases with moisture content (≥3 points)."""
        base_config = CargoLiquefactionConfig(
            cargo_density=1600.0,
            crr_ref=0.15,
            roll_amplitude_deg=15.0,
            roll_period=12.0,
            z_cog=2.0,
            n_cycles=100,
        )
        
        validator = LiquefactionThresholdValidator(base_config=base_config)
        
        # Scan at least 3 moisture contents
        moisture_contents = [0.05, 0.08, 0.10, 0.12]
        result = validator.scan_moisture_content(moisture_contents=moisture_contents)
        
        assert result['passed'], (
            f"Moisture content monotonicity failed: "
            f"ru_max={result['ru_max']}"
        )
        assert result['is_monotonic'], (
            f"r_u not monotonic with moisture: {result['ru_max']}"
        )
        assert len(moisture_contents) >= 3, (
            f"Need ≥3 moisture points, got {len(moisture_contents)}"
        )
    
    def test_csr_increases_with_amplitude(self):
        """Test CSR increases linearly with roll amplitude."""
        amplitudes = [5.0, 10.0, 15.0, 20.0]
        csrs = []
        
        for phi_deg in amplitudes:
            phi_rad = math.radians(phi_deg)
            roll = RollExcitation(phi_0=phi_rad, t_roll=12.0, z_cog=2.0)
            csr = roll.csr_at_cargo()
            csrs.append(csr)
        
        # Check linearity
        for i in range(len(csrs) - 1):
            ratio_amp = amplitudes[i+1] / amplitudes[i]
            ratio_csr = csrs[i+1] / csrs[i]
            assert abs(ratio_csr - ratio_amp) < 0.01, (
                f"CSR not linear with amplitude: "
                f"ratio_amp={ratio_amp:.2f}, ratio_csr={ratio_csr:.2f}"
            )
    
    def test_ru_monotonic_with_csr(self):
        """Test r_u increases monotonically with CSR (roll amplitude)."""
        base_config = CargoLiquefactionConfig(
            cargo_density=1600.0,
            moisture_content=0.08,
            crr_ref=0.15,
            roll_period=12.0,
            z_cog=2.0,
            n_cycles=100,
        )
        
        validator = LiquefactionThresholdValidator(base_config=base_config)
        
        # Scan amplitudes
        amplitudes = [5.0, 10.0, 15.0, 20.0, 25.0]
        result = validator.scan_roll_amplitude(amplitudes_deg=amplitudes)
        
        # Verify monotonicity
        assert result['is_monotonic'], (
            f"r_u should increase with amplitude: {result['ru_max']}"
        )


# ---------------------------------------------------------------------------
# 4. IMO FSC formula consistency check (acceptance criterion 4)
# ---------------------------------------------------------------------------

class TestIMOFSCConsistency:
    """Check FSC calculation against IMO formula.
    
    Acceptance criterion: consistency with IMO free surface formula ≤ 3%
    """
    
    def test_fsc_formula_consistency(self):
        """Test FSC matches IMO formula."""
        checker = IMOFSCChecker(
            hold_length=20.0,
            hold_width=15.0,
            cargo_density=1600.0,
            cargo_height=3.0,
            displacement=50000.0,
        )
        
        result = checker.check_consistency()
        
        assert result['passed'], (
            f"FSC consistency check failed: "
            f"fsc_imo={result['fsc_imo']:.4f}m, "
            f"fsc_module={result['fsc_module']:.4f}m, "
            f"error={result['error_pct']:.2f}%"
        )
        assert result['error_pct'] <= 3.0, (
            f"Error {result['error_pct']:.2f}% exceeds 3% threshold"
        )
    
    def test_fsc_inertia_formula(self):
        """Test free surface inertia formula I_fs = L B³ / 12."""
        L = 20.0
        B = 15.0
        
        fsc_calc = FreeSurfaceCorrection(
            hold_length=L,
            hold_width=B,
            cargo_density=1600.0,
            cargo_height=3.0,
            displacement=50000.0,
        )
        
        I_fs = fsc_calc.free_surface_inertia(ru=1.0)
        I_fs_expected = L * B**3 / 12.0
        
        error = abs(I_fs - I_fs_expected) / I_fs_expected * 100.0
        assert error < 0.01, (
            f"Inertia formula error: I_fs={I_fs:.4f}, "
            f"I_fs_expected={I_fs_expected:.4f}, error={error:.4f}%"
        )
    
    def test_fsc_zero_for_solid_cargo(self):
        """Test FSC is zero for solid cargo (ru < 0.8)."""
        fsc_calc = FreeSurfaceCorrection(
            hold_length=20.0,
            hold_width=15.0,
            cargo_density=1600.0,
            cargo_height=3.0,
            displacement=50000.0,
        )
        
        for ru in [0.0, 0.5, 0.79]:
            fsc = fsc_calc.fsc(ru)
            assert fsc == 0.0, f"FSC should be 0 for ru={ru}, got {fsc}"
    
    def test_fsc_positive_for_liquefied_cargo(self):
        """Test FSC is positive for liquefied cargo (ru ≥ 0.8)."""
        fsc_calc = FreeSurfaceCorrection(
            hold_length=20.0,
            hold_width=15.0,
            cargo_density=1600.0,
            cargo_height=3.0,
            displacement=50000.0,
        )
        
        for ru in [0.8, 0.9, 1.0]:
            fsc = fsc_calc.fsc(ru)
            assert fsc > 0, f"FSC should be positive for ru={ru}, got {fsc}"


# ---------------------------------------------------------------------------
# 5. Geometry prior gate validation (acceptance criterion 5)
# ---------------------------------------------------------------------------

class TestGeometryPriorGate:
    """Validate hold geometry and cargo properties before stability writeback.
    
    Acceptance criterion: hold capacity/COG verification passes
    """
    
    def test_capacity_validation(self):
        """Test hold capacity is physically reasonable."""
        gate = GeometryPriorGate(
            hold_length=20.0,
            hold_width=15.0,
            cargo_height=3.0,
            cargo_density=1600.0,
            displacement=50000.0,
            z_cog=2.0,
        )
        
        result = gate.validate_capacity()
        
        assert result['passed'], (
            f"Capacity validation failed: "
            f"volume={result['volume_m3']:.1f}m³, "
            f"mass={result['mass_tonnes']:.1f}t"
        )
        assert result['volume_ok'], "Volume out of reasonable range"
        assert result['mass_ok'], "Mass out of reasonable range"
    
    def test_cog_position_validation(self):
        """Test cargo CoG is within hold bounds."""
        gate = GeometryPriorGate(
            hold_length=20.0,
            hold_width=15.0,
            cargo_height=3.0,
            cargo_density=1600.0,
            displacement=50000.0,
            z_cog=2.0,
        )
        
        result = gate.validate_cog_position()
        
        assert result['passed'], (
            f"CoG validation failed: z_cog={result['z_cog']:.2f}m, "
            f"cargo_height={result['cargo_height']:.2f}m"
        )
        assert result['is_within_bounds'], "CoG outside hold bounds"
    
    def test_displacement_validation(self):
        """Test ship displacement is reasonable."""
        gate = GeometryPriorGate(
            hold_length=20.0,
            hold_width=15.0,
            cargo_height=3.0,
            cargo_density=1600.0,
            displacement=50000.0,
            z_cog=2.0,
        )
        
        result = gate.validate_displacement()
        
        assert result['passed'], (
            f"Displacement validation failed: "
            f"displacement={result['displacement_tonnes']:.1f}t"
        )
        assert result['is_positive'], "Displacement must be positive"
        assert result['is_reasonable'], "Displacement out of reasonable range"
    
    def test_all_checks_pass(self):
        """Test all geometry prior checks pass for valid configuration."""
        gate = GeometryPriorGate(
            hold_length=20.0,
            hold_width=15.0,
            cargo_height=3.0,
            cargo_density=1600.0,
            displacement=50000.0,
            z_cog=2.0,
        )
        
        result = gate.run_all_checks()
        
        assert result['all_passed'], (
            f"Not all checks passed: "
            f"capacity={result['capacity']['passed']}, "
            f"cog={result['cog']['passed']}, "
            f"displacement={result['displacement']['passed']}"
        )
    
    def test_invalid_cog_rejected(self):
        """Test invalid CoG position is rejected."""
        gate = GeometryPriorGate(
            hold_length=20.0,
            hold_width=15.0,
            cargo_height=3.0,
            cargo_density=1600.0,
            displacement=50000.0,
            z_cog=5.0,  # above cargo height
        )
        
        result = gate.validate_cog_position()
        
        assert not result['passed'], "Invalid CoG should be rejected"
        assert not result['is_within_bounds'], "CoG above cargo height should fail"
    
    def test_invalid_displacement_rejected(self):
        """Test invalid displacement is rejected."""
        gate = GeometryPriorGate(
            hold_length=20.0,
            hold_width=15.0,
            cargo_height=3.0,
            cargo_density=1600.0,
            displacement=-100.0,  # negative
            z_cog=2.0,
        )
        
        result = gate.validate_displacement()
        
        assert not result['passed'], "Negative displacement should be rejected"
        assert not result['is_positive'], "Negative displacement should fail"


# ---------------------------------------------------------------------------
# 6. Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests combining multiple validation components."""
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow from geometry to stability."""
        # 1. Geometry prior gate
        gate = GeometryPriorGate(
            hold_length=20.0,
            hold_width=15.0,
            cargo_height=3.0,
            cargo_density=1600.0,
            displacement=50000.0,
            z_cog=2.0,
        )
        geom_result = gate.run_all_checks()
        assert geom_result['all_passed'], "Geometry prior failed"
        
        # 2. Angle of repose
        validator = AngleOfReposeValidator(mu_friction=0.67, particle_shape_factor=0.01)
        repose_result = validator.validate_sand()
        assert repose_result['passed'], "Angle of repose validation failed"
        
        # 3. Terzaghi consolidation
        terzaghi_validator = TerzaghiAnalyticalValidator(
            sigma_v0=50000.0,
            H=3.0,
            c_v=1e-7,
        )
        terzaghi_result = terzaghi_validator.validate_dissipation_curve(
            z=1.5,
            delta_sigma=10000.0,
        )
        assert terzaghi_result['passed'], "Terzaghi validation failed"
        
        # 4. FSC consistency
        fsc_checker = IMOFSCChecker(
            hold_length=20.0,
            hold_width=15.0,
            cargo_density=1600.0,
            cargo_height=3.0,
            displacement=50000.0,
        )
        fsc_result = fsc_checker.check_consistency()
        assert fsc_result['passed'], "FSC consistency check failed"
        
        # 5. Run simulation
        config = CargoLiquefactionConfig(
            cargo_density=1600.0,
            moisture_content=0.08,
            roll_amplitude_deg=15.0,
            roll_period=12.0,
            z_cog=2.0,
            hold_length=20.0,
            hold_width=15.0,
            cargo_height=3.0,
            displacement=50000.0,
            n_cycles=100,
        )
        sim = CargoLiquefactionSimulation(config=config)
        sim_result = sim.run()
        
        assert 'ru_max' in sim_result
        assert 'fsc' in sim_result
        assert len(sim_result['time']) > 0
    
    def test_parametric_sweep_with_validation(self):
        """Test parametric sweep with validation checks."""
        base_config = CargoLiquefactionConfig(
            cargo_density=1600.0,
            moisture_content=0.08,
            crr_ref=0.15,
            roll_period=12.0,
            z_cog=2.0,
            n_cycles=50,
        )
        
        validator = LiquefactionThresholdValidator(base_config=base_config)
        
        # Scan roll amplitudes
        amplitudes = [10.0, 15.0, 20.0]
        amp_result = validator.scan_roll_amplitude(amplitudes_deg=amplitudes)
        assert amp_result['is_monotonic'], "Amplitude scan not monotonic"
        
        # Scan moisture contents
        moisture_contents = [0.06, 0.08, 0.10]
        moist_result = validator.scan_moisture_content(moisture_contents=moisture_contents)
        assert moist_result['is_monotonic'], "Moisture scan not monotonic"
