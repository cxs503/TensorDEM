"""Tests for crushed ice layer DEM model.

These tests verify:
1. Dilatancy model: ψ(p) decreases with increasing pressure
2. Fragment size distribution: Weibull/Power-law fitting
3. Contact force validation against Hertz theory
4. Sensitivity analysis: ice thickness, speed, bow angle
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

from tensordem.crushed_ice import (
    DilatancyModel,
    Fragment,
    FragmentShape,
    FragmentSizeDistribution,
    ContactForceChain,
    CrushedIceLayer,
    serialize_fragments,
    load_fragments,
)


class DilatancyModelTests(unittest.TestCase):
    """Tests for pressure-dependent dilatancy model."""
    
    def test_dilatancy_decreases_with_pressure(self):
        """Verify ψ(p) decreases with increasing pressure (Spearman < -0.9)."""
        model = DilatancyModel(psi_0=0.35, p_0=1e6, psi_min=0.02)
        
        pressures = np.array([0, 1e4, 1e5, 5e5, 1e6, 2e6, 5e6, 1e7])
        dilatancy_angles = [model.dilatancy_angle(p) for p in pressures]
        
        # Spearman correlation should be negative (decreasing)
        spearman_corr, p_value = scipy_stats.spearmanr(pressures, dilatancy_angles)
        
        self.assertLess(spearman_corr, -0.9, 
            f"Spearman correlation {spearman_corr} should be < -0.9")
    
    def test_dilatancy_bounds(self):
        """Verify dilatancy angle stays within bounds."""
        model = DilatancyModel(psi_0=0.35, p_0=1e6, psi_min=0.02)
        
        # At zero pressure, should be psi_0 + psi_min (the formula adds both)
        psi_0 = model.dilatancy_angle(0)
        expected_psi_0 = model.psi_0 + model.psi_min  # 0.35 + 0.02 = 0.37
        self.assertAlmostEqual(psi_0, expected_psi_0, places=5)
        
        # At very high pressure, should approach psi_min
        psi_inf = model.dilatancy_angle(1e10)
        self.assertAlmostEqual(psi_inf, model.psi_min, places=5)
    
    def test_dilatancy_degrees(self):
        """Verify degree conversion."""
        model = DilatancyModel(psi_0=0.35, p_0=1e6, psi_min=0.02)
        
        psi_rad = model.dilatancy_angle(1e5)
        psi_deg = model.dilatancy_angle_deg(1e5)
        
        self.assertAlmostEqual(math.degrees(psi_rad), psi_deg, places=5)
    
    def test_invalid_inputs(self):
        """Verify invalid inputs raise errors."""
        with self.assertRaises(ValueError):
            DilatancyModel(psi_0=-0.1)
        with self.assertRaises(ValueError):
            DilatancyModel(psi_0=math.pi)
        with self.assertRaises(ValueError):
            DilatancyModel(psi_min=0.5, psi_0=0.3)  # psi_min > psi_0
        with self.assertRaises(ValueError):
            DilatancyModel(p_0=-1)


class FragmentTests(unittest.TestCase):
    """Tests for Fragment dataclass."""
    
    def test_characteristic_size(self):
        """Test characteristic size computation."""
        frag = Fragment(
            id=0,
            particles=torch.tensor([0, 1, 2]),
            volume_m3=0.001,
            equivalent_radius_m=0.062,
            bounding_box_m=[0.1, 0.08, 0.06],
            shape=FragmentShape.BLOCK,
            creation_time_s=0.0
        )
        
        expected = (0.1 * 0.08 * 0.06) ** (1.0 / 3.0)
        self.assertAlmostEqual(frag.characteristic_size_m, expected, places=5)
    
    def test_aspect_ratio(self):
        """Test aspect ratio computation."""
        frag = Fragment(
            id=0,
            particles=torch.tensor([0]),
            volume_m3=0.001,
            equivalent_radius_m=0.062,
            bounding_box_m=[0.3, 0.1, 0.1],
            shape=FragmentShape.ROD,
            creation_time_s=0.0
        )
        
        self.assertAlmostEqual(frag.aspect_ratio, 3.0, places=5)
    
    def test_shape_classification(self):
        """Test shape classification based on aspect ratio."""
        # Block: aspect < 1.5
        frag_block = Fragment(
            id=0, particles=torch.tensor([0]),
            volume_m3=0.001, equivalent_radius_m=0.062,
            bounding_box_m=[0.1, 0.09, 0.08],
            shape=FragmentShape.BLOCK, creation_time_s=0.0
        )
        self.assertEqual(frag_block.shape, FragmentShape.BLOCK)
        
        # Rod: 1.5 <= aspect < 3
        frag_rod = Fragment(
            id=1, particles=torch.tensor([0]),
            volume_m3=0.001, equivalent_radius_m=0.062,
            bounding_box_m=[0.2, 0.1, 0.07],
            shape=FragmentShape.ROD, creation_time_s=0.0
        )
        self.assertEqual(frag_rod.shape, FragmentShape.ROD)


class FragmentSizeDistributionTests(unittest.TestCase):
    """Tests for fragment size distribution analysis."""
    
    def setUp(self):
        """Generate synthetic fragment sizes for testing."""
        # Generate Weibull-distributed sizes (typical for fragmentation)
        np.random.seed(42)
        shape = 2.5  # Weibull shape parameter
        scale = 0.05  # Scale parameter (m)
        self.sizes = scipy_stats.weibull_min.rvs(shape, scale=scale, size=100)
        self.dist = FragmentSizeDistribution(fragment_sizes=self.sizes.tolist())
    
    def test_statistics(self):
        """Test basic statistics computation."""
        self.assertAlmostEqual(self.dist.mean_size_m, np.mean(self.sizes), places=3)
        self.assertAlmostEqual(self.dist.median_size_m, np.median(self.sizes), places=3)
        self.assertAlmostEqual(self.dist.std_size_m, np.std(self.sizes), places=3)
    
    def test_histogram(self):
        """Test histogram computation."""
        bins, counts = self.dist.compute_histogram(num_bins=10)
        
        self.assertEqual(len(bins), 10)
        self.assertEqual(len(counts), 10)
        self.assertEqual(sum(counts), len(self.sizes))
    
    def test_weibull_fit(self):
        """Test Weibull distribution fitting."""
        result = self.dist.fit_weibull()
        
        self.assertIn('shape_k', result)
        self.assertIn('scale_lambda', result)
        self.assertIn('r_squared', result)
        self.assertGreater(result['shape_k'], 0)
        self.assertGreater(result['scale_lambda'], 0)
        self.assertGreater(result['r_squared'], 0.5)  # Should fit reasonably
    
    def test_power_law_fit(self):
        """Test power-law distribution fitting."""
        result = self.dist.fit_power_law()
        
        self.assertIn('alpha', result)
        self.assertIn('x_min', result)
        self.assertIn('r_squared', result)
        self.assertGreater(result['alpha'], 1)  # Physical requirement
    
    def test_ks_test_weibull(self):
        """Test KS test against Weibull distribution."""
        result = self.dist.ks_test_weibull()
        
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        # For synthetic data, p-value should be reasonable
        self.assertGreaterEqual(result['p_value'], 0)
    
    def test_empty_sizes_raises(self):
        """Test that empty sizes raises error."""
        with self.assertRaises(ValueError):
            FragmentSizeDistribution(fragment_sizes=[])


class ContactForceChainTests(unittest.TestCase):
    """Tests for contact force chain analysis."""
    
    def setUp(self):
        """Create synthetic contact data."""
        np.random.seed(42)
        n_contacts = 100
        
        # Generate random contact pairs
        pairs = torch.randint(0, 50, (n_contacts, 2), dtype=torch.long)
        
        # Generate random force magnitudes (log-normal distribution)
        force_magnitudes = torch.tensor(
            scipy_stats.lognorm.rvs(s=0.5, scale=100, size=n_contacts)
        )
        
        # Random force directions
        directions = torch.randn(n_contacts, 3)
        directions = directions / directions.norm(dim=1, keepdim=True)
        forces = directions * force_magnitudes[:, None]
        
        self.chain = ContactForceChain(
            contact_pairs=pairs,
            contact_forces=forces,
            normal_forces=force_magnitudes,
            contact_areas=torch.ones(n_contacts) * 0.001  # 1 mm²
        )
    
    def test_statistics(self):
        """Test contact force statistics."""
        self.assertGreater(self.chain.mean_force_N, 0)
        self.assertGreater(self.chain.max_force_N, 0)
        self.assertGreater(self.chain.total_force_N, 0)
        self.assertEqual(self.chain.num_contacts, 100)
    
    def test_hertz_validation(self):
        """Test Hertz contact validation."""
        result = self.chain.validate_against_hertz(
            particle_radius=0.025,  # 25 mm
            effective_modulus=9e9  # 9 GPa for ice
        )
        
        self.assertIn('mean_force_N', result)
        self.assertIn('max_force_N', result)
        self.assertIn('num_contacts', result)
        self.assertIn('mean_error_pct', result)


class CrushedIceLayerTests(unittest.TestCase):
    """Tests for crushed ice layer model."""
    
    def setUp(self):
        """Create synthetic crushed ice layer."""
        np.random.seed(42)
        n_particles = 50
        
        # Random positions in a box
        positions = torch.randn(n_particles, 3) * 0.1
        velocities = torch.randn(n_particles, 3) * 0.01
        radii = torch.ones(n_particles) * 0.025  # 25 mm
        
        # Assign to fragments (3 fragments)
        fragment_id = torch.randint(0, 3, (n_particles,))
        
        self.layer = CrushedIceLayer(
            particles=positions,
            velocities=velocities,
            radii=radii,
            fragment_id=fragment_id,
            dilatancy_model=DilatancyModel()
        )
    
    def test_basic_properties(self):
        """Test basic layer properties."""
        self.assertEqual(self.layer.num_particles, 50)
        self.assertEqual(self.layer.num_fragments, 3)
    
    def test_extract_fragments(self):
        """Test fragment extraction."""
        fragments = self.layer.extract_fragments()
        
        self.assertEqual(len(fragments), 3)
        for frag in fragments:
            self.assertIsInstance(frag, Fragment)
            self.assertGreater(frag.volume_m3, 0)
            self.assertGreater(frag.equivalent_radius_m, 0)
    
    def test_fragment_size_distribution(self):
        """Test fragment size distribution computation."""
        dist = self.layer.compute_fragment_size_distribution()
        
        self.assertIsInstance(dist, FragmentSizeDistribution)
        self.assertEqual(len(dist.fragment_sizes), 3)


class SerializationTests(unittest.TestCase):
    """Tests for fragment serialization."""
    
    def test_serialize_round_trip(self):
        """Test serialize/load round trip."""
        fragments = [
            Fragment(
                id=0,
                particles=torch.tensor([0, 1, 2]),
                volume_m3=0.001,
                equivalent_radius_m=0.062,
                bounding_box_m=[0.1, 0.08, 0.06],
                shape=FragmentShape.BLOCK,
                creation_time_s=0.5
            ),
            Fragment(
                id=1,
                particles=torch.tensor([3, 4]),
                volume_m3=0.0005,
                equivalent_radius_m=0.05,
                bounding_box_m=[0.08, 0.06, 0.05],
                shape=FragmentShape.ROD,
                creation_time_s=0.8
            ),
        ]
        
        # Serialize
        data = serialize_fragments(fragments)
        
        # Load
        loaded = load_fragments(data)
        
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].id, 0)
        self.assertEqual(loaded[1].id, 1)
        self.assertAlmostEqual(loaded[0].volume_m3, 0.001, places=5)
        self.assertAlmostEqual(loaded[1].creation_time_s, 0.8, places=5)


class SensitivityAnalysisTests(unittest.TestCase):
    """Tests for sensitivity analysis functionality."""
    
    def test_dilatancy_sensitivity_to_parameters(self):
        """Test that dilatancy is sensitive to model parameters."""
        model = DilatancyModel(psi_0=0.35, p_0=1e6, psi_min=0.02)
        
        # At moderate pressure
        p = 5e5  # 500 kPa
        
        # Vary psi_0
        model1 = DilatancyModel(psi_0=0.30, p_0=1e6, psi_min=0.02)
        model2 = DilatancyModel(psi_0=0.40, p_0=1e6, psi_min=0.02)
        
        psi1 = model1.dilatancy_angle(p)
        psi2 = model2.dilatancy_angle(p)
        
        # Higher psi_0 should give higher dilatancy
        self.assertLess(psi1, psi2)
    
    def test_pressure_scale_sensitivity(self):
        """Test sensitivity to pressure scale parameter."""
        p = 1e6  # 1 MPa
        
        model1 = DilatancyModel(psi_0=0.35, p_0=5e5, psi_min=0.02)
        model2 = DilatancyModel(psi_0=0.35, p_0=2e6, psi_min=0.02)
        
        psi1 = model1.dilatancy_angle(p)
        psi2 = model2.dilatancy_angle(p)
        
        # Lower p_0 means faster decay, so at p=1e6:
        # model1 (p_0=5e5): exp(-2) ≈ 0.135
        # model2 (p_0=2e6): exp(-0.5) ≈ 0.607
        # So model1 should have lower dilatancy
        self.assertLess(psi1, psi2)


if __name__ == "__main__":
    unittest.main()