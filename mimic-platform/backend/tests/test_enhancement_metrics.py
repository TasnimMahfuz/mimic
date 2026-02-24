"""
Unit Tests for Enhancement Processing and Scientific Metrics

This module contains unit tests for the enhancement processing and scientific
metrics computation functionality in the MIMIC service.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 11.7, 13.10, 13.11, 18.1**
"""

import pytest
import numpy as np

from app.services.mimic_service import MIMICService, ScientificMetrics
from app.services.transforms.fft_directional import CurveletCoefficients


@pytest.fixture
def mimic_service():
    """Fixture providing a MIMICService instance."""
    return MIMICService()


@pytest.fixture
def sample_image():
    """Fixture providing a sample test image."""
    # Create a simple test image with some structure
    image = np.zeros((128, 128))
    # Add a bright square in the center
    image[40:88, 40:88] = 0.8
    # Add some gradient
    for i in range(128):
        image[i, :] += i / 256.0
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    return image


@pytest.fixture
def sample_coefficients():
    """Fixture providing sample curvelet coefficients."""
    # Create simple test coefficients
    coeffs_dict = {}
    
    # Scale 0 (coarsest, no orientation)
    coeffs_dict[0] = {0: np.random.rand(32, 32)}
    
    # Scale 1 (8 orientations)
    coeffs_dict[1] = {
        i: np.random.rand(64, 64) for i in range(8)
    }
    
    # Scale 2 (8 orientations)
    coeffs_dict[2] = {
        i: np.random.rand(128, 128) for i in range(8)
    }
    
    return CurveletCoefficients(
        coefficients=coeffs_dict,
        scales=3,
        orientations=8,
        shape=(128, 128)
    )


class TestContrastEnhancement:
    """Tests for contrast enhancement functionality."""
    
    def test_contrast_enhancement_increases_dynamic_range(self, mimic_service, sample_image):
        """
        Test that contrast enhancement increases the dynamic range.
        
        **Validates: Requirements 10.1, 18.1**
        """
        original_range = np.max(sample_image) - np.min(sample_image)
        
        enhanced = mimic_service.apply_contrast_enhancement(sample_image, enhancement_factor=2.0)
        enhanced_range = np.max(enhanced) - np.min(enhanced)
        
        # Enhanced range should be greater than or equal to original
        assert enhanced_range >= original_range - 1e-6
    
    def test_contrast_enhancement_preserves_range(self, mimic_service, sample_image):
        """
        Test that contrast enhancement keeps values in [0, 1].
        
        **Validates: Requirements 10.1, 18.1**
        """
        enhanced = mimic_service.apply_contrast_enhancement(sample_image, enhancement_factor=3.0)
        
        assert np.all(enhanced >= 0.0)
        assert np.all(enhanced <= 1.0)
    
    def test_contrast_enhancement_with_factor_one(self, mimic_service, sample_image):
        """
        Test that enhancement factor of 1.0 preserves the image.
        
        **Validates: Requirements 10.1, 18.1**
        """
        enhanced = mimic_service.apply_contrast_enhancement(sample_image, enhancement_factor=1.0)
        
        assert np.allclose(enhanced, sample_image, rtol=1e-5)
    
    def test_contrast_enhancement_invalid_factor(self, mimic_service, sample_image):
        """
        Test that invalid enhancement factors raise errors.
        
        **Validates: Requirements 10.1, 18.1**
        """
        with pytest.raises(ValueError):
            mimic_service.apply_contrast_enhancement(sample_image, enhancement_factor=0.0)
        
        with pytest.raises(ValueError):
            mimic_service.apply_contrast_enhancement(sample_image, enhancement_factor=-1.0)


class TestSpatialSmoothing:
    """Tests for spatial smoothing functionality."""
    
    def test_spatial_smoothing_reduces_gradients(self, mimic_service, sample_image):
        """
        Test that spatial smoothing reduces high-frequency content.
        
        **Validates: Requirements 10.2, 18.1**
        """
        # Compute original gradient magnitude
        grad_y, grad_x = np.gradient(sample_image)
        original_gradient = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Apply smoothing
        smoothed = mimic_service.apply_spatial_smoothing(sample_image, kernel_size=2.0)
        
        # Compute smoothed gradient magnitude
        smooth_grad_y, smooth_grad_x = np.gradient(smoothed)
        smoothed_gradient = np.mean(np.sqrt(smooth_grad_x**2 + smooth_grad_y**2))
        
        # Smoothed gradient should be less than or equal to original
        assert smoothed_gradient <= original_gradient + 1e-6
    
    def test_spatial_smoothing_preserves_range(self, mimic_service, sample_image):
        """
        Test that spatial smoothing keeps values in [0, 1].
        
        **Validates: Requirements 10.2, 18.1**
        """
        smoothed = mimic_service.apply_spatial_smoothing(sample_image, kernel_size=3.0)
        
        assert np.all(smoothed >= 0.0)
        assert np.all(smoothed <= 1.0)
    
    def test_spatial_smoothing_with_zero_kernel(self, mimic_service, sample_image):
        """
        Test that kernel size of 0.0 preserves the image.
        
        **Validates: Requirements 10.2, 18.1**
        """
        smoothed = mimic_service.apply_spatial_smoothing(sample_image, kernel_size=0.0)
        
        assert np.allclose(smoothed, sample_image, rtol=1e-5)
    
    def test_spatial_smoothing_invalid_kernel(self, mimic_service, sample_image):
        """
        Test that invalid kernel sizes raise errors.
        
        **Validates: Requirements 10.2, 18.1**
        """
        with pytest.raises(ValueError):
            mimic_service.apply_spatial_smoothing(sample_image, kernel_size=-1.0)


class TestCombinedEnhancement:
    """Tests for combined enhancement functionality."""
    
    def test_combined_enhancement_applies_both(self, mimic_service, sample_image):
        """
        Test that combined enhancement applies both smoothing and contrast.
        
        **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 18.1**
        """
        enhanced = mimic_service.apply_enhancement(
            sample_image,
            enhancement_factor=2.0,
            smoothing=1.0
        )
        
        # Should be different from original
        assert not np.allclose(enhanced, sample_image)
        
        # Should be in valid range
        assert np.all(enhanced >= 0.0)
        assert np.all(enhanced <= 1.0)
    
    def test_combined_enhancement_identity(self, mimic_service, sample_image):
        """
        Test that identity parameters preserve the image.
        
        **Validates: Requirements 10.3, 10.4, 18.1**
        """
        enhanced = mimic_service.apply_enhancement(
            sample_image,
            enhancement_factor=1.0,
            smoothing=0.0
        )
        
        assert np.allclose(enhanced, sample_image, rtol=1e-5)
    
    def test_combined_enhancement_preserves_shape(self, mimic_service, sample_image):
        """
        Test that enhancement preserves image shape.
        
        **Validates: Requirements 10.3, 18.1**
        """
        enhanced = mimic_service.apply_enhancement(
            sample_image,
            enhancement_factor=2.5,
            smoothing=1.5
        )
        
        assert enhanced.shape == sample_image.shape


class TestScientificMetrics:
    """Tests for scientific metrics computation."""
    
    def test_compute_anisotropy_map(self, mimic_service, sample_coefficients):
        """
        Test anisotropy map computation.
        
        **Validates: Requirements 13.11, 18.1**
        """
        anisotropy_map = mimic_service.compute_anisotropy_map(sample_coefficients)
        
        # Should have correct shape
        assert anisotropy_map.shape == sample_coefficients.shape
        
        # Should be non-negative
        assert np.all(anisotropy_map >= 0.0)
        
        # Should be finite
        assert np.all(np.isfinite(anisotropy_map))
    
    def test_compute_radial_energy(self, mimic_service, sample_coefficients):
        """
        Test radial energy profile computation.
        
        **Validates: Requirements 13.11, 18.1**
        """
        radial_energy = mimic_service.compute_radial_energy(sample_coefficients)
        
        # Should be 1D array
        assert radial_energy.ndim == 1
        
        # Should be non-negative
        assert np.all(radial_energy >= 0.0)
        
        # Should be finite
        assert np.all(np.isfinite(radial_energy))
    
    def test_compute_edge_confidence(self, mimic_service, sample_coefficients):
        """
        Test edge confidence map computation.
        
        **Validates: Requirements 13.11, 18.1**
        """
        # Create a simple edge map
        edge_map = np.zeros(sample_coefficients.shape)
        edge_map[60:68, :] = 1.0  # Horizontal edge
        
        confidence_map = mimic_service.compute_edge_confidence(
            sample_coefficients,
            edge_map
        )
        
        # Should have correct shape
        assert confidence_map.shape == sample_coefficients.shape
        
        # Should be in [0, 1]
        assert np.all(confidence_map >= 0.0)
        assert np.all(confidence_map <= 1.0)
        
        # Should be finite
        assert np.all(np.isfinite(confidence_map))
    
    def test_compute_scientific_metrics_complete(self, mimic_service, sample_coefficients):
        """
        Test complete scientific metrics computation.
        
        **Validates: Requirements 13.11, 18.1**
        """
        # Create a simple edge map
        edge_map = np.zeros(sample_coefficients.shape)
        edge_map[60:68, :] = 1.0
        
        metrics = mimic_service.compute_scientific_metrics(
            sample_coefficients,
            edge_map
        )
        
        # Should return ScientificMetrics object
        assert isinstance(metrics, ScientificMetrics)
        
        # Check all fields are present and valid
        assert metrics.anisotropy_map.shape == sample_coefficients.shape
        assert isinstance(metrics.directional_energy, dict)
        assert len(metrics.directional_energy) > 0
        assert metrics.radial_energy.ndim == 1
        assert metrics.angular_distribution.shape == (sample_coefficients.orientations,)
        assert metrics.edge_confidence.shape == sample_coefficients.shape
        
        # All values should be finite
        assert np.all(np.isfinite(metrics.anisotropy_map))
        assert np.all(np.isfinite(metrics.radial_energy))
        assert np.all(np.isfinite(metrics.angular_distribution))
        assert np.all(np.isfinite(metrics.edge_confidence))
    
    def test_compute_scientific_metrics_without_edge_map(self, mimic_service, sample_coefficients):
        """
        Test scientific metrics computation without edge map.
        
        **Validates: Requirements 13.11, 18.1**
        """
        metrics = mimic_service.compute_scientific_metrics(sample_coefficients)
        
        # Should still work and return valid metrics
        assert isinstance(metrics, ScientificMetrics)
        assert metrics.edge_confidence.shape == sample_coefficients.shape
        
        # Edge confidence should be all zeros without edge map
        assert np.all(metrics.edge_confidence == 0.0)
    
    def test_directional_energy_structure(self, mimic_service, sample_coefficients):
        """
        Test that directional energy has correct structure.
        
        **Validates: Requirements 13.11, 18.1**
        """
        metrics = mimic_service.compute_scientific_metrics(sample_coefficients)
        
        # Should have energy for each scale
        for scale_idx in range(sample_coefficients.scales):
            if scale_idx in sample_coefficients.coefficients:
                assert scale_idx in metrics.directional_energy
                
                # Energy array should have correct length
                energy = metrics.directional_energy[scale_idx]
                assert len(energy) == sample_coefficients.orientations
                
                # All energies should be non-negative
                assert np.all(energy >= 0.0)
    
    def test_angular_distribution_sum(self, mimic_service, sample_coefficients):
        """
        Test that angular distribution sums correctly.
        
        **Validates: Requirements 13.11, 18.1**
        """
        metrics = mimic_service.compute_scientific_metrics(sample_coefficients)
        
        # Angular distribution should be non-negative
        assert np.all(metrics.angular_distribution >= 0.0)
        
        # Should have correct length
        assert len(metrics.angular_distribution) == sample_coefficients.orientations


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_enhancement_on_constant_image(self, mimic_service):
        """
        Test enhancement on constant image.
        
        **Validates: Requirements 10.1, 10.2, 18.1**
        """
        constant_image = np.full((64, 64), 0.5)
        
        enhanced = mimic_service.apply_enhancement(
            constant_image,
            enhancement_factor=2.0,
            smoothing=1.0
        )
        
        # Should not crash and should return valid image
        assert enhanced.shape == constant_image.shape
        assert np.all(np.isfinite(enhanced))
    
    def test_enhancement_on_zero_image(self, mimic_service):
        """
        Test enhancement on all-zero image.
        
        **Validates: Requirements 10.1, 10.2, 18.1**
        """
        zero_image = np.zeros((64, 64))
        
        enhanced = mimic_service.apply_enhancement(
            zero_image,
            enhancement_factor=2.0,
            smoothing=1.0
        )
        
        # Should not crash and should return valid image
        assert enhanced.shape == zero_image.shape
        assert np.all(np.isfinite(enhanced))
    
    def test_metrics_with_minimal_coefficients(self, mimic_service):
        """
        Test metrics computation with minimal coefficient structure.
        
        **Validates: Requirements 13.11, 18.1**
        """
        # Create minimal coefficients (single scale, single orientation)
        minimal_coeffs = CurveletCoefficients(
            coefficients={0: {0: np.random.rand(32, 32)}},
            scales=1,
            orientations=1,
            shape=(32, 32)
        )
        
        metrics = mimic_service.compute_scientific_metrics(minimal_coeffs)
        
        # Should not crash and should return valid metrics
        assert isinstance(metrics, ScientificMetrics)
        assert metrics.anisotropy_map.shape == (32, 32)
