"""
Unit tests for noise modeling functionality in MIMIC service.

Tests validate Requirements 4.1, 4.2, 4.3, 4.4, 4.5:
- Background noise level estimation
- Noise standard deviation computation
- Threshold-based noise filtering
- Noise model visualization generation
- Noise statistics in metadata
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from app.services.mimic_service import MIMICService


@pytest.fixture
def mimic_service():
    """Fixture to provide MIMICService instance."""
    return MIMICService()


@pytest.fixture
def test_image_with_noise():
    """
    Fixture to provide a test image with known noise characteristics.
    
    Creates a 128x128 image with:
    - Background noise at ~0.1 intensity
    - Signal regions at higher intensities
    """
    np.random.seed(42)
    
    # Create background with Gaussian noise
    image = np.random.normal(0.1, 0.02, (128, 128))
    
    # Add signal regions
    image[30:60, 30:60] = np.random.normal(0.6, 0.05, (30, 30))
    image[70:100, 70:100] = np.random.normal(0.8, 0.05, (30, 30))
    
    # Clip to valid range
    image = np.clip(image, 0.0, 1.0)
    
    return image


@pytest.fixture
def temp_output_dir():
    """Fixture to provide temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir)


class TestNoiseEstimation:
    """Tests for noise estimation functionality (Requirements 4.1, 4.2)."""
    
    def test_noise_level_estimation(self, mimic_service, test_image_with_noise):
        """
        Test that background noise level is estimated correctly.
        Validates Requirement 4.1.
        """
        noise_stats = mimic_service.estimate_noise(test_image_with_noise)
        
        # Noise level should be non-negative
        assert noise_stats['noise_level'] >= 0
        
        # For our test image, noise level should be around 0.1 (background mean)
        assert 0.05 <= noise_stats['noise_level'] <= 0.15
        
        # Should be included in metadata
        assert 'noise_level' in noise_stats
    
    def test_noise_std_computation(self, mimic_service, test_image_with_noise):
        """
        Test that noise standard deviation is computed correctly.
        Validates Requirement 4.2.
        """
        noise_stats = mimic_service.estimate_noise(test_image_with_noise)
        
        # Noise std should be non-negative and finite
        assert noise_stats['noise_std'] >= 0
        assert np.isfinite(noise_stats['noise_std'])
        
        # For our test image, noise std should be around 0.02
        assert 0.01 <= noise_stats['noise_std'] <= 0.05
        
        # Should be included in metadata
        assert 'noise_std' in noise_stats
    
    def test_noise_statistics_in_metadata(self, mimic_service, test_image_with_noise):
        """
        Test that noise statistics are included in processing metadata.
        Validates Requirement 4.5.
        """
        noise_stats = mimic_service.estimate_noise(test_image_with_noise)
        
        # All required statistics should be present
        required_keys = [
            'noise_level',
            'noise_std',
            'background_median',
            'background_mad',
            'noise_pixels_fraction'
        ]
        
        for key in required_keys:
            assert key in noise_stats, f"Missing required statistic: {key}"
            assert isinstance(noise_stats[key], (int, float)), \
                f"Statistic {key} should be numeric"
    
    def test_constant_image_noise_estimation(self, mimic_service):
        """Test noise estimation on constant image (edge case)."""
        constant_image = np.ones((100, 100)) * 0.5
        
        noise_stats = mimic_service.estimate_noise(constant_image)
        
        # For constant image, noise std should be very small or zero
        assert noise_stats['noise_std'] < 0.01
        assert noise_stats['noise_level'] >= 0


class TestThresholdFiltering:
    """Tests for threshold-based noise filtering (Requirement 4.3)."""
    
    def test_threshold_filtering_applied(self, mimic_service, test_image_with_noise):
        """
        Test that threshold-based noise filtering is applied when threshold provided.
        Validates Requirement 4.3.
        """
        threshold = 0.3
        noise_stats = mimic_service.estimate_noise(
            test_image_with_noise,
            photon_threshold=threshold
        )
        
        # Filtered image should be present
        assert noise_stats['filtered_image'] is not None
        assert isinstance(noise_stats['filtered_image'], np.ndarray)
        assert noise_stats['filtered_image'].shape == test_image_with_noise.shape
    
    def test_threshold_zeros_below_threshold(self, mimic_service, test_image_with_noise):
        """
        Test that pixels below threshold are zeroed.
        Validates Requirement 4.3.
        """
        threshold = 0.3
        noise_stats = mimic_service.estimate_noise(
            test_image_with_noise,
            photon_threshold=threshold
        )
        
        filtered = noise_stats['filtered_image']
        
        # All pixels below threshold should be zero
        below_threshold = test_image_with_noise < threshold
        assert np.all(filtered[below_threshold] == 0.0)
    
    def test_threshold_preserves_above_threshold(self, mimic_service, test_image_with_noise):
        """
        Test that pixels above threshold are preserved.
        Validates Requirement 4.3.
        """
        threshold = 0.3
        noise_stats = mimic_service.estimate_noise(
            test_image_with_noise,
            photon_threshold=threshold
        )
        
        filtered = noise_stats['filtered_image']
        
        # All pixels above threshold should be preserved
        above_threshold = test_image_with_noise >= threshold
        assert np.allclose(
            filtered[above_threshold],
            test_image_with_noise[above_threshold]
        )
    
    def test_threshold_normalization(self, mimic_service, test_image_with_noise):
        """Test that threshold values > 1.0 are normalized to [0, 1] range."""
        # Provide threshold in percentage (0-100)
        threshold_percent = 30.0  # 30%
        noise_stats = mimic_service.estimate_noise(
            test_image_with_noise,
            photon_threshold=threshold_percent
        )
        
        filtered = noise_stats['filtered_image']
        
        # Should be equivalent to threshold of 0.3
        expected_threshold = 0.3
        below_threshold = test_image_with_noise < expected_threshold
        assert np.all(filtered[below_threshold] == 0.0)
    
    def test_no_filtering_without_threshold(self, mimic_service, test_image_with_noise):
        """Test that no filtering is applied when threshold not provided."""
        noise_stats = mimic_service.estimate_noise(test_image_with_noise)
        
        # Filtered image should be None
        assert noise_stats['filtered_image'] is None


class TestNoiseVisualization:
    """Tests for noise model visualization generation (Requirement 4.4)."""
    
    def test_visualization_file_created(
        self, 
        mimic_service, 
        test_image_with_noise, 
        temp_output_dir
    ):
        """
        Test that noise_model.png visualization is generated.
        Validates Requirement 4.4.
        """
        output_path = temp_output_dir / "noise_model.png"
        
        noise_stats = mimic_service.estimate_noise(
            test_image_with_noise,
            output_path=str(output_path)
        )
        
        # Visualization file should exist
        assert output_path.exists()
        assert output_path.is_file()
        
        # File should not be empty
        assert output_path.stat().st_size > 0
    
    def test_visualization_with_threshold(
        self, 
        mimic_service, 
        test_image_with_noise, 
        temp_output_dir
    ):
        """Test that visualization includes filtered image when threshold provided."""
        output_path = temp_output_dir / "noise_model_filtered.png"
        
        noise_stats = mimic_service.estimate_noise(
            test_image_with_noise,
            photon_threshold=0.3,
            output_path=str(output_path)
        )
        
        # Visualization file should exist
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_visualization_creates_output_directory(
        self, 
        mimic_service, 
        test_image_with_noise, 
        temp_output_dir
    ):
        """Test that output directory is created if it doesn't exist."""
        nested_dir = temp_output_dir / "nested" / "path"
        output_path = nested_dir / "noise_model.png"
        
        # Directory should not exist yet
        assert not nested_dir.exists()
        
        noise_stats = mimic_service.estimate_noise(
            test_image_with_noise,
            output_path=str(output_path)
        )
        
        # Directory should be created
        assert nested_dir.exists()
        assert output_path.exists()


class TestNoiseModelingIntegration:
    """Integration tests for complete noise modeling workflow."""
    
    def test_complete_noise_modeling_workflow(
        self, 
        mimic_service, 
        test_image_with_noise, 
        temp_output_dir
    ):
        """
        Test complete noise modeling workflow with all features.
        Validates Requirements 4.1, 4.2, 4.3, 4.4, 4.5.
        """
        output_path = temp_output_dir / "noise_model_complete.png"
        
        # Run complete noise modeling
        noise_stats = mimic_service.estimate_noise(
            test_image_with_noise,
            photon_threshold=0.25,
            output_path=str(output_path)
        )
        
        # Verify all requirements
        # Requirement 4.1: Background noise level estimated
        assert 'noise_level' in noise_stats
        assert noise_stats['noise_level'] >= 0
        
        # Requirement 4.2: Noise standard deviation computed
        assert 'noise_std' in noise_stats
        assert noise_stats['noise_std'] >= 0
        
        # Requirement 4.3: Threshold-based filtering applied
        assert noise_stats['filtered_image'] is not None
        assert np.all(noise_stats['filtered_image'] >= 0)
        
        # Requirement 4.4: Visualization generated
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Requirement 4.5: Noise statistics in metadata
        required_stats = [
            'noise_level', 'noise_std', 'background_median',
            'background_mad', 'noise_pixels_fraction'
        ]
        for stat in required_stats:
            assert stat in noise_stats
    
    def test_noise_mask_generation(self, mimic_service, test_image_with_noise):
        """Test that noise mask is generated correctly."""
        noise_stats = mimic_service.estimate_noise(test_image_with_noise)
        
        # Noise mask should be present
        assert 'noise_mask' in noise_stats
        assert isinstance(noise_stats['noise_mask'], np.ndarray)
        assert noise_stats['noise_mask'].dtype == bool
        assert noise_stats['noise_mask'].shape == test_image_with_noise.shape
        
        # Noise pixels fraction should match mask
        expected_fraction = np.sum(noise_stats['noise_mask']) / noise_stats['noise_mask'].size
        assert np.isclose(noise_stats['noise_pixels_fraction'], expected_fraction)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_image(self, mimic_service):
        """Test noise estimation on all-zero image."""
        zero_image = np.zeros((50, 50))
        
        noise_stats = mimic_service.estimate_noise(zero_image)
        
        # Should handle gracefully
        assert noise_stats['noise_level'] == 0.0
        assert noise_stats['noise_std'] >= 0
    
    def test_high_noise_image(self, mimic_service):
        """Test noise estimation on very noisy image."""
        np.random.seed(123)
        noisy_image = np.random.uniform(0, 1, (100, 100))
        
        noise_stats = mimic_service.estimate_noise(noisy_image)
        
        # Should still produce valid statistics
        assert 0 <= noise_stats['noise_level'] <= 1
        assert noise_stats['noise_std'] >= 0
    
    def test_small_image(self, mimic_service):
        """Test noise estimation on very small image."""
        small_image = np.random.rand(10, 10)
        
        noise_stats = mimic_service.estimate_noise(small_image)
        
        # Should handle small images
        assert noise_stats['noise_level'] >= 0
        assert noise_stats['noise_std'] >= 0
