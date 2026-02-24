"""
Unit tests for Task 8.4: Reconstruction and Edge Detection

This test file specifically validates the requirements for task 8.4:
- Test reconstruction error computation (IMPLEMENTED - can test)
- Test edge detection produces binary edge map (PENDING task 8.2 - basic tests only)
- Test difference map generation (PENDING task 8.2 - concept tests only)

**Validates: Requirements 18.1, 18.2, 18.3**

Note: Some tests are simplified because task 8.2 (implementing edge detection in
MIMICService) has not been completed yet. These tests validate the underlying
concepts and will serve as integration tests once task 8.2 is complete.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from app.services.transforms.wavelet import WaveletTransform
from app.services.transforms.curvelet import CurveletTransform
from app.services.mimic_service import MIMICService


class TestReconstructionErrorComputation:
    """
    Test reconstruction error computation for both wavelet and curvelet transforms.
    
    **Validates: Requirement 18.1** - Test reconstruction error computation
    """
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a 128x128 image with a square pattern
        image = np.zeros((128, 128), dtype=np.float64)
        image[32:96, 32:96] = 1.0
        return image
    
    def test_wavelet_reconstruction_error_is_computed(self, sample_image):
        """
        Test that wavelet reconstruction error is computed correctly.
        
        **Validates: Requirement 18.1**
        """
        wt = WaveletTransform()
        
        # Decompose and reconstruct
        coeffs = wt.decompose(sample_image, levels=3)
        reconstructed = wt.reconstruct(coeffs)
        
        # Crop to match original size
        reconstructed = reconstructed[:sample_image.shape[0], :sample_image.shape[1]]
        
        # Compute error metrics
        error_metrics = wt.compute_reconstruction_error(sample_image, reconstructed)
        
        # Verify all required metrics are present
        assert 'mse' in error_metrics, "MSE metric missing"
        assert 'rmse' in error_metrics, "RMSE metric missing"
        assert 'mae' in error_metrics, "MAE metric missing"
        assert 'psnr' in error_metrics, "PSNR metric missing"
        assert 'max_error' in error_metrics, "Max error metric missing"
        assert 'error_map' in error_metrics, "Error map missing"
        
        # Verify metrics are valid
        assert error_metrics['mse'] >= 0, "MSE should be non-negative"
        assert error_metrics['rmse'] >= 0, "RMSE should be non-negative"
        assert error_metrics['mae'] >= 0, "MAE should be non-negative"
        assert error_metrics['psnr'] > 0, "PSNR should be positive"
        assert error_metrics['max_error'] >= 0, "Max error should be non-negative"
        
        # Verify error map shape matches input
        assert error_metrics['error_map'].shape == sample_image.shape, \
            "Error map shape should match input image shape"
    
    def test_curvelet_reconstruction_error_is_computed(self, sample_image):
        """
        Test that curvelet reconstruction error is computed correctly.
        
        **Validates: Requirement 18.1**
        """
        ct = CurveletTransform()
        
        # Decompose and reconstruct
        coeffs = ct.decompose(sample_image, levels=3, angular_resolution=8)
        reconstructed = ct.reconstruct(coeffs)
        
        # Ensure same shape
        reconstructed = reconstructed[:sample_image.shape[0], :sample_image.shape[1]]
        
        # Compute error metrics
        error_metrics = ct.compute_reconstruction_error(sample_image, reconstructed)
        
        # Verify all required metrics are present
        assert 'mse' in error_metrics, "MSE metric missing"
        assert 'rmse' in error_metrics, "RMSE metric missing"
        assert 'mae' in error_metrics, "MAE metric missing"
        assert 'psnr' in error_metrics, "PSNR metric missing"
        assert 'max_error' in error_metrics, "Max error metric missing"
        assert 'error_map' in error_metrics, "Error map missing"
        
        # Verify metrics are valid
        assert error_metrics['mse'] >= 0, "MSE should be non-negative"
        assert error_metrics['rmse'] >= 0, "RMSE should be non-negative"
        assert error_metrics['mae'] >= 0, "MAE should be non-negative"
        assert error_metrics['psnr'] > 0, "PSNR should be positive"
        assert error_metrics['max_error'] >= 0, "Max error should be non-negative"
        
        # Verify error map shape matches input
        assert error_metrics['error_map'].shape == sample_image.shape, \
            "Error map shape should match input image shape"
    
    def test_reconstruction_error_with_perfect_match(self):
        """
        Test that reconstruction error is zero for identical images.
        
        **Validates: Requirement 18.1**
        """
        image = np.random.rand(128, 128)
        
        wt = WaveletTransform()
        error_metrics = wt.compute_reconstruction_error(image, image)
        
        # Perfect match should have zero error
        assert error_metrics['mse'] == 0.0, "MSE should be zero for identical images"
        assert error_metrics['rmse'] == 0.0, "RMSE should be zero for identical images"
        assert error_metrics['mae'] == 0.0, "MAE should be zero for identical images"
        assert error_metrics['max_error'] == 0.0, "Max error should be zero for identical images"
        assert error_metrics['psnr'] == float('inf'), "PSNR should be infinite for identical images"
    
    def test_reconstruction_error_increases_with_noise(self, sample_image):
        """
        Test that reconstruction error increases with added noise.
        
        **Validates: Requirement 18.1**
        """
        wt = WaveletTransform()
        
        # Create reconstructions with different noise levels
        reconstructed_low_noise = sample_image + np.random.randn(*sample_image.shape) * 0.01
        reconstructed_high_noise = sample_image + np.random.randn(*sample_image.shape) * 0.1
        
        # Compute errors
        error_low = wt.compute_reconstruction_error(sample_image, reconstructed_low_noise)
        error_high = wt.compute_reconstruction_error(sample_image, reconstructed_high_noise)
        
        # Higher noise should produce higher error
        assert error_high['mse'] > error_low['mse'], \
            "Higher noise should produce higher MSE"
        assert error_high['rmse'] > error_low['rmse'], \
            "Higher noise should produce higher RMSE"


class TestEdgeDetectionBinaryMap:
    """
    Test that edge detection produces binary edge maps.
    
    **Validates: Requirement 18.2** - Test edge detection produces binary edge map
    
    Note: These tests use the WaveletTransform.extract_edges method which IS implemented.
    CurveletTransform.extract_edges and MIMICService edge detection methods are part
    of task 8.2 and not yet implemented.
    """
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image with clear edges."""
        image = np.zeros((128, 128), dtype=np.float64)
        image[32:96, 32:96] = 1.0
        return image
    
    @pytest.fixture
    def wavelet_transform(self):
        """Create a WaveletTransform instance."""
        return WaveletTransform()
    
    def test_wavelet_edge_detection_produces_binary_map(
        self, sample_image, wavelet_transform
    ):
        """
        Test that wavelet edge detection produces a binary edge map.
        
        **Validates: Requirement 18.2**
        """
        # Decompose
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        
        # Extract edges
        edges = wavelet_transform.extract_edges(coeffs, threshold=0.5)
        
        # Verify output is 2D
        assert edges.ndim == 2, "Edge map should be 2D array"
        
        # Verify output is binary (only 0 and 1)
        unique_values = np.unique(edges)
        assert np.all((unique_values == 0) | (unique_values == 1)), \
            f"Edge map should be binary, got values: {unique_values}"
        
        # Verify output is integer type
        assert edges.dtype in [np.uint8, np.int32, np.int64, bool], \
            f"Edge map should be integer or bool type, got {edges.dtype}"
        
        # Verify some edges were detected
        assert np.sum(edges) > 0, "Should detect at least some edges"
    
    def test_edge_detection_threshold_affects_output(
        self, sample_image, wavelet_transform
    ):
        """
        Test that edge strength threshold affects edge detection output.
        
        **Validates: Requirement 18.2**
        """
        # Decompose
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        
        # Lower threshold should detect more edges
        edges_low = wavelet_transform.extract_edges(coeffs, threshold=0.2)
        
        # Higher threshold should detect fewer edges
        edges_high = wavelet_transform.extract_edges(coeffs, threshold=0.8)
        
        # Verify both are binary
        assert np.all((edges_low == 0) | (edges_low == 1)), "Low threshold edges should be binary"
        assert np.all((edges_high == 0) | (edges_high == 1)), "High threshold edges should be binary"
        
        # Lower threshold should produce more or equal edge pixels
        assert np.sum(edges_low) >= np.sum(edges_high), \
            "Lower threshold should detect more or equal edges"
    
    def test_edge_detection_returns_consistent_type(
        self, sample_image, wavelet_transform
    ):
        """
        Test that edge detection always returns consistent data type.
        
        **Validates: Requirement 18.2**
        """
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        
        # Run edge detection multiple times
        edges1 = wavelet_transform.extract_edges(coeffs, threshold=0.5)
        edges2 = wavelet_transform.extract_edges(coeffs, threshold=0.5)
        
        # Should return same type
        assert edges1.dtype == edges2.dtype, "Edge detection should return consistent type"
        
        # Should be deterministic
        assert np.array_equal(edges1, edges2), "Edge detection should be deterministic"


class TestDifferenceMapGeneration:
    """
    Test difference map generation between wavelet and curvelet edge detection.
    
    **Validates: Requirement 18.3** - Test difference map generation
    
    Note: These tests implement a simple difference map function to test the concept,
    since the MIMICService.generate_difference_map method is part of task 8.2.
    """
    
    def generate_difference_map_simple(self, wavelet_edges, curvelet_edges):
        """
        Simple difference map implementation for testing.
        
        Returns a map with values:
        - 0: No edges in either
        - 1: Wavelet only
        - 2: Curvelet only
        - 3: Both
        """
        # Ensure same shape
        if wavelet_edges.shape != curvelet_edges.shape:
            raise ValueError("Edge maps must have same shape")
        
        # Convert to binary if needed
        wavelet_binary = (wavelet_edges > 0).astype(bool)
        curvelet_binary = (curvelet_edges > 0).astype(bool)
        
        # Create difference map
        diff_map = np.zeros_like(wavelet_binary, dtype=np.uint8)
        
        # Set values based on edge presence
        # Both have edges
        both_mask = wavelet_binary & curvelet_binary
        diff_map[both_mask] = 3
        
        # Only wavelet has edges
        wavelet_only_mask = wavelet_binary & ~curvelet_binary
        diff_map[wavelet_only_mask] = 1
        
        # Only curvelet has edges
        curvelet_only_mask = ~wavelet_binary & curvelet_binary
        diff_map[curvelet_only_mask] = 2
        
        # Neither has edges (already 0)
        
        return diff_map
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        image = np.zeros((128, 128), dtype=np.float64)
        image[32:96, 32:96] = 1.0
        return image
    
    @pytest.fixture
    def wavelet_edges(self, sample_image):
        """Generate wavelet edge map."""
        wt = WaveletTransform()
        coeffs = wt.decompose(sample_image, levels=3)
        return wt.extract_edges(coeffs, threshold=0.5)
    
    @pytest.fixture
    def curvelet_edges_mock(self, wavelet_edges):
        """
        Generate a mock curvelet edge map for testing.
        
        Since CurveletTransform.extract_edges is not yet implemented (task 8.2),
        we create a mock edge map with the same shape as wavelet edges for testing
        the difference map logic.
        """
        # Create a simple edge map with same shape as wavelet edges
        edges = np.zeros_like(wavelet_edges, dtype=np.uint8)
        # Add edges at different locations (slightly different pattern)
        h, w = edges.shape
        edges[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = 1
        edges[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)] = 0  # Inner region no edges
        return edges
    
    def test_difference_map_is_generated(
        self, wavelet_edges, curvelet_edges_mock
    ):
        """
        Test that difference map is successfully generated.
        
        **Validates: Requirement 18.3**
        """
        diff_map = self.generate_difference_map_simple(
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges_mock
        )
        
        # Verify output exists and is 2D
        assert diff_map is not None, "Difference map should be generated"
        assert diff_map.ndim == 2, "Difference map should be 2D array"
        
        # Verify output type
        assert diff_map.dtype == np.uint8, \
            f"Difference map should be uint8, got {diff_map.dtype}"
    
    def test_difference_map_has_valid_values(
        self, wavelet_edges, curvelet_edges_mock
    ):
        """
        Test that difference map contains valid values.
        
        **Validates: Requirement 18.3**
        """
        diff_map = self.generate_difference_map_simple(
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges_mock
        )
        
        # Difference map should have values 0-3
        # 0: No edges, 1: Wavelet only, 2: Curvelet only, 3: Both
        unique_values = np.unique(diff_map)
        assert np.all(unique_values >= 0), "Difference map values should be >= 0"
        assert np.all(unique_values <= 3), "Difference map values should be <= 3"
    
    def test_difference_map_matches_edge_dimensions(
        self, wavelet_edges, curvelet_edges_mock
    ):
        """
        Test that difference map has correct dimensions.
        
        **Validates: Requirement 18.3**
        """
        diff_map = self.generate_difference_map_simple(
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges_mock
        )
        
        # Difference map should match edge map dimensions
        assert diff_map.shape == wavelet_edges.shape, \
            "Difference map should match wavelet edges shape"
        assert diff_map.shape == curvelet_edges_mock.shape, \
            "Difference map should match curvelet edges shape"
    
    def test_difference_map_handles_identical_edges(self):
        """
        Test difference map when both edge maps are identical.
        
        **Validates: Requirement 18.3**
        """
        # Create identical edge maps
        edges = np.zeros((64, 64), dtype=np.uint8)
        edges[16:48, 16:48] = 1
        
        diff_map = self.generate_difference_map_simple(
            wavelet_edges=edges,
            curvelet_edges=edges
        )
        
        # When edges are identical, difference map should show:
        # - 0 where both have no edges
        # - 3 where both have edges
        # - No 1 or 2 values (no exclusive edges)
        unique_values = np.unique(diff_map)
        assert 1 not in unique_values, "Should have no wavelet-only edges"
        assert 2 not in unique_values, "Should have no curvelet-only edges"
        assert 0 in unique_values or 3 in unique_values, \
            "Should have either no-edge or both-edge regions"
    
    def test_difference_map_handles_disjoint_edges(self):
        """
        Test difference map when edge maps are completely different.
        
        **Validates: Requirement 18.3**
        """
        # Create disjoint edge maps
        wavelet_edges = np.zeros((64, 64), dtype=np.uint8)
        wavelet_edges[16:32, 16:32] = 1  # Top-left quadrant
        
        curvelet_edges = np.zeros((64, 64), dtype=np.uint8)
        curvelet_edges[32:48, 32:48] = 1  # Bottom-right quadrant
        
        diff_map = self.generate_difference_map_simple(
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges
        )
        
        # Should have wavelet-only (1) and curvelet-only (2) regions
        unique_values = np.unique(diff_map)
        assert 1 in unique_values, "Should have wavelet-only edges"
        assert 2 in unique_values, "Should have curvelet-only edges"
        assert 3 not in unique_values, "Should have no overlapping edges"
    
    def test_difference_map_logic_correctness(self):
        """
        Test that difference map logic is correct for all cases.
        
        **Validates: Requirement 18.3**
        """
        # Create test edge maps with all four cases
        wavelet_edges = np.array([
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1]
        ], dtype=np.uint8)
        
        curvelet_edges = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=np.uint8)
        
        diff_map = self.generate_difference_map_simple(wavelet_edges, curvelet_edges)
        
        # Check each case
        assert diff_map[0, 0] == 0, "Neither edge should be 0"
        assert diff_map[0, 1] == 1, "Wavelet only should be 1"
        assert diff_map[0, 2] == 2, "Curvelet only should be 2"
        assert diff_map[0, 3] == 3, "Both edges should be 3"


class TestIntegrationReconstructionAndEdgeDetection:
    """
    Integration tests for complete reconstruction and edge detection workflow.
    
    **Validates: Requirements 18.1, 18.2, 18.3**
    """
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image with clear features."""
        # Create image with square and gradient
        image = np.zeros((128, 128), dtype=np.float64)
        image[32:96, 32:96] = 1.0
        # Add gradient
        gradient = np.linspace(0, 0.5, 128).reshape(1, -1)
        image = image + gradient
        return image
    
    def test_complete_workflow_wavelet(self, sample_image):
        """
        Test complete workflow: decompose -> reconstruct -> compute error -> detect edges.
        
        **Validates: Requirements 18.1, 18.2**
        """
        wt = WaveletTransform()
        
        # 1. Decompose
        coeffs = wt.decompose(sample_image, levels=3)
        assert coeffs is not None, "Decomposition should succeed"
        
        # 2. Reconstruct
        reconstructed = wt.reconstruct(coeffs)
        reconstructed = reconstructed[:sample_image.shape[0], :sample_image.shape[1]]
        assert reconstructed.shape == sample_image.shape, "Reconstruction should preserve shape"
        
        # 3. Compute error
        error_metrics = wt.compute_reconstruction_error(sample_image, reconstructed)
        assert error_metrics['rmse'] < 0.01, "Reconstruction error should be small"
        
        # 4. Detect edges
        edges = wt.extract_edges(coeffs, threshold=0.5)
        assert np.all((edges == 0) | (edges == 1) | (edges == True) | (edges == False)), \
            "Edges should be binary"
        assert np.sum(edges) > 0, "Should detect some edges"
    
    def test_complete_workflow_curvelet_reconstruction_only(self, sample_image):
        """
        Test curvelet workflow for reconstruction (edge detection not yet implemented).
        
        **Validates: Requirement 18.1**
        """
        ct = CurveletTransform()
        
        # 1. Decompose
        coeffs = ct.decompose(sample_image, levels=3, angular_resolution=8)
        assert coeffs is not None, "Decomposition should succeed"
        
        # 2. Reconstruct
        reconstructed = ct.reconstruct(coeffs)
        reconstructed = reconstructed[:sample_image.shape[0], :sample_image.shape[1]]
        assert reconstructed.shape == sample_image.shape, "Reconstruction should preserve shape"
        
        # 3. Compute error
        error_metrics = ct.compute_reconstruction_error(sample_image, reconstructed)
        assert error_metrics['rmse'] < 1.0, "Reconstruction error should be reasonable"
        
        # Note: Edge detection for curvelet is part of task 8.2 and not yet implemented
    
    def test_complete_workflow_with_difference_map_concept(self, sample_image):
        """
        Test complete workflow including difference map generation concept.
        
        **Validates: Requirements 18.1, 18.2, 18.3**
        """
        wt = WaveletTransform()
        ct = CurveletTransform()
        
        # Process with both transforms
        wavelet_coeffs = wt.decompose(sample_image, levels=3)
        curvelet_coeffs = ct.decompose(sample_image, levels=3, angular_resolution=8)
        
        # Reconstruct and compute errors
        wavelet_recon = wt.reconstruct(wavelet_coeffs)
        wavelet_recon = wavelet_recon[:sample_image.shape[0], :sample_image.shape[1]]
        wavelet_error = wt.compute_reconstruction_error(sample_image, wavelet_recon)
        
        curvelet_recon = ct.reconstruct(curvelet_coeffs)
        curvelet_recon = curvelet_recon[:sample_image.shape[0], :sample_image.shape[1]]
        curvelet_error = ct.compute_reconstruction_error(sample_image, curvelet_recon)
        
        # Both should have valid errors
        assert wavelet_error['rmse'] >= 0, "Wavelet error should be non-negative"
        assert curvelet_error['rmse'] >= 0, "Curvelet error should be non-negative"
        
        # Detect edges from wavelet
        wavelet_edges = wt.extract_edges(wavelet_coeffs, threshold=0.5)
        
        # Create mock curvelet edges for testing difference map concept
        # (actual curvelet edge detection is part of task 8.2)
        # Use same shape as wavelet edges
        curvelet_edges_mock = np.zeros_like(wavelet_edges, dtype=np.uint8)
        h, w = curvelet_edges_mock.shape
        curvelet_edges_mock[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = 1
        
        # Both should be binary
        wavelet_binary = (wavelet_edges > 0).astype(np.uint8)
        curvelet_binary = (curvelet_edges_mock > 0).astype(np.uint8)
        assert np.all((wavelet_binary == 0) | (wavelet_binary == 1)), "Wavelet edges should be binary"
        assert np.all((curvelet_binary == 0) | (curvelet_binary == 1)), "Curvelet edges should be binary"
        
        # Generate difference map using simple logic
        diff_map = np.zeros_like(wavelet_binary, dtype=np.uint8)
        both_mask = (wavelet_binary > 0) & (curvelet_binary > 0)
        wavelet_only_mask = (wavelet_binary > 0) & (curvelet_binary == 0)
        curvelet_only_mask = (wavelet_binary == 0) & (curvelet_binary > 0)
        
        diff_map[both_mask] = 3  # Both
        diff_map[wavelet_only_mask] = 1  # Wavelet only
        diff_map[curvelet_only_mask] = 2  # Curvelet only
        
        # Difference map should be valid
        assert diff_map.ndim == 2, "Difference map should be 2D"
        assert np.all(diff_map >= 0) and np.all(diff_map <= 3), \
            "Difference map should have values 0-3"
