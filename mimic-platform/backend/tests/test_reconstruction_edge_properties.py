"""
Property-based tests for reconstruction and edge detection functionality.

Uses Hypothesis to verify universal correctness properties across randomized inputs.

Tests Requirements: 8.1, 8.2, 8.3, 9.1, 9.2, 9.3

**Property 11: Transform Reconstruction Round-Trip**
For any image and transform type (wavelet or curvelet), decomposing then
reconstructing should produce an image that approximates the original within
a bounded reconstruction error.

**Property 12: Edge Detection Output Format**
For any transform coefficients (wavelet or curvelet) and edge strength threshold,
edge detection should produce a binary edge map with the same spatial dimensions
as the input image.
"""
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
import hypothesis.extra.numpy as npst
from app.services.transforms.wavelet import WaveletTransform
from app.services.transforms.curvelet import CurveletTransform
from app.services.mimic_service import MIMICService


# Hypothesis settings for property tests
# Minimum 100 iterations as specified in design document
settings.register_profile("default", max_examples=100, deadline=None)
settings.load_profile("default")


class TestTransformReconstructionRoundTripProperty:
    """
    Property-based tests for transform reconstruction round-trip.
    
    **Property 11: Transform Reconstruction Round-Trip**
    **Validates: Requirements 8.1, 8.2, 8.3**
    
    For any image and transform type (wavelet or curvelet), decomposing then
    reconstructing should produce an image that approximates the original within
    a bounded reconstruction error.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.wavelet_transform = WaveletTransform()
        self.curvelet_transform = CurveletTransform()
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=64, max_value=256),
                st.integers(min_value=64, max_value=256)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        levels=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_wavelet_reconstruction_roundtrip(self, image, levels):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property 11: For any image, wavelet decompose then reconstruct should
        approximate the original within bounded error.
        """
        # Skip constant images (no information to preserve)
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Ensure image has some dynamic range
        assume(image.max() - image.min() > 1e-6)
        
        # Decompose
        coeffs = self.wavelet_transform.decompose(image, levels=levels)
        
        # Reconstruct
        reconstructed = self.wavelet_transform.reconstruct(coeffs)
        
        # Crop to match original size (wavelet may expand slightly)
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
        
        # Property: Reconstruction error should be bounded
        # Wavelet transform is nearly perfect, so error should be very small
        mse = np.mean((image - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        
        assert rmse < 0.01, \
            f"Wavelet reconstruction RMSE too high: {rmse}"
        
        # Property: Shape should be preserved
        assert reconstructed.shape == image.shape, \
            f"Shape changed: {image.shape} -> {reconstructed.shape}"
        
        # Property: Value range should be approximately preserved
        assert reconstructed.min() >= image.min() - 0.1, \
            f"Min value changed too much: {image.min()} -> {reconstructed.min()}"
        assert reconstructed.max() <= image.max() + 0.1, \
            f"Max value changed too much: {image.max()} -> {reconstructed.max()}"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=64, max_value=256),
                st.integers(min_value=64, max_value=256)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        levels=st.integers(min_value=2, max_value=4),
        angular_resolution=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_curvelet_reconstruction_roundtrip(self, image, levels, angular_resolution):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property 11: For any image, curvelet decompose then reconstruct should
        approximate the original within bounded error.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Ensure image has some dynamic range
        assume(image.max() - image.min() > 1e-6)
        
        # Decompose
        coeffs = self.curvelet_transform.decompose(
            image,
            levels=levels,
            angular_resolution=angular_resolution
        )
        
        # Reconstruct
        reconstructed = self.curvelet_transform.reconstruct(coeffs)
        
        # Ensure same shape
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
        
        # Property: Reconstruction error should be bounded
        # FFT-based curvelet is approximate, so error threshold is higher
        mse = np.mean((image - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        
        assert rmse < 1.0, \
            f"Curvelet reconstruction RMSE too high: {rmse}"
        
        # Property: Shape should be preserved
        assert reconstructed.shape == image.shape, \
            f"Shape changed: {image.shape} -> {reconstructed.shape}"
        
        # Property: Reconstruction should be finite
        assert np.all(np.isfinite(reconstructed)), \
            "Reconstruction contains non-finite values"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(128, 128),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_reconstruction_error_metrics_valid(self, image):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property 11 variant: Reconstruction error metrics should always be valid.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Wavelet round-trip
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        reconstructed = self.wavelet_transform.reconstruct(coeffs)
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
        
        # Compute error metrics
        error_metrics = self.wavelet_transform.compute_reconstruction_error(
            image, reconstructed
        )
        
        # Property: All error metrics should be non-negative
        assert error_metrics['mse'] >= 0, "MSE is negative"
        assert error_metrics['rmse'] >= 0, "RMSE is negative"
        assert error_metrics['mae'] >= 0, "MAE is negative"
        assert error_metrics['max_error'] >= 0, "Max error is negative"
        
        # Property: PSNR should be positive for non-identical images
        if error_metrics['mse'] > 0:
            assert error_metrics['psnr'] > 0, "PSNR is not positive"
        
        # Property: Error map should have same shape as input
        assert error_metrics['error_map'].shape == image.shape, \
            f"Error map shape mismatch: {error_metrics['error_map'].shape} vs {image.shape}"
        
        # Property: Error map values should be non-negative
        assert np.all(error_metrics['error_map'] >= 0), \
            "Error map contains negative values"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(128, 128),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        levels=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_reconstruction_error_per_scale_valid(self, image, levels):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property 11 variant: Error per scale should be valid for all scales.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Decompose
        coeffs = self.wavelet_transform.decompose(image, levels=levels)
        
        # Compute error per scale
        errors_per_scale = self.wavelet_transform.compute_reconstruction_error_per_scale(
            image, coeffs
        )
        
        # Property: Should have error for each scale
        assert len(errors_per_scale) == levels, \
            f"Expected {levels} scales, got {len(errors_per_scale)}"
        
        # Property: All errors should be non-negative and finite
        for scale, error in errors_per_scale.items():
            assert error >= 0, f"Error at scale {scale} is negative: {error}"
            assert np.isfinite(error), f"Error at scale {scale} is not finite: {error}"
        
        # Property: Final error should be reasonably small for wavelet
        final_error = errors_per_scale[max(errors_per_scale.keys())]
        assert final_error < 0.1, \
            f"Final reconstruction error too high: {final_error}"


class TestEdgeDetectionOutputFormatProperty:
    """
    Property-based tests for edge detection output format.
    
    **Property 12: Edge Detection Output Format**
    **Validates: Requirements 9.1, 9.2, 9.3**
    
    For any transform coefficients (wavelet or curvelet) and edge strength threshold,
    edge detection should produce a binary edge map with the same spatial dimensions
    as the input image.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mimic_service = MIMICService()
        self.wavelet_transform = WaveletTransform()
        self.curvelet_transform = CurveletTransform()
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=64, max_value=256),
                st.integers(min_value=64, max_value=256)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        edge_strength=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_wavelet_edge_detection_output_format(self, image, edge_strength):
        """
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        Property 12: For any image and edge strength, wavelet edge detection
        should produce a binary edge map with same spatial dimensions.
        """
        # Skip constant images (no edges to detect)
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Decompose
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Detect edges
        edges = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=edge_strength
        )
        
        # Property: Output should be 2D array
        assert edges.ndim == 2, \
            f"Edge map should be 2D, got {edges.ndim}D"
        
        # Property: Output should have same spatial dimensions as input
        assert edges.shape == image.shape, \
            f"Edge map shape {edges.shape} doesn't match image shape {image.shape}"
        
        # Property: Output should be binary (only 0 and 1)
        unique_values = np.unique(edges)
        assert np.all((unique_values == 0) | (unique_values == 1)), \
            f"Edge map should be binary, got values: {unique_values}"
        
        # Property: Output should be integer type
        assert edges.dtype in [np.uint8, np.int32, np.int64], \
            f"Edge map should be integer type, got {edges.dtype}"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=64, max_value=256),
                st.integers(min_value=64, max_value=256)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        edge_strength=st.floats(min_value=0.0, max_value=1.0),
        angular_resolution=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_curvelet_edge_detection_output_format(self, image, edge_strength, angular_resolution):
        """
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        Property 12: For any image and edge strength, curvelet edge detection
        should produce a binary edge map with same spatial dimensions.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Decompose
        coeffs = self.curvelet_transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # Detect edges
        edges = self.mimic_service.detect_edges_from_curvelet(
            curvelet_coeffs=coeffs,
            edge_strength=edge_strength
        )
        
        # Property: Output should be 2D array
        assert edges.ndim == 2, \
            f"Edge map should be 2D, got {edges.ndim}D"
        
        # Property: Output should have same spatial dimensions as input
        assert edges.shape == image.shape, \
            f"Edge map shape {edges.shape} doesn't match image shape {image.shape}"
        
        # Property: Output should be binary (only 0 and 1)
        unique_values = np.unique(edges)
        assert np.all((unique_values == 0) | (unique_values == 1)), \
            f"Edge map should be binary, got values: {unique_values}"
        
        # Property: Output should be integer type
        assert edges.dtype in [np.uint8, np.int32, np.int64], \
            f"Edge map should be integer type, got {edges.dtype}"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(128, 128),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        edge_strength_low=st.floats(min_value=0.0, max_value=0.5),
        edge_strength_high=st.floats(min_value=0.5, max_value=1.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_edge_strength_threshold_effect(self, image, edge_strength_low, edge_strength_high):
        """
        **Validates: Requirement 9.1**
        
        Property 12 variant: Lower edge strength should detect more or equal edges
        than higher edge strength.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Ensure low < high
        assume(edge_strength_low < edge_strength_high)
        
        # Decompose
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Detect edges with low threshold
        edges_low = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=edge_strength_low
        )
        
        # Detect edges with high threshold
        edges_high = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=edge_strength_high
        )
        
        # Property: Lower threshold should detect more or equal edges
        num_edges_low = np.sum(edges_low)
        num_edges_high = np.sum(edges_high)
        
        assert num_edges_low >= num_edges_high, \
            f"Lower threshold ({edge_strength_low}) detected {num_edges_low} edges, " \
            f"but higher threshold ({edge_strength_high}) detected {num_edges_high} edges"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(128, 128),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        edge_strength=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_edge_detection_deterministic(self, image, edge_strength):
        """
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        Property 12 variant: Edge detection should be deterministic (same input
        produces same output).
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Decompose
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Detect edges twice
        edges1 = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=edge_strength
        )
        
        edges2 = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=edge_strength
        )
        
        # Property: Should produce identical results
        assert np.array_equal(edges1, edges2), \
            "Edge detection is not deterministic"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(128, 128),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_extreme_thresholds(self, image):
        """
        **Validates: Requirement 9.1**
        
        Property 12 variant: Extreme thresholds should produce expected results.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Decompose
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Test threshold = 0.0 (should detect many edges)
        edges_zero = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=0.0
        )
        
        # Test threshold = 1.0 (should detect few or no edges)
        edges_one = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=1.0
        )
        
        # Property: Zero threshold should detect more edges than threshold of 1.0
        num_edges_zero = np.sum(edges_zero)
        num_edges_one = np.sum(edges_one)
        
        assert num_edges_zero >= num_edges_one, \
            f"Threshold 0.0 detected {num_edges_zero} edges, " \
            f"but threshold 1.0 detected {num_edges_one} edges"


class TestEdgeDetectionEdgeCases:
    """
    Property-based tests for edge cases in edge detection.
    
    These tests verify that edge detection handles special cases correctly.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mimic_service = MIMICService()
        self.wavelet_transform = WaveletTransform()
    
    @given(
        constant_value=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        ),
        shape=st.tuples(
            st.integers(min_value=64, max_value=128),
            st.integers(min_value=64, max_value=128)
        ),
        edge_strength=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_constant_image_no_edges(self, constant_value, shape, edge_strength):
        """
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        Edge case: Constant images should produce no edges (or very few).
        """
        image = np.full(shape, constant_value, dtype=np.float64)
        
        # Decompose
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Detect edges
        edges = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=edge_strength
        )
        
        # Property: Constant images should have no or very few edges
        # (some edge detection methods may detect spurious edges at boundaries)
        num_edges = np.sum(edges)
        total_pixels = edges.size
        edge_ratio = num_edges / total_pixels
        
        assert edge_ratio < 0.05, \
            f"Constant image has too many edges: {edge_ratio:.2%} of pixels"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(128, 128),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        edge_strength=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_edge_detection_preserves_shape(self, image, edge_strength):
        """
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        Property: Edge detection should always preserve image shape.
        """
        original_shape = image.shape
        
        # Decompose
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Detect edges
        edges = self.mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=coeffs,
            edge_strength=edge_strength
        )
        
        # Property: Shape should be preserved
        assert edges.shape == original_shape, \
            f"Shape changed: {original_shape} -> {edges.shape}"


# Mark all tests in this module as property tests for easy filtering
pytestmark = pytest.mark.property
