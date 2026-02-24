"""
Property-based tests for flux normalization functionality.

Uses Hypothesis to verify universal properties across randomized inputs.

Tests Requirements: 3.1, 3.2, 3.3, 3.5

**Property 4: Normalization Range Invariant**
For any image, after flux normalization all pixel values should be in [0, 1]
with the minimum value mapping to 0 and the maximum value mapping to 1.

**Property 5: Normalization Round-Trip**
For any image, normalize then denormalize should recover original values
within numerical precision.
"""
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
import hypothesis.extra.numpy as npst
from app.services.mimic_service import MIMICService


# Hypothesis settings for property tests
# Minimum 100 iterations as specified in design document
settings.register_profile("default", max_examples=100, deadline=None)
settings.load_profile("default")


class TestNormalizationRangeInvariantProperty:
    """
    Property-based tests for normalization range invariant.
    
    **Property 4: Normalization Range Invariant**
    **Validates: Requirements 3.1, 3.2, 3.3**
    
    For any image, after flux normalization all pixel values should be in [0, 1]
    with the minimum value mapping to 0 and the maximum value mapping to 1.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = MIMICService()
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=10, max_value=200),
                st.integers(min_value=10, max_value=200)
            ),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_normalization_range_invariant(self, image):
        """
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Property 4: For any image, after normalization all pixels should be in [0, 1]
        with min→0 and max→1.
        """
        # Skip constant images (all pixels same value) as they're a special case
        # handled separately - they normalize to all zeros
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Normalize the image
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Property: All pixel values should be in [0, 1] range
        assert np.all(normalized >= 0.0), \
            f"Found pixels below 0: min={normalized.min()}"
        assert np.all(normalized <= 1.0), \
            f"Found pixels above 1: max={normalized.max()}"
        
        # Property: Minimum value should map to 0
        assert np.isclose(normalized.min(), 0.0, atol=1e-10), \
            f"Minimum value not 0: {normalized.min()}"
        
        # Property: Maximum value should map to 1
        assert np.isclose(normalized.max(), 1.0, atol=1e-10), \
            f"Maximum value not 1: {normalized.max()}"
        
        # Property: Scale parameters should match original image statistics
        assert np.isclose(scale_params['min'], image.min(), rtol=1e-10), \
            f"Scale min doesn't match: {scale_params['min']} vs {image.min()}"
        assert np.isclose(scale_params['max'], image.max(), rtol=1e-10), \
            f"Scale max doesn't match: {scale_params['max']} vs {image.max()}"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(50, 50),
            elements=st.floats(
                min_value=0.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_normalization_range_positive_values(self, image):
        """
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Property 4 variant: Test with positive-only values (common in astronomical data).
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # All values should be in [0, 1]
        assert np.all((normalized >= 0.0) & (normalized <= 1.0))
        
        # Min and max should be at boundaries
        assert np.isclose(normalized.min(), 0.0, atol=1e-10)
        assert np.isclose(normalized.max(), 1.0, atol=1e-10)
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(30, 30),
            elements=st.floats(
                min_value=-500.0,
                max_value=500.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_normalization_range_mixed_values(self, image):
        """
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Property 4 variant: Test with mixed positive/negative values.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # All values should be in [0, 1]
        assert np.all((normalized >= 0.0) & (normalized <= 1.0))
        
        # Min and max should be at boundaries
        assert np.isclose(normalized.min(), 0.0, atol=1e-10)
        assert np.isclose(normalized.max(), 1.0, atol=1e-10)


class TestNormalizationRoundTripProperty:
    """
    Property-based tests for normalization round-trip.
    
    **Property 5: Normalization Round-Trip**
    **Validates: Requirement 3.5**
    
    For any image, normalize then denormalize should recover original values
    within numerical precision.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = MIMICService()
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=10, max_value=200),
                st.integers(min_value=10, max_value=200)
            ),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_normalization_roundtrip(self, image):
        """
        **Validates: Requirement 3.5**
        
        Property 5: For any image, normalize then denormalize should recover
        original values within numerical precision.
        """
        # Skip constant images (special case - they can't be perfectly recovered)
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Normalize the image
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Denormalize back to original range
        denormalized = self.service.denormalize_flux(normalized, scale_params)
        
        # Property: Denormalized should match original within numerical precision
        # Using rtol=1e-9 to account for floating point arithmetic in normalization/denormalization
        assert np.allclose(denormalized, image, rtol=1e-9, atol=1e-9), \
            f"Round-trip failed: max error = {np.max(np.abs(denormalized - image))}"
        
        # Property: Shape should be preserved
        assert denormalized.shape == image.shape, \
            f"Shape changed: {image.shape} -> {denormalized.shape}"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(100, 100),
            elements=st.floats(
                min_value=0.0,
                max_value=10000.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_roundtrip_large_dynamic_range(self, image):
        """
        **Validates: Requirement 3.5**
        
        Property 5 variant: Test round-trip with large dynamic range values.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Ensure we have some dynamic range
        assume(image.max() - image.min() > 1e-6)
        
        normalized, scale_params = self.service.normalize_flux(image)
        denormalized = self.service.denormalize_flux(normalized, scale_params)
        
        # Should recover original values
        assert np.allclose(denormalized, image, rtol=1e-9, atol=1e-9)
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(50, 50),
            elements=st.floats(
                min_value=1e-6,
                max_value=1e-3,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_roundtrip_small_values(self, image):
        """
        **Validates: Requirement 3.5**
        
        Property 5 variant: Test round-trip with very small values.
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Ensure we have some dynamic range
        assume(image.max() - image.min() > 1e-9)
        
        normalized, scale_params = self.service.normalize_flux(image)
        denormalized = self.service.denormalize_flux(normalized, scale_params)
        
        # Should recover original values (may need slightly relaxed tolerance for small values)
        assert np.allclose(denormalized, image, rtol=1e-8, atol=1e-12)


class TestNormalizationEdgeCases:
    """
    Property-based tests for edge cases in normalization.
    
    These tests verify that normalization handles special cases correctly.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = MIMICService()
    
    @given(
        constant_value=st.floats(
            min_value=-1000.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False
        ),
        shape=st.tuples(
            st.integers(min_value=5, max_value=100),
            st.integers(min_value=5, max_value=100)
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_constant_image_normalization(self, constant_value, shape):
        """
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Edge case: Constant images (all pixels same value) should normalize to zeros.
        """
        image = np.full(shape, constant_value, dtype=np.float64)
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Constant images should normalize to all zeros
        assert np.all(normalized == 0.0), \
            f"Constant image didn't normalize to zeros: {np.unique(normalized)}"
        
        # Scale parameters should still be correct
        assert scale_params['min'] == constant_value
        assert scale_params['max'] == constant_value
        assert scale_params['range'] == 0.0
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(50, 50),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_normalization_preserves_shape(self, image):
        """
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Property: Normalization should always preserve image shape.
        """
        original_shape = image.shape
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Shape should be preserved
        assert normalized.shape == original_shape, \
            f"Shape changed: {original_shape} -> {normalized.shape}"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(30, 30),
            elements=st.floats(
                min_value=-1000.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_normalization_idempotent_on_normalized(self, image):
        """
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Property: Normalizing an already normalized image should be idempotent
        (produce the same result).
        """
        # Skip constant images
        if np.all(image == image.flat[0]):
            assume(False)
        
        # First normalization
        normalized1, scale_params1 = self.service.normalize_flux(image)
        
        # Second normalization on already normalized image
        normalized2, scale_params2 = self.service.normalize_flux(normalized1)
        
        # Should produce the same result (already in [0, 1])
        assert np.allclose(normalized2, normalized1, rtol=1e-10)
        
        # Scale parameters should reflect [0, 1] range
        assert np.isclose(scale_params2['min'], 0.0, atol=1e-10)
        assert np.isclose(scale_params2['max'], 1.0, atol=1e-10)


# Mark all tests in this module as property tests for easy filtering
pytestmark = pytest.mark.property
