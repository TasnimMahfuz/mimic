"""
Property-Based Tests for Enhancement Processing

This module contains property-based tests for the enhancement processing
functionality in the MIMIC service, validating that enhancement parameters
have the expected effects on images.

**Validates: Property 13 - Enhancement Parameter Effects**
**Validates: Requirements 10.1, 10.2**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst

from app.services.mimic_service import MIMICService


# Test configuration
MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 128


@pytest.fixture
def mimic_service():
    """Fixture providing a MIMICService instance."""
    return MIMICService()


@given(
    image=npst.arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=MIN_IMAGE_SIZE, max_value=MAX_IMAGE_SIZE),
            st.integers(min_value=MIN_IMAGE_SIZE, max_value=MAX_IMAGE_SIZE)
        ),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ),
    enhancement_factor=st.floats(min_value=1.1, max_value=5.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_property_contrast_enhancement_increases_dynamic_range(mimic_service, image, enhancement_factor):
    """
    **Validates: Requirements 10.1**
    
    Property 13 (Part 1): Enhancement Parameter Effects - Contrast Enhancement
    
    For any image, applying contrast enhancement with factor f > 1 should
    increase the dynamic range (difference between max and min intensities).
    
    This property verifies that contrast enhancement actually enhances contrast
    by stretching the intensity distribution.
    """
    # Skip images with no variation (constant images)
    if np.std(image) < 1e-6:
        return
    
    # Compute original dynamic range
    original_range = np.max(image) - np.min(image)
    
    # Apply contrast enhancement
    enhanced = mimic_service.apply_contrast_enhancement(image, enhancement_factor)
    
    # Compute enhanced dynamic range
    enhanced_range = np.max(enhanced) - np.min(enhanced)
    
    # Verify enhancement increased dynamic range (or kept it the same if already at max)
    # Allow for small numerical errors
    assert enhanced_range >= original_range - 1e-6, \
        f"Contrast enhancement should increase dynamic range: {original_range} -> {enhanced_range}"
    
    # Verify output is in valid range [0, 1]
    assert np.all(enhanced >= 0.0) and np.all(enhanced <= 1.0), \
        "Enhanced image should be in range [0, 1]"
    
    # Verify output shape matches input
    assert enhanced.shape == image.shape, \
        "Enhanced image should have same shape as input"


@given(
    image=npst.arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=MIN_IMAGE_SIZE, max_value=MAX_IMAGE_SIZE),
            st.integers(min_value=MIN_IMAGE_SIZE, max_value=MAX_IMAGE_SIZE)
        ),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ),
    kernel_size=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_property_smoothing_reduces_high_frequency_content(mimic_service, image, kernel_size):
    """
    **Validates: Requirements 10.2**
    
    Property 13 (Part 2): Enhancement Parameter Effects - Spatial Smoothing
    
    For any image, applying spatial smoothing with kernel size k > 0 should
    reduce high-frequency content, measured by the magnitude of gradients.
    
    This property verifies that smoothing actually smooths the image by
    reducing sharp transitions and high-frequency noise.
    """
    # Compute original gradient magnitude (measure of high-frequency content)
    grad_y, grad_x = np.gradient(image)
    original_gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    
    # Apply spatial smoothing
    smoothed = mimic_service.apply_spatial_smoothing(image, kernel_size)
    
    # Compute smoothed gradient magnitude
    smooth_grad_y, smooth_grad_x = np.gradient(smoothed)
    smoothed_gradient_magnitude = np.mean(np.sqrt(smooth_grad_x**2 + smooth_grad_y**2))
    
    # Verify smoothing reduced gradient magnitude (high-frequency content)
    # Allow for small numerical errors and edge cases
    assert smoothed_gradient_magnitude <= original_gradient_magnitude + 1e-6, \
        f"Smoothing should reduce high-frequency content: {original_gradient_magnitude} -> {smoothed_gradient_magnitude}"
    
    # Verify output is in valid range [0, 1]
    assert np.all(smoothed >= 0.0) and np.all(smoothed <= 1.0), \
        "Smoothed image should be in range [0, 1]"
    
    # Verify output shape matches input
    assert smoothed.shape == image.shape, \
        "Smoothed image should have same shape as input"


@given(
    image=npst.arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=MIN_IMAGE_SIZE, max_value=MAX_IMAGE_SIZE),
            st.integers(min_value=MIN_IMAGE_SIZE, max_value=MAX_IMAGE_SIZE)
        ),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ),
    enhancement_factor=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    smoothing=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_property_combined_enhancement_preserves_range(mimic_service, image, enhancement_factor, smoothing):
    """
    **Validates: Requirements 10.1, 10.2**
    
    Property 13 (Part 3): Enhancement Parameter Effects - Combined Enhancement
    
    For any image and enhancement parameters, the combined enhancement
    (smoothing + contrast) should always produce output in the valid range [0, 1].
    
    This property verifies that the enhancement pipeline maintains valid
    intensity values regardless of parameter combinations.
    """
    # Apply combined enhancement
    enhanced = mimic_service.apply_enhancement(image, enhancement_factor, smoothing)
    
    # Verify output is in valid range [0, 1]
    assert np.all(enhanced >= 0.0), \
        f"Enhanced image has values below 0: min={np.min(enhanced)}"
    assert np.all(enhanced <= 1.0), \
        f"Enhanced image has values above 1: max={np.max(enhanced)}"
    
    # Verify output shape matches input
    assert enhanced.shape == image.shape, \
        "Enhanced image should have same shape as input"
    
    # Verify no NaN or Inf values
    assert np.all(np.isfinite(enhanced)), \
        "Enhanced image should not contain NaN or Inf values"


@given(
    image=npst.arrays(
        dtype=np.float64,
        shape=(64, 64),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=50, deadline=None)
def test_property_no_enhancement_preserves_image(mimic_service, image):
    """
    **Validates: Requirements 10.1, 10.2**
    
    Property 13 (Part 4): Enhancement Parameter Effects - Identity Case
    
    For any image, applying enhancement with factor=1.0 and smoothing=0.0
    should preserve the original image (identity transformation).
    
    This property verifies that the enhancement pipeline has a proper
    identity case where no enhancement is applied.
    """
    # Apply "no enhancement" (identity parameters)
    enhanced = mimic_service.apply_enhancement(image, enhancement_factor=1.0, smoothing=0.0)
    
    # Verify output matches input (within numerical precision)
    assert np.allclose(enhanced, image, rtol=1e-5, atol=1e-8), \
        "Enhancement with identity parameters should preserve the image"


@given(
    enhancement_factor=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    smoothing=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_property_enhancement_handles_edge_cases(mimic_service, enhancement_factor, smoothing):
    """
    **Validates: Requirements 10.1, 10.2**
    
    Property 13 (Part 5): Enhancement Parameter Effects - Edge Cases
    
    For any enhancement parameters, the enhancement pipeline should handle
    edge case images (constant, all zeros, all ones) without errors.
    
    This property verifies robustness of the enhancement implementation.
    """
    # Test with constant image
    constant_image = np.full((64, 64), 0.5)
    enhanced_constant = mimic_service.apply_enhancement(constant_image, enhancement_factor, smoothing)
    assert enhanced_constant.shape == constant_image.shape
    assert np.all(np.isfinite(enhanced_constant))
    
    # Test with all zeros
    zero_image = np.zeros((64, 64))
    enhanced_zero = mimic_service.apply_enhancement(zero_image, enhancement_factor, smoothing)
    assert enhanced_zero.shape == zero_image.shape
    assert np.all(np.isfinite(enhanced_zero))
    
    # Test with all ones
    ones_image = np.ones((64, 64))
    enhanced_ones = mimic_service.apply_enhancement(ones_image, enhancement_factor, smoothing)
    assert enhanced_ones.shape == ones_image.shape
    assert np.all(np.isfinite(enhanced_ones))
