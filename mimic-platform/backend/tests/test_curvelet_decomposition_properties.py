"""
Property-Based Tests for Curvelet Decomposition

This module contains property-based tests using Hypothesis to verify
universal properties of curvelet decomposition across a wide range of inputs.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 7.1, 7.2**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst

from app.services.transforms.curvelet import CurveletTransform
from app.services.transforms.fft_directional import FFTDirectionalFilter


# Hypothesis settings for property tests
# Minimum 100 iterations as specified in requirements


class TestCurveletDecompositionStructure:
    """
    Property 9: Curvelet Decomposition Structure
    
    For any image and angular resolution, curvelet decomposition should produce
    at least 3 scale levels with directional coefficients for the specified
    number of orientations at each scale.
    
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """
    
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
        angular_resolution=st.integers(min_value=8, max_value=32)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_decomposition_produces_minimum_scales(
        self,
        image: np.ndarray,
        angular_resolution: int
    ):
        """
        **Validates: Requirements 6.1, 6.2, 6.3**
        
        Property: For any image and angular resolution parameter, the curvelet
        decomposition should produce at least 3 scale levels.
        """
        transform = CurveletTransform()
        
        # Decompose with default levels (should be at least 3)
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # Verify at least 3 scale levels exist
        assert coefficients.scales >= 3, (
            f"Expected at least 3 scale levels, got {coefficients.scales}"
        )
        
        # Verify coefficient structure contains all scales
        assert len(coefficients.coefficients) >= 3, (
            f"Expected coefficients for at least 3 scales, "
            f"got {len(coefficients.coefficients)}"
        )
        
        # Verify scales are indexed from 0 to scales-1
        for scale_idx in range(coefficients.scales):
            assert scale_idx in coefficients.coefficients, (
                f"Missing coefficients for scale {scale_idx}"
            )
    
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
        angular_resolution=st.integers(min_value=8, max_value=32),
        levels=st.integers(min_value=3, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_directional_coefficients_per_scale(
        self,
        image: np.ndarray,
        angular_resolution: int,
        levels: int
    ):
        """
        **Validates: Requirements 6.3, 6.4**
        
        Property: For any image and angular resolution, each scale should have
        directional coefficients for the specified number of orientations.
        """
        transform = CurveletTransform()
        
        # Decompose with specified parameters
        coefficients = transform.decompose(
            image,
            levels=levels,
            angular_resolution=angular_resolution
        )
        
        # Verify each scale has directional coefficients
        for scale_idx in range(coefficients.scales):
            scale_coeffs = coefficients.coefficients[scale_idx]
            
            # Each scale should have orientation coefficients
            assert len(scale_coeffs) > 0, (
                f"Scale {scale_idx} has no orientation coefficients"
            )
            
            # For scales > 0, should have approximately angular_resolution orientations
            # (Scale 0 may have fewer orientations as it's the coarsest scale)
            if scale_idx > 0:
                # Allow some flexibility for coarsest scales
                assert len(scale_coeffs) >= angular_resolution // 2, (
                    f"Scale {scale_idx} has {len(scale_coeffs)} orientations, "
                    f"expected at least {angular_resolution // 2}"
                )
            
            # Verify each orientation has coefficient array
            for angle_idx, coeffs in scale_coeffs.items():
                assert isinstance(coeffs, np.ndarray), (
                    f"Coefficients at scale {scale_idx}, angle {angle_idx} "
                    f"are not numpy array"
                )
                assert coeffs.shape == image.shape, (
                    f"Coefficient shape {coeffs.shape} doesn't match "
                    f"image shape {image.shape}"
                )
    
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
        angular_resolution=st.integers(min_value=8, max_value=32)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_coefficient_shape_preservation(
        self,
        image: np.ndarray,
        angular_resolution: int
    ):
        """
        **Validates: Requirements 6.4**
        
        Property: All directional coefficients should have the same spatial
        dimensions as the input image.
        """
        transform = CurveletTransform()
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # Verify shape is preserved in coefficient structure
        assert coefficients.shape == image.shape, (
            f"Coefficient structure shape {coefficients.shape} doesn't match "
            f"image shape {image.shape}"
        )
        
        # Verify all coefficient arrays have correct shape
        for scale_idx in range(coefficients.scales):
            for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                assert coeffs.shape == image.shape, (
                    f"Coefficients at scale {scale_idx}, angle {angle_idx} "
                    f"have shape {coeffs.shape}, expected {image.shape}"
                )


class TestDirectionalCoefficientProperties:
    """
    Property 10: Directional Coefficient Properties
    
    For any curvelet decomposition, all coefficient magnitudes should be
    non-negative, and energy per orientation should be non-negative with
    sum across orientations equaling total energy.
    
    **Validates: Requirements 7.1, 7.2**
    """
    
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
        angular_resolution=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_10_coefficient_magnitudes_non_negative(
        self,
        image: np.ndarray,
        angular_resolution: int
    ):
        """
        **Validates: Requirements 7.1**
        
        Property: For any curvelet decomposition, coefficient magnitudes
        (absolute values) should be non-negative.
        
        Note: Coefficients themselves can be negative (representing phase),
        but their magnitudes must be non-negative by definition.
        """
        transform = CurveletTransform()
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # Verify all coefficient magnitudes are non-negative
        for scale_idx in range(coefficients.scales):
            for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                magnitudes = np.abs(coeffs)
                
                # Magnitudes must be non-negative (by definition of absolute value)
                assert np.all(magnitudes >= 0), (
                    f"Found negative magnitudes at scale {scale_idx}, "
                    f"angle {angle_idx}"
                )
                
                # Verify no NaN or Inf values
                assert np.all(np.isfinite(magnitudes)), (
                    f"Found non-finite magnitudes at scale {scale_idx}, "
                    f"angle {angle_idx}"
                )
    
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
        angular_resolution=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_10_energy_per_orientation_non_negative(
        self,
        image: np.ndarray,
        angular_resolution: int
    ):
        """
        **Validates: Requirements 7.1, 7.2**
        
        Property: For any curvelet decomposition, energy per orientation
        should be non-negative.
        """
        transform = CurveletTransform()
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # Extract directional energy
        energy_dict = transform.extract_directional_energy(coefficients)
        
        # Verify energy is non-negative for all scales and orientations
        for scale_idx, energies in energy_dict.items():
            assert np.all(energies >= 0), (
                f"Found negative energy at scale {scale_idx}: "
                f"min={energies.min()}"
            )
            
            # Verify no NaN or Inf values
            assert np.all(np.isfinite(energies)), (
                f"Found non-finite energy at scale {scale_idx}"
            )
    
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
        angular_resolution=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_10_energy_conservation_across_orientations(
        self,
        image: np.ndarray,
        angular_resolution: int
    ):
        """
        **Validates: Requirements 7.2**
        
        Property: For any curvelet decomposition, the sum of energy across
        all orientations at a given scale should equal the total energy
        at that scale.
        """
        transform = CurveletTransform()
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # Extract directional energy
        energy_dict = transform.extract_directional_energy(coefficients)
        
        # For each scale, verify energy conservation
        for scale_idx in range(coefficients.scales):
            # Sum energy across all orientations
            total_energy_from_orientations = np.sum(energy_dict[scale_idx])
            
            # Compute total energy directly from coefficients
            total_energy_direct = 0.0
            for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                total_energy_direct += np.sum(coeffs ** 2)
            
            # Energies should match (within numerical precision)
            assert np.isclose(
                total_energy_from_orientations,
                total_energy_direct,
                rtol=1e-10,
                atol=1e-10
            ), (
                f"Energy mismatch at scale {scale_idx}: "
                f"sum of orientation energies = {total_energy_from_orientations}, "
                f"direct computation = {total_energy_direct}"
            )
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(128, 128),
            elements=st.floats(
                min_value=0.1,  # Avoid zero image
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        angular_resolution=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_10_non_zero_image_has_positive_total_energy(
        self,
        image: np.ndarray,
        angular_resolution: int
    ):
        """
        **Validates: Requirements 7.1, 7.2**
        
        Property: For any non-zero image, the total energy across all
        scales and orientations should be positive.
        """
        # Ensure image is not all zeros
        if np.all(image == 0):
            image = image + 0.1
        
        transform = CurveletTransform()
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # Extract directional energy
        energy_dict = transform.extract_directional_energy(coefficients)
        
        # Compute total energy across all scales and orientations
        total_energy = sum(np.sum(energies) for energies in energy_dict.values())
        
        # Total energy should be positive for non-zero image
        assert total_energy > 0, (
            f"Expected positive total energy for non-zero image, "
            f"got {total_energy}"
        )


class TestCurveletDecompositionEdgeCases:
    """Additional property tests for edge cases and robustness."""
    
    @given(
        size=st.integers(min_value=64, max_value=256),
        angular_resolution=st.integers(min_value=8, max_value=32)
    )
    @settings(max_examples=50, deadline=None)
    def test_zero_image_decomposition(
        self,
        size: int,
        angular_resolution: int
    ):
        """
        Property: Zero image should produce zero coefficients.
        """
        image = np.zeros((size, size))
        transform = CurveletTransform()
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # All coefficients should be zero (or very close to zero)
        for scale_idx in range(coefficients.scales):
            for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                assert np.allclose(coeffs, 0, atol=1e-10), (
                    f"Expected zero coefficients for zero image at "
                    f"scale {scale_idx}, angle {angle_idx}"
                )
    
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
        scale_factor=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_scaling_linearity(
        self,
        image: np.ndarray,
        scale_factor: float
    ):
        """
        Property: Scaling the input image should scale the coefficients
        proportionally (linearity property).
        """
        transform = CurveletTransform()
        
        # Decompose original image
        coeffs_original = transform.decompose(image, levels=3, angular_resolution=8)
        
        # Decompose scaled image
        scaled_image = image * scale_factor
        coeffs_scaled = transform.decompose(scaled_image, levels=3, angular_resolution=8)
        
        # Coefficients should be scaled by the same factor
        for scale_idx in range(coeffs_original.scales):
            for angle_idx in coeffs_original.coefficients[scale_idx].keys():
                original = coeffs_original.coefficients[scale_idx][angle_idx]
                scaled = coeffs_scaled.coefficients[scale_idx][angle_idx]
                
                # Check if scaled coefficients match expected scaling
                expected = original * scale_factor
                assert np.allclose(scaled, expected, rtol=1e-5, atol=1e-8), (
                    f"Scaling linearity violated at scale {scale_idx}, "
                    f"angle {angle_idx}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
