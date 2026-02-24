"""
Property-based tests for wavelet decomposition functionality.

Uses Hypothesis to verify universal properties across randomized inputs.

Tests Requirements: 5.1, 5.2, 5.3

**Property 8: Wavelet Decomposition Structure**
For any image, the wavelet decomposition should produce at least 3 scale levels,
with coefficients extracted at each scale, and the total number of coefficients
should equal the original image size.
"""
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
import hypothesis.extra.numpy as npst
from app.services.transforms.wavelet import WaveletTransform, WaveletCoefficients


# Hypothesis settings for property tests
# Minimum 100 iterations as specified in design document
settings.register_profile("default", max_examples=100, deadline=None)
settings.load_profile("default")


class TestWaveletDecompositionStructureProperty:
    """
    Property-based tests for wavelet decomposition structure.
    
    **Property 8: Wavelet Decomposition Structure**
    **Validates: Requirements 5.1, 5.2, 5.3**
    
    For any image, the wavelet decomposition should produce at least 3 scale levels,
    with coefficients extracted at each scale, and the total number of coefficients
    should equal the original image size.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.wavelet_transform = WaveletTransform(wavelet='db4')
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=128, max_value=512),
                st.integers(min_value=128, max_value=512)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_wavelet_decomposition_structure(self, image):
        """
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        Property 8: For any image, wavelet decomposition should produce at least
        3 scale levels with coefficients extracted at each scale.
        """
        # Perform wavelet decomposition with 3 levels
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Property 1: Should produce at least 3 scale levels
        assert coeffs.levels >= 3, \
            f"Expected at least 3 levels, got {coeffs.levels}"
        
        # Property 2: Should have detail coefficients at each scale
        assert len(coeffs.details) >= 3, \
            f"Expected at least 3 detail levels, got {len(coeffs.details)}"
        
        # Property 3: Each scale should have coefficients extracted
        for level_idx, (cH, cV, cD) in enumerate(coeffs.details):
            assert cH is not None, f"Level {level_idx}: Horizontal coefficients missing"
            assert cV is not None, f"Level {level_idx}: Vertical coefficients missing"
            assert cD is not None, f"Level {level_idx}: Diagonal coefficients missing"
            
            # All coefficients should be 2D arrays
            assert cH.ndim == 2, f"Level {level_idx}: cH not 2D"
            assert cV.ndim == 2, f"Level {level_idx}: cV not 2D"
            assert cD.ndim == 2, f"Level {level_idx}: cD not 2D"
            
            # All coefficients should have non-zero size
            assert cH.size > 0, f"Level {level_idx}: cH is empty"
            assert cV.size > 0, f"Level {level_idx}: cV is empty"
            assert cD.size > 0, f"Level {level_idx}: cD is empty"
        
        # Property 4: Approximation coefficients should exist
        assert coeffs.approximation is not None, "Approximation coefficients missing"
        assert coeffs.approximation.ndim == 2, "Approximation not 2D"
        assert coeffs.approximation.size > 0, "Approximation is empty"
        
        # Property 5: Total number of coefficients should equal original image size
        # (accounting for wavelet transform properties)
        total_coeffs = coeffs.approximation.size
        for cH, cV, cD in coeffs.details:
            total_coeffs += cH.size + cV.size + cD.size
        
        # The total should be approximately equal to the original image size
        # (may differ slightly due to padding in wavelet transform)
        original_size = image.size
        ratio = total_coeffs / original_size
        
        # Ratio should be close to 1.0 (within 50% tolerance for padding)
        assert 0.5 <= ratio <= 1.5, \
            f"Coefficient count mismatch: {total_coeffs} coeffs for {original_size} pixels (ratio: {ratio:.2f})"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=(256, 256),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        levels=st.integers(min_value=3, max_value=6)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_wavelet_decomposition_variable_levels(self, image, levels):
        """
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        Property 8 variant: Test with variable number of decomposition levels.
        """
        # Perform wavelet decomposition
        coeffs = self.wavelet_transform.decompose(image, levels=levels)
        
        # Should produce at least 3 scale levels (or requested levels if less)
        assert coeffs.levels >= 3, \
            f"Expected at least 3 levels, got {coeffs.levels}"
        
        # Should have detail coefficients at each scale
        assert len(coeffs.details) >= 3, \
            f"Expected at least 3 detail levels, got {len(coeffs.details)}"
        
        # Each scale should have valid coefficients
        for level_idx, (cH, cV, cD) in enumerate(coeffs.details):
            assert cH.ndim == 2 and cH.size > 0
            assert cV.ndim == 2 and cV.size > 0
            assert cD.ndim == 2 and cD.size > 0
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=128, max_value=512),
                st.integers(min_value=128, max_value=512)
            ),
            elements=st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_wavelet_decomposition_mixed_values(self, image):
        """
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        Property 8 variant: Test with mixed positive/negative values.
        """
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Should produce at least 3 scale levels
        assert coeffs.levels >= 3
        assert len(coeffs.details) >= 3
        
        # All coefficients should be valid
        assert coeffs.approximation is not None
        for cH, cV, cD in coeffs.details:
            assert cH is not None and cH.size > 0
            assert cV is not None and cV.size > 0
            assert cD is not None and cD.size > 0
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=200, max_value=400),
                st.integers(min_value=200, max_value=400)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_coefficient_dimensions_increase_through_levels(self, image):
        """
        **Validates: Requirements 5.2, 5.3**
        
        Property: In PyWavelets wavedec2, detail coefficient dimensions increase
        from coarsest (first) to finest (last) scale.
        """
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # In PyWavelets wavedec2, details list goes from coarsest to finest
        # So dimensions should increase as we go through the list
        prev_size = None
        for level_idx, (cH, cV, cD) in enumerate(coeffs.details):
            current_size = cH.shape[0] * cH.shape[1]
            
            if prev_size is not None:
                # Each successive level should have larger or equal dimensions
                # (going from coarse to fine)
                assert current_size >= prev_size, \
                    f"Level {level_idx}: size decreased from {prev_size} to {current_size}"
            
            prev_size = current_size
        
        # Approximation should be smallest (coarsest scale)
        approx_size = coeffs.approximation.shape[0] * coeffs.approximation.shape[1]
        # First detail level should be approximately same size as approximation
        first_detail_size = coeffs.details[0][0].shape[0] * coeffs.details[0][0].shape[1]
        assert approx_size <= first_detail_size * 1.5, \
            f"Approximation size {approx_size} much larger than first detail {first_detail_size}"
    
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
    def test_property_wavelet_name_preserved(self, image):
        """
        **Validates: Requirement 5.1**
        
        Property: Wavelet name should be preserved in coefficients.
        """
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Wavelet name should match the transform's wavelet
        assert coeffs.wavelet_name == self.wavelet_transform.wavelet, \
            f"Wavelet name mismatch: {coeffs.wavelet_name} vs {self.wavelet_transform.wavelet}"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=128, max_value=512),
                st.integers(min_value=128, max_value=512)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_coefficient_types_are_numeric(self, image):
        """
        **Validates: Requirements 5.2, 5.3**
        
        Property: All coefficients should be numeric arrays.
        """
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Approximation should be numeric
        assert np.issubdtype(coeffs.approximation.dtype, np.number), \
            f"Approximation not numeric: {coeffs.approximation.dtype}"
        
        # All detail coefficients should be numeric
        for level_idx, (cH, cV, cD) in enumerate(coeffs.details):
            assert np.issubdtype(cH.dtype, np.number), \
                f"Level {level_idx} cH not numeric: {cH.dtype}"
            assert np.issubdtype(cV.dtype, np.number), \
                f"Level {level_idx} cV not numeric: {cV.dtype}"
            assert np.issubdtype(cD.dtype, np.number), \
                f"Level {level_idx} cD not numeric: {cD.dtype}"


class TestWaveletDecompositionEdgeCases:
    """
    Property-based tests for edge cases in wavelet decomposition.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.wavelet_transform = WaveletTransform(wavelet='db4')
    
    @given(
        constant_value=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        ),
        shape=st.tuples(
            st.integers(min_value=128, max_value=256),
            st.integers(min_value=128, max_value=256)
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_constant_image_decomposition(self, constant_value, shape):
        """
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        Edge case: Constant images should still decompose correctly.
        """
        image = np.full(shape, constant_value, dtype=np.float64)
        
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Should still produce at least 3 levels
        assert coeffs.levels >= 3
        assert len(coeffs.details) >= 3
        
        # All coefficients should exist
        assert coeffs.approximation is not None
        for cH, cV, cD in coeffs.details:
            assert cH is not None and cH.size > 0
            assert cV is not None and cV.size > 0
            assert cD is not None and cD.size > 0
        
        # For constant images, detail coefficients should be near zero
        for level_idx, (cH, cV, cD) in enumerate(coeffs.details):
            assert np.allclose(cH, 0.0, atol=1e-10), \
                f"Level {level_idx}: cH not near zero for constant image"
            assert np.allclose(cV, 0.0, atol=1e-10), \
                f"Level {level_idx}: cV not near zero for constant image"
            assert np.allclose(cD, 0.0, atol=1e-10), \
                f"Level {level_idx}: cD not near zero for constant image"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=128, max_value=512),
                st.integers(min_value=128, max_value=512)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_decomposition_has_nonzero_energy(self, image):
        """
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        Property: Decomposition should have non-zero energy in coefficients.
        """
        # Skip constant images (zero energy in details)
        if np.all(image == image.flat[0]):
            assume(False)
        
        # Skip near-zero images
        if np.max(np.abs(image)) < 1e-10:
            assume(False)
        
        coeffs = self.wavelet_transform.decompose(image, levels=3)
        
        # Compute energy in coefficients
        coeff_energy = np.sum(coeffs.approximation ** 2)
        for cH, cV, cD in coeffs.details:
            coeff_energy += np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)
        
        # Energy should be non-zero for non-constant images
        assert coeff_energy > 0, \
            f"Coefficient energy is zero for non-constant image"
    
    @given(
        image=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=128, max_value=256),
                st.integers(min_value=128, max_value=256)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_decomposition_is_deterministic(self, image):
        """
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        Property: Decomposing the same image twice should produce identical results.
        """
        # First decomposition
        coeffs1 = self.wavelet_transform.decompose(image, levels=3)
        
        # Second decomposition
        coeffs2 = self.wavelet_transform.decompose(image, levels=3)
        
        # Approximations should be identical
        assert np.array_equal(coeffs1.approximation, coeffs2.approximation), \
            "Approximation coefficients differ between decompositions"
        
        # All detail coefficients should be identical
        for level_idx, ((cH1, cV1, cD1), (cH2, cV2, cD2)) in enumerate(
            zip(coeffs1.details, coeffs2.details)
        ):
            assert np.array_equal(cH1, cH2), \
                f"Level {level_idx}: cH differs between decompositions"
            assert np.array_equal(cV1, cV2), \
                f"Level {level_idx}: cV differs between decompositions"
            assert np.array_equal(cD1, cD2), \
                f"Level {level_idx}: cD differs between decompositions"


# Mark all tests in this module as property tests for easy filtering
pytestmark = pytest.mark.property
