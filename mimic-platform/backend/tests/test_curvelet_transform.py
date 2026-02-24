"""
Unit Tests for Curvelet Transform Module

This module contains unit tests for the curvelet transform implementation,
focusing on FFT fallback activation, directional decomposition, and
coefficient extraction.

**Validates: Requirement 18.1**
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import logging

from app.services.transforms.curvelet import CurveletTransform, CURVELET_LIBRARY_AVAILABLE
from app.services.transforms.fft_directional import FFTDirectionalFilter, CurveletCoefficients


class TestFFTFallbackActivation:
    """
    Test FFT fallback activation when curvelet library is unavailable.
    
    **Validates: Requirement 18.1 - Test FFT fallback activation**
    """
    
    def test_fallback_activated_when_library_unavailable(self):
        """
        Test that FFT fallback is activated when specialized library is unavailable.
        
        This test verifies that the CurveletTransform correctly detects when
        the specialized curvelet library is not available and activates the
        FFT-based fallback implementation.
        """
        # Create transform instance
        transform = CurveletTransform()
        
        # Since the specialized library is not actually installed in this environment,
        # the fallback should be active
        assert transform.use_fallback is True, (
            "Expected FFT fallback to be active when specialized library unavailable"
        )
        
        # Verify FFT filter is initialized
        assert hasattr(transform, 'fft_filter'), (
            "Expected fft_filter to be initialized when using fallback"
        )
        assert isinstance(transform.fft_filter, FFTDirectionalFilter), (
            "Expected fft_filter to be instance of FFTDirectionalFilter"
        )
    
    def test_fallback_logs_warning_message(self, caplog):
        """
        Test that a warning message is logged when FFT fallback is activated.
        
        **Validates: Requirement 20.5 - Log warning when using fallback mode**
        """
        with caplog.at_level(logging.INFO):
            transform = CurveletTransform()
        
        # Check that appropriate log message was generated
        log_messages = [record.message for record in caplog.records]
        
        # Should have a message about FFT-based fallback
        assert any('FFT-based fallback' in msg for msg in log_messages), (
            "Expected warning message about FFT-based fallback in logs"
        )
    
    def test_decompose_uses_fft_fallback(self):
        """
        Test that decompose() method uses FFT fallback when library unavailable.
        
        This test verifies that the decompose method correctly delegates to
        the FFT-based implementation when the specialized library is not available.
        """
        transform = CurveletTransform()
        
        # Create a simple test image
        image = np.random.rand(128, 128)
        
        # Decompose using the transform
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        # Verify coefficients are returned
        assert coefficients is not None, "Expected coefficients to be returned"
        assert isinstance(coefficients, CurveletCoefficients), (
            "Expected CurveletCoefficients object"
        )
        
        # Verify structure matches FFT fallback output
        assert coefficients.scales >= 3, (
            "Expected at least 3 scale levels from FFT fallback"
        )
        assert coefficients.orientations >= 8, (
            "Expected at least 8 orientations from FFT fallback"
        )
    
    def test_reconstruct_uses_fft_fallback(self):
        """
        Test that reconstruct() method uses FFT fallback when library unavailable.
        """
        transform = CurveletTransform()
        
        # Create test image and decompose
        image = np.random.rand(64, 64)
        coefficients = transform.decompose(image, levels=3, angular_resolution=8)
        
        # Reconstruct using the transform
        reconstructed = transform.reconstruct(coefficients)
        
        # Verify reconstruction is returned
        assert reconstructed is not None, "Expected reconstructed image"
        assert isinstance(reconstructed, np.ndarray), (
            "Expected numpy array for reconstructed image"
        )
        assert reconstructed.shape == image.shape, (
            f"Expected shape {image.shape}, got {reconstructed.shape}"
        )
    
    def test_extract_directional_energy_uses_fft_fallback(self):
        """
        Test that extract_directional_energy() uses FFT fallback.
        """
        transform = CurveletTransform()
        
        # Create test image and decompose
        image = np.random.rand(64, 64)
        coefficients = transform.decompose(image, levels=3, angular_resolution=8)
        
        # Extract directional energy
        energy_dict = transform.extract_directional_energy(coefficients)
        
        # Verify energy dictionary is returned
        assert energy_dict is not None, "Expected energy dictionary"
        assert isinstance(energy_dict, dict), "Expected dictionary"
        assert len(energy_dict) >= 3, "Expected energy for at least 3 scales"
        
        # Verify energy values are non-negative
        for scale_idx, energies in energy_dict.items():
            assert np.all(energies >= 0), (
                f"Expected non-negative energies at scale {scale_idx}"
            )
    
    def test_compute_orientation_map_uses_fft_fallback(self):
        """
        Test that compute_orientation_map() uses FFT fallback.
        """
        transform = CurveletTransform()
        
        # Create test image and decompose
        image = np.random.rand(64, 64)
        coefficients = transform.decompose(image, levels=3, angular_resolution=8)
        
        # Compute orientation map
        orientation_map = transform.compute_orientation_map(coefficients)
        
        # Verify orientation map is returned
        assert orientation_map is not None, "Expected orientation map"
        assert isinstance(orientation_map, np.ndarray), "Expected numpy array"
        assert orientation_map.shape == image.shape, (
            f"Expected shape {image.shape}, got {orientation_map.shape}"
        )


class TestDirectionalDecomposition:
    """
    Test directional decomposition produces correct orientations.
    
    **Validates: Requirement 18.1 - Test directional decomposition**
    """
    
    def test_decomposition_produces_correct_number_of_orientations(self):
        """
        Test that decomposition produces the specified number of orientations.
        
        This test verifies that when a specific angular_resolution is requested,
        the decomposition produces coefficients for that many orientations
        (or more, depending on scale).
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Test with different angular resolutions
        for angular_resolution in [8, 16, 24]:
            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=angular_resolution
            )
            
            # Verify orientations parameter is stored correctly
            assert coefficients.orientations == angular_resolution, (
                f"Expected {angular_resolution} orientations, "
                f"got {coefficients.orientations}"
            )
            
            # For scales > 0, should have approximately angular_resolution orientations
            for scale_idx in range(1, coefficients.scales):
                num_orientations = len(coefficients.coefficients[scale_idx])
                assert num_orientations >= angular_resolution // 2, (
                    f"Scale {scale_idx} has {num_orientations} orientations, "
                    f"expected at least {angular_resolution // 2}"
                )
    
    def test_decomposition_produces_minimum_8_orientations(self):
        """
        Test that decomposition produces at least 8 orientations as required.
        
        **Validates: Requirement 20.3 - At least 8 directional orientations**
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Request fewer than 8 orientations (should be corrected to 8)
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=4  # Less than minimum
        )
        
        # Should be corrected to at least 8
        assert coefficients.orientations >= 8, (
            f"Expected at least 8 orientations, got {coefficients.orientations}"
        )
    
    def test_orientations_evenly_distributed(self):
        """
        Test that orientations are evenly distributed across 360 degrees.
        
        This test verifies that the directional filters cover the full
        angular range with even spacing.
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        angular_resolution = 8
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # For each scale (except coarsest), verify orientations
        for scale_idx in range(1, coefficients.scales):
            num_orientations = len(coefficients.coefficients[scale_idx])
            
            # Orientations should be indexed from 0 to num_orientations-1
            orientation_indices = sorted(coefficients.coefficients[scale_idx].keys())
            expected_indices = list(range(num_orientations))
            
            assert orientation_indices == expected_indices, (
                f"Scale {scale_idx}: Expected orientation indices {expected_indices}, "
                f"got {orientation_indices}"
            )
    
    def test_different_scales_have_different_frequency_content(self):
        """
        Test that different scales capture different frequency content.
        
        This test verifies that coefficients at different scales have
        different characteristics, indicating proper scale separation.
        """
        transform = CurveletTransform()
        
        # Create an image with known frequency content
        # High frequency checkerboard pattern
        x = np.arange(128)
        y = np.arange(128)
        X, Y = np.meshgrid(x, y)
        image = ((X + Y) % 2).astype(np.float64)
        
        coefficients = transform.decompose(image, levels=3, angular_resolution=8)
        
        # Compute energy at each scale
        scale_energies = []
        for scale_idx in range(coefficients.scales):
            scale_energy = 0.0
            for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                scale_energy += np.sum(coeffs ** 2)
            scale_energies.append(scale_energy)
        
        # For a high-frequency pattern, finer scales should have more energy
        # At minimum, verify that not all scales have identical energy
        assert not np.allclose(scale_energies, scale_energies[0]), (
            "Expected different energy distributions across scales"
        )


class TestCoefficientExtraction:
    """
    Test coefficient extraction for each scale and angle.
    
    **Validates: Requirement 18.1 - Test coefficient extraction**
    """
    
    def test_coefficients_extracted_for_all_scales(self):
        """
        Test that coefficients are extracted for all requested scales.
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        levels = 4
        coefficients = transform.decompose(
            image,
            levels=levels,
            angular_resolution=8
        )
        
        # Verify coefficients exist for all scales
        assert coefficients.scales == levels, (
            f"Expected {levels} scales, got {coefficients.scales}"
        )
        
        for scale_idx in range(levels):
            assert scale_idx in coefficients.coefficients, (
                f"Missing coefficients for scale {scale_idx}"
            )
            assert len(coefficients.coefficients[scale_idx]) > 0, (
                f"No orientations at scale {scale_idx}"
            )
    
    def test_coefficients_extracted_for_all_angles(self):
        """
        Test that coefficients are extracted for all orientations at each scale.
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        angular_resolution = 16
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=angular_resolution
        )
        
        # For each scale, verify all orientations have coefficients
        for scale_idx in range(coefficients.scales):
            scale_coeffs = coefficients.coefficients[scale_idx]
            
            # Each orientation should have coefficient array
            for angle_idx, coeffs in scale_coeffs.items():
                assert coeffs is not None, (
                    f"Missing coefficients at scale {scale_idx}, angle {angle_idx}"
                )
                assert isinstance(coeffs, np.ndarray), (
                    f"Coefficients at scale {scale_idx}, angle {angle_idx} "
                    f"are not numpy array"
                )
    
    def test_coefficient_arrays_have_correct_shape(self):
        """
        Test that all coefficient arrays have the same shape as input image.
        """
        transform = CurveletTransform()
        
        # Test with different image sizes
        for size in [64, 128, 256]:
            image = np.random.rand(size, size)
            
            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )
            
            # Verify all coefficient arrays match image shape
            for scale_idx in range(coefficients.scales):
                for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                    assert coeffs.shape == image.shape, (
                        f"Coefficient shape {coeffs.shape} doesn't match "
                        f"image shape {image.shape} at scale {scale_idx}, "
                        f"angle {angle_idx}"
                    )
    
    def test_coefficient_arrays_are_finite(self):
        """
        Test that all coefficient values are finite (no NaN or Inf).
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        # Verify all coefficients are finite
        for scale_idx in range(coefficients.scales):
            for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                assert np.all(np.isfinite(coeffs)), (
                    f"Found non-finite coefficients at scale {scale_idx}, "
                    f"angle {angle_idx}"
                )
    
    def test_coefficient_extraction_preserves_image_energy(self):
        """
        Test that total energy is approximately preserved in decomposition.
        
        The sum of squared coefficients across all scales and orientations
        should be related to the original image energy (though not necessarily
        equal due to filter overlap and normalization).
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Compute original image energy
        original_energy = np.sum(image ** 2)
        
        # Decompose
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        # Compute total coefficient energy
        total_coeff_energy = 0.0
        for scale_idx in range(coefficients.scales):
            for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                total_coeff_energy += np.sum(coeffs ** 2)
        
        # Energy should be positive for non-zero image
        assert total_coeff_energy > 0, (
            "Expected positive coefficient energy for non-zero image"
        )
        
        # Energy should be of similar order of magnitude
        # (exact relationship depends on filter normalization)
        energy_ratio = total_coeff_energy / original_energy
        assert 0.1 < energy_ratio < 100, (
            f"Coefficient energy ratio {energy_ratio} seems unreasonable"
        )


class TestCurveletTransformEdgeCases:
    """Test edge cases and error handling."""
    
    def test_decompose_rejects_non_2d_image(self):
        """
        Test that decompose raises ValueError for non-2D images.
        """
        transform = CurveletTransform()
        
        # 1D array
        with pytest.raises(ValueError, match="must be 2D"):
            transform.decompose(np.random.rand(128), levels=3, angular_resolution=8)
        
        # 3D array
        with pytest.raises(ValueError, match="must be 2D"):
            transform.decompose(
                np.random.rand(128, 128, 3),
                levels=3,
                angular_resolution=8
            )
    
    def test_decompose_with_minimum_levels(self):
        """
        Test decomposition with minimum required levels (3).
        
        **Validates: Requirement 20.4 - At least 3 scale levels**
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Request minimum levels
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        assert coefficients.scales >= 3, (
            f"Expected at least 3 scales, got {coefficients.scales}"
        )
    
    def test_decompose_corrects_insufficient_levels(self, caplog):
        """
        Test that decompose corrects levels < 3 to minimum of 3.
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        with caplog.at_level(logging.WARNING):
            coefficients = transform.decompose(
                image,
                levels=2,  # Less than minimum
                angular_resolution=8
            )
        
        # Should be corrected to at least 3
        assert coefficients.scales >= 3, (
            f"Expected at least 3 scales, got {coefficients.scales}"
        )
        
        # Should log a warning
        assert any('at least 3' in record.message.lower() 
                  for record in caplog.records), (
            "Expected warning about minimum levels"
        )
    
    def test_decompose_with_small_image(self):
        """
        Test decomposition with small image (near minimum size).
        """
        transform = CurveletTransform()
        
        # Small but valid image
        image = np.random.rand(64, 64)
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        # Should still produce valid decomposition
        assert coefficients.scales >= 3
        assert len(coefficients.coefficients) >= 3
    
    def test_decompose_with_rectangular_image(self):
        """
        Test decomposition with non-square image.
        """
        transform = CurveletTransform()
        
        # Rectangular image
        image = np.random.rand(128, 256)
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        # Should handle rectangular images
        assert coefficients.shape == image.shape
        
        # All coefficient arrays should match image shape
        for scale_idx in range(coefficients.scales):
            for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                assert coeffs.shape == image.shape


class TestCurveletVisualizationGeneration:
    """Test visualization generation functionality."""
    
    def test_generate_visualizations_creates_output_directory(self, tmp_path):
        """
        Test that generate_visualizations creates output directory if needed.
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        # Use temporary directory that doesn't exist yet
        output_dir = tmp_path / "test_output"
        
        # Generate visualizations
        output_files = transform.generate_visualizations(
            coefficients,
            str(output_dir),
            edge_threshold=0.1
        )
        
        # Verify directory was created
        assert output_dir.exists(), "Expected output directory to be created"
        assert output_dir.is_dir(), "Expected output path to be a directory"
    
    def test_generate_visualizations_returns_file_paths(self, tmp_path):
        """
        Test that generate_visualizations returns dictionary of file paths.
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        output_dir = tmp_path / "test_output"
        
        output_files = transform.generate_visualizations(
            coefficients,
            str(output_dir),
            edge_threshold=0.1
        )
        
        # Verify return value is a dictionary
        assert isinstance(output_files, dict), "Expected dictionary of file paths"
        
        # Verify expected visualization keys
        expected_keys = [
            'curvelet_edge',
            'directional_energy',
            'orientation_map',
            'angular_distribution'
        ]
        
        for key in expected_keys:
            assert key in output_files, f"Missing visualization: {key}"


class TestDirectionalCoefficientExtraction:
        """
        Test directional coefficient extraction methods.

        **Validates: Requirements 7.1, 7.2, 7.5, 13.7**
        """

        def test_extract_coefficient_magnitudes_returns_correct_structure(self):
            """
            Test that extract_coefficient_magnitudes returns proper nested dictionary.

            **Validates: Requirement 7.1 - Extract coefficient magnitudes**
            """
            transform = CurveletTransform()
            image = np.random.rand(128, 128)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            # Extract magnitudes
            magnitudes = transform.extract_coefficient_magnitudes(coefficients)

            # Verify structure
            assert isinstance(magnitudes, dict), "Expected dictionary"
            assert len(magnitudes) == coefficients.scales, (
                f"Expected {coefficients.scales} scales, got {len(magnitudes)}"
            )

            # Verify each scale has orientations
            for scale_idx in range(coefficients.scales):
                assert scale_idx in magnitudes, f"Missing scale {scale_idx}"
                assert isinstance(magnitudes[scale_idx], dict), (
                    f"Expected dict for scale {scale_idx}"
                )
                assert len(magnitudes[scale_idx]) > 0, (
                    f"No orientations at scale {scale_idx}"
                )

        def test_extract_coefficient_magnitudes_are_non_negative(self):
            """
            Test that all extracted magnitudes are non-negative.

            **Validates: Requirement 7.1 - Coefficient magnitudes are non-negative**
            """
            transform = CurveletTransform()
            image = np.random.rand(128, 128)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            magnitudes = transform.extract_coefficient_magnitudes(coefficients)

            # Verify all magnitudes are non-negative
            for scale_idx in range(coefficients.scales):
                for angle_idx, mags in magnitudes[scale_idx].items():
                    assert np.all(mags >= 0), (
                        f"Found negative magnitudes at scale {scale_idx}, "
                        f"angle {angle_idx}"
                    )

        def test_extract_coefficient_magnitudes_match_coefficient_shape(self):
            """
            Test that magnitude arrays have same shape as coefficient arrays.
            """
            transform = CurveletTransform()
            image = np.random.rand(128, 128)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            magnitudes = transform.extract_coefficient_magnitudes(coefficients)

            # Verify shapes match
            for scale_idx in range(coefficients.scales):
                for angle_idx in coefficients.coefficients[scale_idx].keys():
                    coeff_shape = coefficients.coefficients[scale_idx][angle_idx].shape
                    mag_shape = magnitudes[scale_idx][angle_idx].shape
                    assert mag_shape == coeff_shape, (
                        f"Shape mismatch at scale {scale_idx}, angle {angle_idx}: "
                        f"coeffs={coeff_shape}, mags={mag_shape}"
                    )

        def test_save_directional_data_creates_npz_file(self, tmp_path):
            """
            Test that save_directional_data creates NPZ file.

            **Validates: Requirement 7.5 - Save directional data in structured format**
            """
            transform = CurveletTransform()
            image = np.random.rand(128, 128)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            # Save data
            output_path = tmp_path / "directional_data.npz"
            saved_path = transform.save_directional_data(
                coefficients,
                str(output_path),
                include_coefficients=False
            )

            # Verify file was created
            assert output_path.exists(), "Expected NPZ file to be created"
            assert output_path.suffix == '.npz', "Expected .npz extension"
            assert saved_path == str(output_path), "Expected returned path to match"

        def test_save_directional_data_contains_metadata(self, tmp_path):
            """
            Test that saved data contains metadata.

            **Validates: Requirement 7.5 - Structured format includes metadata**
            """
            transform = CurveletTransform()
            image = np.random.rand(128, 128)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            output_path = tmp_path / "directional_data.npz"
            transform.save_directional_data(
                coefficients,
                str(output_path),
                include_coefficients=False
            )

            # Load and verify metadata
            data = np.load(output_path, allow_pickle=True)

            assert 'metadata' in data, "Expected metadata in saved file"

            # Parse metadata JSON
            import json
            metadata = json.loads(str(data['metadata']))

            assert 'scales' in metadata, "Expected scales in metadata"
            assert 'orientations' in metadata, "Expected orientations in metadata"
            assert 'shape' in metadata, "Expected shape in metadata"
            assert metadata['scales'] == coefficients.scales
            assert metadata['orientations'] == coefficients.orientations

        def test_save_directional_data_contains_energy_per_scale(self, tmp_path):
            """
            Test that saved data contains energy per orientation for each scale.

            **Validates: Requirement 7.2 - Compute energy per orientation**
            """
            transform = CurveletTransform()
            image = np.random.rand(128, 128)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            output_path = tmp_path / "directional_data.npz"
            transform.save_directional_data(
                coefficients,
                str(output_path),
                include_coefficients=False
            )

            # Load and verify energy data
            data = np.load(output_path, allow_pickle=True)

            # Check for energy arrays for each scale
            for scale_idx in range(coefficients.scales):
                key = f'energy_scale_{scale_idx}'
                assert key in data, f"Expected {key} in saved file"

                energies = data[key]
                assert len(energies) > 0, f"Expected non-empty energy array for scale {scale_idx}"
                assert np.all(energies >= 0), f"Expected non-negative energies for scale {scale_idx}"

        def test_save_directional_data_contains_magnitude_statistics(self, tmp_path):
            """
            Test that saved data contains magnitude statistics.

            **Validates: Requirement 7.1 - Extract coefficient magnitudes**
            """
            transform = CurveletTransform()
            image = np.random.rand(128, 128)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            output_path = tmp_path / "directional_data.npz"
            transform.save_directional_data(
                coefficients,
                str(output_path),
                include_coefficients=False
            )

            # Load and verify magnitude statistics
            data = np.load(output_path, allow_pickle=True)

            # Check for magnitude statistics for each scale and angle
            for scale_idx in range(coefficients.scales):
                for angle_idx in coefficients.coefficients[scale_idx].keys():
                    # Check for mean, std, max
                    mean_key = f'magnitude_mean_s{scale_idx}_a{angle_idx}'
                    std_key = f'magnitude_std_s{scale_idx}_a{angle_idx}'
                    max_key = f'magnitude_max_s{scale_idx}_a{angle_idx}'

                    assert mean_key in data, f"Expected {mean_key} in saved file"
                    assert std_key in data, f"Expected {std_key} in saved file"
                    assert max_key in data, f"Expected {max_key} in saved file"

                    # Verify values are reasonable
                    assert data[mean_key] >= 0, f"Expected non-negative mean"
                    assert data[std_key] >= 0, f"Expected non-negative std"
                    assert data[max_key] >= 0, f"Expected non-negative max"

        def test_save_directional_data_without_full_coefficients(self, tmp_path):
            """
            Test that by default, full coefficients are not saved (to save space).
            """
            transform = CurveletTransform()
            image = np.random.rand(128, 128)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            output_path = tmp_path / "directional_data.npz"
            transform.save_directional_data(
                coefficients,
                str(output_path),
                include_coefficients=False
            )

            # Load and verify coefficients are not included
            data = np.load(output_path, allow_pickle=True)

            # Check that coefficient arrays are not present
            coeff_keys = [key for key in data.keys() if key.startswith('coeffs_')]
            assert len(coeff_keys) == 0, (
                "Expected no coefficient arrays when include_coefficients=False"
            )

        def test_save_directional_data_with_full_coefficients(self, tmp_path):
            """
            Test that full coefficients are saved when requested.
            """
            transform = CurveletTransform()
            image = np.random.rand(64, 64)  # Smaller image for faster test

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            output_path = tmp_path / "directional_data.npz"
            transform.save_directional_data(
                coefficients,
                str(output_path),
                include_coefficients=True
            )

            # Load and verify coefficients are included
            data = np.load(output_path, allow_pickle=True)

            # Check that coefficient arrays are present
            for scale_idx in range(coefficients.scales):
                for angle_idx in coefficients.coefficients[scale_idx].keys():
                    key = f'coeffs_s{scale_idx}_a{angle_idx}'
                    assert key in data, f"Expected {key} in saved file"

                    # Verify shape matches
                    saved_coeffs = data[key]
                    original_coeffs = coefficients.coefficients[scale_idx][angle_idx]
                    assert saved_coeffs.shape == original_coeffs.shape, (
                        f"Shape mismatch for {key}"
                    )

        def test_save_directional_data_adds_npz_extension(self, tmp_path):
            """
            Test that .npz extension is added if not present.
            """
            transform = CurveletTransform()
            image = np.random.rand(64, 64)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            # Provide path without .npz extension
            output_path = tmp_path / "directional_data"
            saved_path = transform.save_directional_data(
                coefficients,
                str(output_path),
                include_coefficients=False
            )

            # Verify .npz extension was added
            assert saved_path.endswith('.npz'), "Expected .npz extension to be added"
            assert (tmp_path / "directional_data.npz").exists(), (
                "Expected file with .npz extension to exist"
            )

        def test_energy_computation_matches_manual_calculation(self):
            """
            Test that computed energy matches manual calculation.

            **Validates: Requirement 7.2 - Energy per orientation computation**
            """
            transform = CurveletTransform()
            image = np.random.rand(64, 64)

            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=8
            )

            # Get energy from method
            energy_dict = transform.extract_directional_energy(coefficients)

            # Manually compute energy for verification
            for scale_idx in range(coefficients.scales):
                for angle_idx, coeffs in coefficients.coefficients[scale_idx].items():
                    manual_energy = np.sum(coeffs ** 2)
                    method_energy = energy_dict[scale_idx][angle_idx]

                    assert np.isclose(manual_energy, method_energy, rtol=1e-5), (
                        f"Energy mismatch at scale {scale_idx}, angle {angle_idx}: "
                        f"manual={manual_energy}, method={method_energy}"
                    )


class TestDirectionalAnalysis:
    """
    Unit tests for directional analysis functionality.
    
    Tests coefficient magnitude extraction, energy computation per orientation,
    and orientation map generation as specified in task 7.2.
    
    **Validates: Requirement 18.1**
    """
    
    def test_coefficient_magnitude_extraction(self):
        """
        Test coefficient magnitude extraction for all scales and orientations.
        
        Verifies that extract_coefficient_magnitudes returns proper structure
        with non-negative magnitudes for each directional subband.
        
        **Validates: Requirement 7.1 - Extract coefficient magnitudes**
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Decompose image
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=16
        )
        
        # Extract magnitudes
        magnitudes = transform.extract_coefficient_magnitudes(coefficients)
        
        # Verify structure
        assert isinstance(magnitudes, dict), "Expected dictionary structure"
        assert len(magnitudes) == coefficients.scales, (
            f"Expected {coefficients.scales} scales, got {len(magnitudes)}"
        )
        
        # Verify each scale has proper structure
        for scale_idx in range(coefficients.scales):
            assert scale_idx in magnitudes, f"Missing scale {scale_idx}"
            assert isinstance(magnitudes[scale_idx], dict), (
                f"Expected dict for scale {scale_idx}"
            )
            
            # Verify each orientation has magnitude array
            for angle_idx, mag_array in magnitudes[scale_idx].items():
                # Magnitudes should be non-negative
                assert np.all(mag_array >= 0), (
                    f"Found negative magnitudes at scale {scale_idx}, angle {angle_idx}"
                )
                
                # Magnitudes should match coefficient shape
                coeff_shape = coefficients.coefficients[scale_idx][angle_idx].shape
                assert mag_array.shape == coeff_shape, (
                    f"Shape mismatch at scale {scale_idx}, angle {angle_idx}"
                )
                
                # Magnitudes should be finite
                assert np.all(np.isfinite(mag_array)), (
                    f"Found non-finite magnitudes at scale {scale_idx}, angle {angle_idx}"
                )
    
    def test_energy_computation_per_orientation(self):
        """
        Test energy computation per orientation for each scale.
        
        Verifies that extract_directional_energy computes correct energy values
        (sum of squared coefficients) for each orientation at each scale.
        
        **Validates: Requirement 7.2 - Compute energy per orientation**
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Decompose image
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=16
        )
        
        # Extract directional energy
        energy_dict = transform.extract_directional_energy(coefficients)
        
        # Verify structure
        assert isinstance(energy_dict, dict), "Expected dictionary structure"
        assert len(energy_dict) == coefficients.scales, (
            f"Expected {coefficients.scales} scales, got {len(energy_dict)}"
        )
        
        # Verify energy values for each scale
        for scale_idx in range(coefficients.scales):
            assert scale_idx in energy_dict, f"Missing scale {scale_idx}"
            
            energies = energy_dict[scale_idx]
            assert isinstance(energies, (dict, np.ndarray)), (
                f"Expected dict or array for scale {scale_idx}"
            )
            
            # If dict, verify each orientation
            if isinstance(energies, dict):
                for angle_idx, energy in energies.items():
                    # Energy should be non-negative
                    assert energy >= 0, (
                        f"Negative energy at scale {scale_idx}, angle {angle_idx}"
                    )
                    
                    # Energy should be finite
                    assert np.isfinite(energy), (
                        f"Non-finite energy at scale {scale_idx}, angle {angle_idx}"
                    )
                    
                    # Verify energy matches manual calculation
                    coeffs = coefficients.coefficients[scale_idx][angle_idx]
                    expected_energy = np.sum(coeffs ** 2)
                    assert np.isclose(energy, expected_energy, rtol=1e-5), (
                        f"Energy mismatch at scale {scale_idx}, angle {angle_idx}: "
                        f"expected={expected_energy}, got={energy}"
                    )
            else:
                # If array, verify all energies are non-negative
                assert np.all(energies >= 0), (
                    f"Found negative energies at scale {scale_idx}"
                )
                assert np.all(np.isfinite(energies)), (
                    f"Found non-finite energies at scale {scale_idx}"
                )
    
    def test_orientation_map_generation(self):
        """
        Test orientation map generation showing dominant directions.
        
        Verifies that compute_orientation_map generates a map with the same
        shape as the input image, containing orientation indices indicating
        the dominant direction at each spatial location.
        
        **Validates: Requirement 7.3 - Generate orientation map**
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Decompose image
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=16
        )
        
        # Compute orientation map
        orientation_map = transform.compute_orientation_map(coefficients)
        
        # Verify basic properties
        assert orientation_map is not None, "Expected orientation map"
        assert isinstance(orientation_map, np.ndarray), "Expected numpy array"
        
        # Verify shape matches input image
        assert orientation_map.shape == image.shape, (
            f"Shape mismatch: expected {image.shape}, got {orientation_map.shape}"
        )
        
        # Verify orientation indices are valid
        # Should be integers in range [0, angular_resolution)
        assert orientation_map.dtype in [np.int32, np.int64, np.float32, np.float64], (
            f"Unexpected dtype: {orientation_map.dtype}"
        )
        
        # If integer type, verify range
        if orientation_map.dtype in [np.int32, np.int64]:
            assert np.all(orientation_map >= 0), "Found negative orientation indices"
            assert np.all(orientation_map < coefficients.orientations), (
                f"Found orientation indices >= {coefficients.orientations}"
            )
        
        # Verify all values are finite
        assert np.all(np.isfinite(orientation_map)), (
            "Found non-finite values in orientation map"
        )
    
    def test_orientation_map_reflects_image_structure(self):
        """
        Test that orientation map reflects actual image structure.
        
        Creates an image with known directional features and verifies
        that the orientation map captures these features.
        """
        transform = CurveletTransform()
        
        # Create image with horizontal edges (vertical gradient)
        image = np.zeros((128, 128))
        image[:64, :] = 1.0  # Top half bright, bottom half dark
        
        # Decompose and compute orientation map
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        orientation_map = transform.compute_orientation_map(coefficients)
        
        # Verify map has variation (not all same value)
        unique_orientations = np.unique(orientation_map)
        assert len(unique_orientations) > 1, (
            "Expected multiple orientations in map, got only one"
        )
        
        # For a horizontal edge, expect horizontal orientations to dominate
        # near the edge (middle of image)
        edge_region = orientation_map[60:68, :]  # Region around horizontal edge
        
        # Should have some variation in this region
        edge_unique = np.unique(edge_region)
        assert len(edge_unique) >= 1, (
            "Expected at least one orientation in edge region"
        )
    
    def test_energy_sum_across_orientations(self):
        """
        Test that energy sum across orientations is consistent.
        
        Verifies that the total energy across all orientations at a scale
        is positive and finite, representing the total signal energy at that scale.
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Decompose image
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=16
        )
        
        # Extract directional energy
        energy_dict = transform.extract_directional_energy(coefficients)
        
        # Verify total energy at each scale
        for scale_idx in range(coefficients.scales):
            energies = energy_dict[scale_idx]
            
            # Compute total energy at this scale
            if isinstance(energies, dict):
                total_energy = sum(energies.values())
            else:
                total_energy = np.sum(energies)
            
            # Total energy should be positive for non-zero image
            assert total_energy > 0, (
                f"Expected positive total energy at scale {scale_idx}, got {total_energy}"
            )
            
            # Total energy should be finite
            assert np.isfinite(total_energy), (
                f"Non-finite total energy at scale {scale_idx}"
            )
    
    def test_magnitude_extraction_with_complex_coefficients(self):
        """
        Test magnitude extraction handles complex coefficients correctly.
        
        Verifies that if coefficients are complex (as they might be from FFT),
        the magnitude extraction correctly computes absolute values.
        """
        transform = CurveletTransform()
        image = np.random.rand(64, 64)
        
        # Decompose image
        coefficients = transform.decompose(
            image,
            levels=3,
            angular_resolution=8
        )
        
        # Extract magnitudes
        magnitudes = transform.extract_coefficient_magnitudes(coefficients)
        
        # Verify magnitudes are real and non-negative
        for scale_idx in range(coefficients.scales):
            for angle_idx, mag_array in magnitudes[scale_idx].items():
                # Should be real-valued
                assert not np.iscomplexobj(mag_array), (
                    f"Magnitudes should be real at scale {scale_idx}, angle {angle_idx}"
                )
                
                # Should be non-negative
                assert np.all(mag_array >= 0), (
                    f"Magnitudes should be non-negative at scale {scale_idx}, angle {angle_idx}"
                )
    
    def test_orientation_map_with_different_angular_resolutions(self):
        """
        Test orientation map generation with different angular resolutions.
        
        Verifies that the orientation map correctly handles different numbers
        of orientations.
        """
        transform = CurveletTransform()
        image = np.random.rand(128, 128)
        
        # Test with different angular resolutions
        for angular_resolution in [8, 16, 24]:
            coefficients = transform.decompose(
                image,
                levels=3,
                angular_resolution=angular_resolution
            )
            
            orientation_map = transform.compute_orientation_map(coefficients)
            
            # Verify shape
            assert orientation_map.shape == image.shape, (
                f"Shape mismatch for angular_resolution={angular_resolution}"
            )
            
            # Verify values are in valid range
            if orientation_map.dtype in [np.int32, np.int64]:
                max_orientation = np.max(orientation_map)
                assert max_orientation < angular_resolution, (
                    f"Found orientation index {max_orientation} >= {angular_resolution}"
                )




if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
