"""
Unit tests for curvelet visualization generation.

Tests the visualization methods for curvelet transform results,
including edge detection, directional energy, orientation maps,
and angular distribution plots.

**Validates: Requirements 6.6, 6.7, 7.3, 7.4, 11.4, 11.6, 11.10**
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from app.services.transforms.curvelet import CurveletTransform
from app.services.visualization.curvelet_visualizer import CurveletVisualizer


@pytest.fixture
def test_image():
    """Create a simple test image with edges."""
    size = 128
    image = np.zeros((size, size))
    
    # Add horizontal edge
    image[size//2:size//2+5, :] = 1.0
    
    # Add vertical edge
    image[:, size//3:size//3+5] = 0.8
    
    # Add diagonal
    for i in range(size):
        if i < size:
            image[i, i] = 0.6
    
    return image


@pytest.fixture
def curvelet_coefficients(test_image):
    """Generate curvelet coefficients from test image."""
    transform = CurveletTransform()
    return transform.decompose(test_image, levels=3, angular_resolution=8)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestCurveletVisualizer:
    """Test suite for CurveletVisualizer class."""
    
    def test_generate_curvelet_edge(self, curvelet_coefficients, temp_output_dir):
        """
        Test curvelet edge detection visualization generation.
        
        **Validates: Requirements 6.6, 11.4**
        """
        visualizer = CurveletVisualizer()
        output_path = Path(temp_output_dir) / "curvelet_edge.png"
        
        # Generate edge map
        edge_map = visualizer.generate_curvelet_edge(
            curvelet_coefficients,
            threshold=0.1,
            output_path=str(output_path)
        )
        
        # Verify edge map properties
        assert edge_map.shape == curvelet_coefficients.shape
        assert edge_map.dtype == np.uint8
        assert np.all((edge_map == 0) | (edge_map == 1))  # Binary
        
        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_generate_directional_energy(self, curvelet_coefficients, temp_output_dir):
        """
        Test directional energy distribution visualization.
        
        **Validates: Requirements 6.7, 11.6**
        """
        visualizer = CurveletVisualizer()
        output_path = Path(temp_output_dir) / "directional_energy.png"
        
        # Generate energy distribution
        energy_dict = visualizer.generate_directional_energy(
            curvelet_coefficients,
            output_path=str(output_path)
        )
        
        # Verify energy dictionary structure
        assert len(energy_dict) == curvelet_coefficients.scales
        
        for scale_idx in range(curvelet_coefficients.scales):
            assert scale_idx in energy_dict
            energies = energy_dict[scale_idx]
            assert len(energies) == curvelet_coefficients.orientations
            assert np.all(energies >= 0)  # Energy is non-negative
        
        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_generate_orientation_map(self, curvelet_coefficients, temp_output_dir):
        """
        Test orientation map visualization generation.
        
        **Validates: Requirements 7.3, 11.6**
        """
        visualizer = CurveletVisualizer()
        output_path = Path(temp_output_dir) / "orientation_map.png"
        
        # Generate orientation map
        orientation_map = visualizer.generate_orientation_map(
            curvelet_coefficients,
            output_path=str(output_path)
        )
        
        # Verify orientation map properties
        assert orientation_map.shape == curvelet_coefficients.shape
        assert orientation_map.dtype == np.int32
        assert np.all(orientation_map >= 0)
        assert np.all(orientation_map < curvelet_coefficients.orientations)
        
        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_generate_angular_distribution(self, curvelet_coefficients, temp_output_dir):
        """
        Test angular distribution plot generation.
        
        **Validates: Requirements 7.4, 11.10**
        """
        visualizer = CurveletVisualizer()
        output_path = Path(temp_output_dir) / "angular_distribution.png"
        
        # Generate angular distribution
        angular_energy = visualizer.generate_angular_distribution(
            curvelet_coefficients,
            output_path=str(output_path)
        )
        
        # Verify angular energy properties
        assert len(angular_energy) == curvelet_coefficients.orientations
        assert np.all(angular_energy >= 0)  # Energy is non-negative
        assert np.sum(angular_energy) > 0  # Total energy is positive
        
        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_edge_threshold_effect(self, curvelet_coefficients, temp_output_dir):
        """Test that different thresholds produce different edge maps."""
        visualizer = CurveletVisualizer()
        
        # Generate with low threshold
        edge_low = visualizer.generate_curvelet_edge(
            curvelet_coefficients,
            threshold=0.05
        )
        
        # Generate with high threshold
        edge_high = visualizer.generate_curvelet_edge(
            curvelet_coefficients,
            threshold=0.5
        )
        
        # Higher threshold should produce fewer edges
        assert np.sum(edge_high) <= np.sum(edge_low)
    
    def test_energy_conservation(self, curvelet_coefficients):
        """Test that total energy is conserved across orientations."""
        visualizer = CurveletVisualizer()
        
        # Get energy distribution
        energy_dict = visualizer.generate_directional_energy(curvelet_coefficients)
        
        # Get angular distribution (total across scales)
        angular_energy = visualizer.generate_angular_distribution(curvelet_coefficients)
        
        # Total energy should match
        total_from_scales = sum(np.sum(energies) for energies in energy_dict.values())
        total_from_angular = np.sum(angular_energy)
        
        # Should be approximately equal (within numerical precision)
        assert np.isclose(total_from_scales, total_from_angular, rtol=1e-10)


class TestCurveletTransformVisualization:
    """Test suite for integrated visualization generation."""
    
    def test_generate_all_visualizations(self, test_image, temp_output_dir):
        """
        Test integrated visualization generation through CurveletTransform.
        
        **Validates: Requirements 6.6, 6.7, 7.3, 7.4, 11.4, 11.6, 11.10**
        """
        transform = CurveletTransform()
        
        # Decompose image
        coefficients = transform.decompose(test_image, levels=3, angular_resolution=8)
        
        # Generate all visualizations
        output_files = transform.generate_visualizations(
            coefficients,
            output_dir=temp_output_dir,
            edge_threshold=0.1
        )
        
        # Verify all expected files are present
        expected_files = [
            'curvelet_edge',
            'directional_energy',
            'orientation_map',
            'angular_distribution'
        ]
        
        for name in expected_files:
            assert name in output_files
            file_path = Path(output_files[name])
            assert file_path.exists()
            assert file_path.stat().st_size > 0
    
    def test_visualization_with_different_resolutions(self, test_image, temp_output_dir):
        """Test visualizations with different angular resolutions."""
        transform = CurveletTransform()
        
        for angular_resolution in [8, 16, 32]:
            # Decompose with specific resolution
            coefficients = transform.decompose(
                test_image,
                levels=3,
                angular_resolution=angular_resolution
            )
            
            # Generate visualizations
            output_subdir = Path(temp_output_dir) / f"res_{angular_resolution}"
            output_files = transform.generate_visualizations(
                coefficients,
                output_dir=str(output_subdir)
            )
            
            # Verify all files created
            assert len(output_files) == 4
            for file_path in output_files.values():
                assert Path(file_path).exists()


class TestVisualizationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_coefficients(self, temp_output_dir):
        """Test handling of empty coefficient structure."""
        from app.services.transforms.fft_directional import CurveletCoefficients
        
        # Create empty coefficients
        empty_coeffs = CurveletCoefficients(
            coefficients={},
            scales=3,
            orientations=8,
            shape=(128, 128)
        )
        
        visualizer = CurveletVisualizer()
        
        # Should not crash, but produce zero edge map
        edge_map = visualizer.generate_curvelet_edge(empty_coeffs)
        assert np.all(edge_map == 0)
    
    def test_single_scale(self, test_image, temp_output_dir):
        """Test visualization with single scale level."""
        transform = CurveletTransform()
        
        # Decompose with single scale
        coefficients = transform.decompose(test_image, levels=1, angular_resolution=8)
        
        # Generate visualizations
        output_files = transform.generate_visualizations(
            coefficients,
            output_dir=temp_output_dir
        )
        
        # Should still generate all visualizations
        assert len(output_files) == 4
        for file_path in output_files.values():
            assert Path(file_path).exists()
    
    def test_small_image(self, temp_output_dir):
        """Test visualization with small image."""
        # Create small test image
        small_image = np.random.rand(32, 32)
        
        transform = CurveletTransform()
        coefficients = transform.decompose(small_image, levels=2, angular_resolution=4)
        
        # Generate visualizations
        output_files = transform.generate_visualizations(
            coefficients,
            output_dir=temp_output_dir
        )
        
        # Should generate all visualizations
        assert len(output_files) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
