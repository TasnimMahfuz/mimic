"""
Unit tests for comprehensive visualization generator.

Tests the VisualizationGenerator class that generates all required
visualizations for the MIMIC analysis pipeline.

**Validates: Requirements 11.1-11.16, 13.11**
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from app.services.visualization.generator import VisualizationGenerator


@pytest.fixture
def test_image():
    """Create a simple test image."""
    return np.random.rand(128, 128)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_results(test_image):
    """Create sample processing results."""
    return {
        'original_image': test_image,
        'normalized_image': test_image / test_image.max(),
        'wavelet_edges': (np.random.rand(128, 128) > 0.5).astype(np.uint8),
        'curvelet_edges': (np.random.rand(128, 128) > 0.5).astype(np.uint8),
        'reconstructed_wavelet': test_image + np.random.randn(128, 128) * 0.01,
        'reconstructed_curvelet': test_image + np.random.randn(128, 128) * 0.01,
        'radial_energy': np.random.rand(50),
        'coefficient_histogram': np.random.randn(1000),
        'scale_energy': {0: 100.0, 1: 50.0, 2: 25.0},
        'frequency_cone': np.random.rand(128, 128),
        'reconstruction_error': np.random.rand(128, 128) * 0.1,
        'reconstruction_error_curve': {0: 0.01, 1: 0.02, 2: 0.03}
    }


class TestVisualizationGenerator:
    """Test suite for VisualizationGenerator class."""
    
    def test_plot_raw_image(self, test_image, temp_output_dir):
        """Test raw image visualization."""
        generator = VisualizationGenerator()
        output_path = Path(temp_output_dir) / "raw.png"
        
        generator.plot_raw_image(test_image, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_plot_radial_energy(self, temp_output_dir):
        """Test radial energy plot."""
        generator = VisualizationGenerator()
        output_path = Path(temp_output_dir) / "radial_energy.png"
        
        energy = np.random.rand(50)
        generator.plot_radial_energy(energy, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_plot_coefficient_histogram(self, temp_output_dir):
        """Test coefficient histogram plot."""
        generator = VisualizationGenerator()
        output_path = Path(temp_output_dir) / "histogram.png"
        
        coeffs = np.random.randn(1000)
        generator.plot_coefficient_histogram(coeffs, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_plot_scale_energy(self, temp_output_dir):
        """Test scale energy plot."""
        generator = VisualizationGenerator()
        output_path = Path(temp_output_dir) / "scale_energy.png"
        
        energy_per_scale = {0: 100.0, 1: 50.0, 2: 25.0}
        generator.plot_scale_energy(energy_per_scale, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_plot_frequency_cone(self, test_image, temp_output_dir):
        """Test frequency cone plot."""
        generator = VisualizationGenerator()
        output_path = Path(temp_output_dir) / "frequency_cone.png"
        
        generator.plot_frequency_cone(test_image, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_generate_all(self, sample_results, temp_output_dir):
        """Test generation of all visualizations."""
        generator = VisualizationGenerator()
        
        generated_files = generator.generate_all(sample_results, temp_output_dir)
        
        # Verify files were generated
        assert len(generated_files) > 0
        
        for file_path in generated_files:
            path = Path(file_path)
            assert path.exists()
            assert path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
