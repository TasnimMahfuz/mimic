"""
Unit tests for edge detection functionality in mimic_service.

Tests edge detection from wavelet and curvelet coefficients, difference map
generation, and edge overlay visualization.

**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 11.8, 13.9**
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from app.services.mimic_service import MIMICService
from app.services.transforms.wavelet import WaveletTransform
from app.services.transforms.curvelet import CurveletTransform


class TestEdgeDetection:
    """Test suite for edge detection functionality."""
    
    @pytest.fixture
    def mimic_service(self):
        """Create a MIMICService instance for testing."""
        return MIMICService()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image with clear edges."""
        # Create a 128x128 image with a square (clear edges)
        image = np.zeros((128, 128), dtype=np.float64)
        image[32:96, 32:96] = 1.0
        return image
    
    @pytest.fixture
    def wavelet_coeffs(self, sample_image):
        """Generate wavelet coefficients from sample image."""
        wavelet_transform = WaveletTransform(wavelet='db4')
        return wavelet_transform.decompose(sample_image, levels=3)
    
    @pytest.fixture
    def curvelet_coeffs(self, sample_image):
        """Generate curvelet coefficients from sample image."""
        curvelet_transform = CurveletTransform()
        return curvelet_transform.decompose(sample_image, levels=3, angular_resolution=8)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_detect_edges_from_wavelet_returns_binary_map(
        self, mimic_service, wavelet_coeffs
    ):
        """
        Test that wavelet edge detection returns a binary edge map.
        
        **Validates: Requirement 9.2** - Detect edges from wavelet coefficients
        """
        edges = mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=wavelet_coeffs,
            edge_strength=0.5
        )
        
        # Check that edges is a binary map
        assert edges.ndim == 2
        assert edges.dtype in [np.uint8, np.int32, np.int64]
        assert np.all((edges == 0) | (edges == 1))
        
        # Check that some edges were detected
        assert np.sum(edges) > 0
    
    def test_detect_edges_from_wavelet_applies_threshold(
        self, mimic_service, wavelet_coeffs
    ):
        """
        Test that edge_strength threshold affects edge detection.
        
        **Validates: Requirement 9.1** - Apply edge_strength threshold
        """
        # Lower threshold should detect more edges
        edges_low = mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=wavelet_coeffs,
            edge_strength=0.3
        )
        
        # Higher threshold should detect fewer edges
        edges_high = mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=wavelet_coeffs,
            edge_strength=0.7
        )
        
        # Lower threshold should produce more edge pixels
        assert np.sum(edges_low) >= np.sum(edges_high)
    
    def test_detect_edges_from_wavelet_validates_threshold(
        self, mimic_service, wavelet_coeffs
    ):
        """
        Test that invalid edge_strength values are rejected.
        
        **Validates: Requirement 9.1** - Threshold validation
        """
        # Test threshold below valid range
        with pytest.raises(ValueError, match="edge_strength must be in"):
            mimic_service.detect_edges_from_wavelet(
                wavelet_coeffs=wavelet_coeffs,
                edge_strength=-0.1
            )
        
        # Test threshold above valid range
        with pytest.raises(ValueError, match="edge_strength must be in"):
            mimic_service.detect_edges_from_wavelet(
                wavelet_coeffs=wavelet_coeffs,
                edge_strength=1.5
            )
    
    def test_detect_edges_from_wavelet_saves_visualization(
        self, mimic_service, wavelet_coeffs, temp_output_dir
    ):
        """
        Test that wavelet edge visualization is saved when output_path provided.
        
        **Validates: Requirement 11.8** - Generate edge visualizations
        """
        output_path = Path(temp_output_dir) / "wavelet_edge.png"
        
        edges = mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=wavelet_coeffs,
            edge_strength=0.5,
            output_path=str(output_path)
        )
        
        # Check that visualization file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_detect_edges_from_curvelet_returns_binary_map(
        self, mimic_service, curvelet_coeffs
    ):
        """
        Test that curvelet edge detection returns a binary edge map.
        
        **Validates: Requirement 9.3** - Detect edges from curvelet coefficients
        """
        edges = mimic_service.detect_edges_from_curvelet(
            curvelet_coeffs=curvelet_coeffs,
            edge_strength=0.5
        )
        
        # Check that edges is a binary map
        assert edges.ndim == 2
        assert edges.dtype in [np.uint8, np.int32, np.int64]
        assert np.all((edges == 0) | (edges == 1))
        
        # Check that some edges were detected
        assert np.sum(edges) > 0
    
    def test_detect_edges_from_curvelet_applies_threshold(
        self, mimic_service, curvelet_coeffs
    ):
        """
        Test that edge_strength threshold affects curvelet edge detection.
        
        **Validates: Requirement 9.1** - Apply edge_strength threshold
        """
        # Lower threshold should detect more edges
        edges_low = mimic_service.detect_edges_from_curvelet(
            curvelet_coeffs=curvelet_coeffs,
            edge_strength=0.3
        )
        
        # Higher threshold should detect fewer edges
        edges_high = mimic_service.detect_edges_from_curvelet(
            curvelet_coeffs=curvelet_coeffs,
            edge_strength=0.7
        )
        
        # Lower threshold should produce more edge pixels
        assert np.sum(edges_low) >= np.sum(edges_high)
    
    def test_detect_edges_from_curvelet_validates_threshold(
        self, mimic_service, curvelet_coeffs
    ):
        """
        Test that invalid edge_strength values are rejected for curvelet.
        
        **Validates: Requirement 9.1** - Threshold validation
        """
        # Test threshold below valid range
        with pytest.raises(ValueError, match="edge_strength must be in"):
            mimic_service.detect_edges_from_curvelet(
                curvelet_coeffs=curvelet_coeffs,
                edge_strength=-0.1
            )
        
        # Test threshold above valid range
        with pytest.raises(ValueError, match="edge_strength must be in"):
            mimic_service.detect_edges_from_curvelet(
                curvelet_coeffs=curvelet_coeffs,
                edge_strength=1.5
            )
    
    def test_detect_edges_from_curvelet_saves_visualization(
        self, mimic_service, curvelet_coeffs, temp_output_dir
    ):
        """
        Test that curvelet edge visualization is saved when output_path provided.
        
        **Validates: Requirement 11.8** - Generate edge visualizations
        """
        output_path = Path(temp_output_dir) / "curvelet_edge.png"
        
        edges = mimic_service.detect_edges_from_curvelet(
            curvelet_coeffs=curvelet_coeffs,
            edge_strength=0.5,
            output_path=str(output_path)
        )
        
        # Check that visualization file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_generate_difference_map_returns_valid_map(
        self, mimic_service, wavelet_coeffs, curvelet_coeffs
    ):
        """
        Test that difference map generation produces valid output.
        
        **Validates: Requirement 9.4** - Generate difference_map.png
        """
        # Generate edge maps
        wavelet_edges = mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=wavelet_coeffs,
            edge_strength=0.5
        )
        
        curvelet_edges = mimic_service.detect_edges_from_curvelet(
            curvelet_coeffs=curvelet_coeffs,
            edge_strength=0.5
        )
        
        # Generate difference map
        diff_map = mimic_service.generate_difference_map(
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges
        )
        
        # Check that difference map is valid
        assert diff_map.ndim == 2
        assert diff_map.dtype == np.uint8
        
        # Difference map should have values 0-3
        # 0: No edges, 1: Wavelet only, 2: Curvelet only, 3: Both
        assert np.all(diff_map >= 0)
        assert np.all(diff_map <= 3)
    
    def test_generate_difference_map_handles_shape_mismatch(
        self, mimic_service
    ):
        """
        Test that difference map handles edge maps with different shapes.
        
        **Validates: Requirement 9.4** - Handle shape differences
        """
        # Create edge maps with different shapes
        wavelet_edges = np.zeros((64, 64), dtype=np.uint8)
        wavelet_edges[16:48, 16:48] = 1
        
        curvelet_edges = np.zeros((128, 128), dtype=np.uint8)
        curvelet_edges[32:96, 32:96] = 1
        
        # Generate difference map (should resize to match)
        diff_map = mimic_service.generate_difference_map(
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges
        )
        
        # Difference map should match curvelet edges shape
        assert diff_map.shape == curvelet_edges.shape
    
    def test_generate_difference_map_saves_visualization(
        self, mimic_service, wavelet_coeffs, curvelet_coeffs, temp_output_dir
    ):
        """
        Test that difference map visualization is saved.
        
        **Validates: Requirement 9.4** - Save difference_map.png
        """
        output_path = Path(temp_output_dir) / "difference_map.png"
        
        # Generate edge maps
        wavelet_edges = mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=wavelet_coeffs,
            edge_strength=0.5
        )
        
        curvelet_edges = mimic_service.detect_edges_from_curvelet(
            curvelet_coeffs=curvelet_coeffs,
            edge_strength=0.5
        )
        
        # Generate difference map with visualization
        diff_map = mimic_service.generate_difference_map(
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges,
            output_path=str(output_path)
        )
        
        # Check that visualization file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_overlay_edges_on_image_returns_rgb(
        self, mimic_service, sample_image
    ):
        """
        Test that edge overlay produces RGB image.
        
        **Validates: Requirement 9.5** - Overlay edges on original image
        """
        # Create simple edge map
        edges = np.zeros_like(sample_image, dtype=np.uint8)
        edges[32:96, 32] = 1  # Left edge
        edges[32:96, 95] = 1  # Right edge
        edges[32, 32:96] = 1  # Top edge
        edges[95, 32:96] = 1  # Bottom edge
        
        # Overlay edges on image
        rgb_image = mimic_service.overlay_edges_on_image(
            image=sample_image,
            edges=edges,
            edge_color='red'
        )
        
        # Check that output is RGB
        assert rgb_image.ndim == 3
        assert rgb_image.shape[2] == 3
        assert rgb_image.shape[:2] == sample_image.shape
        
        # Check that values are in [0, 1] range
        assert np.all(rgb_image >= 0.0)
        assert np.all(rgb_image <= 1.0)
    
    def test_overlay_edges_on_image_applies_color(
        self, mimic_service, sample_image
    ):
        """
        Test that edge overlay applies the specified color.
        
        **Validates: Requirement 9.5** - Color overlay
        """
        # Create edge map
        edges = np.zeros_like(sample_image, dtype=np.uint8)
        edges[64, 64] = 1  # Single edge pixel
        
        # Test red color
        rgb_red = mimic_service.overlay_edges_on_image(
            image=sample_image,
            edges=edges,
            edge_color='red'
        )
        
        # Edge pixel should be red [1, 0, 0]
        assert rgb_red[64, 64, 0] == 1.0  # Red channel
        assert rgb_red[64, 64, 1] == 0.0  # Green channel
        assert rgb_red[64, 64, 2] == 0.0  # Blue channel
        
        # Test green color
        rgb_green = mimic_service.overlay_edges_on_image(
            image=sample_image,
            edges=edges,
            edge_color='green'
        )
        
        # Edge pixel should be green [0, 1, 0]
        assert rgb_green[64, 64, 0] == 0.0  # Red channel
        assert rgb_green[64, 64, 1] == 1.0  # Green channel
        assert rgb_green[64, 64, 2] == 0.0  # Blue channel
    
    def test_overlay_edges_on_image_handles_shape_mismatch(
        self, mimic_service
    ):
        """
        Test that edge overlay handles shape mismatch between image and edges.
        
        **Validates: Requirement 9.5** - Handle shape differences
        """
        # Create image and edges with different shapes
        image = np.ones((128, 128), dtype=np.float64) * 0.5
        edges = np.zeros((64, 64), dtype=np.uint8)
        edges[16:48, 16:48] = 1
        
        # Overlay edges (should resize to match image)
        rgb_image = mimic_service.overlay_edges_on_image(
            image=image,
            edges=edges
        )
        
        # Output should match image shape
        assert rgb_image.shape[:2] == image.shape
    
    def test_overlay_edges_on_image_saves_visualization(
        self, mimic_service, sample_image, temp_output_dir
    ):
        """
        Test that edge overlay visualization is saved.
        
        **Validates: Requirement 9.5** - Save edge_overlay.png
        """
        output_path = Path(temp_output_dir) / "edge_overlay.png"
        
        # Create edge map
        edges = np.zeros_like(sample_image, dtype=np.uint8)
        edges[32:96, 32:96] = 1
        
        # Overlay edges with visualization
        rgb_image = mimic_service.overlay_edges_on_image(
            image=sample_image,
            edges=edges,
            output_path=str(output_path)
        )
        
        # Check that visualization file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_edge_detection_integration(
        self, mimic_service, sample_image, temp_output_dir
    ):
        """
        Integration test for complete edge detection workflow.
        
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 13.9**
        """
        # 1. Generate transforms
        wavelet_transform = WaveletTransform(wavelet='db4')
        curvelet_transform = CurveletTransform()
        
        wavelet_coeffs = wavelet_transform.decompose(sample_image, levels=3)
        curvelet_coeffs = curvelet_transform.decompose(
            sample_image, levels=3, angular_resolution=8
        )
        
        # 2. Detect edges from both transforms
        wavelet_edges = mimic_service.detect_edges_from_wavelet(
            wavelet_coeffs=wavelet_coeffs,
            edge_strength=0.5,
            output_path=str(Path(temp_output_dir) / "wavelet_edge.png")
        )
        
        curvelet_edges = mimic_service.detect_edges_from_curvelet(
            curvelet_coeffs=curvelet_coeffs,
            edge_strength=0.5,
            output_path=str(Path(temp_output_dir) / "curvelet_edge.png")
        )
        
        # 3. Generate difference map
        diff_map = mimic_service.generate_difference_map(
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges,
            output_path=str(Path(temp_output_dir) / "difference_map.png")
        )
        
        # 4. Overlay edges on original image
        rgb_overlay = mimic_service.overlay_edges_on_image(
            image=sample_image,
            edges=curvelet_edges,
            output_path=str(Path(temp_output_dir) / "edge_overlay.png")
        )
        
        # Verify all outputs were created
        assert (Path(temp_output_dir) / "wavelet_edge.png").exists()
        assert (Path(temp_output_dir) / "curvelet_edge.png").exists()
        assert (Path(temp_output_dir) / "difference_map.png").exists()
        assert (Path(temp_output_dir) / "edge_overlay.png").exists()
        
        # Verify outputs have correct properties
        assert wavelet_edges.ndim == 2
        assert curvelet_edges.ndim == 2
        assert diff_map.ndim == 2
        assert rgb_overlay.ndim == 3
        
        # Verify some edges were detected
        assert np.sum(wavelet_edges) > 0
        assert np.sum(curvelet_edges) > 0
