"""
Integration Tests for Enhancement Overlay Visualization

This module contains integration tests for the enhancement overlay visualization
functionality, verifying that enhancement_overlay.png is generated correctly.

**Validates: Requirements 10.3, 10.4, 11.7, 13.10**
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from app.services.mimic_service import MIMICService
from app.services.visualization.curvelet_visualizer import CurveletVisualizer


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mimic_service():
    """Fixture providing a MIMICService instance."""
    return MIMICService()


@pytest.fixture
def visualizer():
    """Fixture providing a CurveletVisualizer instance."""
    return CurveletVisualizer()


@pytest.fixture
def test_image():
    """Create a test image with some structure."""
    image = np.random.rand(128, 128)
    # Add some structure (edges)
    image[40:60, :] = 0.8
    image[:, 40:60] = 0.2
    return image


def test_enhancement_overlay_generation(mimic_service, visualizer, test_image, temp_output_dir):
    """
    Test that enhancement_overlay.png is generated correctly.
    
    **Validates: Requirements 10.3, 11.7**
    """
    # Apply enhancement
    enhanced = mimic_service.apply_enhancement(
        test_image,
        enhancement_factor=2.0,
        smoothing=1.0
    )
    
    # Generate overlay visualization
    output_path = temp_output_dir / "enhancement_overlay.png"
    difference = visualizer.generate_enhancement_overlay(
        test_image,
        enhanced,
        str(output_path)
    )
    
    # Verify file was created
    assert output_path.exists(), "enhancement_overlay.png should be created"
    assert output_path.stat().st_size > 0, "enhancement_overlay.png should not be empty"
    
    # Verify difference array properties
    assert difference.shape == test_image.shape, "Difference should have same shape as input"
    assert np.all(difference >= 0.0), "Difference should be non-negative"
    assert np.all(difference <= 1.0), "Difference should be normalized to [0, 1]"


def test_enhancement_preserves_unenhanced_results(mimic_service, test_image):
    """
    Test that unenhanced results are preserved for comparison.
    
    **Validates: Requirements 10.4**
    """
    # Store original image
    original = test_image.copy()
    
    # Apply enhancement
    enhanced = mimic_service.apply_enhancement(
        test_image,
        enhancement_factor=2.0,
        smoothing=1.0
    )
    
    # Verify original is unchanged
    assert np.allclose(test_image, original), \
        "Original image should be preserved (not modified in-place)"
    
    # Verify enhanced is different from original
    assert not np.allclose(enhanced, original), \
        "Enhanced image should differ from original"


def test_enhancement_with_different_parameters(mimic_service, visualizer, test_image, temp_output_dir):
    """
    Test enhancement overlay with various parameter combinations.
    
    **Validates: Requirements 10.1, 10.2, 10.3, 13.10**
    """
    test_cases = [
        {"enhancement_factor": 1.0, "smoothing": 0.0},  # No enhancement
        {"enhancement_factor": 2.0, "smoothing": 0.0},  # Only contrast
        {"enhancement_factor": 1.0, "smoothing": 2.0},  # Only smoothing
        {"enhancement_factor": 3.0, "smoothing": 1.5},  # Both
    ]
    
    for i, params in enumerate(test_cases):
        # Apply enhancement
        enhanced = mimic_service.apply_enhancement(
            test_image,
            enhancement_factor=params["enhancement_factor"],
            smoothing=params["smoothing"]
        )
        
        # Generate overlay
        output_path = temp_output_dir / f"enhancement_overlay_{i}.png"
        difference = visualizer.generate_enhancement_overlay(
            test_image,
            enhanced,
            str(output_path)
        )
        
        # Verify file was created
        assert output_path.exists(), f"Overlay {i} should be created"
        assert output_path.stat().st_size > 0, f"Overlay {i} should not be empty"
        
        # Verify difference properties
        assert difference.shape == test_image.shape
        assert np.all(np.isfinite(difference))


def test_enhancement_overlay_without_output_path(visualizer, test_image):
    """
    Test that enhancement overlay can be generated without saving to file.
    
    **Validates: Requirements 10.3**
    """
    enhanced = test_image * 1.5
    enhanced = np.clip(enhanced, 0.0, 1.0)
    
    # Generate overlay without saving
    difference = visualizer.generate_enhancement_overlay(
        test_image,
        enhanced,
        output_path=None
    )
    
    # Verify difference array is returned
    assert difference is not None
    assert difference.shape == test_image.shape
    assert np.all(difference >= 0.0)
    assert np.all(difference <= 1.0)


def test_enhancement_overlay_with_identical_images(visualizer, test_image, temp_output_dir):
    """
    Test enhancement overlay when original and enhanced are identical.
    
    **Validates: Requirements 10.3**
    """
    # Use same image for both
    output_path = temp_output_dir / "enhancement_overlay_identical.png"
    difference = visualizer.generate_enhancement_overlay(
        test_image,
        test_image,
        str(output_path)
    )
    
    # Verify file was created
    assert output_path.exists()
    
    # Verify difference is all zeros (no change)
    assert np.allclose(difference, 0.0), \
        "Difference should be zero when images are identical"


def test_enhancement_overlay_with_extreme_differences(visualizer, temp_output_dir):
    """
    Test enhancement overlay with extreme differences between images.
    
    **Validates: Requirements 10.3**
    """
    # Create images with maximum difference
    original = np.zeros((64, 64))
    enhanced = np.ones((64, 64))
    
    output_path = temp_output_dir / "enhancement_overlay_extreme.png"
    difference = visualizer.generate_enhancement_overlay(
        original,
        enhanced,
        str(output_path)
    )
    
    # Verify file was created
    assert output_path.exists()
    
    # Verify difference is normalized to [0, 1]
    assert np.all(difference >= 0.0)
    assert np.all(difference <= 1.0)
    assert np.max(difference) == 1.0, \
        "Maximum difference should be normalized to 1.0"
