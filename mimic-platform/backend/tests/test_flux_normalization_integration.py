"""
Integration test for flux normalization with actual visualization.

This test creates a sample astronomical-like image and verifies
the complete normalization pipeline including visualization generation.
"""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from app.services.mimic_service import MIMICService


def test_flux_normalization_complete_workflow(tmp_path):
    """Test complete flux normalization workflow with visualization."""
    service = MIMICService()
    
    # Create a synthetic astronomical image with realistic characteristics
    # Simulate a point source with background noise
    size = 128
    image = np.random.normal(100, 10, (size, size))  # Background noise
    
    # Add a bright point source in the center
    center = size // 2
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    point_source = 1000 * np.exp(-r**2 / (2 * 10**2))
    image += point_source
    
    # Ensure all values are positive (like flux measurements)
    image = np.maximum(image, 0)
    
    # Set up output path
    output_dir = tmp_path / "run_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "normalized.png"
    
    # Run normalization
    normalized, scale_params = service.normalize_flux(
        image, 
        output_path=str(output_path)
    )
    
    # Verify normalization properties
    assert normalized.min() == pytest.approx(0.0, abs=1e-10)
    assert normalized.max() == pytest.approx(1.0, abs=1e-10)
    assert normalized.shape == image.shape
    
    # Verify scale parameters
    assert scale_params['min'] == pytest.approx(image.min(), rel=1e-9)
    assert scale_params['max'] == pytest.approx(image.max(), rel=1e-9)
    assert scale_params['range'] > 0
    
    # Verify visualization files were created
    assert output_path.exists(), "normalized.png should be created"
    assert output_path.stat().st_size > 0, "normalized.png should not be empty"
    
    raw_path = output_dir / "normalized_raw.png"
    assert raw_path.exists(), "normalized_raw.png should be created"
    assert raw_path.stat().st_size > 0, "normalized_raw.png should not be empty"
    
    # Verify the raw image can be loaded and has correct properties
    img = Image.open(raw_path)
    assert img.mode == 'L', "Image should be grayscale"
    assert img.size == (size, size), "Image dimensions should match"
    
    # Verify denormalization round-trip
    denormalized = service.denormalize_flux(normalized, scale_params)
    assert np.allclose(denormalized, image, rtol=1e-10)
    
    # Verify the point source is still visible in normalized image
    # The center should have high intensity
    center_value = normalized[center, center]
    assert center_value > 0.8, "Point source should be bright in normalized image"
    
    # Verify background is normalized to low values
    corner_value = normalized[0, 0]
    assert corner_value < 0.2, "Background should be dark in normalized image"


def test_flux_normalization_with_fits_like_data(tmp_path):
    """Test normalization with FITS-like astronomical data characteristics."""
    service = MIMICService()
    
    # Create data similar to Chandra X-ray observations
    # Typically has low count rates with Poisson noise
    size = 64
    
    # Simulate photon counts (Poisson distributed)
    background_rate = 5.0
    image = np.random.poisson(background_rate, (size, size)).astype(np.float64)
    
    # Add a diffuse source
    y, x = np.ogrid[:size, :size]
    source_x, source_y = size // 3, size // 3
    r = np.sqrt((x - source_x)**2 + (y - source_y)**2)
    source = 50 * np.exp(-r**2 / (2 * 8**2))
    image += source
    
    output_path = tmp_path / "normalized.png"
    
    # Normalize
    normalized, scale_params = service.normalize_flux(
        image,
        output_path=str(output_path)
    )
    
    # Verify normalization
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    
    # Verify scale parameters capture the photon count range
    assert scale_params['min'] >= 0, "Photon counts should be non-negative"
    assert scale_params['max'] > scale_params['min']
    
    # Verify visualization was created
    assert output_path.exists()
    
    # Verify the source region has higher normalized intensity
    source_region = normalized[source_y-5:source_y+5, source_x-5:source_x+5]
    background_region = normalized[0:10, 0:10]
    assert source_region.mean() > background_region.mean()


def test_flux_normalization_preserves_structure(tmp_path):
    """Test that normalization preserves spatial structure of features."""
    service = MIMICService()
    
    # Create an image with distinct features
    size = 100
    image = np.zeros((size, size))
    
    # Add three distinct regions with different intensities
    image[10:30, 10:30] = 100  # Low intensity region
    image[40:60, 40:60] = 500  # Medium intensity region
    image[70:90, 70:90] = 1000  # High intensity region
    
    output_path = tmp_path / "normalized.png"
    
    # Normalize
    normalized, scale_params = service.normalize_flux(
        image,
        output_path=str(output_path)
    )
    
    # Verify relative intensities are preserved
    low_region = normalized[10:30, 10:30].mean()
    medium_region = normalized[40:60, 40:60].mean()
    high_region = normalized[70:90, 70:90].mean()
    
    assert low_region < medium_region < high_region
    assert high_region == pytest.approx(1.0, abs=0.01)
    assert low_region > 0.0
    
    # Verify visualization exists
    assert output_path.exists()
