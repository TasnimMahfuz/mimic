"""
Unit tests for flux normalization functionality.

Tests Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 11.2
"""
import numpy as np
import pytest
from pathlib import Path
from app.services.mimic_service import MIMICService


class TestFluxNormalization:
    """Test suite for flux normalization in MIMIC service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = MIMICService()
    
    def test_normalize_flux_basic(self):
        """Test basic flux normalization produces [0, 1] range."""
        # Create test image with known range
        image = np.array([[100, 200], [300, 400]], dtype=np.float64)
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Verify normalization range (Requirements 3.3)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)
        
        # Verify scale parameters are preserved (Requirement 3.5)
        assert scale_params['min'] == 100.0
        assert scale_params['max'] == 400.0
        assert scale_params['range'] == 300.0
    
    def test_normalize_flux_computes_min_max(self):
        """Test that min/max pixel values are computed correctly."""
        # Requirements 3.1, 3.2
        image = np.random.rand(50, 50) * 1000 + 500
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Verify min/max computation
        assert scale_params['min'] == pytest.approx(image.min(), rel=1e-9)
        assert scale_params['max'] == pytest.approx(image.max(), rel=1e-9)
    
    def test_normalize_flux_constant_image(self):
        """Test normalization of constant image (all pixels same value)."""
        image = np.ones((10, 10)) * 42.0
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Constant image should normalize to zeros
        assert np.all(normalized == 0.0)
        assert scale_params['min'] == 42.0
        assert scale_params['max'] == 42.0
    
    def test_denormalize_flux_roundtrip(self):
        """Test that denormalization recovers original values."""
        # Requirement 3.5
        original = np.random.rand(20, 20) * 500 + 100
        
        normalized, scale_params = self.service.normalize_flux(original)
        denormalized = self.service.denormalize_flux(normalized, scale_params)
        
        # Verify round-trip accuracy
        assert np.allclose(denormalized, original, rtol=1e-10)
    
    def test_normalize_flux_saves_visualization(self, tmp_path):
        """Test that normalized.png visualization is saved."""
        # Requirements 3.4, 11.2
        image = np.random.rand(50, 50) * 100
        output_path = tmp_path / "normalized.png"
        
        normalized, scale_params = self.service.normalize_flux(
            image, 
            output_path=str(output_path)
        )
        
        # Verify visualization file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Verify raw image was also saved
        raw_path = tmp_path / "normalized_raw.png"
        assert raw_path.exists()
    
    def test_normalize_flux_negative_values(self):
        """Test normalization with negative pixel values."""
        image = np.array([[-100, -50], [0, 50]], dtype=np.float64)
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Verify normalization works with negative values
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert scale_params['min'] == -100.0
        assert scale_params['max'] == 50.0
    
    def test_normalize_flux_large_dynamic_range(self):
        """Test normalization with large dynamic range."""
        image = np.array([[1e-6, 1e-3], [1.0, 1e6]], dtype=np.float64)
        
        normalized, scale_params = self.service.normalize_flux(image)
        
        # Verify normalization handles large ranges
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert scale_params['range'] == pytest.approx(1e6, rel=1e-9)
    
    def test_normalize_flux_preserves_shape(self):
        """Test that normalization preserves image shape."""
        shapes = [(10, 10), (50, 100), (128, 128)]
        
        for shape in shapes:
            image = np.random.rand(*shape) * 100
            normalized, _ = self.service.normalize_flux(image)
            
            assert normalized.shape == shape
    
    def test_normalize_flux_output_dtype(self):
        """Test that normalized output has appropriate dtype."""
        image = np.random.randint(0, 255, size=(20, 20), dtype=np.uint8)
        
        normalized, _ = self.service.normalize_flux(image.astype(np.float64))
        
        # Normalized should be float type
        assert normalized.dtype in [np.float32, np.float64]
