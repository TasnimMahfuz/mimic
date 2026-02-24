"""
Unit tests for file processing module.
Tests FITS and image file loading, saving, and directory management.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.services.file_processing import (
    FileProcessor,
    create_run_directory,
    load_image,
    save_image,
    save_metadata,
)


class TestFileProcessor:
    """Test suite for FileProcessor class."""
    
    def test_initialization(self):
        """Test FileProcessor initialization."""
        processor = FileProcessor()
        assert processor.base_output_dir.exists()
        assert processor.base_output_dir.name == "outputs"
    
    def test_custom_output_dir(self, tmp_path):
        """Test FileProcessor with custom output directory."""
        custom_dir = tmp_path / "custom_outputs"
        processor = FileProcessor(base_output_dir=str(custom_dir))
        assert processor.base_output_dir == custom_dir
        assert custom_dir.exists()
    
    def test_is_supported_format(self):
        """Test format validation."""
        processor = FileProcessor()
        
        # Supported formats
        assert processor.is_supported_format("image.fits")
        assert processor.is_supported_format("image.fit")
        assert processor.is_supported_format("image.png")
        assert processor.is_supported_format("image.jpg")
        assert processor.is_supported_format("image.jpeg")
        assert processor.is_supported_format("image.tiff")
        assert processor.is_supported_format("image.tif")
        
        # Unsupported formats
        assert not processor.is_supported_format("image.bmp")
        assert not processor.is_supported_format("image.gif")
        assert not processor.is_supported_format("document.pdf")
    
    def test_get_file_format(self):
        """Test file format detection."""
        processor = FileProcessor()
        
        assert processor.get_file_format("image.png") == "png"
        assert processor.get_file_format("image.jpg") == "jpeg"
        assert processor.get_file_format("image.jpeg") == "jpeg"
        assert processor.get_file_format("image.tiff") == "tiff"
        assert processor.get_file_format("image.tif") == "tiff"
        assert processor.get_file_format("image.fits") == "fits"
        assert processor.get_file_format("image.fit") == "fits"


class TestImageSaving:
    """Test suite for image saving functionality."""
    
    def test_save_image_normalized(self, tmp_path):
        """Test saving image with normalization."""
        processor = FileProcessor()
        
        # Create test image
        image = np.random.rand(100, 100) * 1000
        output_path = tmp_path / "test_normalized.png"
        
        # Save image
        saved_path = processor.save_image(image, output_path, normalize=True)
        
        assert saved_path.exists()
        assert saved_path == output_path
        
        # Load and verify
        loaded = processor.load_image(saved_path)
        assert loaded.shape == image.shape
        assert loaded.dtype == np.float64
    
    def test_save_image_without_normalization(self, tmp_path):
        """Test saving image without normalization."""
        processor = FileProcessor()
        
        # Create test image in [0, 255] range
        image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8).astype(np.float64)
        output_path = tmp_path / "test_no_norm.png"
        
        # Save image
        saved_path = processor.save_image(image, output_path, normalize=False)
        
        assert saved_path.exists()
    
    def test_save_image_creates_directory(self, tmp_path):
        """Test that save_image creates parent directories."""
        processor = FileProcessor()
        
        # Create nested path that doesn't exist
        image = np.random.rand(50, 50)
        output_path = tmp_path / "nested" / "dirs" / "image.png"
        
        # Save should create directories
        saved_path = processor.save_image(image, output_path)
        
        assert saved_path.exists()
        assert saved_path.parent.exists()


class TestDirectoryManagement:
    """Test suite for directory management."""
    
    def test_create_run_directory(self, tmp_path):
        """Test run directory creation."""
        processor = FileProcessor(base_output_dir=str(tmp_path))
        
        run_id = "test_run_123"
        run_dir = processor.create_run_directory(run_id)
        
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.name == f"run_{run_id}"
        assert run_dir.parent == tmp_path
    
    def test_create_run_directory_idempotent(self, tmp_path):
        """Test that creating same directory twice doesn't fail."""
        processor = FileProcessor(base_output_dir=str(tmp_path))
        
        run_id = "test_run_456"
        run_dir1 = processor.create_run_directory(run_id)
        run_dir2 = processor.create_run_directory(run_id)
        
        assert run_dir1 == run_dir2
        assert run_dir1.exists()
    
    def test_create_run_directory_convenience(self, tmp_path):
        """Test convenience function for directory creation."""
        run_id = "test_run_789"
        run_dir = create_run_directory(run_id, base_output_dir=str(tmp_path))
        
        assert run_dir.exists()
        assert run_dir.name == f"run_{run_id}"


class TestMetadataSaving:
    """Test suite for metadata saving."""
    
    def test_save_metadata_basic(self, tmp_path):
        """Test basic metadata saving."""
        processor = FileProcessor()
        
        metadata = {
            "run_id": "test_123",
            "parameters": {
                "edge_strength": 0.5,
                "angular_resolution": 16
            },
            "status": "complete"
        }
        
        output_path = tmp_path / "metadata.json"
        saved_path = processor.save_metadata(metadata, output_path)
        
        assert saved_path.exists()
        
        # Load and verify
        with open(saved_path) as f:
            loaded = json.load(f)
        
        assert loaded["run_id"] == "test_123"
        assert loaded["parameters"]["edge_strength"] == 0.5
        assert "timestamp" in loaded  # Should be added automatically
    
    def test_save_metadata_with_timestamp(self, tmp_path):
        """Test metadata saving with existing timestamp."""
        processor = FileProcessor()
        
        custom_timestamp = "2024-01-15T10:30:00"
        metadata = {
            "run_id": "test_456",
            "timestamp": custom_timestamp
        }
        
        output_path = tmp_path / "metadata.json"
        processor.save_metadata(metadata, output_path)
        
        # Load and verify timestamp wasn't overwritten
        with open(output_path) as f:
            loaded = json.load(f)
        
        assert loaded["timestamp"] == custom_timestamp
    
    def test_save_metadata_creates_directory(self, tmp_path):
        """Test that save_metadata creates parent directories."""
        processor = FileProcessor()
        
        metadata = {"test": "data"}
        output_path = tmp_path / "nested" / "path" / "metadata.json"
        
        saved_path = processor.save_metadata(metadata, output_path)
        
        assert saved_path.exists()
        assert saved_path.parent.exists()


class TestImageLoading:
    """Test suite for image loading functionality."""
    
    def test_load_fits_file(self, tmp_path):
        """Test loading FITS file with sample data."""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("Astropy not installed")
        
        processor = FileProcessor()
        
        # Create a sample FITS file
        test_data = np.random.rand(100, 100) * 1000
        test_path = tmp_path / "test.fits"
        
        # Write FITS file
        hdu = fits.PrimaryHDU(data=test_data)
        hdul = fits.HDUList([hdu])
        hdul.writeto(test_path)
        
        # Load and verify
        loaded = processor.load_fits(test_path)
        
        assert loaded.shape == test_data.shape
        assert loaded.dtype == np.float64
        assert np.allclose(loaded, test_data)
    
    def test_load_fits_with_nan_values(self, tmp_path):
        """Test loading FITS file containing NaN values."""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("Astropy not installed")
        
        processor = FileProcessor()
        
        # Create FITS file with NaN values
        test_data = np.random.rand(50, 50)
        test_data[10:20, 10:20] = np.nan
        test_path = tmp_path / "test_nan.fits"
        
        hdu = fits.PrimaryHDU(data=test_data)
        hdul = fits.HDUList([hdu])
        hdul.writeto(test_path)
        
        # Load and verify NaN values are replaced with 0
        loaded = processor.load_fits(test_path)
        
        assert not np.any(np.isnan(loaded))
        assert loaded[10, 10] == 0.0
    
    def test_load_fits_with_extension(self, tmp_path):
        """Test loading FITS file with data in extension HDU."""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("Astropy not installed")
        
        processor = FileProcessor()
        
        # Create FITS file with empty primary and data in extension
        test_data = np.random.rand(80, 80) * 500
        test_path = tmp_path / "test_ext.fits"
        
        primary_hdu = fits.PrimaryHDU()  # Empty primary
        image_hdu = fits.ImageHDU(data=test_data)
        hdul = fits.HDUList([primary_hdu, image_hdu])
        hdul.writeto(test_path)
        
        # Load and verify
        loaded = processor.load_fits(test_path)
        
        assert loaded.shape == test_data.shape
        assert np.allclose(loaded, test_data)
    
    def test_load_png_image(self, tmp_path):
        """Test loading PNG image."""
        processor = FileProcessor()
        
        # Create and save a test PNG
        test_image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        test_path = tmp_path / "test.png"
        
        from PIL import Image
        img = Image.fromarray(test_image, mode='L')
        img.save(test_path)
        
        # Load and verify
        loaded = processor.load_image(test_path)
        
        assert loaded.shape == test_image.shape
        assert loaded.dtype == np.float64
        assert np.allclose(loaded, test_image, atol=1)
    
    def test_load_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise exception."""
        processor = FileProcessor()
        
        # Create a file with unsupported extension
        test_path = tmp_path / "test.bmp"
        test_path.write_text("fake data")
        
        with pytest.raises(HTTPException) as exc_info:
            processor.load_image(test_path)
        
        assert exc_info.value.status_code == 415
        assert "Unsupported file format" in exc_info.value.detail


class TestConvenienceFunctions:
    """Test suite for convenience wrapper functions."""
    
    def test_save_image_convenience(self, tmp_path):
        """Test save_image convenience function."""
        image = np.random.rand(50, 50)
        output_path = tmp_path / "convenience.png"
        
        saved_path = save_image(image, output_path)
        
        assert saved_path.exists()
    
    def test_save_metadata_convenience(self, tmp_path):
        """Test save_metadata convenience function."""
        metadata = {"test": "data"}
        output_path = tmp_path / "metadata.json"
        
        saved_path = save_metadata(metadata, output_path)
        
        assert saved_path.exists()


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises exception."""
        processor = FileProcessor()
        
        with pytest.raises(HTTPException) as exc_info:
            processor.load_image("/nonexistent/path/image.png")
        
        assert exc_info.value.status_code == 400
    
    def test_save_image_invalid_data(self, tmp_path):
        """Test saving invalid image data."""
        processor = FileProcessor()
        
        # Try to save non-array data
        with pytest.raises(Exception):
            processor.save_image("not an array", tmp_path / "test.png")


# Integration test
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self, tmp_path):
        """Test complete save-load-save workflow."""
        processor = FileProcessor(base_output_dir=str(tmp_path))
        
        # Create run directory
        run_id = "integration_test"
        run_dir = processor.create_run_directory(run_id)
        
        # Create and save original image
        original_image = np.random.rand(128, 128) * 1000
        image_path = run_dir / "original.png"
        processor.save_image(original_image, image_path)
        
        # Load image back
        loaded_image = processor.load_image(image_path)
        
        # Save processed version
        processed_path = run_dir / "processed.png"
        processor.save_image(loaded_image * 0.5, processed_path)
        
        # Save metadata
        metadata = {
            "run_id": run_id,
            "original_shape": original_image.shape,
            "files": ["original.png", "processed.png"]
        }
        metadata_path = run_dir / "metadata.json"
        processor.save_metadata(metadata, metadata_path)
        
        # Verify everything exists
        assert image_path.exists()
        assert processed_path.exists()
        assert metadata_path.exists()
        
        # Verify metadata content
        with open(metadata_path) as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata["run_id"] == run_id
        assert "timestamp" in loaded_metadata
