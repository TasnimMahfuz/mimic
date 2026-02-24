"""
Property-based tests for file processing module.
Uses Hypothesis to verify universal properties across randomized inputs.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6**
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
from PIL import Image

from app.services.file_processing import FileProcessor


# Hypothesis settings for property tests
# Minimum 100 iterations as specified in design document


class TestFileFormatAcceptanceProperties:
    """
    Property-based tests for file format acceptance.
    
    **Property 1: Supported Image Format Acceptance**
    For any valid image file in a supported format (FITS, PNG, JPEG, TIFF),
    the system should successfully accept and load the file into a numerical array.
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6**
    """
    
    @given(
        image_data=npst.arrays(
            dtype=npst.floating_dtypes(endianness='=', sizes=(32, 64)),
            shape=st.tuples(
                st.integers(min_value=64, max_value=512),
                st.integers(min_value=64, max_value=512)
            ),
            elements=st.floats(
                min_value=0.0,
                max_value=65535.0,
                allow_nan=False,
                allow_infinity=False
            )
        ),
        file_format=st.sampled_from(['png', 'jpeg', 'tiff'])
    )
    @settings(max_examples=100, deadline=None)
    def test_property_supported_format_acceptance_standard_images(
        self,
        image_data: np.ndarray,
        file_format: str
    ):
        """
        Property 1: Supported Image Format Acceptance (PNG, JPEG, TIFF)
        
        For any valid image array and supported format (PNG, JPEG, TIFF),
        the system should:
        1. Successfully save the image in that format
        2. Successfully load the image back
        3. Return a numpy array with the same shape
        4. Return data in float64 dtype
        
        **Validates: Requirements 2.2, 2.3, 2.4, 2.6**
        """
        processor = FileProcessor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Normalize image data to [0, 255] for saving
            img_min = np.min(image_data)
            img_max = np.max(image_data)
            
            if img_max > img_min:
                normalized = ((image_data - img_min) / (img_max - img_min) * 255)
            else:
                normalized = np.zeros_like(image_data)
            
            image_uint8 = normalized.astype(np.uint8)
            
            # Save image using PIL
            file_path = tmp_path / f"test_image.{file_format}"
            img = Image.fromarray(image_uint8, mode='L')
            
            # Handle JPEG quality
            if file_format == 'jpeg':
                img.save(file_path, format='JPEG', quality=95)
            else:
                img.save(file_path, format=file_format.upper())
            
            # Property verification: Load the image
            loaded_data = processor.load_image(file_path)
            
            # Assertions
            assert loaded_data is not None, "Loaded data should not be None"
            assert isinstance(loaded_data, np.ndarray), "Loaded data should be numpy array"
            assert loaded_data.shape == image_data.shape, \
                f"Shape mismatch: expected {image_data.shape}, got {loaded_data.shape}"
            assert loaded_data.dtype == np.float64, \
                f"Dtype should be float64, got {loaded_data.dtype}"
            
            # Verify data is in valid range
            assert np.all(np.isfinite(loaded_data)), "All values should be finite"
            assert np.all(loaded_data >= 0), "All values should be non-negative"
    
    @given(
        image_data=npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=64, max_value=256),
                st.integers(min_value=64, max_value=256)
            ),
            elements=st.floats(
                min_value=-1000.0,
                max_value=10000.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_fits_format_acceptance(self, image_data: np.ndarray):
        """
        Property 1: Supported Image Format Acceptance (FITS)
        
        For any valid image array, the system should:
        1. Successfully save the image as FITS format
        2. Successfully load the FITS file back
        3. Return a numpy array with the same shape
        4. Return data in float64 dtype
        5. Preserve the original data values (within numerical precision)
        
        **Validates: Requirements 2.1, 2.6**
        """
        processor = FileProcessor()
        
        # Check if astropy is available
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("Astropy not installed")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "test_image.fits"
            
            # Save as FITS file
            hdu = fits.PrimaryHDU(data=image_data)
            hdu.writeto(file_path, overwrite=True)
            
            # Property verification: Load the FITS file
            loaded_data = processor.load_fits(file_path)
            
            # Assertions
            assert loaded_data is not None, "Loaded data should not be None"
            assert isinstance(loaded_data, np.ndarray), "Loaded data should be numpy array"
            assert loaded_data.shape == image_data.shape, \
                f"Shape mismatch: expected {image_data.shape}, got {loaded_data.shape}"
            assert loaded_data.dtype == np.float64, \
                f"Dtype should be float64, got {loaded_data.dtype}"
            
            # Verify data preservation (FITS should preserve exact values)
            assert np.allclose(loaded_data, image_data, rtol=1e-10), \
                "FITS should preserve original data values"
            
            # Verify data is finite
            assert np.all(np.isfinite(loaded_data)), "All values should be finite"
    
    @given(
        width=st.integers(min_value=32, max_value=512),
        height=st.integers(min_value=32, max_value=512),
        pixel_values=st.lists(
            st.integers(min_value=0, max_value=255),
            min_size=1024,  # 32x32 minimum
            max_size=262144  # 512x512 maximum
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_format_acceptance_with_various_dimensions(
        self,
        width: int,
        height: int,
        pixel_values: list
    ):
        """
        Property 1: Supported Image Format Acceptance (Various Dimensions)
        
        For any valid image dimensions and pixel values, the system should
        successfully load images of different sizes.
        
        **Validates: Requirements 2.2, 2.3, 2.4, 2.6**
        """
        processor = FileProcessor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create image with specified dimensions
            total_pixels = width * height
            
            # Adjust pixel_values to match dimensions
            if len(pixel_values) < total_pixels:
                # Repeat values to fill
                pixel_values = (pixel_values * (total_pixels // len(pixel_values) + 1))[:total_pixels]
            else:
                pixel_values = pixel_values[:total_pixels]
            
            image_array = np.array(pixel_values, dtype=np.uint8).reshape(height, width)
            
            # Test with PNG format
            file_path = tmp_path / "test_dimensions.png"
            img = Image.fromarray(image_array, mode='L')
            img.save(file_path, format='PNG')
            
            # Property verification: Load the image
            loaded_data = processor.load_image(file_path)
            
            # Assertions
            assert loaded_data is not None, "Loaded data should not be None"
            assert loaded_data.shape == (height, width), \
                f"Shape mismatch: expected ({height}, {width}), got {loaded_data.shape}"
            assert loaded_data.dtype == np.float64, \
                f"Dtype should be float64, got {loaded_data.dtype}"
    
    @given(
        file_format=st.sampled_from(['png', 'jpg', 'jpeg', 'tiff', 'tif', 'fits', 'fit'])
    )
    @settings(max_examples=50)
    def test_property_format_detection(self, file_format: str):
        """
        Property: Format Detection Correctness
        
        For any supported file extension, the system should correctly
        identify it as a supported format.
        
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
        """
        processor = FileProcessor()
        
        filename = f"test_image.{file_format}"
        
        # Property verification: Format should be recognized as supported
        assert processor.is_supported_format(filename), \
            f"Format '{file_format}' should be recognized as supported"
    
    @given(
        image_data=npst.arrays(
            dtype=np.float64,
            shape=(128, 128),
            elements=st.floats(
                min_value=0.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_load_image_universal_interface(self, image_data: np.ndarray):
        """
        Property: Universal Load Interface
        
        The load_image() method should work for all supported formats
        through a single interface, returning consistent output format.
        
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6**
        """
        processor = FileProcessor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Normalize for saving
            img_min = np.min(image_data)
            img_max = np.max(image_data)
            
            if img_max > img_min:
                normalized = ((image_data - img_min) / (img_max - img_min) * 255)
            else:
                normalized = np.zeros_like(image_data)
            
            image_uint8 = normalized.astype(np.uint8)
            
            # Test all supported formats through the same interface
            formats_to_test = ['png', 'tiff']
            
            for fmt in formats_to_test:
                file_path = tmp_path / f"test_universal.{fmt}"
                img = Image.fromarray(image_uint8, mode='L')
                img.save(file_path, format=fmt.upper())
                
                # Property verification: load_image should work for all formats
                loaded = processor.load_image(file_path)
                
                # All formats should return the same output type and shape
                assert isinstance(loaded, np.ndarray), \
                    f"Format {fmt}: should return numpy array"
                assert loaded.dtype == np.float64, \
                    f"Format {fmt}: should return float64"
                assert loaded.shape == image_data.shape, \
                    f"Format {fmt}: should preserve shape"
                assert np.all(np.isfinite(loaded)), \
                    f"Format {fmt}: should return finite values"


class TestUnsupportedFormatRejectionProperties:
    """
    Property-based tests for unsupported format rejection.
    
    **Property 2: Unsupported Format Rejection**
    For any file with an unsupported format, the system should reject
    the upload and return an error message.
    
    **Validates: Requirements 2.5**
    """
    
    @given(
        unsupported_ext=st.sampled_from([
            'bmp', 'gif', 'svg', 'webp', 'ico', 'pdf', 'txt', 'doc', 'mp4'
        ])
    )
    @settings(max_examples=50)
    def test_property_unsupported_format_rejection(self, unsupported_ext: str):
        """
        Property 2: Unsupported Format Rejection
        
        For any unsupported file extension, the system should:
        1. Recognize it as unsupported
        2. Return False from is_supported_format()
        
        **Validates: Requirements 2.5**
        """
        processor = FileProcessor()
        
        filename = f"test_file.{unsupported_ext}"
        
        # Property verification: Format should be rejected
        assert not processor.is_supported_format(filename), \
            f"Format '{unsupported_ext}' should be rejected as unsupported"
    
    @given(
        unsupported_ext=st.text(
            alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd')),
            min_size=2,
            max_size=5
        ).filter(lambda x: x.lower() not in {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'fits', 'fit'})
    )
    @settings(max_examples=50)
    def test_property_arbitrary_extension_rejection(self, unsupported_ext: str):
        """
        Property 2: Arbitrary Extension Rejection
        
        For any arbitrary file extension that is not in the supported list,
        the system should reject it.
        
        **Validates: Requirements 2.5**
        """
        processor = FileProcessor()
        
        filename = f"test_file.{unsupported_ext}"
        
        # Property verification: Arbitrary extensions should be rejected
        assert not processor.is_supported_format(filename), \
            f"Arbitrary extension '{unsupported_ext}' should be rejected"


# Mark all tests in this module as property tests for easy filtering
pytestmark = pytest.mark.property
