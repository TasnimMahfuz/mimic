"""
File processing module for MIMIC system.
Handles loading and saving of astronomical FITS files and standard image formats.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    File I/O handler for MIMIC system.
    Supports FITS, PNG, JPEG, and TIFF formats.
    """
    
    SUPPORTED_FORMATS = {'.fits', '.fit', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialize file processor.
        
        Args:
            base_output_dir: Base directory for output files
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fits(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load FITS file and extract image data.
        
        Args:
            file_path: Path to FITS file
        
        Returns:
            Image data as numpy array
        
        Raises:
            HTTPException: If file cannot be loaded or is invalid
        
        Validates: Requirements 2.1, 2.6
        """
        try:
            from astropy.io import fits
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Astropy not installed. Cannot load FITS files."
            )
        
        try:
            file_path = Path(file_path)
            logger.info(f"Loading FITS file: {file_path}")
            
            with fits.open(file_path) as hdul:
                # Get primary HDU data
                image_data = hdul[0].data
                
                if image_data is None:
                    # Try first extension if primary is empty
                    if len(hdul) > 1:
                        image_data = hdul[1].data
                
                if image_data is None:
                    raise ValueError("No image data found in FITS file")
                
                # Convert to float64 for processing
                image_data = np.asarray(image_data, dtype=np.float64)
                
                # Handle NaN values
                if np.any(np.isnan(image_data)):
                    logger.warning("FITS file contains NaN values, replacing with 0")
                    image_data = np.nan_to_num(image_data, nan=0.0)
                
                logger.info(f"Loaded FITS image: shape={image_data.shape}, dtype={image_data.dtype}")
                return image_data
        
        except Exception as e:
            logger.error(f"Failed to load FITS file {file_path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load FITS file: {str(e)}"
            )
    
    def load_image(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load standard image formats (PNG, JPEG, TIFF).
        
        Args:
            file_path: Path to image file
        
        Returns:
            Image data as numpy array (grayscale, float64)
        
        Raises:
            HTTPException: If file cannot be loaded or format unsupported
        
        Validates: Requirements 2.2, 2.3, 2.4
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        # Check if format is supported
        if ext not in self.SUPPORTED_FORMATS:
            supported = ", ".join(sorted(self.SUPPORTED_FORMATS))
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format '{ext}'. Supported formats: {supported}"
            )
        
        # Handle FITS files
        if ext in {'.fits', '.fit'}:
            return self.load_fits(file_path)
        
        # Handle standard image formats
        try:
            # Try OpenCV first (faster)
            try:
                import cv2
                logger.info(f"Loading image with OpenCV: {file_path}")
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    raise ValueError("OpenCV failed to load image")
                
                # Convert to float64
                image_data = image.astype(np.float64)
            
            except (ImportError, ValueError):
                # Fallback to PIL
                from PIL import Image
                logger.info(f"Loading image with PIL: {file_path}")
                
                with Image.open(file_path) as img:
                    # Convert to grayscale
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    image_data = np.array(img, dtype=np.float64)
            
            logger.info(f"Loaded image: shape={image_data.shape}, dtype={image_data.dtype}")
            return image_data
        
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load image: {str(e)}"
            )
    
    def save_image(
        self,
        image: np.ndarray,
        path: Union[str, Path],
        format: str = 'png',
        normalize: bool = True
    ) -> Path:
        """
        Save image to file with percentile stretch for better contrast.
        
        Uses 1st-99th percentile stretch to preserve scientific contrast
        and prevent black outputs from very small coefficient values.
        
        Args:
            image: Image data as numpy array
            path: Output file path
            format: Image format ('png', 'jpeg', 'tiff')
            normalize: Whether to apply percentile stretch (default: True)
        
        Returns:
            Path to saved file
        
        Raises:
            HTTPException: If save operation fails
        
        Validates: Requirements 17.3, 17.4
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare image for saving
            if normalize:
                # Handle NaN values
                img = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Apply percentile stretch for better contrast
                # This preserves scientific features better than min-max
                p1, p99 = np.percentile(img, (1, 99))
                
                # Clip to percentile range
                img_clipped = np.clip(img, p1, p99)
                
                # Normalize to [0, 1] range
                if p99 > p1:
                    img_normalized = (img_clipped - p1) / (p99 - p1 + 1e-8)
                else:
                    # Constant image or very small range
                    img_normalized = np.zeros_like(img)
                
                # Scale to [0, 255] and convert to uint8
                image_to_save = (img_normalized * 255).astype(np.uint8)
            else:
                # Direct conversion without normalization
                image_to_save = np.clip(image, 0, 255).astype(np.uint8)
            
            # Save using OpenCV for consistency
            try:
                import cv2
                cv2.imwrite(str(path), image_to_save)
                abs_path = path.resolve()
                logger.info(f"Saved image to {path} (absolute: {abs_path})")
            
            except ImportError:
                # Fallback to PIL
                from PIL import Image
                img = Image.fromarray(image_to_save, mode='L')
                img.save(path, format=format.upper())
                abs_path = path.resolve()
                logger.info(f"Saved image to {path} (absolute: {abs_path}) using PIL")
            
            return path
        
        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save image: {str(e)}"
            )
    
    def create_run_directory(self, run_id: str) -> Path:
        """
        Create output directory for a processing run.
        
        Args:
            run_id: Unique identifier for the run
        
        Returns:
            Path to created directory
        
        Raises:
            HTTPException: If directory creation fails
        
        Validates: Requirements 17.1, 17.2
        """
        try:
            run_dir = self.base_output_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created run directory: {run_dir}")
            return run_dir
        
        except Exception as e:
            logger.error(f"Failed to create run directory for {run_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create output directory: {str(e)}"
            )
    
    def save_metadata(
        self,
        metadata: Dict,
        path: Union[str, Path]
    ) -> Path:
        """
        Save processing metadata as JSON.
        
        Args:
            metadata: Dictionary containing metadata
            path: Output file path
        
        Returns:
            Path to saved metadata file
        
        Raises:
            HTTPException: If save operation fails
        
        Validates: Requirements 17.5, 17.6, 17.7
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp if not present
            if 'timestamp' not in metadata:
                metadata['timestamp'] = datetime.utcnow().isoformat()
            
            # Write JSON with pretty formatting
            with open(path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved metadata to {path}")
            return path
        
        except Exception as e:
            logger.error(f"Failed to save metadata to {path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save metadata: {str(e)}"
            )
    
    def get_file_format(self, filename: str) -> str:
        """
        Determine file format from filename.
        
        Args:
            filename: Name of file
        
        Returns:
            File format (lowercase extension without dot)
        """
        ext = Path(filename).suffix.lower()
        
        # Normalize extensions
        if ext in {'.jpg', '.jpeg'}:
            return 'jpeg'
        elif ext in {'.tif', '.tiff'}:
            return 'tiff'
        elif ext in {'.fits', '.fit'}:
            return 'fits'
        elif ext == '.png':
            return 'png'
        else:
            return ext.lstrip('.')
    
    def is_supported_format(self, filename: str) -> bool:
        """
        Check if file format is supported.
        
        Args:
            filename: Name of file
        
        Returns:
            True if format is supported, False otherwise
        
        Validates: Requirements 2.5
        """
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_FORMATS


# Convenience functions for backward compatibility
def load_fits(file_path: Union[str, Path]) -> np.ndarray:
    """Load FITS file. Convenience wrapper."""
    processor = FileProcessor()
    return processor.load_fits(file_path)


def load_image(file_path: Union[str, Path]) -> np.ndarray:
    """Load image file. Convenience wrapper."""
    processor = FileProcessor()
    return processor.load_image(file_path)


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    format: str = 'png',
    normalize: bool = True
) -> Path:
    """Save image file. Convenience wrapper."""
    processor = FileProcessor()
    return processor.save_image(image, path, format, normalize)


def create_run_directory(run_id: str, base_output_dir: str = "outputs") -> Path:
    """Create run directory. Convenience wrapper."""
    processor = FileProcessor(base_output_dir)
    return processor.create_run_directory(run_id)


def save_metadata(metadata: Dict, path: Union[str, Path]) -> Path:
    """Save metadata. Convenience wrapper."""
    processor = FileProcessor()
    return processor.save_metadata(metadata, path)
