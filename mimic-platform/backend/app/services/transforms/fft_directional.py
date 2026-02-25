"""
FFT-Based Directional Filtering Module

This module implements FFT-based directional filtering to simulate curvelet transform
behavior. It provides a reliable fallback implementation when specialized curvelet
libraries are unavailable, using frequency-domain directional bandpass filters.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 20.1, 20.2, 20.3, 20.4**
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import logging
import cv2

logger = logging.getLogger(__name__)


@dataclass
class CurveletCoefficients:
    """
    Container for curvelet-like directional decomposition coefficients.
    
    Attributes:
        coefficients: Nested dict mapping scale -> angle -> coefficient array
                     Structure: {scale_idx: {angle_idx: np.ndarray}}
        scales: Number of scale levels
        orientations: Number of directional orientations per scale
        shape: Original image shape (height, width)
    """
    coefficients: Dict[int, Dict[int, np.ndarray]]
    scales: int
    orientations: int
    shape: Tuple[int, int]


class FFTDirectionalFilter:
    """
    FFT-based directional filtering for curvelet transform simulation.
    
    This class implements directional decomposition using frequency-domain filtering.
    It creates bandpass filters at multiple scales and orientations, applies them
    in the Fourier domain, and extracts directional coefficients that approximate
    curvelet transform behavior.
    
    The implementation provides a reliable fallback when specialized curvelet
    libraries (like CurveLab) are unavailable, ensuring the system always works.
    """
    
    def __init__(self):
        """Initialize the FFT directional filter."""
        pass
    
    def create_directional_filter(
        self,
        shape: Tuple[int, int],
        angle: float,
        scale: int,
        total_scales: int,
        angular_width: float = 0.3
    ) -> np.ndarray:
        """
        Create a directional bandpass filter in frequency domain.
        
        Constructs a 2D filter in the Fourier domain that selectively passes
        frequencies in a specific direction (angle) and at a specific scale
        (radial frequency band). The filter has a wedge shape in frequency space,
        with angular selectivity determined by angular_width.
        
        **Validates: Requirements 20.2**
        
        Args:
            shape: Image shape (height, width)
            angle: Orientation angle in radians (0 to 2π)
            scale: Scale level (0 = coarsest, higher = finer)
            total_scales: Total number of scale levels
            angular_width: Angular selectivity parameter (default: 0.3)
                          Smaller values = narrower directional selectivity
        
        Returns:
            2D frequency-domain filter as numpy array
        """
        h, w = shape
        
        # Create frequency grid centered at origin
        # Use fftshift convention: DC at center
        fy = np.fft.fftfreq(h).reshape(-1, 1)
        fx = np.fft.fftfreq(w).reshape(1, -1)
        
        # Shift to center for easier filter design
        fy = np.fft.fftshift(fy)
        fx = np.fft.fftshift(fx)
        
        # Compute radial frequency and angle in frequency domain
        freq_radius = np.sqrt(fx**2 + fy**2)
        freq_angle = np.arctan2(fy, fx)
        
        # Radial bandpass filter (scale selectivity)
        # Define frequency bands for each scale
        # Scale 0 (coarsest): low frequencies
        # Higher scales: progressively higher frequencies
        
        # Frequency band boundaries
        # Use logarithmic spacing for scale bands
        max_freq = 0.5  # Nyquist frequency
        min_freq = max_freq / (2 ** total_scales)
        
        # Compute band edges for this scale
        if scale == 0:
            # Coarsest scale: DC to first band
            freq_low = 0.0
            freq_high = min_freq * 2
        else:
            # Finer scales: logarithmically spaced bands
            freq_low = min_freq * (2 ** (scale - 1))
            freq_high = min_freq * (2 ** scale)
        
        # Create smooth radial bandpass using Gaussian-like transition
        radial_filter = np.exp(-((freq_radius - (freq_low + freq_high) / 2) ** 2) / 
                               (2 * ((freq_high - freq_low) / 2) ** 2))
        
        # Ensure frequencies outside band are suppressed
        radial_filter[freq_radius < freq_low * 0.5] = 0
        radial_filter[freq_radius > freq_high * 1.5] = 0
        
        # Angular filter (directional selectivity)
        # Compute angular distance from target angle
        # Handle angle wrapping (angles are periodic with period 2π)
        angle_diff = np.abs(freq_angle - angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        
        # Create smooth angular filter using Gaussian
        angular_filter = np.exp(-(angle_diff ** 2) / (2 * angular_width ** 2))
        
        # Also include opposite direction (180° symmetry in Fourier domain)
        angle_diff_opposite = np.abs(freq_angle - (angle + np.pi))
        angle_diff_opposite = np.minimum(angle_diff_opposite, 
                                        2 * np.pi - angle_diff_opposite)
        angular_filter_opposite = np.exp(-(angle_diff_opposite ** 2) / 
                                         (2 * angular_width ** 2))
        angular_filter = np.maximum(angular_filter, angular_filter_opposite)
        
        # Combine radial and angular filters
        directional_filter = radial_filter * angular_filter
        
        # Shift back to FFT convention (DC at corner)
        directional_filter = np.fft.ifftshift(directional_filter)
        
        return directional_filter
    
    def simulate_curvelet(
        self,
        image: np.ndarray,
        levels: int = 3,
        orientations: int = 8
    ) -> CurveletCoefficients:
        """
        Simulate curvelet transform using FFT-based directional filtering.
        
        Performs multiscale directional decomposition by applying directional
        bandpass filters in the frequency domain. This approximates curvelet
        transform behavior without requiring specialized libraries.
        
        The method:
        1. Computes 2D FFT of the image
        2. For each scale and orientation:
           - Creates a directional bandpass filter
           - Applies filter in frequency domain
           - Extracts filtered coefficients via inverse FFT
        3. Returns coefficients organized by scale and orientation
        
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 20.1, 20.3, 20.4**
        
        Args:
            image: Input image as 2D numpy array (grayscale)
            levels: Number of scale levels (default: 3, minimum: 3)
            orientations: Number of directional orientations (default: 8, minimum: 8)
        
        Returns:
            CurveletCoefficients object containing directional coefficients
        
        Raises:
            ValueError: If image is not 2D or parameters are invalid
        """
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}")
        
        if levels < 3:
            logger.warning(
                f"Requested {levels} levels, but requirement specifies at least 3. "
                f"Using 3 levels."
            )
            levels = max(3, levels)
        
        if orientations < 8:
            logger.warning(
                f"Requested {orientations} orientations, but requirement specifies "
                f"at least 8. Using 8 orientations."
            )
            orientations = max(8, orientations)
        
        logger.info(
            f"Starting FFT-based curvelet simulation: "
            f"shape={image.shape}, levels={levels}, orientations={orientations}"
        )
        
        # Compute 2D FFT of image
        image_fft = np.fft.fft2(image)
        
        # Initialize coefficient storage
        coefficients = {}
        
        # Process each scale level
        for scale in range(levels):
            coefficients[scale] = {}
            
            # Determine number of orientations for this scale
            # Coarsest scale (0) typically has fewer orientations
            if scale == 0:
                scale_orientations = max(4, orientations // 2)
            else:
                scale_orientations = orientations
            
            # Process each orientation at this scale
            for angle_idx in range(scale_orientations):
                # Compute angle in radians
                angle = 2 * np.pi * angle_idx / scale_orientations
                
                # Create directional filter for this scale and angle
                directional_filter = self.create_directional_filter(
                    shape=image.shape,
                    angle=angle,
                    scale=scale,
                    total_scales=levels
                )
                
                # Apply filter in frequency domain
                filtered_fft = image_fft * directional_filter
                
                # Transform back to spatial domain
                filtered_spatial = np.fft.ifft2(filtered_fft)
                
                # Extract real part as coefficients
                # (imaginary part is typically small due to filter symmetry)
                coeffs = np.real(filtered_spatial)
                
                # Store coefficients
                coefficients[scale][angle_idx] = coeffs
                
                logger.debug(
                    f"Extracted coefficients: scale={scale}, angle_idx={angle_idx}, "
                    f"angle={np.degrees(angle):.1f}°, "
                    f"coeff_range=[{coeffs.min():.3f}, {coeffs.max():.3f}]"
                )
        
        logger.info(
            f"FFT-based curvelet simulation complete: "
            f"{levels} scales, {orientations} orientations"
        )
        
        return CurveletCoefficients(
            coefficients=coefficients,
            scales=levels,
            orientations=orientations,
            shape=image.shape
        )
    
    def reconstruct(self, coefficients: CurveletCoefficients) -> np.ndarray:
        """
        Reconstruct image from directional coefficients.
        
        Performs approximate reconstruction by summing all directional
        coefficients across scales and orientations. This is an approximate
        inverse since the filters are not perfectly orthogonal.
        
        Args:
            coefficients: CurveletCoefficients object from simulate_curvelet()
        
        Returns:
            Reconstructed image as 2D numpy array
        """
        # Initialize reconstruction with zeros
        reconstructed = np.zeros(coefficients.shape, dtype=np.float64)
        
        # Sum all directional coefficients
        for scale in range(coefficients.scales):
            for angle_idx in coefficients.coefficients[scale].keys():
                coeff = coefficients.coefficients[scale][angle_idx]
                
                # Resize coefficient to match output shape if needed
                if coeff.shape != coefficients.shape:
                    coeff = cv2.resize(
                        coeff,
                        (coefficients.shape[1], coefficients.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                reconstructed += coeff
        
        # Normalize by number of orientations to approximate original scale
        # This is a heuristic normalization
        total_filters = sum(len(coefficients.coefficients[scale]) 
                          for scale in range(coefficients.scales))
        reconstructed = reconstructed / total_filters
        
        logger.debug(
            f"Reconstruction complete: shape={reconstructed.shape}, "
            f"range=[{reconstructed.min():.3f}, {reconstructed.max():.3f}]"
        )
        
        return reconstructed
    
    def extract_directional_energy(
        self,
        coefficients: CurveletCoefficients
    ) -> Dict[int, np.ndarray]:
        """
        Extract energy distribution across orientations for each scale.
        
        Computes the energy (sum of squared coefficients) for each directional
        subband at each scale. This reveals the dominant orientations in the
        image at different scales.
        
        Args:
            coefficients: CurveletCoefficients object from simulate_curvelet()
        
        Returns:
            Dictionary mapping scale -> array of energies per orientation
        """
        energy_per_scale = {}
        
        for scale in range(coefficients.scales):
            num_angles = len(coefficients.coefficients[scale])
            energies = np.zeros(num_angles)
            
            for angle_idx in range(num_angles):
                coeffs = coefficients.coefficients[scale][angle_idx]
                energies[angle_idx] = np.sum(coeffs ** 2)
            
            energy_per_scale[scale] = energies
        
        return energy_per_scale
    
    def compute_orientation_map(
        self,
        coefficients: CurveletCoefficients
    ) -> np.ndarray:
        """
        Compute dominant orientation at each spatial location.
        
        For each pixel, determines which orientation has the strongest
        coefficient magnitude across all scales. This produces a map
        showing the dominant edge/feature direction at each location.
        
        Args:
            coefficients: CurveletCoefficients object from simulate_curvelet()
        
        Returns:
            2D array of dominant orientation indices
        """
        h, w = coefficients.shape
        orientation_map = np.zeros((h, w), dtype=np.int32)
        max_magnitude = np.zeros((h, w), dtype=np.float64)
        
        # For each scale and orientation, track maximum magnitude
        for scale in range(coefficients.scales):
            for angle_idx, coeffs in coefficients.coefficients[scale].items():
                magnitude = np.abs(coeffs)
                
                # Resize magnitude to match output shape if needed
                if magnitude.shape != (h, w):
                    magnitude = cv2.resize(
                        magnitude,
                        (w, h),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Update orientation map where this orientation is strongest
                mask = magnitude > max_magnitude
                orientation_map[mask] = angle_idx
                max_magnitude[mask] = magnitude[mask]
        
        return orientation_map
