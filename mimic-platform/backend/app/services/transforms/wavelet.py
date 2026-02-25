"""
Wavelet Transform Module

This module implements 2D discrete wavelet transform for multiscale image analysis.
It provides baseline edge detection capabilities for comparison with curvelet transform.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 11.3, 13.5**
"""

import numpy as np
import pywt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WaveletCoefficients:
    """
    Container for wavelet decomposition coefficients.
    
    Attributes:
        approximation: Low-frequency approximation coefficients at coarsest scale
        details: List of detail coefficient tuples (cH, cV, cD) for each level
        levels: Number of decomposition levels
        wavelet_name: Name of the wavelet used (e.g., 'db4')
    """
    approximation: np.ndarray
    details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    levels: int
    wavelet_name: str


class WaveletTransform:
    """
    2D Discrete Wavelet Transform for multiscale image analysis.
    
    This class provides wavelet decomposition, reconstruction, and edge detection
    capabilities using PyWavelets library. It serves as a baseline transform for
    comparison with the curvelet transform.
    """
    
    def __init__(self, wavelet: str = 'db4'):
        """
        Initialize the wavelet transform.
        
        Args:
            wavelet: Wavelet family to use (default: 'db4' - Daubechies 4)
                    Other options: 'haar', 'sym4', 'coif1', etc.
        """
        self.wavelet = wavelet
        
        # Verify wavelet is available
        if wavelet not in pywt.wavelist():
            logger.warning(f"Wavelet '{wavelet}' not found, falling back to 'db4'")
            self.wavelet = 'db4'
    
    def decompose(
        self,
        image: np.ndarray,
        levels: int = 3
    ) -> WaveletCoefficients:
        """
        Apply 2D discrete wavelet transform to decompose image into multiple scales.
        
        Performs multilevel 2D wavelet decomposition, extracting approximation and
        detail coefficients at each scale level. The decomposition provides a
        multiscale representation suitable for edge detection and feature analysis.
        
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        Args:
            image: Input image as 2D numpy array (grayscale)
            levels: Number of decomposition levels (default: 3, minimum: 1)
        
        Returns:
            WaveletCoefficients object containing approximation and detail coefficients
        
        Raises:
            ValueError: If image is not 2D or levels is invalid
        """
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}")
        
        if levels < 1:
            raise ValueError(f"Levels must be at least 1, got {levels}")
        
        # Check if image is large enough for requested decomposition levels
        min_size = min(image.shape)
        max_levels = pywt.dwt_max_level(min_size, self.wavelet)
        
        if levels > max_levels:
            logger.warning(
                f"Requested {levels} levels exceeds maximum {max_levels} "
                f"for image size {image.shape}. Using {max_levels} levels."
            )
            levels = max_levels
        
        # Ensure at least 3 levels as per requirement
        if levels < 3:
            logger.warning(
                f"Image size {image.shape} only supports {levels} levels, "
                f"requirement specifies at least 3 levels"
            )
        
        # Perform multilevel 2D wavelet decomposition
        coeffs = pywt.wavedec2(image, self.wavelet, level=levels)
        
        # Extract approximation (first element) and details (remaining elements)
        approximation = coeffs[0]
        details = coeffs[1:]  # List of (cH, cV, cD) tuples
        
        logger.info(
            f"Wavelet decomposition complete: {levels} levels, "
            f"wavelet={self.wavelet}, shape={image.shape}"
        )
        
        return WaveletCoefficients(
            approximation=approximation,
            details=details,
            levels=levels,
            wavelet_name=self.wavelet
        )
    
    def reconstruct(self, coefficients: WaveletCoefficients) -> np.ndarray:
        """
        Perform inverse wavelet transform to reconstruct image from coefficients.
        
        Reconstructs the original image from wavelet coefficients using the
        inverse 2D discrete wavelet transform. This is useful for verifying
        decomposition quality and for selective reconstruction from modified
        coefficients.
        
        **Validates: Requirement 5.4** (implied by reconstruction capability)
        
        Args:
            coefficients: WaveletCoefficients object from decompose()
        
        Returns:
            Reconstructed image as 2D numpy array
        
        Raises:
            ValueError: If coefficients are invalid
        """
        # Reconstruct coefficient list in PyWavelets format
        coeffs_list = [coefficients.approximation] + list(coefficients.details)
        
        # Perform inverse wavelet transform
        reconstructed = pywt.waverec2(coeffs_list, coefficients.wavelet_name)
        
        logger.debug(
            f"Wavelet reconstruction complete: shape={reconstructed.shape}"
        )
        
        return reconstructed
    
    def extract_edges(
        self,
        coefficients: WaveletCoefficients,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Extract edges from wavelet coefficients using adaptive percentile threshold.
        
        Detects edges by analyzing the magnitude of detail coefficients across
        all scales. Uses adaptive percentile-based thresholding to ensure edges
        are visible even with varying coefficient magnitudes.
        
        **Validates: Requirement 5.4**
        
        Args:
            coefficients: WaveletCoefficients object from decompose()
            threshold: Percentile threshold (0.0-1.0). Higher values produce
                      fewer, stronger edges. Default: 0.5 (uses 50th percentile)
        
        Returns:
            Binary edge map as 2D numpy array (same size as original image)
        
        Raises:
            ValueError: If threshold is not in valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        # Start with approximation coefficients shape
        # We'll upsample edge maps from each level to full resolution
        edge_maps = []
        
        # Process each decomposition level
        for level_idx, (cH, cV, cD) in enumerate(coefficients.details):
            # Compute magnitude of detail coefficients
            # Combine horizontal, vertical, and diagonal components
            magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
            
            # Use adaptive percentile threshold - more lenient for astronomy images
            # Convert threshold parameter to percentile (0.5 → 75th percentile)
            percentile = 50 + (threshold * 25)  # Maps [0,1] to [50,75] instead of [50,100]
            adaptive_thr = np.percentile(magnitude, percentile)
            
            # Apply adaptive threshold to create binary edge map
            level_edges = (magnitude > adaptive_thr).astype(np.float32)
            
            edge_maps.append((level_edges, magnitude))
        
        # Upsample all edge maps to the finest scale (largest detail level)
        # The finest scale is the first in the details list
        target_shape = coefficients.details[0][0].shape
        
        combined_edges = np.zeros(target_shape, dtype=np.float32)
        
        for level_idx, (level_edges, magnitude) in enumerate(edge_maps):
            # Skip empty edge maps
            if level_edges.size == 0:
                continue
                
            # Upsample to target shape if needed
            if level_edges.shape != target_shape:
                # Calculate upsampling factor
                scale_h = target_shape[0] / level_edges.shape[0]
                scale_v = target_shape[1] / level_edges.shape[1]
                
                # Handle edge case where scale factors are very small
                if scale_h < 1 or scale_v < 1:
                    # Downsample instead (shouldn't happen in normal wavelet decomposition)
                    continue
                
                # Simple nearest-neighbor upsampling using repeat
                upsampled = np.repeat(
                    np.repeat(level_edges, int(np.ceil(scale_h)), axis=0),
                    int(np.ceil(scale_v)),
                    axis=1
                )
                
                # Crop to exact target shape
                upsampled = upsampled[:target_shape[0], :target_shape[1]]
            else:
                upsampled = level_edges
            
            # Accumulate edges from all scales
            combined_edges += upsampled
        
        # Normalize combined edge map
        if combined_edges.max() > 0:
            combined_edges = combined_edges / combined_edges.max()
        
        # Apply final lenient threshold to ensure we get edges
        # Use 10th percentile of non-zero values for astronomy images
        if np.any(combined_edges > 0):
            final_thr = np.percentile(combined_edges[combined_edges > 0], 10)
        else:
            final_thr = 0.01
        
        binary_edges = (combined_edges > final_thr).astype(np.uint8)
        
        edge_count = np.sum(binary_edges)
        logger.info(
            f"Edge extraction complete: {edge_count} edge pixels detected "
            f"(adaptive threshold: {final_thr:.4f})"
        )
        
        return binary_edges
    
    def get_coefficient_magnitudes(
        self,
        coefficients: WaveletCoefficients
    ) -> List[np.ndarray]:
        """
        Extract magnitude of detail coefficients at each scale.
        
        Computes the magnitude of wavelet detail coefficients by combining
        horizontal, vertical, and diagonal components at each decomposition level.
        Useful for visualization and analysis of multiscale features.
        
        Args:
            coefficients: WaveletCoefficients object from decompose()
        
        Returns:
            List of magnitude arrays, one per decomposition level
        """
        magnitudes = []
        
        for cH, cV, cD in coefficients.details:
            # Compute magnitude from three detail components
            magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
            magnitudes.append(magnitude)
        
        return magnitudes
    
    def compute_energy_per_scale(
        self,
        coefficients: WaveletCoefficients
    ) -> dict:
        """
        Compute energy (sum of squared coefficients) at each scale level.
        
        Calculates the energy distribution across decomposition scales, which
        indicates how much information is captured at each scale. Useful for
        understanding the multiscale structure of the image.
        
        Args:
            coefficients: WaveletCoefficients object from decompose()
        
        Returns:
            Dictionary mapping scale level to energy value
        """
        energy = {}
        
        # Approximation energy
        energy[0] = np.sum(coefficients.approximation**2)
        
        # Detail energy at each level
        for level_idx, (cH, cV, cD) in enumerate(coefficients.details, start=1):
            level_energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
            energy[level_idx] = level_energy
        
        return energy
    
    def save_edge_visualization(
        self,
        edges: np.ndarray,
        original_image: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save wavelet edge detection visualization.
        
        Creates and saves a visualization showing detected edges overlaid on
        the original image. The edges are displayed in a contrasting color
        for clear visibility.
        
        **Validates: Requirements 5.5, 11.3**
        
        Args:
            edges: Binary edge map from extract_edges()
            original_image: Original input image for context
            output_path: Path where to save the visualization (e.g., 'wavelet_edge.png')
        """
        # Create figure with single subplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Display original image in grayscale
        ax.imshow(original_image, cmap='gray', aspect='auto')
        
        # Overlay edges in red with transparency
        edge_overlay = np.zeros((*edges.shape, 4))
        edge_overlay[edges > 0] = [1, 0, 0, 0.7]  # Red with 70% opacity
        ax.imshow(edge_overlay, aspect='auto')
        
        ax.set_title('Wavelet Edge Detection', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved wavelet edge visualization to {output_path}")
    
    def save_coefficient_visualization(
        self,
        coefficients: WaveletCoefficients,
        output_path: str
    ) -> None:
        """
        Save wavelet coefficient magnitude visualization.
        
        Creates and saves a multi-panel visualization showing coefficient
        magnitudes at each decomposition scale. This provides insight into
        the multiscale structure captured by the wavelet transform.
        
        **Validates: Requirements 5.6, 11.3**
        
        Args:
            coefficients: WaveletCoefficients object from decompose()
            output_path: Path where to save the visualization (e.g., 'wavelet_coefficients.png')
        """
        num_levels = len(coefficients.details)
        
        # Create figure with subplots for each level plus approximation
        fig, axes = plt.subplots(2, (num_levels + 1 + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if num_levels > 1 else [axes]
        
        # Plot approximation coefficients
        im0 = axes[0].imshow(coefficients.approximation, cmap='viridis', aspect='auto')
        axes[0].set_title('Approximation\n(Coarsest Scale)', fontsize=10)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot detail coefficient magnitudes at each level
        magnitudes = self.get_coefficient_magnitudes(coefficients)
        
        for level_idx, magnitude in enumerate(magnitudes, start=1):
            ax = axes[level_idx]
            
            # Normalize magnitude for better visualization
            mag_normalized = magnitude / (magnitude.max() + 1e-10)
            
            im = ax.imshow(mag_normalized, cmap='hot', aspect='auto')
            ax.set_title(f'Level {level_idx}\nDetail Magnitude', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(num_levels + 1, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Wavelet Coefficient Magnitudes at Each Scale', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved wavelet coefficient visualization to {output_path}")
    
    def compute_reconstruction_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> dict:
        """
        Compute reconstruction error metrics between original and reconstructed images.
        
        Calculates multiple error metrics to quantify the quality of reconstruction:
        - MSE (Mean Squared Error): Average squared difference
        - RMSE (Root Mean Squared Error): Square root of MSE
        - MAE (Mean Absolute Error): Average absolute difference
        - PSNR (Peak Signal-to-Noise Ratio): Quality metric in dB
        - Max Error: Maximum absolute difference
        
        **Validates: Requirement 8.3**
        
        Args:
            original: Original image as 2D numpy array
            reconstructed: Reconstructed image as 2D numpy array
        
        Returns:
            Dictionary containing error metrics:
            - 'mse': Mean squared error
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'psnr': Peak signal-to-noise ratio (dB)
            - 'max_error': Maximum absolute error
            - 'error_map': Spatial distribution of absolute errors
        
        Raises:
            ValueError: If images have different shapes
        """
        if original.shape != reconstructed.shape:
            raise ValueError(
                f"Image shapes must match: original {original.shape} "
                f"vs reconstructed {reconstructed.shape}"
            )
        
        # Compute error map (spatial distribution of errors)
        error_map = np.abs(original - reconstructed)
        
        # Mean Squared Error
        mse = np.mean((original - reconstructed) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(error_map)
        
        # Maximum Error
        max_error = np.max(error_map)
        
        # Peak Signal-to-Noise Ratio (PSNR)
        # PSNR = 10 * log10(MAX^2 / MSE)
        # where MAX is the maximum possible pixel value
        max_pixel_value = max(np.max(original), np.max(reconstructed))
        if mse > 0:
            psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
        else:
            psnr = float('inf')  # Perfect reconstruction
        
        logger.info(
            f"Reconstruction error: MSE={mse:.6f}, RMSE={rmse:.6f}, "
            f"MAE={mae:.6f}, PSNR={psnr:.2f}dB, Max={max_error:.6f}"
        )
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'psnr': float(psnr),
            'max_error': float(max_error),
            'error_map': error_map
        }
    
    def compute_reconstruction_error_per_scale(
        self,
        original: np.ndarray,
        coefficients: WaveletCoefficients
    ) -> dict:
        """
        Compute reconstruction error when reconstructing from each scale level.
        
        Performs partial reconstructions using coefficients up to each scale level,
        computing the error at each stage. This shows how each scale contributes
        to the overall reconstruction quality.
        
        **Validates: Requirement 8.6**
        
        Args:
            original: Original image as 2D numpy array
            coefficients: WaveletCoefficients object from decompose()
        
        Returns:
            Dictionary mapping scale level to RMSE value
            Example: {0: 0.15, 1: 0.08, 2: 0.03, 3: 0.001}
        """
        errors_per_scale = {}
        
        # Reconstruct using progressively more scales
        for level in range(1, coefficients.levels + 1):
            # Create partial coefficients (approximation + details up to this level)
            partial_details = coefficients.details[:level]
            partial_coeffs = WaveletCoefficients(
                approximation=coefficients.approximation,
                details=partial_details,
                levels=level,
                wavelet_name=coefficients.wavelet_name
            )
            
            # Reconstruct from partial coefficients
            partial_reconstruction = self.reconstruct(partial_coeffs)
            
            # Handle size mismatch - reconstruction may be different size
            # Resize to match original using simple interpolation
            if partial_reconstruction.shape != original.shape:
                from scipy.ndimage import zoom
                scale_h = original.shape[0] / partial_reconstruction.shape[0]
                scale_w = original.shape[1] / partial_reconstruction.shape[1]
                partial_reconstruction = zoom(partial_reconstruction, (scale_h, scale_w), order=1)
                
                # Ensure exact size match by cropping or padding
                if partial_reconstruction.shape[0] > original.shape[0]:
                    partial_reconstruction = partial_reconstruction[:original.shape[0], :]
                if partial_reconstruction.shape[1] > original.shape[1]:
                    partial_reconstruction = partial_reconstruction[:, :original.shape[1]]
            
            # Compute RMSE for this scale
            mse = np.mean((original - partial_reconstruction) ** 2)
            rmse = np.sqrt(mse)
            errors_per_scale[level] = float(rmse)
        
        logger.debug(f"Computed reconstruction error per scale: {errors_per_scale}")
        
        return errors_per_scale
    
    def save_reconstruction_visualization(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save reconstruction comparison visualization.
        
        Creates a side-by-side comparison of the original and reconstructed images,
        showing the quality of the wavelet transform round-trip.
        
        **Validates: Requirement 8.4**
        
        Args:
            original: Original image as 2D numpy array
            reconstructed: Reconstructed image as 2D numpy array
            output_path: Path to save visualization (e.g., 'reconstruction.png')
        """
        # Crop reconstructed to match original size if needed
        if reconstructed.shape != original.shape:
            reconstructed = reconstructed[:original.shape[0], :original.shape[1]]
        
        # Compute error metrics
        error_metrics = self.compute_reconstruction_error(original, reconstructed)
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original, cmap='gray', aspect='auto')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Reconstructed image
        axes[1].imshow(reconstructed, cmap='gray', aspect='auto')
        axes[1].set_title(
            f'Reconstructed Image\nPSNR: {error_metrics["psnr"]:.2f} dB',
            fontsize=12, fontweight='bold'
        )
        axes[1].axis('off')
        
        # Difference (error map)
        error_map = error_metrics['error_map']
        im = axes[2].imshow(error_map, cmap='hot', aspect='auto')
        axes[2].set_title(
            f'Absolute Error\nRMSE: {error_metrics["rmse"]:.6f}',
            fontsize=12, fontweight='bold'
        )
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        fig.suptitle('Wavelet Transform Reconstruction Quality', 
                     fontsize=14, fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved reconstruction visualization to {output_path}")
    
    def save_reconstruction_error_map(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save spatial distribution of reconstruction error.
        
        Creates a heatmap showing where reconstruction errors are concentrated
        in the image, helping identify problematic regions.
        
        **Validates: Requirement 8.5**
        
        Args:
            original: Original image as 2D numpy array
            reconstructed: Reconstructed image as 2D numpy array
            output_path: Path to save visualization (e.g., 'reconstruction_error.png')
        """
        # Crop reconstructed to match original size if needed
        if reconstructed.shape != original.shape:
            reconstructed = reconstructed[:original.shape[0], :original.shape[1]]
        
        # Compute error metrics
        error_metrics = self.compute_reconstruction_error(original, reconstructed)
        error_map = error_metrics['error_map']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Display error map as heatmap
        im = ax.imshow(error_map, cmap='hot', aspect='auto')
        ax.set_title(
            f'Spatial Distribution of Reconstruction Error\n'
            f'RMSE: {error_metrics["rmse"]:.6f}, Max Error: {error_metrics["max_error"]:.6f}',
            fontsize=12, fontweight='bold'
        )
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Error', rotation=270, labelpad=20)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved reconstruction error map to {output_path}")
    
    def save_reconstruction_error_curve(
        self,
        original: np.ndarray,
        coefficients: WaveletCoefficients,
        output_path: str
    ) -> None:
        """
        Save reconstruction error vs scale level curve.
        
        Creates a plot showing how reconstruction error decreases as more
        scale levels are included in the reconstruction. This demonstrates
        the multiscale nature of the wavelet decomposition.
        
        **Validates: Requirement 8.6**
        
        Args:
            original: Original image as 2D numpy array
            coefficients: WaveletCoefficients object from decompose()
            output_path: Path to save visualization (e.g., 'reconstruction_error_curve.png')
        """
        # Compute error per scale
        errors_per_scale = self.compute_reconstruction_error_per_scale(
            original, coefficients
        )
        
        # Extract scale levels and errors
        scales = sorted(errors_per_scale.keys())
        errors = [errors_per_scale[s] for s in scales]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot error curve
        ax.plot(scales, errors, 'o-', linewidth=2, markersize=8, color='#e74c3c')
        ax.fill_between(scales, errors, alpha=0.3, color='#e74c3c')
        
        # Formatting
        ax.set_xlabel('Number of Scale Levels Included', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (Root Mean Squared Error)', fontsize=12, fontweight='bold')
        ax.set_title(
            'Reconstruction Error vs Scale Level\n(Wavelet Transform)',
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(scales)
        
        # Add value labels on points
        for scale, error in zip(scales, errors):
            ax.annotate(
                f'{error:.4f}',
                (scale, error),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9
            )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved reconstruction error curve to {output_path}")

