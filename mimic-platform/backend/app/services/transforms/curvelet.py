"""
Curvelet Transform Module

This module provides the main curvelet transform interface that attempts to use
specialized curvelet libraries but falls back to FFT-based implementation when
unavailable. It provides a unified API for curvelet decomposition, reconstruction,
and directional analysis.

**Validates: Requirements 13.6, 20.5, 20.6**
"""

import numpy as np
from typing import Dict, Optional
import logging

# Import the FFT-based fallback implementation
from .fft_directional import FFTDirectionalFilter, CurveletCoefficients

logger = logging.getLogger(__name__)

# Try to import specialized curvelet library
CURVELET_LIBRARY_AVAILABLE = False
try:
    # Attempt to import specialized curvelet library (e.g., pycurvelab)
    # This is a placeholder - actual library may vary
    import pycurvelab
    CURVELET_LIBRARY_AVAILABLE = True
    logger.info("Specialized curvelet library available")
except ImportError:
    logger.warning(
        "Specialized curvelet library not available. "
        "Using FFT-based directional filtering as fallback."
    )


class CurveletTransform:
    """
    Main curvelet transform interface with automatic fallback.
    
    This class provides a unified API for curvelet transform operations.
    It attempts to use specialized curvelet libraries (like CurveLab/pycurvelab)
    for optimal performance and accuracy, but automatically falls back to
    FFT-based directional filtering when specialized libraries are unavailable.
    
    The fallback ensures the system always works, even in environments where
    specialized curvelet libraries cannot be installed.
    
    **Validates: Requirements 13.6, 20.5, 20.6**
    """
    
    def __init__(self):
        """
        Initialize the curvelet transform.
        
        Checks for specialized library availability and sets up the
        appropriate implementation (specialized or FFT fallback).
        """
        self.use_fallback = not CURVELET_LIBRARY_AVAILABLE
        
        if self.use_fallback:
            self.fft_filter = FFTDirectionalFilter()
            logger.info("CurveletTransform initialized with FFT-based fallback")
        else:
            logger.info("CurveletTransform initialized with specialized library")
    
    def decompose(
        self,
        image: np.ndarray,
        levels: int = 3,
        angular_resolution: int = 16
    ) -> CurveletCoefficients:
        """
        Decompose image using curvelet transform.
        
        Performs multiscale directional decomposition of the input image.
        Attempts to use specialized curvelet library if available, otherwise
        falls back to FFT-based directional filtering.
        
        **Validates: Requirements 13.6, 20.5**
        
        Args:
            image: Input image as 2D numpy array (grayscale)
            levels: Number of scale levels (default: 3, minimum: 3)
            angular_resolution: Number of directional orientations (default: 16)
        
        Returns:
            CurveletCoefficients object containing directional coefficients
        
        Raises:
            ValueError: If image is not 2D or parameters are invalid
        """
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}")
        
        if self.use_fallback:
            # Use FFT-based fallback
            logger.debug(
                f"Using FFT-based fallback for curvelet decomposition: "
                f"levels={levels}, angular_resolution={angular_resolution}"
            )
            return self.fft_filter.simulate_curvelet(
                image=image,
                levels=levels,
                orientations=angular_resolution
            )
        else:
            # Use specialized curvelet library
            logger.debug(
                f"Using specialized library for curvelet decomposition: "
                f"levels={levels}, angular_resolution={angular_resolution}"
            )
            # Placeholder for specialized library implementation
            # This would call the actual library's decomposition function
            # For now, fall back to FFT method
            logger.warning(
                "Specialized library interface not yet implemented, "
                "using FFT fallback"
            )
            if not hasattr(self, 'fft_filter'):
                self.fft_filter = FFTDirectionalFilter()
            return self.fft_filter.simulate_curvelet(
                image=image,
                levels=levels,
                orientations=angular_resolution
            )
    
    def reconstruct(self, coefficients: CurveletCoefficients) -> np.ndarray:
        """
        Reconstruct image from curvelet coefficients.
        
        Performs inverse curvelet transform to reconstruct the image from
        its directional coefficients. Uses the same implementation (specialized
        or fallback) that was used for decomposition.
        
        **Validates: Requirements 13.6**
        
        Args:
            coefficients: CurveletCoefficients object from decompose()
        
        Returns:
            Reconstructed image as 2D numpy array
        """
        if self.use_fallback or not CURVELET_LIBRARY_AVAILABLE:
            # Use FFT-based fallback reconstruction
            if not hasattr(self, 'fft_filter'):
                self.fft_filter = FFTDirectionalFilter()
            return self.fft_filter.reconstruct(coefficients)
        else:
            # Use specialized curvelet library reconstruction
            logger.warning(
                "Specialized library interface not yet implemented, "
                "using FFT fallback"
            )
            if not hasattr(self, 'fft_filter'):
                self.fft_filter = FFTDirectionalFilter()
            return self.fft_filter.reconstruct(coefficients)
    
    def extract_directional_energy(
        self,
        coefficients: CurveletCoefficients
    ) -> Dict[int, np.ndarray]:
        """
        Compute energy per orientation for each scale.
        
        Extracts the energy distribution across orientations, revealing
        the dominant directional features at each scale. Energy is computed
        as the sum of squared coefficient magnitudes for each directional
        subband.
        
        **Validates: Requirements 20.5**
        
        Args:
            coefficients: CurveletCoefficients object from decompose()
        
        Returns:
            Dictionary mapping scale index -> array of energies per orientation
            Example: {0: [e0, e1, ...], 1: [e0, e1, ...], ...}
        """
        if self.use_fallback or not CURVELET_LIBRARY_AVAILABLE:
            # Use FFT-based fallback
            if not hasattr(self, 'fft_filter'):
                self.fft_filter = FFTDirectionalFilter()
            return self.fft_filter.extract_directional_energy(coefficients)
        else:
            # Use specialized curvelet library
            logger.warning(
                "Specialized library interface not yet implemented, "
                "using FFT fallback"
            )
            if not hasattr(self, 'fft_filter'):
                self.fft_filter = FFTDirectionalFilter()
            return self.fft_filter.extract_directional_energy(coefficients)
    
    def compute_orientation_map(
        self,
        coefficients: CurveletCoefficients
    ) -> np.ndarray:
        """
        Compute dominant orientation at each spatial location.
        
        Creates a map showing the dominant edge/feature direction at each
        pixel location. For each pixel, determines which orientation has
        the strongest coefficient magnitude across all scales.
        
        **Validates: Requirements 20.6**
        
        Args:
            coefficients: CurveletCoefficients object from decompose()
        
        Returns:
            2D array of dominant orientation indices (same shape as input image)
            Values are orientation indices (0 to angular_resolution-1)
        """
        if self.use_fallback or not CURVELET_LIBRARY_AVAILABLE:
            # Use FFT-based fallback
            if not hasattr(self, 'fft_filter'):
                self.fft_filter = FFTDirectionalFilter()
            return self.fft_filter.compute_orientation_map(coefficients)
        else:
            # Use specialized curvelet library
            logger.warning(
                "Specialized library interface not yet implemented, "
                "using FFT fallback"
            )
            if not hasattr(self, 'fft_filter'):
                self.fft_filter = FFTDirectionalFilter()
            return self.fft_filter.compute_orientation_map(coefficients)
    def extract_coefficient_magnitudes(
        self,
        coefficients: CurveletCoefficients
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Extract coefficient magnitudes for each directional subband.

        Computes the magnitude (absolute value) of coefficients for each
        scale and orientation. This is useful for analyzing the strength
        of directional features at different scales.

        **Validates: Requirements 7.1, 13.7**

        Args:
            coefficients: CurveletCoefficients object from decompose()

        Returns:
            Dictionary mapping scale -> angle -> magnitude array
            Structure: {scale_idx: {angle_idx: np.ndarray}}
        """
        magnitudes = {}

        for scale in range(coefficients.scales):
            magnitudes[scale] = {}
            for angle_idx, coeffs in coefficients.coefficients[scale].items():
                magnitudes[scale][angle_idx] = np.abs(coeffs)

        logger.debug(
            f"Extracted coefficient magnitudes for {coefficients.scales} scales"
        )

        return magnitudes

    def save_directional_data(
        self,
        coefficients: CurveletCoefficients,
        output_path: str,
        include_coefficients: bool = False
    ) -> str:
        """
        Save directional coefficient data in structured format.

        Saves directional analysis data to disk in NPZ format (compressed numpy).
        Includes energy per orientation, coefficient magnitudes, and metadata.
        Optionally includes full coefficient arrays (can be large).

        **Validates: Requirements 7.5, 13.7**

        Args:
            coefficients: CurveletCoefficients object from decompose()
            output_path: Path to save the data file (should end in .npz)
            include_coefficients: Whether to include full coefficient arrays
                                 (default: False, as they can be very large)

        Returns:
            Path to the saved file
        """
        from pathlib import Path
        import json

        # Ensure output path has .npz extension
        output_path = str(Path(output_path).with_suffix('.npz'))

        # Extract energy per orientation
        energy_data = self.extract_directional_energy(coefficients)

        # Extract coefficient magnitudes
        magnitude_data = self.extract_coefficient_magnitudes(coefficients)

        # Prepare data for saving
        save_dict = {}

        # Save metadata
        save_dict['metadata'] = json.dumps({
            'scales': coefficients.scales,
            'orientations': coefficients.orientations,
            'shape': coefficients.shape,
            'includes_coefficients': include_coefficients
        })

        # Save energy per orientation for each scale
        for scale, energies in energy_data.items():
            save_dict[f'energy_scale_{scale}'] = energies

        # Save coefficient magnitudes for each scale and angle
        for scale in range(coefficients.scales):
            for angle_idx, magnitudes in magnitude_data[scale].items():
                # Compute statistics instead of full arrays to save space
                save_dict[f'magnitude_mean_s{scale}_a{angle_idx}'] = np.mean(magnitudes)
                save_dict[f'magnitude_std_s{scale}_a{angle_idx}'] = np.std(magnitudes)
                save_dict[f'magnitude_max_s{scale}_a{angle_idx}'] = np.max(magnitudes)

        # Optionally save full coefficient arrays
        if include_coefficients:
            for scale in range(coefficients.scales):
                for angle_idx, coeffs in coefficients.coefficients[scale].items():
                    save_dict[f'coeffs_s{scale}_a{angle_idx}'] = coeffs

        # Save to compressed numpy format
        np.savez_compressed(output_path, **save_dict)

        logger.info(
            f"Saved directional coefficient data to {output_path} "
            f"(include_coefficients={include_coefficients})"
        )

        return output_path

    def generate_visualizations(
        self,
        coefficients: CurveletCoefficients,
        output_dir: str,
        edge_threshold: float = 0.1
    ) -> Dict[str, str]:
        """
        Generate all curvelet visualizations.
        
        Convenience method to generate all required curvelet visualizations
        in one call. Creates curvelet_edge.png, directional_energy.png,
        orientation_map.png, and angular_distribution.png.
        
        **Validates: Requirements 6.6, 6.7, 7.3, 7.4, 11.4, 11.6, 11.10**
        
        Args:
            coefficients: CurveletCoefficients from decompose()
            output_dir: Directory to save visualizations
            edge_threshold: Threshold for edge detection (default: 0.1)
        
        Returns:
            Dictionary mapping visualization name to file path
        """
        from pathlib import Path
        from ..visualization.curvelet_visualizer import CurveletVisualizer
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        visualizer = CurveletVisualizer()
        
        # Generate all visualizations
        output_files = {}
        
        # 1. Curvelet edge detection
        edge_path = str(output_path / "curvelet_edge.png")
        visualizer.generate_curvelet_edge(
            coefficients,
            threshold=edge_threshold,
            output_path=edge_path
        )
        output_files['curvelet_edge'] = edge_path
        
        # 2. Directional energy distribution
        energy_path = str(output_path / "directional_energy.png")
        visualizer.generate_directional_energy(
            coefficients,
            output_path=energy_path
        )
        output_files['directional_energy'] = energy_path
        
        # 3. Orientation map
        orientation_path = str(output_path / "orientation_map.png")
        visualizer.generate_orientation_map(
            coefficients,
            output_path=orientation_path
        )
        output_files['orientation_map'] = orientation_path
        
        # 4. Angular distribution
        angular_path = str(output_path / "angular_distribution.png")
        visualizer.generate_angular_distribution(
            coefficients,
            output_path=angular_path
        )
        output_files['angular_distribution'] = angular_path
        
        logger.info(f"Generated {len(output_files)} curvelet visualizations in {output_dir}")
        
        return output_files
    
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
            f"Curvelet reconstruction error: MSE={mse:.6f}, RMSE={rmse:.6f}, "
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
        coefficients: CurveletCoefficients
    ) -> dict:
        """
        Compute reconstruction error when reconstructing from each scale level.
        
        Performs partial reconstructions using coefficients up to each scale level,
        computing the error at each stage. This shows how each scale contributes
        to the overall reconstruction quality.
        
        **Validates: Requirement 8.6**
        
        Args:
            original: Original image as 2D numpy array
            coefficients: CurveletCoefficients object from decompose()
        
        Returns:
            Dictionary mapping scale level to RMSE value
            Example: {0: 0.15, 1: 0.08, 2: 0.03, 3: 0.001}
        """
        errors_per_scale = {}
        
        # Reconstruct using progressively more scales
        for level in range(1, coefficients.scales + 1):
            # Create partial coefficients (only scales up to this level)
            partial_coeffs_dict = {}
            for scale in range(level):
                partial_coeffs_dict[scale] = coefficients.coefficients[scale]
            
            partial_coeffs = CurveletCoefficients(
                coefficients=partial_coeffs_dict,
                scales=level,
                orientations=coefficients.orientations,
                shape=coefficients.shape
            )
            
            # Reconstruct from partial coefficients
            partial_reconstruction = self.reconstruct(partial_coeffs)
            
            # Ensure same shape as original
            if partial_reconstruction.shape != original.shape:
                partial_reconstruction = partial_reconstruction[
                    :original.shape[0], :original.shape[1]
                ]
            
            # Compute RMSE for this scale
            mse = np.mean((original - partial_reconstruction) ** 2)
            rmse = np.sqrt(mse)
            errors_per_scale[level] = float(rmse)
        
        logger.debug(f"Computed curvelet reconstruction error per scale: {errors_per_scale}")
        
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
        showing the quality of the curvelet transform round-trip.
        
        **Validates: Requirement 8.4**
        
        Args:
            original: Original image as 2D numpy array
            reconstructed: Reconstructed image as 2D numpy array
            output_path: Path to save visualization (e.g., 'reconstruction.png')
        """
        import matplotlib.pyplot as plt
        
        # Ensure same shape
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
        
        fig.suptitle('Curvelet Transform Reconstruction Quality', 
                     fontsize=14, fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved curvelet reconstruction visualization to {output_path}")
    
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
        import matplotlib.pyplot as plt
        
        # Ensure same shape
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
        
        logger.info(f"Saved curvelet reconstruction error map to {output_path}")
    
    def save_reconstruction_error_curve(
        self,
        original: np.ndarray,
        coefficients: CurveletCoefficients,
        output_path: str
    ) -> None:
        """
        Save reconstruction error vs scale level curve.
        
        Creates a plot showing how reconstruction error decreases as more
        scale levels are included in the reconstruction. This demonstrates
        the multiscale nature of the curvelet decomposition.
        
        **Validates: Requirement 8.6**
        
        Args:
            original: Original image as 2D numpy array
            coefficients: CurveletCoefficients object from decompose()
            output_path: Path to save visualization (e.g., 'reconstruction_error_curve.png')
        """
        import matplotlib.pyplot as plt
        
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
        ax.plot(scales, errors, 'o-', linewidth=2, markersize=8, color='#3498db')
        ax.fill_between(scales, errors, alpha=0.3, color='#3498db')
        
        # Formatting
        ax.set_xlabel('Number of Scale Levels Included', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (Root Mean Squared Error)', fontsize=12, fontweight='bold')
        ax.set_title(
            'Reconstruction Error vs Scale Level\n(Curvelet Transform)',
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
        
        logger.info(f"Saved curvelet reconstruction error curve to {output_path}")
