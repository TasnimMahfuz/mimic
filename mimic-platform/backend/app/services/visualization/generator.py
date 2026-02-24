"""
Visualization Generator Module

This module provides comprehensive visualization generation for the MIMIC
analysis pipeline, including all required plots and images for transform
analysis, edge detection, and scientific metrics.

**Validates: Requirements 11.1-11.16, 13.11**
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """
    Comprehensive visualization generator for MIMIC analysis pipeline.
    
    Generates all required visualizations including raw images, transform
    results, edge detection, spectral analysis, and scientific metrics.
    
    **Validates: Requirements 11.15, 11.16, 13.11**
    """
    
    def __init__(self):
        """Initialize the visualization generator."""
        plt.style.use('default')
        self.dpi = 150
        
    def generate_all(
        self,
        results: Dict,
        output_dir: str
    ) -> List[str]:
        """
        Generate all visualizations for a processing run.
        
        **Validates: Requirements 11.1-11.16**
        
        Args:
            results: Dictionary containing all processing results
            output_dir: Directory to save visualizations
        
        Returns:
            List of generated file paths
        """
        logger.info(f"Generating all visualizations in {output_dir}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Generate each visualization
        try:
            # Raw and normalized images
            if 'original_image' in results:
                path = str(output_path / 'raw.png')
                self.plot_raw_image(results['original_image'], path)
                generated_files.append(path)
            
            if 'normalized_image' in results:
                path = str(output_path / 'normalized.png')
                self.plot_normalized_image(results['normalized_image'], path)
                generated_files.append(path)
            
            # Transform visualizations
            if 'wavelet_coeffs' in results:
                path = str(output_path / 'wavelet_coefficients.png')
                self.plot_wavelet_decomposition(results['wavelet_coeffs'], path)
                generated_files.append(path)
            
            # Edge detection results
            if 'wavelet_edges' in results:
                path = str(output_path / 'wavelet_edge.png')
                self.plot_edge_map(results['wavelet_edges'], path, 'Wavelet Edge Detection')
                generated_files.append(path)
            
            if 'curvelet_edges' in results:
                path = str(output_path / 'curvelet_edge.png')
                self.plot_edge_map(results['curvelet_edges'], path, 'Curvelet Edge Detection')
                generated_files.append(path)
            
            # Reconstruction visualizations
            if 'reconstructed_wavelet' in results and 'original_image' in results:
                path = str(output_path / 'reconstruction.png')
                self.plot_reconstruction(
                    results['original_image'],
                    results['reconstructed_wavelet'],
                    results.get('reconstructed_curvelet'),
                    path
                )
                generated_files.append(path)
            
            # Scientific metrics visualizations
            if 'radial_energy' in results:
                path = str(output_path / 'radial_energy.png')
                self.plot_radial_energy(results['radial_energy'], path)
                generated_files.append(path)
            
            if 'coefficient_histogram' in results:
                path = str(output_path / 'coefficient_histogram.png')
                self.plot_coefficient_histogram(results['coefficient_histogram'], path)
                generated_files.append(path)
            
            if 'scale_energy' in results:
                path = str(output_path / 'scale_energy.png')
                self.plot_scale_energy(results['scale_energy'], path)
                generated_files.append(path)
            
            if 'frequency_cone' in results:
                path = str(output_path / 'frequency_cone.png')
                self.plot_frequency_cone(results['frequency_cone'], path)
                generated_files.append(path)
            
            # Reconstruction error
            if 'reconstruction_error' in results:
                path = str(output_path / 'reconstruction_error.png')
                self.plot_reconstruction_error(results['reconstruction_error'], path)
                generated_files.append(path)
            
            if 'reconstruction_error_curve' in results:
                path = str(output_path / 'reconstruction_error_curve.png')
                self.plot_reconstruction_error_curve(results['reconstruction_error_curve'], path)
                generated_files.append(path)
            
            # Difference map
            if 'wavelet_edges' in results and 'curvelet_edges' in results:
                path = str(output_path / 'difference_map.png')
                self.plot_difference_map(
                    results['wavelet_edges'],
                    results['curvelet_edges'],
                    path
                )
                generated_files.append(path)
            
            # Edge overlay
            if 'original_image' in results and 'curvelet_edges' in results:
                path = str(output_path / 'edge_overlay.png')
                self.plot_edge_overlay(
                    results['original_image'],
                    results['curvelet_edges'],
                    path
                )
                generated_files.append(path)
            
            logger.info(f"Generated {len(generated_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
        
        return generated_files
    
    def plot_raw_image(
        self,
        image: np.ndarray,
        path: str
    ) -> None:
        """
        Plot the original uploaded image.
        
        **Validates: Requirements 11.1**
        
        Args:
            image: Original image array
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray', interpolation='nearest')
        ax.set_title('Original Image', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved raw image to {path}")
    
    def plot_normalized_image(
        self,
        image: np.ndarray,
        path: str
    ) -> None:
        """
        Plot the flux-normalized image.
        
        **Validates: Requirements 11.2**
        
        Args:
            image: Normalized image array
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray', interpolation='nearest')
        ax.set_title('Flux Normalized Image', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved normalized image to {path}")
    
    def plot_wavelet_decomposition(
        self,
        coeffs: Dict,
        path: str
    ) -> None:
        """
        Plot wavelet decomposition coefficients.
        
        **Validates: Requirements 5.6**
        
        Args:
            coeffs: Wavelet coefficients dictionary
            path: Output file path
        """
        # Extract coefficient levels
        if hasattr(coeffs, 'details'):
            details = coeffs.details
        else:
            details = coeffs.get('details', [])
        
        num_levels = len(details)
        fig, axes = plt.subplots(1, num_levels, figsize=(5 * num_levels, 5))
        
        if num_levels == 1:
            axes = [axes]
        
        for i, detail_coeffs in enumerate(details):
            # Combine horizontal, vertical, diagonal coefficients
            if isinstance(detail_coeffs, tuple) and len(detail_coeffs) == 3:
                cH, cV, cD = detail_coeffs
                combined = np.sqrt(cH**2 + cV**2 + cD**2)
            else:
                combined = np.abs(detail_coeffs)
            
            axes[i].imshow(combined, cmap='viridis', interpolation='nearest')
            axes[i].set_title(f'Level {i+1}', fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Wavelet Coefficients', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved wavelet decomposition to {path}")
    
    def plot_edge_map(
        self,
        edges: np.ndarray,
        path: str,
        title: str = 'Edge Detection'
    ) -> None:
        """
        Plot edge detection result.
        
        **Validates: Requirements 11.3, 11.4**
        
        Args:
            edges: Binary edge map
            path: Output file path
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(edges, cmap='gray', interpolation='nearest')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved edge map to {path}")
    
    def plot_reconstruction(
        self,
        original: np.ndarray,
        wavelet_recon: np.ndarray,
        curvelet_recon: Optional[np.ndarray],
        path: str
    ) -> None:
        """
        Plot reconstruction comparison.
        
        **Validates: Requirements 11.5**
        
        Args:
            original: Original image
            wavelet_recon: Wavelet reconstruction
            curvelet_recon: Curvelet reconstruction (optional)
            path: Output file path
        """
        num_plots = 3 if curvelet_recon is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        
        if num_plots == 2:
            axes = list(axes)
        
        axes[0].imshow(original, cmap='gray', interpolation='nearest')
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(wavelet_recon, cmap='gray', interpolation='nearest')
        axes[1].set_title('Wavelet Reconstruction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        if curvelet_recon is not None:
            axes[2].imshow(curvelet_recon, cmap='gray', interpolation='nearest')
            axes[2].set_title('Curvelet Reconstruction', fontsize=12, fontweight='bold')
            axes[2].axis('off')
        
        plt.suptitle('Multi-Scale Reconstruction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved reconstruction to {path}")
    
    def plot_radial_energy(
        self,
        energy: np.ndarray,
        path: str
    ) -> None:
        """
        Plot radial energy distribution.
        
        **Validates: Requirements 11.8**
        
        Args:
            energy: Radial energy array
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        radii = np.arange(len(energy))
        ax.plot(radii, energy, 'b-', linewidth=2)
        ax.fill_between(radii, energy, alpha=0.3)
        
        ax.set_xlabel('Radial Frequency', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title('Radial Energy Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved radial energy to {path}")
    
    def plot_coefficient_histogram(
        self,
        coeffs: np.ndarray,
        path: str
    ) -> None:
        """
        Plot coefficient magnitude histogram.
        
        **Validates: Requirements 11.9**
        
        Args:
            coeffs: Coefficient array
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Flatten coefficients and compute histogram
        flat_coeffs = coeffs.flatten()
        ax.hist(flat_coeffs, bins=100, color='blue', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Coefficient Magnitude', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Coefficient Magnitude Distribution', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved coefficient histogram to {path}")
    
    def plot_scale_energy(
        self,
        energy_per_scale: Dict[int, float],
        path: str
    ) -> None:
        """
        Plot energy vs scale level.
        
        **Validates: Requirements 11.11**
        
        Args:
            energy_per_scale: Dictionary mapping scale -> energy
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scales = sorted(energy_per_scale.keys())
        energies = [energy_per_scale[s] for s in scales]
        
        ax.bar(scales, energies, color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Scale Level', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title('Energy Distribution Across Scales', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved scale energy to {path}")
    
    def plot_frequency_cone(
        self,
        frequency_data: np.ndarray,
        path: str
    ) -> None:
        """
        Plot directional frequency decomposition (frequency cone).
        
        **Validates: Requirements 11.12**
        
        Args:
            frequency_data: 2D frequency domain data
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display frequency domain magnitude
        magnitude = np.log1p(np.abs(frequency_data))
        
        im = ax.imshow(magnitude, cmap='viridis', interpolation='nearest')
        ax.set_title('Directional Frequency Decomposition', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Log Magnitude')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved frequency cone to {path}")
    
    def plot_reconstruction_error(
        self,
        error_map: np.ndarray,
        path: str
    ) -> None:
        """
        Plot spatial distribution of reconstruction error.
        
        **Validates: Requirements 8.5**
        
        Args:
            error_map: 2D error map
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        im = ax.imshow(error_map, cmap='hot', interpolation='nearest')
        ax.set_title('Reconstruction Error Map', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Error')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved reconstruction error to {path}")
    
    def plot_reconstruction_error_curve(
        self,
        error_per_scale: Dict[int, float],
        path: str
    ) -> None:
        """
        Plot reconstruction error vs scale level.
        
        **Validates: Requirements 8.6, 11.13**
        
        Args:
            error_per_scale: Dictionary mapping scale -> error
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scales = sorted(error_per_scale.keys())
        errors = [error_per_scale[s] for s in scales]
        
        ax.plot(scales, errors, 'ro-', linewidth=2, markersize=8)
        
        ax.set_xlabel('Scale Level', fontsize=12)
        ax.set_ylabel('Reconstruction Error (RMSE)', fontsize=12)
        ax.set_title('Reconstruction Error vs Scale', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved reconstruction error curve to {path}")
    
    def plot_difference_map(
        self,
        wavelet_edges: np.ndarray,
        curvelet_edges: np.ndarray,
        path: str
    ) -> None:
        """
        Plot difference between wavelet and curvelet edge detection.
        
        **Validates: Requirements 9.4, 11.14**
        
        Args:
            wavelet_edges: Wavelet edge map
            curvelet_edges: Curvelet edge map
            path: Output file path
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Wavelet edges
        axes[0].imshow(wavelet_edges, cmap='gray', interpolation='nearest')
        axes[0].set_title('Wavelet Edges', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Curvelet edges
        axes[1].imshow(curvelet_edges, cmap='gray', interpolation='nearest')
        axes[1].set_title('Curvelet Edges', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Difference map
        difference = np.abs(curvelet_edges.astype(float) - wavelet_edges.astype(float))
        im = axes[2].imshow(difference, cmap='hot', interpolation='nearest')
        axes[2].set_title('Difference Map', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.suptitle('Transform Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved difference map to {path}")
    
    def plot_edge_overlay(
        self,
        image: np.ndarray,
        edges: np.ndarray,
        path: str
    ) -> None:
        """
        Plot detected edges overlaid on original image.
        
        **Validates: Requirements 9.5**
        
        Args:
            image: Original image
            edges: Binary edge map
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display original image
        ax.imshow(image, cmap='gray', alpha=0.7, interpolation='nearest')
        
        # Overlay edges in red
        edge_overlay = np.zeros((*edges.shape, 4))
        edge_overlay[edges > 0] = [1, 0, 0, 0.8]  # Red with alpha
        ax.imshow(edge_overlay, interpolation='nearest')
        
        ax.set_title('Edge Detection Overlay', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved edge overlay to {path}")
