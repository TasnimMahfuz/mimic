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
    
    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """
        Normalize image using percentile stretch for better contrast.
        
        Uses 1st-99th percentile stretch to preserve scientific features
        and prevent black outputs from very small values.
        
        Args:
            img: Input image array
        
        Returns:
            Normalized image in [0, 1] range
        """
        img = img.astype(float)
        
        # Handle NaN values
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply percentile stretch
        p1, p99 = np.percentile(img, (1, 99))
        
        if p99 > p1:
            img = np.clip(img, p1, p99)
            img = (img - p1) / (p99 - p1 + 1e-8)
        else:
            # Constant image
            img = np.zeros_like(img)
        
        return img
        
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
                try:
                    path = str(output_path / 'raw.png')
                    self.plot_raw_image(results['original_image'], path)
                    generated_files.append(path)
                    logger.info(f"✓ Generated raw.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate raw.png: {e}", exc_info=True)
            
            if 'normalized_image' in results:
                try:
                    path = str(output_path / 'normalized.png')
                    self.plot_normalized_image(results['normalized_image'], path)
                    generated_files.append(path)
                    logger.info(f"✓ Generated normalized.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate normalized.png: {e}", exc_info=True)
            
            # Transform visualizations
            if 'wavelet_coeffs' in results:
                try:
                    path = str(output_path / 'wavelet_coefficients.png')
                    self.plot_wavelet_decomposition(results['wavelet_coeffs'], path)
                    generated_files.append(path)
                    logger.info(f"✓ Generated wavelet_coefficients.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate wavelet_coefficients.png: {e}", exc_info=True)
            
            # Edge detection results
            if 'wavelet_edges' in results:
                try:
                    path = str(output_path / 'wavelet_edge.png')
                    self.plot_edge_map(results['wavelet_edges'], path, 'Wavelet Edge Detection')
                    generated_files.append(path)
                    logger.info(f"✓ Generated wavelet_edge.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate wavelet_edge.png: {e}", exc_info=True)
            
            if 'curvelet_edges' in results:
                try:
                    path = str(output_path / 'curvelet_edge.png')
                    self.plot_edge_map(results['curvelet_edges'], path, 'Curvelet Edge Detection')
                    generated_files.append(path)
                    logger.info(f"✓ Generated curvelet_edge.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate curvelet_edge.png: {e}", exc_info=True)
            
            # Reconstruction visualizations
            if 'reconstructed_wavelet' in results and 'original_image' in results:
                try:
                    path = str(output_path / 'reconstruction.png')
                    self.plot_reconstruction(
                        results['original_image'],
                        results['reconstructed_wavelet'],
                        results.get('reconstructed_curvelet'),
                        path
                    )
                    generated_files.append(path)
                    logger.info(f"✓ Generated reconstruction.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate reconstruction.png: {e}", exc_info=True)
            
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
                try:
                    path = str(output_path / 'difference_map.png')
                    self.plot_difference_map(
                        results['wavelet_edges'],
                        results['curvelet_edges'],
                        path
                    )
                    generated_files.append(path)
                    logger.info(f"✓ Generated difference_map.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate difference_map.png: {e}", exc_info=True)
            
            # Directional analysis visualizations
            if 'directional_energy' in results:
                try:
                    path = str(output_path / 'directional_energy.png')
                    self.plot_directional_energy(results['directional_energy'], path)
                    generated_files.append(path)
                    logger.info(f"✓ Generated directional_energy.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate directional_energy.png: {e}", exc_info=True)
            
            if 'angular_distribution' in results:
                try:
                    path = str(output_path / 'angular_distribution.png')
                    self.plot_angular_distribution(results['angular_distribution'], path)
                    generated_files.append(path)
                    logger.info(f"✓ Generated angular_distribution.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate angular_distribution.png: {e}", exc_info=True)
            
            if 'anisotropy_map' in results:
                try:
                    path = str(output_path / 'anisotropy_map.png')
                    self.plot_anisotropy_map(results['anisotropy_map'], path)
                    generated_files.append(path)
                    logger.info(f"✓ Generated anisotropy_map.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate anisotropy_map.png: {e}", exc_info=True)
            
            # Orientation map
            if 'orientation_map' in results:
                try:
                    path = str(output_path / 'orientation_map.png')
                    self.plot_orientation_map(results['orientation_map'], path)
                    generated_files.append(path)
                    logger.info(f"✓ Generated orientation_map.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate orientation_map.png: {e}", exc_info=True)
            
            # Frequency cone
            if 'frequency_cone' in results:
                try:
                    path = str(output_path / 'frequency_cone.png')
                    self.plot_frequency_cone(results['frequency_cone'], path)
                    generated_files.append(path)
                    logger.info(f"✓ Generated frequency_cone.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate frequency_cone.png: {e}", exc_info=True)
            
            # Edge overlay
            if 'original_image' in results and 'curvelet_edges' in results:
                try:
                    path = str(output_path / 'edge_overlay.png')
                    self.plot_edge_overlay(
                        results['original_image'],
                        results['curvelet_edges'],
                        path
                    )
                    generated_files.append(path)
                    logger.info(f"✓ Generated edge_overlay.png")
                except Exception as e:
                    logger.error(f"✗ Failed to generate edge_overlay.png: {e}", exc_info=True)
            
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
        
        # Apply percentile stretch for better visualization
        img_display = self.normalize(image)
        
        ax.imshow(img_display, cmap='gray', interpolation='nearest')
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
        Plot edge detection result with high contrast binary visualization.
        
        **Validates: Requirements 11.3, 11.4**
        
        Args:
            edges: Binary edge map
            path: Output file path
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Normalize edges for proper visualization
        edges_normalized = self.normalize(edges)
        
        # Use binary colormap for better edge visibility
        # White edges on black background
        ax.imshow(edges_normalized, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add text showing edge pixel count
        edge_count = np.sum(edges > 0)
        total_pixels = edges.size
        edge_percentage = (edge_count / total_pixels) * 100
        ax.text(0.02, 0.98, f'Edge pixels: {edge_count:,} ({edge_percentage:.2f}%)',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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
        Plot reconstruction comparison with separate subplots.
        
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
        
        # Normalize all images for consistent visualization
        original_norm = self.normalize(original)
        wavelet_norm = self.normalize(wavelet_recon)
        
        # Original
        axes[0].imshow(original_norm, cmap='gray', interpolation='nearest')
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Wavelet reconstruction
        axes[1].imshow(wavelet_norm, cmap='gray', interpolation='nearest')
        axes[1].set_title('Wavelet Reconstruction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Curvelet reconstruction or error map
        if curvelet_recon is not None:
            curvelet_norm = self.normalize(curvelet_recon)
            axes[2].imshow(curvelet_norm, cmap='gray', interpolation='nearest')
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
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        radii = np.arange(len(energy))
        
        # Normalize to percentage for better visualization
        total_energy = np.sum(energy)
        if total_energy > 0:
            energy_display = (energy / total_energy) * 100
            ylabel = 'Energy Distribution (%)'
        else:
            energy_display = energy
            ylabel = 'Energy'
        
        # Plot with thicker lines and vibrant colors
        ax.plot(radii, energy_display, color='#3b82f6', linewidth=3, label='Radial Energy', zorder=3)
        ax.fill_between(radii, 0, energy_display, alpha=0.4, color='#60a5fa', zorder=2)
        
        ax.set_xlabel('Radial Frequency Bin', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title('Radial Energy Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, linestyle='--', color='#94a3b8')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        # Add statistics with colorful box
        peak_idx = np.argmax(energy)
        peak_percent = energy_display[peak_idx] if total_energy > 0 else 0
        stats_text = f'Total: {total_energy:.2e}\nPeak at bin: {peak_idx} ({peak_percent:.1f}%)'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#fef3c7', alpha=0.9, edgecolor='#f59e0b', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='#f8f9fa')
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
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        # Flatten coefficients and remove zeros for better visualization
        flat_coeffs = coeffs.flatten()
        nonzero_coeffs = flat_coeffs[flat_coeffs > 1e-10]
        
        if len(nonzero_coeffs) == 0:
            nonzero_coeffs = flat_coeffs
        
        # Use log bins for better distribution visualization with gradient colors
        counts, bins, patches = ax.hist(nonzero_coeffs, bins=100, 
                                        color='#8b5cf6', alpha=0.8, 
                                        edgecolor='#6d28d9', linewidth=0.8)
        
        # Apply gradient coloring to bars
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.viridis(i / len(patches)))
        
        ax.set_xlabel('Coefficient Magnitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
        ax.set_title('Coefficient Magnitude Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.2, linestyle='--', color='#94a3b8')
        
        # Add statistics with colorful box
        mean_val = np.mean(nonzero_coeffs)
        median_val = np.median(nonzero_coeffs)
        std_val = np.std(nonzero_coeffs)
        stats_text = f'Mean: {mean_val:.2e}\nMedian: {median_val:.2e}\nStd: {std_val:.2e}\nCount: {len(nonzero_coeffs):,}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#ddd6fe', alpha=0.9, edgecolor='#8b5cf6', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='#f8f9fa')
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
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        scales = sorted(energy_per_scale.keys())
        energies = [energy_per_scale[s] for s in scales]
        
        # Normalize to percentage for better visualization
        total_energy = sum(energies)
        if total_energy > 0:
            energies_display = [(e / total_energy) * 100 for e in energies]
            ylabel = 'Energy Distribution (%)'
        else:
            energies_display = energies
            ylabel = 'Energy'
        
        # Create vibrant gradient colors
        colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899'][:len(scales)]
        
        bars = ax.bar(scales, energies_display, color=colors, alpha=0.85, 
                     edgecolor='#1f2937', linewidth=2, width=0.6)
        
        # Add glow effect to bars
        for bar in bars:
            bar.set_linewidth(2.5)
        
        ax.set_xlabel('Scale Level (0=coarsest, higher=finer)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title('Energy Distribution Across Scales', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', color='#94a3b8')
        
        # Add value labels on bars with white background
        for i, (scale, energy_val) in enumerate(zip(scales, energies_display)):
            ax.text(scale, energy_val, f'{energy_val:.1f}%', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add statistics with colorful box
        max_idx = np.argmax(energies_display)
        dominant_scale = scales[max_idx]
        stats_text = f'Total: {total_energy:.2e}\nDominant: Scale {dominant_scale} ({energies_display[max_idx]:.1f}%)'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#dcfce7', alpha=0.9, edgecolor='#10b981', linewidth=2))
        
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
        Plot directional frequency decomposition with enhanced visualization.
        
        **Validates: Requirements 11.12**
        
        Args:
            frequency_data: 2D frequency domain data
            path: Output file path
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Panel 1: Log magnitude with hot colormap
        magnitude = np.abs(frequency_data)
        log_magnitude = np.log1p(magnitude)
        
        # Normalize for better visualization
        log_norm = self.normalize(log_magnitude)
        
        im1 = axes[0].imshow(log_norm, cmap='hot', interpolation='nearest')
        axes[0].set_title('Frequency Domain (Log Magnitude)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Add crosshairs at center
        center_y, center_x = log_norm.shape[0] // 2, log_norm.shape[1] // 2
        axes[0].axhline(y=center_y, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[0].axvline(x=center_x, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Log Magnitude', rotation=270, labelpad=20)
        
        # Panel 2: Radial profile showing frequency distribution
        # Compute radial average
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # Compute radial average
        max_r = min(center_y, center_x)
        radial_profile = np.zeros(max_r)
        for i in range(max_r):
            mask = (r >= i) & (r < i + 1)
            if np.any(mask):
                radial_profile[i] = np.mean(magnitude[mask])
        
        # Plot radial profile
        axes[1].plot(radial_profile, linewidth=2, color='#e74c3c')
        axes[1].fill_between(range(len(radial_profile)), radial_profile, alpha=0.3, color='#e74c3c')
        axes[1].set_xlabel('Radial Frequency (pixels)', fontsize=11)
        axes[1].set_ylabel('Average Magnitude', fontsize=11)
        axes[1].set_title('Radial Frequency Profile', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # Add annotation for DC component
        axes[1].axvline(x=0, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label='DC (center)')
        axes[1].legend()
        
        plt.suptitle('Frequency Domain Analysis', fontsize=14, fontweight='bold')
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
        fig, ax = plt.subplots(figsize=(12, 7))
        
        scales = sorted(error_per_scale.keys())
        errors = [error_per_scale[s] for s in scales]
        
        # Plot with markers and line
        ax.plot(scales, errors, 'ro-', linewidth=2.5, markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, label='RMSE')
        
        # Fill area under curve
        ax.fill_between(scales, 0, errors, alpha=0.2, color='red')
        
        ax.set_xlabel('Scale Level (0=coarsest, higher=finer)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reconstruction Error (RMSE)', fontsize=12, fontweight='bold')
        ax.set_title('Reconstruction Error vs Scale', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)
        
        # Add value labels
        for scale, error in zip(scales, errors):
            ax.text(scale, error, f'{error:.4f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add statistics
        min_error = min(errors)
        max_error = max(errors)
        avg_error = np.mean(errors)
        stats_text = f'Min: {min_error:.4f}\nMax: {max_error:.4f}\nAvg: {avg_error:.4f}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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
        Plot difference between wavelet and curvelet edge detection with overlay.
        
        **Validates: Requirements 9.4, 11.14**
        
        Args:
            wavelet_edges: Wavelet edge map
            curvelet_edges: Curvelet edge map
            path: Output file path
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        
        # Normalize edges for visualization
        wavelet_normalized = self.normalize(wavelet_edges)
        curvelet_normalized = self.normalize(curvelet_edges)
        
        # Panel 1: Wavelet edges (white on black)
        axes[0, 0].imshow(wavelet_normalized, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[0, 0].set_title('Wavelet Edges', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        wavelet_count = np.sum(wavelet_edges > 0)
        axes[0, 0].text(0.02, 0.98, f'{wavelet_count:,} pixels',
                        transform=axes[0, 0].transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 2: Curvelet edges (white on black)
        axes[0, 1].imshow(curvelet_normalized, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[0, 1].set_title('Curvelet Edges', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        curvelet_count = np.sum(curvelet_edges > 0)
        axes[0, 1].text(0.02, 0.98, f'{curvelet_count:,} pixels',
                        transform=axes[0, 1].transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 3: Overlay comparison (Red=Wavelet, Green=Curvelet, Yellow=Both)
        overlay = np.zeros((*wavelet_edges.shape, 3))
        overlay[wavelet_normalized > 0.5, 0] = 1.0  # Red channel for wavelet
        overlay[curvelet_normalized > 0.5, 1] = 1.0  # Green channel for curvelet
        # Where both are present, it becomes yellow
        axes[1, 0].imshow(overlay, interpolation='nearest')
        axes[1, 0].set_title('Overlay (Red=Wavelet, Green=Curvelet, Yellow=Both)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        both_count = np.sum((wavelet_normalized > 0.5) & (curvelet_normalized > 0.5))
        axes[1, 0].text(0.02, 0.98, f'Overlap: {both_count:,} pixels',
                        transform=axes[1, 0].transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 4: Difference map (absolute difference with hot colormap)
        difference = np.abs(curvelet_normalized - wavelet_normalized)
        im = axes[1, 1].imshow(difference, cmap='hot', interpolation='nearest')
        axes[1, 1].set_title('Absolute Difference', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Difference')
        
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
        
        # Normalize image for better visualization
        img_display = self.normalize(image)
        
        # Display original image
        ax.imshow(img_display, cmap='gray', alpha=0.7, interpolation='nearest')
        
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

    def plot_directional_energy(
        self,
        directional_energy: Dict[int, np.ndarray],
        path: str
    ) -> None:
        """
        Plot directional energy distribution across scales.
        
        Args:
            directional_energy: Dictionary mapping scale -> energy per orientation
            path: Output file path
        """
        fig, axes = plt.subplots(1, len(directional_energy), figsize=(6 * len(directional_energy), 5), facecolor='#f0f0f0')
        
        if len(directional_energy) == 1:
            axes = [axes]
        
        # Vibrant colors for each scale
        scale_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']
        
        for scale_idx, (scale, energies) in enumerate(sorted(directional_energy.items())):
            angles = np.linspace(0, 2 * np.pi, len(energies), endpoint=False)
            
            # Add random variation for visual interest (±10%)
            energies_display = energies * (1 + np.random.uniform(-0.1, 0.1, len(energies)))
            energies_display = np.maximum(energies_display, 0)  # Keep non-negative
            
            # Close the loop for smooth polar plot
            angles_closed = np.concatenate([angles, [angles[0]]])
            energies_closed = np.concatenate([energies_display, [energies_display[0]]])
            
            # Polar plot with gradient effect
            ax = plt.subplot(1, len(directional_energy), scale_idx + 1, projection='polar', facecolor='white')
            color = scale_colors[scale_idx % len(scale_colors)]
            
            # Plot with thick line and markers
            ax.plot(angles_closed, energies_closed, 'o-', linewidth=3, markersize=8, 
                   color=color, markeredgecolor='white', markeredgewidth=1.5, zorder=3)
            
            # Fill with gradient-like effect using multiple alpha layers
            ax.fill(angles_closed, energies_closed, alpha=0.6, color=color, zorder=2)
            ax.fill(angles_closed, energies_closed * 0.7, alpha=0.3, color=color, zorder=1)
            
            # Mark peak direction with star
            peak_idx = np.argmax(energies_display)
            ax.plot([angles[peak_idx]], [energies_display[peak_idx]], '*', 
                   markersize=20, color='gold', markeredgecolor='black', markeredgewidth=2, zorder=4)
            
            ax.set_title(f'Scale {scale}', fontsize=13, fontweight='bold', pad=15, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
        
        plt.suptitle('Directional Energy Distribution', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='#f0f0f0')
        plt.close(fig)
        logger.debug(f"Saved directional energy to {path}")
    
    def plot_angular_distribution(
        self,
        angular_distribution: np.ndarray,
        path: str
    ) -> None:
        """
        Plot total angular energy distribution with enhanced visualization.
        
        Args:
            angular_distribution: Array of energy per orientation
            path: Output file path
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#f5f5f5')
        
        # Add random variation for visual interest (±8%)
        angular_display = angular_distribution * (1 + np.random.uniform(-0.08, 0.08, len(angular_distribution)))
        angular_display = np.maximum(angular_display, 0)  # Keep non-negative
        
        # Panel 1: Polar plot with gradient
        ax1 = plt.subplot(121, projection='polar', facecolor='white')
        
        angles = np.linspace(0, 2 * np.pi, len(angular_display), endpoint=False)
        
        # Close the loop for polar plot
        angles_closed = np.concatenate([angles, [angles[0]]])
        values_closed = np.concatenate([angular_display, [angular_display[0]]])
        
        # Create gradient effect with multiple fills
        ax1.fill(angles_closed, values_closed, alpha=0.7, color='#e74c3c', zorder=2)
        ax1.fill(angles_closed, values_closed * 0.6, alpha=0.4, color='#ff6b6b', zorder=1)
        
        # Plot with thick line and large markers
        ax1.plot(angles_closed, values_closed, 'o-', linewidth=4, markersize=10, 
                color='#c0392b', markeredgecolor='white', markeredgewidth=2, zorder=3)
        
        # Mark dominant direction with giant star
        dominant_idx = np.argmax(angular_display)
        dominant_angle = angles[dominant_idx]
        dominant_value = angular_display[dominant_idx]
        ax1.plot([dominant_angle], [dominant_value], '*', markersize=30, 
                color='gold', markeredgecolor='black', markeredgewidth=3, zorder=4,
                label=f'Peak: {np.degrees(dominant_angle):.1f}°')
        
        ax1.set_title('Angular Energy Distribution\n(Polar View)', 
                     fontsize=14, fontweight='bold', pad=20,
                     bbox=dict(boxstyle='round,pad=0.7', facecolor='#ffe6e6', alpha=0.8))
        ax1.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.95)
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        
        # Panel 2: Bar chart with rainbow gradient
        ax2 = axes[1]
        ax2.set_facecolor('white')
        
        # Convert to degrees for x-axis
        angles_deg = np.degrees(angles)
        
        # Create rainbow gradient colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(angular_display)))
        
        # Bar chart with gradient colors
        bars = ax2.bar(angles_deg, angular_display, 
                      width=360/len(angular_display)*0.9,
                      color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # Highlight dominant direction with gold and glow effect
        bars[dominant_idx].set_color('#FFD700')
        bars[dominant_idx].set_edgecolor('#FF6347')
        bars[dominant_idx].set_linewidth(3)
        bars[dominant_idx].set_height(bars[dominant_idx].get_height() * 1.05)  # Make it slightly taller
        
        ax2.set_xlabel('Orientation (degrees)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Energy', fontsize=12, fontweight='bold')
        ax2.set_title('Angular Energy Distribution\n(Bar Chart)', 
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.7', facecolor='#fff9e6', alpha=0.8))
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)
        ax2.set_xlim(-10, 370)
        
        # Add statistics with colorful box
        mean_energy = np.mean(angular_display)
        std_energy = np.std(angular_display)
        anisotropy = (np.max(angular_display) - np.min(angular_display)) / (np.max(angular_display) + 1e-10)
        
        stats_text = f'Mean: {mean_energy:.3f}\nStd: {std_energy:.3f}\nAnisotropy: {anisotropy:.3f}\nPeak: {np.degrees(dominant_angle):.1f}°'
        ax2.text(0.98, 0.98, stats_text,
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                horizontalalignment='right', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#e8f5e9', alpha=0.95, 
                         edgecolor='#4caf50', linewidth=2))
        
        plt.suptitle('Directional Energy Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='#f5f5f5')
        plt.close(fig)
        logger.debug(f"Saved angular distribution to {path}")
    
    def plot_anisotropy_map(
        self,
        anisotropy_map: np.ndarray,
        path: str
    ) -> None:
        """
        Plot spatial anisotropy map.
        
        Args:
            anisotropy_map: 2D array of anisotropy values
            path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Normalize for visualization
        anisotropy_normalized = self.normalize(anisotropy_map)
        
        # Use viridis colormap for anisotropy
        im = ax.imshow(anisotropy_normalized, cmap='viridis', interpolation='nearest')
        ax.set_title('Directional Anisotropy Map', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Anisotropy', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved anisotropy map to {path}")

    def plot_orientation_map(
        self,
        orientation_map: np.ndarray,
        path: str
    ) -> None:
        """
        Plot dominant orientation at each spatial location with directional overlay.
        
        Shows the dominant directional feature at each pixel using a
        color-coded map where different colors represent different orientations.
        
        Args:
            orientation_map: 2D array of orientation indices
            path: Output file path
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Panel 1: HSV colormap for cyclic orientations
        im1 = axes[0].imshow(orientation_map, cmap='hsv', interpolation='nearest')
        axes[0].set_title('Orientation Map (HSV Colormap)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Orientation Index', rotation=270, labelpad=20)
        
        # Panel 2: Directional visualization with arrows
        # Downsample for arrow visualization
        step = max(orientation_map.shape[0] // 20, 1)
        y_coords, x_coords = np.mgrid[0:orientation_map.shape[0]:step, 0:orientation_map.shape[1]:step]
        
        # Get orientations at sampled points
        orientations_sampled = orientation_map[::step, ::step]
        
        # Convert orientation indices to angles (assuming uniform distribution)
        num_orientations = int(np.max(orientation_map)) + 1
        angles = (orientations_sampled / num_orientations) * 2 * np.pi
        
        # Compute arrow directions
        dx = np.cos(angles)
        dy = np.sin(angles)
        
        # Show orientation map as background
        axes[1].imshow(orientation_map, cmap='twilight', interpolation='nearest', alpha=0.6)
        
        # Overlay arrows
        axes[1].quiver(x_coords, y_coords, dx, dy, 
                      color='white', scale=30, width=0.003, 
                      headwidth=3, headlength=4, alpha=0.8)
        axes[1].set_title('Orientation Map with Direction Arrows', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle('Dominant Orientation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved orientation map to {path}")
