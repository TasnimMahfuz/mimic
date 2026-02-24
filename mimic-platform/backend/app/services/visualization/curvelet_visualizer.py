"""
Curvelet Visualization Module

This module provides visualization methods for curvelet transform results,
including edge detection, directional energy distributions, orientation maps,
and angular distribution plots.

**Validates: Requirements 6.6, 6.7, 7.3, 7.4, 11.4, 11.6, 11.10**
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional
import logging
from pathlib import Path

from ..transforms.fft_directional import CurveletCoefficients

logger = logging.getLogger(__name__)


class CurveletVisualizer:
    """
    Visualization generator for curvelet transform results.
    
    Provides methods to generate visualizations of curvelet decomposition
    results, including edge detection, directional energy distributions,
    orientation maps, and angular distribution plots.
    
    **Validates: Requirements 6.6, 6.7, 7.3, 7.4, 11.4, 11.6, 11.10**
    """
    
    def __init__(self):
        """Initialize the curvelet visualizer."""
        # Set matplotlib style for scientific plots
        plt.style.use('default')
        
    def generate_curvelet_edge(
        self,
        coefficients: CurveletCoefficients,
        threshold: float = 0.1,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate edge detection visualization from curvelet coefficients.
        
        Detects edges by thresholding the magnitude of curvelet coefficients
        across all scales and orientations. Strong coefficients indicate
        the presence of edges or curvilinear features.
        
        **Validates: Requirements 6.6, 11.4**
        
        Args:
            coefficients: CurveletCoefficients from decomposition
            threshold: Threshold for edge detection (0.0-1.0)
            output_path: Optional path to save the visualization
        
        Returns:
            Binary edge map as 2D numpy array
        """
        logger.info(f"Generating curvelet edge detection with threshold={threshold}")
        
        # Initialize edge map
        edge_map = np.zeros(coefficients.shape, dtype=np.float64)
        
        # Accumulate coefficient magnitudes across all scales and orientations
        for scale_idx in range(coefficients.scales):
            for angle_idx in range(coefficients.orientations):
                if scale_idx in coefficients.coefficients:
                    if angle_idx in coefficients.coefficients[scale_idx]:
                        coeff = coefficients.coefficients[scale_idx][angle_idx]
                        # Add magnitude to edge map
                        edge_map += np.abs(coeff)
        
        # Normalize to [0, 1]
        if edge_map.max() > 0:
            edge_map = edge_map / edge_map.max()
        
        # Apply threshold to create binary edge map
        binary_edges = (edge_map > threshold).astype(np.uint8)
        
        # Save visualization if path provided
        if output_path:
            self._save_edge_visualization(binary_edges, output_path)
            logger.info(f"Saved curvelet edge visualization to {output_path}")
        
        return binary_edges
    
    def generate_directional_energy(
        self,
        coefficients: CurveletCoefficients,
        output_path: Optional[str] = None
    ) -> Dict[int, np.ndarray]:
        """
        Generate directional energy distribution visualization.
        
        Computes and visualizes the energy distribution across orientations
        for each scale level. Shows which directions contain the most energy,
        revealing dominant edge orientations in the image.
        
        **Validates: Requirements 6.7, 11.6**
        
        Args:
            coefficients: CurveletCoefficients from decomposition
            output_path: Optional path to save the visualization
        
        Returns:
            Dictionary mapping scale -> array of energies per orientation
        """
        logger.info("Generating directional energy distribution")
        
        # Compute energy per orientation for each scale
        energy_dict = {}
        
        for scale_idx in range(coefficients.scales):
            energies = np.zeros(coefficients.orientations)
            
            for angle_idx in range(coefficients.orientations):
                if scale_idx in coefficients.coefficients:
                    if angle_idx in coefficients.coefficients[scale_idx]:
                        coeff = coefficients.coefficients[scale_idx][angle_idx]
                        # Energy is sum of squared magnitudes
                        energies[angle_idx] = np.sum(np.abs(coeff) ** 2)
            
            energy_dict[scale_idx] = energies
        
        # Save visualization if path provided
        if output_path:
            self._save_directional_energy_visualization(
                energy_dict,
                coefficients.orientations,
                output_path
            )
            logger.info(f"Saved directional energy visualization to {output_path}")
        
        return energy_dict
    
    def generate_orientation_map(
        self,
        coefficients: CurveletCoefficients,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate orientation map showing dominant directions.
        
        Creates a spatial map showing the dominant edge orientation at each
        pixel location. For each pixel, determines which orientation has the
        strongest coefficient magnitude across all scales.
        
        **Validates: Requirements 7.3, 11.6**
        
        Args:
            coefficients: CurveletCoefficients from decomposition
            output_path: Optional path to save the visualization
        
        Returns:
            2D array of dominant orientation indices
        """
        logger.info("Generating orientation map")
        
        # Initialize arrays to track maximum magnitude and corresponding orientation
        max_magnitude = np.zeros(coefficients.shape, dtype=np.float64)
        dominant_orientation = np.zeros(coefficients.shape, dtype=np.int32)
        
        # Find dominant orientation at each spatial location
        for scale_idx in range(coefficients.scales):
            for angle_idx in range(coefficients.orientations):
                if scale_idx in coefficients.coefficients:
                    if angle_idx in coefficients.coefficients[scale_idx]:
                        coeff = coefficients.coefficients[scale_idx][angle_idx]
                        magnitude = np.abs(coeff)
                        
                        # Update dominant orientation where this magnitude is larger
                        mask = magnitude > max_magnitude
                        dominant_orientation[mask] = angle_idx
                        max_magnitude[mask] = magnitude[mask]
        
        # Save visualization if path provided
        if output_path:
            self._save_orientation_map_visualization(
                dominant_orientation,
                coefficients.orientations,
                output_path
            )
            logger.info(f"Saved orientation map to {output_path}")
        
        return dominant_orientation
    
    def generate_angular_distribution(
        self,
        coefficients: CurveletCoefficients,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate angular distribution plot showing energy vs angle.
        
        Creates a plot showing the total energy as a function of orientation
        angle, summed across all scales. Reveals the dominant directional
        features in the image.
        
        **Validates: Requirements 7.4, 11.10**
        
        Args:
            coefficients: CurveletCoefficients from decomposition
            output_path: Optional path to save the visualization
        
        Returns:
            Array of total energies per orientation angle
        """
        logger.info("Generating angular distribution plot")
        
        # Compute total energy per orientation (summed across scales)
        total_energy = np.zeros(coefficients.orientations)
        
        for scale_idx in range(coefficients.scales):
            for angle_idx in range(coefficients.orientations):
                if scale_idx in coefficients.coefficients:
                    if angle_idx in coefficients.coefficients[scale_idx]:
                        coeff = coefficients.coefficients[scale_idx][angle_idx]
                        total_energy[angle_idx] += np.sum(np.abs(coeff) ** 2)
        
        # Save visualization if path provided
        if output_path:
            self._save_angular_distribution_visualization(
                total_energy,
                coefficients.orientations,
                output_path
            )
            logger.info(f"Saved angular distribution plot to {output_path}")
        
        return total_energy
    
    def _save_edge_visualization(
        self,
        edge_map: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save edge detection visualization to file.
        
        Args:
            edge_map: Binary edge map
            output_path: Path to save the image
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display edge map
        ax.imshow(edge_map, cmap='gray', interpolation='nearest')
        ax.set_title('Curvelet Edge Detection', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _save_directional_energy_visualization(
        self,
        energy_dict: Dict[int, np.ndarray],
        num_orientations: int,
        output_path: str
    ) -> None:
        """
        Save directional energy distribution visualization.
        
        Args:
            energy_dict: Dictionary mapping scale -> energies per orientation
            num_orientations: Number of orientations
            output_path: Path to save the image
        """
        num_scales = len(energy_dict)
        
        fig, axes = plt.subplots(1, num_scales, figsize=(5 * num_scales, 5))
        
        if num_scales == 1:
            axes = [axes]
        
        # Angles in degrees for plotting
        angles = np.linspace(0, 360, num_orientations, endpoint=False)
        
        for scale_idx, ax in enumerate(axes):
            if scale_idx in energy_dict:
                energies = energy_dict[scale_idx]
                
                # Create polar plot
                ax_polar = plt.subplot(1, num_scales, scale_idx + 1, projection='polar')
                ax_polar.plot(np.deg2rad(angles), energies, 'b-', linewidth=2)
                ax_polar.fill(np.deg2rad(angles), energies, alpha=0.3)
                ax_polar.set_title(f'Scale {scale_idx}', fontsize=12, fontweight='bold')
                ax_polar.set_theta_zero_location('N')
                ax_polar.set_theta_direction(-1)
        
        plt.suptitle('Directional Energy Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _save_orientation_map_visualization(
        self,
        orientation_map: np.ndarray,
        num_orientations: int,
        output_path: str
    ) -> None:
        """
        Save orientation map visualization.
        
        Args:
            orientation_map: 2D array of dominant orientation indices
            num_orientations: Number of orientations
            output_path: Path to save the image
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create custom colormap for orientations (HSV-like)
        cmap = plt.cm.hsv
        
        # Display orientation map
        im = ax.imshow(
            orientation_map,
            cmap=cmap,
            interpolation='nearest',
            vmin=0,
            vmax=num_orientations - 1
        )
        
        ax.set_title('Dominant Orientation Map', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar with angle labels
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Orientation Angle', rotation=270, labelpad=20)
        
        # Set colorbar ticks to show angles in degrees
        tick_positions = np.linspace(0, num_orientations - 1, min(8, num_orientations))
        tick_labels = [f"{int(360 * i / num_orientations)}°" for i in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _save_angular_distribution_visualization(
        self,
        total_energy: np.ndarray,
        num_orientations: int,
        output_path: str
    ) -> None:
        """
        Save angular distribution plot.
        
        Args:
            total_energy: Array of total energies per orientation
            num_orientations: Number of orientations
            output_path: Path to save the image
        """
        fig = plt.figure(figsize=(10, 8))
        
        # Create polar plot
        ax = plt.subplot(111, projection='polar')
        
        # Angles in radians
        angles = np.linspace(0, 2 * np.pi, num_orientations, endpoint=False)
        
        # Plot energy distribution
        ax.plot(angles, total_energy, 'b-', linewidth=2, label='Energy')
        ax.fill(angles, total_energy, alpha=0.3)
        
        # Formatting
        ax.set_title('Angular Energy Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.grid(True, alpha=0.3)
        
        # Add angle labels
        angle_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(angle_labels)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


    def generate_enhancement_overlay(
        self,
        original_image: np.ndarray,
        enhanced_image: np.ndarray,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate enhancement overlay visualization showing enhanced features.

        Creates a visualization that overlays enhanced features on the original
        image, allowing comparison between unenhanced and enhanced results.

        **Validates: Requirements 10.3, 11.7**

        Args:
            original_image: Original normalized image
            enhanced_image: Enhanced image after processing
            output_path: Optional path to save the visualization

        Returns:
            Overlay visualization as 2D numpy array
        """
        logger.info("Generating enhancement overlay visualization")

        # Compute difference to highlight enhanced features
        difference = np.abs(enhanced_image - original_image)

        # Normalize difference for visualization
        if difference.max() > 0:
            difference = difference / difference.max()

        # Save visualization if path provided
        if output_path:
            self._save_enhancement_overlay_visualization(
                original_image,
                enhanced_image,
                difference,
                output_path
            )
            logger.info(f"Saved enhancement overlay to {output_path}")

        return difference

    def _save_enhancement_overlay_visualization(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        difference: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save enhancement overlay visualization to file.

        Args:
            original: Original image
            enhanced: Enhanced image
            difference: Difference map
            output_path: Path to save the image
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(original, cmap='gray', interpolation='nearest')
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Enhanced image
        axes[1].imshow(enhanced, cmap='gray', interpolation='nearest')
        axes[1].set_title('Enhanced', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Difference overlay (enhanced features highlighted)
        axes[2].imshow(original, cmap='gray', alpha=0.7, interpolation='nearest')
        axes[2].imshow(difference, cmap='hot', alpha=0.5, interpolation='nearest')
        axes[2].set_title('Enhancement Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle('Feature Enhancement Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

