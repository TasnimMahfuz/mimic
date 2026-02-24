"""
MIMIC Service Module

This module provides the main service layer for the MIMIC analysis pipeline,
including enhancement processing and scientific metrics computation.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 11.7, 13.1, 13.2, 13.3, 13.4, 13.10, 13.11**
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import cv2
from scipy import ndimage
import logging
import uuid
from datetime import datetime
from pathlib import Path

from app.models.mimic_run import MIMICRun
from app.services.transforms.fft_directional import CurveletCoefficients
from app.services.transforms.wavelet import WaveletTransform, WaveletCoefficients
from app.services.transforms.curvelet import CurveletTransform
from app.services.file_processing import FileProcessor
from app.services.visualization.generator import VisualizationGenerator

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """
    Container for complete processing pipeline results.
    
    Attributes:
        run_id: Unique identifier for this run
        original_image: Original input image
        normalized_image: Flux-normalized image
        wavelet_coeffs: Wavelet transform coefficients
        curvelet_coeffs: Curvelet transform coefficients
        wavelet_edges: Edge map from wavelet transform
        curvelet_edges: Edge map from curvelet transform
        reconstructed_wavelet: Reconstructed image from wavelet coefficients
        reconstructed_curvelet: Reconstructed image from curvelet coefficients
        scientific_metrics: Computed scientific metrics
        output_dir: Path to output directory
        visualization_paths: List of generated visualization file paths
    """
    run_id: str
    original_image: np.ndarray
    normalized_image: np.ndarray
    wavelet_coeffs: WaveletCoefficients
    curvelet_coeffs: CurveletCoefficients
    wavelet_edges: np.ndarray
    curvelet_edges: np.ndarray
    reconstructed_wavelet: np.ndarray
    reconstructed_curvelet: np.ndarray
    scientific_metrics: 'ScientificMetrics'
    output_dir: Path
    visualization_paths: List[str]


@dataclass
class ScientificMetrics:
    """
    Container for scientific metrics computed from curvelet analysis.
    
    Attributes:
        anisotropy_map: Spatial map of directional anisotropy
        directional_energy: Energy distribution per scale and angle
        radial_energy: Energy as a function of radial frequency
        angular_distribution: Total energy vs orientation angle
        edge_confidence: Confidence map for detected edges
    """
    anisotropy_map: np.ndarray
    directional_energy: Dict[int, np.ndarray]
    radial_energy: np.ndarray
    angular_distribution: np.ndarray
    edge_confidence: np.ndarray


class MIMICService:
    """
    Main service for MIMIC analysis pipeline orchestration.
    
    **Validates: Requirements 13.1, 13.2, 13.3, 13.4**
    """
    
    def __init__(self):
        """Initialize MIMIC service with required components."""
        from app.services.file_processing import FileProcessor
        from app.services.transforms.wavelet import WaveletTransform
        from app.services.transforms.curvelet import CurveletTransform
        from app.services.visualization.generator import VisualizationGenerator
    
        self.file_processor = FileProcessor()
        self.wavelet_transform = WaveletTransform()
        self.curvelet_transform = CurveletTransform()
        self.visualizer = VisualizationGenerator()
        logger.info("MIMIC Service initialized")


    def run(self, db, user_id, dataset):
        """Legacy run method for backward compatibility."""
        run = MIMICRun(
            dataset_name=dataset,
            result_path="fake/output/path",
            user_id=user_id
        )

        db.add(run)
        db.commit()

        return run
    
    def run_pipeline(
        self,
        file_path: str,
        edge_strength: float = 0.5,
        angular_resolution: int = 16,
        smoothing: float = 1.0,
        photon_threshold: float = 10.0,
        enhancement_factor: float = 2.0,
        run_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Execute complete MIMIC analysis pipeline.
        
        Orchestrates all six processing stages:
        1. Image loading and normalization
        2. Wavelet transform (baseline)
        3. Curvelet transform (primary)
        4. Edge detection
        5. Multi-scale reconstruction
        6. Scientific enhancements and metrics
        
        **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 17.5, 17.6, 17.7**
        
        Args:
            file_path: Path to input image file
            edge_strength: Edge detection threshold (0.0-1.0)
            angular_resolution: Number of directional orientations
            smoothing: Spatial smoothing kernel size
            photon_threshold: Photon count threshold for noise filtering
            enhancement_factor: Contrast enhancement multiplier
            run_id: Optional run identifier (generated if not provided)
        
        Returns:
            ProcessingResult containing all outputs and metadata
        """
        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        logger.info(f"Starting MIMIC pipeline for run_id={run_id}")
        
        # Create output directory
        output_dir = self.file_processor.create_run_directory(run_id)
        
        # Stage 1: Load and normalize
        logger.info("Stage 1: Loading and normalizing image")
        original_image = self.file_processor.load_image(file_path)
        normalized_image, scale_params = self.normalize_flux(original_image)
        
        # Apply noise filtering if threshold provided
        if photon_threshold > 0:
            normalized_image = self.apply_noise_filter(normalized_image, photon_threshold)
        
        # Stage 2: Wavelet transform
        logger.info("Stage 2: Performing wavelet transform")
        wavelet_coeffs = self.wavelet_transform.decompose(normalized_image, levels=3)
        reconstructed_wavelet = self.wavelet_transform.reconstruct(wavelet_coeffs)
        wavelet_edges = self.wavelet_transform.extract_edges(wavelet_coeffs, edge_strength)
        
        # Stage 3: Curvelet transform
        logger.info("Stage 3: Performing curvelet transform")
        curvelet_coeffs = self.curvelet_transform.decompose(
            normalized_image,
            levels=3,
            angular_resolution=angular_resolution
        )
        reconstructed_curvelet = self.curvelet_transform.reconstruct(curvelet_coeffs)
        
        # Stage 4: Edge detection from curvelet
        logger.info("Stage 4: Detecting edges")
        curvelet_edges = self.detect_edges_from_curvelet(curvelet_coeffs, edge_strength)
        
        # Stage 5: Apply enhancements
        logger.info("Stage 5: Applying enhancements")
        enhanced_image = self.apply_enhancement(
            normalized_image,
            enhancement_factor=enhancement_factor,
            smoothing=smoothing
        )
        
        # Stage 6: Compute scientific metrics
        logger.info("Stage 6: Computing scientific metrics")
        scientific_metrics = self.compute_scientific_metrics(
            curvelet_coeffs,
            edge_map=curvelet_edges
        )
        
        # Generate all visualizations
        logger.info("Generating visualizations")
        
        # Convert to dictionary format for visualization generator
        viz_data = {
            'original_image': original_image,
            'normalized_image': normalized_image,
            'wavelet_coeffs': wavelet_coeffs,
            'curvelet_coeffs': curvelet_coeffs,
            'wavelet_edges': wavelet_edges,
            'curvelet_edges': curvelet_edges,
            'reconstructed_wavelet': reconstructed_wavelet,
            'reconstructed_curvelet': reconstructed_curvelet,
            'scientific_metrics': scientific_metrics
        }
        
        visualization_paths = self.visualizer.generate_all(viz_data, str(output_dir))
        
        result = ProcessingResult(
            run_id=run_id,
            original_image=original_image,
            normalized_image=normalized_image,
            wavelet_coeffs=wavelet_coeffs,
            curvelet_coeffs=curvelet_coeffs,
            wavelet_edges=wavelet_edges,
            curvelet_edges=curvelet_edges,
            reconstructed_wavelet=reconstructed_wavelet,
            reconstructed_curvelet=reconstructed_curvelet,
            scientific_metrics=scientific_metrics,
            output_dir=output_dir,
            visualization_paths=visualization_paths
        )
        
        # Save metadata
        metadata = {
            'run_id': run_id,
            'timestamp': datetime.utcnow().isoformat(),
            'parameters': {
                'edge_strength': edge_strength,
                'angular_resolution': angular_resolution,
                'smoothing': smoothing,
                'photon_threshold': photon_threshold,
                'enhancement_factor': enhancement_factor
            },
            'input_file': file_path,
            'image_shape': list(original_image.shape),
            'scale_params': scale_params,
            'visualizations': visualization_paths
        }
        self.file_processor.save_metadata(metadata, output_dir / "metadata.json")
        
        logger.info(f"Pipeline complete for run_id={run_id}")
        
        return result
    
    def normalize_flux(
        self,
        image: np.ndarray,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Normalize image flux to [0, 1] range.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        
        Args:
            image: Input image array
            output_path: Optional path to save normalized image
        
        Returns:
            Tuple of (normalized_image, scale_parameters)
        """
        img_min = np.min(image)
        img_max = np.max(image)
        img_range = img_max - img_min
        
        if img_range > 0:
            normalized = (image - img_min) / img_range
        else:
            # Constant image
            normalized = np.zeros_like(image)
        
        scale_params = {
            'min': float(img_min),
            'max': float(img_max),
            'range': float(img_range)
        }
        
        if output_path:
            self.file_processor.save_image(normalized, output_path)
        
        logger.info(f"Normalized flux: [{img_min:.2f}, {img_max:.2f}] -> [0.0, 1.0]")
        
        return normalized, scale_params
    
    def denormalize_flux(
        self,
        normalized: np.ndarray,
        scale_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Denormalize image back to original intensity scale.
        
        **Validates: Requirements 3.5**
        
        Args:
            normalized: Normalized image in [0, 1] range
            scale_params: Scale parameters from normalize_flux
        
        Returns:
            Denormalized image in original intensity range
        """
        img_min = scale_params['min']
        img_range = scale_params['range']
        
        denormalized = normalized * img_range + img_min
        
        return denormalized
    
    def apply_noise_filter(
        self,
        image: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Apply threshold-based noise filtering.
        
        **Validates: Requirements 4.3**
        
        Args:
            image: Input image
            threshold: Threshold value (pixels below this are zeroed)
        
        Returns:
            Filtered image
        """
        filtered = image.copy()
        filtered[filtered < threshold] = 0.0
        
        logger.info(f"Applied noise filter with threshold {threshold}")
        
        return filtered
    
    def detect_edges_from_curvelet(
        self,
        coefficients: CurveletCoefficients,
        threshold: float
    ) -> np.ndarray:
        """
        Detect edges from curvelet coefficients.
        
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        Args:
            coefficients: Curvelet transform coefficients
            threshold: Edge strength threshold
        
        Returns:
            Binary edge map
        """
        height, width = coefficients.shape
        edge_map = np.zeros((height, width))
        
        # Accumulate coefficient magnitudes across scales and orientations
        for scale_idx in range(coefficients.scales):
            if scale_idx not in coefficients.coefficients:
                continue
            
            scale_coeffs = coefficients.coefficients[scale_idx]
            
            for angle_idx in scale_coeffs:
                coeff = scale_coeffs[angle_idx]
                
                # Resize to match output dimensions if needed
                if coeff.shape != (height, width):
                    coeff_resized = cv2.resize(
                        np.abs(coeff),
                        (width, height),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    coeff_resized = np.abs(coeff)
                
                edge_map += coeff_resized
        
        # Normalize and threshold
        if np.max(edge_map) > 0:
            edge_map /= np.max(edge_map)
        
        # Apply threshold to create binary edge map
        edges = (edge_map > threshold).astype(np.float64)
        
        logger.info(f"Detected edges with threshold {threshold}")
        
        return edges
    
    def apply_contrast_enhancement(
        self,
        image: np.ndarray,
        enhancement_factor: float = 2.0
    ) -> np.ndarray:
        """
        Apply contrast enhancement to improve feature visibility.
        
        **Validates: Requirements 10.1**
        """
        if enhancement_factor <= 0:
            raise ValueError("enhancement_factor must be positive")
        
        mean_intensity = np.mean(image)
        enhanced = mean_intensity + enhancement_factor * (image - mean_intensity)
        enhanced = np.clip(enhanced, 0.0, 1.0)
        
        logger.info(f"Applied contrast enhancement with factor {enhancement_factor}")
        
        return enhanced
    
    def apply_spatial_smoothing(
        self,
        image: np.ndarray,
        kernel_size: float = 1.0
    ) -> np.ndarray:
        """
        Apply spatial smoothing to reduce high-frequency noise.
        
        **Validates: Requirements 10.2**
        """
        if kernel_size < 0:
            raise ValueError("kernel_size must be non-negative")
        
        if kernel_size == 0:
            return image.copy()
        
        smoothed = ndimage.gaussian_filter(image, sigma=kernel_size)
        
        logger.info(f"Applied spatial smoothing with kernel size {kernel_size}")
        
        return smoothed
    
    def apply_enhancement(
        self,
        image: np.ndarray,
        enhancement_factor: float = 2.0,
        smoothing: float = 1.0
    ) -> np.ndarray:
        """
        Apply combined enhancement processing to improve feature visibility.
        
        **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 11.7, 13.10**
        """
        if smoothing > 0:
            processed = self.apply_spatial_smoothing(image, smoothing)
        else:
            processed = image.copy()
        
        if enhancement_factor != 1.0:
            processed = self.apply_contrast_enhancement(processed, enhancement_factor)
        
        return processed
    
    def compute_anisotropy_map(
        self,
        coefficients: CurveletCoefficients
    ) -> np.ndarray:
        """
        Compute spatial map of directional anisotropy.
        
        **Validates: Requirements 13.11**
        """
        height, width = coefficients.shape
        anisotropy_map = np.zeros((height, width))
        
        for scale_idx in range(1, coefficients.scales):
            if scale_idx not in coefficients.coefficients:
                continue
            
            scale_coeffs = coefficients.coefficients[scale_idx]
            energies = []
            
            for angle_idx in range(coefficients.orientations):
                if angle_idx in scale_coeffs:
                    coeff = scale_coeffs[angle_idx]
                    if coeff.shape != (height, width):
                        coeff_resized = cv2.resize(
                            np.abs(coeff),
                            (width, height),
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        coeff_resized = np.abs(coeff)
                    energies.append(coeff_resized)
            
            if len(energies) > 0:
                energies = np.array(energies)
                max_energy = np.max(energies, axis=0)
                mean_energy = np.mean(energies, axis=0)
                
                anisotropy = np.divide(
                    max_energy,
                    mean_energy,
                    out=np.ones_like(max_energy),
                    where=mean_energy > 1e-10
                )
                
                anisotropy_map += anisotropy
        
        if coefficients.scales > 1:
            anisotropy_map /= (coefficients.scales - 1)
        
        logger.info("Computed anisotropy map")
        
        return anisotropy_map
    
    def compute_radial_energy(
        self,
        coefficients: CurveletCoefficients
    ) -> np.ndarray:
        """
        Compute radial energy profile (energy vs radial frequency).
        
        **Validates: Requirements 13.11**
        """
        num_bins = 50
        radial_energy = np.zeros(num_bins)
        
        for scale_idx in range(coefficients.scales):
            if scale_idx not in coefficients.coefficients:
                continue
            
            scale_coeffs = coefficients.coefficients[scale_idx]
            scale_energy = 0.0
            
            for angle_idx in scale_coeffs:
                coeff = scale_coeffs[angle_idx]
                scale_energy += np.sum(np.abs(coeff) ** 2)
            
            bin_idx = min(scale_idx * (num_bins // coefficients.scales), num_bins - 1)
            radial_energy[bin_idx] += scale_energy
        
        logger.info("Computed radial energy profile")
        
        return radial_energy
    
    def compute_edge_confidence(
        self,
        coefficients: CurveletCoefficients,
        edge_map: np.ndarray
    ) -> np.ndarray:
        """
        Compute confidence map for detected edges.
        
        **Validates: Requirements 13.11**
        """
        height, width = coefficients.shape
        confidence_map = np.zeros((height, width))
        
        for scale_idx in range(coefficients.scales):
            if scale_idx not in coefficients.coefficients:
                continue
            
            scale_coeffs = coefficients.coefficients[scale_idx]
            
            for angle_idx in scale_coeffs:
                coeff = scale_coeffs[angle_idx]
                
                if coeff.shape != (height, width):
                    coeff_resized = cv2.resize(
                        np.abs(coeff),
                        (width, height),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    coeff_resized = np.abs(coeff)
                
                confidence_map += coeff_resized
        
        if np.max(confidence_map) > 0:
            confidence_map /= np.max(confidence_map)
        
        confidence_map = confidence_map * edge_map
        
        logger.info("Computed edge confidence map")
        
        return confidence_map
    
    def compute_scientific_metrics(
        self,
        coefficients: CurveletCoefficients,
        edge_map: Optional[np.ndarray] = None
    ) -> ScientificMetrics:
        """
        Compute comprehensive scientific metrics from curvelet analysis.
        
        **Validates: Requirements 13.11**
        """
        logger.info("Computing scientific metrics")
        
        anisotropy_map = self.compute_anisotropy_map(coefficients)
        
        directional_energy = {}
        angular_distribution = np.zeros(coefficients.orientations)
        
        for scale_idx in range(coefficients.scales):
            if scale_idx not in coefficients.coefficients:
                continue
            
            scale_coeffs = coefficients.coefficients[scale_idx]
            scale_energy = np.zeros(coefficients.orientations)
            
            for angle_idx in range(coefficients.orientations):
                if angle_idx in scale_coeffs:
                    coeff = scale_coeffs[angle_idx]
                    energy = np.sum(np.abs(coeff) ** 2)
                    scale_energy[angle_idx] = energy
                    angular_distribution[angle_idx] += energy
            
            directional_energy[scale_idx] = scale_energy
        
        radial_energy = self.compute_radial_energy(coefficients)
        
        if edge_map is not None:
            edge_confidence = self.compute_edge_confidence(coefficients, edge_map)
        else:
            edge_confidence = np.zeros(coefficients.shape)
        
        metrics = ScientificMetrics(
            anisotropy_map=anisotropy_map,
            directional_energy=directional_energy,
            radial_energy=radial_energy,
            angular_distribution=angular_distribution,
            edge_confidence=edge_confidence
        )
        
        logger.info("Scientific metrics computation complete")
        
        return metrics
