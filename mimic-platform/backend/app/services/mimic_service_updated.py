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
import time

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
        run_id: Optional[str] = None,
        db_session = None
    ) -> ProcessingResult:
        """
        Execute complete MIMIC analysis pipeline with detailed logging.
        
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
            db_session: Optional database session for status updates
        
        Returns:
            ProcessingResult containing all outputs and metadata
        """
        pipeline_start = time.time()
        
        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        logger.info(f"=" * 80)
        logger.info(f"MIMIC PIPELINE STARTED - run_id={run_id}")
        logger.info(f"Parameters: edge_strength={edge_strength}, angular_resolution={angular_resolution}")
        logger.info(f"            smoothing={smoothing}, photon_threshold={photon_threshold}")
        logger.info(f"            enhancement_factor={enhancement_factor}")
        logger.info(f"=" * 80)
        
        # Create output directory
        stage_start = time.time()
        output_dir = self.file_processor.create_run_directory(run_id)
        logger.info(f"Created output directory: {output_dir} ({time.time() - stage_start:.2f}s)")
        
        # Stage 1: Load and normalize
        stage_start = time.time()
        logger.info(f"[STAGE 1/6] Loading and normalizing image from {file_path}")
        original_image = self.file_processor.load_image(file_path)
        logger.info(f"  → Loaded image: shape={original_image.shape}, dtype={original_image.dtype}")
        
        normalized_image, scale_params = self.normalize_flux(original_image)
        logger.info(f"  → Normalized flux: min={scale_params['min']:.2f}, max={scale_params['max']:.2f}")
        
        # Apply noise filtering if threshold provided
        if photon_threshold > 0:
            normalized_image = self.apply_noise_filter(normalized_image, photon_threshold)
            logger.info(f"  → Applied noise filter: threshold={photon_threshold}")
        
        stage_time = time.time() - stage_start
        logger.info(f"[STAGE 1/6] COMPLETE - Image loading and normalization ({stage_time:.2f}s)")
        
        # Update database status
        if db_session:
            self._update_db_status(db_session, run_id, "processing", "Stage 1/6: Image loaded and normalized")
        
        # Stage 2: Wavelet transform
        stage_start = time.time()
        logger.info(f"[STAGE 2/6] Performing wavelet transform (3 levels)")
        wavelet_coeffs = self.wavelet_transform.decompose(normalized_image, levels=3)
        logger.info(f"  → Wavelet decomposition complete: {wavelet_coeffs.levels} levels")
        
        reconstructed_wavelet = self.wavelet_transform.reconstruct(wavelet_coeffs)
        logger.info(f"  → Wavelet reconstruction complete")
        
        wavelet_edges = self.wavelet_transform.extract_edges(wavelet_coeffs, edge_strength)
        logger.info(f"  → Wavelet edge extraction complete")
        
        stage_time = time.time() - stage_start
        logger.info(f"[STAGE 2/6] COMPLETE - Wavelet transform ({stage_time:.2f}s)")
        
        # Update database status
        if db_session:
            self._update_db_status(db_session, run_id, "processing", "Stage 2/6: Wavelet transform complete")
        
        # Stage 3: Curvelet transform
        stage_start = time.time()
        logger.info(f"[STAGE 3/6] Performing curvelet transform (3 levels, {angular_resolution} orientations)")
        curvelet_coeffs = self.curvelet_transform.decompose(
            normalized_image,
            levels=3,
            angular_resolution=angular_resolution
        )
        logger.info(f"  → Curvelet decomposition complete: {curvelet_coeffs.scales} scales, {curvelet_coeffs.orientations} orientations")
        
        reconstructed_curvelet = self.curvelet_transform.reconstruct(curvelet_coeffs)
        logger.info(f"  → Curvelet reconstruction complete")
        
        stage_time = time.time() - stage_start
        logger.info(f"[STAGE 3/6] COMPLETE - Curvelet transform ({stage_time:.2f}s)")
        
        # Update database status
        if db_session:
            self._update_db_status(db_session, run_id, "processing", "Stage 3/6: Curvelet transform complete")
        
        # Stage 4: Edge detection from curvelet
        stage_start = time.time()
        logger.info(f"[STAGE 4/6] Detecting edges from curvelet coefficients")
        curvelet_edges = self.detect_edges_from_curvelet(curvelet_coeffs, edge_strength)
        edge_pixels = np.sum(curvelet_edges > 0)
        logger.info(f"  → Edge detection complete: {edge_pixels} edge pixels detected")
        
        stage_time = time.time() - stage_start
        logger.info(f"[STAGE 4/6] COMPLETE - Edge detection ({stage_time:.2f}s)")
        
        # Update database status
        if db_session:
            self._update_db_status(db_session, run_id, "processing", "Stage 4/6: Edge detection complete")
        
        # Stage 5: Apply enhancements
        stage_start = time.time()
        logger.info(f"[STAGE 5/6] Applying enhancements (factor={enhancement_factor}, smoothing={smoothing})")
        enhanced_image = self.apply_enhancement(
            normalized_image,
            enhancement_factor=enhancement_factor,
            smoothing=smoothing
        )
        logger.info(f"  → Enhancement complete")
        
        stage_time = time.time() - stage_start
        logger.info(f"[STAGE 5/6] COMPLETE - Enhancement processing ({stage_time:.2f}s)")
        
        # Update database status
        if db_session:
            self._update_db_status(db_session, run_id, "processing", "Stage 5/6: Enhancement complete")
        
        # Stage 6: Compute scientific metrics
        stage_start = time.time()
        logger.info(f"[STAGE 6/6] Computing scientific metrics")
        scientific_metrics = self.compute_scientific_metrics(
            curvelet_coeffs,
            edge_map=curvelet_edges
        )
        logger.info(f"  → Anisotropy map computed: shape={scientific_metrics.anisotropy_map.shape}")
        logger.info(f"  → Directional energy computed: {len(scientific_metrics.directional_energy)} scales")
        logger.info(f"  → Radial energy profile computed: {len(scientific_metrics.radial_energy)} bins")
        logger.info(f"  → Angular distribution computed: {len(scientific_metrics.angular_distribution)} angles")
        
        stage_time = time.time() - stage_start
        logger.info(f"[STAGE 6/6] COMPLETE - Scientific metrics ({stage_time:.2f}s)")
        
        # Update database status
        if db_session:
            self._update_db_status(db_session, run_id, "processing", "Stage 6/6: Metrics computed, generating visualizations")
        
        # Generate all visualizations
        stage_start = time.time()
        logger.info(f"Generating visualizations")
        
        # Compute orientation map from curvelet coefficients
        orientation_map = self.curvelet_transform.compute_orientation_map(curvelet_coeffs)
        
        # Compute frequency cone (FFT magnitude of original image)
        frequency_cone = np.fft.fftshift(np.abs(np.fft.fft2(normalized_image)))
        
        # Compute additional metrics for visualization
        # Coefficient histogram from curvelet coefficients
        all_coeffs = []
        for scale_idx in curvelet_coeffs.coefficients:
            for angle_idx in curvelet_coeffs.coefficients[scale_idx]:
                coeff = curvelet_coeffs.coefficients[scale_idx][angle_idx]
                all_coeffs.append(np.abs(coeff).flatten())
        coefficient_histogram = np.concatenate(all_coeffs) if all_coeffs else np.array([0])
        
        # Scale energy from curvelet coefficients
        scale_energy = {}
        for scale_idx in curvelet_coeffs.coefficients:
            scale_coeffs = curvelet_coeffs.coefficients[scale_idx]
            total_energy = 0.0
            for angle_idx in scale_coeffs:
                coeff = scale_coeffs[angle_idx]
                total_energy += np.sum(np.abs(coeff) ** 2)
            scale_energy[scale_idx] = total_energy
        
        # Reconstruction error per scale (using wavelet)
        reconstruction_error_curve = self.wavelet_transform.compute_reconstruction_error_per_scale(
            normalized_image, wavelet_coeffs
        )
        
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
            'scientific_metrics': scientific_metrics,
            # Extract directional analysis data from scientific_metrics
            'directional_energy': scientific_metrics.directional_energy,
            'angular_distribution': scientific_metrics.angular_distribution,
            'anisotropy_map': scientific_metrics.anisotropy_map,
            'radial_energy': scientific_metrics.radial_energy,
            # Add orientation map and frequency cone
            'orientation_map': orientation_map,
            'frequency_cone': frequency_cone,
            # Add computed metrics
            'coefficient_histogram': coefficient_histogram,
            'scale_energy': scale_energy,
            'reconstruction_error_curve': reconstruction_error_curve,
        }
        
        visualization_paths = self.visualizer.generate_all(viz_data, str(output_dir))
        logger.info(f"  → Generated {len(visualization_paths)} visualizations")
        
        stage_time = time.time() - stage_start
        logger.info(f"Visualization generation complete ({stage_time:.2f}s)")
        
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
        total_time = time.time() - pipeline_start
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
            'visualizations': visualization_paths,
            'execution_time_seconds': total_time
        }
        self.file_processor.save_metadata(metadata, output_dir / "metadata.json")
        
        logger.info(f"=" * 80)
        logger.info(f"MIMIC PIPELINE COMPLETE - run_id={run_id}")
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Visualizations: {len(visualization_paths)} files")
        logger.info(f"=" * 80)
        
        return result
    
    def _update_db_status(self, db_session, run_id: str, status: str, message: str = None):
        """Update database run status with optional message."""
        try:
            db_run = db_session.query(MIMICRun).filter(MIMICRun.run_id == run_id).first()
            if db_run:
                db_run.status = status
                if message:
                    if not db_run.metrics:
                        db_run.metrics = {}
                    db_run.metrics['progress_message'] = message
                db_session.commit()
                logger.info(f"Database status updated: {status} - {message}")
        except Exception as e:
            logger.warning(f"Failed to update database status: {e}")
            db_session.rollback()
    
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
        Detect edges from curvelet coefficients using adaptive percentile threshold.
        
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        Args:
            coefficients: Curvelet transform coefficients
            threshold: Edge strength parameter (0.0-1.0)
        
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
        
        # Normalize edge map
        if np.max(edge_map) > 0:
            edge_map /= np.max(edge_map)
        
        # Apply lenient adaptive percentile threshold for astronomy images
        # Convert threshold parameter to percentile (0.5 → 62.5th percentile)
        percentile = 50 + (threshold * 25)  # Maps [0,1] to [50,75] instead of [50,100]
        
        if np.any(edge_map > 0):
            # FIX #2: Use 3rd percentile instead of 10th for better edge detection
            min_thr = np.percentile(edge_map[edge_map > 0], 3)
            adaptive_thr = max(np.percentile(edge_map[edge_map > 0], percentile), min_thr)
        else:
            adaptive_thr = 0.01
        
        # Create binary edge map
        edges = (edge_map > adaptive_thr).astype(np.float64)
        
        edge_count = np.sum(edges)
        logger.info(
            f"Detected curvelet edges: {edge_count} pixels "
            f"(adaptive threshold: {adaptive_thr:.4f})"
        )
        
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
        
        # Compute energy per scale and orientation
        for scale_idx in range(coefficients.scales):
            if scale_idx not in coefficients.coefficients:
                continue
            
            scale_coeffs = coefficients.coefficients[scale_idx]
            scale_energy = np.zeros(coefficients.orientations)
            
            for angle_idx in range(coefficients.orientations):
                if angle_idx in scale_coeffs:
                    coeff = scale_coeffs[angle_idx]
                    # Use L2 norm (energy) for each orientation
                    energy = np.sum(np.abs(coeff) ** 2)
                    scale_energy[angle_idx] = energy
                    
                    # Weight finer scales more (they capture directional features better)
                    # Scale 0 = coarsest, higher scales = finer details
                    scale_weight = (scale_idx + 1) ** 1.5  # Emphasize finer scales
                    angular_distribution[angle_idx] += energy * scale_weight
            
            directional_energy[scale_idx] = scale_energy
        
        # Normalize angular distribution to [0, 1] for better visualization
        if np.max(angular_distribution) > 0:
            angular_distribution = angular_distribution / np.max(angular_distribution)
        
        # Apply smoothing to reduce noise in angular distribution
        # Use circular convolution to respect periodicity
        from scipy.ndimage import convolve1d
        kernel = np.array([0.25, 0.5, 0.25])  # Simple smoothing kernel
        angular_distribution = convolve1d(angular_distribution, kernel, mode='wrap')
        
        logger.info(f"Angular distribution range: [{np.min(angular_distribution):.4f}, {np.max(angular_distribution):.4f}]")
        logger.info(f"Dominant orientation: {np.argmax(angular_distribution)} (energy: {np.max(angular_distribution):.4f})")
        
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
