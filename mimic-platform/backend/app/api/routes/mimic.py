"""
MIMIC API Router

This module provides REST API endpoints for the MIMIC analysis pipeline.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11, 19.1, 19.8**
"""

import logging
import uuid
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.services.mimic_service import MIMICService
from app.schemas.mimic import ProcessingParameters, AnalysisResponse, ResultsResponse
from app.models.mimic_run import MIMICRun

logger = logging.getLogger(__name__)

router = APIRouter()


def get_db():
    """Database dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_service():
    """Service dependency."""
    return MIMICService()


@router.post("/run", response_model=AnalysisResponse)
async def run_mimic_analysis(
    file: UploadFile = File(..., description="Image file (FITS, PNG, JPEG, TIFF)"),
    edge_strength: float = Form(0.5, ge=0.0, le=1.0),
    angular_resolution: int = Form(16, ge=8, le=32),
    smoothing: float = Form(1.0, ge=0.0, le=5.0),
    photon_threshold: float = Form(10.0, ge=0.0, le=100.0),
    enhancement_factor: float = Form(2.0, ge=1.0, le=5.0),
    db: Session = Depends(get_db),
    service: MIMICService = Depends(get_service)
):
    """
    Run MIMIC analysis pipeline on uploaded image.
    
    Accepts an image file and processing parameters, executes the complete
    six-stage analysis pipeline, and returns visualization URLs.
    
    **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11**
    
    Args:
        file: Uploaded image file (FITS, PNG, JPEG, TIFF)
        edge_strength: Edge detection threshold (0.0-1.0)
        angular_resolution: Number of directional bins (8-32)
        smoothing: Spatial smoothing kernel size (0.0-5.0)
        photon_threshold: Photon count threshold (0.0-100.0)
        enhancement_factor: Contrast enhancement multiplier (1.0-5.0)
        db: Database session
    
    Returns:
        AnalysisResponse with run_id, status, and visualization URLs
    
    Raises:
        HTTPException 400: Invalid file format or processing error
        HTTPException 500: Internal server error
    """
    run_id = str(uuid.uuid4())
    
    try:
        logger.info(f"=" * 80)
        logger.info(f"RECEIVED ANALYSIS REQUEST")
        logger.info(f"  run_id: {run_id}")
        logger.info(f"  filename: {file.filename}")
        logger.info(f"  content_type: {file.content_type}")
        logger.info(f"  parameters: edge_strength={edge_strength}, angular_resolution={angular_resolution}")
        logger.info(f"              smoothing={smoothing}, photon_threshold={photon_threshold}")
        logger.info(f"              enhancement_factor={enhancement_factor}")
        logger.info(f"=" * 80)
        
        # Validate file format
        if not service.file_processor.is_supported_format(file.filename):
            logger.error(f"Unsupported file format: {file.filename}")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format. Supported formats: FITS, PNG, JPEG, TIFF"
            )
        
        # Save uploaded file temporarily
        temp_dir = Path("outputs") / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / f"{run_id}_{file.filename}"
        
        file_start = time.time()
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"FILE SAVED: {temp_file_path} ({file_size_mb:.2f} MB, {time.time() - file_start:.2f}s)")
        
        # Create database record
        db_start = time.time()
        db_run = MIMICRun(
            run_id=run_id,
            dataset_name=file.filename,
            result_path=f"outputs/run_{run_id}",
            status="processing",
            parameters={
                'edge_strength': edge_strength,
                'angular_resolution': angular_resolution,
                'smoothing': smoothing,
                'photon_threshold': photon_threshold,
                'enhancement_factor': enhancement_factor
            },
            created_at=datetime.utcnow()
        )
        db.add(db_run)
        db.commit()
        logger.info(f"DATABASE RECORD CREATED: run_id={run_id}, status=processing ({time.time() - db_start:.2f}s)")
        
        # Run pipeline with database session for status updates
        try:
            logger.info(f"STARTING PIPELINE EXECUTION")
            result = service.run_pipeline(
                file_path=str(temp_file_path),
                edge_strength=edge_strength,
                angular_resolution=angular_resolution,
                smoothing=smoothing,
                photon_threshold=photon_threshold,
                enhancement_factor=enhancement_factor,
                run_id=run_id
            )
            
            # Update database record with success
            db_start = time.time()
            db_run.status = "complete"
            db_run.completed_at = datetime.utcnow()
            db_run.metrics = {
                'num_visualizations': len(result.visualization_paths),
                'image_shape': list(result.original_image.shape),
                'edge_pixels': int(np.sum(result.curvelet_edges > 0))
            }
            db.commit()
            logger.info(f"DATABASE STATUS UPDATED: status=complete ({time.time() - db_start:.2f}s)")
            
            # Generate visualization URLs
            visualization_urls = [
                f"/outputs/run_{run_id}/{Path(p).name}"
                for p in result.visualization_paths
            ]
            
            logger.info(f"=" * 80)
            logger.info(f"ANALYSIS COMPLETE")
            logger.info(f"  run_id: {run_id}")
            logger.info(f"  visualizations: {len(visualization_urls)}")
            logger.info(f"  status: complete")
            logger.info(f"=" * 80)
            
            return AnalysisResponse(
                run_id=run_id,
                status="complete",
                message="Analysis completed successfully",
                visualization_urls=visualization_urls
            )
        
        except Exception as e:
            # Update database record with failure
            db_run.status = "failed"
            db_run.error_message = str(e)
            db_run.completed_at = datetime.utcnow()
            db.commit()
            
            logger.error(f"=" * 80)
            logger.error(f"PIPELINE EXECUTION FAILED")
            logger.error(f"  run_id: {run_id}")
            logger.error(f"  error: {str(e)}")
            logger.error(f"=" * 80)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR in run_mimic_analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/run/{run_id}/results", response_model=ResultsResponse)
def get_mimic_results(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get results for a specific MIMIC analysis run.
    
    Returns processing status, parameters, visualizations, and metrics
    for the specified run.
    
    **Validates: Requirements 12.1**
    
    Args:
        run_id: Unique identifier for the analysis run
        db: Database session
    
    Returns:
        ResultsResponse with status, visualizations, and metrics
    
    Raises:
        HTTPException 404: Run not found
    """
    logger.info(f"Fetching results for run_id={run_id}")
    
    # Query database for run
    db_run = db.query(MIMICRun).filter(MIMICRun.run_id == run_id).first()
    
    if not db_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    
    # Build response
    response = ResultsResponse(
        run_id=run_id,
        status=db_run.status,
        parameters=ProcessingParameters(**db_run.parameters) if db_run.parameters else None,
        metrics=db_run.metrics or {},
        timestamp=db_run.completed_at.isoformat() if db_run.completed_at else None,
        error_message=db_run.error_message
    )
    
    # If complete, load visualization paths
    if db_run.status == "complete":
        output_dir = Path(db_run.result_path)
        if output_dir.exists():
            viz_files = list(output_dir.glob("*.png"))
            response.visualizations = {
                f.stem: f"/outputs/run_{run_id}/{f.name}"
                for f in viz_files
            }
    
    logger.info(f"Returning results for run_id={run_id}, status={db_run.status}")
    
    return response