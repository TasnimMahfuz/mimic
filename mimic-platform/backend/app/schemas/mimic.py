"""
Pydantic schemas for MIMIC API endpoints.

**Validates: Requirements 12.3, 12.4, 12.5, 12.6, 12.7**
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ProcessingParameters(BaseModel):
    """
    Processing parameters for MIMIC analysis pipeline.
    
    **Validates: Requirements 12.3, 12.4, 12.5, 12.6, 12.7**
    """
    edge_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for edge detection (0.0-1.0)"
    )
    angular_resolution: int = Field(
        default=16,
        ge=8,
        le=32,
        description="Number of directional bins in curvelet decomposition"
    )
    smoothing: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Spatial smoothing kernel size"
    )
    photon_threshold: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Photon count threshold for noise filtering"
    )
    enhancement_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Contrast enhancement multiplier"
    )


class AnalysisResponse(BaseModel):
    """
    Response model for POST /mimic/run endpoint.
    
    **Validates: Requirements 12.8, 12.9, 12.10**
    """
    run_id: str = Field(description="Unique identifier for the analysis run")
    status: str = Field(description="Processing status: 'processing', 'complete', 'failed'")
    message: str = Field(description="Human-readable status message")
    visualization_urls: List[str] = Field(
        default_factory=list,
        description="URLs to generated visualization files"
    )


class ResultsResponse(BaseModel):
    """
    Response model for GET /mimic/run/{run_id}/results endpoint.
    
    **Validates: Requirements 12.1**
    """
    run_id: str = Field(description="Unique identifier for the analysis run")
    status: str = Field(description="Processing status: 'processing', 'complete', 'failed'")
    parameters: Optional[ProcessingParameters] = Field(
        default=None,
        description="Processing parameters used for this run"
    )
    visualizations: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of visualization names to URLs"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Computed scientific metrics"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp of run completion"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'"
    )
