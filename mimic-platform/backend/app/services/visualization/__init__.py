"""
Visualization generation modules for MIMIC Analysis Dashboard.

This package contains visualization generators for:
- Transform coefficient visualizations
- Edge detection overlays
- Directional energy distributions
- Spectral analysis plots
- Comparison visualizations
"""

from .generator import VisualizationGenerator
from .curvelet_visualizer import CurveletVisualizer

__all__ = ['VisualizationGenerator', 'CurveletVisualizer']
