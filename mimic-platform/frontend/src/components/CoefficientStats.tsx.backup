import React, { useState, useEffect } from 'react';

interface CoefficientStatsProps {
  runId: string;
  baseUrl: string;
}

interface CoefficientStatistics {
  transform_info: {
    wavelet: {
      type: string;
      levels: number;
      total_subbands: number;
      coefficients_per_level: { [key: string]: number };
    };
    curvelet: {
      type: string;
      scales: number;
      orientations: number;
      total_subbands: number;
    };
  };
  edge_detection: {
    wavelet_edges: number;
    curvelet_edges: number;
    wavelet_edge_density_percent: number;
    curvelet_edge_density_percent: number;
  };
  energy_distribution: {
    total_energy: number;
    energy_per_scale: { [key: string]: number };
    scale_energy_percentages: { [key: string]: number };
  };
  directional_analysis: {
    dominant_orientation_index: number;
    dominant_angle_degrees: number;
    angular_resolution: number;
    anisotropy_mean: number;
    anisotropy_std: number;
  };
  reconstruction_quality: {
    wavelet: {
      mse: number;
      psnr_db: number;
      quality_rating: string;
    };
    curvelet: {
      mse: number;
      psnr_db: number;
      quality_rating: string;
    };
  };
  image_properties: {
    total_pixels: number;
    width: number;
    height: number;
    intensity_range: number;
    intensity_min: number;
    intensity_max: number;
  };
}

interface MetadataStats {
  image_shape: number[];
  coefficient_statistics: CoefficientStatistics;
}

export const CoefficientStats: React.FC<CoefficientStatsProps> = ({ runId, baseUrl }) => {
  const [stats, setStats] = useState<MetadataStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${baseUrl}/outputs/run_${runId}/metadata.json`);
        if (!response.ok) throw new Error('Failed to load metadata');
        const data = await response.json();
        setStats(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load statistics');
      } finally {
        setLoading(false);
      }
    };

    if (runId) {
      fetchMetadata();
    }
  }, [runId, baseUrl]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading coefficient statistics...</p>
        </div>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p className="mt-2 text-sm">{error || 'Coefficient statistics not available'}</p>
          <p className="mt-1 text-xs text-gray-400">Run a new analysis to generate statistics</p>
        </div>
      </div>
    );
  }

  const cs = stats.coefficient_statistics;

  if (!cs) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p className="mt-2 text-sm">Coefficient statistics not available</p>
          <p className="mt-1 text-xs text-gray-400">Run a new analysis to generate statistics</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6 bg-gray-50">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Transform Coefficient Statistics</h2>
          <p className="text-gray-600">Quantitative analysis of wavelet and curvelet decomposition</p>
        </div>

        {/* Image Properties */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Image Properties</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="border-l-4 border-blue-500 pl-4">
              <p className="text-sm text-gray-600">Dimensions</p>
              <p className="text-xl font-bold text-gray-900">{cs.image_properties.width} × {cs.image_properties.height}</p>
            </div>
            <div className="border-l-4 border-purple-500 pl-4">
              <p className="text-sm text-gray-600">Total Pixels</p>
              <p className="text-xl font-bold text-gray-900">{cs.image_properties.total_pixels.toLocaleString()}</p>
            </div>
            <div className="border-l-4 border-green-500 pl-4">
              <p className="text-sm text-gray-600">Intensity Range</p>
              <p className="text-xl font-bold text-gray-900">{cs.image_properties.intensity_range.toFixed(1)}</p>
            </div>
          </div>
        </div>

        {/* Transform Information */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Transform Decomposition</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Wavelet Transform */}
            <div className="border border-gray-200 rounded-lg p-4 bg-gradient-to-br from-blue-50 to-white">
              <h4 className="font-semibold text-gray-900 mb-3">Wavelet Transform</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Type:</span>
                  <span className="font-medium text-gray-900">{cs.transform_info.wavelet.type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Decomposition Levels:</span>
                  <span className="font-medium text-gray-900">{cs.transform_info.wavelet.levels}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Subbands:</span>
                  <span className="font-medium text-gray-900">{cs.transform_info.wavelet.total_subbands}</span>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t border-gray-200">
                <p className="text-xs text-gray-600 mb-2">Coefficients per Level:</p>
                {Object.entries(cs.transform_info.wavelet.coefficients_per_level).map(([level, count]) => (
                  <div key={level} className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600">Level {level}:</span>
                    <span className="font-medium text-gray-900">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Curvelet Transform */}
            <div className="border border-gray-200 rounded-lg p-4 bg-gradient-to-br from-purple-50 to-white">
              <h4 className="font-semibold text-gray-900 mb-3">Curvelet Transform</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Type:</span>
                  <span className="font-medium text-gray-900 text-xs">{cs.transform_info.curvelet.type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Scale Levels:</span>
                  <span className="font-medium text-gray-900">{cs.transform_info.curvelet.scales}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Orientations:</span>
                  <span className="font-medium text-gray-900">{cs.transform_info.curvelet.orientations}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Subbands:</span>
                  <span className="font-medium text-gray-900">{cs.transform_info.curvelet.total_subbands}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Edge Detection Results */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Edge Detection Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Wavelet Edge Detection</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Edge Pixels:</span>
                  <span className="text-lg font-bold text-gray-900">{cs.edge_detection.wavelet_edges.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Edge Density:</span>
                  <span className="text-lg font-bold text-blue-600">{cs.edge_detection.wavelet_edge_density_percent.toFixed(2)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                  <div 
                    className="bg-blue-600 h-3 rounded-full transition-all" 
                    style={{ width: `${Math.min(cs.edge_detection.wavelet_edge_density_percent, 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Curvelet Edge Detection</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Edge Pixels:</span>
                  <span className="text-lg font-bold text-gray-900">{cs.edge_detection.curvelet_edges.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Edge Density:</span>
                  <span className="text-lg font-bold text-purple-600">{cs.edge_detection.curvelet_edge_density_percent.toFixed(2)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                  <div 
                    className="bg-purple-600 h-3 rounded-full transition-all" 
                    style={{ width: `${Math.min(cs.edge_detection.curvelet_edge_density_percent, 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 p-3 bg-gray-50 rounded border border-gray-200">
            <p className="text-xs text-gray-600">
              <span className="font-semibold">Interpretation:</span> Edge density indicates the percentage of pixels identified as edges. 
              Typical values range from 5-15% for astronomical images. Higher values may indicate noise or fine structure.
            </p>
          </div>
        </div>

        {/* Energy Distribution */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Energy Distribution Across Scales</h3>
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600">Total Energy:</span>
              <span className="text-lg font-bold text-gray-900">{cs.energy_distribution.total_energy.toExponential(2)}</span>
            </div>
          </div>
          <div className="space-y-3">
            {Object.entries(cs.energy_distribution.scale_energy_percentages).map(([scale, percentage]) => {
              const scaleLabel = scale === '0' ? 'Coarse (Low Freq)' : scale === '1' ? 'Medium' : 'Fine (High Freq)';
              return (
                <div key={scale}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-700">Scale {scale} - {scaleLabel}</span>
                    <span className="font-semibold text-gray-900">{percentage.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-green-500 to-green-600 h-2 rounded-full transition-all" 
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-4 p-3 bg-gray-50 rounded border border-gray-200">
            <p className="text-xs text-gray-600">
              <span className="font-semibold">Interpretation:</span> Energy distribution shows how image information is distributed across frequency scales. 
              Coarse scales capture large-scale structures, while fine scales capture details and edges.
            </p>
          </div>
        </div>

        {/* Directional Analysis */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Directional Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="border-l-4 border-indigo-500 pl-4">
              <p className="text-sm text-gray-600">Dominant Direction</p>
              <p className="text-2xl font-bold text-gray-900">{cs.directional_analysis.dominant_angle_degrees.toFixed(1)}°</p>
              <p className="text-xs text-gray-500">Orientation index: {cs.directional_analysis.dominant_orientation_index}</p>
            </div>
            <div className="border-l-4 border-pink-500 pl-4">
              <p className="text-sm text-gray-600">Mean Anisotropy</p>
              <p className="text-2xl font-bold text-gray-900">{cs.directional_analysis.anisotropy_mean.toFixed(3)}</p>
              <p className="text-xs text-gray-500">Directional preference strength</p>
            </div>
            <div className="border-l-4 border-orange-500 pl-4">
              <p className="text-sm text-gray-600">Anisotropy Std Dev</p>
              <p className="text-2xl font-bold text-gray-900">{cs.directional_analysis.anisotropy_std.toFixed(3)}</p>
              <p className="text-xs text-gray-500">Spatial variation</p>
            </div>
          </div>
          <div className="p-3 bg-gray-50 rounded border border-gray-200">
            <p className="text-xs text-gray-600">
              <span className="font-semibold">Interpretation:</span> Dominant direction indicates the primary orientation of features in the image. 
              Anisotropy measures directional preference (higher values = stronger directional features). 
              Values near 1.0 indicate isotropic (no preferred direction), while higher values indicate strong directional structure.
            </p>
          </div>
        </div>

        {/* Reconstruction Quality */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Reconstruction Quality Assessment</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Wavelet Quality */}
            <div className="border border-gray-200 rounded-lg p-4 bg-gradient-to-br from-orange-50 to-white">
              <h4 className="font-semibold text-gray-900 mb-3">Wavelet Reconstruction</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">PSNR:</span>
                  <span className="text-xl font-bold text-gray-900">{cs.reconstruction_quality.wavelet.psnr_db.toFixed(2)} dB</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">MSE:</span>
                  <span className="text-sm font-medium text-gray-900">{cs.reconstruction_quality.wavelet.mse.toExponential(3)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Quality Rating:</span>
                  <span className={`text-sm font-bold ${
                    cs.reconstruction_quality.wavelet.quality_rating === 'Excellent' ? 'text-green-600' :
                    cs.reconstruction_quality.wavelet.quality_rating === 'Good' ? 'text-blue-600' : 'text-yellow-600'
                  }`}>{cs.reconstruction_quality.wavelet.quality_rating}</span>
                </div>
              </div>
            </div>

            {/* Curvelet Quality */}
            <div className="border border-gray-200 rounded-lg p-4 bg-gradient-to-br from-teal-50 to-white">
              <h4 className="font-semibold text-gray-900 mb-3">Curvelet Reconstruction</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">PSNR:</span>
                  <span className="text-xl font-bold text-gray-900">{cs.reconstruction_quality.curvelet.psnr_db.toFixed(2)} dB</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">MSE:</span>
                  <span className="text-sm font-medium text-gray-900">{cs.reconstruction_quality.curvelet.mse.toExponential(3)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Quality Rating:</span>
                  <span className={`text-sm font-bold ${
                    cs.reconstruction_quality.curvelet.quality_rating === 'Excellent' ? 'text-green-600' :
                    cs.reconstruction_quality.curvelet.quality_rating === 'Good' ? 'text-blue-600' : 'text-yellow-600'
                  }`}>{cs.reconstruction_quality.curvelet.quality_rating}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 p-3 bg-gray-50 rounded border border-gray-200">
            <p className="text-xs text-gray-600">
              <span className="font-semibold">Interpretation:</span> PSNR (Peak Signal-to-Noise Ratio) measures reconstruction fidelity. 
              Values above 40 dB indicate excellent quality, 30-40 dB is good, below 30 dB may indicate information loss. 
              MSE (Mean Squared Error) quantifies pixel-wise reconstruction error (lower is better).
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};
