import React, { useState, useEffect } from 'react';

interface CoefficientStatsTabsProps {
  runId: string;
  baseUrl: string;
}

interface CoefficientStatistics {
  transform_info: any;
  edge_detection: any;
  energy_distribution: any;
  directional_analysis: any;
  reconstruction_quality: any;
  image_properties: any;
}

interface MetadataStats {
  image_shape: number[];
  coefficient_statistics: CoefficientStatistics;
}

export const CoefficientStatsTabs: React.FC<CoefficientStatsTabsProps> = ({ runId, baseUrl }) => {
  const [stats, setStats] = useState<MetadataStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('image-properties');

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

  const downloadPDF = () => {
    if (!stats?.coefficient_statistics) return;
    
    // Create text content
    const cs = stats.coefficient_statistics;
    let content = 'MIMIC ANALYSIS - COEFFICIENT STATISTICS REPORT\n';
    content += '='.repeat(60) + '\n\n';
    content += `Run ID: ${runId}\n`;
    content += `Generated: ${new Date().toLocaleString()}\n\n`;
    
    content += 'IMAGE PROPERTIES\n' + '-'.repeat(60) + '\n';
    content += `Dimensions: ${cs.image_properties.width} × ${cs.image_properties.height}\n`;
    content += `Total Pixels: ${cs.image_properties.total_pixels.toLocaleString()}\n`;
    content += `Intensity Range: ${cs.image_properties.intensity_range.toFixed(1)}\n\n`;
    
    content += 'TRANSFORM INFORMATION\n' + '-'.repeat(60) + '\n';
    content += `Wavelet: ${cs.transform_info.wavelet.type}, ${cs.transform_info.wavelet.levels} levels\n`;
    content += `Curvelet: ${cs.transform_info.curvelet.scales} scales, ${cs.transform_info.curvelet.orientations} orientations\n\n`;
    
    content += 'EDGE DETECTION\n' + '-'.repeat(60) + '\n';
    content += `Wavelet Edges: ${cs.edge_detection.wavelet_edges.toLocaleString()} (${cs.edge_detection.wavelet_edge_density_percent.toFixed(2)}%)\n`;
    content += `Curvelet Edges: ${cs.edge_detection.curvelet_edges.toLocaleString()} (${cs.edge_detection.curvelet_edge_density_percent.toFixed(2)}%)\n\n`;
    
    content += 'ENERGY DISTRIBUTION\n' + '-'.repeat(60) + '\n';
    content += `Total Energy: ${cs.energy_distribution.total_energy.toExponential(2)}\n`;
    Object.entries(cs.energy_distribution.scale_energy_percentages).forEach(([scale, pct]: [string, any]) => {
      content += `Scale ${scale}: ${pct.toFixed(1)}%\n`;
    });
    content += '\n';
    
    content += 'DIRECTIONAL ANALYSIS\n' + '-'.repeat(60) + '\n';
    content += `Dominant Direction: ${cs.directional_analysis.dominant_angle_degrees.toFixed(1)}°\n`;
    content += `Mean Anisotropy: ${cs.directional_analysis.anisotropy_mean.toFixed(3)}\n`;
    content += `Anisotropy Std Dev: ${cs.directional_analysis.anisotropy_std.toFixed(3)}\n\n`;
    
    content += 'RECONSTRUCTION QUALITY\n' + '-'.repeat(60) + '\n';
    content += `Wavelet: PSNR ${cs.reconstruction_quality.wavelet.psnr_db.toFixed(2)} dB (${cs.reconstruction_quality.wavelet.quality_rating})\n`;
    content += `Curvelet: PSNR ${cs.reconstruction_quality.curvelet.psnr_db.toFixed(2)} dB (${cs.reconstruction_quality.curvelet.quality_rating})\n`;
    
    // Download as text file (PDF generation requires library)
    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mimic_stats_${runId}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

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

  if (error || !stats?.coefficient_statistics) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <p className="mt-2 text-sm">{error || 'Coefficient statistics not available'}</p>
          <p className="mt-1 text-xs text-gray-400">Run a new analysis to generate statistics</p>
        </div>
      </div>
    );
  }

  const cs = stats.coefficient_statistics;

  const tabs = [
    { id: 'image-properties', label: 'Image Properties' },
    { id: 'transform-info', label: 'Transform Info' },
    { id: 'edge-detection', label: 'Edge Detection' },
    { id: 'energy-distribution', label: 'Energy Distribution' },
    { id: 'directional-analysis', label: 'Directional Analysis' },
    { id: 'reconstruction-quality', label: 'Reconstruction Quality' },
    { id: 'download', label: 'Download Report' },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Tab Navigation */}
      <div className="border-b border-gray-200 bg-white px-4">
        <nav className="flex space-x-2 overflow-x-auto" aria-label="Tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors ${
                activeTab === tab.id
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
        {activeTab === 'image-properties' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Image Properties</h2>
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                <div className="border-l-4 border-blue-500 pl-4">
                  <p className="text-sm text-gray-600">Dimensions</p>
                  <p className="text-2xl font-bold text-gray-900">{cs.image_properties.width} × {cs.image_properties.height}</p>
                </div>
                <div className="border-l-4 border-purple-500 pl-4">
                  <p className="text-sm text-gray-600">Total Pixels</p>
                  <p className="text-2xl font-bold text-gray-900">{cs.image_properties.total_pixels.toLocaleString()}</p>
                </div>
                <div className="border-l-4 border-green-500 pl-4">
                  <p className="text-sm text-gray-600">Intensity Range</p>
                  <p className="text-2xl font-bold text-gray-900">{cs.image_properties.intensity_range.toFixed(1)}</p>
                </div>
                <div className="border-l-4 border-orange-500 pl-4">
                  <p className="text-sm text-gray-600">Min Intensity</p>
                  <p className="text-2xl font-bold text-gray-900">{cs.image_properties.intensity_min.toFixed(1)}</p>
                </div>
                <div className="border-l-4 border-red-500 pl-4">
                  <p className="text-sm text-gray-600">Max Intensity</p>
                  <p className="text-2xl font-bold text-gray-900">{cs.image_properties.intensity_max.toFixed(1)}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'transform-info' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Transform Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Wavelet Transform</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Type:</span>
                    <span className="font-medium">{cs.transform_info.wavelet.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Levels:</span>
                    <span className="font-medium">{cs.transform_info.wavelet.levels}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Subbands:</span>
                    <span className="font-medium">{cs.transform_info.wavelet.total_subbands}</span>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Curvelet Transform</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Type:</span>
                    <span className="font-medium text-sm">{cs.transform_info.curvelet.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Scales:</span>
                    <span className="font-medium">{cs.transform_info.curvelet.scales}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Orientations:</span>
                    <span className="font-medium">{cs.transform_info.curvelet.orientations}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Subbands:</span>
                    <span className="font-medium">{cs.transform_info.curvelet.total_subbands}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'edge-detection' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Edge Detection Analysis</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Wavelet Edge Detection</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-600">Edge Pixels:</span>
                      <span className="text-xl font-bold">{cs.edge_detection.wavelet_edges.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-600">Edge Density:</span>
                      <span className="text-xl font-bold text-blue-600">{cs.edge_detection.wavelet_edge_density_percent.toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div 
                        className="bg-blue-600 h-4 rounded-full" 
                        style={{ width: `${Math.min(cs.edge_detection.wavelet_edge_density_percent, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Curvelet Edge Detection</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-600">Edge Pixels:</span>
                      <span className="text-xl font-bold">{cs.edge_detection.curvelet_edges.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-600">Edge Density:</span>
                      <span className="text-xl font-bold text-purple-600">{cs.edge_detection.curvelet_edge_density_percent.toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div 
                        className="bg-purple-600 h-4 rounded-full" 
                        style={{ width: `${Math.min(cs.edge_detection.curvelet_edge_density_percent, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'energy-distribution' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Energy Distribution Across Scales</h2>
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="mb-6">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Total Energy:</span>
                  <span className="text-2xl font-bold">{cs.energy_distribution.total_energy.toExponential(2)}</span>
                </div>
              </div>
              <div className="space-y-4">
                {Object.entries(cs.energy_distribution.scale_energy_percentages).map(([scale, percentage]: [string, any]) => {
                  const scaleLabel = scale === '0' ? 'Coarse (Low Freq)' : scale === '1' ? 'Medium' : 'Fine (High Freq)';
                  return (
                    <div key={scale}>
                      <div className="flex justify-between mb-2">
                        <span className="text-gray-700 font-medium">Scale {scale} - {scaleLabel}</span>
                        <span className="font-bold text-gray-900">{percentage.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full" 
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'directional-analysis' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Directional Analysis</h2>
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="border-l-4 border-indigo-500 pl-4">
                  <p className="text-sm text-gray-600">Dominant Direction</p>
                  <p className="text-3xl font-bold text-gray-900">{cs.directional_analysis.dominant_angle_degrees.toFixed(1)}°</p>
                  <p className="text-xs text-gray-500 mt-1">Index: {cs.directional_analysis.dominant_orientation_index}</p>
                </div>
                <div className="border-l-4 border-pink-500 pl-4">
                  <p className="text-sm text-gray-600">Mean Anisotropy</p>
                  <p className="text-3xl font-bold text-gray-900">{cs.directional_analysis.anisotropy_mean.toFixed(3)}</p>
                  <p className="text-xs text-gray-500 mt-1">Directional preference</p>
                </div>
                <div className="border-l-4 border-orange-500 pl-4">
                  <p className="text-sm text-gray-600">Anisotropy Std Dev</p>
                  <p className="text-3xl font-bold text-gray-900">{cs.directional_analysis.anisotropy_std.toFixed(3)}</p>
                  <p className="text-xs text-gray-500 mt-1">Spatial variation</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'reconstruction-quality' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Reconstruction Quality Assessment</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Wavelet Reconstruction</h3>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-gray-600">PSNR:</span>
                    <span className="text-2xl font-bold">{cs.reconstruction_quality.wavelet.psnr_db.toFixed(2)} dB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">MSE:</span>
                    <span className="font-medium">{cs.reconstruction_quality.wavelet.mse.toExponential(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Quality:</span>
                    <span className={`font-bold ${
                      cs.reconstruction_quality.wavelet.quality_rating === 'Excellent' ? 'text-green-600' :
                      cs.reconstruction_quality.wavelet.quality_rating === 'Good' ? 'text-blue-600' : 'text-yellow-600'
                    }`}>{cs.reconstruction_quality.wavelet.quality_rating}</span>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Curvelet Reconstruction</h3>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-gray-600">PSNR:</span>
                    <span className="text-2xl font-bold">{cs.reconstruction_quality.curvelet.psnr_db.toFixed(2)} dB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">MSE:</span>
                    <span className="font-medium">{cs.reconstruction_quality.curvelet.mse.toExponential(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Quality:</span>
                    <span className={`font-bold ${
                      cs.reconstruction_quality.curvelet.quality_rating === 'Excellent' ? 'text-green-600' :
                      cs.reconstruction_quality.curvelet.quality_rating === 'Good' ? 'text-blue-600' : 'text-yellow-600'
                    }`}>{cs.reconstruction_quality.curvelet.quality_rating}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'download' && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-lg p-8 border border-gray-200 text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Download Statistics Report</h2>
              <p className="text-gray-600 mb-6">Download all coefficient statistics as a text file</p>
              <button
                onClick={downloadPDF}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                Download Report (.txt)
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
