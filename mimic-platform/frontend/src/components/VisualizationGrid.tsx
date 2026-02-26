import React, { useState } from 'react';
import { CoefficientStats } from './CoefficientStats';
import { PipelineFlow } from './PipelineFlow';

interface VisualizationGridProps {
  runId: string | null;
  baseUrl: string;
}

interface TabConfig {
  id: string;
  label: string;
  visualizations: Array<{
    filename: string;
    title: string;
  }>;
}

const tabs: TabConfig[] = [
  {
    id: 'transform-comparison',
    label: 'MIMIC Pipeline Overview',
    visualizations: [], // Special tab - uses PipelineFlow component instead
  },
  {
    id: 'directional-analysis',
    label: 'Directional Analysis',
    visualizations: [
      { filename: 'directional_energy.png', title: 'Directional Energy' },
      { filename: 'angular_distribution.png', title: 'Angular Distribution' },
    ],
  },
  {
    id: 'image-properties',
    label: 'Image Properties',
    visualizations: [],
  },
  {
    id: 'transform-info',
    label: 'Transform Info',
    visualizations: [],
  },
  {
    id: 'edge-detection',
    label: 'Edge Detection',
    visualizations: [],
  },
  {
    id: 'energy',
    label: 'Energy',
    visualizations: [],
  },
  {
    id: 'directional',
    label: 'Directional',
    visualizations: [],
  },
  {
    id: 'download',
    label: 'Download',
    visualizations: [],
  },
];

export const VisualizationGrid: React.FC<VisualizationGridProps> = ({ runId, baseUrl }) => {
  const [activeTab, setActiveTab] = useState<string>('transform-comparison');
  const [imageErrors, setImageErrors] = useState<Set<string>>(new Set());
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Clear errors when runId changes
  React.useEffect(() => {
    setImageErrors(new Set());
  }, [runId]);

  // Fetch coefficient stats from separate file
  React.useEffect(() => {
    const fetchStats = async () => {
      if (!runId) return;
      try {
        setLoading(true);
        const response = await fetch(`${baseUrl}/outputs/run_${runId}/coefficient_stats.json`);
        if (response.ok) {
          const data = await response.json();
          console.log('Coefficient stats loaded:', data);
          setStats(data);
        } else {
          setStats(null);
        }
      } catch (err) {
        console.error('Failed to load coefficient stats:', err);
        setStats(null);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, [runId, baseUrl]);

  const handleImageError = (filename: string) => {
    setImageErrors((prev) => new Set(prev).add(filename));
  };

  const handleImageLoad = (filename: string) => {
    // Image loaded successfully - could add logic here if needed
  };

  const handleImageDownload = async (imageUrl: string, filename: string) => {
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download image:', error);
    }
  };

  const activeTabConfig = tabs.find((tab) => tab.id === activeTab);

  if (!runId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <p className="mt-2 text-sm">Upload an image and run analysis to see visualizations</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Tab Navigation */}
      <div className="border-b border-gray-200 bg-white">
        <nav className="flex space-x-4 px-6 py-3" aria-label="Tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                px-3 py-2 text-sm font-medium rounded-md transition-colors
                ${
                  activeTab === tab.id
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }
              `}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Visualization Grid */}
      <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
        {activeTab === 'transform-comparison' ? (
          <PipelineFlow runId={runId} />
        ) : activeTab === 'image-properties' ? (
          <div className="max-w-4xl mx-auto">
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              </div>
            ) : stats?.image_properties ? (
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Image Properties</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="border-l-4 border-blue-500 pl-4">
                    <p className="text-sm text-gray-600">Dimensions</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {stats.image_properties.width} × {stats.image_properties.height}
                    </p>
                  </div>
                  <div className="border-l-4 border-purple-500 pl-4">
                    <p className="text-sm text-gray-600">Total Pixels</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {stats.image_properties.total_pixels.toLocaleString()}
                    </p>
                  </div>
                  <div className="border-l-4 border-green-500 pl-4">
                    <p className="text-sm text-gray-600">Intensity Range</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {stats.image_properties.intensity_range.toFixed(1)}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <p>No statistics available. Run a new analysis to generate statistics.</p>
              </div>
            )}
          </div>
        ) : activeTab === 'transform-info' ? (
          <div className="max-w-4xl mx-auto">
            {stats?.transform_info ? (
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Transform Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="border border-gray-200 rounded-lg p-4 bg-gradient-to-br from-blue-50 to-white">
                    <h4 className="font-semibold text-gray-900 mb-3">Wavelet Transform</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Type:</span>
                        <span className="font-medium text-gray-900">{stats.transform_info.wavelet.type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Levels:</span>
                        <span className="font-medium text-gray-900">{stats.transform_info.wavelet.levels}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Subbands:</span>
                        <span className="font-medium text-gray-900">{stats.transform_info.wavelet.total_subbands}</span>
                      </div>
                    </div>
                  </div>
                  <div className="border border-gray-200 rounded-lg p-4 bg-gradient-to-br from-purple-50 to-white">
                    <h4 className="font-semibold text-gray-900 mb-3">Curvelet Transform</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Scales:</span>
                        <span className="font-medium text-gray-900">{stats.transform_info.curvelet.scales}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Orientations:</span>
                        <span className="font-medium text-gray-900">{stats.transform_info.curvelet.orientations}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Subbands:</span>
                        <span className="font-medium text-gray-900">{stats.transform_info.curvelet.total_subbands}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <p>No statistics available.</p>
              </div>
            )}
          </div>
        ) : activeTab === 'edge-detection' ? (
          <div className="max-w-4xl mx-auto">
            {stats?.edge_detection ? (
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Edge Detection</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-3">Wavelet</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Edge Pixels:</span>
                        <span className="text-lg font-bold text-gray-900">{stats.edge_detection.wavelet_edges.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Density:</span>
                        <span className="text-lg font-bold text-blue-600">{stats.edge_detection.wavelet_edge_density_percent.toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-3">Curvelet</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Edge Pixels:</span>
                        <span className="text-lg font-bold text-gray-900">{stats.edge_detection.curvelet_edges.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Density:</span>
                        <span className="text-lg font-bold text-purple-600">{stats.edge_detection.curvelet_edge_density_percent.toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <p>No statistics available.</p>
              </div>
            )}
          </div>
        ) : activeTab === 'energy' ? (
          <div className="max-w-4xl mx-auto">
            {stats?.energy_distribution ? (
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Energy Distribution</h3>
                <div className="mb-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total Energy:</span>
                    <span className="text-lg font-bold text-gray-900">{stats.energy_distribution.total_energy.toExponential(2)}</span>
                  </div>
                </div>
                <div className="space-y-3">
                  {Object.entries(stats.energy_distribution.scale_energy_percentages).map(([scale, percentage]: [string, any]) => (
                    <div key={scale}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-700">Scale {scale}</span>
                        <span className="font-semibold text-gray-900">{percentage.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-green-600 h-2 rounded-full" style={{ width: `${percentage}%` }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <p>No statistics available.</p>
              </div>
            )}
          </div>
        ) : activeTab === 'directional' ? (
          <div className="max-w-4xl mx-auto">
            {stats?.directional_analysis ? (
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Directional Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="border-l-4 border-indigo-500 pl-4">
                    <p className="text-sm text-gray-600">Dominant Direction</p>
                    <p className="text-2xl font-bold text-gray-900">{stats.directional_analysis.dominant_angle_degrees.toFixed(1)}°</p>
                  </div>
                  <div className="border-l-4 border-pink-500 pl-4">
                    <p className="text-sm text-gray-600">Mean Anisotropy</p>
                    <p className="text-2xl font-bold text-gray-900">{stats.directional_analysis.anisotropy_mean.toFixed(3)}</p>
                  </div>
                  <div className="border-l-4 border-orange-500 pl-4">
                    <p className="text-sm text-gray-600">Anisotropy Std</p>
                    <p className="text-2xl font-bold text-gray-900">{stats.directional_analysis.anisotropy_std.toFixed(3)}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <p>No statistics available.</p>
              </div>
            )}
          </div>
        ) : activeTab === 'download' ? (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Download Statistics Report</h3>
              <p className="text-gray-600 mb-6">Download all coefficient statistics as a formatted text report.</p>
              <button
                onClick={() => {
                  if (!stats) return;
                  
                  const content = `COEFFICIENT STATISTICS REPORT
Run ID: ${runId}
Generated: ${new Date().toLocaleString()}

================================================================================
IMAGE PROPERTIES
================================================================================
Dimensions:        ${stats.image_properties?.width} × ${stats.image_properties?.height}
Total Pixels:      ${stats.image_properties?.total_pixels.toLocaleString()}
Intensity Min:     ${stats.image_properties?.intensity_min.toFixed(2)}
Intensity Max:     ${stats.image_properties?.intensity_max.toFixed(2)}
Intensity Range:   ${stats.image_properties?.intensity_range.toFixed(2)}

================================================================================
TRANSFORM INFORMATION
================================================================================
Wavelet Transform:
  Type:            ${stats.transform_info?.wavelet.type}
  Levels:          ${stats.transform_info?.wavelet.levels}
  Total Subbands:  ${stats.transform_info?.wavelet.total_subbands}

Curvelet Transform:
  Type:            ${stats.transform_info?.curvelet.type}
  Scales:          ${stats.transform_info?.curvelet.scales}
  Orientations:    ${stats.transform_info?.curvelet.orientations}
  Total Subbands:  ${stats.transform_info?.curvelet.total_subbands}

================================================================================
EDGE DETECTION
================================================================================
Wavelet Edges:
  Edge Pixels:     ${stats.edge_detection?.wavelet_edges.toLocaleString()}
  Density:         ${stats.edge_detection?.wavelet_edge_density_percent.toFixed(2)}%

Curvelet Edges:
  Edge Pixels:     ${stats.edge_detection?.curvelet_edges.toLocaleString()}
  Density:         ${stats.edge_detection?.curvelet_edge_density_percent.toFixed(2)}%

================================================================================
ENERGY DISTRIBUTION
================================================================================
Total Energy:      ${stats.energy_distribution?.total_energy.toExponential(2)}

Energy per Scale:
${Object.entries(stats.energy_distribution?.scale_energy_percentages || {}).map(([scale, pct]: [string, any]) => 
  `  Scale ${scale}:        ${pct.toFixed(2)}%`
).join('\n')}

================================================================================
DIRECTIONAL ANALYSIS
================================================================================
Dominant Direction:  ${stats.directional_analysis?.dominant_angle_degrees.toFixed(1)}°
Angular Resolution:  ${stats.directional_analysis?.angular_resolution}
Mean Anisotropy:     ${stats.directional_analysis?.anisotropy_mean.toFixed(3)}
Anisotropy Std Dev:  ${stats.directional_analysis?.anisotropy_std.toFixed(3)}

================================================================================
END OF REPORT
================================================================================
`;
                  
                  const blob = new Blob([content], { type: 'text/plain' });
                  const url = window.URL.createObjectURL(blob);
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = `coefficient_stats_${runId}.txt`;
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);
                  window.URL.revokeObjectURL(url);
                }}
                disabled={!stats}
                className={`px-6 py-3 rounded-lg transition-colors ${
                  stats 
                    ? 'bg-blue-600 text-white hover:bg-blue-700' 
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                Download Report (.txt)
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {activeTabConfig?.visualizations.map((viz) => {
            const imageUrl = `${baseUrl}/outputs/run_${runId}/${viz.filename}?t=${Date.now()}`;
            const hasError = imageErrors.has(viz.filename);

            return (
              <div
                key={viz.filename}
                className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden"
              >
                <div className="p-4 border-b border-gray-200 flex justify-between items-center">
                  <h3 className="text-sm font-semibold text-gray-900">{viz.title}</h3>
                  {!hasError && (
                    <button
                      onClick={() => handleImageDownload(imageUrl, viz.filename)}
                      className="text-blue-600 hover:text-blue-800 transition-colors"
                      title="Download image"
                    >
                      <svg
                        className="h-5 w-5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                        />
                      </svg>
                    </button>
                  )}
                </div>
                <div className="p-4">
                  {hasError ? (
                    <div className="flex items-center justify-center h-64 bg-gray-100 rounded">
                      <div className="text-center text-gray-500">
                        <svg
                          className="mx-auto h-8 w-8 text-gray-400"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                        <p className="mt-2 text-xs">Image not available</p>
                        <p className="mt-1 text-xs text-gray-400">{viz.filename}</p>
                      </div>
                    </div>
                  ) : (
                    <div 
                      className="relative group cursor-pointer"
                      onClick={() => handleImageDownload(imageUrl, viz.filename)}
                      title="Click to download"
                    >
                      <img
                        src={imageUrl}
                        alt={viz.title}
                        className="w-full h-auto rounded"
                        onLoad={() => handleImageLoad(viz.filename)}
                        onError={() => handleImageError(viz.filename)}
                        style={{ 
                          minHeight: '200px', 
                          backgroundColor: '#f3f4f6',
                          // FIX #1: Auto-brighten edge images more aggressively for visibility
                          filter: viz.filename.includes('edge') 
                            ? 'brightness(3) contrast(2)'  // Much stronger for edge images
                            : activeTab === 'transform-comparison' 
                              ? 'brightness(1.5) contrast(1.2)'  // Moderate for other transform images
                              : 'none'  // Natural for plots
                        }}
                      />
                      {/* Download overlay on hover */}
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all duration-200 rounded flex items-center justify-center">
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-white rounded-full p-3 shadow-lg">
                          <svg
                            className="h-6 w-6 text-blue-600"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                            />
                          </svg>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        )}
      </div>
    </div>
  );
};
