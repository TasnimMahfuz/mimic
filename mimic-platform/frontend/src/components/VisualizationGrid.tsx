import React, { useState } from 'react';

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
    label: 'Transform Comparison',
    visualizations: [
      { filename: 'raw.png', title: 'Original Image' },
      { filename: 'normalized.png', title: 'Normalized' },
      { filename: 'wavelet_edge.png', title: 'Wavelet Edges' },
      { filename: 'curvelet_edge.png', title: 'Curvelet Edges' },
      { filename: 'difference_map.png', title: 'Difference Map' },
      { filename: 'reconstruction.png', title: 'Reconstruction' },
    ],
  },
  {
    id: 'directional-analysis',
    label: 'Directional Analysis',
    visualizations: [
      { filename: 'orientation_map.png', title: 'Orientation Map' },
      { filename: 'directional_energy.png', title: 'Directional Energy' },
      { filename: 'angular_distribution.png', title: 'Angular Distribution' },
      { filename: 'frequency_cone.png', title: 'Frequency Cone' },
    ],
  },
  {
    id: 'spectral-analysis',
    label: 'Spectral Analysis',
    visualizations: [
      { filename: 'radial_energy.png', title: 'Radial Energy' },
      { filename: 'scale_energy.png', title: 'Scale Energy' },
      { filename: 'coefficient_histogram.png', title: 'Coefficient Histogram' },
      { filename: 'wavelet_coefficients.png', title: 'Wavelet Coefficients' },
    ],
  },
  {
    id: 'edge-results',
    label: 'Edge Results',
    visualizations: [
      { filename: 'edge_overlay.png', title: 'Edge Overlay' },
      { filename: 'enhancement_overlay.png', title: 'Enhancement Overlay' },
      { filename: 'reconstruction_error.png', title: 'Reconstruction Error' },
      { filename: 'reconstruction_error_curve.png', title: 'Error Curve' },
    ],
  },
];

export const VisualizationGrid: React.FC<VisualizationGridProps> = ({ runId, baseUrl }) => {
  const [activeTab, setActiveTab] = useState<string>('transform-comparison');
  const [imageErrors, setImageErrors] = useState<Set<string>>(new Set());

  const handleImageError = (filename: string) => {
    setImageErrors((prev) => new Set(prev).add(filename));
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {activeTabConfig?.visualizations.map((viz) => {
            const imageUrl = `${baseUrl}/outputs/run_${runId}/${viz.filename}`;
            const hasError = imageErrors.has(viz.filename);

            return (
              <div
                key={viz.filename}
                className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden"
              >
                <div className="p-4 border-b border-gray-200">
                  <h3 className="text-sm font-semibold text-gray-900">{viz.title}</h3>
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
                      </div>
                    </div>
                  ) : (
                    <img
                      src={imageUrl}
                      alt={viz.title}
                      className="w-full h-auto rounded"
                      onError={() => handleImageError(viz.filename)}
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
