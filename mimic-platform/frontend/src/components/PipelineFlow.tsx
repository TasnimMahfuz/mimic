import React, { useState, useEffect } from 'react';

interface PipelineFlowProps {
  runId: string | null;
}

export const PipelineFlow: React.FC<PipelineFlowProps> = ({ runId }) => {
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    if (!runId) return;

    // Cycle through stages every 2 seconds
    const interval = setInterval(() => {
      setActiveStage((prev) => (prev + 1) % 7); // 7 stages total (0-6)
    }, 2000);

    return () => clearInterval(interval);
  }, [runId]);

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
              d="M13 10V3L4 14h7v7l9-11h-7z"
            />
          </svg>
          <p className="mt-2 text-sm">Upload an image to see the processing pipeline</p>
        </div>
      </div>
    );
  }

  const stages = [
    { id: 0, label: 'Image Loading', color: 'blue' },
    { id: 1, label: 'Normalization', color: 'green' },
    { id: 2, label: 'Wavelet Transform', color: 'purple' },
    { id: 3, label: 'Curvelet Transform', color: 'orange' },
    { id: 4, label: 'Edge Detection', color: 'red' },
    { id: 5, label: 'Enhancement', color: 'pink' },
    { id: 6, label: 'Scientific Metrics', color: 'teal' },
  ];

  const getColorClasses = (color: string, active: boolean) => {
    const colors: Record<string, { bg: string; border: string; text: string; activeBg: string }> = {
      blue: { bg: 'bg-blue-50', border: 'border-blue-500', text: 'text-blue-700', activeBg: 'bg-blue-500' },
      green: { bg: 'bg-green-50', border: 'border-green-500', text: 'text-green-700', activeBg: 'bg-green-500' },
      purple: { bg: 'bg-purple-50', border: 'border-purple-500', text: 'text-purple-700', activeBg: 'bg-purple-500' },
      orange: { bg: 'bg-orange-50', border: 'border-orange-500', text: 'text-orange-700', activeBg: 'bg-orange-500' },
      red: { bg: 'bg-red-50', border: 'border-red-500', text: 'text-red-700', activeBg: 'bg-red-500' },
      pink: { bg: 'bg-pink-50', border: 'border-pink-500', text: 'text-pink-700', activeBg: 'bg-pink-500' },
      teal: { bg: 'bg-teal-50', border: 'border-teal-500', text: 'text-teal-700', activeBg: 'bg-teal-500' },
    };
    
    const c = colors[color] || colors.blue;
    return active ? `${c.activeBg} text-white border-4` : `${c.bg} ${c.text} border-2 ${c.border}`;
  };

  return (
    <div className="h-full overflow-y-auto p-8 bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold text-center mb-8 text-gray-800">
          MIMIC Analysis Pipeline
        </h2>

        {/* Branching Flow Diagram */}
        <div className="relative">
          
          {/* Stage 1: Image Loading */}
          <div className={`transition-all duration-500 ${activeStage === 0 ? 'scale-110 opacity-100' : 'scale-100 opacity-40'}`}>
            <div className={`${getColorClasses('blue', activeStage === 0)} rounded-lg p-6 mb-4 shadow-lg transform transition-all duration-500`}>
              <div className="text-center">
                <h3 className="text-xl font-bold">Stage 1: Image Loading</h3>
                <p className="text-sm mt-1">Load astronomical image data</p>
              </div>
            </div>
          </div>

          {/* Arrow */}
          <div className="flex justify-center mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 3a1 1 0 011 1v10.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V4a1 1 0 011-1z" clipRule="evenodd" />
            </svg>
          </div>

          {/* Stage 2: Normalization */}
          <div className={`transition-all duration-500 ${activeStage === 1 ? 'scale-110 opacity-100' : 'scale-100 opacity-40'}`}>
            <div className={`${getColorClasses('green', activeStage === 1)} rounded-lg p-6 mb-4 shadow-lg transform transition-all duration-500`}>
              <div className="text-center">
                <h3 className="text-xl font-bold">Stage 2: Normalization & Filtering</h3>
                <p className="text-sm mt-1">Normalize flux (0-1) + noise filter</p>
              </div>
            </div>
          </div>

          {/* Branching Arrow */}
          <div className="flex justify-center mb-4">
            <div className="flex items-center space-x-4">
              <svg className="w-8 h-8 text-gray-400 transform -rotate-45" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 3a1 1 0 011 1v10.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V4a1 1 0 011-1z" clipRule="evenodd" />
              </svg>
              <span className="text-gray-500 font-semibold">PARALLEL PROCESSING</span>
              <svg className="w-8 h-8 text-gray-400 transform rotate-45" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 3a1 1 0 011 1v10.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V4a1 1 0 011-1z" clipRule="evenodd" />
              </svg>
            </div>
          </div>

          {/* Parallel Branches */}
          <div className="grid grid-cols-2 gap-8 mb-4">
            
            {/* Left Branch: Wavelet */}
            <div className={`transition-all duration-500 ${activeStage === 2 ? 'scale-110 opacity-100' : 'scale-100 opacity-40'}`}>
              <div className={`${getColorClasses('purple', activeStage === 2)} rounded-lg p-6 shadow-lg transform transition-all duration-500`}>
                <div className="text-center">
                  <h3 className="text-lg font-bold">Wavelet Transform</h3>
                  <p className="text-xs mt-1">3 levels decomposition</p>
                  <p className="text-xs">Baseline method</p>
                </div>
              </div>
            </div>

            {/* Right Branch: Curvelet */}
            <div className={`transition-all duration-500 ${activeStage === 3 ? 'scale-110 opacity-100' : 'scale-100 opacity-40'}`}>
              <div className={`${getColorClasses('orange', activeStage === 3)} rounded-lg p-6 shadow-lg transform transition-all duration-500`}>
                <div className="text-center">
                  <h3 className="text-lg font-bold">Curvelet Transform</h3>
                  <p className="text-xs mt-1">3 scales, 16 orientations</p>
                  <p className="text-xs">Primary method</p>
                </div>
              </div>
            </div>
          </div>

          {/* Merge Arrow */}
          <div className="flex justify-center mb-4">
            <div className="flex items-center space-x-4">
              <svg className="w-8 h-8 text-gray-400 transform rotate-45" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 3a1 1 0 011 1v10.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V4a1 1 0 011-1z" clipRule="evenodd" />
              </svg>
              <span className="text-gray-500 font-semibold">MERGE</span>
              <svg className="w-8 h-8 text-gray-400 transform -rotate-45" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 3a1 1 0 011 1v10.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V4a1 1 0 011-1z" clipRule="evenodd" />
              </svg>
            </div>
          </div>

          {/* Stage 4: Edge Detection */}
          <div className={`transition-all duration-500 ${activeStage === 4 ? 'scale-110 opacity-100' : 'scale-100 opacity-40'}`}>
            <div className={`${getColorClasses('red', activeStage === 4)} rounded-lg p-6 mb-4 shadow-lg transform transition-all duration-500`}>
              <div className="text-center">
                <h3 className="text-xl font-bold">Stage 4: Edge Detection</h3>
                <p className="text-sm mt-1">Multi-scale edge extraction + confidence map</p>
              </div>
            </div>
          </div>

          {/* Arrow */}
          <div className="flex justify-center mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 3a1 1 0 011 1v10.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V4a1 1 0 011-1z" clipRule="evenodd" />
            </svg>
          </div>

          {/* Stage 5: Enhancement */}
          <div className={`transition-all duration-500 ${activeStage === 5 ? 'scale-110 opacity-100' : 'scale-100 opacity-40'}`}>
            <div className={`${getColorClasses('pink', activeStage === 5)} rounded-lg p-6 mb-4 shadow-lg transform transition-all duration-500`}>
              <div className="text-center">
                <h3 className="text-xl font-bold">Stage 5: Enhancement</h3>
                <p className="text-sm mt-1">Contrast enhancement + spatial smoothing</p>
              </div>
            </div>
          </div>

          {/* Arrow */}
          <div className="flex justify-center mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 3a1 1 0 011 1v10.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V4a1 1 0 011-1z" clipRule="evenodd" />
            </svg>
          </div>

          {/* Stage 6: Scientific Metrics */}
          <div className={`transition-all duration-500 ${activeStage === 6 ? 'scale-110 opacity-100' : 'scale-100 opacity-40'}`}>
            <div className={`${getColorClasses('teal', activeStage === 6)} rounded-lg p-6 shadow-lg transform transition-all duration-500`}>
              <div className="text-center">
                <h3 className="text-xl font-bold">Stage 6: Scientific Metrics</h3>
                <p className="text-sm mt-1">Anisotropy • Directional Energy • Radial Profile • Angular Distribution</p>
              </div>
            </div>
          </div>

        </div>

        {/* Stage Indicator */}
        <div className="mt-8 text-center">
          <div className="inline-flex space-x-2">
            {stages.map((stage) => (
              <div
                key={stage.id}
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  activeStage === stage.id ? 'bg-blue-600 scale-150' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
          <p className="mt-4 text-sm text-gray-600">
            Currently showing: <span className="font-bold">{stages[activeStage].label}</span>
          </p>
        </div>
      </div>
    </div>
  );
};
