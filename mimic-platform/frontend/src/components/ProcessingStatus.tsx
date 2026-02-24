import React from 'react';

export type ProcessingStage =
  | 'idle'
  | 'uploading'
  | 'curvelet_decomposition'
  | 'directional_extraction'
  | 'spectral_analysis'
  | 'wavelet_processing'
  | 'edge_detection'
  | 'enhancement'
  | 'visualization'
  | 'complete'
  | 'error';

interface ProcessingStatusProps {
  stage: ProcessingStage;
  error?: string;
}

const statusMessages: Record<ProcessingStage, string> = {
  idle: 'Ready to process',
  uploading: 'Uploading file...',
  curvelet_decomposition: 'Performing Curvelet Decomposition...',
  directional_extraction: 'Extracting Curvilinear Singularities...',
  spectral_analysis: 'Computing Directional Energy Spectrum...',
  wavelet_processing: 'Performing Wavelet Transform...',
  edge_detection: 'Detecting Edges and Boundaries...',
  enhancement: 'Enhancing Feature Visibility...',
  visualization: 'Generating Visualizations...',
  complete: 'Analysis Complete',
  error: 'Processing Failed',
};

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ stage, error }) => {
  const isProcessing = !['idle', 'complete', 'error'].includes(stage);
  const isComplete = stage === 'complete';
  const isError = stage === 'error';

  return (
    <div className="w-full">
      <div
        className={`
          rounded-lg p-4 border-2
          ${isError ? 'bg-red-50 border-red-200' : ''}
          ${isComplete ? 'bg-green-50 border-green-200' : ''}
          ${isProcessing ? 'bg-blue-50 border-blue-200' : ''}
          ${stage === 'idle' ? 'bg-gray-50 border-gray-200' : ''}
        `}
      >
        <div className="flex items-center space-x-3">
          {isProcessing && (
            <div className="flex-shrink-0">
              <svg
                className="animate-spin h-5 w-5 text-blue-600"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            </div>
          )}
          {isComplete && (
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-green-600"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
          )}
          {isError && (
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-red-600"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
          )}
          <div className="flex-1">
            <p
              className={`
                text-sm font-medium
                ${isError ? 'text-red-800' : ''}
                ${isComplete ? 'text-green-800' : ''}
                ${isProcessing ? 'text-blue-800' : ''}
                ${stage === 'idle' ? 'text-gray-600' : ''}
              `}
            >
              {statusMessages[stage]}
            </p>
            {error && (
              <p className="text-xs text-red-600 mt-1">{error}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
