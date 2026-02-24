import React, { useState } from 'react';
import { FileUpload } from '../components/FileUpload';
import { ParameterPanel } from '../components/ParameterPanel';
import { ProcessingStatus } from '../components/ProcessingStatus';
import type { ProcessingStage } from '../components/ProcessingStatus';
import { VisualizationGrid } from '../components/VisualizationGrid';
import { mimicApi } from '../api/mimic';
import type { ProcessingParameters } from '../api/mimic';

const API_BASE_URL = 'http://localhost:8000';

export const MIMICAnalysis: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [parameters, setParameters] = useState<ProcessingParameters>({
    edge_strength: 0.5,
    angular_resolution: 16,
    smoothing: 1.0,
    photon_threshold: 10.0,
    enhancement_factor: 2.0,
  });
  const [runId, setRunId] = useState<string | null>(null);
  const [stage, setStage] = useState<ProcessingStage>('idle');
  const [error, setError] = useState<string | undefined>(undefined);

  const handleFileSelect = (selectedFile: File | null) => {
    setFile(selectedFile);
    setError(undefined);
  };

  const handleParameterChange = (
    param: keyof ProcessingParameters,
    value: number
  ) => {
    setParameters((prev) => ({
      ...prev,
      [param]: value,
    }));
  };

  const simulateProcessingStages = async () => {
    const stages: ProcessingStage[] = [
      'curvelet_decomposition',
      'directional_extraction',
      'spectral_analysis',
      'wavelet_processing',
      'edge_detection',
      'enhancement',
      'visualization',
    ];

    for (const currentStage of stages) {
      setStage(currentStage);
      await new Promise((resolve) => setTimeout(resolve, 800));
    }

    setStage('complete');
  };

  const handleRunAnalysis = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    try {
      setStage('uploading');
      setError(undefined);

      const response = await mimicApi.runAnalysis(file, parameters);
      
      if (response.run_id) {
        setRunId(response.run_id);
        await simulateProcessingStages();
      } else {
        throw new Error('No run ID received from server');
      }
    } catch (err) {
      setStage('error');
      setError(err instanceof Error ? err.message : 'An error occurred during processing');
      console.error('Analysis error:', err);
    }
  };

  const isProcessing = !['idle', 'complete', 'error'].includes(stage);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-full mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold text-gray-900">MIMIC Analysis Lab</h1>
          <p className="text-sm text-gray-600 mt-1">
            Multi-scale Image analysis with Multidirectional Intelligent Curvelet processing
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex h-[calc(100vh-88px)]">
        {/* Left Panel - Controls */}
        <div className="w-96 bg-white border-r border-gray-200 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* File Upload */}
            <FileUpload onFileSelect={handleFileSelect} disabled={isProcessing} />

            {/* Parameters */}
            <ParameterPanel
              parameters={parameters}
              onChange={handleParameterChange}
              disabled={isProcessing}
            />

            {/* Run Button */}
            <button
              onClick={handleRunAnalysis}
              disabled={!file || isProcessing}
              className={`
                w-full py-3 px-4 rounded-lg font-medium text-white
                transition-colors duration-200
                ${
                  !file || isProcessing
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
                }
              `}
            >
              {isProcessing ? 'Processing...' : 'Run Analysis'}
            </button>

            {/* Status */}
            <ProcessingStatus stage={stage} error={error} />
          </div>
        </div>

        {/* Right Panel - Visualizations */}
        <div className="flex-1 overflow-hidden">
          <VisualizationGrid runId={runId} baseUrl={API_BASE_URL} />
        </div>
      </div>
    </div>
  );
};
