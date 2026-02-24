import React from 'react';
import type { ProcessingParameters } from '../api/mimic';

interface ParameterPanelProps {
  parameters: ProcessingParameters;
  onChange: (param: keyof ProcessingParameters, value: number) => void;
  disabled?: boolean;
}

interface SliderConfig {
  key: keyof ProcessingParameters;
  label: string;
  min: number;
  max: number;
  step: number;
  description: string;
}

const sliderConfigs: SliderConfig[] = [
  {
    key: 'edge_strength',
    label: 'Edge Strength',
    min: 0.0,
    max: 1.0,
    step: 0.05,
    description: 'Threshold for edge detection sensitivity',
  },
  {
    key: 'angular_resolution',
    label: 'Angular Resolution',
    min: 8,
    max: 32,
    step: 2,
    description: 'Number of directional bins in curvelet decomposition',
  },
  {
    key: 'smoothing',
    label: 'Smoothing',
    min: 0.0,
    max: 5.0,
    step: 0.1,
    description: 'Spatial smoothing kernel size',
  },
  {
    key: 'photon_threshold',
    label: 'Photon Threshold',
    min: 0.0,
    max: 100.0,
    step: 1.0,
    description: 'Photon count threshold for noise filtering',
  },
  {
    key: 'enhancement_factor',
    label: 'Enhancement Factor',
    min: 1.0,
    max: 5.0,
    step: 0.1,
    description: 'Contrast enhancement multiplier',
  },
];

export const ParameterPanel: React.FC<ParameterPanelProps> = ({
  parameters,
  onChange,
  disabled = false,
}) => {
  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">Processing Parameters</h3>
      {sliderConfigs.map((config) => (
        <div key={config.key} className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium text-gray-700">
              {config.label}
            </label>
            <span className="text-sm font-mono text-gray-600 bg-gray-100 px-2 py-1 rounded">
              {parameters[config.key].toFixed(config.step < 1 ? 2 : 0)}
            </span>
          </div>
          <input
            type="range"
            min={config.min}
            max={config.max}
            step={config.step}
            value={parameters[config.key]}
            onChange={(e) => onChange(config.key, parseFloat(e.target.value))}
            disabled={disabled}
            className={`
              w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer
              disabled:opacity-50 disabled:cursor-not-allowed
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:w-4
              [&::-webkit-slider-thumb]:h-4
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-blue-600
              [&::-webkit-slider-thumb]:cursor-pointer
              [&::-moz-range-thumb]:w-4
              [&::-moz-range-thumb]:h-4
              [&::-moz-range-thumb]:rounded-full
              [&::-moz-range-thumb]:bg-blue-600
              [&::-moz-range-thumb]:border-0
              [&::-moz-range-thumb]:cursor-pointer
            `}
          />
          <p className="text-xs text-gray-500">{config.description}</p>
        </div>
      ))}
    </div>
  );
};
