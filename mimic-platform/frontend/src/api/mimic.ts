import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface ProcessingParameters {
  edge_strength: number;
  angular_resolution: number;
  smoothing: number;
  photon_threshold: number;
  enhancement_factor: number;
}

export interface AnalysisResponse {
  run_id: string;
  status: string;
  message: string;
  visualization_urls?: string[];
}

export interface ResultsResponse {
  run_id: string;
  status: 'processing' | 'complete' | 'failed';
  parameters: ProcessingParameters;
  visualizations: Record<string, string>;
  metrics?: Record<string, number>;
  timestamp: string;
}

export class MIMICApi {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async runAnalysis(
    file: File,
    parameters: ProcessingParameters
  ): Promise<AnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('edge_strength', parameters.edge_strength.toString());
    formData.append('angular_resolution', parameters.angular_resolution.toString());
    formData.append('smoothing', parameters.smoothing.toString());
    formData.append('photon_threshold', parameters.photon_threshold.toString());
    formData.append('enhancement_factor', parameters.enhancement_factor.toString());

    const response = await axios.post<AnalysisResponse>(
      `${this.baseURL}/mimic/run`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  async getResults(runId: string): Promise<ResultsResponse> {
    const response = await axios.get<ResultsResponse>(
      `${this.baseURL}/mimic/run/${runId}/results`
    );
    return response.data;
  }

  getVisualizationUrl(runId: string, filename: string): string {
    return `${this.baseURL}/outputs/run_${runId}/${filename}`;
  }
}

export const mimicApi = new MIMICApi();
