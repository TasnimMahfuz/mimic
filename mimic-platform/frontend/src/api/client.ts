import axios from 'axios';
import type { AxiosInstance, AxiosRequestConfig } from 'axios';

const API_BASE_URL = 'http://localhost:8000';
const TOKEN_KEY = 'mimic_token';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add token to every request
    this.client.interceptors.request.use((config) => {
      const token = this.getToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Handle response errors
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          this.clearToken();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  setToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token);
  }

  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  clearToken(): void {
    localStorage.removeItem(TOKEN_KEY);
  }

  isAuthenticated(): boolean {
    return !!this.getToken();
  }

  // Auth endpoints
  async register(email: string, password: string, role: string = 'student') {
    const response = await this.client.post('/auth/register', {
      email,
      password,
      role,
    });
    return response.data;
  }

  async login(email: string, password: string) {
    const response = await this.client.post('/auth/login', {
      email,
      password,
    });
    return response.data;
  }

  // Material endpoints
  async uploadMaterial(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post('/materials/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Chat endpoints
  async chatQuery(query: string) {
    const response = await this.client.post('/chat/query', {
      query,
    });
    return response.data;
  }

  // Generic request method for testing
  async request(method: string, path: string, data?: unknown, config?: AxiosRequestConfig) {
    const response = await this.client.request({
      method: method.toUpperCase(),
      url: path,
      data,
      ...config,
    });
    return response;
  }
}

export const api = new APIClient();
export default APIClient;
