import axios, { AxiosError, AxiosInstance, AxiosResponse } from 'axios';
import type {
  TokenResponse, LoginRequest, RegisterRequest, User,
  FullAnalysisResponse, PaginatedHistory,
  ModelInfo, ModelStatus,
  TrainingStatus, PretrainConfig, ClassifierTrainConfig,
  EvaluationResult, EvaluationMetrics,
  ExplainabilityData,
} from '@/types';

const BASE_URL = '/api';

// ── Axios Instance ─────────────────────────────────────────────────────────────
const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000,
});

// ── Request Interceptor: attach JWT ──────────────────────────────────────────
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Response Interceptor: handle 401 ─────────────────────────────────────────
api.interceptors.response.use(
  (res: AxiosResponse) => res,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ── Helper to extract error message ───────────────────────────────────────────
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data;
    if (typeof data?.detail === 'string') return data.detail;
    if (Array.isArray(data?.detail)) return data.detail.map((d: { msg: string }) => d.msg).join(', ');
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return 'An unknown error occurred';
}

// ═════════════════════════════════════════════════════════════════════════════
// AUTH SERVICE
// ═════════════════════════════════════════════════════════════════════════════
export const authService = {
  async register(data: RegisterRequest): Promise<TokenResponse> {
    const res = await api.post<TokenResponse>('/auth/register', data);
    return res.data;
  },
  async login(data: LoginRequest): Promise<TokenResponse> {
    const res = await api.post<TokenResponse>('/auth/login', data);
    return res.data;
  },
  async logout(): Promise<void> {
    await api.post('/auth/logout');
  },
  async getMe(): Promise<User> {
    const res = await api.get<User>('/auth/me');
    return res.data;
  },
};

// ═════════════════════════════════════════════════════════════════════════════
// DETECTION SERVICE
// ═════════════════════════════════════════════════════════════════════════════
export const detectionService = {
  async analyzeImage(file: File): Promise<FullAnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post<FullAnalysisResponse>('/detection/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  async analyzeVideo(file: File): Promise<FullAnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post<FullAnalysisResponse>('/detection/video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 min for video
    });
    return res.data;
  },
  async getAnalysis(analysisId: string): Promise<FullAnalysisResponse> {
    const res = await api.get<FullAnalysisResponse>(`/detection/${analysisId}`);
    return res.data;
  },
  async getHistory(page = 1, pageSize = 20): Promise<PaginatedHistory> {
    const res = await api.get<PaginatedHistory>('/detection/history/list', {
      params: { page, page_size: pageSize },
    });
    return res.data;
  },
  async deleteAnalysis(analysisId: string): Promise<void> {
    await api.delete(`/detection/${analysisId}`);
  },
};

// ═════════════════════════════════════════════════════════════════════════════
// MODEL SERVICE
// ═════════════════════════════════════════════════════════════════════════════
export const modelService = {
  async getInfo(): Promise<ModelInfo> {
    const res = await api.get<ModelInfo>('/model/info');
    return res.data;
  },
  async getStatus(): Promise<ModelStatus> {
    const res = await api.get<ModelStatus>('/model/status');
    return res.data;
  },
  async getCheckpoints(): Promise<string[]> {
    const res = await api.get<string[]>('/model/checkpoints');
    return res.data;
  },
};

// ═════════════════════════════════════════════════════════════════════════════
// TRAINING SERVICE
// ═════════════════════════════════════════════════════════════════════════════
export const trainingService = {
  async startPretrain(config: PretrainConfig): Promise<{ job_id: string; status: string; message: string }> {
    const res = await api.post('/training/pretrain', config);
    return res.data;
  },
  async startClassifier(config: ClassifierTrainConfig): Promise<{ job_id: string; status: string; message: string }> {
    const res = await api.post('/training/classifier', config);
    return res.data;
  },
  async getStatus(jobId: string): Promise<TrainingStatus> {
    const res = await api.get<TrainingStatus>(`/training/status/${jobId}`);
    return res.data;
  },
  async stop(jobId: string): Promise<void> {
    await api.post(`/training/stop/${jobId}`);
  },
  async getHistory(): Promise<TrainingStatus[]> {
    const res = await api.get<TrainingStatus[]>('/training/history');
    return res.data;
  },
};

// ═════════════════════════════════════════════════════════════════════════════
// EVALUATION SERVICE
// ═════════════════════════════════════════════════════════════════════════════
export const evaluationService = {
  async runEvaluation(payload: {
    model_version_id: string;
    eval_type: string;
    dataset_path: string;
    dataset_name: string;
  }): Promise<void> {
    await api.post('/evaluation/run', payload);
  },
  async getResults(): Promise<EvaluationResult[]> {
    const res = await api.get<EvaluationResult[]>('/evaluation/results');
    return res.data;
  },
  async getMetrics(resultId: string): Promise<EvaluationMetrics> {
    const res = await api.get<EvaluationMetrics>(`/evaluation/metrics/${resultId}`);
    return res.data;
  },
};

// ═════════════════════════════════════════════════════════════════════════════
// EXPLAINABILITY SERVICE
// ═════════════════════════════════════════════════════════════════════════════
export const explainabilityService = {
  async getExplainability(analysisId: string): Promise<ExplainabilityData> {
    const res = await api.get<ExplainabilityData>(`/explainability/${analysisId}`);
    return res.data;
  },
  async getHeatmap(analysisId: string): Promise<{ heatmap_base64: string; format: string }> {
    const res = await api.get(`/explainability/${analysisId}/heatmap`);
    return res.data;
  },
  async getGraphData(analysisId: string): Promise<{
    analysis_id: string;
    patch_relevance: number[];
    graph_data: unknown;
    num_patches: number;
    grid_size: number;
  }> {
    const res = await api.get(`/explainability/${analysisId}/graph`);
    return res.data;
  },
};

export default api;
