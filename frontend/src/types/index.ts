// ── Auth ───────────────────────────────────────────────────────────────────────
export interface User {
  id: string;
  email: string;
  username: string;
  full_name: string | null;
  role: 'user' | 'admin';
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  full_name?: string;
}

// ── Analysis / Detection ───────────────────────────────────────────────────────
export type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed';
export type Prediction = 'REAL' | 'FAKE';

export interface AnalysisSummary {
  id: string;
  status: AnalysisStatus;
  error_message?: string | null;
  prediction?: Prediction;
  real_probability?: number;
  fake_probability?: number;
  confidence?: number;
  face_count?: number;
  file_name?: string;
  file_type?: 'image' | 'video';
  created_at: string;
  processing_duration_ms?: number;
}

export interface VideoFrameResult {
  frame_index: number;
  timestamp_seconds: number;
  prediction: Prediction;
  real_probability: number;
  fake_probability: number;
  face_detected: boolean;
}

export interface DetectionResult {
  prediction: Prediction;
  real_probability: number;
  fake_probability: number;
  confidence: number;
  raw_logits?: number[];
  detector_name?: string | null;
  detector_notice?: string | null;
  face_count: number;
  heatmap_available: boolean;
  graph_data_available: boolean;
  patch_relevance?: number[];
}

export interface FullAnalysisResponse {
  analysis: AnalysisSummary;
  detection?: DetectionResult;
  video_frames?: VideoFrameResult[];
  suspicious_timestamps?: number[];
  model_version?: string;
}

export interface PaginatedHistory {
  items: AnalysisSummary[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}

// ── Model ──────────────────────────────────────────────────────────────────────
export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  embed_dim: number;
  patch_size: number;
  image_size: number;
  gcn_layers: number;
  transformer_blocks: number;
  attention_heads: number;
  knn_k: number;
  training_status: string;
  checkpoint_path?: string;
  pretrain_checkpoint_path?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface ModelStatus {
  model_loaded: boolean;
  pretrained_fallback_available: boolean;
  training_status: string;
  device: string;
  checkpoint_exists: boolean;
  pretrain_checkpoint_exists: boolean;
  active_model_id?: string;
}

// ── Training ──────────────────────────────────────────────────────────────────
export interface PretrainConfig {
  dataset_path: string;
  batch_size: number;
  epochs: number;
  learning_rate?: number;
  teacher_momentum: number;
  mask_ratio: number;
  save_every: number;
  seed: number;
  model_name: string;
}

export interface ClassifierTrainConfig {
  dataset_path: string;
  pretrain_checkpoint?: string;
  batch_size: number;
  epochs: number;
  learning_rate: number;
  freeze_backbone: boolean;
  save_every: number;
  seed: number;
  model_name: string;
}

export interface TrainingMetric {
  epoch: number;
  train_loss: number;
  val_loss?: number;
  train_cls_loss?: number;
  train_mim_loss?: number;
  learning_rate?: number;
}

export interface TrainingStatus {
  job_id: string;
  job_type: 'pretrain' | 'classifier';
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopped';
  current_epoch: number;
  total_epochs: number;
  progress_pct: number;
  best_val_loss?: number;
  started_at?: string;
  metrics: TrainingMetric[];
}

// ── Evaluation ────────────────────────────────────────────────────────────────
export interface EvaluationResult {
  id: string;
  eval_type: string;
  dataset_name: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auc_roc?: number;
  false_positive_rate?: number;
  false_negative_rate?: number;
  evaluated_at: string;
}

export interface EvaluationMetrics extends EvaluationResult {
  confusion_matrix?: number[][];
  roc_curve_data?: { fpr: number[]; tpr: number[]; thresholds: number[] };
  extra_metrics?: Record<string, unknown>;
}

// ── Explainability ────────────────────────────────────────────────────────────
export interface ExplainabilityData {
  analysis_id: string;
  prediction: Prediction;
  confidence: number;
  heatmap_base64?: string;
  patch_relevance?: number[];
  graph_nodes?: number;
  graph_edges?: number;
  attention_weights?: number[][][];
  explanation_text: string;
}

// ── Dashboard Stats ────────────────────────────────────────────────────────────
export interface DashboardStats {
  total_analyses: number;
  real_count: number;
  fake_count: number;
  pending_count: number;
  failed_count: number;
}

// ── API Error ──────────────────────────────────────────────────────────────────
export interface ApiError {
  detail: string | { msg: string; type: string }[];
}
