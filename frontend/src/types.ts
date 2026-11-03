// ─── Shared API types ─────────────────────────────────────────────────────────

export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

export const SMART_FEATURES = [
  'smart_5_raw',
  'smart_187_raw',
  'smart_188_raw',
  'smart_197_raw',
  'smart_198_raw',
] as const;

export type SmartFeature = (typeof SMART_FEATURES)[number];

export type SmartValues = Record<SmartFeature, number>;

export interface DriveTelemetry extends SmartValues {
  serial_number?: string;
  drive_model?: string;
}

export interface EnrichedPrediction {
  id: string | null;
  serial_number: string | null;
  ttf_days: number;
  rul_days: number;
  risk_level: RiskLevel;
  log_time: number;
  novelty_score: number | null;
  novelty_distance: number | null;
  is_novel: boolean | null;
  out_of_range_features: string[];
  fired_alert_ids: string[];
  drift_summary: unknown | null;
}

export interface BatchPredictionResponse {
  count: number;
  risk_counts: Record<RiskLevel, number>;
  novel_count: number;
  predictions: EnrichedPrediction[];
}

export interface Alert {
  id: string;
  alert_type: 'LATENT_NOVELTY' | 'FEATURE_DRIFT';
  severity: 'WARNING' | 'CRITICAL';
  status: 'ACTIVE' | 'RESOLVED';
  message: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string | null;
  resolved_at: string | null;
  escalation_count: number;
  occurrences: number;
}

export interface FeatureDriftScore {
  feature: string;
  ks_statistic: number;
  ks_p_value: number;
  psi: number;
  severity: 'stable' | 'minor' | 'major';
  is_drifting: boolean;
}

export interface HealthData {
  status: 'healthy' | 'degraded' | 'unavailable';
  model_loaded: boolean;
  fingerprint_id: string | null;
  total_inferences: number;
  drift_window_count: number;
  faiss: { is_built: boolean; threshold: number | null };
  drift: {
    overall_drift_detected: boolean;
    drifting_features: string[];
    window_size: number;
    feature_scores: FeatureDriftScore[];
  } | null;
  alerts: {
    total_active: number;
    latent_novelty: number;
    feature_drift: number;
    critical_count: number;
  };
}

export interface Fingerprint {
  fingerprint_id: string;
  n_probe_samples: number;
  rul_distribution: { mean: number; std: number; median: number; p10: number; p90: number };
  prediction_entropy: number;
  uncertainty_rate: number;
  class_bias: Record<string, number>;
  feature_importance: Record<string, number>;
  histogram: { bins: number[]; counts: number[] };
}

export interface Topology {
  n_trees: number;
  total_nodes: number;
  total_leaves: number;
  max_depth: number;
  tree_summaries: Array<{ tree_id: number; n_nodes: number; n_leaves: number; max_depth: number }>;
}

export interface PredictionSummary {
  total_predictions: number;
  risk_counts: Record<RiskLevel, number>;
  novel_count: number;
  novelty_rate: number;
  mean_rul_days: number | null;
  tracked_drives: number;
}

export interface DriftSnapshot {
  id: string;
  created_at: string | null;
  window_size: number;
  overall_drift_detected: boolean;
  drifting_features: string[];
  feature_scores: Array<{ feature: string; psi: number; ks_statistic: number; severity: string }>;
}

export interface ApiMeta {
  version: string;
  admin_enabled: boolean;
  max_batch_size: number;
  model_sha256: string | null;
}

export const FEATURE_LABELS: Record<SmartFeature, string> = {
  smart_5_raw: 'SMART 5 (Reallocated Sectors)',
  smart_187_raw: 'SMART 187 (Uncorrectable Errors)',
  smart_188_raw: 'SMART 188 (Command Timeout)',
  smart_197_raw: 'SMART 197 (Pending Sectors)',
  smart_198_raw: 'SMART 198 (Uncorrectable Sectors)',
};

export const RISK_ORDER: Record<RiskLevel, number> = { critical: 0, high: 1, medium: 2, low: 3 };

/** Short display metadata for each SMART attribute (used by the input grid). */
export const FEATURE_META: Record<SmartFeature, { code: string; name: string; hint: string }> = {
  smart_5_raw: { code: 'SMART 5', name: 'Reallocated sectors', hint: 'bad sectors remapped to spares' },
  smart_187_raw: { code: 'SMART 187', name: 'Uncorrectable errors', hint: 'errors ECC could not recover' },
  smart_188_raw: { code: 'SMART 188', name: 'Command timeouts', hint: 'operations aborted by timeout' },
  smart_197_raw: { code: 'SMART 197', name: 'Pending sectors', hint: 'unstable, awaiting remap' },
  smart_198_raw: { code: 'SMART 198', name: 'Offline uncorrectable', hint: 'failed offline surface scan' },
};
