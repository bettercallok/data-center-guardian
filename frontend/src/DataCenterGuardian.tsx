import { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Line, ReferenceLine, Legend, AreaChart, Area
} from 'recharts';

// ─── Types ────────────────────────────────────────────────────────────────────

interface DriveTelemetry {
  smart_5_raw: number;
  smart_187_raw: number;
  smart_188_raw: number;
  smart_197_raw: number;
  smart_198_raw: number;
}

interface EnrichedPrediction {
  ttf_days: number;
  rul_days: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  log_time: number;
  novelty_score: number | null;
  novelty_distance: number | null;
  is_novel: boolean | null;
  fired_alert_ids: string[];
  drift_summary: unknown | null;
}

interface Alert {
  id: string;
  alert_type: 'LATENT_NOVELTY' | 'FEATURE_DRIFT';
  severity: 'WARNING' | 'CRITICAL';
  status: 'ACTIVE' | 'RESOLVED';
  message: string;
  metadata: Record<string, unknown>;
  created_at: string;
  escalation_count: number;
}

interface FeatureDriftScore {
  feature: string;
  ks_statistic: number;
  ks_p_value: number;
  psi: number;
  severity: 'stable' | 'minor' | 'major';
  is_drifting: boolean;
}

interface HealthData {
  status: string;
  total_inferences: number;
  faiss: { is_built: boolean; threshold: number };
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

interface Fingerprint {
  fingerprint_id: string;
  n_probe_samples: number;
  rul_distribution: { mean: number; std: number; median: number; p10: number; p90: number };
  prediction_entropy: number;
  uncertainty_rate: number;
  class_bias: Record<string, number>;
  feature_importance: Record<string, number>;
  histogram: { bins: number[]; counts: number[] };
}



// ─── Constants ────────────────────────────────────────────────────────────────

// Empty string = relative URLs (same origin). Works for any port in Docker.
// Set VITE_API_URL at build time to point to a remote backend if needed.
const API_URL = import.meta.env.VITE_API_URL || '';
const FEATURE_LABELS: Record<string, string> = {
  smart_5_raw: 'SMART 5 (Reallocated Sectors)',
  smart_187_raw: 'SMART 187 (Uncorrectable Errors)',
  smart_188_raw: 'SMART 188 (Command Timeout)',
  smart_197_raw: 'SMART 197 (Pending Sectors)',
  smart_198_raw: 'SMART 198 (Uncorrectable Sectors)',
};

// ─── Sub-components ────────────────────────────────────────────────────────────

const RiskBadge = ({ level }: { level: string }) => {
  const colors: Record<string, string> = {
    low: 'var(--risk-low)',
    medium: 'var(--risk-medium)',
    high: 'var(--risk-high)',
    critical: 'var(--risk-critical)',
  };
  return (
    <span className={`badge badge-${level}`} style={{ background: colors[level] }}>
      {level}
    </span>
  );
};

const NoveltyIndicator = ({ score, isNovel }: { score: number | null; isNovel: boolean | null }) => {
  if (score === null) return <span className="novelty-na">—</span>;
  const pct = Math.min(score * 100, 150);
  return (
    <div className="novelty-bar-wrap">
      <div className="novelty-bar-track">
        <div
          className="novelty-bar-fill"
          style={{ width: `${Math.min(pct, 100)}%`, background: isNovel ? 'var(--risk-critical)' : 'var(--accent-glow)' }}
        />
        <div className="novelty-threshold-line" />
      </div>
      <span className={`novelty-label ${isNovel ? 'novel' : 'normal'}`}>
        {isNovel ? '⚠ NOVEL' : '✓ normal'} ({score.toFixed(2)})
      </span>
    </div>
  );
};

const StatCard = ({ label, value, sub }: { label: string; value: string | number; sub?: string }) => (
  <div className="stat-card">
    <span className="stat-label">{label}</span>
    <span className="stat-value">{value}</span>
    {sub && <span className="stat-sub">{sub}</span>}
  </div>
);

// ─── Tabs ─────────────────────────────────────────────────────────────────────

const InferenceTab = () => {
  const [telemetry, setTelemetry] = useState<DriveTelemetry>({
    smart_5_raw: 12, smart_187_raw: 2, smart_188_raw: 0, smart_197_raw: 8, smart_198_raw: 8,
  });
  const [prediction, setPrediction] = useState<EnrichedPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState<Array<EnrichedPrediction & { id: number }>>([]);

  const runPrediction = async () => {
    setLoading(true); setError('');
    try {
      const res = await fetch(`${API_URL}/api/models/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(telemetry),
      });
      if (!res.ok) throw new Error(`API Error ${res.status}`);
      const data: EnrichedPrediction = await res.json();
      setPrediction(data);
      setHistory(h => [{ ...data, id: Date.now() }, ...h].slice(0, 30));
    } catch (e) {
      setError('failed to connect to inference server.');
    } finally {
      setLoading(false);
    }
  };

  const historyChartData = history.slice().reverse().map((p, i) => ({
    i: i + 1,
    rul: p.rul_days,
    novelty: p.novelty_score ?? 0,
    risk: p.risk_level,
  }));

  return (
    <div className="tab-content stagger-enter">
      <div className="two-col">
        {/* Left: Inputs */}
        <div className="card">
          <h2 className="card-title">smart telemetry inputs</h2>
          {Object.keys(telemetry).map(key => (
            <div className="input-group" key={key}>
              <label>{FEATURE_LABELS[key] || key}</label>
              <input
                type="number" min={0} max={65535}
                value={telemetry[key as keyof DriveTelemetry]}
                onChange={e => setTelemetry({ ...telemetry, [key]: Number(e.target.value) })}
              />
            </div>
          ))}
          <button className="btn btn-primary" onClick={runPrediction} disabled={loading}>
            {loading ? 'running inference...' : 'run prediction'}
          </button>
          {error && <p className="error-msg">{error}</p>}
        </div>

        {/* Right: Results */}
        <div className="card">
          <h2 className="card-title">prediction results</h2>
          {prediction ? (
            <>
              <div className="metric-grid" style={{ marginBottom: '1.5rem' }}>
                <StatCard label="remaining useful life" value={`${prediction.rul_days}d`} />
                <StatCard label="time to failure" value={`${prediction.ttf_days}d`} />
                <div className="stat-card">
                  <span className="stat-label">risk level</span>
                  <RiskBadge level={prediction.risk_level} />
                </div>
                <StatCard label="log survival time" value={prediction.log_time} />
              </div>
              <div style={{ marginBottom: '1rem' }}>
                <span className="stat-label">latent space novelty</span>
                <NoveltyIndicator score={prediction.novelty_score} isNovel={prediction.is_novel} />
              </div>
              {prediction.fired_alert_ids.length > 0 && (
                <div className="alert-banner">
                  ⚠ {prediction.fired_alert_ids.length} alert(s) fired
                </div>
              )}
            </>
          ) : (
            <p className="muted">enter telemetry and run prediction.</p>
          )}
        </div>
      </div>

      {/* Inference history chart */}
      {history.length > 1 && (
        <div className="card" style={{ marginTop: '1.5rem' }}>
          <h2 className="card-title">inference timeline (last {history.length})</h2>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={historyChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="rulGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="i" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6 }}
                labelStyle={{ color: 'var(--text-muted)' }}
              />
              <Area type="monotone" dataKey="rul" stroke="var(--accent)" fill="url(#rulGrad)" name="RUL (days)" dot={false} />
              <Line type="monotone" dataKey="novelty" stroke="var(--risk-high)" dot={false} name="novelty score" strokeDasharray="4 2" />
              <ReferenceLine y={1} stroke="var(--risk-critical)" strokeDasharray="6 3" label={{ value: 'novelty threshold', fill: 'var(--risk-critical)', fontSize: 10 }} />
              <Legend wrapperStyle={{ color: 'var(--text-muted)', fontSize: 12 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};


const MonitoringTab = ({ health, alerts, onResolve }: {
  health: HealthData | null;
  alerts: Alert[];
  onResolve: (id: string) => void;
}) => {
  const driftData = health?.drift?.feature_scores?.map(s => ({
    name: s.feature.replace('smart_', 'S').replace('_raw', ''),
    psi: s.psi,
    ks: s.ks_statistic,
    severity: s.severity,
  })) || [];


  return (
    <div className="tab-content stagger-enter">
      {/* System health summary */}
      <div className="metric-grid" style={{ marginBottom: '1.5rem' }}>
        <StatCard
          label="system status"
          value={health?.status || '—'}
          sub={health?.status === 'healthy' ? '✓ all systems nominal' : '⚠ degraded'}
        />
        <StatCard label="total inferences" value={health?.total_inferences || 0} />
        <StatCard label="active alerts" value={health?.alerts?.total_active || 0} sub={`${health?.alerts?.critical_count || 0} critical`} />
        <StatCard
          label="faiss index"
          value={health?.faiss?.is_built ? 'built' : 'not built'}
          sub={health?.faiss?.threshold ? `threshold: ${health.faiss.threshold.toFixed(4)}` : ''}
        />
      </div>

      {/* Drift Charts */}
      {driftData.length > 0 && (
        <div className="card">
          <h2 className="card-title">per-feature drift scores (ks + psi)</h2>
          <p className="muted" style={{ marginBottom: '1rem' }}>
            KS statistic and PSI per SMART feature vs. baseline LHS probe distribution.
            PSI &gt; 0.25 triggers a FEATURE_DRIFT alert.
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={driftData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6 }}
              />
              <ReferenceLine y={0.25} stroke="var(--risk-critical)" strokeDasharray="4 2" label={{ value: 'PSI drift threshold', fill: 'var(--risk-critical)', fontSize: 10 }} />
              <Bar dataKey="psi" name="PSI" fill="var(--accent)" radius={[3, 3, 0, 0]} />
              <Bar dataKey="ks" name="KS statistic" fill="var(--accent-dim)" radius={[3, 3, 0, 0]} />
              <Legend wrapperStyle={{ color: 'var(--text-muted)', fontSize: 12 }} />
            </BarChart>
          </ResponsiveContainer>
          <div className="drift-feature-grid">
            {health?.drift?.feature_scores?.map(fs => (
              <div key={fs.feature} className={`drift-chip severity-${fs.severity}`}>
                <span>{fs.feature.replace('_raw', '')}</span>
                <span className="drift-chip-val">{fs.severity}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Active Alerts */}
      <div className="card">
        <h2 className="card-title">active alerts</h2>
        {alerts.length === 0 ? (
          <p className="muted">no active alerts — system is nominal.</p>
        ) : (
          <div className="alert-list">
            {alerts.map(a => (
              <div key={a.id} className={`alert-item alert-${a.alert_type.toLowerCase()} severity-${a.severity.toLowerCase()}`}>
                <div className="alert-header">
                  <span className="alert-type-badge">{a.alert_type}</span>
                  <span className={`alert-severity ${a.severity.toLowerCase()}`}>{a.severity}</span>
                  <span className="alert-time">{new Date(a.created_at).toLocaleTimeString()}</span>
                  <button className="btn btn-ghost btn-sm" onClick={() => onResolve(a.id)}>resolve</button>
                </div>
                <p className="alert-msg">{a.message}</p>
                {a.escalation_count > 0 && (
                  <p className="alert-escalation">escalated {a.escalation_count}× — consecutive occurrences</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};


const AutopsyTab = ({ fingerprint }: { fingerprint: Fingerprint | null }) => {
  const [topology, setTopology] = useState<Record<string, unknown> | null>(null);
  const [loadingProbe, setLoadingProbe] = useState(false);
  const [probeResult, setProbeResult] = useState<string>('');

  const runProbe = async () => {
    setLoadingProbe(true); setProbeResult('');
    try {
      const res = await fetch(`${API_URL}/api/models/probe`, { method: 'POST' });
      if (!res.ok) throw new Error('Probe failed');
      const data = await res.json();
      setProbeResult(`probe complete — fingerprint: ${data.fingerprint_id} | threshold: ${data.novelty_threshold?.toFixed(4)}`);
    } catch {
      setProbeResult('probe failed — check server logs.');
    } finally {
      setLoadingProbe(false);
    }
  };

  useEffect(() => {
    fetch(`${API_URL}/api/models/topology`)
      .then(r => r.json())
      .then(setTopology)
      .catch(() => {});
  }, []);

  const histogramData = fingerprint?.histogram?.bins?.map((bin, i) => ({
    day: Math.round(bin), count: fingerprint.histogram.counts[i] || 0,
  })).filter(d => d.count > 0) || [];

  const featureImportanceData = Object.entries(fingerprint?.feature_importance || {}).map(([feat, val]) => ({
    name: feat.replace('smart_', 'S').replace('_raw', ''),
    importance: Math.round((val as number) * 100),
  })).sort((a, b) => b.importance - a.importance);

  return (
    <div className="tab-content stagger-enter">
      <div className="card">
        <h2 className="card-title">model autopsy & fingerprint</h2>
        <p className="muted" style={{ marginBottom: '1rem' }}>
          Run a 500-sample Latin Hypercube Sampling probe across the full SMART feature space.
          This builds the FAISS novelty index and behavioral fingerprint.
        </p>
        <button className="btn btn-primary" onClick={runProbe} disabled={loadingProbe}>
          {loadingProbe ? 'probing...' : 'run lhs probe sweep'}
        </button>
        {probeResult && <p className="success-msg" style={{ marginTop: '0.75rem' }}>{probeResult}</p>}
      </div>

      {fingerprint && (
        <>
          {/* Fingerprint summary */}
          <div className="metric-grid">
            <StatCard label="fingerprint id" value={fingerprint.fingerprint_id} />
            <StatCard label="probe samples" value={fingerprint.n_probe_samples} />
            <StatCard label="prediction entropy" value={fingerprint.prediction_entropy.toFixed(3)} sub="0=uniform, 1=max spread" />
            <StatCard label="uncertainty rate" value={`${(fingerprint.uncertainty_rate * 100).toFixed(1)}%`} sub="fraction predicting <90d" />
          </div>

          {/* RUL histogram */}
          <div className="card">
            <h2 className="card-title">rul prediction histogram (lhs probe)</h2>
            <div className="rul-distribution-row" style={{ marginBottom: '1rem' }}>
              {[
                { l: 'mean', v: fingerprint.rul_distribution.mean.toFixed(0) + 'd' },
                { l: 'p10', v: fingerprint.rul_distribution.p10.toFixed(0) + 'd' },
                { l: 'median', v: fingerprint.rul_distribution.median.toFixed(0) + 'd' },
                { l: 'p90', v: fingerprint.rul_distribution.p90.toFixed(0) + 'd' },
              ].map(({ l, v }) => <StatCard key={l} label={l} value={v} />)}
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={histogramData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="day" stroke="var(--text-muted)" tick={{ fontSize: 10 }} label={{ value: 'days', position: 'insideBottomRight', fill: 'var(--text-muted)', fontSize: 11 }} />
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6 }} />
                <Bar dataKey="count" fill="var(--accent)" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Feature importance */}
          <div className="two-col">
            <div className="card">
              <h2 className="card-title">feature importance (gain)</h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={featureImportanceData} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                  <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 11 }} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="name" stroke="var(--text-muted)" tick={{ fontSize: 11 }} width={40} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6 }} formatter={(v) => `${v}%`} />
                  <Bar dataKey="importance" fill="var(--accent)" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Class bias */}
            <div className="card">
              <h2 className="card-title">risk tier distribution (class bias)</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
                {Object.entries(fingerprint.class_bias).map(([tier, frac]) => (
                  <div key={tier}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <RiskBadge level={tier} />
                      <span className="muted">{(frac * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bias-bar-track">
                      <div className={`bias-bar-fill bias-${tier}`} style={{ width: `${frac * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* XGBoost Topology */}
      {topology && (
        <div className="card">
          <h2 className="card-title">xgboost ensemble topology</h2>
          <div className="metric-grid" style={{ marginBottom: '1.5rem' }}>
            <StatCard label="n trees" value={(topology.n_trees as number) || 0} />
            <StatCard label="total nodes" value={(topology.total_nodes as number) || 0} />
            <StatCard label="total leaves" value={(topology.total_leaves as number) || 0} />
            <StatCard label="max depth" value={(topology.max_depth as number) || 0} />
          </div>
          <div className="topology-trees">
            {((topology.tree_summaries as Array<Record<string, number>>) || []).map((t) => (
              <div key={t.tree_id} className="tree-chip">
                <span className="tree-id">tree {t.tree_id}</span>
                <span className="tree-meta">{t.n_nodes} nodes · {t.n_leaves} leaves · depth {t.max_depth}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function DataCenterGuardian() {
  const [activeTab, setActiveTab] = useState('inference');
  const [health, setHealth] = useState<HealthData | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [fingerprint, setFingerprint] = useState<Fingerprint | null>(null);

  const tabs = [
    { id: 'inference', label: 'inference' },
    { id: 'monitoring', label: 'monitoring' },
    { id: 'autopsy', label: 'model autopsy' },
  ];

  const fetchHealth = useCallback(async () => {
    try {
      const [healthRes, alertsRes, fpRes] = await Promise.all([
        fetch(`${API_URL}/api/models/health`),
        fetch(`${API_URL}/api/models/alerts`),
        fetch(`${API_URL}/api/models/fingerprint`),
      ]);
      if (healthRes.ok) setHealth(await healthRes.json());
      if (alertsRes.ok) setAlerts(await alertsRes.json());
      if (fpRes.ok) setFingerprint(await fpRes.json());
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 5000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  const resolveAlert = async (id: string) => {
    try {
      await fetch(`${API_URL}/api/models/alerts/${id}`, { method: 'DELETE' });
      fetchHealth();
    } catch { /* silent */ }
  };

  const criticalCount = health?.alerts?.critical_count || 0;
  const activeAlertCount = health?.alerts?.total_active || 0;

  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1 className="title">data center guardian</h1>
          <p className="subtitle">predictive maintenance · mlops observability platform</p>
        </div>
        <div className="header-right">
          {activeAlertCount > 0 && (
            <div className={`global-alert-badge ${criticalCount > 0 ? 'critical' : 'warning'}`}>
              {criticalCount > 0 ? '🔴' : '🟡'} {activeAlertCount} alert{activeAlertCount > 1 ? 's' : ''} active
            </div>
          )}
          <div className={`status-dot ${health?.status === 'healthy' ? 'green' : 'red'}`} title={health?.status || 'connecting...'} />
          <nav className="tabs">
            {tabs.map(t => (
              <button key={t.id} className={`tab-btn ${activeTab === t.id ? 'active' : ''}`} onClick={() => setActiveTab(t.id)}>
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main>
        {activeTab === 'inference' && <InferenceTab />}
        {activeTab === 'monitoring' && <MonitoringTab health={health} alerts={alerts} onResolve={resolveAlert} />}
        {activeTab === 'autopsy' && <AutopsyTab fingerprint={fingerprint} />}
      </main>
    </div>
  );
}
