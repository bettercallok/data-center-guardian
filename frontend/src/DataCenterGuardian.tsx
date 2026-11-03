import { useState, useEffect, useCallback, useRef, type FormEvent } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Line, LineChart, ReferenceLine, Legend, AreaChart, Area
} from 'recharts';

import { API_URL, apiFetch, errorMessage, getAdminKey, setAdminKey } from './api';
import { CSV_TEMPLATE, MAX_CSV_BYTES, downloadText, parseFleetCsv } from './csv';
import {
  FEATURE_LABELS, RISK_ORDER, SMART_FEATURES,
  type Alert, type ApiMeta, type BatchPredictionResponse, type DriftSnapshot, type DriveTelemetry,
  type EnrichedPrediction, type Fingerprint, type HealthData, type PredictionSummary,
  type RiskLevel, type SmartFeature, type SmartValues, type Topology,
} from './types';

// ─── Constants ────────────────────────────────────────────────────────────────

const POLL_MS = 5000;

const PRESETS: Array<{ id: string; label: string; values: SmartValues }> = [
  { id: 'healthy', label: 'healthy', values: { smart_5_raw: 0, smart_187_raw: 0, smart_188_raw: 0, smart_197_raw: 0, smart_198_raw: 0 } },
  { id: 'degrading', label: 'degrading', values: { smart_5_raw: 24, smart_187_raw: 3, smart_188_raw: 0, smart_197_raw: 16, smart_198_raw: 16 } },
  { id: 'failing', label: 'failing', values: { smart_5_raw: 320, smart_187_raw: 120, smart_188_raw: 600, smart_197_raw: 280, smart_198_raw: 280 } },
];

const TOOLTIP_STYLE = { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6 };
const FEATURE_COLORS = ['#4f8ef7', '#22c55e', '#f59e0b', '#f97316', '#a78bfa'];

const shortFeature = (f: string) => f.replace('smart_', 'S').replace('_raw', '');
const clampSmart = (raw: string) => {
  const n = Math.floor(Number(raw));
  return Number.isFinite(n) ? Math.min(65535, Math.max(0, n)) : 0;
};

// ─── Sub-components ────────────────────────────────────────────────────────────

const RiskBadge = ({ level }: { level: string }) => (
  <span className={`badge badge-${level}`}>{level}</span>
);

const NoveltyIndicator = ({ score, isNovel }: { score: number | null; isNovel: boolean | null }) => {
  if (score === null) return <span className="novelty-na">—</span>;
  // Track spans 0 → 1.5; the threshold marker sits at 1.0 (2/3 of the track).
  const pct = Math.min(score / 1.5, 1) * 100;
  return (
    <div className="novelty-bar-wrap">
      <div
        className="novelty-bar-track"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1.5}
        aria-valuenow={Number(score.toFixed(2))}
        aria-label="novelty score"
      >
        <div
          className="novelty-bar-fill"
          style={{ width: `${pct}%`, background: isNovel ? 'var(--risk-critical)' : 'var(--accent-glow)' }}
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

// ─── Inference Tab ────────────────────────────────────────────────────────────

const InferenceTab = () => {
  const [telemetry, setTelemetry] = useState<SmartValues>(PRESETS[1].values);
  const [serial, setSerial] = useState('');
  const [prediction, setPrediction] = useState<EnrichedPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState<Array<EnrichedPrediction & { key: number }>>([]);

  const runPrediction = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true); setError('');
    try {
      const body: DriveTelemetry = { ...telemetry, ...(serial.trim() ? { serial_number: serial.trim() } : {}) };
      const data = await apiFetch<EnrichedPrediction>('/api/models/predict', { method: 'POST', body });
      setPrediction(data);
      setHistory(h => [{ ...data, key: Date.now() }, ...h].slice(0, 30));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const historyChartData = history.slice().reverse().map((p, i) => ({
    i: i + 1,
    rul: p.rul_days,
    novelty: p.novelty_score ?? 0,
  }));

  return (
    <div className="tab-content stagger-enter">
      <div className="two-col">
        {/* Left: Inputs */}
        <form className="card" onSubmit={runPrediction}>
          <h2 className="card-title">smart telemetry inputs</h2>
          <div className="preset-row" role="group" aria-label="telemetry presets">
            <span className="stat-label">presets</span>
            {PRESETS.map(p => (
              <button type="button" key={p.id} className="btn btn-ghost btn-sm" onClick={() => setTelemetry(p.values)}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="input-group">
            <label htmlFor="serial_number">serial number (optional)</label>
            <input
              id="serial_number" type="text" maxLength={64} placeholder="e.g. ZA1234XY"
              value={serial} onChange={e => setSerial(e.target.value)}
              pattern="[A-Za-z0-9][A-Za-z0-9 ._\-/]{0,63}"
              title="letters, digits, space, . _ - /"
            />
          </div>
          {SMART_FEATURES.map((key: SmartFeature) => (
            <div className="input-group" key={key}>
              <label htmlFor={key}>{FEATURE_LABELS[key]}</label>
              <input
                id={key} type="number" min={0} max={65535} step={1} inputMode="numeric"
                value={telemetry[key]}
                onChange={e => setTelemetry(t => ({ ...t, [key]: clampSmart(e.target.value) }))}
              />
            </div>
          ))}
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'running inference...' : 'run prediction'}
          </button>
          {error && <p className="error-msg" role="alert">{error}</p>}
        </form>

        {/* Right: Results */}
        <div className="card" aria-live="polite">
          <h2 className="card-title">prediction results</h2>
          {prediction ? (
            <>
              <div className="metric-grid" style={{ marginBottom: '1.5rem' }}>
                <StatCard label="remaining useful life" value={`${prediction.rul_days}d`} sub="ttf minus 30d maintenance buffer" />
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
                {prediction.out_of_range_features.length > 0 && (
                  <p className="muted" style={{ marginTop: '0.4rem' }}>
                    outside probed operating range: {prediction.out_of_range_features.map(shortFeature).join(', ')}
                  </p>
                )}
              </div>
              {prediction.fired_alert_ids.length > 0 && (
                <div className="alert-banner" role="status">
                  ⚠ {prediction.fired_alert_ids.length} alert(s) fired — see monitoring
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
              <YAxis yAxisId="rul" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="nov" orientation="right" stroke="var(--text-muted)" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={{ color: 'var(--text-muted)' }} />
              <Area yAxisId="rul" type="monotone" dataKey="rul" stroke="var(--accent)" fill="url(#rulGrad)" name="RUL (days)" dot={false} />
              <Line yAxisId="nov" type="monotone" dataKey="novelty" stroke="var(--risk-high)" dot={false} name="novelty score" strokeDasharray="4 2" />
              <ReferenceLine yAxisId="nov" y={1} stroke="var(--risk-critical)" strokeDasharray="6 3" label={{ value: 'novelty threshold', fill: 'var(--risk-critical)', fontSize: 10 }} />
              <Legend wrapperStyle={{ color: 'var(--text-muted)', fontSize: 12 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// ─── Fleet Scan Tab (batch CSV) ───────────────────────────────────────────────

type FleetRow = EnrichedPrediction & { drive_model?: string; row: number };

const FleetTab = ({ maxBatch }: { maxBatch: number }) => {
  const [drives, setDrives] = useState<DriveTelemetry[]>([]);
  const [parseErrors, setParseErrors] = useState<string[]>([]);
  const [fileName, setFileName] = useState('');
  const [results, setResults] = useState<FleetRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);

  const onFile = async (file: File | undefined) => {
    setResults([]); setError(''); setParseErrors([]); setDrives([]);
    if (!file) return;
    setFileName(file.name);
    if (file.size > MAX_CSV_BYTES) {
      setParseErrors([`file is larger than ${MAX_CSV_BYTES / 1024} KB.`]);
      return;
    }
    const parsed = parseFleetCsv(await file.text());
    setDrives(parsed.drives);
    setParseErrors(parsed.errors);
  };

  const runScan = async () => {
    setLoading(true); setError(''); setResults([]);
    try {
      const out: FleetRow[] = [];
      for (let start = 0; start < drives.length; start += maxBatch) {
        const chunk = drives.slice(start, start + maxBatch);
        const res = await apiFetch<BatchPredictionResponse>('/api/models/predict/batch', {
          method: 'POST', body: { drives: chunk },
        });
        res.predictions.forEach((p, i) => out.push({ ...p, drive_model: chunk[i].drive_model, row: start + i + 1 }));
      }
      out.sort((a, b) => RISK_ORDER[a.risk_level] - RISK_ORDER[b.risk_level] || a.rul_days - b.rul_days);
      setResults(out);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const counts = results.reduce<Record<RiskLevel, number>>(
    (acc, r) => ({ ...acc, [r.risk_level]: acc[r.risk_level] + 1 }),
    { critical: 0, high: 0, medium: 0, low: 0 },
  );

  const exportResults = () => {
    const header = 'row,serial_number,drive_model,risk_level,rul_days,ttf_days,novelty_score,is_novel';
    const esc = (v: unknown) => {
      const s = v === null || v === undefined ? '' : String(v);
      return /^[=+\-@\t\r]/.test(s) && Number.isNaN(Number(s)) ? `'${s}` : s;
    };
    const lines = results.map(r =>
      [r.row, r.serial_number, r.drive_model, r.risk_level, r.rul_days, r.ttf_days, r.novelty_score, r.is_novel].map(esc).join(','),
    );
    downloadText('fleet_scan_results.csv', [header, ...lines].join('\n') + '\n');
  };

  return (
    <div className="tab-content stagger-enter">
      <div className="card">
        <h2 className="card-title">fleet scan — batch csv upload</h2>
        <p className="muted" style={{ marginBottom: '1rem' }}>
          Upload a CSV of drives (columns: serial_number, drive_model and the five SMART attributes) to score
          the whole fleet at once. Results are sorted by urgency. Files are parsed in your browser; up to 1,000 rows.
        </p>
        <div className="fleet-actions">
          <input
            ref={fileInput} id="fleet-csv" type="file" accept=".csv,text/csv" className="visually-hidden"
            onChange={e => void onFile(e.target.files?.[0])}
          />
          <button type="button" className="btn btn-ghost" onClick={() => fileInput.current?.click()}>
            {fileName ? `📄 ${fileName}` : 'choose csv file'}
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => downloadText('fleet_template.csv', CSV_TEMPLATE)}>
            download template
          </button>
          <a className="btn btn-ghost" href={`${API_URL}/api/models/predictions/export`} download>
            export prediction history
          </a>
          <button type="button" className="btn btn-primary" style={{ marginTop: 0 }} disabled={!drives.length || loading} onClick={runScan}>
            {loading ? 'scanning...' : drives.length ? `scan ${drives.length} drive${drives.length > 1 ? 's' : ''}` : 'scan fleet'}
          </button>
        </div>
        {parseErrors.length > 0 && (
          <ul className="error-list" role="alert">
            {parseErrors.map(e => <li key={e}>{e}</li>)}
          </ul>
        )}
        {error && <p className="error-msg" role="alert">{error}</p>}
      </div>

      {results.length > 0 && (
        <>
          <div className="metric-grid">
            {(['critical', 'high', 'medium', 'low'] as RiskLevel[]).map(tier => (
              <div className="stat-card" key={tier}>
                <span className="stat-label">{tier} risk</span>
                <span className={`stat-value risk-text-${tier}`}>{counts[tier]}</span>
              </div>
            ))}
            <StatCard label="novel inputs" value={results.filter(r => r.is_novel).length} sub="outside training distribution" />
          </div>
          <div className="card">
            <div className="card-title-row">
              <h2 className="card-title">scan results ({results.length})</h2>
              <button type="button" className="btn btn-ghost btn-sm" onClick={exportResults}>download results</button>
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">row</th><th scope="col">serial</th><th scope="col">model</th>
                    <th scope="col">risk</th><th scope="col" className="num">rul</th>
                    <th scope="col" className="num">ttf</th><th scope="col" className="num">novelty</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map(r => (
                    <tr key={r.row}>
                      <td className="mono">{r.row}</td>
                      <td className="mono">{r.serial_number ?? '—'}</td>
                      <td>{r.drive_model ?? '—'}</td>
                      <td><RiskBadge level={r.risk_level} /></td>
                      <td className="num mono">{r.rul_days}d</td>
                      <td className="num mono">{r.ttf_days}d</td>
                      <td className={`num mono ${r.is_novel ? 'risk-text-critical' : ''}`}>
                        {r.novelty_score === null ? '—' : r.novelty_score.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// ─── Monitoring Tab ───────────────────────────────────────────────────────────

const MonitoringTab = ({ health, alerts, summary, driftHistory, alertFilter, onAlertFilter, onResolve }: {
  health: HealthData | null;
  alerts: Alert[];
  summary: PredictionSummary | null;
  driftHistory: DriftSnapshot[];
  alertFilter: 'active' | 'all';
  onAlertFilter: (f: 'active' | 'all') => void;
  onResolve: (id: string) => Promise<string | null>;
}) => {
  const [resolveError, setResolveError] = useState('');

  const driftData = health?.drift?.feature_scores?.map(s => ({
    name: shortFeature(s.feature),
    psi: s.psi,
    ks: s.ks_statistic,
  })) || [];

  const trendData = driftHistory.map((snap, i) => {
    const row: Record<string, number | string> = { i: i + 1 };
    snap.feature_scores.forEach(fs => { row[shortFeature(fs.feature)] = fs.psi; });
    return row;
  });

  const resolve = async (id: string) => {
    setResolveError('');
    const err = await onResolve(id);
    if (err) setResolveError(err);
  };

  return (
    <div className="tab-content stagger-enter">
      {/* System health summary */}
      <div className="metric-grid" style={{ marginBottom: '1.5rem' }}>
        <StatCard
          label="system status"
          value={health?.status || '—'}
          sub={health?.status === 'healthy' ? '✓ all systems nominal' : health ? '⚠ attention needed' : 'connecting...'}
        />
        <StatCard label="total inferences" value={health?.total_inferences ?? 0} sub="since last restart" />
        <StatCard label="active alerts" value={health?.alerts?.total_active ?? 0} sub={`${health?.alerts?.critical_count ?? 0} critical`} />
        <StatCard
          label="faiss index"
          value={health?.faiss?.is_built ? 'built' : 'not built'}
          sub={health?.faiss?.threshold ? `threshold: ${health.faiss.threshold.toFixed(4)}` : ''}
        />
      </div>

      {summary && summary.total_predictions > 0 && (
        <div className="card">
          <h2 className="card-title">prediction history (all time)</h2>
          <div className="metric-grid" style={{ marginBottom: 0 }}>
            <StatCard label="predictions stored" value={summary.total_predictions} />
            <StatCard label="tracked drives" value={summary.tracked_drives} sub="with a serial number" />
            <StatCard label="mean rul" value={summary.mean_rul_days !== null ? `${summary.mean_rul_days}d` : '—'} />
            <StatCard label="novelty rate" value={`${(summary.novelty_rate * 100).toFixed(1)}%`} />
          </div>
        </div>
      )}

      {/* Drift Charts */}
      {driftData.length > 0 ? (
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
              <Tooltip contentStyle={TOOLTIP_STYLE} />
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
      ) : (
        <div className="card">
          <h2 className="card-title">per-feature drift scores (ks + psi)</h2>
          <p className="muted">
            collecting samples — drift statistics appear after 30 inferences
            ({health?.drift_window_count ?? 0}/30 in the window).
          </p>
        </div>
      )}

      {trendData.length > 1 && (
        <div className="card">
          <h2 className="card-title">psi trend (persisted drift snapshots)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trendData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="i" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <ReferenceLine y={0.25} stroke="var(--risk-critical)" strokeDasharray="4 2" />
              {SMART_FEATURES.map((f, i) => (
                <Line key={f} type="monotone" dataKey={shortFeature(f)} stroke={FEATURE_COLORS[i]} dot={false} />
              ))}
              <Legend wrapperStyle={{ color: 'var(--text-muted)', fontSize: 12 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Alerts */}
      <div className="card">
        <div className="card-title-row">
          <h2 className="card-title">{alertFilter === 'active' ? 'active alerts' : 'alert history'}</h2>
          <div className="segmented" role="group" aria-label="alert filter">
            {(['active', 'all'] as const).map(f => (
              <button
                type="button" key={f} aria-pressed={alertFilter === f}
                className={`seg-btn ${alertFilter === f ? 'active' : ''}`}
                onClick={() => onAlertFilter(f)}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
        {resolveError && <p className="error-msg" role="alert" style={{ marginBottom: '0.75rem' }}>{resolveError}</p>}
        {alerts.length === 0 ? (
          <p className="muted">{alertFilter === 'active' ? 'no active alerts — system is nominal.' : 'no alerts recorded yet.'}</p>
        ) : (
          <div className="alert-list">
            {alerts.map(a => (
              <div key={a.id} className={`alert-item alert-${a.alert_type.toLowerCase()} severity-${a.severity.toLowerCase()} ${a.status === 'RESOLVED' ? 'resolved' : ''}`}>
                <div className="alert-header">
                  <span className="alert-type-badge">{a.alert_type}</span>
                  <span className={`alert-severity ${a.severity.toLowerCase()}`}>{a.severity}</span>
                  {a.status === 'RESOLVED' && <span className="alert-resolved-tag">resolved</span>}
                  <time className="alert-time" dateTime={a.updated_at ?? a.created_at}>
                    {new Date(a.updated_at ?? a.created_at).toLocaleTimeString()}
                  </time>
                  {a.status === 'ACTIVE' && (
                    <button type="button" className="btn btn-ghost btn-sm" onClick={() => void resolve(a.id)}>resolve</button>
                  )}
                </div>
                <p className="alert-msg">{a.message}</p>
                {a.occurrences > 1 && (
                  <p className="alert-escalation">seen {a.occurrences}× — escalated {a.escalation_count}×</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// ─── Model Autopsy Tab ────────────────────────────────────────────────────────

const AutopsyTab = ({ fingerprint, onProbed }: { fingerprint: Fingerprint | null; onProbed: () => void }) => {
  const [topology, setTopology] = useState<Topology | null>(null);
  const [loadingProbe, setLoadingProbe] = useState(false);
  const [probeResult, setProbeResult] = useState<{ ok: boolean; text: string } | null>(null);

  const runProbe = async () => {
    setLoadingProbe(true); setProbeResult(null);
    try {
      const data = await apiFetch<{
        fingerprint_id: string; novelty_threshold: number | null; wasserstein_to_previous: number | null;
      }>('/api/models/probe', { method: 'POST', admin: true });
      const shift = data.wasserstein_to_previous !== null ? ` | baseline shift (W1): ${data.wasserstein_to_previous.toFixed(2)}` : '';
      setProbeResult({
        ok: true,
        text: `probe complete — fingerprint: ${data.fingerprint_id} | threshold: ${data.novelty_threshold?.toFixed(4)}${shift}`,
      });
      onProbed();
    } catch (err) {
      setProbeResult({ ok: false, text: `probe failed — ${errorMessage(err)}` });
    } finally {
      setLoadingProbe(false);
    }
  };

  useEffect(() => {
    const ctrl = new AbortController();
    apiFetch<Topology>('/api/models/topology', { signal: ctrl.signal })
      .then(setTopology)
      .catch(() => { /* topology is optional */ });
    return () => ctrl.abort();
  }, []);

  const histogramData = fingerprint?.histogram?.bins?.map((bin, i) => ({
    day: Math.round(bin), count: fingerprint.histogram.counts[i] || 0,
  })).filter(d => d.count > 0) || [];

  const featureImportanceData = Object.entries(fingerprint?.feature_importance || {}).map(([feat, val]) => ({
    name: shortFeature(feat),
    importance: Math.round(val * 100),
  })).sort((a, b) => b.importance - a.importance);

  return (
    <div className="tab-content stagger-enter">
      <div className="card">
        <h2 className="card-title">model autopsy & fingerprint</h2>
        <p className="muted" style={{ marginBottom: '1rem' }}>
          Run a 500-sample Latin Hypercube Sampling probe across the full SMART feature space.
          This rebuilds the FAISS novelty index and behavioral fingerprint. Requires the admin key.
        </p>
        <button type="button" className="btn btn-primary" onClick={runProbe} disabled={loadingProbe}>
          {loadingProbe ? 'probing...' : 'run lhs probe sweep'}
        </button>
        {probeResult && (
          <p className={probeResult.ok ? 'success-msg' : 'error-msg'} role="status" style={{ marginTop: '0.75rem' }}>
            {probeResult.text}
          </p>
        )}
      </div>

      {fingerprint && (
        <>
          {/* Fingerprint summary */}
          <div className="metric-grid">
            <StatCard label="fingerprint id" value={fingerprint.fingerprint_id} />
            <StatCard label="probe samples" value={fingerprint.n_probe_samples} />
            <StatCard label="prediction entropy" value={fingerprint.prediction_entropy.toFixed(3)} sub="0=concentrated, 1=max spread" />
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
                <Tooltip contentStyle={TOOLTIP_STYLE} />
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
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => `${v}%`} />
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
            <StatCard label="n trees" value={topology.n_trees || 0} />
            <StatCard label="total nodes" value={topology.total_nodes || 0} />
            <StatCard label="total leaves" value={topology.total_leaves || 0} />
            <StatCard label="max depth" value={topology.max_depth || 0} />
          </div>
          <div className="topology-trees">
            {(topology.tree_summaries || []).map(t => (
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

// ─── Admin key panel ──────────────────────────────────────────────────────────

const AdminPanel = ({ adminEnabled, onClose }: { adminEnabled: boolean | null; onClose: () => void }) => {
  const [value, setValue] = useState(getAdminKey());
  const [saved, setSaved] = useState(false);

  const save = (e: FormEvent) => {
    e.preventDefault();
    setAdminKey(value.trim());
    setSaved(true);
  };

  return (
    <form className="admin-panel card" onSubmit={save} aria-label="admin access">
      <div className="input-group" style={{ marginBottom: 0, flex: 1 }}>
        <label htmlFor="admin-key">admin api key</label>
        <input
          id="admin-key" type="password" autoComplete="off" value={value}
          onChange={e => { setValue(e.target.value); setSaved(false); }}
          placeholder="required for probe sweeps and resolving alerts"
        />
      </div>
      <div className="admin-actions">
        <button type="submit" className="btn btn-primary" style={{ marginTop: 0 }}>{value ? 'save' : 'clear'}</button>
        <button type="button" className="btn btn-ghost" onClick={onClose}>close</button>
      </div>
      <p className="muted admin-note">
        {saved ? '✓ saved for this browser tab. ' : ''}
        stored in session storage only — cleared when the tab closes.
        {adminEnabled === false && ' note: the server has no admin key configured, so admin actions are disabled.'}
      </p>
    </form>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────────────

type TabId = 'inference' | 'fleet' | 'monitoring' | 'autopsy';

const TABS: Array<{ id: TabId; label: string }> = [
  { id: 'inference', label: 'inference' },
  { id: 'fleet', label: 'fleet scan' },
  { id: 'monitoring', label: 'monitoring' },
  { id: 'autopsy', label: 'model autopsy' },
];

export default function DataCenterGuardian() {
  const [activeTab, setActiveTab] = useState<TabId>('inference');
  const [health, setHealth] = useState<HealthData | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alertFilter, setAlertFilter] = useState<'active' | 'all'>('active');
  const [fingerprint, setFingerprint] = useState<Fingerprint | null>(null);
  const [summary, setSummary] = useState<PredictionSummary | null>(null);
  const [driftHistory, setDriftHistory] = useState<DriftSnapshot[]>([]);
  const [meta, setMeta] = useState<ApiMeta | null>(null);
  const [showAdmin, setShowAdmin] = useState(false);

  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([
      apiFetch<HealthData>('/api/models/health'),
      apiFetch<Alert[]>(`/api/models/alerts?status=${alertFilter}`),
      apiFetch<Fingerprint>('/api/models/fingerprint'),
      apiFetch<PredictionSummary>('/api/models/predictions/summary'),
      apiFetch<DriftSnapshot[]>('/api/models/drift/history?limit=30'),
    ]);
    const [h, a, fp, s, dh] = results;
    setOnline(h.status === 'fulfilled');
    if (h.status === 'fulfilled') setHealth(h.value);
    if (a.status === 'fulfilled') setAlerts(a.value);
    if (fp.status === 'fulfilled') setFingerprint(fp.value);
    if (s.status === 'fulfilled') setSummary(s.value);
    if (dh.status === 'fulfilled') setDriftHistory(dh.value);
  }, [alertFilter]);

  // Poll every few seconds, but only while the browser tab is visible.
  useEffect(() => {
    const tick = () => { if (!document.hidden) void refresh(); };
    const first = setTimeout(tick, 0);
    const interval = setInterval(tick, POLL_MS);
    document.addEventListener('visibilitychange', tick);
    return () => {
      clearTimeout(first);
      clearInterval(interval);
      document.removeEventListener('visibilitychange', tick);
    };
  }, [refresh]);

  useEffect(() => {
    apiFetch<ApiMeta>('/api').then(setMeta).catch(() => { /* optional */ });
  }, []);

  const resolveAlert = async (id: string): Promise<string | null> => {
    try {
      await apiFetch(`/api/models/alerts/${encodeURIComponent(id)}`, { method: 'DELETE', admin: true });
      void refresh();
      return null;
    } catch (err) {
      return errorMessage(err);
    }
  };

  const criticalCount = health?.alerts?.critical_count || 0;
  const activeAlertCount = health?.alerts?.total_active || 0;
  const statusLabel = online === false ? 'offline' : health?.status || 'connecting...';
  const statusColor = online === false ? 'red' : health?.status === 'healthy' ? 'green' : health ? 'amber' : 'grey';

  return (
    <div className="dashboard-container">
      <a href="#main" className="skip-link">skip to content</a>
      <header className="header">
        <div>
          <h1 className="title">data center guardian</h1>
          <p className="subtitle">predictive maintenance · mlops observability platform</p>
        </div>
        <div className="header-right">
          {activeAlertCount > 0 && (
            <button
              type="button"
              className={`global-alert-badge ${criticalCount > 0 ? 'critical' : 'warning'}`}
              onClick={() => setActiveTab('monitoring')}
            >
              {criticalCount > 0 ? '🔴' : '🟡'} {activeAlertCount} alert{activeAlertCount > 1 ? 's' : ''} active
            </button>
          )}
          <span className="status-wrap" title={statusLabel}>
            <span className={`status-dot ${statusColor}`} aria-hidden="true" />
            <span className="status-text">{statusLabel}</span>
          </span>
          <button
            type="button" className={`btn btn-ghost btn-sm ${showAdmin ? 'active' : ''}`}
            aria-expanded={showAdmin} onClick={() => setShowAdmin(s => !s)}
          >
            🔒 admin
          </button>
          <nav className="tabs" role="tablist" aria-label="dashboard sections">
            {TABS.map(t => (
              <button
                key={t.id} type="button" role="tab" id={`tab-${t.id}`}
                aria-selected={activeTab === t.id} aria-controls="main"
                className={`tab-btn ${activeTab === t.id ? 'active' : ''}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {showAdmin && <AdminPanel adminEnabled={meta ? meta.admin_enabled : null} onClose={() => setShowAdmin(false)} />}

      {online === false && (
        <div className="offline-banner" role="status">
          cannot reach the inference server — retrying every {POLL_MS / 1000}s.
        </div>
      )}

      <main id="main" role="tabpanel" aria-labelledby={`tab-${activeTab}`}>
        {activeTab === 'inference' && <InferenceTab />}
        {activeTab === 'fleet' && <FleetTab maxBatch={meta?.max_batch_size || 100} />}
        {activeTab === 'monitoring' && (
          <MonitoringTab
            health={health} alerts={alerts} summary={summary} driftHistory={driftHistory}
            alertFilter={alertFilter} onAlertFilter={setAlertFilter} onResolve={resolveAlert}
          />
        )}
        {activeTab === 'autopsy' && <AutopsyTab fingerprint={fingerprint} onProbed={() => void refresh()} />}
      </main>

      <footer className="footer">
        <span>xgboost aft survival model{meta?.model_sha256 ? ` · model ${meta.model_sha256.slice(0, 8)}` : ''}{meta?.version ? ` · api v${meta.version}` : ''}</span>
        <span>predictions are decision support, not a guarantee — always keep backups.</span>
      </footer>
    </div>
  );
}
