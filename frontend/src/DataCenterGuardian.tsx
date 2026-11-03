import { useState, useEffect, useCallback, useRef, type DragEvent, type FormEvent } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Line, LineChart, ReferenceLine, Legend, AreaChart, Area, Cell
} from 'recharts';

import { API_URL, apiFetch, errorMessage, getAdminKey, setAdminKey } from './api';
import { CSV_TEMPLATE, MAX_CSV_BYTES, downloadText, parseFleetCsv } from './csv';
import { RISK_ACTION, humanDuration } from './format';
import { SERIES_COLORS, applyTheme, axisProps, gridProps, initialTheme, legendProps, tooltipProps, type ThemeMode } from './theme';
import {
  FEATURE_META, RISK_ORDER, SMART_FEATURES,
  type Alert, type ApiMeta, type BatchPredictionResponse, type DriftSnapshot, type DriveTelemetry,
  type EnrichedPrediction, type Fingerprint, type HealthData, type PredictionSummary,
  type RiskLevel, type SmartFeature, type SmartValues, type Topology,
} from './types';
import { CardHeader, EmptyState, Icon, LifeGauge, NoveltyIndicator, RiskBadge, StatCard, type IconName, type Tone } from './ui';

// ─── Constants ────────────────────────────────────────────────────────────────

const POLL_MS = 5000;

const PRESETS: Array<{ id: string; label: string; values: SmartValues }> = [
  { id: 'healthy', label: 'healthy', values: { smart_5_raw: 0, smart_187_raw: 0, smart_188_raw: 0, smart_197_raw: 0, smart_198_raw: 0 } },
  { id: 'degrading', label: 'degrading', values: { smart_5_raw: 24, smart_187_raw: 3, smart_188_raw: 0, smart_197_raw: 16, smart_198_raw: 16 } },
  { id: 'failing', label: 'failing', values: { smart_5_raw: 320, smart_187_raw: 120, smart_188_raw: 600, smart_197_raw: 280, smart_198_raw: 280 } },
];

const RISK_TIERS: RiskLevel[] = ['critical', 'high', 'medium', 'low'];

const shortFeature = (f: string) => f.replace('smart_', 'S').replace('_raw', '');
const clampSmart = (raw: string) => {
  const n = Math.floor(Number(raw));
  return Number.isFinite(n) ? Math.min(65535, Math.max(0, n)) : 0;
};
const fmtDays = (d: number) => `${Math.round(d).toLocaleString()}d`;

// ─── Inference Tab ────────────────────────────────────────────────────────────

const PredictionHero = ({ p }: { p: EnrichedPrediction }) => (
  <div className={`hero tone-${p.risk_level}`}>
    <div className="hero-top">
      <div>
        <span className="eyebrow">remaining useful life</span>
        <div className="hero-value">
          <span className="hero-number">{Math.round(p.rul_days).toLocaleString()}</span>
          <span className="hero-unit">days</span>
        </div>
        <span className="hero-sub">{humanDuration(p.rul_days)} · includes a 30-day maintenance buffer</span>
      </div>
      <RiskBadge level={p.risk_level} />
    </div>
    <LifeGauge ttfDays={p.ttf_days} />
    <p className="hero-action"><Icon name={p.risk_level === 'low' ? 'check' : 'alert'} size={15} />{RISK_ACTION[p.risk_level]}</p>
  </div>
);

const InferenceTab = () => {
  const [telemetry, setTelemetry] = useState<SmartValues>(PRESETS[1].values);
  const [serial, setSerial] = useState('');
  const [prediction, setPrediction] = useState<EnrichedPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState<Array<EnrichedPrediction & { key: number }>>([]);

  const activePreset = PRESETS.find(p => SMART_FEATURES.every(f => p.values[f] === telemetry[f]))?.id;

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
    <div className="tab-content">
      <div className="split">
        {/* Inputs */}
        <form className="card" onSubmit={runPrediction}>
          <CardHeader title="SMART telemetry" description="Raw counter values reported by the drive." />
          <div className="field-row">
            <span className="field-label">presets</span>
            <div className="chip-group" role="group" aria-label="telemetry presets">
              {PRESETS.map(p => (
                <button
                  type="button" key={p.id} aria-pressed={activePreset === p.id}
                  className={`chip ${activePreset === p.id ? 'active' : ''}`}
                  onClick={() => setTelemetry(p.values)}
                >
                  <span className={`chip-dot dot-${p.id}`} aria-hidden="true" />{p.label}
                </button>
              ))}
            </div>
          </div>
          <div className="field">
            <label htmlFor="serial_number">Serial number <span className="optional">optional</span></label>
            <input
              id="serial_number" type="text" maxLength={64} placeholder="e.g. ZA1234XY"
              value={serial} onChange={e => setSerial(e.target.value)}
              pattern="[A-Za-z0-9][A-Za-z0-9 ._\-/]{0,63}" title="letters, digits, space, . _ - /"
            />
          </div>
          <div className="field-grid">
            {SMART_FEATURES.map((key: SmartFeature) => (
              <div className="field" key={key}>
                <label htmlFor={key}>
                  {FEATURE_META[key].name}
                  <span className="code-chip">{FEATURE_META[key].code}</span>
                </label>
                <input
                  id={key} type="number" min={0} max={65535} step={1} inputMode="numeric"
                  value={telemetry[key]} aria-describedby={`${key}-hint`}
                  onChange={e => setTelemetry(t => ({ ...t, [key]: clampSmart(e.target.value) }))}
                />
                <span className="field-hint" id={`${key}-hint`}>{FEATURE_META[key].hint}</span>
              </div>
            ))}
          </div>
          <div className="form-footer">
            <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
              {loading ? <span className="spinner" aria-hidden="true" /> : <Icon name="activity" />}
              {loading ? 'Running inference…' : 'Run prediction'}
            </button>
            {error && <p className="error-msg" role="alert">{error}</p>}
          </div>
        </form>

        {/* Results */}
        <section className="card" aria-live="polite" aria-label="prediction results">
          <CardHeader title="Prediction" description={prediction?.serial_number ? `Drive ${prediction.serial_number}` : 'XGBoost accelerated-failure-time survival model.'} />
          {prediction ? (
            <>
              <PredictionHero p={prediction} />
              <div className="metric-grid compact">
                <StatCard label="time to failure" value={fmtDays(prediction.ttf_days)} />
                <StatCard label="log survival time" value={prediction.log_time.toFixed(3)} />
                <StatCard label="alerts fired" value={prediction.fired_alert_ids.length} tone={prediction.fired_alert_ids.length ? 'danger' : 'neutral'} />
              </div>
              <div className="subsection">
                <span className="field-label">latent space novelty</span>
                <NoveltyIndicator score={prediction.novelty_score} isNovel={prediction.is_novel} />
                {prediction.out_of_range_features.length > 0 && (
                  <p className="field-hint">
                    Outside the probed operating range: {prediction.out_of_range_features.map(shortFeature).join(', ')}
                  </p>
                )}
              </div>
            </>
          ) : (
            <EmptyState icon="activity" title="No prediction yet">
              Choose a preset or enter SMART values, then run a prediction to estimate remaining useful life.
            </EmptyState>
          )}
        </section>
      </div>

      {history.length > 1 && (
        <section className="card">
          <CardHeader title="Session timeline" description={`Last ${history.length} predictions in this browser session.`} />
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={historyChartData} margin={{ top: 10, right: 8, left: -8, bottom: 0 }}>
              <defs>
                <linearGradient id="rulGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.28} />
                  <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="i" {...axisProps} />
              <YAxis yAxisId="rul" {...axisProps} />
              <YAxis yAxisId="nov" orientation="right" {...axisProps} domain={[0, 'auto']} />
              <Tooltip {...tooltipProps} />
              <Area yAxisId="rul" type="monotone" dataKey="rul" stroke="var(--accent)" strokeWidth={2} fill="url(#rulGrad)" name="RUL (days)" dot={false} />
              <Line yAxisId="nov" type="monotone" dataKey="novelty" stroke="var(--risk-high)" strokeWidth={1.5} dot={false} name="novelty score" strokeDasharray="4 3" />
              <ReferenceLine yAxisId="nov" y={1} stroke="var(--risk-critical)" strokeDasharray="6 4" strokeOpacity={0.7} />
              <Legend {...legendProps} />
            </AreaChart>
          </ResponsiveContainer>
        </section>
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
  const [dragging, setDragging] = useState(false);
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

  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDragging(false);
    void onFile(e.dataTransfer.files?.[0]);
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
    <div className="tab-content">
      <section className="card">
        <CardHeader
          title="Fleet scan"
          description="Score many drives at once from a CSV. Files are parsed in your browser (up to 1,000 rows)."
          actions={
            <>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => downloadText('fleet_template.csv', CSV_TEMPLATE)}>
                <Icon name="file" size={14} />Template
              </button>
              <a className="btn btn-ghost btn-sm" href={`${API_URL}/api/models/predictions/export`} download>
                <Icon name="download" size={14} />Export history
              </a>
            </>
          }
        />
        <label
          htmlFor="fleet-csv"
          className={`dropzone ${dragging ? 'dragging' : ''} ${drives.length ? 'loaded' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={fileInput} id="fleet-csv" type="file" accept=".csv,text/csv" className="visually-hidden"
            onChange={e => void onFile(e.target.files?.[0])}
          />
          <span className="dropzone-icon"><Icon name={drives.length ? 'file' : 'upload'} size={22} /></span>
          {fileName ? (
            <span className="dropzone-title">{fileName}</span>
          ) : (
            <span className="dropzone-title">Drop a CSV here or <u>browse</u></span>
          )}
          <span className="dropzone-sub">
            {drives.length
              ? `${drives.length} drive${drives.length > 1 ? 's' : ''} ready to scan`
              : 'columns: serial_number, drive_model, smart_5_raw … smart_198_raw'}
          </span>
        </label>
        <div className="form-footer row">
          <button type="button" className="btn btn-primary" disabled={!drives.length || loading} onClick={runScan}>
            {loading ? <span className="spinner" aria-hidden="true" /> : <Icon name="server" />}
            {loading ? 'Scanning…' : drives.length ? `Scan ${drives.length} drive${drives.length > 1 ? 's' : ''}` : 'Scan fleet'}
          </button>
          {error && <p className="error-msg" role="alert">{error}</p>}
        </div>
        {parseErrors.length > 0 && (
          <ul className="error-list" role="alert">
            {parseErrors.map(e => <li key={e}>{e}</li>)}
          </ul>
        )}
      </section>

      {results.length > 0 && (
        <>
          <div className="metric-grid">
            {RISK_TIERS.map(tier => (
              <StatCard key={tier} label={`${tier} risk`} value={counts[tier]} tone={tier}
                sub={`${Math.round((counts[tier] / results.length) * 100)}% of fleet`} />
            ))}
            <StatCard label="novel inputs" value={results.filter(r => r.is_novel).length} sub="outside training distribution" tone="accent" />
          </div>
          <section className="card flush">
            <CardHeader
              title={`Results · ${results.length} drives`}
              description="Sorted by urgency: highest risk and shortest remaining life first."
              actions={<button type="button" className="btn btn-ghost btn-sm" onClick={exportResults}><Icon name="download" size={14} />Download</button>}
            />
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">#</th><th scope="col">serial</th><th scope="col">model</th>
                    <th scope="col">risk</th><th scope="col" className="num">rul</th>
                    <th scope="col" className="num">ttf</th><th scope="col" className="num">novelty</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map(r => (
                    <tr key={r.row} className={`row-${r.risk_level}`}>
                      <td className="mono dim">{r.row}</td>
                      <td className="mono strong">{r.serial_number ?? '—'}</td>
                      <td>{r.drive_model ?? '—'}</td>
                      <td><RiskBadge level={r.risk_level} /></td>
                      <td className="num mono strong">{fmtDays(r.rul_days)}</td>
                      <td className="num mono">{fmtDays(r.ttf_days)}</td>
                      <td className={`num mono ${r.is_novel ? 'text-danger' : ''}`}>
                        {r.novelty_score === null ? '—' : r.novelty_score.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
};

// ─── Monitoring Tab ───────────────────────────────────────────────────────────

const STATUS_TONE: Record<string, Tone> = { healthy: 'success', degraded: 'warning', unavailable: 'danger' };

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
    <div className="tab-content">
      <div className="metric-grid">
        <StatCard
          label="system status" tone={STATUS_TONE[health?.status ?? ''] ?? 'neutral'}
          value={health?.status || '—'}
          sub={health?.status === 'healthy' ? 'all systems nominal' : health ? 'attention needed' : 'connecting…'}
        />
        <StatCard label="inferences" value={(health?.total_inferences ?? 0).toLocaleString()} sub="since last restart" />
        <StatCard
          label="active alerts" value={health?.alerts?.total_active ?? 0}
          tone={(health?.alerts?.critical_count ?? 0) > 0 ? 'danger' : (health?.alerts?.total_active ?? 0) > 0 ? 'warning' : 'neutral'}
          sub={`${health?.alerts?.critical_count ?? 0} critical`}
        />
        <StatCard
          label="novelty index" value={health?.faiss?.is_built ? 'ready' : 'not built'}
          tone={health?.faiss?.is_built ? 'accent' : 'warning'}
          sub={health?.faiss?.threshold ? `threshold ${health.faiss.threshold.toFixed(3)}` : ''}
        />
      </div>

      {summary && summary.total_predictions > 0 && (
        <section className="card">
          <CardHeader title="Prediction history" description="All predictions stored in the database." />
          <div className="summary-row">
            <div className="summary-stats">
              <StatCard label="stored" value={summary.total_predictions.toLocaleString()} />
              <StatCard label="tracked drives" value={summary.tracked_drives} sub="with a serial number" />
              <StatCard label="mean rul" value={summary.mean_rul_days !== null ? fmtDays(summary.mean_rul_days) : '—'} />
              <StatCard label="novelty rate" value={`${(summary.novelty_rate * 100).toFixed(1)}%`} />
            </div>
            <div className="stacked-bar-wrap">
              <span className="field-label">risk mix</span>
              <div className="stacked-bar" role="img"
                aria-label={RISK_TIERS.map(t => `${t} ${summary.risk_counts[t]}`).join(', ')}>
                {RISK_TIERS.map(t => summary.risk_counts[t] > 0 && (
                  <span key={t} className={`seg seg-${t}`} style={{ flexGrow: summary.risk_counts[t] }}
                    title={`${t}: ${summary.risk_counts[t]}`} />
                ))}
              </div>
              <div className="legend-row">
                {RISK_TIERS.map(t => (
                  <span key={t} className="legend-item"><span className={`legend-dot seg-${t}`} />{t} <b>{summary.risk_counts[t]}</b></span>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      <div className="split">
        <section className="card">
          <CardHeader title="Feature drift" description="KS statistic and PSI vs. the probe baseline. PSI above 0.25 raises a FEATURE_DRIFT alert." />
          {driftData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={driftData} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={3}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="name" {...axisProps} />
                  <YAxis {...axisProps} />
                  <Tooltip {...tooltipProps} />
                  <ReferenceLine y={0.25} stroke="var(--risk-critical)" strokeDasharray="5 4" strokeOpacity={0.8} />
                  <Bar dataKey="psi" name="PSI" fill="var(--accent)" radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
                  <Bar dataKey="ks" name="KS" fill="var(--accent-soft-strong)" radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
                  <Legend {...legendProps} />
                </BarChart>
              </ResponsiveContainer>
              <div className="chip-row">
                {health?.drift?.feature_scores?.map(fs => (
                  <span key={fs.feature} className={`status-chip sev-${fs.severity}`}>
                    {shortFeature(fs.feature)} · {fs.severity}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <EmptyState icon="pulse" title="Collecting samples">
              Drift statistics appear after 30 inferences ({health?.drift_window_count ?? 0}/30 in the window).
            </EmptyState>
          )}
        </section>

        <section className="card">
          <CardHeader title="PSI trend" description="Persisted drift snapshots over time." />
          {trendData.length > 1 ? (
            <ResponsiveContainer width="100%" height={262}>
              <LineChart data={trendData} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="i" {...axisProps} />
                <YAxis {...axisProps} />
                <Tooltip {...tooltipProps} />
                <ReferenceLine y={0.25} stroke="var(--risk-critical)" strokeDasharray="5 4" strokeOpacity={0.8} />
                {SMART_FEATURES.map((f, i) => (
                  <Line key={f} type="monotone" dataKey={shortFeature(f)} stroke={SERIES_COLORS[i]} strokeWidth={2} dot={false} isAnimationActive={false} />
                ))}
                <Legend {...legendProps} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState icon="activity" title="Not enough snapshots yet">
              A snapshot is stored every drift-check interval. The trend appears after two.
            </EmptyState>
          )}
        </section>
      </div>

      <section className="card">
        <CardHeader
          title={alertFilter === 'active' ? 'Active alerts' : 'Alert history'}
          description="Repeat occurrences update a single alert and escalate it to CRITICAL."
          actions={
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
          }
        />
        {resolveError && <p className="error-msg" role="alert" style={{ marginBottom: '0.75rem' }}>{resolveError}</p>}
        {alerts.length === 0 ? (
          <EmptyState icon="check" title={alertFilter === 'active' ? 'All clear' : 'No alerts recorded'}>
            {alertFilter === 'active' ? 'No active alerts. The model is operating within its expected envelope.' : 'Alerts will appear here once they fire.'}
          </EmptyState>
        ) : (
          <ul className="alert-list">
            {alerts.map(a => (
              <li key={a.id} className={`alert-item sev-${a.severity.toLowerCase()} ${a.status === 'RESOLVED' ? 'resolved' : ''}`}>
                <span className="alert-icon"><Icon name={a.status === 'RESOLVED' ? 'check' : 'alert'} size={16} /></span>
                <div className="alert-body">
                  <div className="alert-header">
                    <span className="alert-type">{a.alert_type.replace('_', ' ').toLowerCase()}</span>
                    <span className={`pill pill-${a.severity.toLowerCase()}`}>{a.severity}</span>
                    {a.status === 'RESOLVED' && <span className="pill pill-resolved">resolved</span>}
                    {a.occurrences > 1 && <span className="alert-count">×{a.occurrences}</span>}
                  </div>
                  <p className="alert-msg">{a.message}</p>
                </div>
                <div className="alert-side">
                  <time className="alert-time" dateTime={a.updated_at ?? a.created_at}>
                    {new Date(a.updated_at ?? a.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </time>
                  {a.status === 'ACTIVE' && (
                    <button type="button" className="btn btn-ghost btn-sm" onClick={() => void resolve(a.id)}>Resolve</button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
};

// ─── Model Autopsy Tab ────────────────────────────────────────────────────────

const TIER_COLOR: Record<string, string> = {
  low: 'var(--risk-low)', medium: 'var(--risk-medium)', high: 'var(--risk-high)', critical: 'var(--risk-critical)',
};

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
      const shift = data.wasserstein_to_previous !== null ? ` · baseline shift W1 ${data.wasserstein_to_previous.toFixed(2)}` : '';
      setProbeResult({ ok: true, text: `Fingerprint ${data.fingerprint_id} · threshold ${data.novelty_threshold?.toFixed(3)}${shift}` });
      onProbed();
    } catch (err) {
      setProbeResult({ ok: false, text: `Probe failed — ${errorMessage(err)}` });
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
  })) || [];

  const featureImportanceData = Object.entries(fingerprint?.feature_importance || {}).map(([feat, val]) => ({
    name: shortFeature(feat),
    importance: Math.round(val * 100),
  })).sort((a, b) => b.importance - a.importance);

  return (
    <div className="tab-content">
      <section className="card">
        <CardHeader
          title="Model autopsy"
          description="A Latin Hypercube probe sweep across the SMART feature space rebuilds the novelty index and behavioral fingerprint. Requires the admin key."
          actions={
            <button type="button" className="btn btn-primary" onClick={runProbe} disabled={loadingProbe}>
              {loadingProbe ? <span className="spinner" aria-hidden="true" /> : <Icon name="spark" />}
              {loadingProbe ? 'Probing…' : 'Run probe sweep'}
            </button>
          }
        />
        {probeResult && (
          <p className={`notice ${probeResult.ok ? 'notice-success' : 'notice-error'}`} role="status">
            <Icon name={probeResult.ok ? 'check' : 'alert'} size={15} />{probeResult.text}
          </p>
        )}
        {fingerprint && (
          <div className="metric-grid compact" style={{ marginTop: probeResult ? '1rem' : 0 }}>
            <StatCard label="fingerprint" value={<span className="mono small">{fingerprint.fingerprint_id}</span>} />
            <StatCard label="probe samples" value={fingerprint.n_probe_samples} />
            <StatCard label="prediction entropy" value={fingerprint.prediction_entropy.toFixed(3)} sub="0 concentrated · 1 max spread" />
            <StatCard label="uncertainty rate" value={`${(fingerprint.uncertainty_rate * 100).toFixed(1)}%`} sub="probes predicting < 90 days" />
          </div>
        )}
      </section>

      {fingerprint && (
        <>
          <section className="card">
            <CardHeader title="Predicted time-to-failure distribution" description="Model output across the probe sweep (clipped at 3,000 days)." />
            <div className="pill-stats">
              {[
                { l: 'p10', v: fingerprint.rul_distribution.p10 },
                { l: 'median', v: fingerprint.rul_distribution.median },
                { l: 'mean', v: fingerprint.rul_distribution.mean },
                { l: 'p90', v: fingerprint.rul_distribution.p90 },
              ].map(({ l, v }) => (
                <span key={l} className="pill-stat"><span>{l} rul</span><b className="mono">{fmtDays(v)}</b></span>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={histogramData} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barCategoryGap={1}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="day" {...axisProps} interval="preserveStartEnd" minTickGap={24} tickFormatter={v => `${v}d`} />
                <YAxis {...axisProps} allowDecimals={false} />
                <Tooltip {...tooltipProps} labelFormatter={v => `≈ ${v} days`} />
                <Bar dataKey="count" name="probes" fill="var(--accent)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </section>

          <div className="split">
            <section className="card">
              <CardHeader title="Feature importance" description="Share of total split gain per SMART attribute." />
              <ResponsiveContainer width="100%" height={210}>
                <BarChart data={featureImportanceData} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid {...gridProps} vertical horizontal={false} />
                  <XAxis type="number" {...axisProps} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="name" {...axisProps} width={44} />
                  <Tooltip {...tooltipProps} formatter={(v) => [`${v}%`, 'gain share']} />
                  <Bar dataKey="importance" fill="var(--accent)" radius={[0, 4, 4, 0]} maxBarSize={22} isAnimationActive={false}>
                    {featureImportanceData.map((d, i) => (
                      <Cell key={d.name} fillOpacity={1 - i * 0.14} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="card">
              <CardHeader title="Risk tier distribution" description="Share of probes landing in each tier (class bias)." />
              <div className="bias-list">
                {Object.entries(fingerprint.class_bias).map(([tier, frac]) => (
                  <div key={tier} className="bias-row">
                    <RiskBadge level={tier} />
                    <div className="bias-track">
                      <div className="bias-fill" style={{ width: `${frac * 100}%`, background: TIER_COLOR[tier] }} />
                    </div>
                    <span className="mono bias-pct">{(frac * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </>
      )}

      {topology && (
        <section className="card">
          <CardHeader title="Ensemble topology" description="Structure of the gradient-boosted tree ensemble." />
          <div className="metric-grid compact">
            <StatCard label="trees" value={topology.n_trees.toLocaleString()} />
            <StatCard label="nodes" value={topology.total_nodes.toLocaleString()} />
            <StatCard label="leaves" value={topology.total_leaves.toLocaleString()} />
            <StatCard label="max depth" value={topology.max_depth} />
          </div>
          <div className="tree-row">
            {(topology.tree_summaries || []).map(t => (
              <div key={t.tree_id} className="tree-chip">
                <span className="tree-id">tree {t.tree_id}</span>
                <span className="tree-meta">{t.n_nodes} nodes · {t.n_leaves} leaves · depth {t.max_depth}</span>
              </div>
            ))}
          </div>
        </section>
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
    <form className="admin-panel" onSubmit={save} aria-label="admin access">
      <span className="admin-icon"><Icon name="lock" size={18} /></span>
      <div className="field admin-field">
        <label htmlFor="admin-key">Admin API key</label>
        <input
          id="admin-key" type="password" autoComplete="off" value={value}
          onChange={e => { setValue(e.target.value); setSaved(false); }}
          placeholder="required for probe sweeps and resolving alerts"
        />
        <span className="field-hint">
          {saved ? '✓ Saved for this tab. ' : ''}Kept in session storage only and cleared when the tab closes.
          {adminEnabled === false && ' The server has no admin key configured, so admin actions are disabled.'}
        </span>
      </div>
      <div className="admin-actions">
        <button type="submit" className="btn btn-primary">{value ? 'Save' : 'Clear'}</button>
        <button type="button" className="btn btn-ghost" onClick={onClose}>Close</button>
      </div>
    </form>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────────────

type TabId = 'inference' | 'fleet' | 'monitoring' | 'autopsy';

const TABS: Array<{ id: TabId; label: string; icon: IconName }> = [
  { id: 'inference', label: 'Inference', icon: 'activity' },
  { id: 'fleet', label: 'Fleet scan', icon: 'server' },
  { id: 'monitoring', label: 'Monitoring', icon: 'pulse' },
  { id: 'autopsy', label: 'Model autopsy', icon: 'layers' },
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
  const [theme, setTheme] = useState<ThemeMode>(initialTheme);

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    applyTheme(next, true);
    setTheme(next);
  };

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
  const statusLabel = online === false ? 'offline' : health?.status || 'connecting';
  const statusTone = online === false ? 'danger' : health?.status === 'healthy' ? 'success' : health ? 'warning' : 'neutral';

  return (
    <div className="app">
      <a href="#main" className="skip-link">Skip to content</a>

      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="brand-mark"><Icon name="logo" size={20} /></span>
            <div>
              <h1 className="brand-name">Data Center Guardian</h1>
              <p className="brand-sub">Predictive maintenance · MLOps observability</p>
            </div>
          </div>
          <div className="topbar-actions">
            {activeAlertCount > 0 && (
              <button
                type="button" className={`status-chip ${criticalCount > 0 ? 'tone-danger' : 'tone-warning'} clickable`}
                onClick={() => setActiveTab('monitoring')}
              >
                <Icon name="alert" size={13} />
                {activeAlertCount} alert{activeAlertCount > 1 ? 's' : ''}
              </button>
            )}
            <span className={`status-chip tone-${statusTone}`} role="status">
              <span className="status-dot" aria-hidden="true" />{statusLabel}
            </span>
            <button
              type="button" className={`icon-btn ${showAdmin ? 'active' : ''}`}
              aria-expanded={showAdmin} aria-label="admin access" title="Admin access"
              onClick={() => setShowAdmin(s => !s)}
            >
              <Icon name="lock" />
            </button>
            <button
              type="button" className="icon-btn" onClick={toggleTheme}
              aria-label={`switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            >
              <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
            </button>
          </div>
        </div>
        <nav className="tabbar" role="tablist" aria-label="dashboard sections">
          {TABS.map(t => (
            <button
              key={t.id} type="button" role="tab" id={`tab-${t.id}`}
              aria-selected={activeTab === t.id} aria-controls="main"
              className={`tab ${activeTab === t.id ? 'active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              <Icon name={t.icon} size={15} />
              {t.label}
              {t.id === 'monitoring' && activeAlertCount > 0 && (
                <span className={`tab-badge ${criticalCount > 0 ? 'critical' : ''}`}>{activeAlertCount}</span>
              )}
            </button>
          ))}
        </nav>
      </header>

      <div className="page">
        {showAdmin && <AdminPanel adminEnabled={meta ? meta.admin_enabled : null} onClose={() => setShowAdmin(false)} />}

        {online === false && (
          <div className="notice notice-error banner" role="status">
            <Icon name="alert" size={15} />Cannot reach the inference server — retrying every {POLL_MS / 1000}s.
          </div>
        )}

        <main id="main" role="tabpanel" aria-labelledby={`tab-${activeTab}`} key={activeTab}>
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
          <span>
            XGBoost AFT survival model
            {meta?.model_sha256 ? <> · model <span className="mono">{meta.model_sha256.slice(0, 8)}</span></> : null}
            {meta?.version ? ` · API v${meta.version}` : ''}
          </span>
          <span>Predictions are decision support, not a guarantee. Always keep backups.</span>
        </footer>
      </div>
    </div>
  );
}
