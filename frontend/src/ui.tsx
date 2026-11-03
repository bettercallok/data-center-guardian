import type { ReactNode } from 'react';
import { GAUGE_TIERS, gaugePosition } from './format';

// ─── Icons (inline SVG, 24px grid, stroke = currentColor) ─────────────────────

const ICON_PATHS: Record<string, ReactNode> = {
  logo: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="3" />
      <path d="M3 10h18M7 7h.01M7 14h4" />
      <circle cx="16.5" cy="15" r="2.5" />
    </>
  ),
  activity: <path d="M3 12h4l3-8 4 16 3-8h4" />,
  server: (
    <>
      <rect x="3" y="4" width="18" height="7" rx="2" />
      <rect x="3" y="13" width="18" height="7" rx="2" />
      <path d="M7 7.5h.01M7 16.5h.01" />
    </>
  ),
  pulse: (
    <>
      <path d="M12 21s-7-4.35-9-9.5C1.6 7.9 4 5 7 5c2 0 3.5 1.2 5 3 1.5-1.8 3-3 5-3 3 0 5.4 2.9 4 6.5C19 16.65 12 21 12 21z" />
      <path d="M7 12h2.5l1.5-2.5 2 4 1.5-1.5H17" />
    </>
  ),
  layers: (
    <>
      <path d="M12 3 3 8l9 5 9-5-9-5z" />
      <path d="m3 13 9 5 9-5" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  moon: <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />,
  lock: (
    <>
      <rect x="4" y="11" width="16" height="10" rx="2" />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" />
    </>
  ),
  upload: <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />,
  download: <path d="M12 4v12m0 0-4-4m4 4 4-4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />,
  alert: (
    <>
      <path d="M10.3 3.9 2.4 17.5A2 2 0 0 0 4.1 20.5h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      <path d="M12 9v4M12 17h.01" />
    </>
  ),
  check: <path d="M20 6 9 17l-5-5" />,
  spark: <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" />,
  file: (
    <>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
    </>
  ),
};

export type IconName = keyof typeof ICON_PATHS;

export const Icon = ({ name, size = 16, className }: { name: IconName; size?: number; className?: string }) => (
  <svg
    width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round"
    className={className} aria-hidden="true" focusable="false"
  >
    {ICON_PATHS[name]}
  </svg>
);

// ─── Small building blocks ────────────────────────────────────────────────────

export type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'low' | 'medium' | 'high' | 'critical';

export const RiskBadge = ({ level }: { level: string }) => (
  <span className={`badge badge-${level}`}>
    <span className="badge-dot" aria-hidden="true" />
    {level}
  </span>
);

export const StatCard = ({ label, value, sub, tone = 'neutral' }: {
  label: string; value: ReactNode; sub?: ReactNode; tone?: Tone;
}) => (
  <div className={`stat-card tone-${tone}`}>
    <span className="stat-label">{label}</span>
    <span className="stat-value">{value}</span>
    {sub && <span className="stat-sub">{sub}</span>}
  </div>
);

export const CardHeader = ({ title, description, actions }: {
  title: string; description?: ReactNode; actions?: ReactNode;
}) => (
  <div className="card-header">
    <div className="card-header-text">
      <h2 className="card-title">{title}</h2>
      {description && <p className="card-desc">{description}</p>}
    </div>
    {actions && <div className="card-actions">{actions}</div>}
  </div>
);

export const EmptyState = ({ icon, title, children }: { icon: IconName; title: string; children?: ReactNode }) => (
  <div className="empty-state">
    <span className="empty-icon"><Icon name={icon} size={22} /></span>
    <p className="empty-title">{title}</p>
    {children && <p className="empty-body">{children}</p>}
  </div>
);

export const NoveltyIndicator = ({ score, isNovel }: { score: number | null; isNovel: boolean | null }) => {
  if (score === null) return <span className="muted">novelty scoring unavailable</span>;
  // Track spans 0 → 1.5; the threshold marker sits at 1.0 (2/3 of the track).
  const pct = Math.min(score / 1.5, 1) * 100;
  return (
    <div className="novelty">
      <div
        className="novelty-track" role="meter" aria-label="novelty score"
        aria-valuemin={0} aria-valuemax={1.5} aria-valuenow={Number(score.toFixed(2))}
      >
        <div className={`novelty-fill ${isNovel ? 'is-novel' : ''}`} style={{ width: `${pct}%` }} />
        <div className="novelty-threshold" title="novelty threshold (1.0)" />
      </div>
      <span className={`novelty-label ${isNovel ? 'is-novel' : ''}`}>
        <Icon name={isNovel ? 'alert' : 'check'} size={13} />
        {isNovel ? 'novel input' : 'in distribution'}
        <span className="mono"> {score.toFixed(2)}</span>
      </span>
    </div>
  );
};

// ─── Life gauge ───────────────────────────────────────────────────────────────

export const LifeGauge = ({ ttfDays }: { ttfDays: number }) => (
  <div className="gauge">
    <div className="gauge-track" aria-hidden="true">
      {GAUGE_TIERS.map(t => <span key={t.level} className={`gauge-band band-${t.level}`} />)}
      <span className="gauge-marker" style={{ left: `${gaugePosition(ttfDays)}%` }} />
    </div>
    <div className="gauge-labels" aria-hidden="true">
      {GAUGE_TIERS.map(t => <span key={t.level}>{t.label}</span>)}
    </div>
  </div>
);
