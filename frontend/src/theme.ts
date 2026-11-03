// ─── Theme + chart styling ───────────────────────────────────────────────────

export type ThemeMode = 'light' | 'dark';

const THEME_KEY = 'dcg-theme';

/** Stored preference, else the OS preference. */
export function initialTheme(): ThemeMode {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch { /* storage unavailable */ }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function applyTheme(mode: ThemeMode, persist = false): void {
  document.documentElement.dataset.theme = mode;
  if (!persist) return;
  try { localStorage.setItem(THEME_KEY, mode); } catch { /* ignore */ }
}

// Shared Recharts props so every chart matches the design system in both themes.
export const axisProps = {
  stroke: 'var(--border-strong)',
  tick: { fontSize: 11, fill: 'var(--text-3)' },
  tickLine: false,
  axisLine: false,
} as const;

export const gridProps = { strokeDasharray: '3 4', stroke: 'var(--grid)', vertical: false } as const;

export const tooltipProps = {
  contentStyle: {
    background: 'var(--surface-raised)',
    border: '1px solid var(--border)',
    borderRadius: 10,
    boxShadow: 'var(--shadow-lg)',
    fontSize: 12,
    color: 'var(--text-1)',
  },
  labelStyle: { color: 'var(--text-3)', marginBottom: 4 },
  itemStyle: { color: 'var(--text-1)' },
  cursor: { fill: 'var(--hover)' },
} as const;

export const legendProps = {
  iconType: 'circle' as const,
  iconSize: 8,
  wrapperStyle: { fontSize: 12, color: 'var(--text-2)', paddingTop: 8 },
};

export const SERIES_COLORS = ['var(--accent)', 'var(--risk-low)', 'var(--risk-medium)', 'var(--risk-high)', 'var(--violet)'];
