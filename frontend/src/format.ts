import type { RiskLevel } from './types';

// Each risk tier gets an equal-width band so the short critical window stays readable.
export const GAUGE_TIERS: Array<{ level: RiskLevel; from: number; to: number; label: string }> = [
  { level: 'critical', from: 0, to: 90, label: '< 90d' },
  { level: 'high', from: 90, to: 365, label: '90d – 1y' },
  { level: 'medium', from: 365, to: 1000, label: '1 – 2.7y' },
  { level: 'low', from: 1000, to: 3000, label: '> 2.7y' },
];

export function gaugePosition(days: number): number {
  const band = 100 / GAUGE_TIERS.length;
  for (let i = 0; i < GAUGE_TIERS.length; i++) {
    const t = GAUGE_TIERS[i];
    if (days < t.to || i === GAUGE_TIERS.length - 1) {
      const frac = Math.min(1, Math.max(0, (days - t.from) / (t.to - t.from)));
      return band * i + frac * band;
    }
  }
  return 100;
}

export const RISK_ACTION: Record<RiskLevel, string> = {
  critical: 'Replace immediately and verify backups — failure is likely within weeks.',
  high: 'Schedule a replacement this quarter and increase monitoring frequency.',
  medium: 'Healthy for now. Keep monitoring and plan replacement in the next cycle.',
  low: 'Drive looks healthy. Routine monitoring is sufficient.',
};

export function humanDuration(days: number): string {
  if (days < 1) return 'less than a day';
  if (days < 60) return `about ${Math.round(days)} days`;
  if (days < 730) return `about ${Math.round(days / 30.4)} months`;
  return `about ${(days / 365.25).toFixed(1)} years`;
}
