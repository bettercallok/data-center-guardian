import { SMART_FEATURES, type DriveTelemetry } from './types';

export const MAX_CSV_BYTES = 1024 * 1024; // 1 MB
export const MAX_CSV_ROWS = 1000;

export const CSV_TEMPLATE =
  'serial_number,drive_model,smart_5_raw,smart_187_raw,smart_188_raw,smart_197_raw,smart_198_raw\n' +
  'ZA1001,ST4000DM000,0,0,0,0,0\n' +
  'ZA1002,ST4000DM000,24,3,0,16,16\n' +
  'ZA1003,ST12000NM0008,320,120,600,280,280\n';

const IDENT_RE = /^[A-Za-z0-9][A-Za-z0-9 ._\-/]{0,63}$/;

/** Minimal RFC-4180 line splitter (handles quoted fields and escaped quotes). */
function splitLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur);
  return out.map(s => s.trim());
}

export interface ParseResult {
  drives: DriveTelemetry[];
  errors: string[];
}

/** Parse a fleet CSV into validated telemetry rows. Missing SMART columns default to 0. */
export function parseFleetCsv(text: string): ParseResult {
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter(l => l.trim() !== '');
  const errors: string[] = [];
  if (lines.length < 2) return { drives: [], errors: ['csv must have a header row and at least one drive.'] };

  const header = splitLine(lines[0]).map(h => h.toLowerCase());
  const col = (name: string) => header.indexOf(name);
  if (!SMART_FEATURES.some(f => col(f) >= 0)) {
    return { drives: [], errors: [`header must include at least one of: ${SMART_FEATURES.join(', ')}`] };
  }

  const drives: DriveTelemetry[] = [];
  const rows = lines.slice(1);
  if (rows.length > MAX_CSV_ROWS) errors.push(`only the first ${MAX_CSV_ROWS} rows were loaded.`);

  rows.slice(0, MAX_CSV_ROWS).forEach((line, idx) => {
    const cells = splitLine(line);
    const rowNo = idx + 2;
    const drive: DriveTelemetry = {
      smart_5_raw: 0, smart_187_raw: 0, smart_188_raw: 0, smart_197_raw: 0, smart_198_raw: 0,
    };
    for (const f of SMART_FEATURES) {
      const i = col(f);
      if (i < 0 || cells[i] === undefined || cells[i] === '') continue;
      const n = Number(cells[i]);
      if (!Number.isInteger(n) || n < 0 || n > 65535) {
        errors.push(`row ${rowNo}: ${f} must be an integer 0–65535.`);
        return;
      }
      drive[f] = n;
    }
    for (const key of ['serial_number', 'drive_model'] as const) {
      const i = col(key);
      const v = i >= 0 ? cells[i] : '';
      if (!v) continue;
      if (!IDENT_RE.test(v)) {
        errors.push(`row ${rowNo}: ${key} contains unsupported characters.`);
        return;
      }
      drive[key] = v;
    }
    drives.push(drive);
  });

  return { drives, errors: errors.slice(0, 10) };
}

export function downloadText(filename: string, text: string, mime = 'text/csv'): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
