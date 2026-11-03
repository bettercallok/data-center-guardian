// ─── Thin API client ─────────────────────────────────────────────────────────
//
// Empty string = relative URLs (same origin). Works for any port in Docker.
// Set VITE_API_URL at build time to point to a remote backend if needed.
export const API_URL: string = import.meta.env.VITE_API_URL || '';

const ADMIN_KEY_STORAGE = 'dcg-admin-key';

// The admin key lives in sessionStorage: it survives reloads but is dropped
// when the browser tab closes, and is never persisted to localStorage.
export function getAdminKey(): string {
  try {
    return sessionStorage.getItem(ADMIN_KEY_STORAGE) || '';
  } catch {
    return '';
  }
}

export function setAdminKey(key: string): void {
  try {
    if (key) sessionStorage.setItem(ADMIN_KEY_STORAGE, key);
    else sessionStorage.removeItem(ADMIN_KEY_STORAGE);
  } catch {
    /* storage unavailable (private mode) — key just won't persist */
  }
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function describeError(status: number, body: unknown): string {
  if (status === 401) return 'admin key required — set it via the "admin" button in the header.';
  if (status === 503 && typeof body === 'object' && body && 'detail' in body) {
    return String((body as { detail: unknown }).detail);
  }
  if (status === 429) return 'rate limit exceeded — slow down and retry shortly.';
  if (status === 413 && typeof body === 'object' && body && 'detail' in body) {
    return String((body as { detail: unknown }).detail);
  }
  if (status === 422 && typeof body === 'object' && body && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as { loc?: unknown[]; msg?: string };
      const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : 'input';
      return `invalid ${String(field)}: ${first.msg ?? 'validation failed'}`;
    }
  }
  return `server error (${status})`;
}

export async function apiFetch<T>(
  path: string,
  options: { method?: string; body?: unknown; admin?: boolean; signal?: AbortSignal } = {},
): Promise<T> {
  const headers: Record<string, string> = {};
  if (options.body !== undefined) headers['Content-Type'] = 'application/json';
  if (options.admin) {
    const key = getAdminKey();
    if (key) headers['X-API-Key'] = key;
  }

  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method: options.method ?? 'GET',
      headers,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
      signal: options.signal,
    });
  } catch {
    throw new ApiError(0, 'failed to connect to the inference server.');
  }

  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, describeError(res.status, body));
  }
  return (await res.json()) as T;
}

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : 'unexpected error.';
}
