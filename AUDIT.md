# Project audit — November 2026

A full review of the backend API, monitoring stack, ML/ETL pipeline, CI, container and frontend.
Each finding lists its severity, what was wrong, and how it was resolved in this change.

Baseline before the audit: 16/17 tests passed (the OOD novelty test failed), `npm run lint` reported
2 errors, and `npm audit` reported high-severity advisories. No test exercised the HTTP API.

After: 70 tests pass (API, security, alert engine, ETL safety, training), lint is clean, and production
npm dependencies have 0 known vulnerabilities.

## Security

| # | Severity | Finding | Resolution |
|---|---|---|---|
| S1 | High | `POST /api/models/probe` and `DELETE /api/models/alerts/{id}` were unauthenticated. Anyone could rebuild the monitoring baseline (CPU-heavy) or silently clear alerts. | Admin API key (`X-API-Key`, constant-time compare). Fails closed with `503` when `GUARDIAN_ADMIN_API_KEY` is unset. |
| S2 | High | Every novel inference created a **new** alert. Eviction only removed *resolved* alerts, so a flood of anomalous requests grew memory without bound. | Alerts are de-duplicated per type (repeat firings update and escalate the active alert), with a hard cap that evicts the oldest entries. |
| S3 | High | CI could publish a model trained on fake data: on any ETL error the pipeline wrote a 4-row synthetic dataset and exited 0, then CI trained on it and pushed the weights to `main`. | ETL exits non-zero on failure (fallback only with `--allow-fallback`). Training has a data-quality gate and atomic export, and CI runs the test-suite before committing a model. |
| S4 | Medium | Rate limiting only covered `/api/models/predict`, through a second `Limiter` instance separate from the app's. Behind the HF proxy every client shared one IP bucket. | One shared limiter, configurable limits on every route, and uvicorn `--proxy-headers` with configurable trusted proxies. |
| S5 | Medium | Unbounded or unvalidated query params (`limit` could be negative; `status` and `n_samples` were free-form). | `Query(ge=, le=)`, `Literal` enums and path patterns everywhere. |
| S6 | Medium | No security headers. | CSP, `X-Frame-Options: DENY`, `nosniff`, Referrer-Policy, Permissions-Policy, `Cache-Control: no-store` on the API, and a sanitised `X-Request-ID`. |
| S7 | Medium | ETL: no network timeouts, `urlretrieve` reused partial downloads, no zip path-traversal or zip-bomb guard, and no host allow-list. | Timeouts, temp-file and atomic rename, `safe_extract` with traversal and size checks, URL allow-list. |
| S8 | Medium | npm advisories (brace-expansion, browserslist, …). Docker used `npm install`, ignoring the lockfile. | `npm audit fix` (0 production vulnerabilities); Docker uses `npm ci`. |
| S9 | Low | Validation errors echoed raw input back. Unhandled exceptions had no generic handler. | Custom handlers that return field/message only, plus a generic 500 carrying the request ID. |
| S10 | Low | CSV exports were open to spreadsheet formula injection (CWE-1236). | Cells starting with `= + - @` (non-numeric) are prefixed with `'`. Identifiers are restricted by pattern. |
| S11 | Low | CORS origins were hard-coded, and `X-API-Key` was not an allowed header. | `CORS_ORIGINS` env var; header allowed. |
| S12 | Low | `extract_columns.py` hard-coded a developer's home-directory paths. | `--input-dir` / `--output` CLI arguments. |
| S13 | Info | The database URL (possibly with credentials) was logged at startup. | Credentials redacted in logs. |

## Correctness bugs

| # | Finding | Resolution |
|---|---|---|
| B1 | **Healthy drives were flagged as NOVEL.** The FAISS index measured L2 distance between raw leaf *IDs*, which are categorical labels, and uniform LHS probes almost never cover the near-zero region where healthy drives live. An all-zero drive scored 3× the novelty threshold. | Leaves are one-hot encoded, so FAISS L2² equals 2 × Hamming over trees, normalised to [0, 1]. The novelty index includes zero-inflated, log-scaled probes, and the threshold uses the 99th percentile instead of the noisy max. The false-novel rate on realistic healthy traffic is now ~0%. |
| B2 | **Extreme OOD inputs were not flagged** (failing test). Trees saturate past their last split, so `65535` lands in the same leaves as `500`. | Feature-range guard against the probe envelope. The response now includes `out_of_range_features`. |
| B3 | The legacy `POST /predict` route was broken: an untyped `request` param became a query parameter, and the handler was called without a DB session. | Rewritten on the shared inference pipeline and covered by tests. |
| B4 | `drift_summary` was only returned when drift was **not** detected. | Always returned on drift-check cycles. |
| B5 | The `AlertRecord` and `DriftSnapshot` tables existed but were never written. Alerts were lost on restart. | Alerts are persisted on fire, update and resolve, and active alerts are restored at startup. Drift snapshots are stored on each check (`GET /api/models/drift/history`). |
| B6 | CPU-bound inference and probing ran inside `async def` handlers, blocking the event loop. | Handlers are sync `def` (threadpool). Shared state is guarded by locks, and only one probe can run at a time (`409`). |
| B7 | The training split sliced `group_by` output, whose order is arbitrary, so it was not a valid random split. | Seeded shuffled split, plus a seeded booster. |
| B8 | The novelty bar's threshold marker (at 2/3 of the track) didn't match the fill scale, so the bar reached the marker at a score of 0.67. | Track rescaled to 0–1.5 with the marker at 1.0. |
| B9 | Pydantic v2 deprecation warnings (`class Config`); the ETL used the deprecated `LazyFrame.columns`. | `ConfigDict` and `collect_schema()`; polars requirement raised to ≥ 1.0. |
| B10 | An existing SQLite DB would break when new columns were added (`create_all` doesn't migrate). | Minimal additive migration for new nullable columns. |

## New features

- **Batch prediction**: `POST /api/models/predict/batch` and a **Fleet Scan** tab (client-side CSV parsing, template download, urgency-sorted results table, results CSV download).
- **Drive tracking**: optional `serial_number` / `drive_model`; `GET /api/drives/{serial}/history` with an RUL trend.
- **Exports and stats**: `GET /api/models/predictions/export` (CSV) and `GET /api/models/predictions/summary`.
- **Drift trend**: persisted snapshots plus a PSI trend chart.
- **Ops**: `/healthz` readiness probe, Docker `HEALTHCHECK`, model SHA-256 and fingerprint in `GET /api`, `X-Request-ID` / `X-Process-Time-Ms` headers, and the Wasserstein baseline shift reported after each probe.
- **CI**: a new `ci.yml` (pytest, lint, build, `npm audit` on PRs). The MLOps workflow gained concurrency control, a timeout and a test gate.

## Frontend / design (small changes)

- Page title, meta description and theme colour.
- Telemetry presets, an optional serial field, input clamping, and form submit on Enter.
- Real error messages from the API (validation, 401, 429, offline) instead of a generic one, plus an offline banner.
- An admin key panel (stored in `sessionStorage` only) and an active/all toggle for alert history.
- Polling pauses while the browser tab is hidden, and both lint errors are fixed.
- Accessibility: labelled inputs, ARIA tablist/tabs, a skip link, `:focus-visible` outlines, `prefers-reduced-motion`, and a status label next to the status dot.
- Mobile: scrollable tab bar, tighter stat values, no horizontal overflow at 390px.

## Recommended follow-ups (not done here)

1. **Drift baseline**: drift is measured against the synthetic uniform LHS distribution, so real fleet traffic (mostly zeros) will tend to register as drifted. Calibrate the drift detector on a sample of real training telemetry exported by the CT pipeline.
2. **Model registry / shadow deploy**: store each fingerprint, and block auto-promotion when the Wasserstein shift exceeds a threshold.
3. **Postgres in production**: the HF container's SQLite file is ephemeral. Point `DATABASE_URL` at a managed Postgres to keep history across restarts.
4. **Code-split the frontend**: lazy-load recharts to remove the >500 kB chunk warning.
5. **ESLint major upgrade**: one moderate dev-only advisory (`minimatch` via eslint) needs a breaking upgrade.
