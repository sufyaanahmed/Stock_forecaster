/**
 * api.js — QuantML Frontend API Client
 * =====================================
 * Design principles:
 *   1. All URLs are RELATIVE (/api/*) — Vite proxy forwards to FastAPI.
 *      This eliminates CORS issues regardless of which port Vite picks.
 *   2. In-memory request cache with TTL to avoid hammering the backend.
 *   3. In-flight deduplication — parallel callers wait on one shared Promise.
 *   4. Automatic retry with exponential back-off for transient failures.
 *   5. Debounced ticker searches to avoid firing on every keystroke.
 *   6. Meaningful error messages surfaced to the UI.
 */

// ── Cache ─────────────────────────────────────────────────────────────────────
const _cache = new Map();      // key → { data, expiresAt }
const _inflight = new Map();   // key → Promise (dedup parallel requests)

const TTL = {
  health:   15_000,   // 15s  — health changes rarely
  models:   30_000,   // 30s  — model list changes only after training
  analysis: 60_000,   // 1 min — analysis is expensive (full data fetch)
  chart:   120_000,   // 2 min — chart data doesn't change intra-minute
  status:    3_000,   // 3s   — training status must be fresh
};

function cacheGet(key) {
  const entry = _cache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) { _cache.delete(key); return null; }
  return entry.data;
}

function cacheSet(key, data, ttl) {
  _cache.set(key, { data, expiresAt: Date.now() + ttl });
}

export function cacheClear(keyPrefix) {
  for (const k of _cache.keys()) {
    if (k.startsWith(keyPrefix)) _cache.delete(k);
  }
}

// ── Core fetch with retry + dedup ─────────────────────────────────────────────
const RETRY_DELAYS = [500, 1500, 4000];   // ms between retries

async function _fetchWithRetry(url, opts = {}, retries = 2) {
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), opts.timeout ?? 90_000);

  try {
    const res = await fetch(url, { ...opts, signal: controller.signal });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = body.detail ?? body.message ?? detail;
      } catch {}
      throw new ApiError(detail, res.status);
    }

    return await res.json();
  } catch (err) {
    clearTimeout(timeoutId);

    // Don't retry client errors (4xx) or aborts from our own timeout
    const isClientError = err instanceof ApiError && err.status >= 400 && err.status < 500;
    if (isClientError || retries <= 0) throw err;

    // Network error or 5xx → wait then retry
    const delay = RETRY_DELAYS[RETRY_DELAYS.length - retries] ?? 4000;
    console.warn(`[api] retrying ${url} in ${delay}ms (${retries} left)…`);
    await sleep(delay);
    return _fetchWithRetry(url, opts, retries - 1);
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/** Fetch with cache + in-flight deduplication */
async function cachedFetch(key, url, opts, ttl, retries = 2) {
  const cached = cacheGet(key);
  if (cached !== null) return cached;

  if (_inflight.has(key)) return _inflight.get(key);

  const promise = _fetchWithRetry(url, opts, retries)
    .then(data => {
      cacheSet(key, data, ttl);
      _inflight.delete(key);
      return data;
    })
    .catch(err => {
      _inflight.delete(key);
      throw err;
    });

  _inflight.set(key, promise);
  return promise;
}

// ── Public error type ─────────────────────────────────────────────────────────
export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name  = 'ApiError';
    this.status = status;
  }
}

// ── API calls ─────────────────────────────────────────────────────────────────

/** Check if FastAPI is reachable. Never throws — returns true/false. */
export async function fetchHealth() {
  try {
    const data = await cachedFetch('health', '/api/health', {}, TTL.health, 0);
    return data?.status === 'ok';
  } catch {
    return false;
  }
}

/** List all trained model checkpoints. */
export async function fetchModels() {
  return cachedFetch('models', '/api/models', {}, TTL.models);
}

/**
 * Full analysis for a ticker.
 * Expensive: downloads data + runs inference. Cached for 1 min.
 */
export async function fetchAnalysis(ticker) {
  const key = `analysis:${ticker}`;
  return cachedFetch(
    key,
    `/api/analyze/${ticker}`,
    { timeout: 120_000 },
    TTL.analysis,
  );
}

/**
 * OHLCV + indicator chart data.
 * Cached per ticker+period for 2 minutes.
 */
export async function fetchChart(ticker, period = '1y') {
  const key = `chart:${ticker}:${period}`;
  return cachedFetch(
    key,
    `/api/chart/${ticker}?period=${period}`,
    { timeout: 60_000 },
    TTL.chart,
  );
}

/** Trigger background training for a ticker. Clears relevant caches. */
export async function triggerTrain(ticker, epochs = 100, start = '2015-01-01') {
  cacheClear(`analysis:${ticker}`);
  cacheClear('models');

  const data = await _fetchWithRetry('/api/train', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ ticker, start, epochs, seq_len: 60 }),
    timeout: 15_000,
  }, 1);

  return data;
}

/**
 * Poll training status. Short TTL (3s) so UI stays responsive.
 * Once status is 'done' or 'error', cache for 30s to stop hammering.
 */
export async function fetchTrainStatus(ticker) {
  const key = `status:${ticker}`;
  const data = await cachedFetch(key, `/api/train/status/${ticker}`, {}, TTL.status, 0);

  // Extend cache TTL once terminal state reached
  const s = data?.status ?? '';
  if (s.startsWith('done') || s.startsWith('error')) {
    cacheSet(key, data, TTL.models);   // cache for 30s, no more polling needed
  }

  return data;
}

// ── Debounce utility (exported for use in components) ────────────────────────
export function debounce(fn, ms = 400) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
