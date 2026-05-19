/**
 * api.js — QuantML Frontend API Client v2
 * =========================================
 * Dual-mode: supports both legacy LSTM and market ranker pipelines.
 *
 * Design:
 *   1. All URLs are RELATIVE (/api/*) — Vite proxy forwards to FastAPI.
 *   2. In-memory request cache with TTL to avoid hammering the backend.
 *   3. In-flight deduplication — parallel callers wait on one shared Promise.
 *   4. Automatic retry with exponential back-off for transient failures.
 *   5. Debounced ticker searches to avoid firing on every keystroke.
 *   6. Mode state is stored in memory + synced with backend.
 */

// ── Cache ─────────────────────────────────────────────────────────────────────
const _cache   = new Map();    // key → { data, expiresAt }
const _inflight = new Map();   // key → Promise (dedup parallel requests)

const TTL = {
  health:   15_000,   // 15s
  models:   30_000,   // 30s
  analysis: 60_000,   // 1 min
  chart:   120_000,   // 2 min
  status:    3_000,   // 3s
  market:   30_000,   // 30s — market analysis
  backtest: 60_000,   // 1 min
  logs:      0,       // never cache — always fresh
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
const RETRY_DELAYS = [500, 1500, 4000];

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
    const isClientError = err instanceof ApiError && err.status >= 400 && err.status < 500;
    if (isClientError || retries <= 0) throw err;
    const delay = RETRY_DELAYS[RETRY_DELAYS.length - retries] ?? 4000;
    console.warn(`[api] retrying ${url} in ${delay}ms (${retries} left)…`);
    await sleep(delay);
    return _fetchWithRetry(url, opts, retries - 1);
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

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
    this.name   = 'ApiError';
    this.status = status;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// LEGACY ENDPOINTS (single-stock LSTM)
// ══════════════════════════════════════════════════════════════════════════════

/** Check if FastAPI is reachable. Never throws — returns true/false. */
export async function fetchHealth() {
  try {
    const data = await cachedFetch('health', '/api/health', {}, TTL.health, 0);
    return data?.status === 'ok';
  } catch {
    return false;
  }
}

/** Get current active mode from backend. */
export async function fetchMode() {
  try {
    return await _fetchWithRetry('/api/mode', {}, 0);
  } catch {
    return { mode: 'legacy' };
  }
}

/** Switch active mode on backend. */
export async function setMode(mode) {
  cacheClear('health');
  return _fetchWithRetry('/api/mode', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ mode }),
    timeout: 10_000,
  }, 1);
}

/** List all trained LSTM model checkpoints. */
export async function fetchModels() {
  return cachedFetch('models', '/api/models', {}, TTL.models);
}

/** Full LSTM analysis for a ticker. */
export async function fetchAnalysis(ticker) {
  const key = `analysis:${ticker}`;
  return cachedFetch(key, `/api/analyze/${ticker}`, { timeout: 120_000 }, TTL.analysis);
}

/** OHLCV + indicator chart data. */
export async function fetchChart(ticker, period = '1y') {
  const key = `chart:${ticker}:${period}`;
  return cachedFetch(key, `/api/chart/${ticker}?period=${period}`, { timeout: 60_000 }, TTL.chart);
}

/** Trigger background training for a ticker. */
export async function triggerTrain(ticker, epochs = 100, start = '2015-01-01') {
  cacheClear(`analysis:${ticker}`);
  cacheClear('models');
  return _fetchWithRetry('/api/train', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ ticker, start, epochs, seq_len: 60 }),
    timeout: 15_000,
  }, 1);
}

/** Poll training status. */
export async function fetchTrainStatus(ticker) {
  const key = `status:${ticker}`;
  const data = await cachedFetch(key, `/api/train/status/${ticker}`, {}, TTL.status, 0);
  const s = data?.status ?? '';
  if (s.startsWith('done') || s.startsWith('error')) {
    cacheSet(key, data, TTL.models);
  }
  return data;
}

// ══════════════════════════════════════════════════════════════════════════════
// MARKET ENDPOINTS (multi-stock ranker)
// ══════════════════════════════════════════════════════════════════════════════

/** Full market analysis — top longs/shorts + macro context. */
export async function fetchMarketAnalysis(universe = 'sp500_tech', topN = 5, bottomN = 5) {
  const key = `market:${universe}:${topN}:${bottomN}`;
  return cachedFetch(
    key,
    `/market/analyze?universe=${universe}&top_n=${topN}&bottom_n=${bottomN}`,
    { timeout: 120_000 },
    TTL.market,
  );
}

/** Train market ranking model. */
export async function trainMarketModel(universe = 'sp500_tech', numRounds = 100) {
  cacheClear('market:');
  return _fetchWithRetry('/market/train', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ universe, model_name: 'market_v2', num_boost_rounds: numRounds }),
    timeout: 300_000,  // 5 min — training can be slow
  }, 0);
}

/**
 * Run market backtest.
 * Returns metrics including strategy_return, buy_hold_return, alpha, sharpe,
 * max_drawdown, sortino, hit_rate, win_loss_ratio, equity_curve.
 */
export async function fetchMarketBacktest(
  universe = 'sp500_tech',
  strategyType = 'long_short',
  longN = 5,
  shortN = 5,
) {
  const key = `backtest:${universe}:${strategyType}:${longN}:${shortN}`;
  return cachedFetch(
    key,
    '/market/backtest',
    {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ universe, strategy_type: strategyType, long_n: longN, short_n: shortN }),
      timeout: 180_000,
    },
    TTL.backtest,
    0,
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// LIVE LOGS (polling)
// ══════════════════════════════════════════════════════════════════════════════

/** Fetch log records newer than `sinceTs` (epoch-ms). */
export async function fetchLogs(sinceTs = 0, limit = 200) {
  return _fetchWithRetry(`/api/logs?since=${sinceTs}&limit=${limit}`, {}, 0);
}

// ── Debounce utility ──────────────────────────────────────────────────────────
export function debounce(fn, ms = 400) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
