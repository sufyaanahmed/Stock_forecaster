/**
 * App.jsx — QuantML Forecaster v2
 * =================================
 * Dual-mode: [ Single Stock LSTM ] | [ Market Ranker ]
 *
 * Mode switching:
 *   - Tabs in navbar control the active mode
 *   - Mode is persisted to backend via POST /api/mode
 *   - Only ONE pipeline is active at runtime
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  fetchHealth, fetchModels, fetchAnalysis, fetchChart,
  cacheClear, debounce, ApiError,
  fetchMode, setMode as apiSetMode,
} from './api';
import MetricCard      from './components/MetricCard';
import SignalBadge     from './components/SignalBadge';
import ChartPanel      from './components/ChartPanel';
import TrainPanel      from './components/TrainPanel';
import ModelsList      from './components/ModelsList';
import Loader          from './components/Loader';
import MarketDashboard from './components/MarketDashboard';
import LiveLogPanel    from './components/LiveLogPanel';
import styles          from './App.module.css';

// ── Inline SVG icons ──────────────────────────────────────────────────────────
function IconActivity()   { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>; }
function IconBrain()      { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9.5 2C7 2 5 4 5 6.5c-2 0-3.5 1.5-3.5 3.5S3 13.5 5 13.5c0 2.5 2 4.5 4.5 4.5H13c3.3 0 6-2.7 6-6 0-2.5-1.5-4.5-3.5-5.2C14.9 4 13 2 11 2z"/></svg>; }
function IconZap()        { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>; }
function IconShield()     { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>; }
function IconTrending()   { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>; }
function IconRefresh()    { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>; }
function IconBarChart()   { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>; }

// ── Status indicator ──────────────────────────────────────────────────────────
function StatusDot({ online, checking }) {
  return (
    <div className={`${styles.statusDot} ${online ? styles.online : styles.offline}`}>
      <span className={styles.dotPulse} style={checking ? { animationDuration: '0.5s' } : {}} />
      {checking ? 'CONNECTING…' : online ? 'API ONLINE' : 'API OFFLINE'}
    </div>
  );
}

// ── Error display component ────────────────────────────────────────────────────
function ErrorBanner({ message, onRetry }) {
  const isOffline = message?.toLowerCase().includes('fetch') || message?.toLowerCase().includes('network');
  return (
    <div className={styles.error}>
      <span className={styles.errorIcon}>!</span>
      <div className={styles.errorContent}>
        <div className={styles.errorMsg}>
          {isOffline
            ? 'Cannot connect to the API server. Make sure the backend is running on port 8000.'
            : message}
        </div>
        {isOffline && (
          <div className={styles.errorHint}>
            Run: <code>uvicorn api.main:app --reload</code> from the project root.
          </div>
        )}
      </div>
      {onRetry && (
        <button className={styles.retryBtn} onClick={onRetry}>
          <IconRefresh /> RETRY
        </button>
      )}
    </div>
  );
}

// ── Constants ─────────────────────────────────────────────────────────────────
const QUICK_TICKERS   = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'BTC-USD'];
const HEALTH_INTERVAL = 20_000;

// Feature card data with importance & interpretation
const FEATURE_DATA = [
  { name: 'Log Return 1d',  type: 'Returns',    importance: '14.2%', desc: 'ln(Pₜ/Pₜ₋₁) — daily momentum signal' },
  { name: 'Log Return 5d',  type: 'Returns',    importance: '11.8%', desc: 'Weekly momentum proxy' },
  { name: 'Vol Ratio',      type: 'Volatility', importance: '11.7%', desc: 'Vol-10/Vol-60 — detects volatility regime shifts' },
  { name: 'Volume Ratio',   type: 'Volume',     importance: '8.2%',  desc: 'Detects unusual participation and accumulation' },
  { name: 'RSI-14',         type: 'Momentum',   importance: '8.0%',  desc: 'Wilder RSI — overbought/oversold signal' },
  { name: 'MACD Norm',      type: 'Trend',      importance: '7.6%',  desc: '(EMA12−EMA26−Signal)/Price — trend direction' },
  { name: 'BB Position',    type: 'Mean-Rev',   importance: '7.1%',  desc: '(P − SMA20) / (2σ) — band position' },
  { name: 'Vol-20',         type: 'Volatility', importance: '6.9%',  desc: 'Rolling 20d σ of returns — risk regime' },
  { name: 'RSI-28',         type: 'Momentum',   importance: '5.8%',  desc: 'Slower RSI — confirms or diverges from RSI-14' },
  { name: 'HL Ratio',       type: 'Intraday',   importance: '5.3%',  desc: 'ln(High/Low) — intraday range & volatility' },
  { name: 'BB Width',       type: 'Mean-Rev',   importance: '4.9%',  desc: '2σ/SMA20 — Bollinger squeeze indicator' },
  { name: 'Log Return 10d', type: 'Returns',    importance: '3.9%',  desc: 'Bi-weekly momentum' },
  { name: 'Vol-10',         type: 'Volatility', importance: '2.9%',  desc: 'Short-term volatility estimate' },
  { name: 'Vol-60',         type: 'Volatility', importance: '1.8%',  desc: 'Long-term volatility baseline' },
  { name: 'Log Return 2d',  type: 'Returns',    importance: '1.5%',  desc: 'ln(Pₜ/Pₜ₋₂)' },
];

// ═════════════════════════════════════════════════════════════════════════════
export default function App() {
  // ── Mode state ────────────────────────────────────────────────────────────
  const [mode,          setModeState]  = useState('legacy');  // 'legacy' | 'market'
  const [universe,      setUniverse]   = useState('sp500_tech');

  // ── API / health ──────────────────────────────────────────────────────────
  const [apiOnline,     setApiOnline]  = useState(false);
  const [apiChecking,   setApiChecking]= useState(true);

  // ── Legacy mode state ─────────────────────────────────────────────────────
  const [ticker,        setTicker]     = useState('AAPL');
  const [inputVal,      setInputVal]   = useState('AAPL');
  const [models,        setModels]     = useState([]);
  const [analysis,      setAnalysis]   = useState(null);
  const [chartData,     setChartData]  = useState(null);
  const [period,        setPeriod]     = useState('1y');
  const [loading,       setLoading]    = useState(false);
  const [chartLoading,  setChartLoading] = useState(false);
  const [error,         setError]      = useState(null);
  const [chartError,    setChartError] = useState(null);
  const [activeTab,     setActiveTab]  = useState('dashboard');

  const tickerRef = useRef(ticker);
  useEffect(() => { tickerRef.current = ticker; }, [ticker]);

  // ── Health polling ────────────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    async function check() {
      setApiChecking(true);
      const ok = await fetchHealth();
      if (mounted) { setApiOnline(ok); setApiChecking(false); }
    }
    check();
    const t = setInterval(check, HEALTH_INTERVAL);
    return () => { mounted = false; clearInterval(t); };
  }, []);

  // ── Sync mode with backend on load ────────────────────────────────────────
  useEffect(() => {
    fetchMode().then(d => { if (d?.mode) setModeState(d.mode); });
  }, []);

  // ── Mode switch ───────────────────────────────────────────────────────────
  async function handleModeSwitch(newMode) {
    setModeState(newMode);
    try { await apiSetMode(newMode); } catch { /* best-effort */ }
  }

  // ── Model list ────────────────────────────────────────────────────────────
  const loadModels = useCallback(async () => {
    try {
      const m = await fetchModels();
      setModels(Array.isArray(m) ? m : []);
    } catch (e) { console.warn('[models]', e.message); }
  }, []);
  useEffect(() => { loadModels(); }, [loadModels]);

  // ── Legacy analysis ───────────────────────────────────────────────────────
  const analyze = useCallback(async (t) => {
    if (!t) return;
    setLoading(true);
    setError(null);
    setAnalysis(null);
    try {
      const res = await fetchAnalysis(t);
      if (tickerRef.current === t) setAnalysis(res);
    } catch (e) {
      if (tickerRef.current === t) {
        setError(e instanceof ApiError ? e.message : 'Failed to connect to API. Is the backend running on port 8000?');
      }
    } finally {
      if (tickerRef.current === t) setLoading(false);
    }
  }, []);
  useEffect(() => { if (mode === 'legacy') analyze(ticker); }, [ticker, analyze, mode]);

  // ── Chart data ────────────────────────────────────────────────────────────
  const loadChart = useCallback(async (t, p) => {
    if (!t) return;
    setChartLoading(true);
    setChartError(null);
    try {
      const res = await fetchChart(t, p);
      if (tickerRef.current === t) setChartData(res);
    } catch (e) {
      if (tickerRef.current === t) setChartError(e.message);
    } finally {
      if (tickerRef.current === t) setChartLoading(false);
    }
  }, []);
  useEffect(() => { if (mode === 'legacy') loadChart(ticker, period); }, [ticker, period, loadChart, mode]);

  // ── Debounced search ──────────────────────────────────────────────────────
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const debouncedAnalyze = useCallback(debounce((t) => { setTicker(t); }, 500), []);

  function handleSearch(e) {
    e.preventDefault();
    const t = inputVal.trim().toUpperCase();
    if (t && t !== ticker) {
      cacheClear(`analysis:${t}`);
      setTicker(t);
    } else if (t === ticker) {
      cacheClear(`analysis:${ticker}`);
      analyze(ticker);
    }
  }

  function handleTickerClick(t) {
    setInputVal(t);
    setTicker(t);
  }

  function handleTrainComplete() {
    cacheClear('models');
    cacheClear(`analysis:${ticker}`);
    loadModels();
    analyze(ticker);
  }

  const hasModel = analysis?.model_trained === true;
  const a        = analysis;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={styles.root}>
      {/* ── Navbar ──────────────────────────────────────────────────────── */}
      <nav className={styles.nav}>
        <div className={styles.navBrand}>
          <div className={styles.navLogo}>
            <span className={styles.logoIcon}>◈</span>
            <span className={styles.logoText}>QuantML</span>
            <span className={styles.logoBadge}>{mode === 'market' ? 'RANKER v2' : 'LSTM v1'}</span>
          </div>

          {/* ── Mode tabs ──────────────────────────────────────────────── */}
          <div className={styles.modeTabs}>
            <button
              id="mode-legacy"
              className={`${styles.modeTab} ${mode === 'legacy' ? styles.modeTabActive : ''}`}
              onClick={() => handleModeSwitch('legacy')}
            >
              📈 Single Stock
            </button>
            <button
              id="mode-market"
              className={`${styles.modeTab} ${mode === 'market' ? styles.modeTabActiveMarket : ''}`}
              onClick={() => handleModeSwitch('market')}
            >
              <IconBarChart /> Market Ranker
            </button>
          </div>
        </div>

        {/* Search (only shown in legacy mode) */}
        {mode === 'legacy' && (
          <form className={styles.searchForm} onSubmit={handleSearch}>
            <div className={styles.searchWrap}>
              <span className={styles.searchPrefix}>$</span>
              <input
                id="ticker-search"
                className={styles.searchInput}
                value={inputVal}
                onChange={e => setInputVal(e.target.value.toUpperCase())}
                placeholder="TICKER"
                autoComplete="off"
              />
              <button type="submit" className={styles.searchBtn}>ANALYZE</button>
            </div>
          </form>
        )}

        <div className={styles.navRight}>
          <StatusDot online={apiOnline} checking={apiChecking} />
        </div>
      </nav>

      {/* ── API offline banner ──────────────────────────────────────────── */}
      {!apiOnline && !apiChecking && (
        <div className={styles.offlineBanner}>
          <span>⚠</span>
          Backend API is offline. Run{' '}
          <code>uvicorn api.main:app --reload</code>{' '}
          from the project root directory, then refresh.
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════
          MARKET RANKER MODE
          ══════════════════════════════════════════════════════════════════ */}
      {mode === 'market' && (
        <div className={styles.marketContainer}>
          <MarketDashboard universe={universe} onUniverseChange={setUniverse} />
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════
          LEGACY SINGLE-STOCK MODE
          ══════════════════════════════════════════════════════════════════ */}
      {mode === 'legacy' && (
        <>
          {/* Ticker ribbon */}
          <div className={styles.tickerRibbon}>
            <div className={styles.ribbonInner}>
              {QUICK_TICKERS.map(t => (
                <button
                  key={t}
                  className={`${styles.quickTicker} ${ticker === t ? styles.activeQT : ''}`}
                  onClick={() => handleTickerClick(t)}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Layout */}
          <div className={styles.layout}>
            {/* Sidebar */}
            <aside className={styles.sidebar}>
              <div className={styles.sideSection}>
                <div className={styles.sideTitle}>TRAINED MODELS</div>
                <ModelsList models={models} activeTicker={ticker} onSelect={handleTickerClick} />
              </div>

              <div className={styles.sideSection}>
                <div className={styles.sideTitle}>ARCHITECTURE</div>
                <div className={styles.archTable}>
                  {[
                    ['Type',      'LSTM + Attn'],
                    ['Layers',    '2'],
                    ['Hidden',    '128'],
                    ['Seq Len',   '60 days'],
                    ['Features',  '15'],
                    ['Params',    '278,978'],
                    ['Target',    'Log Return T+1'],
                    ['Optimizer', 'AdamW'],
                    ['Scheduler', 'Cosine LR'],
                    ['Dropout',   '0.3'],
                  ].map(([k, v]) => (
                    <div key={k} className={styles.archRow}>
                      <span className={styles.archKey}>{k}</span>
                      <span className={styles.archVal}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </aside>

            {/* Main */}
            <main className={styles.main}>
              {/* Tabs */}
              <div className={styles.mainTabs}>
                {[['dashboard', 'DASHBOARD'], ['train', 'TRAIN MODEL'], ['logs', 'LIVE LOGS']].map(([id, label]) => (
                  <button
                    key={id}
                    id={`tab-${id}`}
                    className={`${styles.mainTab} ${activeTab === id ? styles.mainTabActive : ''}`}
                    onClick={() => setActiveTab(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {/* Train tab */}
              {activeTab === 'train' ? (
                <div className={styles.trainView}>
                  <TrainPanel ticker={ticker} onTrainComplete={handleTrainComplete} />
                </div>

              /* Live Logs tab */
              ) : activeTab === 'logs' ? (
                <div className={styles.trainView}>
                  <div className={styles.sectionTitle}>LIVE SYSTEM LOGS</div>
                  <LiveLogPanel maxLines={300} />
                </div>

              /* Dashboard tab */
              ) : (
                <>
                  <div className={styles.tickerHeader}>
                    <div className={styles.thLeft}>
                      <div className={styles.thTicker}>{ticker}</div>
                      {a?.latest_close ? (
                        <div className={styles.thPrice}>${Number(a.latest_close).toFixed(2)}</div>
                      ) : null}
                      {a?.cache_hit && <div className={styles.cacheTag}>CACHED</div>}
                    </div>
                    {!loading && !error && !hasModel && a !== null && (
                      <div className={styles.noModelBanner}>
                        No trained model for <strong>{ticker}</strong>.
                        Switch to <strong>Train Model</strong> tab to train one.
                      </div>
                    )}
                  </div>

                  {loading ? (
                    <Loader text={`ANALYZING ${ticker}…`} />
                  ) : error ? (
                    <ErrorBanner message={error} onRetry={() => { cacheClear(`analysis:${ticker}`); analyze(ticker); }} />
                  ) : (
                    <>
                      {/* Signal badge */}
                      {hasModel && a && (
                        <div className={`${styles.section} animate-slide-up`} style={{ animationDelay: '0ms' }}>
                          <div className={styles.sectionTitle}>NEXT-DAY SIGNAL</div>
                          <SignalBadge
                            direction={a.predicted_direction}
                            returnPct={a.predicted_return_pct}
                            confidence={a.confidence_score}
                          />
                        </div>
                      )}

                      {/* Metrics grid */}
                      {hasModel && a && (
                        <div className={`${styles.section} animate-slide-up`} style={{ animationDelay: '60ms' }}>
                          <div className={styles.sectionTitle}>MODEL PERFORMANCE</div>
                          <div className={styles.metricsGrid}>
                            <MetricCard
                              label="IC (Spearman)"
                              value={Number(a.ic).toFixed(4)}
                              sub={a.ic_significant ? '[sig] p<0.05' : '[not sig]'}
                              color={a.ic > 0.05 ? 'green' : a.ic > 0 ? 'accent' : 'red'}
                              icon={<IconActivity />}
                            />
                            <MetricCard
                              label="Direction Accuracy"
                              value={`${(Number(a.direction_accuracy) * 100).toFixed(1)}%`}
                              sub="baseline: 50.0%"
                              color={a.direction_accuracy > 0.52 ? 'green' : a.direction_accuracy > 0.5 ? 'accent' : 'red'}
                              icon={<IconZap />}
                            />
                            <MetricCard
                              label="Sharpe Ratio"
                              value={Number(a.sharpe).toFixed(3)}
                              sub="annualised"
                              color={a.sharpe > 1 ? 'green' : a.sharpe > 0 ? 'amber' : 'red'}
                              icon={<IconBrain />}
                            />
                            <MetricCard
                              label="Max Drawdown"
                              value={`${(Number(a.max_drawdown) * 100).toFixed(1)}%`}
                              sub="peak-to-trough"
                              color={a.max_drawdown > -0.15 ? 'green' : a.max_drawdown > -0.3 ? 'amber' : 'red'}
                              icon={<IconShield />}
                            />
                            <MetricCard
                              label="Strategy Return"
                              value={`${(Number(a.total_return) * 100).toFixed(1)}%`}
                              sub="test period"
                              color={a.total_return > 0 ? 'green' : 'red'}
                              icon={<IconTrending />}
                            />
                            <MetricCard
                              label="Alpha vs B&H"
                              value={`${((a.total_return - a.buy_hold_return) * 100).toFixed(1)}%`}
                              sub={`B&H: ${(Number(a.buy_hold_return) * 100).toFixed(1)}%`}
                              color={(a.total_return - a.buy_hold_return) > 0 ? 'green' : 'red'}
                              icon={<IconActivity />}
                            />
                          </div>
                        </div>
                      )}

                      {/* Chart */}
                      <div className={`${styles.section} animate-slide-up`} style={{ animationDelay: '120ms' }}>
                        <div className={styles.sectionTitle}>TECHNICAL ANALYSIS</div>
                        {chartLoading ? (
                          <Loader text="LOADING CHART…" />
                        ) : chartError ? (
                          <div className={styles.chartErrorBox}>
                            <span>Chart unavailable: {chartError}</span>
                            <button
                              className={styles.retryBtnSmall}
                              onClick={() => { cacheClear(`chart:${ticker}:${period}`); loadChart(ticker, period); }}
                            >
                              <IconRefresh /> Retry
                            </button>
                          </div>
                        ) : (
                          <ChartPanel
                            chartData={chartData}
                            period={period}
                            onPeriodChange={p => { cacheClear(`chart:${ticker}:${p}`); setPeriod(p); }}
                          />
                        )}
                      </div>

                      {/* Feature engineering table — with importance */}
                      {hasModel && (
                        <div className={`${styles.section} animate-slide-up`} style={{ animationDelay: '180ms' }}>
                          <div className={styles.sectionTitle}>FEATURE IMPORTANCE (15 FEATURES)</div>
                          <div className={styles.featGrid}>
                            {FEATURE_DATA.map(f => (
                              <div key={f.name} className={styles.featCard}>
                                <div className={styles.featHeader}>
                                  <div className={styles.featName}>{f.name}</div>
                                  <div className={styles.featImportance}>{f.importance}</div>
                                </div>
                                <div className={styles.featImportanceBar}>
                                  <div
                                    className={styles.featImportanceFill}
                                    style={{ width: f.importance }}
                                  />
                                </div>
                                <div className={styles.featType}>{f.type}</div>
                                <div className={styles.featDesc}>{f.desc}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </>
              )}
            </main>
          </div>
        </>
      )}

      {/* Footer */}
      <footer className={styles.footer}>
        <span>QuantML Forecaster · {mode === 'market' ? 'LambdaRank · Multi-Stock' : 'LSTM · PyTorch · Next-Day Return'}</span>
        <span>Not financial advice. For research use only.</span>
      </footer>
    </div>
  );
}
