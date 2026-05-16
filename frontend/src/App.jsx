import { useState, useEffect, useCallback } from 'react';
import { fetchHealth, fetchModels, fetchAnalysis, fetchChart } from './api';
import MetricCard from './components/MetricCard';
import SignalBadge from './components/SignalBadge';
import ChartPanel from './components/ChartPanel';
import TrainPanel from './components/TrainPanel';
import ModelsList from './components/ModelsList';
import Loader from './components/Loader';
import styles from './App.module.css';

// --- Icons (inline SVG to avoid extra dep) ---
function IconActivity() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>;
}
function IconBrain() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9.5 2C7 2 5 4 5 6.5c-2 0-3.5 1.5-3.5 3.5S3 13.5 5 13.5c0 2.5 2 4.5 4.5 4.5H13c3.3 0 6-2.7 6-6 0-2.5-1.5-4.5-3.5-5.2C14.9 4 13 2 11 2z"/></svg>;
}
function IconZap() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>;
}
function IconShield() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>;
}
function IconTrendingUp() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>;
}

function StatusDot({ online }) {
  return (
    <div className={`${styles.statusDot} ${online ? styles.online : styles.offline}`}>
      <span className={styles.dotPulse} />
      {online ? 'API ONLINE' : 'API OFFLINE'}
    </div>
  );
}

// Ticker search bar
const QUICK_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'BTC-USD'];

export default function App() {
  const [apiOnline,   setApiOnline]   = useState(false);
  const [ticker,      setTicker]      = useState('AAPL');
  const [inputVal,    setInputVal]    = useState('AAPL');
  const [models,      setModels]      = useState([]);
  const [analysis,    setAnalysis]    = useState(null);
  const [chartData,   setChartData]   = useState(null);
  const [period,      setPeriod]      = useState('1y');
  const [loading,     setLoading]     = useState(false);
  const [chartLoading,setChartLoading]= useState(false);
  const [error,       setError]       = useState(null);
  const [activeTab,   setActiveTab]   = useState('dashboard'); // 'dashboard' | 'train'

  // Health check
  useEffect(() => {
    (async () => {
      try { await fetchHealth(); setApiOnline(true); }
      catch { setApiOnline(false); }
    })();
    const t = setInterval(async () => {
      try { await fetchHealth(); setApiOnline(true); }
      catch { setApiOnline(false); }
    }, 15000);
    return () => clearInterval(t);
  }, []);

  // Load models list
  const loadModels = useCallback(async () => {
    try { setModels(await fetchModels()); }
    catch {}
  }, []);

  useEffect(() => { loadModels(); }, [loadModels]);

  // Auto-analyze on ticker change
  const analyze = useCallback(async (t) => {
    setLoading(true);
    setError(null);
    setAnalysis(null);
    try {
      const res = await fetchAnalysis(t);
      setAnalysis(res);
    } catch(e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { analyze(ticker); }, [ticker, analyze]);

  // Load chart data
  const loadChart = useCallback(async (t, p) => {
    setChartLoading(true);
    try {
      const res = await fetchChart(t, p);
      setChartData(res);
    } catch {}
    finally { setChartLoading(false); }
  }, []);

  useEffect(() => { loadChart(ticker, period); }, [ticker, period, loadChart]);

  function handleSearch(e) {
    e.preventDefault();
    const t = inputVal.trim().toUpperCase();
    if (t) { setTicker(t); }
  }

  const hasModel = analysis?.model_trained;
  const a = analysis;

  return (
    <div className={styles.root}>
      {/* ── Navbar ───────────────────────── */}
      <nav className={styles.nav}>
        <div className={styles.navBrand}>
          <div className={styles.navLogo}>
            <span className={styles.logoIcon}>◈</span>
            <span className={styles.logoText}>QuantML</span>
            <span className={styles.logoBadge}>LSTM v1</span>
          </div>
          <div className={styles.navModel}>
            <span className={styles.modelTag}>2L LSTM · ATT · SEQ-60</span>
          </div>
        </div>

        {/* Search */}
        <form className={styles.searchForm} onSubmit={handleSearch}>
          <div className={styles.searchWrap}>
            <span className={styles.searchPrefix}>$</span>
            <input
              className={styles.searchInput}
              value={inputVal}
              onChange={e => setInputVal(e.target.value.toUpperCase())}
              placeholder="TICKER"
              id="ticker-search"
            />
            <button type="submit" className={styles.searchBtn}>ANALYZE</button>
          </div>
        </form>

        <div className={styles.navRight}>
          <StatusDot online={apiOnline} />
        </div>
      </nav>

      {/* ── Ticker Ribbon ─────────────────── */}
      <div className={styles.tickerRibbon}>
        <div className={styles.ribbonInner}>
          {[...QUICK_TICKERS, ...QUICK_TICKERS].map((t, i) => (
            <button
              key={i}
              className={`${styles.quickTicker} ${ticker === t && i < QUICK_TICKERS.length ? styles.activeQT : ''}`}
              onClick={() => { setTicker(t); setInputVal(t); }}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* ── Main Layout ───────────────────── */}
      <div className={styles.layout}>
        {/* Sidebar */}
        <aside className={styles.sidebar}>
          <div className={styles.sideSection}>
            <div className={styles.sideTitle}>TRAINED MODELS</div>
            <ModelsList models={models} activeTicker={ticker} onSelect={(t) => { setTicker(t); setInputVal(t); }} />
          </div>

          <div className={styles.sideSection}>
            <div className={styles.sideTitle}>ARCHITECTURE</div>
            <div className={styles.archTable}>
              {[
                ['Type',       'LSTM + Attn'],
                ['Layers',     '2'],
                ['Hidden',     '128'],
                ['Seq Len',    '60 days'],
                ['Features',   '15'],
                ['Params',     '278,978'],
                ['Target',     'Log Return T+1'],
                ['Optimizer',  'AdamW'],
                ['Scheduler',  'Cosine LR'],
              ].map(([k, v]) => (
                <div key={k} className={styles.archRow}>
                  <span className={styles.archKey}>{k}</span>
                  <span className={styles.archVal}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main className={styles.main}>
          {/* Tabs */}
          <div className={styles.mainTabs}>
            {['DASHBOARD', 'TRAIN MODEL'].map((t, i) => {
              const id = i === 0 ? 'dashboard' : 'train';
              return (
                <button
                  key={id}
                  className={`${styles.mainTab} ${activeTab === id ? styles.mainTabActive : ''}`}
                  onClick={() => setActiveTab(id)}
                  id={`tab-${id}`}
                >
                  {t}
                </button>
              );
            })}
          </div>

          {activeTab === 'train' ? (
            <div className={styles.trainView}>
              <TrainPanel ticker={ticker} onTrainComplete={() => { loadModels(); analyze(ticker); }} />
            </div>
          ) : (
            <>
              {/* Ticker Header */}
              <div className={styles.tickerHeader}>
                <div className={styles.thLeft}>
                  <div className={styles.thTicker}>{ticker}</div>
                  {a?.latest_close && (
                    <div className={styles.thPrice}>${a.latest_close?.toFixed(2)}</div>
                  )}
                </div>
                {!hasModel && !loading && (
                  <div className={styles.noModelBanner}>
                    No model trained yet. Go to <strong>Train Model</strong> tab.
                  </div>
                )}
              </div>

              {loading ? (
                <Loader text={`ANALYZING ${ticker}...`} />
              ) : error ? (
                <div className={styles.error}>
                  <span className={styles.errorIcon}>!</span> {error}
                </div>
              ) : (
                <>
                  {/* Signal */}
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

                  {/* IC + Stats grid */}
                  {hasModel && a && (
                    <div className={`${styles.section} animate-slide-up`} style={{ animationDelay: '60ms' }}>
                      <div className={styles.sectionTitle}>MODEL PERFORMANCE</div>
                      <div className={styles.metricsGrid}>
                        <MetricCard
                          label="IC (Spearman)"
                          value={a.ic?.toFixed(4)}
                          sub={a.ic_significant ? '[sig] p<0.05' : '[not sig]'}
                          color={a.ic > 0.05 ? 'green' : a.ic > 0 ? 'accent' : 'red'}
                          icon={<IconActivity />}
                        />
                        <MetricCard
                          label="Direction Accuracy"
                          value={`${(a.direction_accuracy * 100)?.toFixed(1)}%`}
                          sub="baseline: 50.0%"
                          color={a.direction_accuracy > 0.52 ? 'green' : a.direction_accuracy > 0.5 ? 'accent' : 'red'}
                          icon={<IconZap />}
                        />
                        <MetricCard
                          label="Sharpe Ratio"
                          value={a.sharpe?.toFixed(3)}
                          sub="annualised"
                          color={a.sharpe > 1 ? 'green' : a.sharpe > 0 ? 'amber' : 'red'}
                          icon={<IconBrain />}
                        />
                        <MetricCard
                          label="Max Drawdown"
                          value={`${(a.max_drawdown * 100)?.toFixed(1)}%`}
                          sub="peak-to-trough"
                          color={a.max_drawdown > -0.15 ? 'green' : a.max_drawdown > -0.3 ? 'amber' : 'red'}
                          icon={<IconShield />}
                        />
                        <MetricCard
                          label="Strategy Return"
                          value={`${(a.total_return * 100)?.toFixed(1)}%`}
                          sub="test period"
                          color={a.total_return > 0 ? 'green' : 'red'}
                          icon={<IconTrendingUp />}
                        />
                        <MetricCard
                          label="Alpha vs B&H"
                          value={`${((a.total_return - a.buy_hold_return) * 100)?.toFixed(1)}%`}
                          sub={`B&H: ${(a.buy_hold_return * 100)?.toFixed(1)}%`}
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
                      <Loader text="LOADING CHART..." />
                    ) : (
                      <ChartPanel
                        chartData={chartData}
                        period={period}
                        onPeriodChange={(p) => setPeriod(p)}
                      />
                    )}
                  </div>

                  {/* Features info */}
                  {hasModel && (
                    <div className={`${styles.section} animate-slide-up`} style={{ animationDelay: '180ms' }}>
                      <div className={styles.sectionTitle}>FEATURE ENGINEERING</div>
                      <div className={styles.featGrid}>
                        {[
                          { name: 'Log Return 1d',  type: 'Returns',     desc: 'ln(P_t / P_{t-1})' },
                          { name: 'Log Return 2d',  type: 'Returns',     desc: 'ln(P_t / P_{t-2})' },
                          { name: 'Log Return 5d',  type: 'Returns',     desc: 'Weekly momentum' },
                          { name: 'Log Return 10d', type: 'Returns',     desc: 'Bi-weekly momentum' },
                          { name: 'Vol-10',         type: 'Volatility',  desc: 'Rolling 10d std of returns' },
                          { name: 'Vol-20',         type: 'Volatility',  desc: 'Rolling 20d std of returns' },
                          { name: 'Vol-60',         type: 'Volatility',  desc: 'Rolling 60d std of returns' },
                          { name: 'Vol Ratio',      type: 'Volatility',  desc: 'Vol-10 / Vol-60 (regime)' },
                          { name: 'RSI-14',         type: 'Momentum',    desc: 'Wilder RSI, 14-period EWM' },
                          { name: 'RSI-28',         type: 'Momentum',    desc: 'Slower RSI signal' },
                          { name: 'MACD Norm',      type: 'Trend',       desc: '(EMA12-EMA26-Sig)/Price' },
                          { name: 'BB Position',    type: 'Mean-Rev',    desc: '(P - SMA20) / (2σ)' },
                          { name: 'BB Width',       type: 'Mean-Rev',    desc: '2σ / SMA20' },
                          { name: 'Volume Ratio',   type: 'Volume',      desc: 'ln(Vol / EMA20_Vol)' },
                          { name: 'HL Ratio',       type: 'Intraday',    desc: 'ln(High / Low)' },
                        ].map(f => (
                          <div key={f.name} className={styles.featCard}>
                            <div className={styles.featName}>{f.name}</div>
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

      {/* Footer */}
      <footer className={styles.footer}>
        <span>QuantML Forecaster · LSTM · PyTorch · Next-Day Log Return Prediction</span>
        <span>Not financial advice. For research use only.</span>
      </footer>
    </div>
  );
}
