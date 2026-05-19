/**
 * MarketDashboard.jsx
 * ====================
 * Market Ranker mode dashboard.
 *
 * Shows:
 *   - Top Longs / Top Shorts
 *   - Macro Regime + Factor Exposures
 *   - Strategy Metrics (IC, Sharpe, alpha vs benchmark)
 *   - Backtest equity curve + benchmark comparison
 *   - Strategy Accuracy section
 *   - Live Logs
 */

import { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import { fetchMarketAnalysis, trainMarketModel, fetchMarketBacktest, cacheClear } from '../api';
import MetricCard from './MetricCard';
import LiveLogPanel from './LiveLogPanel';
import Loader from './Loader';
import styles from './MarketDashboard.module.css';

// ── Icons ─────────────────────────────────────────────────────────────────────
function IconTrend()    { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>; }
function IconDown()     { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>; }
function IconActivity() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>; }
function IconShield()   { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>; }
function IconZap()      { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>; }
function IconRefresh()  { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>; }

const UNIVERSE_LABELS = {
  sp500_tech: 'S&P 500 Tech',
  nasdaq_100: 'NASDAQ 100',
  nifty_50:   'NIFTY 50',
};

// ── Custom tooltip for equity curve ──────────────────────────────────────────
function EquityTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className={styles.chartTooltip}>
      <div className={styles.chartTooltipDate}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
          {p.name}: {(p.value * 100).toFixed(2)}%
        </div>
      ))}
    </div>
  );
}

export default function MarketDashboard({ universe = 'sp500_tech', onUniverseChange }) {
  const [analysis,   setAnalysis]   = useState(null);
  const [backtest,   setBacktest]   = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [btLoading,  setBtLoading]  = useState(false);
  const [training,   setTraining]   = useState(false);
  const [error,      setError]      = useState(null);
  const [btError,    setBtError]    = useState(null);
  const [activeTab,  setActiveTab]  = useState('overview');
  const [stratType,  setStratType]  = useState('long_short');

  // ── Fetch market analysis ─────────────────────────────────────────────────
  const loadAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchMarketAnalysis(universe, 5, 5);
      setAnalysis(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [universe]);

  useEffect(() => { loadAnalysis(); }, [loadAnalysis]);

  // ── Fetch backtest ────────────────────────────────────────────────────────
  const loadBacktest = useCallback(async () => {
    setBtLoading(true);
    setBtError(null);
    try {
      const data = await fetchMarketBacktest(universe, stratType, 5, 5);
      setBacktest(data);
    } catch (e) {
      setBtError(e.message);
    } finally {
      setBtLoading(false);
    }
  }, [universe, stratType]);

  // ── Train model ───────────────────────────────────────────────────────────
  async function handleTrain() {
    setTraining(true);
    try {
      await trainMarketModel(universe, 100);
      cacheClear('market:');
      cacheClear('backtest:');
      await loadAnalysis();
    } catch (e) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  }

  // ── Derived metrics ───────────────────────────────────────────────────────
  const m = analysis?.metrics ?? {};
  const bt = backtest?.metrics ?? {};
  const equityCurve = backtest?.equity_curve ?? [];

  // Alpha vs benchmark
  const stratReturn = bt.total_return ?? m.strategy_return ?? null;
  const bhReturn    = bt.buy_hold_return ?? m.buy_hold_return ?? null;
  const alpha       = stratReturn != null && bhReturn != null ? stratReturn - bhReturn : null;

  return (
    <div className={styles.dashboard}>
      {/* ── Top toolbar ─────────────────────────────────────────────────── */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <select
            className={styles.universeSelect}
            value={universe}
            onChange={e => onUniverseChange?.(e.target.value)}
          >
            {Object.entries(UNIVERSE_LABELS).map(([val, label]) => (
              <option key={val} value={val}>{label}</option>
            ))}
          </select>
          <div className={styles.modeBadge}>MARKET RANKER</div>
        </div>
        <div className={styles.toolbarRight}>
          <button
            className={styles.refreshBtn}
            onClick={() => { cacheClear('market:'); loadAnalysis(); }}
            disabled={loading}
          >
            <IconRefresh /> REFRESH
          </button>
          <button
            className={`${styles.trainBtn} ${training ? styles.trainingActive : ''}`}
            onClick={handleTrain}
            disabled={training}
          >
            {training ? '⟳ TRAINING…' : '⚡ TRAIN MODEL'}
          </button>
        </div>
      </div>

      {/* ── Sub-tabs ────────────────────────────────────────────────────── */}
      <div className={styles.subTabs}>
        {[
          ['overview',  'OVERVIEW'],
          ['backtest',  'BACKTEST'],
          ['accuracy',  'ACCURACY'],
          ['logs',      'LIVE LOGS'],
        ].map(([id, label]) => (
          <button
            key={id}
            className={`${styles.subTab} ${activeTab === id ? styles.subTabActive : ''}`}
            onClick={() => { setActiveTab(id); if (id === 'backtest' && !backtest) loadBacktest(); }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          OVERVIEW TAB
          ══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'overview' && (
        <>
          {loading ? <Loader text="ANALYZING MARKET…" /> : error ? (
            <div className={styles.errorBox}>
              <span>⚠ {error}</span>
              <button className={styles.retrySmall} onClick={loadAnalysis}><IconRefresh /> Retry</button>
            </div>
          ) : (
            <>
              {/* Long/Short picks */}
              <div className={styles.picksGrid}>
                <div className={styles.picksPanel}>
                  <div className={styles.pickTitle} style={{ color: 'var(--green)' }}>
                    ▲ TOP LONGS
                  </div>
                  <div className={styles.picksList}>
                    {(analysis?.long ?? []).length > 0
                      ? analysis.long.map((t, i) => (
                          <div key={t} className={styles.pickRow}>
                            <span className={styles.pickRank}>#{i + 1}</span>
                            <span className={styles.pickTicker}>{t}</span>
                            <span className={styles.pickBadge} style={{ color: 'var(--green)', borderColor: 'rgba(0,255,136,0.3)' }}>LONG</span>
                          </div>
                        ))
                      : <div className={styles.emptyPicks}>No data yet — train a model first</div>
                    }
                  </div>
                </div>

                <div className={styles.picksPanel}>
                  <div className={styles.pickTitle} style={{ color: 'var(--red)' }}>
                    ▼ TOP SHORTS
                  </div>
                  <div className={styles.picksList}>
                    {(analysis?.short ?? []).length > 0
                      ? analysis.short.map((t, i) => (
                          <div key={t} className={styles.pickRow}>
                            <span className={styles.pickRank}>#{i + 1}</span>
                            <span className={styles.pickTicker}>{t}</span>
                            <span className={styles.pickBadge} style={{ color: 'var(--red)', borderColor: 'rgba(255,51,102,0.3)' }}>SHORT</span>
                          </div>
                        ))
                      : <div className={styles.emptyPicks}>No shorts — train a model first</div>
                    }
                  </div>
                </div>

                {/* Macro context */}
                <div className={styles.macroPanel}>
                  <div className={styles.pickTitle} style={{ color: 'var(--accent)' }}>
                    ◎ MACRO REGIME
                  </div>
                  {analysis?.macro_context && Object.keys(analysis.macro_context).length > 0 ? (
                    <div className={styles.macroGrid}>
                      {Object.entries(analysis.macro_context).map(([k, v]) => (
                        <div key={k} className={styles.macroRow}>
                          <span className={styles.macroKey}>{k.replace(/_/g, ' ').toUpperCase()}</span>
                          <span className={styles.macroVal}>
                            {typeof v === 'number' ? v.toFixed(3) : String(v)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className={styles.emptyPicks}>Macro data unavailable</div>
                  )}
                </div>
              </div>

              {/* Strategy metrics */}
              {Object.keys(m).length > 0 && (
                <div className={styles.section}>
                  <div className={styles.sectionTitle}>STRATEGY METRICS</div>
                  <div className={styles.metricsGrid}>
                    {m.rank_ic != null && (
                      <MetricCard label="Rank IC" value={Number(m.rank_ic).toFixed(4)}
                        color={m.rank_ic > 0.05 ? 'green' : m.rank_ic > 0 ? 'accent' : 'red'}
                        icon={<IconActivity />} />
                    )}
                    {bt.sharpe != null && (
                      <MetricCard label="Sharpe Ratio" value={Number(bt.sharpe).toFixed(3)}
                        sub="annualised"
                        color={bt.sharpe > 1 ? 'green' : bt.sharpe > 0 ? 'amber' : 'red'}
                        icon={<IconShield />} />
                    )}
                    {stratReturn != null && (
                      <MetricCard label="Strategy Return" value={`${(stratReturn * 100).toFixed(1)}%`}
                        sub="test period"
                        color={stratReturn > 0 ? 'green' : 'red'}
                        icon={<IconTrend />} />
                    )}
                    {bhReturn != null && (
                      <MetricCard label="Buy & Hold" value={`${(bhReturn * 100).toFixed(1)}%`}
                        sub="benchmark (SPY)"
                        color="amber" icon={<IconDown />} />
                    )}
                    {alpha != null && (
                      <MetricCard label="Alpha" value={`${(alpha * 100).toFixed(1)}%`}
                        sub="vs buy & hold"
                        color={alpha > 0 ? 'green' : 'red'}
                        icon={<IconZap />} />
                    )}
                    {bt.max_drawdown != null && (
                      <MetricCard label="Max Drawdown" value={`${(bt.max_drawdown * 100).toFixed(1)}%`}
                        sub="peak-to-trough"
                        color={bt.max_drawdown > -0.15 ? 'green' : bt.max_drawdown > -0.3 ? 'amber' : 'red'}
                        icon={<IconShield />} />
                    )}
                  </div>
                </div>
              )}

              {/* Alpha vs B&H highlight */}
              {alpha != null && (
                <div className={styles.alphaBanner} style={{ borderColor: alpha > 0 ? 'rgba(0,255,136,0.3)' : 'rgba(255,51,102,0.3)', background: alpha > 0 ? 'rgba(0,255,136,0.04)' : 'rgba(255,51,102,0.04)' }}>
                  <div className={styles.alphaStat}>
                    <span className={styles.alphaLabel}>STRATEGY RETURN</span>
                    <span className={styles.alphaValue} style={{ color: stratReturn > 0 ? 'var(--green)' : 'var(--red)' }}>
                      {stratReturn >= 0 ? '+' : ''}{(stratReturn * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className={styles.alphaDivider}>vs</div>
                  <div className={styles.alphaStat}>
                    <span className={styles.alphaLabel}>BUY & HOLD</span>
                    <span className={styles.alphaValue} style={{ color: 'var(--amber)' }}>
                      {bhReturn >= 0 ? '+' : ''}{(bhReturn * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className={styles.alphaDivider}>→</div>
                  <div className={styles.alphaStat}>
                    <span className={styles.alphaLabel}>ALPHA</span>
                    <span className={styles.alphaValue} style={{ color: alpha > 0 ? 'var(--green)' : 'var(--red)', fontSize: '2rem' }}>
                      {alpha >= 0 ? '+' : ''}{(alpha * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════════════
          BACKTEST TAB
          ══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'backtest' && (
        <div className={styles.backtestSection}>
          <div className={styles.btToolbar}>
            <div>
              <label className={styles.btLabel}>STRATEGY TYPE</label>
              <select
                className={styles.universeSelect}
                value={stratType}
                onChange={e => { setStratType(e.target.value); setBacktest(null); }}
              >
                <option value="long_short">Long/Short</option>
                <option value="long_only">Long Only</option>
              </select>
            </div>
            <button
              className={styles.trainBtn}
              onClick={loadBacktest}
              disabled={btLoading}
            >
              {btLoading ? '⟳ RUNNING…' : '▶ RUN BACKTEST'}
            </button>
          </div>

          {btLoading ? <Loader text="RUNNING BACKTEST…" /> : btError ? (
            <div className={styles.errorBox}>
              <span>⚠ {btError}</span>
              <button className={styles.retrySmall} onClick={loadBacktest}><IconRefresh /> Retry</button>
            </div>
          ) : backtest ? (
            <>
              {/* Metrics */}
              <div className={styles.metricsGrid}>
                {[
                  ['Annual Return',     bt.annual_return,    v => `${(v*100).toFixed(1)}%`,  v => v > 0 ? 'green' : 'red'],
                  ['Sharpe Ratio',      bt.sharpe,           v => v.toFixed(3),               v => v > 1 ? 'green' : v > 0 ? 'amber' : 'red'],
                  ['Max Drawdown',      bt.max_drawdown,     v => `${(v*100).toFixed(1)}%`,  v => v > -0.15 ? 'green' : v > -0.3 ? 'amber' : 'red'],
                  ['Win Rate',          bt.win_rate,         v => `${(v*100).toFixed(1)}%`,  v => v > 0.5 ? 'green' : 'red'],
                  ['Hit Rate',          bt.hit_rate,         v => `${(v*100).toFixed(1)}%`,  v => v > 0.55 ? 'green' : 'amber'],
                  ['Annual Volatility', bt.annual_volatility,v => `${(v*100).toFixed(1)}%`,  () => 'amber'],
                ].filter(([, v]) => v != null).map(([label, val, fmt, colorFn]) => (
                  <MetricCard key={label} label={label} value={fmt(val)}
                    color={colorFn(val)} size="sm" />
                ))}
              </div>

              {/* Alpha comparison */}
              {bt.total_return != null && bt.buy_hold_return != null && (
                <div className={styles.alphaBanner} style={{
                  borderColor: (bt.total_return - bt.buy_hold_return) > 0 ? 'rgba(0,255,136,0.3)' : 'rgba(255,51,102,0.3)',
                  background:  (bt.total_return - bt.buy_hold_return) > 0 ? 'rgba(0,255,136,0.04)' : 'rgba(255,51,102,0.04)',
                }}>
                  <div className={styles.alphaStat}>
                    <span className={styles.alphaLabel}>STRATEGY RETURN</span>
                    <span className={styles.alphaValue} style={{ color: bt.total_return > 0 ? 'var(--green)' : 'var(--red)' }}>
                      {bt.total_return >= 0 ? '+' : ''}{(bt.total_return * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className={styles.alphaDivider}>vs</div>
                  <div className={styles.alphaStat}>
                    <span className={styles.alphaLabel}>BUY & HOLD (SPY)</span>
                    <span className={styles.alphaValue} style={{ color: 'var(--amber)' }}>
                      +{(bt.buy_hold_return * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className={styles.alphaDivider}>→</div>
                  <div className={styles.alphaStat}>
                    <span className={styles.alphaLabel}>ALPHA</span>
                    <span className={styles.alphaValue} style={{ color: (bt.total_return - bt.buy_hold_return) > 0 ? 'var(--green)' : 'var(--red)', fontSize: '2rem' }}>
                      {(bt.total_return - bt.buy_hold_return) >= 0 ? '+' : ''}{((bt.total_return - bt.buy_hold_return) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}

              {/* Equity curve */}
              {equityCurve.length > 0 && (
                <div className={styles.section}>
                  <div className={styles.sectionTitle}>EQUITY CURVE vs BENCHMARK</div>
                  <div className={styles.chartWrap}>
                    <ResponsiveContainer width="100%" height={260}>
                      <AreaChart data={equityCurve} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                        <defs>
                          <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor="var(--green)" stopOpacity={0.25} />
                            <stop offset="95%" stopColor="var(--green)" stopOpacity={0} />
                          </linearGradient>
                          <linearGradient id="bhGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor="var(--amber)" stopOpacity={0.15} />
                            <stop offset="95%" stopColor="var(--amber)" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.06)" />
                        <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }} tickLine={false} />
                        <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }} tickLine={false} />
                        <Tooltip content={<EquityTooltip />} />
                        <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" />
                        <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }} />
                        <Area type="monotone" dataKey="strategy" name="Strategy" stroke="var(--green)" fill="url(#stratGrad)" strokeWidth={2} dot={false} />
                        <Area type="monotone" dataKey="benchmark" name="Buy & Hold" stroke="var(--amber)" fill="url(#bhGrad)" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className={styles.emptyPicks} style={{ padding: '40px', textAlign: 'center' }}>
              Click <strong>RUN BACKTEST</strong> to simulate the strategy.
            </div>
          )}
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════
          ACCURACY TAB
          ══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'accuracy' && (
        <div className={styles.section}>
          <div className={styles.sectionTitle}>STRATEGY ACCURACY</div>
          <div className={styles.metricsGrid}>
            {[
              { label: 'Direction Acc.',  val: m.direction_accuracy, fmt: v => `${(v*100).toFixed(1)}%`, color: v => v > 0.52 ? 'green' : 'amber' },
              { label: 'Rank IC',         val: m.rank_ic,            fmt: v => v.toFixed(4),              color: v => v > 0.05 ? 'green' : v > 0 ? 'accent' : 'red' },
              { label: 'Precision@Top5', val: m.precision_top5,     fmt: v => `${(v*100).toFixed(1)}%`, color: v => v > 0.6 ? 'green' : 'amber' },
              { label: 'Long Hit Rate',  val: m.long_hit_rate,      fmt: v => `${(v*100).toFixed(1)}%`, color: v => v > 0.55 ? 'green' : 'amber' },
              { label: 'Short Hit Rate', val: m.short_hit_rate,     fmt: v => `${(v*100).toFixed(1)}%`, color: v => v > 0.55 ? 'green' : 'amber' },
              { label: 'Win/Loss Ratio', val: m.win_loss_ratio,     fmt: v => v.toFixed(2),              color: v => v > 1 ? 'green' : 'red' },
              { label: 'Hit Rate',       val: bt.hit_rate,          fmt: v => `${(v*100).toFixed(1)}%`, color: v => v > 0.55 ? 'green' : 'amber' },
              { label: 'Information Ratio', val: m.information_ratio, fmt: v => v.toFixed(3),            color: v => v > 0.5 ? 'green' : 'amber' },
            ].filter(({ val }) => val != null).map(({ label, val, fmt, color }) => (
              <MetricCard key={label} label={label} value={fmt(val)} color={color(val)} size="sm" />
            ))}
          </div>
          {Object.keys(m).length === 0 && (
            <div className={styles.emptyPicks} style={{ marginTop: 24, textAlign: 'center' }}>
              No accuracy metrics yet — train a market model first.
            </div>
          )}
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════
          LIVE LOGS TAB
          ══════════════════════════════════════════════════════════════════ */}
      {activeTab === 'logs' && (
        <div className={styles.section}>
          <div className={styles.sectionTitle}>LIVE SYSTEM LOGS</div>
          <LiveLogPanel maxLines={300} />
        </div>
      )}
    </div>
  );
}
