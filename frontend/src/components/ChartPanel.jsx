import { useState } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts';
import styles from './ChartPanel.module.css';

const PERIODS = ['3m', '6m', '1y', '2y'];

function QuantTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className={styles.tooltip}>
      <div className={styles.ttDate}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className={styles.ttRow}>
          <span className={styles.ttDot} style={{ background: p.color }} />
          <span className={styles.ttLabel}>{p.name}</span>
          <span className={styles.ttVal}>{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function ChartPanel({ chartData, onPeriodChange, period }) {
  const [activeTab, setActiveTab] = useState('price');

  if (!chartData) return null;

  const { dates, ohlcv, indicators, log_returns } = chartData;

  const priceData = dates.map((d, i) => ({
    date: d,
    close: ohlcv.close[i],
    volume: ohlcv.volume[i],
  }));

  const indicatorData = dates.map((d, i) => ({
    date: d,
    rsi: indicators.rsi_14[i],
    macd: indicators.macd_norm[i],
    bbPos: indicators.bb_position[i],
    vol: indicators.vol_20[i],
    logRet: log_returns[i],
  }));

  // Sample every N for perf
  const sample = (arr, n = 1) => arr.filter((_, i) => i % n === 0);
  const stride = dates.length > 500 ? 3 : 1;
  const pData = sample(priceData, stride);
  const iData = sample(indicatorData, stride);

  const tabs = [
    { id: 'price',   label: 'PRICE' },
    { id: 'returns', label: 'LOG RETURNS' },
    { id: 'rsi',     label: 'RSI' },
    { id: 'macd',    label: 'MACD' },
    { id: 'bb',      label: 'BOLLINGER' },
    { id: 'vol',     label: 'VOLATILITY' },
  ];

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.tabs}>
          {tabs.map(t => (
            <button
              key={t.id}
              className={`${styles.tab} ${activeTab === t.id ? styles.active : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className={styles.periods}>
          {PERIODS.map(p => (
            <button
              key={p}
              className={`${styles.period} ${period === p ? styles.activePeriod : ''}`}
              onClick={() => onPeriodChange(p)}
            >
              {p.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.chart}>
        <ResponsiveContainer width="100%" height={280}>
          {activeTab === 'price' ? (
            <AreaChart data={pData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="closeGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="var(--accent)" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10 }} tickLine={false} width={60} tickFormatter={v => `$${v?.toFixed(0)}`} />
              <Tooltip content={<QuantTooltip />} />
              <Area type="monotone" dataKey="close" name="Close" stroke="var(--accent)" fill="url(#closeGrad)" strokeWidth={1.5} dot={false} />
            </AreaChart>
          ) : activeTab === 'returns' ? (
            <AreaChart data={iData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="retGradPos" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="var(--green)" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="var(--green)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10 }} tickLine={false} width={60} tickFormatter={v => `${(v*100).toFixed(2)}%`} />
              <Tooltip content={<QuantTooltip />} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
              <Area type="monotone" dataKey="logRet" name="Log Return" stroke="var(--green)" fill="url(#retGradPos)" strokeWidth={1} dot={false} />
            </AreaChart>
          ) : activeTab === 'rsi' ? (
            <LineChart data={iData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} tickLine={false} width={40} />
              <Tooltip content={<QuantTooltip />} />
              <ReferenceLine y={70} stroke="rgba(255,51,102,0.4)" strokeDasharray="4 4" label={{ value: 'OB', fill: 'var(--red)', fontSize: 10 }} />
              <ReferenceLine y={30} stroke="rgba(0,255,136,0.4)" strokeDasharray="4 4" label={{ value: 'OS', fill: 'var(--green)', fontSize: 10 }} />
              <Line type="monotone" dataKey="rsi" name="RSI-14" stroke="var(--purple)" strokeWidth={1.5} dot={false} />
            </LineChart>
          ) : activeTab === 'macd' ? (
            <LineChart data={iData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10 }} tickLine={false} width={60} tickFormatter={v => v?.toExponential(1)} />
              <Tooltip content={<QuantTooltip />} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
              <Line type="monotone" dataKey="macd" name="MACD Norm" stroke="var(--amber)" strokeWidth={1.5} dot={false} />
            </LineChart>
          ) : activeTab === 'bb' ? (
            <LineChart data={iData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} interval="preserveStartEnd" />
              <YAxis domain={[-1.5, 1.5]} tick={{ fontSize: 10 }} tickLine={false} width={40} />
              <Tooltip content={<QuantTooltip />} />
              <ReferenceLine y={1}  stroke="rgba(255,51,102,0.3)" strokeDasharray="4 4" />
              <ReferenceLine y={-1} stroke="rgba(0,255,136,0.3)" strokeDasharray="4 4" />
              <ReferenceLine y={0}  stroke="rgba(255,255,255,0.1)" />
              <Line type="monotone" dataKey="bbPos" name="BB Position" stroke="var(--accent)" strokeWidth={1.5} dot={false} />
            </LineChart>
          ) : (
            <AreaChart data={iData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="var(--amber)" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="var(--amber)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10 }} tickLine={false} width={60} tickFormatter={v => `${(v*100).toFixed(2)}%`} />
              <Tooltip content={<QuantTooltip />} />
              <Area type="monotone" dataKey="vol" name="Vol-20" stroke="var(--amber)" fill="url(#volGrad)" strokeWidth={1.5} dot={false} />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}
