/**
 * LiveLogPanel.jsx
 * ================
 * Real-time streaming log terminal with color-coded levels.
 * Polls GET /api/logs every 2 seconds and appends new records.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchLogs } from '../api';
import styles from './LiveLogPanel.module.css';

const LEVEL_COLORS = {
  INFO:    'var(--text-secondary)',
  SUCCESS: 'var(--green)',
  WARNING: 'var(--amber)',
  ERROR:   'var(--red)',
};

const SOURCE_COLORS = {
  training: 'var(--purple)',
  backtest: 'var(--accent)',
  features: '#22d3ee',
  market:   '#f59e0b',
  macro:    '#a78bfa',
  cache:    'var(--text-muted)',
  api:      'var(--green)',
  ranking:  '#fb923c',
};

const POLL_MS = 2000;

export default function LiveLogPanel({ maxLines = 200 }) {
  const [logs, setLogs]         = useState([]);
  const [paused, setPaused]     = useState(false);
  const [filter, setFilter]     = useState('ALL');
  const [sinceTs, setSinceTs]   = useState(0);
  const bottomRef               = useRef(null);
  const pausedRef               = useRef(false);

  pausedRef.current = paused;

  const fetchNew = useCallback(async () => {
    if (pausedRef.current) return;
    try {
      const data = await fetchLogs(sinceTs, 100);
      const newRecords = data?.logs ?? [];
      if (newRecords.length > 0) {
        const maxTs = Math.max(...newRecords.map(r => r.ts));
        setSinceTs(maxTs);
        setLogs(prev => {
          const combined = [...prev, ...newRecords];
          return combined.slice(-maxLines);
        });
      }
    } catch {
      // silently ignore poll errors
    }
  }, [sinceTs, maxLines]);

  // Initial load
  useEffect(() => {
    fetchNew();
  }, []); // eslint-disable-line

  // Polling
  useEffect(() => {
    const timer = setInterval(fetchNew, POLL_MS);
    return () => clearInterval(timer);
  }, [fetchNew]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (!paused) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, paused]);

  const levels   = ['ALL', 'INFO', 'SUCCESS', 'WARNING', 'ERROR'];
  const filtered = filter === 'ALL' ? logs : logs.filter(r => r.level === filter);

  function fmtTime(ts) {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  return (
    <div className={styles.panel}>
      {/* Toolbar */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <span className={styles.termDot} style={{ background: 'var(--red)' }} />
          <span className={styles.termDot} style={{ background: 'var(--amber)' }} />
          <span className={styles.termDot} style={{ background: 'var(--green)' }} />
          <span className={styles.termTitle}>LIVE LOGS</span>
          <span className={styles.recordCount}>{filtered.length} records</span>
        </div>
        <div className={styles.toolbarRight}>
          {/* Level filter */}
          {levels.map(l => (
            <button
              key={l}
              className={`${styles.filterBtn} ${filter === l ? styles.filterActive : ''}`}
              onClick={() => setFilter(l)}
              style={filter === l && l !== 'ALL' ? { color: LEVEL_COLORS[l] } : {}}
            >
              {l}
            </button>
          ))}
          <button
            className={`${styles.filterBtn} ${paused ? styles.pauseActive : ''}`}
            onClick={() => setPaused(p => !p)}
          >
            {paused ? '▶ RESUME' : '⏸ PAUSE'}
          </button>
          <button
            className={styles.filterBtn}
            onClick={() => { setLogs([]); setSinceTs(0); }}
          >
            CLEAR
          </button>
        </div>
      </div>

      {/* Log lines */}
      <div className={styles.terminal}>
        {filtered.length === 0 ? (
          <div className={styles.emptyState}>
            <span className={styles.cursor}>_</span>
            Waiting for log events…
          </div>
        ) : (
          filtered.map((r, i) => (
            <div key={`${r.ts}-${i}`} className={styles.logLine}>
              <span className={styles.logTime}>{fmtTime(r.ts)}</span>
              <span
                className={styles.logLevel}
                style={{ color: LEVEL_COLORS[r.level] ?? 'var(--text-secondary)' }}
              >
                {r.level.padEnd(7)}
              </span>
              <span
                className={styles.logSource}
                style={{ color: SOURCE_COLORS[r.source] ?? 'var(--text-muted)' }}
              >
                [{r.source}]
              </span>
              <span className={styles.logMsg}>{r.message}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {paused && (
        <div className={styles.pauseBanner}>
          ⏸ Scrolling paused — click RESUME to continue live updates
        </div>
      )}
    </div>
  );
}
