import styles from './SignalBadge.module.css';

export default function SignalBadge({ direction, returnPct, confidence }) {
  const isUp   = direction === 'UP';
  const isDown = direction === 'DOWN';
  const isNeut = direction === 'NEUTRAL';

  const colorClass = isUp ? styles.up : isDown ? styles.down : styles.neutral;
  const arrow      = isUp ? '▲' : isDown ? '▼' : '—';
  const label      = isUp ? 'LONG' : isDown ? 'SHORT' : 'FLAT';

  return (
    <div className={`${styles.wrapper} ${colorClass}`}>
      <div className={styles.pulse} />
      <div className={styles.arrow}>{arrow}</div>
      <div className={styles.content}>
        <div className={styles.label}>{label} SIGNAL</div>
        <div className={styles.ret}>
          {returnPct >= 0 ? '+' : ''}{returnPct?.toFixed(3)}% expected
        </div>
        <div className={styles.conf}>
          Confidence: <strong>{(confidence * 100).toFixed(1)}%</strong>
        </div>
      </div>
    </div>
  );
}
