import styles from './MetricCard.module.css';

export default function MetricCard({ label, value, sub, color = 'accent', icon, delta, size = 'md' }) {
  const colorMap = {
    accent: 'var(--accent)',
    green:  'var(--green)',
    red:    'var(--red)',
    amber:  'var(--amber)',
    purple: 'var(--purple)',
  };
  const c = colorMap[color] || colorMap.accent;

  return (
    <div className={`${styles.card} ${styles[size]}`} style={{ '--card-color': c }}>
      <div className={styles.glow} />
      <div className={styles.scanLine} />
      {icon && <div className={styles.icon}>{icon}</div>}
      <div className={styles.label}>{label}</div>
      <div className={styles.value}>{value}</div>
      {sub && <div className={styles.sub}>{sub}</div>}
      {delta !== undefined && (
        <div className={`${styles.delta} ${delta >= 0 ? styles.pos : styles.neg}`}>
          {delta >= 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(3)}
        </div>
      )}
      <div className={styles.corner} />
    </div>
  );
}
