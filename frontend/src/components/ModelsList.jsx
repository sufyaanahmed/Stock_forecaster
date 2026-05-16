import styles from './ModelsList.module.css';

export default function ModelsList({ models, activeTicker, onSelect }) {
  if (!models?.length) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>⊘</div>
        <div className={styles.emptyText}>No trained models found.<br/>Run training first.</div>
      </div>
    );
  }
  return (
    <div className={styles.list}>
      {models.map(m => (
        <button
          key={m.ticker}
          className={`${styles.item} ${activeTicker === m.ticker ? styles.active : ''}`}
          onClick={() => onSelect(m.ticker)}
          id={`model-${m.ticker}`}
        >
          <div className={styles.ticker}>{m.ticker}</div>
          <div className={styles.stats}>
            <span className={`${styles.stat} ${m.test_ic > 0 ? styles.pos : styles.neg}`}>
              IC {m.test_ic > 0 ? '+' : ''}{m.test_ic?.toFixed(4)}
            </span>
            <span className={`${styles.stat} ${m.test_dir_acc > 0.5 ? styles.pos : styles.neg}`}>
              DA {(m.test_dir_acc * 100)?.toFixed(1)}%
            </span>
          </div>
          <div className={styles.arrow}>›</div>
        </button>
      ))}
    </div>
  );
}
