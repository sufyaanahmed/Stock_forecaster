import styles from './Loader.module.css';

export default function Loader({ text = 'PROCESSING...' }) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.matrix}>
        {[...Array(12)].map((_, i) => (
          <div key={i} className={styles.bar} style={{ animationDelay: `${i * 0.08}s` }} />
        ))}
      </div>
      <div className={styles.text}>{text}</div>
    </div>
  );
}
