import { useState } from 'react';
import { triggerTrain, fetchTrainStatus } from '../api';
import styles from './TrainPanel.module.css';

export default function TrainPanel({ ticker, onTrainComplete }) {
  const [epochs, setEpochs] = useState(100);
  const [status, setStatus] = useState(null); // null | 'running' | 'done' | 'error'
  const [msg, setMsg] = useState('');

  async function handleTrain() {
    setStatus('running');
    setMsg('Dispatching training job...');
    try {
      const res = await triggerTrain(ticker, epochs);
      if (res.status === 'already_running') {
        setMsg('Training already in progress. Polling...');
      } else {
        setMsg('Training started. Polling for completion...');
      }
      // Poll every 5s
      const poll = setInterval(async () => {
        const s = await fetchTrainStatus(ticker);
        if (s.status.startsWith('done')) {
          clearInterval(poll);
          setStatus('done');
          setMsg(`Complete: ${s.status.replace('done|', '')}`);
          onTrainComplete && onTrainComplete();
        } else if (s.status.startsWith('error')) {
          clearInterval(poll);
          setStatus('error');
          setMsg(s.status.replace('error|', ''));
        } else {
          setMsg(`Status: ${s.status}...`);
        }
      }, 5000);
    } catch (e) {
      setStatus('error');
      setMsg(e.message);
    }
  }

  return (
    <div className={styles.panel}>
      <div className={styles.title}>TRAIN MODEL</div>
      <div className={styles.row}>
        <div className={styles.field}>
          <label className={styles.label}>TICKER</label>
          <div className={styles.tickerVal}>{ticker}</div>
        </div>
        <div className={styles.field}>
          <label className={styles.label}>EPOCHS</label>
          <input
            type="number"
            className={styles.input}
            value={epochs}
            min={5}
            max={500}
            onChange={e => setEpochs(+e.target.value)}
            disabled={status === 'running'}
          />
        </div>
        <button
          className={`${styles.btn} ${status === 'running' ? styles.running : ''}`}
          onClick={handleTrain}
          disabled={status === 'running'}
          id="train-btn"
        >
          {status === 'running' ? (
            <><span className={styles.spinner} /> TRAINING...</>
          ) : (
            'RUN TRAINING'
          )}
        </button>
      </div>
      {msg && (
        <div className={`${styles.msg} ${styles[status]}`}>
          <span className={styles.msgDot} />
          {msg}
        </div>
      )}
    </div>
  );
}
