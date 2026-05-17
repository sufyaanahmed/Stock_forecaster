import { useState, useRef, useEffect } from 'react';
import { triggerTrain, fetchTrainStatus, cacheClear, ApiError } from '../api';
import styles from './TrainPanel.module.css';

export default function TrainPanel({ ticker, onTrainComplete }) {
  const [epochs,  setEpochs]  = useState(100);
  const [start,   setStart]   = useState('2015-01-01');
  const [status,  setStatus]  = useState(null);   // null | 'running' | 'done' | 'error'
  const [msg,     setMsg]     = useState('');
  const [elapsed, setElapsed] = useState(0);

  const pollRef    = useRef(null);
  const timerRef   = useRef(null);
  const startedAt  = useRef(null);

  // Clean up polling on unmount
  useEffect(() => () => {
    clearInterval(pollRef.current);
    clearInterval(timerRef.current);
  }, []);

  async function handleTrain() {
    clearInterval(pollRef.current);
    clearInterval(timerRef.current);
    setStatus('running');
    setElapsed(0);
    startedAt.current = Date.now();
    setMsg('Dispatching training job to API…');

    // Elapsed timer
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt.current) / 1000));
    }, 1000);

    try {
      const res = await triggerTrain(ticker, epochs, start);

      if (res.status === 'already_running') {
        setMsg('Training is already running. Polling for completion…');
      } else {
        setMsg('Training started. This may take several minutes on CPU…');
      }

      // Poll every 5 s for status
      pollRef.current = setInterval(async () => {
        try {
          const s = await fetchTrainStatus(ticker);
          const st = s.status ?? 'unknown';

          if (st.startsWith('done')) {
            clearInterval(pollRef.current);
            clearInterval(timerRef.current);
            setStatus('done');
            const detail = st.replace('done|', '');
            setMsg(`Training complete (${_formatElapsed()}) — ${detail}`);
            cacheClear('models');
            onTrainComplete?.();
          } else if (st.startsWith('error')) {
            clearInterval(pollRef.current);
            clearInterval(timerRef.current);
            setStatus('error');
            setMsg(st.replace('error|', ''));
          } else if (st === 'running') {
            setMsg(`Training in progress… ${_formatElapsed()}`);
          } else {
            setMsg(`Status: ${st}`);
          }
        } catch (e) {
          setMsg(`Poll failed: ${e.message} — will retry…`);
        }
      }, 5000);

    } catch (e) {
      clearInterval(timerRef.current);
      setStatus('error');
      const isNetErr = !(e instanceof ApiError);
      setMsg(isNetErr
        ? 'Cannot reach API. Make sure uvicorn is running on port 8000.'
        : e.message
      );
    }
  }

  function _formatElapsed() {
    const s = Math.floor((Date.now() - startedAt.current) / 1000);
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
  }

  function handleCancel() {
    clearInterval(pollRef.current);
    clearInterval(timerRef.current);
    setStatus(null);
    setMsg('');
  }

  const isRunning = status === 'running';

  return (
    <div className={styles.panel}>
      <div className={styles.title}>TRAIN MODEL</div>

      <div className={styles.row}>
        {/* Ticker */}
        <div className={styles.field}>
          <label className={styles.label}>TICKER</label>
          <div className={styles.tickerVal}>{ticker}</div>
        </div>

        {/* Start date */}
        <div className={styles.field}>
          <label className={styles.label}>START DATE</label>
          <input
            type="date"
            className={styles.input}
            value={start}
            onChange={e => setStart(e.target.value)}
            disabled={isRunning}
            style={{ width: 140 }}
          />
        </div>

        {/* Epochs */}
        <div className={styles.field}>
          <label className={styles.label}>EPOCHS</label>
          <input
            type="number"
            className={styles.input}
            value={epochs}
            min={5}
            max={500}
            step={5}
            onChange={e => setEpochs(Math.max(5, +e.target.value))}
            disabled={isRunning}
          />
        </div>

        {/* Action buttons */}
        <button
          className={`${styles.btn} ${isRunning ? styles.running : ''}`}
          onClick={isRunning ? handleCancel : handleTrain}
          id="train-btn"
        >
          {isRunning ? (
            <><span className={styles.spinner} /> CANCEL</>
          ) : (
            'RUN TRAINING'
          )}
        </button>
      </div>

      {/* Status message */}
      {msg && (
        <div className={`${styles.msg} ${styles[status ?? '']}`}>
          <span className={styles.msgDot} />
          <span className={styles.msgText}>{msg}</span>
          {isRunning && elapsed > 0 && (
            <span className={styles.elapsed}>{elapsed}s</span>
          )}
        </div>
      )}

      {/* Config note */}
      <div className={styles.note}>
        Training downloads historical data from 2015, runs {epochs} epochs on CPU.
        Expect 3–15 min depending on hardware. Results are saved as a checkpoint and
        immediately available for inference.
      </div>
    </div>
  );
}
