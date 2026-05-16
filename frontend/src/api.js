const BASE = 'http://localhost:8001';

export async function fetchHealth() {
  const r = await fetch(`${BASE}/api/health`);
  return r.json();
}

export async function fetchModels() {
  const r = await fetch(`${BASE}/api/models`);
  return r.json();
}

export async function fetchAnalysis(ticker) {
  const r = await fetch(`${BASE}/api/analyze/${ticker}`);
  if (!r.ok) {
    const err = await r.json();
    throw new Error(err.detail || 'Analysis failed');
  }
  return r.json();
}

export async function fetchChart(ticker, period = '1y') {
  const r = await fetch(`${BASE}/api/chart/${ticker}?period=${period}`);
  if (!r.ok) throw new Error('Chart data fetch failed');
  return r.json();
}

export async function triggerTrain(ticker, epochs = 100) {
  const r = await fetch(`${BASE}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker, start: '2015-01-01', epochs, seq_len: 60 }),
  });
  return r.json();
}

export async function fetchTrainStatus(ticker) {
  const r = await fetch(`${BASE}/api/train/status/${ticker}`);
  return r.json();
}
