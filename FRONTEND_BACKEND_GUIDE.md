# FRONTEND + BACKEND INTEGRATION GUIDE

## 🔴 Error: `[proxy] FastAPI unreachable: connect ECONNREFUSED 127.0.0.1:8000`

### Root Cause
**ECONNREFUSED** = Connection Refused. The Vite proxy (port 5173) tried to forward your `/api/*` requests to FastAPI on port 8000, but **FastAPI isn't running there**.

```
Browser (http://localhost:5173)
    ↓ (makes request to /api/chart)
Vite Dev Server (port 5173) ← running ✅
    ↓ (tries to proxy to /api/chart)
FastAPI Backend (port 8000) ← NOT running ❌ ← ERROR HERE
    ↓ (would return data if running)
Browser displays response
```

**Why it happens**:
1. You started Vite frontend: `npm run dev` (listening on 5173)
2. You did NOT start FastAPI backend: `uvicorn api.main:app --reload` (should listen on 8000)
3. Browser makes API call → Vite tries to proxy → No server listening → ECONNREFUSED

---

## ✅ FIX: Exact Steps to Run Both Together

### Step 1: Start FastAPI Backend (Terminal 1)
```bash
cd c:\Users\USER\Desktop\stock-forecaster

# Verify requirements installed
pip install -r requirements.txt

# Start FastAPI server
uvicorn api.main:app --reload --port 8000
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

✅ Backend now listening on `http://127.0.0.1:8000`

### Step 2: Start Vite Frontend (Terminal 2)
```bash
cd c:\Users\USER\Desktop\stock-forecaster\frontend

npm run dev
```

**Expected output**:
```
VITE v5.0.0  ready in 123 ms

➜  Local:   http://localhost:5173/
➜  press h + enter to show help
```

✅ Frontend now listening on `http://localhost:5173`

### Step 3: Verify Connection
Browser test:
```
Visit: http://localhost:5173
```

Should load without proxy errors. Check browser console (F12):
- If no proxy errors → ✅ Connected successfully
- If still seeing proxy errors → FastAPI not running (go back to Step 1)

### Step 4: Test Endpoints
```bash
# Terminal 3: Verify backend is responding
curl http://localhost:8000/api/health
# Should return: {"status": "ok"}

curl http://localhost:8000/api/models
# Should return: list of trained models

curl http://localhost:8000/api/chart/AAPL
# Should return: OHLCV + indicators
```

---

## 🎯 Architecture: How They Work Together

```
┌─────────────────────────────────────────────────────────────┐
│ USER BROWSER                                                │
│ http://localhost:5173                                       │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ React App (ChartPanel, ModelsList, TrainPanel)      │   │
│ │ - Renders UI                                         │   │
│ │ - Makes API calls: /api/chart, /api/models, etc    │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          ↓ (Request to /api/chart/AAPL)
          ↑ (Response with OHLCV + RSI + MACD)

┌─────────────────────────────────────────────────────────────┐
│ VITE DEV SERVER (Port 5173)                                 │
│ - Serves React HTML/CSS/JS                                  │
│ - Intercepts /api/* requests                                │
│ - Proxies them to FastAPI                                   │
│ - Returns response to browser                               │
│ - Eliminates CORS (same origin for browser)                │
└─────────────────────────────────────────────────────────────┘
          ↓ (Proxied: GET http://127.0.0.1:8000/api/chart/AAPL)
          ↑ (Proxied response)

┌─────────────────────────────────────────────────────────────┐
│ FASTAPI BACKEND (Port 8000)                                 │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ routes/legacy_routes.py & routes/analyze.py        │   │
│ │ - /api/chart/{ticker}  → Fetch OHLCV + compute RSI │   │
│ │ - /api/models          → List trained models        │   │
│ │ - /api/analyze/{ticker}→ Full analysis + predict    │   │
│ │ - /api/train           → Start background training  │   │
│ └──────────────────────────────────────────────────────┘   │
│          ↓                                                  │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ DATA LAYER                                          │   │
│ │ - yfinance: Download OHLCV                          │   │
│ │ - pipeline.py: Preprocess + features               │   │
│ │ - Server-side cache: Avoids re-downloading         │   │
│ └──────────────────────────────────────────────────────┘   │
│          ↓                                                  │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ ML LAYER                                            │   │
│ │ - models/lstm.py: Neural network                    │   │
│ │ - checkpoints/*.pt: Trained model weights          │   │
│ │ - Model cache: Avoid reloading on every request    │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principle: **Vite Proxy Eliminates CORS**

Without proxy:
```
Browser (http://localhost:5173)
    ↓ (direct fetch to http://127.0.0.1:8000)
    ↓ CORS BLOCKED: Different port = different origin
    ↓ Browser says "No, this violates same-origin policy"
```

With Vite proxy:
```
Browser (http://localhost:5173)
    ↓ (fetch to http://localhost:5173/api/chart)
    ✅ Same origin! Browser allows it
Vite Server ← running on 5173
    ↓ (converts to http://127.0.0.1:8000/api/chart)
FastAPI Backend ← running on 8000
    ↓ (server-to-server, no CORS)
Vite forwards response back to browser
```

---

## 📡 API Flow: What Happens When You Request `/api/chart`

### Complete Flow Walkthrough

```
USER CLICKS "Load Chart"
  ↓
React component calls: fetchChart("AAPL")
  ↓
api.js: fetch("/api/chart/AAPL")
  ↓
1. Check browser cache: Is /api/chart/AAPL already cached? (TTL=120s)
   - YES → Return cached data immediately
   - NO  → Continue to next step
  ↓
2. Check if request already in-flight (dedup parallel requests)
   - YES → Wait on existing Promise
   - NO  → Continue to next step
  ↓
3. Browser makes HTTP request to http://localhost:5173/api/chart/AAPL
   (Note: RELATIVE URL, same origin as browser)
  ↓
Vite Proxy Intercepts Request
  ↓
4. Vite proxy matches /api/* pattern in vite.config.js
  ↓
5. Vite rewrites URL to: http://127.0.0.1:8000/api/chart/AAPL
  ↓
6. Vite makes server-to-server request to FastAPI backend
   (No CORS here: server-to-server is allowed)
  ↓
FastAPI Backend (port 8000) Processes Request
  ↓
7. routes/legacy_routes.py handles GET /api/chart/{ticker}
  ↓
8. Check server-side data cache: Is AAPL data already cached? (TTL=300s)
   - YES → Skip data fetch, use cached data
   - NO  → Continue to next step
  ↓
9. Download AAPL data from yfinance (5 years of daily OHLCV)
  ↓
10. Compute features: RSI, MACD, Bollinger Bands
  ↓
11. Format as JSON: { ohlcv: [...], rsi: [...], macd: [...] }
  ↓
12. Store in server-side cache (TTL=300s)
  ↓
13. Return to Vite proxy
  ↓
Vite Proxy Returns Response to Browser
  ↓
14. Vite receives response from FastAPI
  ↓
15. Vite forwards response to browser
  ↓
Browser Receives & Displays
  ↓
16. api.js receives response
  ↓
17. Cache data in browser (TTL=120s)
  ↓
18. React component renders ChartPanel with OHLCV + indicators
  ↓
CHART APPEARS ON SCREEN ✅
```

### API Flow for 3 Key Endpoints

#### Endpoint 1: GET `/api/models`
```
Browser → Vite Proxy → FastAPI → routes/legacy_routes.py
  ↓
def get_models():
    checkpoints = list(Path("checkpoints").glob("*_lstm.pt"))
    return { "models": [c.stem for c in checkpoints] }
  ↓
Response: { "models": ["AAPL_lstm", "NVDA_lstm", ...] }
  ↓
Browser cache: 30s TTL
  ↓
React ModelsList displays model buttons
```

#### Endpoint 2: GET `/api/chart/{ticker}`
```
Browser → Vite Proxy → FastAPI → routes/legacy_routes.py
  ↓
def get_chart(ticker):
    # Server-side cache check (300s TTL)
    if ticker in _data_cache:
        return _data_cache[ticker]
    
    # Download OHLCV from yfinance
    df = yf.download(ticker, period="5y", progress=False)
    
    # Compute indicators
    rsi = compute_rsi(df.close)
    macd = compute_macd(df.close)
    
    # Format response
    data = {
        "ohlcv": df[["open", "high", "low", "close"]].to_dict(),
        "rsi": rsi.to_dict(),
        "macd": macd.to_dict(),
        "volume": df.volume.to_dict()
    }
    
    # Cache server-side
    _data_cache[ticker] = data
    return data
  ↓
Response: 5 years of OHLCV + indicators (JSON)
  ↓
Browser cache: 120s TTL
  ↓
ChartPanel renders with Chart.js or Recharts
```

#### Endpoint 3: GET `/api/analyze/{ticker}`
```
Browser → Vite Proxy → FastAPI → routes/legacy_routes.py
  ↓
def analyze(ticker):
    # Get chart data
    chart_data = get_chart(ticker)
    
    # Load pre-trained LSTM model from checkpoint
    model = load_lstm_model(ticker)
    
    # Get latest data point
    X_latest = preprocess_latest(chart_data)
    
    # Predict next day's return
    with torch.no_grad():
        prediction = model(X_latest)  # e.g., +0.5%
    
    # Compute evaluation metrics
    metrics = full_evaluation(ticker)
    
    return {
        "ticker": ticker,
        "prediction": prediction,
        "confidence": metrics["ic"],
        "metrics": metrics
    }
  ↓
Response: { "prediction": +0.5%, "confidence": 0.043, ... }
  ↓
Browser cache: 60s TTL
  ↓
React displays prediction + historical IC/Sharpe metrics
```

---

## 📋 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ECONNREFUSED 127.0.0.1:8000` | FastAPI not running | Run `uvicorn api.main:app --reload --port 8000` in Terminal 1 |
| `ENOTFOUND localhost:8000` | Typo in vite.config.js target | Check `target: 'http://127.0.0.1:8000'` is correct |
| `Module not found: /api/chart` | Vite proxy path mismatch | Ensure vite.config.js has `/api: { target: ... }` |
| `Timeout on /api/analyze` | Model loading slow | Increase timeout: `timeout: 120000` in vite.config.js |
| `CORS error in browser console` | FastAPI CORS not configured | Check `app.add_middleware(CORSMiddleware, ...)` in api/main.py |
| `Cannot find module pydantic` | Requirements not installed | Run `pip install -r requirements.txt` |
| `Port 8000 already in use` | Another process on 8000 | `netstat -ano \| findstr :8000` then `taskkill /PID <pid>` |
| `npm ERR! code ENOENT` | frontend/node_modules missing | Run `cd frontend && npm install` |
| `Vite port 5173 already in use` | Another Vite instance running | Add `--port 5174` or kill existing process |

---

## 🚀 TLDR: Working Setup

### Terminal 1: FastAPI Backend
```bash
cd c:\Users\USER\Desktop\stock-forecaster
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```
✅ Server ready on http://127.0.0.1:8000

### Terminal 2: Vite Frontend
```bash
cd c:\Users\USER\Desktop\stock-forecaster\frontend
npm install  # if needed
npm run dev
```
✅ Server ready on http://localhost:5173

### Browser
```
Navigate to: http://localhost:5173
Expected:   App loads without proxy errors
```

✅ **Both working together**

---

## 🏗️ Project Structure

```
stock-forecaster/
├── api/
│   └── main.py              # FastAPI app (port 8000)
│       ├── routes/legacy_routes.py     # /api/* endpoints
│       ├── routes/analyze.py           # /market/* endpoints
│       └── CORS middleware configured
│
├── models/
│   ├── lstm.py              # Neural network model
│   └── *.pt checkpoint files
│
├── data/
│   └── pipeline.py          # Data loading & preprocessing
│
├── frontend/
│   ├── vite.config.js       # Proxy config (forwards /api/* to :8000)
│   ├── package.json
│   ├── src/
│   │   ├── api.js           # API client (makes /api/* requests)
│   │   ├── App.jsx          # React root component
│   │   └── components/
│   │       ├── ChartPanel.jsx       # Renders chart
│   │       ├── ModelsList.jsx       # Lists /api/models
│   │       ├── TrainPanel.jsx       # Trains models
│   │       └── ...
│   └── package-lock.json
│
└── requirements.txt         # Python dependencies
```

---

## 📊 Data Flow Summary

```
TRAINING:
  user clicks "Train AAPL"
    → POST /api/train
    → Backend starts background task
    → Loads historical data from yfinance
    → Computes features (RSI, MACD, etc.)
    → Trains LSTM on train set
    → Evaluates on test set
    → Saves checkpoint as AAPL_lstm.pt
    → Returns training status

INFERENCE (Chart Display):
  user selects "AAPL" from dropdown
    → GET /api/chart/AAPL
    → Backend checks 5-year cache (TTL=300s)
    → If cached: return cached OHLCV + indicators
    → If not: fetch from yfinance, compute indicators, cache
    → Return JSON with open/high/low/close/rsi/macd/volume
    → Frontend renders chart with Chart.js or Recharts

ANALYSIS (Prediction):
  user clicks "Analyze AAPL"
    → GET /api/analyze/AAPL
    → Backend gets latest data (from /api/chart cache)
    → Loads AAPL_lstm.pt model from disk/cache
    → Prepares X_latest (30-day lookback window)
    → Runs model inference: y_pred = LSTM(X_latest)
    → Computes confidence (Rank IC from test set)
    → Returns { prediction: +0.5%, confidence: 0.043 }
    → Frontend displays prediction + metrics
```

---

## 🧠 Q&A: System Design Interview Prep

### Q1: Why does the frontend need a proxy?
**A**: Without a proxy, the browser would make cross-origin requests (port 5173 → 8000), violating CORS policy. The proxy acts as a middleman:
- Browser talks to port 5173 (same origin, allowed)
- Vite forwards to port 8000 server-to-server (no CORS)
- Response comes back to browser
Result: No CORS errors, cleaner architecture.

### Q2: What happens if the user requests `/api/chart/AAPL` twice in 30 seconds?
**A**: 
1. First request: Cache miss → Download from yfinance → Store in server cache (TTL=300s)
2. Second request: Cache hit → Return cached data immediately (no yfinance call)
Impact: 98% faster, reduces bandwidth/API rate limits, improves UX.

### Q3: Why use PyTorch LSTM instead of LightGBM?
**A**: 
- LSTM: Captures temporal patterns, good for sequence prediction
- LightGBM: Fast gradient boosting, but needs multiple time-series grouping
For single-stock prediction (legacy mode), LSTM is more interpretable. New market mode uses LambdaRank (see ARCHITECTURE.md).

### Q4: How does the frontend handle slow model inference (30s)?
**A**: 
- api.js sets timeout to 90s (default)
- vite.config.js sets proxy timeout to 120s
- React shows loading spinner during request
- If timeout: Show error message + retry button
Result: User knows something is happening, not hanging.

### Q5: If backend crashes, what does frontend show?
**A**:
- api.js catches fetch errors
- Retries twice with exponential backoff (500ms, 1500ms, 4000ms)
- After 3 failures: Shows error message "API unavailable"
- User can retry manually
Result: Graceful degradation.

### Q6: Why is model caching important?
**A**:
- Loading PyTorch model from disk: ~100ms
- Keeping in memory: ~1ms
- 100 daily requests: 100ms × 100 = 10s wasted loading vs 1s in cache
- Cache strategy: Keep latest 5 models in memory, evict oldest
Result: 10× faster inference, better UX.

### Q7: How do you debug this system?
**A**:
```bash
# 1. Check FastAPI is running
curl http://localhost:8000/api/health
# Should return: {"status": "ok"}

# 2. Check Vite proxy config
cat frontend/vite.config.js | grep proxy

# 3. Check browser console for proxy errors (F12)
# Should see: No proxy errors

# 4. Check network tab (F12 → Network)
# Watch requests to http://localhost:5173/api/*
# Should see 200 responses, not 502 (proxy error)

# 5. Check backend logs
# Should see: "GET /api/chart/AAPL - 200 OK"
```

### Q8: What's the difference between /api/* and /market/* routes?
**A**:
- `/api/*` (LEGACY): Single-stock LSTM prediction, backward compatible
  - GET /api/chart/{ticker} → Returns OHLCV + indicators
  - GET /api/analyze/{ticker} → Returns LSTM prediction
  
- `/market/*` (NEW): Multi-stock macro-aware ranking
  - GET /market/analyze → Full market analysis (all stocks ranked)
  - GET /market/analyze/{ticker} → Single stock within market context
  - POST /market/train → Train LambdaRank ranker
  
Frontend currently uses /api/* (legacy). To switch to market mode: Change urls in api.js.

### Q9: Why does ChartPanel use `key={ticker}` in React?
**A**:
```jsx
// Without key: React reuses same DOM element
// Problem: Chart.js canvas persists old data
// Solution: key={ticker} forces React to destroy+recreate component
// Result: Fresh chart every time ticker changes
```

### Q10: How would you add authentication?
**A**:
```bash
1. Backend: Add JWT token endpoint
   POST /api/login → Returns token

2. Frontend: Store token in localStorage
   const token = localStorage.getItem("token")

3. All API calls: Include Authorization header
   fetch("/api/chart/AAPL", {
     headers: { "Authorization": `Bearer ${token}` }
   })

4. FastAPI: Check token before returning data
   @app.get("/api/chart/{ticker}")
   async def get_chart(ticker: str, token: Annotated[str, Depends(verify_token)]):
       ...
```

---

## 🎯 Next Steps

1. ✅ Start FastAPI: `uvicorn api.main:app --reload --port 8000`
2. ✅ Start Vite: `npm run dev` (in frontend dir)
3. ✅ Test browser: http://localhost:5173
4. ✅ Check console (F12) for proxy errors
5. ✅ Load a stock and verify chart displays
6. ✅ Click "Analyze" and verify prediction loads
7. ✅ Check Network tab (F12 → Network) to see request flow

If still seeing proxy errors after these steps: See "Common Errors & Fixes" section above.

---

## 📚 Related Documentation

- `Readme.md` - Project overview
- `ARCHITECTURE.md` - System design details
- `vite.config.js` - Proxy configuration
- `api.js` - Frontend API client
- `api/main.py` - Backend server
- `requirements.txt` - Python dependencies
