# COMPLETE SOLUTION: Frontend-Backend Integration

## 🔴 Your Error: ECONNREFUSED 127.0.0.1:8000

### What It Means
```
Connection to 127.0.0.1 (localhost) on port 8000 was refused.
Why? No server is listening on port 8000.
Where? FastAPI backend should be running there.
```

### Why It Happened
1. You started Vite: `npm run dev` ✅ (listening on port 5173)
2. You did NOT start FastAPI: `uvicorn api.main:app` ❌ (should listen on port 8000)
3. Browser made request → Vite tried to proxy to backend → Backend not running → ERROR

---

## ✅ IMMEDIATE FIX (30 seconds)

### Windows Users: Double-Click One of These
```
RUN_ALL.bat              ← Easiest (Command prompt)
RUN_ALL.ps1             ← More reliable (PowerShell)
```

**What it does**:
- ✅ Checks Python & Node.js installed
- ✅ Installs dependencies automatically
- ✅ Opens 2 terminal windows (one for backend, one for frontend)
- ✅ Prints when both are ready

**Then**: Open browser → **http://localhost:5173**

---

## ✅ MANUAL FIX (if scripts don't work)

### Terminal 1: FastAPI Backend
```bash
cd c:\Users\USER\Desktop\stock-forecaster
uvicorn api.main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Terminal 2: Vite Frontend
```bash
cd c:\Users\USER\Desktop\stock-forecaster\frontend
npm run dev
```

Expected output:
```
➜  Local:   http://localhost:5173/
```

### Browser
```
Visit: http://localhost:5173
```

---

## 🔍 Verification: Is It Working?

### Check 1: Backend Running?
```bash
curl http://localhost:8000/api/health
# Should return: {"status": "ok"}
```

### Check 2: Frontend Can Reach Backend?
```bash
# Open browser console (F12)
# Should see NO proxy errors
# Should see successful API calls in Network tab
```

### Check 3: Full System Working?
```
✅ App loads in browser (http://localhost:5173)
✅ Can select a stock ticker
✅ Can click "Load Chart" and see chart appears
✅ Can click "Analyze" and see prediction loads
✅ No red errors in browser console
```

If all 5 checks pass → **You're done!** ✅

---

## 🏗️ How It All Works Together

### Architecture: 3-Layer System

```
┌──────────────────────────────────┐
│  BROWSER (http://localhost:5173) │
│  ┌────────────────────────────┐  │
│  │ React App                  │  │
│  │ - ChartPanel              │  │
│  │ - ModelsList              │  │
│  │ - TrainPanel              │  │
│  └────────────────────────────┘  │
│  Clicks "Load Chart"             │
└──────────────────────────────────┘
          ↓ (makes HTTP request)
         [SAME ORIGIN]
┌──────────────────────────────────┐
│  VITE DEV SERVER (port 5173)     │
│  - Serves React HTML/CSS/JS      │
│  - Intercepts /api/* requests    │
│  - **Proxies them to backend**   │
│  - Returns response to browser   │
└──────────────────────────────────┘
          ↓ (proxied request)
      [DIFFERENT PORT OK]
┌──────────────────────────────────┐
│  FASTAPI BACKEND (port 8000)     │
│  - Receives /api/chart/AAPL      │
│  - Loads data from yfinance      │
│  - Computes indicators (RSI)     │
│  - Returns JSON: OHLCV + RSI     │
└──────────────────────────────────┘
          ↓ (response)
    Returns to Vite proxy
          ↓
   Returns to browser
          ↓
    React renders chart
          ↓
   CHART APPEARS ON SCREEN ✅
```

### Why Vite Proxy?

**Without proxy** (CORS error):
```
Browser on localhost:5173
    → tries to fetch from localhost:8000
    → CORS BLOCKED: Different port = different origin
    → Browser: "No, you can't do that!"
```

**With Vite proxy** (no CORS):
```
Browser on localhost:5173
    → fetches from localhost:5173/api/chart ✅ Same origin!
    → Vite forwards to localhost:8000/api/chart
    → FastAPI returns data
    → Vite returns to browser
    → No CORS issues!
```

**Key insight**: Proxy makes it same-origin for browser, server-to-server for backend.

---

## 📡 API Flow: What Happens When You Request `/api/chart`

### Step-by-Step Walkthrough

```
1. USER CLICKS "Load Chart"
   ↓
2. React component calls: fetchChart("AAPL")
   ↓
3. api.js: fetch("/api/chart/AAPL")
   ↓
4. Browser check: Is this URL already cached? (TTL=120 seconds)
   - YES → Return cache, done immediately
   - NO  → Continue to next step
   ↓
5. Browser makes HTTP request to http://localhost:5173/api/chart/AAPL
   (Note: Same port as page, so no CORS issue)
   ↓
6. Vite proxy intercepts: "Oh, this is /api/*, I know what to do"
   ↓
7. Vite converts URL: http://127.0.0.1:8000/api/chart/AAPL
   ↓
8. Vite makes server-to-server request to FastAPI
   (No CORS issue: server-to-server is always allowed)
   ↓
9. FastAPI receives: GET /api/chart/AAPL
   ↓
10. routes/legacy_routes.py handles it
   ↓
11. Check server-side cache: Is AAPL data cached? (TTL=300 seconds)
    - YES → Return cached data, skip yfinance call
    - NO  → Continue to next step
    ↓
12. Download AAPL data from yfinance (5 years of daily OHLCV)
   ↓
13. Compute indicators: RSI, MACD, Bollinger Bands
   ↓
14. Format response: {
      "ohlcv": [...],
      "rsi": [...],
      "macd": [...],
      "volume": [...]
    }
   ↓
15. Store in server cache (TTL=300 seconds)
   ↓
16. Return to Vite proxy
   ↓
17. Vite forwards to browser
   ↓
18. api.js receives response
   ↓
19. Browser cache: Store data (TTL=120 seconds)
   ↓
20. React component: Render ChartPanel
   ↓
21. Chart appears on screen ✅
```

### Cache Strategy (2-Level Caching)

```
Level 1: Browser Cache (frontend/src/api.js)
  - TTL: 15s-120s depending on endpoint
  - Purpose: Avoid hammering Vite proxy
  - Cleared manually or on TTL expiry

Level 2: Server Cache (api/main.py)
  - TTL: 300s (5 minutes)
  - Purpose: Avoid downloading same data repeatedly from yfinance
  - Saves bandwidth and improves response time

Example: First request takes 3 seconds, second request same data takes 5ms ✅
```

---

## 📋 Three Key API Endpoints

### 1. GET `/api/models`
**What it does**: Lists all trained models

```
Request:  GET /api/models
Response: {
  "models": [
    "AAPL_lstm",
    "NVDA_lstm", 
    "META_lstm",
    "QQQ_lstm"
  ]
}
Browser cache TTL: 30 seconds
```

**Used by**: ModelsList component (shows buttons for each model)

---

### 2. GET `/api/chart/{ticker}`
**What it does**: Returns OHLCV + indicators for charting

```
Request:  GET /api/chart/AAPL
Response: {
  "ohlcv": {
    "2023-01-01": {open: 150, high: 151, low: 149, close: 150.5},
    "2023-01-02": {open: 150.5, high: 152, low: 150, close: 151.2},
    ...
  },
  "rsi": {
    "2023-01-01": 45.2,
    "2023-01-02": 47.1,
    ...
  },
  "macd": {
    "2023-01-01": {macd: 0.5, signal: 0.4, histogram: 0.1},
    ...
  },
  "volume": {
    "2023-01-01": 1000000,
    ...
  }
}
Browser cache TTL: 120 seconds
Server cache TTL: 300 seconds
```

**Used by**: ChartPanel component (renders with Chart.js or Recharts)

---

### 3. GET `/api/analyze/{ticker}`
**What it does**: Full analysis + LSTM prediction

```
Request:  GET /api/analyze/AAPL
Response: {
  "ticker": "AAPL",
  "prediction": 0.005,           ← +0.5% next day
  "confidence": 0.043,           ← Rank IC from test set
  "metrics": {
    "annual_return": 0.12,
    "sharpe": 0.95,
    "max_drawdown": -0.18,
    "accuracy": 0.52
  }
}
Browser cache TTL: 60 seconds
```

**Used by**: Analysis section (shows prediction + metrics)

---

## 🛠️ Common Problems & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| `ECONNREFUSED 127.0.0.1:8000` | Backend not running | Run: `uvicorn api.main:app --reload --port 8000` |
| Proxy timeout on `/api/analyze` | Model loading slow | Increase timeout in vite.config.js |
| Port 8000 already in use | Another process on 8000 | `netstat -ano \| findstr :8000` → `taskkill /PID <id>` |
| Port 5173 already in use | Another Vite instance | Kill old process or use `npm run dev -- --port 5174` |
| `ModuleNotFoundError: lightgbm` | Dependency missing | `pip install lightgbm>=4.0.0` |
| `npm ERR! code ENOENT` | node_modules missing | `cd frontend && npm install` |
| CORS error in browser | Proxy misconfigured | Check vite.config.js has correct target |
| API returns 404 | Endpoint doesn't exist | Check endpoint exists in api/main.py |
| Stale data in chart | Cache not expiring | Hard refresh browser (Ctrl+Shift+R) or wait 120s |
| Backend crashes on API call | Python error | Check backend terminal for traceback |

---

## 🎓 System Design Q&A

### Q: Why does Vite proxy exist?
**A**: To eliminate CORS issues. Browser can't fetch from different port directly, but proxy makes it same-origin for browser while doing server-to-server for backend.

### Q: What if backend is slow?
**A**: Vite proxy has 120s timeout. If inference takes >120s, it times out. Solution: Increase timeout in vite.config.js or use GPU for faster inference.

### Q: How does caching avoid redundant API calls?
**A**: 2-level caching:
- Browser: Don't refetch same data for 120s
- Server: Don't redownload from yfinance for 300s
- Result: 100 requests in 5 minutes = only 1 yfinance call

### Q: Why run both servers instead of one?
**A**: Separation of concerns:
- Frontend: Vite dev server (hot reload, file watching)
- Backend: FastAPI (data/ML processing)
- Easy to develop/debug independently

### Q: Can this work in production?
**A**: Not as-is. In production:
- Remove Vite (use built React bundle)
- Serve frontend from same port as backend or different domain with proper CORS
- Use HTTPS, authentication, rate limiting

### Q: How do I add a new API endpoint?
**A**:
1. Add function in routes/legacy_routes.py:
   ```python
   @router.get("/api/new-endpoint")
   async def new_endpoint():
       return {"data": "value"}
   ```
2. Test: `curl http://localhost:8000/api/new-endpoint`
3. Frontend automatically proxied (no Vite config change needed)

### Q: Why LSTM for single-stock vs LambdaRank for multi-stock?
**A**: 
- LSTM: Good at learning sequences (yesterday's price → tomorrow's price)
- LambdaRank: Good at learning rankings (stock A outperforms stock B given macro context)
- Different problems = different models

---

## 📚 Files You Edited/Created

```
Created:
  ✅ START_HERE.md - Quick 2-minute fix
  ✅ FRONTEND_BACKEND_GUIDE.md - 2800+ line comprehensive guide
  ✅ RUN_ALL.bat - Automated Windows batch script
  ✅ RUN_ALL.ps1 - Automated PowerShell script

Already Existed:
  📄 vite.config.js - Proxy configuration
  📄 api/main.py - FastAPI server
  📄 frontend/src/api.js - Frontend API client
  📄 routes/legacy_routes.py - /api/* endpoints
  📄 routes/analyze.py - /market/* endpoints
```

---

## 🚀 Next Steps

1. **Run it**: Double-click RUN_ALL.bat or RUN_ALL.ps1
2. **Verify**: Open http://localhost:5173, should load without errors
3. **Test**: Load a chart, analyze a stock, train a model
4. **Explore**: Read FRONTEND_BACKEND_GUIDE.md for deep dive
5. **Customize**: Edit vite.config.js or api/main.py as needed

---

## 📞 If Still Stuck

### Debug Checklist

```bash
# 1. Check backend is running
curl http://localhost:8000/api/health
# Should return: {"status": "ok"}

# 2. Check frontend proxy config
cat frontend/vite.config.js | grep -A5 proxy

# 3. Check ports are correct
# vite.config.js should have: target: 'http://127.0.0.1:8000'

# 4. Check network tab (browser F12)
# Should see requests to http://localhost:5173/api/*
# With status 200 (not 502)

# 5. Check backend logs
# Terminal 1 should show: "GET /api/chart/AAPL - 200 OK"
```

### Get Help

- **Quick start**: START_HERE.md
- **Full guide**: FRONTEND_BACKEND_GUIDE.md
- **Code**: Check docstrings in api.js, vite.config.js, api/main.py

---

## ✅ Success Criteria

You're done when you see:

```
✅ Browser loads http://localhost:5173
✅ No proxy errors in browser console (F12)
✅ Can select a stock ticker
✅ Chart displays OHLCV data
✅ Can click "Analyze" and see prediction
✅ Backend terminal shows: "GET /api/* - 200 OK"
```

---

**Status**: ✅ Ready to use!

Run `RUN_ALL.bat` or `RUN_ALL.ps1`, then open http://localhost:5173

You got this! 🎉
