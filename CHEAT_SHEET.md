# CHEAT SHEET: Frontend + Backend

## 🚀 FASTEST WAY TO RUN

### Option A: Automatic (Recommended)
```
Double-click: RUN_ALL.bat
              (or RUN_ALL.ps1)
Then: Open http://localhost:5173
```

### Option B: Manual
```
Terminal 1:   uvicorn api.main:app --reload --port 8000
Terminal 2:   cd frontend && npm run dev
Browser:      http://localhost:5173
```

---

## ✅ VERIFY IT WORKS

```bash
# Check backend running
curl http://localhost:8000/api/health
# → should return: {"status": "ok"}

# Check frontend can reach backend
curl http://localhost:8000/api/models
# → should return: {"models": [...]}
```

---

## 🔍 TROUBLESHOOT

| Error | Fix |
|-------|-----|
| `ECONNREFUSED 127.0.0.1:8000` | Start backend (see Option B above) |
| Port 8000 in use | `netstat -ano \| findstr :8000` → `taskkill /PID <id>` |
| Port 5173 in use | Try 5174: `npm run dev -- --port 5174` |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `npm ERR! ENOENT` | `cd frontend && npm install` |
| No chart displays | Check browser F12 console for errors |

---

## 🏗️ HOW IT WORKS

```
Browser (5173)
    ↓ /api/chart request
Vite Proxy (5173) ← converts to 127.0.0.1:8000/api/chart
    ↓
FastAPI Backend (8000)
    ↓ returns data
→ CHART DISPLAYS ✅
```

**Key**: Browser talks to Vite (same port), Vite talks to FastAPI.

---

## 📡 API ENDPOINTS

```
GET  /api/health        → Server status
GET  /api/models        → List trained models
GET  /api/chart/{ticker}    → OHLCV + indicators
GET  /api/analyze/{ticker}  → Prediction + metrics
POST /api/train             → Start training
GET  /api/train/status/{ticker} → Training progress
```

---

## 🎯 EXPECTED OUTPUT

### Backend Ready
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Frontend Ready
```
➜  Local:   http://localhost:5173/
```

### Browser
```
Page loads without proxy errors
Chart displays
Predictions load
```

---

## 🧠 KEY CONCEPTS

- **Vite Proxy**: Converts browser request (port 5173) to backend (port 8000)
- **Browser Cache**: Stores API responses for 15-120s
- **Server Cache**: Stores data for 300s to avoid re-downloading
- **2 Ports Needed**: 5173 for frontend, 8000 for backend
- **Both Must Run**: One without the other causes ECONNREFUSED

---

## 💾 FILE LOCATIONS

```
c:\Users\USER\Desktop\stock-forecaster\

Backend files:
  api/main.py              ← FastAPI server
  routes/legacy_routes.py  ← /api/* endpoints
  vite.config.js          ← Proxy config
  requirements.txt        ← Python deps

Frontend files:
  frontend/src/api.js     ← API client
  frontend/package.json   ← NPM deps
  frontend/src/App.jsx    ← React root
  frontend/src/components/  ← UI components
```

---

## ⚡ QUICK COMMANDS

```bash
# Install everything
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Start everything
RUN_ALL.bat

# Or manually
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev

# Test endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/api/models
curl "http://localhost:8000/api/chart/AAPL"

# Browser
http://localhost:5173
```

---

## 🐛 DEBUG STEPS

1. Check both servers running: See "VERIFY IT WORKS" above
2. Check browser console (F12): Look for red proxy errors
3. Check Network tab (F12): Look for failed requests
4. Check backend terminal: Look for "GET /api/* - 200 OK"
5. Hard refresh browser: Ctrl+Shift+R to clear cache

---

## 📊 COMMON MISTAKES

❌ Starting only frontend → Proxy error (ECONNREFUSED)
✅ Start both frontend AND backend

❌ Vite proxy not forwarding → Check vite.config.js
✅ Ensure target is correct: http://127.0.0.1:8000

❌ Port already in use → Connection refused
✅ Check what's using the port, kill it, try again

❌ Dependencies missing → ModuleNotFoundError
✅ Run: pip install -r requirements.txt && npm install

---

## 📚 DOCS

- **Quick start**: START_HERE.md
- **Full guide**: SOLUTION_SUMMARY.md
- **Deep dive**: FRONTEND_BACKEND_GUIDE.md
- **Architecture**: ARCHITECTURE.md

---

## ✅ SUCCESS = ALL 5 CHECK

- [ ] Backend running on 8000 (see terminal 1)
- [ ] Frontend running on 5173 (see terminal 2)
- [ ] Browser loads http://localhost:5173
- [ ] No errors in browser console (F12)
- [ ] Can load chart and see data

**All checked?** → You're done! 🎉

---

**Bookmark this page!** 📌
