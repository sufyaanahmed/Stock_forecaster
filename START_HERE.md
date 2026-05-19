# QUICK FIX: Running Frontend + Backend

## 🔴 Problem
```
[proxy] FastAPI unreachable: connect ECONNREFUSED 127.0.0.1:8000
```

## ✅ Solution: Start Both Servers

### Option 1: Automated (Easiest)
**Windows**:
```bash
# Double-click one of these in File Explorer:
RUN_ALL.bat
# OR
RUN_ALL.ps1
```

This will automatically:
1. ✅ Check Python & Node.js
2. ✅ Install dependencies
3. ✅ Start FastAPI backend (port 8000)
4. ✅ Start Vite frontend (port 5173)

Then open browser: **http://localhost:5173**

---

### Option 2: Manual (Two Terminals)

**Terminal 1: FastAPI Backend**
```bash
cd c:\Users\USER\Desktop\stock-forecaster
uvicorn api.main:app --reload --port 8000
```

**Terminal 2: Vite Frontend**
```bash
cd c:\Users\USER\Desktop\stock-forecaster\frontend
npm run dev
```

**Browser**: http://localhost:5173

---

## ✔️ Verification Checklist

- [ ] Terminal 1 shows: `INFO: Application startup complete`
- [ ] Terminal 2 shows: `➜ Local: http://localhost:5173/`
- [ ] Browser opens http://localhost:5173 without proxy errors
- [ ] No red errors in browser console (F12)
- [ ] Can load a stock chart
- [ ] Can click "Analyze" and see prediction

---

## 🔧 If Still Not Working

### Issue: Port 8000 Already in Use
```bash
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with number from above)
taskkill /PID <PID> /F

# Try again
uvicorn api.main:app --reload --port 8000
```

### Issue: Port 5173 Already in Use
```bash
# Vite will try port 5174 automatically
# Or manually specify:
npm run dev -- --port 5174
```

### Issue: Still Getting Proxy Errors
```bash
# Check backend is actually running
curl http://localhost:8000/api/health
# Should return: {"status": "ok"}

# Check frontend can reach it
curl http://localhost:8000/api/models
# Should return: {"models": [...]}
```

### Issue: `ModuleNotFoundError: No module named lightgbm`
```bash
pip install lightgbm>=4.0.0
```

### Issue: `npm ERR! code ENOENT`
```bash
cd frontend
npm install
npm run dev
```

---

## 📊 Architecture Summary

```
Browser (http://localhost:5173)
    ↓ makes request to /api/chart
Vite Proxy (port 5173) ← Converts to http://127.0.0.1:8000/api/chart
    ↓ forwards to
FastAPI Backend (port 8000) ← Processes request, returns data
    ↓
Vite returns response to browser
    ↓
React component displays chart
```

**Key**: Browser talks to Vite (port 5173), Vite talks to FastAPI (port 8000).
No CORS errors because Vite proxy handles it server-to-server.

---

## 📚 Full Documentation

For detailed explanation, see:
- `FRONTEND_BACKEND_GUIDE.md` - Complete architecture & API flows
- `api.js` - Frontend API client code
- `vite.config.js` - Proxy configuration
- `api/main.py` - Backend server

---

## ⚡ TL;DR

Run this command **once**:
```bash
# Windows: Run one of these
RUN_ALL.bat
# OR
powershell -ExecutionPolicy Bypass -File RUN_ALL.ps1
```

Then open: **http://localhost:5173**

Done! 🎉
