@echo off
REM ============================================================================
REM RUN_ALL.bat — Start FastAPI Backend + Vite Frontend Together
REM ============================================================================
REM This script starts both servers in two separate terminal windows.
REM Use this instead of running them manually in different terminals.

echo ============================================================================
echo  QUANTML STOCK FORECASTER — Starting Both Backend + Frontend
echo ============================================================================
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found. 
    echo Please run this script from: c:\Users\USER\Desktop\stock-forecaster
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo ERROR: frontend\package.json not found.
    echo Please run this script from: c:\Users\USER\Desktop\stock-forecaster
    pause
    exit /b 1
)

echo [1/4] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

echo [2/4] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Install from nodejs.org
    pause
    exit /b 1
)

echo [3/4] Installing Python dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements. Check requirements.txt
    pause
    exit /b 1
)

echo [4/4] Installing frontend dependencies...
cd frontend
if not exist "node_modules" (
    npm install >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to install npm packages. Check internet connection.
        pause
        exit /b 1
    )
)
cd ..

echo.
echo ============================================================================
echo  ✅ All dependencies ready
echo ============================================================================
echo.
echo Starting servers in 3 seconds...
echo   Terminal 1: FastAPI Backend (http://127.0.0.1:8000)
echo   Terminal 2: Vite Frontend (http://localhost:5173)
echo.
echo NOTE: Two command windows will open. Keep both running.
echo.
timeout /t 3 /nobreak

REM Start FastAPI backend in new window
start "FastAPI Backend - Port 8000" cmd /k "title FastAPI Backend - http://127.0.0.1:8000 && uvicorn api.main:app --reload --port 8000"

REM Wait for backend to start
timeout /t 2 /nobreak

REM Start Vite frontend in new window
start "Vite Frontend - Port 5173" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================================================
echo  ✅ Both servers started!
echo ============================================================================
echo.
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://localhost:5173
echo.
echo   → Open browser: http://localhost:5173
echo.
echo   To stop: Close either terminal window (or Ctrl+C in each)
echo.
echo ============================================================================
pause
