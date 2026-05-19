#!/usr/bin/env pwsh
<#
.SYNOPSIS
    RUN_ALL.ps1 — Start FastAPI Backend + Vite Frontend Together (PowerShell)
    
.DESCRIPTION
    This script starts both servers in separate terminal windows on Windows.
    More reliable than .bat for error handling.
    
.USAGE
    # Option 1: From terminal
    powershell -ExecutionPolicy Bypass -File RUN_ALL.ps1
    
    # Option 2: Right-click RUN_ALL.ps1 → Run with PowerShell
#>

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  QUANTML STOCK FORECASTER — Starting Both Backend + Frontend" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "ERROR: requirements.txt not found." -ForegroundColor Red
    Write-Host "Please run from: c:\Users\USER\Desktop\stock-forecaster" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "frontend\package.json")) {
    Write-Host "ERROR: frontend\package.json not found." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Python
Write-Host "[1/4] Checking Python..." -ForegroundColor Cyan
$pythonExists = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
if (-not $pythonExists) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
python --version

# Check Node.js
Write-Host "[2/4] Checking Node.js..." -ForegroundColor Cyan
$nodeExists = $null -ne (Get-Command node -ErrorAction SilentlyContinue)
if (-not $nodeExists) {
    Write-Host "ERROR: Node.js not found. Install from nodejs.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
node --version

# Install Python deps
Write-Host "[3/4] Installing Python dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install Python dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✅ Python dependencies ready" -ForegroundColor Green

# Install Node deps
Write-Host "[4/4] Installing frontend dependencies..." -ForegroundColor Cyan
if (-not (Test-Path "frontend\node_modules")) {
    Push-Location frontend
    npm install --silent
    Pop-Location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install npm packages" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host "✅ Frontend dependencies ready" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  ✅ All dependencies ready. Starting servers..." -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Start FastAPI backend
Write-Host "Starting FastAPI Backend (port 8000)..." -ForegroundColor Yellow
$backendCmd = "uvicorn api.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd `
    -WindowStyle Normal

Start-Sleep -Seconds 2

# Start Vite frontend
Write-Host "Starting Vite Frontend (port 5173)..." -ForegroundColor Yellow
$frontendCmd = "cd frontend; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd `
    -WindowStyle Normal

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  ✅ Both servers started!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend:  http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "  → Open browser and visit: http://localhost:5173" -ForegroundColor Green
Write-Host ""
Write-Host "  To stop: Close either terminal window (or press Ctrl+C)" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan

Read-Host "Press Enter to exit this script (servers will continue running)"
