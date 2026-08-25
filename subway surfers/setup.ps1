# One-time setup for the ACM stall machine.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { py -3.13 -m venv .venv }

Write-Host "Installing dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install mediapipe opencv-python pydirectinput

if (-not (Test-Path "models\pose_landmarker_lite.task")) {
  Write-Host "Downloading pose model..." -ForegroundColor Cyan
  New-Item -ItemType Directory -Force models | Out-Null
  Invoke-WebRequest -OutFile "models\pose_landmarker_lite.task" `
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
}

Write-Host "`nRunning self-test..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe test_logic.py
.\.venv\Scripts\python.exe test_browser.py

Write-Host "`nSetup complete. Double-click run.bat to start the stall." -ForegroundColor Green
