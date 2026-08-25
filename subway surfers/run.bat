@echo off
cd /d "%~dp0"
title ACM - Subway Surfers Body Controller
if not exist ".venv\Scripts\python.exe" (
  echo First run: please run setup.ps1 once.
  pause
  exit /b 1
)
.venv\Scripts\python.exe pose_controller.py
pause
