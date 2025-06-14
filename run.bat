@echo off
echo Starting Face Recognition System...

:: Start backend
start "Backend Server" uv run backend.py

:: Wait for backend to initialize
timeout /t 15

:: Start frontend
start "Frontend GUI" uv run frontend.py

:: Wait for frontend to close
:WAIT
tasklist | find "frontend.py" >nul
if errorlevel 1 (
    :: Frontend closed, kill backend
    taskkill /F /FI "WINDOWTITLE eq Backend Server*" >nul 2>&1
    exit
) else (
    timeout /t 1 /nobreak >nul
    goto WAIT
)