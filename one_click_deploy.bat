@echo off
title Automatic Manga Translation One-Click Deployer
color 0B
echo ======================================================================
echo    Automatic Manga Translation - One-Click Environment Setup ^& Startup
echo ======================================================================
echo.

:: 1. Check Node.js
echo [*] Checking Node.js environment...
where node >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Node.js is not installed!
    echo Please download and install Node.js from: https://nodejs.org/
    echo Recommended version: v18 or v20 LTS.
    echo.
    pause
    exit /b 1
)
echo [+] Node.js is ready.
echo.

:: 2. Check Python
echo [*] Checking Python environment...
where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [+] Python is ready [Global].
) else if exist "python_env\python.exe" (
    echo [+] Python is ready [Portable].
) else (
    echo [!] Global Python not found. A portable Python environment will be set up automatically during server launch.
)
echo.

:: 3. Setup Frontend
echo [*] Setting up frontend dependencies (React + Vite)...
if not exist node_modules\vite (
    echo [!] Frontend dependencies missing or incomplete. Installing...
    call npm install
) else (
    echo [+] Frontend dependencies already installed.
)
echo.

:: 4. Setup & Launch Services
color 0A
echo [+] Environment setup check complete!
echo [*] Starting services in separate windows...
echo.

:: Start OCR Server (via start_ocr_server.bat to ensure isolated GPU env)
echo [*] Launching Local OCR Server...
start "AetherLens Local OCR Server" start_ocr_server.bat

:: Start Frontend Dev Server
echo [*] Launching Frontend Web App...
start "AetherLens Frontend Server" cmd /c "npm run dev"

echo.
echo ======================================================================
echo   Both services are now starting. 
echo   - The Web UI will be available at: http://localhost:5173
echo   - The OCR API is running at: http://127.0.0.1:5000
echo.
echo   Keep the newly opened command windows running while using the app!
echo ======================================================================
echo.
pause
