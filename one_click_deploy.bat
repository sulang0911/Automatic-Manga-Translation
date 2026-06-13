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
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed!
    echo Please download and install Python from: https://www.python.org/
    echo Please make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
echo [+] Python is ready.
echo.

:: 3. Setup Frontend
echo [*] Setting up frontend dependencies (React + Vite)...
if not exist node_modules (
    echo [!] node_modules not found. Installing dependencies...
    call npm install
) else (
    echo [+] Frontend dependencies already installed.
)
echo.

:: 4. Setup Backend
echo [*] Setting up Python dependencies for OCR server...
python -c "import flask, flask_cors, cv2, numpy, torch, torchvision, paddleocr" >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Some Python packages are missing. Installing backend dependencies...
    python -m pip install --upgrade pip
    python -m pip install flask flask-cors opencv-python numpy torch torchvision paddlepaddle paddleocr
) else (
    echo [+] Python dependencies are ready.
)
echo.

:: 5. Launch Servers
color 0A
echo [+] Environment setup complete!
echo [*] Starting services in separate windows...
echo.

:: Start OCR Server
echo [*] Launching Local OCR Server...
start "AetherLens Local OCR Server" cmd /c "python ocr_server.py"

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
