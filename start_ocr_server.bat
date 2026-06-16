@echo off
title AetherLens Local High-Precision OCR Server
echo ======================================================
echo   AetherLens Local High-Precision OCR Server (PaddleOCR)
echo ======================================================
echo.

set VENV_PYTHON=python_env\python.exe

if not exist "python_env\.setup_complete" (
    echo [!] Python 3.10 environment setup is missing or incomplete.
    echo [*] Starting automatic environment setup...
    echo.
    
    :: Check if global python is available
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        python setup_venv.py
    ) else (
        echo [!] Global Python not found in PATH.
        echo [*] Bootstrapping portable Python environment using PowerShell...
        
        :: 1. Download zip
        powershell -NoProfile -ExecutionPolicy Bypass -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; echo '[*] Downloading Python 3.10.11 embeddable package...'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile 'python_embed.zip' }"
        
        if exist python_embed.zip (
            :: 2. Extract zip
            echo [*] Extracting Python to python_env...
            mkdir python_env >nul 2>nul
            powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path 'python_embed.zip' -DestinationPath 'python_env' -Force"
            del python_embed.zip
            
            :: 3. Configure python310._pth
            if exist "python_env\python310._pth" (
                echo [*] Enabling site-packages in portable environment...
                powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-Content python_env\python310._pth) -replace '#import site', 'import site' | Set-Content python_env\python310._pth"
            )
            
            :: 4. Run the rest of setup using the bootstrapped python
            if exist "%VENV_PYTHON%" (
                echo [*] Bootstrapping completed. Running dependency installer...
                "%VENV_PYTHON%" setup_venv.py
            )
        )
    )
    
    if not exist "python_env\.setup_complete" (
        echo.
        echo [X] Error: Failed to set up Python environment automatically.
        echo [!] Please install Python 3.10 globally, or configure the portable environment manually.
        echo.
        pause
        exit /b 1
    )
)

:: Clean up any orphaned processes on port 5000 and 5001 before starting
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001') do taskkill /f /pid %%a >nul 2>&1

echo [*] Starting python OCR backend server using isolated environment...
echo [*] Close this window to stop the server.
echo.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Start inpainting server in background
start /b "" "%VENV_PYTHON%" inpaint_server.py

:: Start main OCR server in foreground
"%VENV_PYTHON%" ocr_server.py

echo.
echo [!] Stopping background services...
:: Clean up inpainting server on port 5001
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001') do taskkill /f /pid %%a >nul 2>&1

echo.
echo [!] Server stopped.
pause
