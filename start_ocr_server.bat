@echo off
title AetherLens Local High-Precision OCR Server
echo ======================================================
echo   AetherLens Local High-Precision OCR Server (PaddleOCR)
echo ======================================================
echo.

set VENV_PYTHON=venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo [!] Python 3.10 virtual environment not found.
    echo [*] Starting automatic environment setup...
    echo.
    
    :: Check if global python is available
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        python setup_venv.py
    ) else (
        echo [!] Global Python not found in PATH.
        echo [*] Attempting to run setup using PowerShell download/execution...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process python -ArgumentList 'setup_venv.py' -NoNewWindow -Wait" 2>nul
        if %errorlevel% neq 0 (
            echo [!] Failed to execute setup. Downloading Python installer directly via PowerShell...
            powershell -NoProfile -ExecutionPolicy Bypass -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python_installer.exe' }"
            if exist python_installer.exe (
                echo [*] Installing Python 3.10.11 silently to python_env...
                mkdir python_env >nul 2>nul
                python_installer.exe /quiet InstallAllUsers=0 TargetDir="%CD%\python_env" PrependPath=0 Include_doc=0 Include_test=0
                del python_installer.exe
                if exist "%CD%\python_env\python.exe" (
                    echo [*] Creating virtual environment...
                    "%CD%\python_env\python.exe" -m venv venv
                    if exist "%VENV_PYTHON%" (
                        echo [*] Installing dependencies in virtual environment...
                        "%VENV_PYTHON%" setup_venv.py
                    )
                )
            )
        )
    )
    
    if not exist "%VENV_PYTHON%" (
        echo.
        echo [X] Error: Failed to set up virtual environment automatically.
        echo [!] Please make sure you have Python installed globally, or run setup_venv.py manually.
        echo.
        pause
        exit /b 1
    )
)

echo [*] Starting python OCR backend server using isolated environment...
echo [*] Close this window to stop the server.
echo.
set PYTHONIOENCODING=utf-8
"%VENV_PYTHON%" ocr_server.py
echo.
echo [!] Server stopped.
pause

