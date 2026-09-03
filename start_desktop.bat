@echo off
title AetherLens - Native Manga Translator Desktop Client
color 0B
echo ======================================================================
echo    AetherLens Manga Translation - Apple Minimalist Desktop App
echo ======================================================================
echo.

cd /d "%~dp0"

:: Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo [*] Launching AetherLens PyQt6 Desktop Client...
python "%~dp0run_desktop.py"

if %errorlevel% neq 0 (
    echo.
    echo [!] Application exited with code %errorlevel%.
    pause
)