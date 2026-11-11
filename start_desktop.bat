@echo off
title AetherLens - Native Manga Translator Desktop Client
color 0B
echo ======================================================================
echo    AetherLens Manga Translation - Apple Minimalist Desktop App
echo ======================================================================
echo.

cd /d "%~dp0"

:: Enforce single-threaded and low-memory execution on 8GB RAM systems
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set TORCH_NUM_THREADS=1
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONHASHSEED=0

:: Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo [*] Launching AetherLens PyQt6 Desktop Client in single-thread memory safe mode...
python "%~dp0run_desktop.py"

if %errorlevel% neq 0 (
    echo.
    echo [!] Application exited with code %errorlevel%.
    pause
)