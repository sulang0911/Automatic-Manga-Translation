@echo off
title AetherLens Manga Translation Project Uninstaller
color 0C
echo ======================================================================
echo          AetherLens Manga Translation Project - Uninstaller
echo ======================================================================
echo.
echo WARNING: This script will remove dependencies and cache files for this project.
echo.
set /p confirm="Are you sure you want to proceed with uninstallation? (Y/N): "
if /i "%confirm%" neq "Y" (
    echo [!] Uninstallation cancelled.
    pause
    exit /b 0
)

echo.
echo ======================================================================
echo 1. Cleaning local project files...
echo ======================================================================
if exist node_modules (
    echo [*] Removing node_modules (frontend dependencies)...
    rmdir /s /q node_modules
    echo [+] node_modules removed.
) else (
    echo [-] node_modules not found.
)

if exist dist (
    echo [*] Removing dist (build artifacts)...
    rmdir /s /q dist
    echo [+] dist removed.
) else (
    echo [-] dist not found.
)

echo.
echo ======================================================================
echo 2. Cleaning PaddleOCR model cache files...
echo ======================================================================
echo This will free up around 1-2 GB of disk space by deleting downloaded
echo model weights in your user profile folder (.paddlex & .paddleocr).
echo.
set /p clean_models="Do you want to delete PaddleOCR cached model files? (Y/N): "
if /i "%clean_models%"=="Y" (
    if exist "%USERPROFILE%\.paddlex" (
        echo [*] Removing %USERPROFILE%\.paddlex...
        rmdir /s /q "%USERPROFILE%\.paddlex"
    )
    if exist "%USERPROFILE%\.paddleocr" (
        echo [*] Removing %USERPROFILE%\.paddleocr...
        rmdir /s /q "%USERPROFILE%\.paddleocr"
    )
    echo [+] Model cache cleaned.
) else (
    echo [-] Skipped model cache cleaning.
)

echo.
echo ======================================================================
echo 3. Cleaning Python dependencies...
echo ======================================================================
echo This will delete the isolated Python environment (python_env) and 
echo virtual environment (venv) created inside this project.
echo.
set /p clean_python="Do you want to delete the isolated Python and virtual environment? (Y/N): "
if /i "%clean_python%"=="Y" (
    if exist python_env (
        echo [*] Removing python_env...
        rmdir /s /q python_env
    )
    if exist venv (
        echo [*] Removing venv...
        rmdir /s /q venv
    )
    echo [+] Isolated Python environment and virtual environment removed.
) else (
    echo [-] Skipped isolated Python cleaning.
)

echo.
echo ======================================================================
echo Uninstallation completed successfully!
echo ======================================================================
echo.
pause
