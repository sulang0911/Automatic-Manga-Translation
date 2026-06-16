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
echo WARNING: Only choose YES if you do not use these libraries for other 
echo python projects on this machine.
echo.
set /p clean_python="Do you want to uninstall Python packages (flask, flask-cors, opencv-python, paddlepaddle-gpu)? (Y/N): "
if /i "%clean_python%"=="Y" (
    echo [*] Uninstalling Python dependencies...
    python -m pip uninstall -y flask flask-cors opencv-python paddlepaddle-gpu nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12
    echo [+] Python packages uninstalled.
) else (
    echo [-] Skipped Python dependencies uninstall.
)

echo.
echo ======================================================================
echo Uninstallation completed successfully!
echo ======================================================================
echo.
pause
