@echo off
title AetherLens Local High-Precision OCR Server
echo ======================================================
echo   AetherLens Local High-Precision OCR Server (PaddleOCR)
echo ======================================================
echo.
echo [*] Starting python OCR backend server...
echo [*] Close this window to stop the server.
echo.
set PYTHONIOENCODING=utf-8
python ocr_server.py
echo.
echo [!] Server stopped.
pause
