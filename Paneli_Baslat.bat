@echo off
setlocal
cd /d "%~dp0"
title Baret Tespit Sistemi

start "" /min "_tarayici_ac.bat"

echo ============================================================
echo   BARET TESPIT SISTEMI baslatiliyor...
echo   Modeller yuklenirken (10-20 saniye) bekleyin - tarayici
echo   panel hazir olunca KENDILIGINDEN acilacak.
echo ============================================================
echo.

python panel.py

echo.
echo ------------------------------------------------------------
echo   Panel durdu (veya bir hata olustu - yukaridaki mesaja bak).
echo ------------------------------------------------------------
pause
