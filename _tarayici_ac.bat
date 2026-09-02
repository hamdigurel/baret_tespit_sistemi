@echo off
REM Panel arka planda ayaga kalkarken (model yukleme ~10-20 sn surer)
REM bir sure bekleyip tarayiciyi otomatik acar. Elle calistirmana gerek
REM yok - Paneli_Baslat.bat bunu kendisi cagirir.
timeout /t 12 /nobreak >nul
start "" http://localhost:8000
