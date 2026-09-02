@echo off
setlocal
cd /d "%~dp0"
title Masaustu Kisayolu Olustur

echo Masaustune kisayol ekleniyor...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Baret Tespit Sistemi.lnk'); $s.TargetPath='%~dp0Paneli_Baslat.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0baret_simge.ico'; $s.Description='Baret Tespit Sistemi panelini baslatir'; $s.Save()"

if errorlevel 1 (
    echo.
    echo HATA: kisayol olusturulamadi.
) else (
    echo.
    echo Bitti. Masaustunde "Baret Tespit Sistemi" adinda bir simge var artik.
    echo Panelini baslatmak icin bundan sonra SADECE o simgeye cift tikla -
    echo terminal acmana gerek yok.
    echo.
    echo Bu dosyayi ^(Kisayol_Olustur.bat^) bir daha calistirmana gerek yok,
    echo istersen silebilirsin.
)

echo.
pause
