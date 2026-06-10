@echo off
chcp 65001 > nul
echo ============================================================
echo   Türkçe Morfolojik Çözümleme Projesi
echo   Turkish Morphological Disambiguation
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] Gerekli kütüphaneler kontrol ediliyor...
pip install -r requirements.txt -q

echo.
echo [2/2] Proje çalıştırılıyor...
echo.

python morphological_disambiguator.py

echo.
echo ============================================================
echo   Tamamlandı! Çıktılar "output/" klasöründe.
echo ============================================================
pause
