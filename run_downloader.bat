@echo off
echo Image Downloader with Playwright
echo ================================
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Installing Playwright browsers...
playwright install

echo.
echo Starting image downloader...
python image_downloader.py

echo.
echo Press any key to exit...
pause >nul 