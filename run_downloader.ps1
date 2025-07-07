Write-Host "Image Downloader with Playwright" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Yellow
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if pip is available
try {
    $pipVersion = pip --version 2>&1
    Write-Host "Pip found: $pipVersion" -ForegroundColor Yellow
} catch {
    Write-Host "Error: Pip is not available" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Installing Playwright browsers..." -ForegroundColor Cyan
playwright install

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install Playwright browsers" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting image downloader..." -ForegroundColor Cyan
python image_downloader.py

Write-Host ""
Write-Host "Process completed!" -ForegroundColor Green
Read-Host "Press Enter to exit" 