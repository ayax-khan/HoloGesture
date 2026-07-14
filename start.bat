@echo off
cd /d "%~dp0"
echo HoloGesture v1.0.0 - Starting...
echo.
python main.py
if errorlevel 1 (
    echo.
    echo Failed to start. Make sure Python is installed.
    echo Install dependencies: pip install -r requirements.txt
    pause
)
