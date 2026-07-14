@echo off
cd /d "%~dp0"
echo HoloGesture v1.0.0 - Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting HoloGesture...
python main.py
if errorlevel 1 (
    echo.
    echo Failed to start. Make sure Python is installed.
    pause
)
pause
