@echo off
REM Ka-myii Quick Start Script for Windows

echo ================================
echo Ka-myii VTuber Model Generator
echo ================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Dependencies not found. Installing...
    pip install -r requirements.txt
    echo Dependencies installed!
    echo.
)

REM Ask user which mode to run
echo Select mode:
echo 1) Dummy mode (no GPU required, for testing)
echo 2) Production mode (requires GPU)
set /p mode="Enter choice (1 or 2): "

echo.
echo Starting Ka-myii...
echo.

if "%mode%"=="1" (
    python app.py --dummy
) else (
    python app.py
)

pause
