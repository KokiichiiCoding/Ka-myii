@echo off
REM Simple Ka-myii launcher - for troubleshooting
REM If run.bat fails, try this simpler version

echo ========================================
echo Ka-myii - Simple Launch
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Install from: https://www.python.org/
    echo Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo Python detected:
python --version
echo.

REM Create venv if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Try: python -m pip install --upgrade pip
        pause
        exit /b 1
    )
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Choose mode:
echo 1 = Demo mode (no GPU, lightweight, instant)
echo 2 = Full mode (installs PyTorch, GPU support)
echo.
set /p choice="Enter 1 or 2: "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto full

echo Invalid choice
pause
exit /b 1

:demo
echo.
echo Installing demo dependencies (lightweight)...
pip install Flask flask-cors flask-socketio Pillow numpy psd-tools pytoshop six python-dotenv
if errorlevel 1 (
    echo.
    echo ERROR during installation. Try:
    echo 1. Close all Python programs
    echo 2. Run as Administrator
    echo 3. Disable antivirus temporarily
    pause
    exit /b 1
)
echo.
echo Starting Ka-myii in DEMO mode...
echo.
python app.py --dummy
goto end

:full
echo.
echo Installing full dependencies (this may take 5-10 minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR during installation. Common fixes:
    echo 1. Make sure you have 2GB+ free disk space
    echo 2. Try running as Administrator
    echo 3. Check internet connection
    echo 4. Disable antivirus temporarily
    pause
    exit /b 1
)
echo.
echo Starting Ka-myii in FULL mode...
echo.
python app.py
goto end

:end
echo.
echo Ka-myii has stopped
pause
