@echo off
SETLOCAL EnableDelayedExpansion
cd /d "%~dp0"
set FLASK_ENV=production
set PYTHONIOENCODING=utf-8

REM ================================
REM Ka-myii Enhanced Setup & Launch
REM ================================

echo.
echo ================================================
echo    Ka-myii VTuber Model Generator
echo    Enhanced Setup with CUDA Detection
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo [OK] Python detected
python --version
echo.

REM Reuse the existing environment; prefer the same .venv used for development.
set "KAMYII_VENV=.venv"
if not exist ".venv\Scripts\python.exe" if exist "venv\Scripts\python.exe" set "KAMYII_VENV=venv"
if not exist "%KAMYII_VENV%\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    python -m venv "%KAMYII_VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
) else (
    echo [OK] Virtual environment found
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call "%KAMYII_VENV%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Check if dependencies are installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo [INFO] Dependencies not found. Installing...
    goto :install_deps
) else (
    echo [OK] Dependencies already installed
    goto :check_pytorch
)

:install_deps
echo.
echo ================================================
echo    Installing Dependencies
echo ================================================
echo.

REM Upgrade pip first
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Pip upgrade failed, but continuing anyway...
    echo If you see errors below, try running: python -m pip install --upgrade pip --user
    echo.
) else (
    echo [OK] Pip upgraded successfully
)
echo.

REM Check for CUDA/GPU support
echo [INFO] Checking for NVIDIA GPU support...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NVIDIA GPU not detected or nvidia-smi not available
    echo [INFO] Will install CPU-only version of PyTorch
    set CUDA_AVAILABLE=0
) else (
    echo [OK] NVIDIA GPU detected!
    nvidia-smi
    echo.
    set CUDA_AVAILABLE=1
)

echo.
echo ================================================
echo    Installing PyTorch
echo ================================================
echo.

if !CUDA_AVAILABLE! equ 1 (
    echo [INFO] Installing PyTorch with CUDA 12.8 support...
    echo This may take a few minutes...
    pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo [WARNING] CUDA installation failed, falling back to CPU version...
        pip install torch torchvision torchaudio
    )
) else (
    echo [INFO] Installing PyTorch (CPU version)...
    pip install torch torchvision torchaudio
)

echo.
echo [INFO] Installing other dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo [OK] All dependencies installed successfully!
echo.

:check_pytorch
REM Optional: See-through engine for real layer decomposition of AI-generated characters
if not exist "engines\see-through\.venv" (
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo ================================================
        echo    Optional: See-through layer decomposition
        echo ================================================
        echo Splits generated characters into 20+ clean, inpainted layers
        echo ^(SIGGRAPH 2026^). Needs an 8 GB+ NVIDIA GPU and ~9 GB download.
        echo Without it, real images use a rough geometric split.
        echo.
        set /p install_st="Install See-through now? (Y/n): "
        if /i not "!install_st!"=="n" (
            python scripts\install_seethrough.py
            if errorlevel 1 echo [WARNING] See-through install failed - Ka-myii will use the basic splitter.
        )
        echo.
    )
)

REM Verify PyTorch installation and CUDA availability
echo ================================================
echo    PyTorch Configuration
echo ================================================
echo.

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"

echo.

REM Ask user which mode to run
echo ================================================
echo    Launch Options
echo ================================================
echo.
echo Select mode:
echo   1) Production mode (with GPU/CUDA if available)
echo   2) Dummy mode (no AI generation, for testing UI)
echo   3) Exit
echo.

set /p mode="Enter choice (1, 2, or 3): "

if "%mode%"=="3" (
    echo.
    echo Exiting...
    pause
    exit /b 0
)

if "%mode%"=="2" (
    echo.
    echo ================================================
    echo    Starting Ka-myii in DUMMY MODE
    echo    (No actual AI generation)
    echo ================================================
    echo.
    python app.py --dummy
) else if "%mode%"=="1" (
    echo.
    echo ================================================
    echo    Starting Ka-myii in PRODUCTION MODE
    echo ================================================
    echo.

    REM Check if CUDA is available
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
    if errorlevel 1 (
        echo [WARNING] CUDA not available - generation will be SLOW on CPU
        echo Press Ctrl+C to cancel, or any key to continue with CPU...
        pause
    ) else (
        echo [OK] CUDA available - GPU acceleration enabled
    )

    echo.
    python app.py
) else (
    echo [ERROR] Invalid choice. Please run the script again.
    pause
    exit /b 1
)

REM If we get here, the server has stopped
echo.
echo ================================================
echo    Ka-myii has stopped
echo ================================================
echo.
pause
