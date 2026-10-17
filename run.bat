@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set FLASK_ENV=production
set PYTHONIOENCODING=utf-8

echo.
echo ================================================
echo    Ka-myii - VTuber Model Studio
echo ================================================
echo.

REM --- Virtual environment (.venv; an older venv\ is reused) ---------------
set "VENV=.venv"
if not exist ".venv\Scripts\python.exe" if exist "venv\Scripts\python.exe" set "VENV=venv"
if exist "%VENV%\Scripts\python.exe" goto :have_venv

REM PyTorch 2.8 (used for CUDA 12.8 on 8 GB GPUs) has builds for Python 3.10 - 3.13 only.
set "PYEXE="
for %%v in (3.12 3.11 3.13 3.10) do (
    if not defined PYEXE (
        py -%%v -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PYEXE=py -%%v"
    )
)
if not defined PYEXE (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=python"
)
if not defined PYEXE (
    echo [ERROR] Ka-myii needs Python 3.10 to 3.13 ^(3.12 recommended^).
    echo         Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    echo         Newer Pythons can stay installed; the launcher picks a supported one.
    goto :fail
)
echo [INFO] Creating .venv with %PYEXE% ...
%PYEXE% -m venv .venv
if errorlevel 1 goto :fail
set "VENV=.venv"

:have_venv
set "VPY=%VENV%\Scripts\python.exe"
"%VPY%" --version

REM --- Dependencies: first run, or whenever requirements.txt changed ---------
fc /b requirements.txt "%VENV%\kamyii-requirements.txt" >nul 2>&1
if errorlevel 1 (
    call :install
    if errorlevel 1 goto :fail
) else (
    echo [OK] Dependencies up to date
)

set "HAS_GPU=0"
nvidia-smi >nul 2>&1
if not errorlevel 1 set "HAS_GPU=1"

REM --- Optional See-through engine (real layer decomposition) ----------------
if "%HAS_GPU%"=="1" if not exist "engines\see-through\.venv\Scripts\python.exe" if not exist "engines\see-through.skip" (
    echo.
    echo ================================================
    echo    Optional: See-through layer decomposition
    echo ================================================
    echo Splits each character into 20+ clean, inpainted layers ^(SIGGRAPH 2026^).
    echo Needs an 8 GB+ NVIDIA GPU and about 9 GB of downloads. Without it,
    echo generated characters get a rough geometric split.
    echo.
    set "install_st=y"
    set /p install_st="Install See-through now? [Y/n]: "
    if /i "!install_st!"=="n" (
        if not exist engines mkdir engines
        echo declined> "engines\see-through.skip"
        echo [INFO] Skipped. Install later with:  %VPY% scripts\install_seethrough.py
    ) else (
        "%VPY%" scripts\install_seethrough.py
        if errorlevel 1 echo [WARNING] See-through install failed - Ka-myii will use the basic splitter.
    )
)

REM --- Launch ----------------------------------------------------------------
echo.
echo   1^) Studio - real generation ^(NVIDIA GPU^)
echo   2^) Demo   - no GPU or downloads; a drawn character runs the full pipeline
echo.
set "mode=2"
if "%HAS_GPU%"=="1" set "mode=1"
set /p mode="Choose 1 or 2 [!mode!]: "
echo.
echo Ka-myii opens in your browser. Keep this window open; close it to stop the server.
echo.
if "!mode!"=="2" (
    "%VPY%" app.py --dummy --open
) else (
    "%VPY%" app.py --open
)
echo.
echo Ka-myii has stopped.
pause
exit /b 0

:install
echo.
echo [INFO] Installing dependencies ^(first run or an update^). This can take a while...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo [INFO] No NVIDIA GPU found: installing CPU PyTorch ^(demo mode works; real generation needs a GPU^).
        "%VPY%" -m pip install torch torchvision
    ) else (
        echo [INFO] Installing PyTorch with CUDA 12.8 ^(about 3 GB^)...
        "%VPY%" -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    )
    if errorlevel 1 exit /b 1
)
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
copy /y requirements.txt "%VENV%\kamyii-requirements.txt" >nul
echo [OK] Dependencies installed
exit /b 0

:fail
echo.
echo [ERROR] Setup did not finish. Scroll up for the message above, or see README.md - Troubleshooting.
pause
exit /b 1
