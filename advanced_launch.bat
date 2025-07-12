@echo off
setlocal enabledelayedexpansion
title YOLO Building & Population Estimator - Advanced Launcher
color 0A

:: Main Menu
:MAIN_MENU
cls
echo ============================================
echo    YOLO Building & Population Estimator
echo           Advanced Launcher v1.0
echo ============================================
echo.
echo Please select an option:
echo.
echo [1] Quick Launch (Install dependencies + Run)
echo [2] Run Application Only
echo [3] Install/Update Dependencies Only
echo [4] System Check
echo [5] View Requirements
echo [6] Create Desktop Shortcut
echo [7] Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto QUICK_LAUNCH
if "%choice%"=="2" goto RUN_ONLY
if "%choice%"=="3" goto INSTALL_DEPS
if "%choice%"=="4" goto SYSTEM_CHECK
if "%choice%"=="5" goto VIEW_REQUIREMENTS
if "%choice%"=="6" goto CREATE_SHORTCUT
if "%choice%"=="7" goto EXIT
goto MAIN_MENU

:QUICK_LAUNCH
cls
echo ============================================
echo         Quick Launch Mode
echo ============================================
echo.
call :SYSTEM_CHECK_INTERNAL
if !errorlevel! neq 0 goto MAIN_MENU
call :INSTALL_DEPS_INTERNAL
if !errorlevel! neq 0 goto MAIN_MENU
call :RUN_APP_INTERNAL
goto MAIN_MENU

:RUN_ONLY
cls
echo ============================================
echo         Run Application Only
echo ============================================
echo.
call :RUN_APP_INTERNAL
goto MAIN_MENU

:INSTALL_DEPS
cls
echo ============================================
echo       Install/Update Dependencies
echo ============================================
echo.
call :INSTALL_DEPS_INTERNAL
pause
goto MAIN_MENU

:SYSTEM_CHECK
cls
echo ============================================
echo           System Check
echo ============================================
echo.
call :SYSTEM_CHECK_INTERNAL
pause
goto MAIN_MENU

:VIEW_REQUIREMENTS
cls
echo ============================================
echo          System Requirements
echo ============================================
echo.
echo Required Software:
echo - Python 3.8 or higher
echo - pip (Python package manager)
echo.
echo Required Python Packages:
echo - tkinter (GUI framework)
echo - Pillow (Image processing)
echo - opencv-python (Computer vision)
echo - matplotlib (Plotting)
echo - numpy (Numerical computing)
echo - ultralytics (YOLO model)
echo - rasterio (Geospatial data)
echo.
echo Required Files:
echo - dhm.py (Main application)
echo - runs\detect\train\weights\best.pt (YOLO model)
echo.
echo Optional Files:
echo - DSM file (.tif/.tiff) for height analysis
echo - DTM file (.tif/.tiff) for height analysis
echo.
pause
goto MAIN_MENU

:CREATE_SHORTCUT
cls
echo ============================================
echo        Create Desktop Shortcut
echo ============================================
echo.
set SCRIPT_PATH=%~dp0%~nx0
set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT_NAME=YOLO Population Estimator.lnk

echo Creating desktop shortcut...
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP%\%SHORTCUT_NAME%'); $Shortcut.TargetPath = '%SCRIPT_PATH%'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Description = 'YOLO Building & Population Estimator'; $Shortcut.Save()"

if exist "%DESKTOP%\%SHORTCUT_NAME%" (
    echo ✓ Desktop shortcut created successfully!
    echo Location: %DESKTOP%\%SHORTCUT_NAME%
) else (
    echo ✗ Failed to create desktop shortcut
)
echo.
pause
goto MAIN_MENU

:EXIT
echo.
echo Thank you for using YOLO Building & Population Estimator!
echo.
pause
exit /b 0

:: Internal Functions
:SYSTEM_CHECK_INTERNAL
echo Performing system check...
echo.

:: Set script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: Check Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ✗ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    exit /b 1
)
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo ✓ Python !PYTHON_VERSION! is installed

:: Check pip
echo [2/4] Checking pip installation...
pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ✗ ERROR: pip is not available
    echo Please ensure pip is installed with Python
    exit /b 1
)
echo ✓ pip is available

:: Check dhm.py
echo [3/4] Checking for dhm.py...
if not exist "dhm.py" (
    echo ✗ ERROR: dhm.py not found in current directory
    echo Please ensure dhm.py is in the same folder as this batch file
    exit /b 1
)
echo ✓ dhm.py found

:: Check YOLO model
echo [4/4] Checking for YOLO model...
if not exist "runs\detect\train\weights\best.pt" (
    echo ⚠ WARNING: YOLO model not found at runs\detect\train\weights\best.pt
    echo The application may not work without the model file
    echo Please ensure your trained YOLO model is in the correct location
) else (
    echo ✓ YOLO model found
)

echo.
echo System check completed!
echo.
exit /b 0

:INSTALL_DEPS_INTERNAL
echo Installing/Updating dependencies...
echo This may take a few minutes...
echo.

:: Create requirements file
echo Creating requirements file...
(
echo tkinter
echo Pillow
echo opencv-python
echo matplotlib
echo numpy
echo ultralytics
echo rasterio
) > requirements.txt

:: Install packages
echo Installing Python packages...
pip install -r requirements.txt --upgrade
if !errorlevel! neq 0 (
    echo ✗ ERROR: Failed to install required packages
    echo Please check your internet connection and try again
    del requirements.txt
    exit /b 1
)

:: Clean up
del requirements.txt
echo.
echo ✓ All dependencies installed/updated successfully!
echo.
exit /b 0

:RUN_APP_INTERNAL
echo ============================================
echo   Launching YOLO Population Estimator...
echo ============================================
echo.
echo Instructions:
echo 1. Click "Choose Image File" to select your satellite/aerial image
echo 2. Optionally upload DSM and DTM files for height analysis
echo 3. Adjust "People/Residential Building" if needed (default: 4.23)
echo 4. Click "START ANALYSIS" to begin processing
echo.
echo The application window will open shortly...
echo.

:: Small delay
timeout /t 2 /nobreak >nul

:: Launch the application
python dhm.py

if !errorlevel! neq 0 (
    echo.
    echo ============================================
    echo ERROR: Application failed to start
    echo ============================================
    echo.
    echo Possible issues:
    echo - Missing YOLO model file
    echo - Missing required packages
    echo - Python environment issues
    echo.
    echo Please run System Check to diagnose the problem.
    echo.
    pause
    exit /b 1
)

echo.
echo Application closed successfully!
echo.
exit /b 0