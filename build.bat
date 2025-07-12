@echo off
echo Building YOLO Population Estimator...
echo.

echo Installing/Updating requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing requirements!
    pause
    exit /b 1
)

echo.
echo Building executable with PyInstaller...
pyinstaller dhm.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo Error building executable!
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executable location: dist\YOLO_Population_Estimator.exe
echo.
pause
