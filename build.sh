#!/bin/bash
echo "Building YOLO Population Estimator..."
echo

echo "Installing/Updating requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing requirements!"
    exit 1
fi

echo
echo "Building executable with PyInstaller..."
pyinstaller dhm.spec --clean --noconfirm
if [ $? -ne 0 ]; then
    echo "Error building executable!"
    exit 1
fi

echo
echo "Build completed successfully!"
echo "Executable location: dist/YOLO_Population_Estimator"
echo
