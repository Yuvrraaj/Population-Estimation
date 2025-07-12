"""
Build script to create executable from the YOLO Population Estimator application
Run this script to generate the .exe file
"""

import os
import sys
import subprocess
import shutil

def create_spec_file():
    """Create PyInstaller spec file for better control over the build process"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['dhm.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('runs/detect/train/weights/best.pt', 'runs/detect/train/weights/'),
    ],
    hiddenimports=[
        'PIL._tkinter_finder',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'cv2',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.figure',
        'numpy',
        'ultralytics',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'rasterio',
        'rasterio.windows',
        'threading',
        'math',
        'os',
        'time'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='YOLO_Population_Estimator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico' if os.path.exists('app_icon.ico') else None,
)
'''
    
    with open('dhm.spec', 'w') as f:
        f.write(spec_content)
    
    print("✓ Created dhm.spec file")

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'dhm.py',
        'runs/detect/train/weights/best.pt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✓ All required files found")
    return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building executable...")
    try:
        # Use the spec file for building
        subprocess.run([sys.executable, '-m', 'PyInstaller', 'dhm.spec', '--clean'], 
                      check=True)
        print("✓ Executable built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build executable: {e}")
        return False

def create_distribution_folder():
    """Create a distribution folder with the executable and required files"""
    dist_folder = "YOLO_Population_Estimator_Distribution"
    
    # Create distribution folder
    if os.path.exists(dist_folder):
        shutil.rmtree(dist_folder)
    os.makedirs(dist_folder)
    
    # Copy executable
    if os.path.exists("dist/YOLO_Population_Estimator.exe"):
        shutil.copy2("dist/YOLO_Population_Estimator.exe", dist_folder)
        print("✓ Copied executable to distribution folder")
    
    # Create README file
    readme_content = """YOLO Population Estimator
========================

This application estimates population based on building detection using YOLO.

How to use:
1. Run YOLO_Population_Estimator.exe
2. Click "Choose Image File" to select your satellite/aerial image
3. (Optional) Upload DSM and DTM files for floor-based estimation
4. Set the "People/Residential Building" value (default: 4.23)
5. Click "START ANALYSIS" to begin processing

Features:
- Area-based population estimation
- Floor-based population estimation (requires DSM/DTM files)
- Building classification (Residential/Non-Residential)
- Statistical charts and analysis
- Modern, user-friendly interface

Supported file formats:
- Images: PNG, JPG, JPEG, BMP, TIFF, TIF
- Height data: TIFF files (DSM/DTM)

Requirements:
- Windows 10/11
- Minimum 4GB RAM
- 2GB free disk space

For support or questions, please contact the development team.
"""
    
    with open(os.path.join(dist_folder, "README.txt"), 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Created distribution folder: {dist_folder}")
    print(f"✓ Added README.txt with usage instructions")

def main():
    """Main build process"""
    print("🚀 Starting YOLO Population Estimator Build Process")
    print("=" * 60)
    
    # Check if running from correct directory
    if not os.path.exists('dhm.py'):
        print("❌ Please run this script from the same directory as dhm.py")
        return
    
    # Step 1: Check requirements
    if not check_requirements():
        print("❌ Build failed: Missing required files")
        return
    
    # Step 2: Create requirements.txt if it doesn't exist
    if not os.path.exists('requirements.txt'):
        print("Creating requirements.txt...")
        # Create requirements.txt content here or copy from the artifact above
    
    # Step 3: Install requirements
    if not install_requirements():
        print("❌ Build failed: Could not install requirements")
        return
    
    # Step 4: Create spec file
    create_spec_file()
    
    # Step 5: Build executable
    if not build_executable():
        print("❌ Build failed: Could not create executable")
        return
    
    # Step 6: Create distribution folder
    create_distribution_folder()
    
    print("\n" + "=" * 60)
    print("🎉 BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Your executable is ready in the 'YOLO_Population_Estimator_Distribution' folder")
    print("You can now distribute this folder to users.")
    print("\nTo test the executable:")
    print("1. Navigate to the distribution folder")
    print("2. Double-click 'YOLO_Population_Estimator.exe'")

if __name__ == "__main__":
    main()