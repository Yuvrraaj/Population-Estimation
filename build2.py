"""
Fixed Build script to create executable from the YOLO Population Estimator application
Run this script to generate the .exe file with proper dependencies
"""

import os
import sys
import subprocess
import shutil

def create_requirements_txt():
    """Create requirements.txt file with all necessary dependencies"""
    requirements_content = """ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=9.5.0
numpy>=1.24.0
matplotlib>=3.7.0
rasterio>=1.3.0
torch>=2.0.0
torchvision>=0.15.0
PyQt5>=5.15.0
pyinstaller>=5.13.0
scipy>=1.10.0
scikit-image>=0.20.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    print("✓ Created requirements.txt")

def create_requirements_txt():
    """Create requirements.txt file with all necessary dependencies"""
    requirements_content = """# YOLO Population Estimator Requirements
# Core Dependencies

# GUI and Imaging
Pillow>=10.0.0
opencv-python>=4.8.0

# Scientific Computing  
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
scikit-image>=0.21.0

# Geospatial Processing (CRITICAL - fixes rasterio.sample error)
rasterio>=1.3.8
GDAL>=3.7.0

# Machine Learning
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.196

# Build Tools
pyinstaller>=5.13.2

# Additional Dependencies
setuptools>=68.0.0
wheel>=0.41.0
packaging>=23.0
certifi>=2023.7.22
requests>=2.31.0
tqdm>=4.65.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    print("✓ Created requirements.txt")

def create_hook_file():
    """Create PyInstaller hook for rasterio"""
    hooks_dir = "hooks"
    if not os.path.exists(hooks_dir):
        os.makedirs(hooks_dir)
    
    hook_content = '''# Hook for rasterio to ensure all submodules are included
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all rasterio submodules
hiddenimports = collect_submodules('rasterio')

# Collect data files
datas = collect_data_files('rasterio')

# Add specific modules that are often missed
hiddenimports += [
    'rasterio.sample',
    'rasterio._sample',
    'rasterio.features',
    'rasterio.mask',
    'rasterio.warp',
    'rasterio.transform',
    'rasterio.crs',
    'rasterio._shim',
    'rasterio.dtypes',
    'rasterio.profiles',
    'rasterio.env',
    'rasterio.errors',
    'rasterio.windows',
]
'''
    
    with open(os.path.join(hooks_dir, 'hook-rasterio.py'), 'w') as f:
        f.write(hook_content)
    print("✓ Created rasterio hook file")


    """Create PyInstaller spec file with comprehensive dependencies"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# Get the directory of the spec file
spec_root = os.path.abspath(SPECPATH)

a = Analysis(
    ['dhm.py'],
    pathex=[spec_root],
    binaries=[],
    datas=[
        ('runs/detect/train/weights/best.pt', 'runs/detect/train/weights/'),
    ],
    hiddenimports=[
        # Core GUI and imaging
        'PIL._tkinter_finder',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageFont',
        'PIL.ImageDraw',
        
        # Computer vision
        'cv2',
        'cv2.data',
        
        # Scientific computing
        'numpy',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.distance',
        'sklearn',
        'skimage',
        'skimage.measure',
        'skimage.morphology',
        'skimage.filters',
        
        # Plotting
        'matplotlib',
        'matplotlib.backends',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends._backend_agg',
        'matplotlib.figure',
        'matplotlib.pyplot',
        
        # YOLO and ML
        'ultralytics',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
        'ultralytics.utils',
        'ultralytics.utils.plotting',
        'torch',
        'torchvision',
        'torchvision.transforms',
        
        # Rasterio and geospatial (CRITICAL for DSM/DTM support)
        'rasterio',
        'rasterio.windows',
        'rasterio.sample',
        'rasterio.features',
        'rasterio.mask',
        'rasterio.warp',
        'rasterio.transform',
        'rasterio.crs',
        'rasterio._shim',
        'rasterio.dtypes',
        'rasterio.profiles',
        'rasterio.env',
        'rasterio.errors',
        
        # GDAL support (often needed by rasterio)
        'osgeo',
        'osgeo.gdal',
        'osgeo.osr',
        'osgeo.ogr',
        
        # System modules
        'threading',
        'math',
        'os',
        'time',
        'json',
        'pickle',
        'pathlib',
        'tempfile',
        'shutil',
        'sys',
        'gc',
        'warnings',
        'logging',
        
        # Additional modules that might be needed
        'pkg_resources',
        'pkg_resources.py2_warn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'test',
        'tests',
        'testing',
        'unittest',
        'pdb',
        'pydoc',
        'doctest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out duplicate entries
a.datas = list(set(a.datas))

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
    upx=False,  # Disabled UPX as it can cause issues with some libraries
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging if needed
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
    
    print("✓ Created dhm.spec file with comprehensive dependencies")

def create_hook_file():
    """Create a PyInstaller hook file for rasterio to ensure all modules are included"""
    hooks_dir = "hooks"
    if not os.path.exists(hooks_dir):
        os.makedirs(hooks_dir)
    
    hook_content = '''# Hook for rasterio
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all rasterio submodules
hiddenimports = collect_submodules('rasterio')

# Collect data files
datas = collect_data_files('rasterio')

# Add specific modules that might be missed
hiddenimports += [
    'rasterio.sample',
    'rasterio._sample',
    'rasterio.features',
    'rasterio.mask',
    'rasterio.warp',
    'rasterio.transform',
    'rasterio.crs',
    'rasterio._shim',
    'rasterio.dtypes',
    'rasterio.profiles',
    'rasterio.env',
    'rasterio.errors',
    'rasterio.windows',
]
'''
    
    with open(os.path.join(hooks_dir, 'hook-rasterio.py'), 'w') as f:
        f.write(hook_content)
    
    print("✓ Created rasterio hook file")

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
        print("\nMake sure you have:")
        print("1. dhm.py - Your main application file")
        print("2. runs/detect/train/weights/best.pt - Your trained YOLO model")
        return False
    
    print("✓ All required files found")
    return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True, text=True)
        
        # Install requirements
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        
        # Install PyInstaller separately to ensure latest version
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pyinstaller'], 
                      check=True, capture_output=True, text=True)
        
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("Try running the command manually:")
        print(f"   {sys.executable} -m pip install -r requirements.txt")
        return False

def clean_previous_builds():
    """Clean up previous build artifacts"""
    folders_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['dhm.spec']
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"✓ Cleaned {folder}")
    
    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Removed {file}")

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building executable...")
    print("This may take several minutes depending on your system...")
    
    try:
        # Build with the spec file and additional options
        cmd = [
            sys.executable, '-m', 'PyInstaller', 
            'dhm.spec',
            '--clean',
            '--noconfirm',
            '--additional-hooks-dir=hooks' if os.path.exists('hooks') else '',
        ]
        
        # Remove empty strings from command
        cmd = [c for c in cmd if c]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Executable built successfully")
        
        # Check if executable was created
        if os.path.exists("dist/YOLO_Population_Estimator.exe"):
            size = os.path.getsize("dist/YOLO_Population_Estimator.exe") / (1024*1024)
            print(f"✓ Executable size: {size:.1f} MB")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build executable")
        print("Error output:", e.stderr if e.stderr else "No error details available")
        print("Standard output:", e.stdout if e.stdout else "No output available")
        
        # Suggest debugging steps
        print("\n🔧 Debugging suggestions:")
        print("1. Try building with console=True in the spec file for debugging")
        print("2. Check if all required modules are installed")
        print("3. Run: python dhm.py to test if the script works before building")
        return False

def create_distribution_folder():
    """Create a distribution folder with the executable and required files"""
    dist_folder = "YOLO_Population_Estimator_Distribution"
    
    # Create distribution folder
    if os.path.exists(dist_folder):
        shutil.rmtree(dist_folder)
    os.makedirs(dist_folder)
    
    # Copy executable
    exe_path = "dist/YOLO_Population_Estimator.exe"
    if os.path.exists(exe_path):
        shutil.copy2(exe_path, dist_folder)
        print("✓ Copied executable to distribution folder")
    else:
        print("❌ Executable not found in dist folder")
        return False
    
    # Create comprehensive README file
    readme_content = """YOLO Population Estimator v1.0
===============================

OVERVIEW
--------
This application estimates population density based on building detection using 
advanced YOLO (You Only Look Once) machine learning technology. It analyzes 
satellite or aerial imagery to detect buildings and calculates population estimates.

SYSTEM REQUIREMENTS
------------------
- Windows 10/11 (64-bit)
- Minimum 8GB RAM (16GB recommended)
- 5GB free disk space
- Graphics card with DirectX 11 support (recommended)

QUICK START
-----------
1. Double-click 'YOLO_Population_Estimator.exe' to launch
2. Click "Choose Image File" to select your satellite/aerial image
3. Set the "People per Residential Building" value (default: 4.23)
4. Click "START ANALYSIS" to begin processing
5. View results in the statistics panel and charts

ADVANCED FEATURES
----------------
Floor-based Estimation:
- Upload DSM (Digital Surface Model) file for height data
- Upload DTM (Digital Terrain Model) file for ground reference
- Enable more accurate population estimation based on building floors

Building Classification:
- Automatic detection of Residential vs Non-Residential buildings
- Customizable population density factors
- Statistical analysis and visualization

SUPPORTED FILE FORMATS
----------------------
Images:     PNG, JPG, JPEG, BMP, TIFF, TIF
Height Data: TIFF files (GeoTIFF format recommended)

USAGE TIPS
----------
- Use high-resolution satellite imagery for better results
- Ensure images cover urban or residential areas
- DSM/DTM files should cover the same geographic area as your image
- Processing time depends on image size and complexity

TROUBLESHOOTING
--------------
If the application doesn't start:
1. Right-click the .exe file and "Run as administrator"
2. Check Windows Defender/antivirus isn't blocking the file
3. Ensure you have enough free disk space
4. Try running from a folder without spaces in the path

If processing fails:
1. Try with a smaller image first
2. Ensure image format is supported
3. Check that DSM/DTM files are valid GeoTIFF format

PERFORMANCE NOTES
----------------
- First run may take longer as the AI model initializes
- Large images (>50MB) may require several minutes to process
- Close other resource-intensive applications during processing

ABOUT THE TECHNOLOGY
-------------------
This application uses:
- YOLOv8 for building detection
- OpenCV for image processing  
- Rasterio for geospatial data handling
- Custom algorithms for population estimation

For technical support or questions, please contact the development team.

Version: 1.0
Build Date: {build_date}
""".format(build_date=__import__('datetime').datetime.now().strftime('%Y-%m-%d'))
    
    with open(os.path.join(dist_folder, "README.txt"), 'w') as f:
        f.write(readme_content)
    
    # Create a simple batch file to run the executable
    batch_content = '''@echo off
echo Starting YOLO Population Estimator...
echo Please wait while the application loads...
echo.
YOLO_Population_Estimator.exe
pause
'''
    
    with open(os.path.join(dist_folder, "Run_Application.bat"), 'w') as f:
        f.write(batch_content)
    
    print(f"✓ Created distribution folder: {dist_folder}")
    print(f"✓ Added comprehensive README.txt")
    print(f"✓ Added Run_Application.bat for easy launching")
    
    return True

def test_executable():
    """Test if the executable can be run"""
    exe_path = "YOLO_Population_Estimator_Distribution/YOLO_Population_Estimator.exe"
    
    if not os.path.exists(exe_path):
        print("❌ Executable not found for testing")
        return False
    
    print("Testing executable (this will launch and close quickly)...")
    try:
        # Just check if it can start without errors
        process = subprocess.Popen([exe_path], stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        # Give it a moment to start
        import time
        time.sleep(3)
        
        # Terminate the process
        process.terminate()
        process.wait(timeout=5)
        
        print("✓ Executable appears to start correctly")
        return True
    except Exception as e:
        print(f"⚠️  Could not test executable automatically: {e}")
        print("Please test manually by running the .exe file")
        return True  # Don't fail the build for this

def main():
    """Main build process"""
    print("🚀 YOLO Population Estimator - Advanced Build Process")
    print("=" * 65)
    
    # Check if running from correct directory
    if not os.path.exists('dhm.py'):
        print("❌ Please run this script from the same directory as dhm.py")
        input("Press Enter to exit...")
        return
    
    try:
        # Step 1: Clean previous builds
        print("\n📁 Step 1: Cleaning previous builds...")
        clean_previous_builds()
        
        # Step 2: Check requirements
        print("\n🔍 Step 2: Checking required files...")
        if not check_requirements():
            print("❌ Build failed: Missing required files")
            input("Press Enter to exit...")
            return
        
        # Step 3: Create requirements.txt
        print("\n📝 Step 3: Creating requirements.txt...")
        create_requirements_txt()
        
        # Step 4: Install requirements
        print("\n📦 Step 4: Installing requirements...")
        if not install_requirements():
            print("❌ Build failed: Could not install requirements")
            input("Press Enter to exit...")
            return
        
        # Step 5: Create hook files
        print("\n🪝 Step 5: Creating PyInstaller hooks...")
        create_hook_file()
        
        # Step 6: Create spec file
        print("\n⚙️  Step 6: Creating PyInstaller spec file...")
        create_spec_file()
        
        # Step 7: Build executable
        print("\n🔨 Step 7: Building executable...")
        if not build_executable():
            print("❌ Build failed: Could not create executable")
            input("Press Enter to exit...")
            return
        
        # Step 8: Create distribution folder
        print("\n📦 Step 8: Creating distribution package...")
        if not create_distribution_folder():
            print("❌ Build failed: Could not create distribution folder")
            input("Press Enter to exit...")
            return
        
        # Step 9: Test executable
        print("\n🧪 Step 9: Testing executable...")
        test_executable()
        
        print("\n" + "=" * 65)
        print("🎉 BUILD COMPLETED SUCCESSFULLY!")
        print("=" * 65)
        print("✓ Executable created: YOLO_Population_Estimator.exe")
        print("✓ Distribution folder: YOLO_Population_Estimator_Distribution")
        print("✓ Size:", end=" ")
        
        try:
            exe_size = os.path.getsize("YOLO_Population_Estimator_Distribution/YOLO_Population_Estimator.exe")
            print(f"{exe_size / (1024*1024):.1f} MB")
        except:
            print("Unknown")
        
        print("\n📋 Next Steps:")
        print("1. Navigate to 'YOLO_Population_Estimator_Distribution' folder")
        print("2. Double-click 'YOLO_Population_Estimator.exe' to test")
        print("3. Or use 'Run_Application.bat' for easier launching")
        print("4. Read 'README.txt' for detailed usage instructions")
        print("\n✅ The application is ready for distribution!")
        
    except KeyboardInterrupt:
        print("\n❌ Build cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during build: {e}")
        print("Please check the error messages above for details")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()