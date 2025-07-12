"""
Test script to verify all required modules can be imported
Run this before building the executable to catch import issues early
"""

import sys

def test_import(module_name, description=""):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {module_name:<25} {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name:<25} FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name:<25} WARNING: {str(e)}")
        return True

def main():
    print("🧪 Testing Module Imports for YOLO Population Estimator")
    print("=" * 60)
    
    failed_imports = []
    
    # Core GUI modules
    print("\n📱 GUI and Imaging Modules:")
    modules = [
        ("tkinter", "GUI framework"),
        ("tkinter.filedialog", "File dialogs"),
        ("tkinter.messagebox", "Message boxes"),
        ("tkinter.ttk", "Themed widgets"),
        ("PIL", "Python Imaging Library"),
        ("PIL.Image", "Image processing"),
        ("PIL.ImageTk", "Tkinter image support"),
    ]
    
    for module, desc in modules:
        if not test_import(module, desc):
            failed_imports.append(module)
    
    # Scientific computing
    print("\n🔬 Scientific Computing:")
    modules = [
        ("numpy", "Numerical computing"),
        ("scipy", "Scientific computing"),
        ("matplotlib", "Plotting library"),
        ("matplotlib.pyplot", "Plotting interface"),
        ("cv2", "OpenCV computer vision"),
    ]
    
    for module, desc in modules:
        if not test_import(module, desc):
            failed_imports.append(module)
    
    # Machine Learning
    print("\n🤖 Machine Learning:")
    modules = [
        ("torch", "PyTorch deep learning"),
        ("torchvision", "Computer vision for PyTorch"),
        ("ultralytics", "YOLO implementation"),
    ]
    
    for module, desc in modules:
        if not test_import(module, desc):
            failed_imports.append(module)
    
    # Geospatial modules (CRITICAL)
    print("\n🗺️  Geospatial Modules (Critical for DSM/DTM):")
    modules = [
        ("rasterio", "Geospatial raster data"),
        ("rasterio.sample", "Rasterio sampling"),
        ("rasterio.windows", "Rasterio windowing"),
        ("rasterio.features", "Rasterio features"),
        ("rasterio.mask", "Rasterio masking"),
        ("rasterio.warp", "Rasterio warping"),
    ]
    
    for module, desc in modules:
        if not test_import(module, desc):
            failed_imports.append(module)
    
    # Test specific functionality
    print("\n🔧 Testing Specific Functionality:")
    
    # Test rasterio.sample specifically (this was causing the error)
    try:
        import rasterio.sample
        print("✓ rasterio.sample        Direct import successful")
    except Exception as e:
        print(f"❌ rasterio.sample        CRITICAL ERROR: {e}")
        failed_imports.append("rasterio.sample")
    
    # Test YOLO model loading capability
    try:
        from ultralytics import YOLO
        print("✓ YOLO class import      Model loading capability")
    except Exception as e:
        print(f"❌ YOLO class import     ERROR: {e}")
        failed_imports.append("ultralytics.YOLO")
    
    # Test matplotlib backend
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # This is what the app uses
        print("✓ matplotlib TkAgg       Backend configuration")
    except Exception as e:
        print(f"❌ matplotlib TkAgg      ERROR: {e}")
        failed_imports.append("matplotlib.TkAgg")
    
    print("\n" + "=" * 60)
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} MODULES FAILED TO IMPORT")
        print("Failed modules:", ", ".join(failed_imports))
        print("\n🔧 To fix these issues:")
        print("1. Run: pip install -r requirements.txt")
        print("2. For rasterio issues: pip install rasterio --force-reinstall")
        print("3. For PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("4. For OpenCV: pip install opencv-python")
        print("\n❌ DO NOT BUILD EXECUTABLE until all imports pass!")
        return False
    else:
        print("✅ ALL MODULES IMPORTED SUCCESSFULLY!")
        print("✅ Ready to build executable")
        
        # Additional system info
        print(f"\n📊 System Information:")
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        
        # Test file access
        if __import__('os').path.exists('runs/detect/train/weights/best.pt'):
            print("✓ YOLO model file found")
        else:
            print("❌ YOLO model file missing: runs/detect/train/weights/best.pt")
            return False
        
        return True

if __name__ == "__main__":
    success = main()
    input(f"\nPress Enter to {'continue with build' if success else 'exit'}...")