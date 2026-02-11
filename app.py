import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dst
from ultralytics import YOLO
import threading
import rasterio
import rasterio.warp
import rasterio.windows
from rasterio.windows import Window, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling
import tempfile
import shutil
import csv
from datetime import datetime
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import sys
import os

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Update any file path references in your code to use resource_path()
# For example, if you have icon loading:
# icon_path = resource_path("assets/app.ico")
# root.iconbitmap(icon_path)

# Add this after your imports in app4.py
import os
import logging
import sys
from pathlib import Path

# Setup application data directory
def setup_app_directories():
    """Setup application directories in user space"""
    base_dir = Path(os.getenv("LOCALAPPDATA")) / "PopulationEstimator"
    logs_dir = base_dir / "logs"
    cache_dir = base_dir / "cache"
    
    # Create directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = logs_dir / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Population Estimator started")
    logging.info(f"App data directory: {base_dir}")
    logging.info(f"Log file: {log_file}")
    
    return base_dir, cache_dir

# Call this early in your app
try:
    APP_DATA_DIR, CACHE_DIR = setup_app_directories()
except Exception as e:
    logging.warning(f"Could not setup app directories: {e}")
    APP_DATA_DIR = Path.cwd()
    CACHE_DIR = Path.cwd() / "cache"
    CACHE_DIR.mkdir(exist_ok=True)


# Try to import optional packages
try:
    import seaborn as sns
    import pandas as pd
    from scipy import stats
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False
    print("Advanced visualization libraries not available. Basic charts will work.")

# Set matplotlib style for better looking charts
plt.style.use('default')
if HAS_ADVANCED_LIBS:
    sns.set_palette("husl")

# Configure matplotlib for better chart appearance
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

# ==== Load YOLO Model ====
model = YOLO("best.pt")
  # Update this path

# ==== Constants ====
colors = {0: (255, 0, 255), 1: (0, 165, 255)}
class_names = {0: 'Residential', 1: 'Non-Residential'}

class ScrollableFrame:
    """A scrollable frame that can contain other widgets"""
    def __init__(self, parent, bg='white'):
        # Main frame
        self.main_frame = tk.Frame(parent, bg=bg)
        
        # Create canvas and scrollbars
        self.canvas = tk.Canvas(self.main_frame, bg=bg, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.main_frame, orient="horizontal", command=self.canvas.xview)
        
        # Create the scrollable frame
        self.scrollable_frame = tk.Frame(self.canvas, bg=bg)
        
        # Configure canvas
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Bind mousewheel to canvas
        self.canvas.bind('<Button-4>', self._on_mousewheel)
        self.canvas.bind('<Button-5>', self._on_mousewheel)
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        
        # Create window in canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure scrolling
        self.scrollable_frame.bind('<Configure>', self._configure_scrollregion)
        self.canvas.bind('<Configure>', self._configure_canvas_size)
        
        # Pack elements
        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.delta:
            delta = -1 * int(event.delta / 120)
        else:
            delta = -1 if event.num == 4 else 1
        self.canvas.yview_scroll(delta, "units")

    def _configure_scrollregion(self, event):
        """Update scroll region when content changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _configure_canvas_size(self, event):
        """Resize the canvas window to fit the canvas"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def pack(self, **kwargs):
        self.main_frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.main_frame.grid(**kwargs)

class BuildingCountPopulationEstimator:
    def __init__(self, root):
        self.root = root
        self.validation_label = None
        
        # Parameters - Updated for building count approach
        self.people_per_household = tk.DoubleVar(value=4.23)
        self.ground_sample_distance = 0.5  # meters per pixel (for area calculation)
        self.standard_house_area = tk.DoubleVar(value=100.0)  # Standard house area in m²
        
        # Tile area configuration for TIF processing
        self.tile_area_sqm = tk.DoubleVar(value=10000.0)  # Default 10,000 m² (100m x 100m)
        
        # Area multiplier thresholds
        self.medium_house_threshold = tk.DoubleVar(value=150.0)  # 1.5x standard = 2x multiplier
        self.large_house_threshold = tk.DoubleVar(value=250.0)   # 2.5x standard = 3x multiplier
        
        # Floor calculation parameters
        self.floor_height = tk.DoubleVar(value=3.0)  # Average floor height in meters
        self.use_height_calculation = tk.BooleanVar(value=False)  # Enable/disable height-based floors
        
        # File handling
        self.file_path = None
        self.dhm_path = None
        self.dtm_path = None
        self.tiles_save_path = None
        self.is_tif_file = False
        self.tiles_generated = False
        self.height_data_aligned = False
        
        # Height calculation data
        self.dsm_data = None
        self.dtm_data = None
        self.height_transform = None
        self.height_bounds = None
        
        # **NEW**: Coordinate-aware tile processing
        self.tile_metadata_list = []  # Store tile metadata with coordinates
        
        # Analysis results storage
        self.analysis_results = None
        self.building_data = []
        
        # UI State
        self.progress_var = tk.StringVar()
        self.progress_value = tk.IntVar()
        
        self.setup_window()
        self.create_widgets()

    def setup_window(self):
        """Setup main window with responsive design"""
        self.root.title("🏘️ Population Estimator - Coordinate-Aware Building Count with Floor Calculation")

        # Try to maximize in a cross-platform safe way
        try:
            # Works on Windows
            self.root.state('zoomed')
        except tk.TclError:
            try:
                # Sometimes works on Linux (X11 backends)
                self.root.attributes('-zoomed', True)
            except tk.TclError:
                # Fallback → just open normally
                self.root.state('normal')

        self.root.configure(bg="#2c3e50")
        self.root.minsize(1200, 700)
        
        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Set window icon (if available)
        try:
            self.root.iconbitmap('icon.ico')
        except Exception:
            pass


    def create_widgets(self):
        """Create the main widget structure with improved layout"""
        # Main container
        main_container = tk.Frame(self.root, bg="#2c3e50")
        main_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        
        # Header
        self.create_header(main_container)
        
        # Create main paned window for resizable panels
        self.paned_window = tk.PanedWindow(main_container, orient=tk.HORIZONTAL,
                                         bg="#2c3e50", sashwidth=8, sashrelief=tk.RAISED)
        self.paned_window.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5, 0))
        
        # Left panel (controls) - Scrollable
        self.create_left_panel()
        
        # Right panel (results) - Scrollable
        self.create_right_panel()
        
        # Bottom status bar
        self.create_status_bar(main_container)
        
        # Progress bar
        self.create_progress_bar(main_container)

    def create_header(self, parent):
        """Create header with modern design"""
        header_frame = tk.Frame(parent, bg="#34495e", height=70)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Logo/Icon area
        icon_frame = tk.Frame(header_frame, bg="#34495e", width=70)
        icon_frame.pack(side="left", fill="y", padx=10)
        icon_frame.pack_propagate(False)
        
        icon_label = tk.Label(icon_frame, text="🏘️", font=('Arial', 28),
                            bg="#34495e", fg="white")
        icon_label.pack(expand=True)
        
        # Title area
        title_frame = tk.Frame(header_frame, bg="#34495e")
        title_frame.pack(side="left", expand=True, fill="both", padx=(0, 10))
        
        title_label = tk.Label(title_frame, text="Population Estimator - Coordinate-Aware Building Count with Floor Calculation",
                             font=('Arial', 18, 'bold'), fg="white", bg="#34495e")
        title_label.pack(anchor="w", pady=(8, 2))
        
        subtitle_label = tk.Label(title_frame,
                                text="AI-powered population estimation with coordinate-aware processing and height-based floor calculation",
                                font=('Arial', 10), fg="#bdc3c7", bg="#34495e")
        subtitle_label.pack(anchor="w")
        
        # Quick stats area (will be updated during analysis)
        self.quick_stats_frame = tk.Frame(header_frame, bg="#34495e")
        self.quick_stats_frame.pack(side="right", padx=10)

    def create_left_panel(self):
        """Create left control panel with scrolling"""
        # Left panel main frame
        left_main = tk.Frame(self.paned_window, bg="white", relief=tk.RIDGE, bd=2)
        
        # Create scrollable frame for left panel
        self.left_scroll = ScrollableFrame(left_main, bg="white")
        self.left_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create control sections in scrollable frame
        self.create_control_sections(self.left_scroll.scrollable_frame)
        
        # Add to paned window
        self.paned_window.add(left_main, minsize=450, width=500)

    def create_right_panel(self):
        """Create right results panel with scrolling"""
        # Right panel main frame
        right_main = tk.Frame(self.paned_window, bg="white", relief=tk.RIDGE, bd=2)
        
        # Create scrollable frame for right panel
        self.right_scroll = ScrollableFrame(right_main, bg="white")
        self.right_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create results sections in scrollable frame
        self.create_results_sections(self.right_scroll.scrollable_frame)
        
        # Add to paned window
        self.paned_window.add(right_main, minsize=600)

    def create_control_sections(self, parent):
        """Create all control sections with improved styling"""
        # File Upload Section
        self.create_file_section(parent)
        
        # Separator
        self.create_separator(parent)
        
        # Height Data Section (DHM/DTM)
        self.create_height_data_section(parent)
        
        # Separator
        self.create_separator(parent)
        
        # Parameters Section
        self.create_parameters_section(parent)
        
        # Separator
        self.create_separator(parent)
        
        # TIF Processing Section
        self.create_tif_section(parent)
        
        # Separator
        self.create_separator(parent)
        
        # NEW: Tile Upload Section
        self.create_tile_upload_section(parent)
        
        # Separator
        self.create_separator(parent)
        
        # Analysis Section
        self.create_analysis_section(parent)
        
        # Separator
        self.create_separator(parent)
        
        # CSV Export Section
        self.create_csv_export_section(parent)
        
        # Separator
        self.create_separator(parent)
        
        # Visualization Controls
        self.create_visualization_controls(parent)
    

    def create_separator(self, parent):
        """Create a visual separator"""
        separator = tk.Frame(parent, height=2, bg="#bdc3c7")
        separator.pack(fill="x", padx=20, pady=10)

    def create_file_section(self, parent):
        """Create file upload section with enhanced styling"""
        section_frame = tk.LabelFrame(parent, text="📁 Primary Image File", font=('Arial', 12, 'bold'),
                                    bg="white", fg="#2c3e50", padx=15, pady=15)
        section_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Upload button with hover effect
        self.upload_btn = tk.Button(section_frame, text="📁 Choose Image File",
                                  command=self.choose_image, bg="#3498db", fg="white",
                                  font=('Arial', 11, 'bold'), height=2, relief=tk.RAISED,
                                  cursor="hand2", activebackground="#2980b9")
        self.upload_btn.pack(fill="x", pady=(0, 10))
        
        # File info display
        info_frame = tk.Frame(section_frame, bg="white")
        info_frame.pack(fill="x")
        
        self.file_info = tk.Label(info_frame, text="No file selected",
                                bg="white", font=('Arial', 10), wraplength=400, justify="left",
                                fg="#7f8c8d")
        self.file_info.pack(anchor="w", pady=(0, 5))
        
        self.file_type_info = tk.Label(info_frame, text="",
                                     bg="white", font=('Arial', 9, 'italic'), fg="#95a5a6")
        self.file_type_info.pack(anchor="w")

    def create_height_data_section(self, parent):
        """Create height data (DHM/DTM) upload section"""
        self.height_frame = tk.LabelFrame(parent, text="🗻 Height Data (Optional - For Floor Calculation)",
                                        font=('Arial', 12, 'bold'),
                                        bg="white", fg="#2c3e50", padx=15, pady=15)
        self.height_frame.pack(fill="x", padx=10, pady=5)
        self.height_frame.pack_forget()  # keep hidden until a TIF is uploaded ✅

        
        # Enable/Disable height calculation
        enable_frame = tk.Frame(self.height_frame, bg="white")
        enable_frame.pack(fill="x", pady=(0, 10))
        
        self.enable_height_cb = tk.Checkbutton(enable_frame, text="Enable Floor-based Population Calculation",
                                             variable=self.use_height_calculation,
                                             command=self.toggle_height_calculation,
                                             bg="white", font=('Arial', 11, 'bold'),
                                             fg="#2c3e50", selectcolor="#3498db", cursor="hand2")
        self.enable_height_cb.pack(anchor="w")
        
        # Height calculation frame (initially disabled)
        self.height_calc_frame = tk.Frame(self.height_frame, bg="white")
        self.height_calc_frame.pack(fill="x", pady=(10, 0))
        
        # DHM File Upload
        dhm_frame = tk.Frame(self.height_calc_frame, bg="white")
        dhm_frame.pack(fill="x", pady=(0, 10))

        tk.Label(dhm_frame, text="📐 DSM File (Digital Surface Model):", bg="white",
               font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w")
        
        dhm_input_frame = tk.Frame(dhm_frame, bg="white")
        dhm_input_frame.pack(fill="x", pady=(5, 0))
        
        self.dhm_btn = tk.Button(dhm_input_frame, text="📁 Select DSM File",
                               command=self.choose_dhm_file, bg="#e67e22", fg="white",
                               font=('Arial', 10, 'bold'), cursor="hand2",
                               relief=tk.RAISED, activebackground="#d35400")
        self.dhm_btn.pack(side="left")
        
        self.dhm_info = tk.Label(dhm_input_frame, text="No DSM file selected",
                               bg="white", font=('Arial', 10), fg="#7f8c8d")
        self.dhm_info.pack(side="left", padx=(15, 0))
        
        # DTM File Upload
        dtm_frame = tk.Frame(self.height_calc_frame, bg="white")
        dtm_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(dtm_frame, text="🌍 DTM File (Digital Terrain Model):", bg="white",
               font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w")
        
        dtm_input_frame = tk.Frame(dtm_frame, bg="white")
        dtm_input_frame.pack(fill="x", pady=(5, 0))
        
        self.dtm_btn = tk.Button(dtm_input_frame, text="📁 Select DTM File",
                               command=self.choose_dtm_file, bg="#e67e22", fg="white",
                               font=('Arial', 10, 'bold'), cursor="hand2",
                               relief=tk.RAISED, activebackground="#d35400")
        self.dtm_btn.pack(side="left")
        
        self.dtm_info = tk.Label(dtm_input_frame, text="No DTM file selected",
                               bg="white", font=('Arial', 10), fg="#7f8c8d")
        self.dtm_info.pack(side="left", padx=(15, 0))
        
        # Floor height parameter
        floor_frame = tk.Frame(self.height_calc_frame, bg="white")
        floor_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(floor_frame, text="📏 Average Floor Height (meters):", bg="white",
               font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w")
        
        self.floor_height_entry = tk.Entry(floor_frame, textvariable=self.floor_height,
                                         font=('Arial', 10), width=10, relief=tk.SOLID, bd=1)
        self.floor_height_entry.pack(anchor="w", pady=(5, 0))
        
        # Alignment status
        self.alignment_status = tk.Label(self.height_calc_frame, text="",
                                       bg="white", font=('Arial', 10))
        self.alignment_status.pack(anchor="w", pady=(10, 0))
        
        # Info box about coordinate-aware height calculation
        self.create_info_box(self.height_calc_frame, "Coordinate-Aware Height-based Floor Calculation:", [
            "• Tiles saved as GeoTIFF with coordinate information",
            "• DHM - DTM = Building Height above ground per tile",
            "• Building Height ÷ Floor Height = Number of Floors",
            "• Population = Base Population × Number of Floors",
            "• Files must have matching coordinate systems",
            "• Coordinate alignment is automatically verified"
        ])
        
        # Initially disable the height calculation frame
        self.toggle_height_calculation()

    def toggle_height_calculation(self):
        """Toggle height calculation controls"""
        if self.use_height_calculation.get():
            # Enable all controls in height calculation frame
            for widget in self.height_calc_frame.winfo_children():
                self.enable_widget_tree(widget)
        else:
            # Disable all controls in height calculation frame
            for widget in self.height_calc_frame.winfo_children():
                self.disable_widget_tree(widget)
        
        self.update_validation_status()

    def enable_widget_tree(self, widget):
        """Recursively enable widget and its children"""
        try:
            widget.config(state='normal')
        except:
            pass
        for child in widget.winfo_children():
            self.enable_widget_tree(child)

    def disable_widget_tree(self, widget):
        """Recursively disable widget and its children"""
        try:
            widget.config(state='disabled')
        except:
            pass
        for child in widget.winfo_children():
            self.disable_widget_tree(child)

    def choose_dhm_file(self):
        """Choose DHM file"""
        file_path = filedialog.askopenfilename(
            title="Select DHM File (Digital Height Model)",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            self.dhm_path = file_path
            filename = os.path.basename(file_path)
            self.dhm_info.config(text=f"📐 {filename}", fg="#27ae60")
            self.update_status(f"DHM file selected: {filename}", "📐")
            self.check_height_data_alignment()

    def choose_dtm_file(self):
        """Choose DTM file"""
        file_path = filedialog.askopenfilename(
            title="Select DTM File (Digital Terrain Model)",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            self.dtm_path = file_path
            filename = os.path.basename(file_path)
            self.dtm_info.config(text=f"🌍 {filename}", fg="#27ae60")
            self.update_status(f"DTM file selected: {filename}", "🌍")
            self.check_height_data_alignment()

    def check_height_data_alignment(self):
        """Check if DHM and DTM files are coordinate-aligned"""
        if not (self.dhm_path and self.dtm_path and self.file_path and self.is_tif_file):
            self.alignment_status.config(text="", fg="#7f8c8d")
            self.height_data_aligned = False
            return
        
        try:
            # Check coordinate alignment
            with rasterio.open(self.file_path) as src_img:
                with rasterio.open(self.dhm_path) as src_dhm:
                    with rasterio.open(self.dtm_path) as src_dtm:
                        # Check CRS compatibility
                        img_crs = src_img.crs
                        dhm_crs = src_dhm.crs
                        dtm_crs = src_dtm.crs
                        
                        # Check bounds overlap
                        img_bounds = src_img.bounds
                        dhm_bounds = src_dhm.bounds
                        dtm_bounds = src_dtm.bounds
                        
                        # Calculate overlap
                        overlap_exists = self.check_bounds_overlap(img_bounds, dhm_bounds) and \
                                       self.check_bounds_overlap(img_bounds, dtm_bounds) and \
                                       self.check_bounds_overlap(dhm_bounds, dtm_bounds)
                        
                        if overlap_exists:
                            self.alignment_status.config(text="✅ Height data alignment verified", fg="#27ae60")
                            self.height_data_aligned = True
                            self.update_status("Height data files are properly aligned", "✅")
                        else:
                            self.alignment_status.config(text="❌ Height data not aligned with image", fg="#e74c3c")
                            self.height_data_aligned = False
                            self.update_status("Warning: Height data files not aligned with image", "⚠️")
                            messagebox.showwarning("Alignment Warning",
                                                 "Height data files (DHM/DTM) do not appear to be properly aligned with the image file.\n"
                                                 "This may result in incorrect floor calculations.\n\n"
                                                 "Please ensure all files have matching coordinate systems and overlapping areas.")
        
        except Exception as e:
            self.alignment_status.config(text="❌ Error checking alignment", fg="#e74c3c")
            self.height_data_aligned = False
            print(f"Error checking alignment: {e}")

    def check_bounds_overlap(self, bounds1, bounds2):
        """Check if two bounding boxes overlap"""
        return not (bounds1[2] <= bounds2[0] or bounds2[2] <= bounds1[0] or
                   bounds1[3] <= bounds2[1] or bounds2[3] <= bounds1[1])

    # **NEW**: Coordinate-Aware Tile Creation Method
    def create_tiles_from_tif_with_coordinates(self, tif_path, output_dir, tile_area_sqm):
        """Create tiles based on AREA (square meters) while preserving coordinate information"""
        import os
        import numpy as np
        import rasterio
        from rasterio.windows import Window

        os.makedirs(output_dir, exist_ok=True)
        tile_metadata_list = []

        try:
            with rasterio.open(tif_path) as src:
                # Store original transform and CRS
                original_transform = src.transform
                original_crs = src.crs
                total_width = src.width
                total_height = src.height
                bounds = src.bounds
                
                # CRITICAL: Get the Ground Sample Distance (GSD) from the transform
                # The transform contains pixel size in map units (usually meters)
                pixel_width_meters = abs(original_transform[0])  # X-direction pixel size
                pixel_height_meters = abs(original_transform[4])  # Y-direction pixel size
                
                # Use average if pixels are not square (though they usually are)
                gsd = (pixel_width_meters + pixel_height_meters) / 2.0
                
                print(f"\n{'='*70}")
                print(f"AREA-BASED TILE GENERATION")
                print(f"{'='*70}")
                print(f"Image path: {os.path.basename(tif_path)}")
                print(f"Image dimensions: {total_width}x{total_height} pixels")
                print(f"CRS: {original_crs}")
                print(f"Extent: X({bounds.left:.2f} to {bounds.right:.2f}), Y({bounds.bottom:.2f} to {bounds.top:.2f})")
                print(f"Ground Sample Distance (GSD): {pixel_width_meters} x {pixel_height_meters} meters/pixel")
                print(f"Average GSD: {gsd:.4f} meters/pixel")
                
                # Calculate tile size in PIXELS based on desired AREA
                tile_side_meters = math.sqrt(tile_area_sqm)
                
                # Convert meters to pixels using GSD
                tile_size_pixels = int(round(tile_side_meters / gsd))
                
                # Ensure minimum tile size
                if tile_size_pixels < 64:
                    print(f"⚠️  WARNING: Calculated tile size ({tile_size_pixels}px) is very small!")
                    tile_size_pixels = 64
                
                print(f"\nTile Configuration:")
                print(f"  Requested area: {tile_area_sqm:,.0f} m²")
                print(f"  Target tile side: {tile_side_meters:.2f} meters")
                print(f"  Tile size: {tile_size_pixels}x{tile_size_pixels} pixels")
                print(f"  Actual tile area: {(tile_size_pixels * gsd)**2:.2f} m²")
                print(f"{'='*70}\n")

                tile_count = 0
                for row in range(0, total_height, tile_size_pixels):
                    for col in range(0, total_width, tile_size_pixels):
                        # Calculate actual tile dimensions (handle edge cases)
                        actual_width = min(tile_size_pixels, total_width - col)
                        actual_height = min(tile_size_pixels, total_height - row)

                        # Skip tiles that are too small (less than 25% of target size)
                        min_acceptable_size = tile_size_pixels // 4
                        if actual_width < min_acceptable_size or actual_height < min_acceptable_size:
                            continue

                        # Calculate tile bounds in world coordinates
                        window = Window(col, row, actual_width, actual_height)
                        tile_transform = rasterio.windows.transform(window, original_transform)
                        tile_bounds = rasterio.windows.bounds(window, original_transform)
                        
                        # Calculate actual tile area in square meters
                        tile_width_meters = actual_width * gsd
                        tile_height_meters = actual_height * gsd
                        actual_tile_area = tile_width_meters * tile_height_meters

                        # Create tile filename with coordinate metadata
                        tile_filename = f"tile_{row:04d}_{col:04d}_area{int(actual_tile_area)}sqm.tif"
                        tile_path = os.path.join(output_dir, tile_filename)

                        # Read tile data WITHOUT any modification
                        tile_data = src.read(window=window)

                        # Save as GeoTIFF with coordinate info
                        profile = src.profile.copy()
                        profile.update({
                            'driver': 'GTiff',
                            'height': actual_height,
                            'width': actual_width,
                            'transform': tile_transform,
                            'crs': original_crs,
                            'compress': 'lzw'
                        })

                        with rasterio.open(tile_path, 'w', **profile) as dst:
                            dst.write(tile_data)

                        # Store metadata for processing
                        tile_metadata = {
                            'path': tile_path,
                            'row': row,
                            'col': col,
                            'transform': tile_transform,
                            'crs': original_crs,
                            'bounds': tile_bounds,
                            'shape': (actual_height, actual_width),
                            'area_sqm': actual_tile_area,  # Store actual area
                            'gsd': gsd  # Store GSD for reference
                        }
                        tile_metadata_list.append(tile_metadata)
                        tile_count += 1

                        if tile_count % 10 == 0:
                            print(f"Created {tile_count} area-based tiles...")

                print(f"Successfully created {len(tile_metadata_list)} area-based tiles")
                print(f"Average tile area: {np.mean([t['area_sqm'] for t in tile_metadata_list]):.2f} m²")

        except Exception as e:
            print(f"Error creating area-based tiles: {e}")
            raise

        return tile_metadata_list

    # def normalize_tile_data(self, tile_data):
    #     """Normalize tile data for better visualization"""
    #     # For single-band or RGB (3-band) data
    #     bands, height, width = tile_data.shape
    #     normalized = np.empty_like(tile_data, dtype=np.uint8)
        
    #     for i in range(bands):
    #         band = tile_data[i]
    #         min_val = np.nanmin(band)
    #         max_val = np.nanmax(band)
    #         if max_val > min_val:
    #             norm_band = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    #         else:
    #             norm_band = np.zeros_like(band, dtype=np.uint8)
    #         normalized[i] = norm_band
        
    #     return normalized

    
    # **NEW**: Extract Height Data for Specific Tile
    def extract_height_data_for_tile(self, tile_metadata):
        """Extract DHM/DTM data for a specific tile"""
        try:
            tile_bounds = tile_metadata['bounds']
            tile_transform = tile_metadata['transform']
            tile_shape = tile_metadata['shape']
            tile_crs = tile_metadata['crs']
            
            # Extract and align DHM data
            dhm_data = self._extract_aligned_height_data(
                self.dhm_path, tile_bounds, tile_transform, tile_shape, tile_crs
            )
            
            # Extract and align DTM data
            dtm_data = self._extract_aligned_height_data(
                self.dtm_path, tile_bounds, tile_transform, tile_shape, tile_crs
            )
            
            return dhm_data, dtm_data, tile_transform
            
        except Exception as e:
            print(f"Error extracting height data for tile: {e}")
            return None, None, None

    # **NEW**: Helper method for height data extraction and alignment
    def _extract_aligned_height_data(self, height_file_path, tile_bounds, tile_transform, tile_shape, tile_crs):
        """Extract and align height data to tile coordinate system"""
        try:
            with rasterio.open(height_file_path) as height_src:
                # Get window covering the tile bounds
                try:
                    height_window = from_bounds(*tile_bounds, height_src.transform)
                    height_data = height_src.read(1, window=height_window)
                except Exception:
                    # Fallback: read entire height data and crop
                    height_data = height_src.read(1)
                    height_window = None
                
                # Check if reprojection is needed
                if height_src.crs != tile_crs:
                    print(f"Reprojecting height data from {height_src.crs} to {tile_crs}")
                    
                    # Create output array
                    aligned_data = np.empty(tile_shape, dtype=np.float32)
                    
                    # Reproject height data to tile coordinate system
                    rasterio.warp.reproject(
                        source=height_data,
                        destination=aligned_data,
                        src_transform=height_src.window_transform(height_window) if height_window else height_src.transform,
                        src_crs=height_src.crs,
                        dst_transform=tile_transform,
                        dst_crs=tile_crs,
                        resampling=rasterio.warp.Resampling.bilinear
                    )
                    
                    return aligned_data
                else:
                    # Same CRS, just resize if needed
                    if height_data.shape != tile_shape:
                        from scipy.ndimage import zoom
                        zoom_factors = (tile_shape[0] / height_data.shape[0], 
                                      tile_shape[1] / height_data.shape[1])
                        height_data = zoom(height_data, zoom_factors, order=1)
                    
                    return height_data.astype(np.float32)
                    
        except Exception as e:
            print(f"Error processing height data: {e}")
            # Return zeros as fallback
            return np.zeros(tile_shape, dtype=np.float32)

    # **NEW**: Coordinate-aware building floor calculation
    def calculate_building_floors_from_tile_data(self, x_center, y_center, dhm_data, dtm_data, transform):
        """Calculate building floors using tile-specific height data and coordinates"""
        try:
            # Ensure coordinates are within bounds
            height, width = dhm_data.shape
            x_center = max(0, min(int(x_center), width - 1))
            y_center = max(0, min(int(y_center), height - 1))
            
            # Sample height data around the building center
            sample_radius = min(5, height // 10, width // 10)  # Adaptive sampling radius
            
            # Extract samples in a window around the center
            y_start = max(0, y_center - sample_radius)
            y_end = min(height, y_center + sample_radius + 1)
            x_start = max(0, x_center - sample_radius)
            x_end = min(width, x_center + sample_radius + 1)
            
            # Get height samples
            dhm_samples = dhm_data[y_start:y_end, x_start:x_end]
            dtm_samples = dtm_data[y_start:y_end, x_start:x_end]
            
            # Filter out invalid values (NaN, extreme values)
            valid_mask = (~np.isnan(dhm_samples)) & (~np.isnan(dtm_samples))
            valid_mask &= (dhm_samples > -1000) & (dhm_samples < 10000)
            valid_mask &= (dtm_samples > -1000) & (dtm_samples < 10000)
            
            if not valid_mask.any():
                print(f"No valid height data at position ({x_center}, {y_center})")
                return 1  # Default to 1 floor
            
            # Calculate building height
            dhm_height = np.median(dhm_samples[valid_mask])
            dtm_height = np.median(dtm_samples[valid_mask])
            building_height = dhm_height - dtm_height
            
            # Validate building height
            if building_height < 2.5:  # Minimum building height
                return 1
            
            # Calculate floors
            estimated_floors = max(1, round(building_height / self.floor_height.get()))
            
            # Cap maximum floors (reasonable limit)
            estimated_floors = min(estimated_floors, 50)
            
            print(f"Building at ({x_center}, {y_center}): height={building_height:.1f}m, floors={estimated_floors}")
            
            return estimated_floors
            
        except Exception as e:
            print(f"Error calculating floors: {e}")
            return 1  # Default fallback

    # **NEW**: Coordinate-aware tile analysis
    def analyze_tile_with_coordinates(self, tile_metadata):
        """Analyze tile with proper coordinate-aware floor calculation"""
        tile_path = tile_metadata['path']
        print(f"Analyzing coordinate-aware tile: {os.path.basename(tile_path)}")

        try:
            # Extract height data
            dhm_data, dtm_data, tile_transform = self.extract_height_data_for_tile(tile_metadata)
            if dhm_data is None or dtm_data is None:
                dhm_data = dtm_data = None

            # ✅ Always preprocess tile before YOLO
            preprocessed_path = self.preprocess_tile_for_yolo(tile_path)

            # Run YOLO detection on safe RGB
            with Image.open(preprocessed_path) as im:
                rgb_array = np.array(im.convert("RGB"))

            results = model.predict(source=rgb_array, save=False, imgsz=640, verbose=False)
            # ✅ Generate annotated image from YOLO results
            if results and len(results) > 0:
                result = results[0]
                annotated_image = result.plot()  # Draw bounding boxes (BGR array)

                # Save temporarily so GUI can show it
                import cv2, tempfile
                annotated_path = tempfile.mktemp(suffix=".png")
                cv2.imwrite(annotated_path, annotated_image)

                # Store path for later display in GUI
                self.annotated_result_path = annotated_path

            if not results or len(results) == 0:
                return []

            detection_results = []
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    try:
                        x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        if (self.use_height_calculation.get() and 
                            dhm_data is not None and dtm_data is not None):
                            floors = self.calculate_building_floors_from_tile_data(
                                int(x_center), int(y_center), dhm_data, dtm_data, tile_transform
                            )
                        else:
                            floors = 1

                        world_x, world_y = rasterio.transform.xy(tile_transform, y_center, x_center)

                        detection_result = {
                            'tile_path': tile_path,
                            'tile_row': tile_metadata['row'],
                            'tile_col': tile_metadata['col'],
                            'detection_id': i,
                            'pixel_x': float(x_center),
                            'pixel_y': float(y_center),
                            'world_x': float(world_x),
                            'world_y': float(world_y),
                            'width': float(width),
                            'height': float(height),
                            'confidence': confidence,
                            'class_id': class_id,
                            'estimated_floors': floors,
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        }
                        detection_results.append(detection_result)

                    except Exception as e:
                        print(f"Error processing detection {i} in tile {tile_path}: {e}")
                        continue

            print(f"Found {len(detection_results)} buildings in coordinate-aware tile")
            return detection_results

        except Exception as e:
            print(f"Error analyzing coordinate-aware tile {tile_path}: {e}")
            return []


    # Continuing with the rest of the original methods but with coordinate-aware modifications...
    def create_parameters_section(self, parent):
        """Create parameters section with improved layout"""
        section_frame = tk.LabelFrame(parent, text="⚙️ Building Count Parameters",
                                    font=('Arial', 12, 'bold'),
                                    bg="white", fg="#2c3e50", padx=15, pady=15)
        section_frame.pack(fill="x", padx=10, pady=5)
        
        # Create parameter grid
        params = [
            ("People per Household:", self.people_per_household, "Average number of people per household"),
            ("Standard House Area (m²):", self.standard_house_area, "Baseline house area for calculations"),
            ("Medium House Threshold (m²):", self.medium_house_threshold, "2x population multiplier"),
            ("Large House Threshold (m²):", self.large_house_threshold, "3x population multiplier")
        ]
        
        for i, (label, var, tooltip) in enumerate(params):
            # Parameter frame
            param_frame = tk.Frame(section_frame, bg="white")
            param_frame.pack(fill="x", pady=(0, 12))
            
            # Label
            label_widget = tk.Label(param_frame, text=label, bg="white",
                                  font=('Arial', 10, 'bold'), fg="#2c3e50")
            label_widget.pack(anchor="w")
            
            # Entry with validation
            entry = tk.Entry(param_frame, textvariable=var, font=('Arial', 10),
                           relief=tk.SOLID, bd=1, bg="#ecf0f1")
            entry.pack(fill="x", pady=(2, 2))
            
            # Store entry reference
            if i == 0:
                self.household_entry = entry
            elif i == 1:
                self.standard_area_entry = entry
            elif i == 2:
                self.medium_threshold_entry = entry
            else:
                self.large_threshold_entry = entry
            
            # Tooltip
            tooltip_label = tk.Label(param_frame, text=f"💡 {tooltip}",
                                   bg="white", font=('Arial', 8), fg="#7f8c8d")
            tooltip_label.pack(anchor="w", pady=(2, 0))
        
        # Methodology info
        self.create_info_box(section_frame,
                           "Coordinate-Aware Methodology:", [
            "• Base Population = Building Count × Multiplier × People/Household",
            "• With Height Data: Final Population = Base Population × Floor Count",
            "• Tiles processed with coordinate preservation (GeoTIFF format)",
            "• Floor calculation uses tile-specific DHM/DTM alignment",
            "• Standard houses (1x): < Medium threshold",
            "• Medium houses (2x): Medium to Large threshold", 
            "• Large houses (3x): > Large threshold",
            "• Non-residential buildings = 0 population"
        ])

    def create_info_box(self, parent, title, items):
        """Create an information box with bullet points"""
        info_frame = tk.Frame(parent, bg="#ecf0f1", relief=tk.SUNKEN, bd=1)
        info_frame.pack(fill="x", pady=(10, 0))
        
        title_label = tk.Label(info_frame, text=title, bg="#ecf0f1",
                             font=('Arial', 9, 'bold'), fg="#2c3e50")
        title_label.pack(anchor="w", padx=8, pady=(5, 2))
        
        for item in items:
            item_label = tk.Label(info_frame, text=item, bg="#ecf0f1",
                                font=('Arial', 8), fg="#34495e", justify="left")
            item_label.pack(anchor="w", padx=8)
        
        # Padding
        tk.Label(info_frame, text="", bg="#ecf0f1", height=1).pack()

    def create_tif_section(self, parent):
        """Create TIF processing section with enhanced UI"""
        self.tif_frame = tk.LabelFrame(parent, text="🗂️ Coordinate-Aware TIF Processing & Tile Configuration",
                                     font=('Arial', 12, 'bold'),
                                     bg="white", fg="#2c3e50", padx=15, pady=15)
        self.tif_frame.pack(fill="x", padx=10, pady=5)
        self.tif_frame.pack_forget()  # Hidden by default
        
        # Tile Area Configuration
        self.create_tile_config_section()
        
        # Separator
        separator = tk.Frame(self.tif_frame, height=2, bg="#bdc3c7")
        separator.pack(fill="x", pady=15)
        
        # Tile Generation Section
        self.create_tile_generation_section()
        
    def create_tile_upload_section(self, parent):
        """Create section for uploading individual tiles for analysis"""
        self.tile_upload_frame = tk.LabelFrame(parent, text="📂 Individual Tile Upload & Analysis",
                                            font=('Arial', 12, 'bold'),
                                            bg="white", fg="#2c3e50", padx=15, pady=15)
        self.tile_upload_frame.pack(fill="x", padx=10, pady=5)
        self.tile_upload_frame.pack_forget()  # Hidden by default
        
        # Info
        info_label = tk.Label(self.tile_upload_frame, 
                            text="📌 Upload individual tiles generated from TIF file for analysis",
                            bg="white", font=('Arial', 10), fg="#7f8c8d")
        info_label.pack(anchor="w", pady=(0, 10))
        
        # Upload button
        self.upload_tiles_btn = tk.Button(self.tile_upload_frame, text="📁 Upload Individual Tiles",
                                        command=self.upload_individual_tiles, bg="#3498db", fg="white",
                                        font=('Arial', 11, 'bold'), height=2, cursor="hand2",
                                        relief=tk.RAISED, activebackground="#2980b9")
        self.upload_tiles_btn.pack(fill="x", pady=(0, 10))
        
        # Tile list frame
        list_frame = tk.Frame(self.tile_upload_frame, bg="white")
        list_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        tk.Label(list_frame, text="Uploaded Tiles:", bg="white",
                font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w")
        
        # Listbox for tiles
        listbox_frame = tk.Frame(list_frame, bg="white")
        listbox_frame.pack(fill="both", expand=True, pady=(5, 0))
        
        self.tile_listbox = tk.Listbox(listbox_frame, height=6, font=('Arial', 9),
                                    selectmode=tk.SINGLE, relief=tk.SOLID, bd=1)
        tile_scrollbar = tk.Scrollbar(listbox_frame, orient="vertical", command=self.tile_listbox.yview)
        self.tile_listbox.configure(yscrollcommand=tile_scrollbar.set)
        
        self.tile_listbox.pack(side="left", fill="both", expand=True)
        tile_scrollbar.pack(side="right", fill="y")
        
        # Tile selection info
        self.tile_selection_info = tk.Label(self.tile_upload_frame, text="No tiles uploaded",
                                        bg="white", font=('Arial', 10), fg="#7f8c8d")
        self.tile_selection_info.pack(anchor="w", pady=(5, 0))
        
        # Bind selection event
        self.tile_listbox.bind('<<ListboxSelect>>', self.on_tile_select)
        
        # Store uploaded tiles
        self.uploaded_tiles = []

    def upload_individual_tiles(self):
        """Upload individual tile files for analysis"""
        file_paths = filedialog.askopenfilenames(
            title="Select Tile Files",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All Supported", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            # Clear previous tiles
            self.uploaded_tiles = []
            self.tile_listbox.delete(0, tk.END)
            
            # Add new tiles
            for file_path in file_paths:
                self.uploaded_tiles.append(file_path)
                filename = os.path.basename(file_path)
                self.tile_listbox.insert(tk.END, filename)
            
            self.tile_selection_info.config(
                text=f"✅ {len(self.uploaded_tiles)} tile(s) uploaded - Select a tile and run analysis",
                fg="#27ae60"
            )
            self.update_status(f"Uploaded {len(self.uploaded_tiles)} individual tiles", "📂")
            self.update_validation_status()

    def on_tile_select(self, event):
        """Handle tile selection from listbox"""
        selection = self.tile_listbox.curselection()
        if selection:
            selected_index = selection[0]
            selected_file = self.uploaded_tiles[selected_index]
            filename = os.path.basename(selected_file)
            
            self.tile_selection_info.config(
                text=f"📌 Selected: {filename} - Ready for analysis",
                fg="#3498db"
            )
            
            # Load preview of selected tile
            self.load_tile_preview(selected_file)

    def load_tile_preview(self, tile_path):
        """Load preview of selected tile with proper channel handling"""
        try:
            if tile_path.lower().endswith(('.tif', '.tiff')):
                # Handle TIFF tile with proper channel management
                with rasterio.open(tile_path) as src:
                    # Ensure RGB output
                    if src.count >= 3:
                        data = src.read([1, 2, 3])
                        data = np.transpose(data, (1, 2, 0))
                    else:
                        single_band = src.read(1)
                        data = np.stack([single_band, single_band, single_band], axis=2)
                    
                    # Normalize for display
                    data = data.astype(np.float32)
                    data_min = np.nanmin(data)
                    data_max = np.nanmax(data)
                    
                    if data_max > data_min:
                        data = ((data - data_min) / (data_max - data_min)) * 255
                    else:
                        data = np.full_like(data, 128)
                    
                    data = np.clip(data, 0, 255).astype(np.uint8)
                    
                    # Verify 3 channels
                    if data.shape[2] != 3:
                        raise Exception(f"Failed to create 3-channel image, got {data.shape}")
                    
                    image_pil = Image.fromarray(data, mode='RGB')
            else:
                # Handle regular image files
                image = cv2.imread(tile_path)
                if image is not None:
                    # Ensure RGB
                    if len(image.shape) == 3:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    image_pil = Image.fromarray(image_rgb)
                else:
                    raise Exception("Could not load image")
            
            # Resize and display
            image_pil.thumbnail((300, 150), Image.Resampling.LANCZOS)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.original_label.config(image=image_tk, text="")
            self.original_label.image = image_tk
            
        except Exception as e:
            error_msg = f"❌ Error loading tile preview:\n{str(e)}"
            self.original_label.config(text=error_msg)
            print(f"Error loading tile preview: {e}")


        
    def preprocess_tile_for_yolo(self, tile_path):
        """Preprocess tile to ensure it's in RGB format for YOLO"""
        try:
            if tile_path.lower().endswith(('.tif', '.tiff')):
                # Handle TIFF tile
                with rasterio.open(tile_path) as src:
                    # Always ensure we get 3 channels
                    if src.count >= 3:
                        # Read first 3 bands as RGB
                        data = src.read([1, 2, 3])
                        data = np.transpose(data, (1, 2, 0))
                    elif src.count == 1:
                        # Single band - replicate to create RGB
                        single_band = src.read(1)
                        data = np.stack([single_band, single_band, single_band], axis=2)
                    else:
                        # Handle 2 bands or other cases
                        available_bands = src.read()
                        if available_bands.shape[0] == 2:
                            # Use first band for R and G, second for B
                            data = np.stack([available_bands, available_bands[1], available_bands], axis=2)
                        else:
                            # Fallback: use first band for all channels
                            first_band = available_bands
                            data = np.stack([first_band, first_band, first_band], axis=2)
                        data = np.transpose(data, (1, 2, 0))
                    
                    # Ensure data is in the right format and range
                    data = data.astype(np.float32)
                    
                    # Normalize to 0-255 range if needed
                    if data.max() > 255 or data.min() < 0:
                        # Handle different data ranges
                        data_min = np.nanmin(data)
                        data_max = np.nanmax(data)
                        if data_max > data_min:
                            data = ((data - data_min) / (data_max - data_min)) * 255
                        else:
                            data = np.full_like(data, 128)
                    
                    # Ensure values are in valid range and convert to uint8
                    data = np.clip(data, 0, 255).astype(np.uint8)
                    
                    # Verify we have exactly 3 channels
                    if len(data.shape) != 3 or data.shape[2] != 3:
                        raise ValueError(f"Expected RGB image with 3 channels, got shape: {data.shape}")
                    
                    # Save as temporary RGB image for YOLO
                    import tempfile
                    temp_path = tempfile.mktemp(suffix='.png')
                    image_pil = Image.fromarray(data, mode='RGB')
                    image_pil.save(temp_path)
                    return temp_path
                    
            else:
                # Regular image file - ensure it's RGB
                image = cv2.imread(tile_path)
                if image is None:
                    raise Exception("Could not load image file")
                
                # Convert BGR to RGB and ensure 3 channels
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2:
                    # Grayscale image - convert to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    raise Exception(f"Unexpected image format: {image.shape}")
                
                # Save as temporary RGB image
                import tempfile
                temp_path = tempfile.mktemp(suffix='.png')
                image_pil = Image.fromarray(image_rgb, mode='RGB')
                image_pil.save(temp_path)
                return temp_path
                
        except Exception as e:
            print(f"Error preprocessing tile: {e}")
            # Fallback: create a simple RGB image if all else fails
            try:
                import tempfile
                temp_path = tempfile.mktemp(suffix='.png')
                fallback_image = np.full((640, 640, 3), 128, dtype=np.uint8)
                image_pil = Image.fromarray(fallback_image, mode='RGB')
                image_pil.save(temp_path)
                return temp_path
            except:
                return tile_path

    def validate_image_channels(self, image_path):
        """Validate that image has exactly 3 channels before YOLO processing"""
        try:
            # Test load with PIL for more reliable format detection
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Check dimensions
                width, height = img.size
                channels = len(img.getbands())
                
                if channels != 3:
                    return False, f"Image has {channels} channels, expected 3 (RGB)"
                
                if width < 32 or height < 32:
                    return False, f"Image too small: {width}x{height}, minimum 32x32"
                
                return True, f"Valid RGB image: {width}x{height}x{channels}"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def debug_image_info(self, image_path):
        """Debug method to check image properties"""
        try:
            print(f"    🔍 DEBUG INFO for {os.path.basename(image_path)}")
            
            if image_path.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(image_path) as src:
                    print(f"    📊 TIFF - Bands: {src.count}, Shape: {src.shape}, CRS: {src.crs}")
                    print(f"    📊 Data type: {src.dtypes}, Bounds: {src.bounds}")
                    
                    # Sample data from first few bands
                    for i in range(min(3, src.count)):
                        try:
                            band_data = src.read(i + 1)
                            print(f"    📊 Band {i+1} - Min: {np.nanmin(band_data):.2f}, Max: {np.nanmax(band_data):.2f}, Shape: {band_data.shape}")
                        except Exception as e:
                            print(f"    ❌ Error reading band {i+1}: {e}")
            else:
                # Regular image
                image = cv2.imread(image_path)
                if image is not None:
                    print(f"    📊 CV2 - Shape: {image.shape}, Type: {image.dtype}")
                    print(f"    📊 Min: {image.min()}, Max: {image.max()}")
                else:
                    print(f"    ❌ CV2 - Could not load image")
                    
                # Also try with PIL
                try:
                    with Image.open(image_path) as img:
                        print(f"    📊 PIL - Size: {img.size}, Mode: {img.mode}, Format: {img.format}")
                except Exception as e:
                    print(f"    ❌ PIL error: {e}")
                    
        except Exception as e:
            print(f"    ❌ Debug error: {e}")



    def create_tile_config_section(self):
        """Create tile configuration section"""
        config_frame = tk.LabelFrame(self.tif_frame, text="📐 Coordinate-Aware Tile Size Configuration",
                                   font=('Arial', 11, 'bold'),
                                   bg="white", fg="#2c3e50", padx=12, pady=12)
        config_frame.pack(fill="x", pady=(0, 10))
        
        # Tile area input
        input_frame = tk.Frame(config_frame, bg="white")
        input_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(input_frame, text="Tile Area (m²):", bg="white",
               font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w")
        
        # Area input with live feedback
        area_input_frame = tk.Frame(input_frame, bg="white")
        area_input_frame.pack(fill="x", pady=(5, 0))
        
        self.tile_area_entry = tk.Entry(area_input_frame, textvariable=self.tile_area_sqm,
                                      font=('Arial', 11), width=12, relief=tk.SOLID, bd=1)
        self.tile_area_entry.pack(side="left")
        
        self.tile_dimensions_label = tk.Label(area_input_frame, text="", bg="white",
                                            font=('Arial', 10), fg="#27ae60")
        self.tile_dimensions_label.pack(side="left", padx=(10, 0))
        
        # Bind events for live update
        self.tile_area_entry.bind('<KeyRelease>', self.update_tile_dimensions)
        self.tile_area_entry.bind('<FocusOut>', self.update_tile_dimensions)
        
        # Preset buttons
        preset_frame = tk.Frame(config_frame, bg="white")
        preset_frame.pack(fill="x", pady=(10, 0))
        
        tk.Label(preset_frame, text="Quick Presets:", bg="white",
               font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w", pady=(0, 5))
        
        button_frame = tk.Frame(preset_frame, bg="white")
        button_frame.pack(fill="x")
        
        presets = [
            ("50×50m", 2500, "#e67e22"),
            ("100×100m", 10000, "#27ae60"),
            ("200×200m", 40000, "#3498db"),
            ("500×500m", 250000, "#9b59b6")
        ]
        
        for preset_name, preset_area, color in presets:
            btn = tk.Button(button_frame, text=preset_name,
                          command=lambda area=preset_area: self.set_tile_area(area),
                          bg=color, fg="white", font=('Arial', 9, 'bold'),
                          width=10, height=1, cursor="hand2", relief=tk.RAISED,
                          activebackground=color)
            btn.pack(side="left", padx=(0, 8), pady=2)
        
        # Info box
        self.create_info_box(config_frame, "Coordinate-Aware Tile Guidelines:", [
            "• Tiles saved as GeoTIFF with coordinate information preserved",
            "• Smaller tiles (50×50m): Faster processing, more files, better coordinate precision",
            "• Medium tiles (100×100m): Balanced processing (Recommended)",
            "• Larger tiles (200×200m+): Slower processing, fewer files",
            "• Very large tiles may cause memory issues",
            "• Coordinate metadata enables accurate height-based floor calculation"
        ])
        
        # Initialize display
        self.update_tile_dimensions()

    def create_tile_generation_section(self):
        """Create tile generation controls"""
        gen_frame = tk.LabelFrame(self.tif_frame, text="🔧 Coordinate-Aware Tile Generation",
                                font=('Arial', 11, 'bold'),
                                bg="white", fg="#2c3e50", padx=12, pady=12)
        gen_frame.pack(fill="x")
        
        # Save location
        location_frame = tk.Frame(gen_frame, bg="white")
        location_frame.pack(fill="x", pady=(0, 15))
        
        tk.Label(location_frame, text="Save Location:", bg="white",
               font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w")
        
        location_input_frame = tk.Frame(location_frame, bg="white")
        location_input_frame.pack(fill="x", pady=(5, 0))
        
        self.browse_btn = tk.Button(location_input_frame, text="📁 Browse Folder",
                                  command=self.browse_save_location, bg="#e67e22", fg="white",
                                  font=('Arial', 10, 'bold'), cursor="hand2",
                                  relief=tk.RAISED, activebackground="#d35400")
        self.browse_btn.pack(side="left")
        
        self.save_location_info = tk.Label(location_input_frame, text="No location selected",
                                         bg="white", font=('Arial', 10), fg="#7f8c8d")
        self.save_location_info.pack(side="left", padx=(15, 0))
        
        # Generate button
        self.generate_tiles_btn = tk.Button(gen_frame, text="🔄 Generate Coordinate-Aware Tiles",
                                          command=self.generate_tiles, bg="#27ae60", fg="white",
                                          font=('Arial', 11, 'bold'), height=2, cursor="hand2",
                                          relief=tk.RAISED, activebackground="#229954")
        self.generate_tiles_btn.pack(fill="x", pady=(0, 10))
        
        # Status
        self.tiles_status = tk.Label(gen_frame, text="", bg="white",
                                   font=('Arial', 10), fg="#27ae60")
        self.tiles_status.pack(anchor="w")

    def create_analysis_section(self, parent):
        """Create analysis section with prominent button"""
        section_frame = tk.LabelFrame(parent, text="🔍 Coordinate-Aware Analysis", font=('Arial', 12, 'bold'),
                                    bg="white", fg="#2c3e50", padx=15, pady=15)
        section_frame.pack(fill="x", padx=10, pady=5)
        
        # Analysis button
        self.analyze_btn = tk.Button(section_frame, text="START ANALYSIS",
                                   command=self.start_analysis, bg="#e74c3c", fg="white",
                                   font=('Arial', 14, 'bold'), height=3, cursor="hand2",
                                   relief=tk.RAISED, activebackground="#c0392b")
        self.analyze_btn.pack(fill="x", pady=(0, 10))
        
        # Quick validation info
        self.validation_label = tk.Label(section_frame, text="✅ Ready for coordinate-aware analysis",
                                       bg="white", font=('Arial', 10), fg="#27ae60")
        self.validation_label.pack(anchor="w")

    def create_visualization_controls(self, parent):
        """Create visualization controls section"""
        viz_frame = tk.LabelFrame(parent, text="📊 Advanced Visualizations",
                                font=('Arial', 12, 'bold'),
                                bg="white", fg="#2c3e50", padx=15, pady=15)
        viz_frame.pack(fill="x", padx=10, pady=5)
        
        # Chart selection
        tk.Label(viz_frame, text="Select Chart Types:", bg="white",
               font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w", pady=(0, 8))
        
        self.chart_vars = {}
        chart_types = [
            ("📊 Building Count Analysis", "building_count"),
            ("👥 Population Distribution", "population_dist"),
            ("📏 Size Category Analysis", "size_analysis"),
            ("🏢 Coordinate-Aware Floor Analysis (if height data available)", "floor_analysis"),
            ("📈 Statistical Comparisons", "statistical_comp")
        ]
        
        for chart_name, chart_key in chart_types:
            var = tk.BooleanVar(value=True)
            self.chart_vars[chart_key] = var
            cb = tk.Checkbutton(viz_frame, text=chart_name, variable=var,
                              bg="white", font=('Arial', 10), fg="#2c3e50",
                              selectcolor="#3498db", cursor="hand2")
            cb.pack(anchor="w", padx=10, pady=2)
        
        # Generate button
        self.generate_viz_btn = tk.Button(viz_frame, text="🎨 Generate Advanced Charts",
                                        command=self.generate_all_visualizations,
                                        bg="#9b59b6", fg="white",
                                        font=('Arial', 11, 'bold'), height=2,
                                        state='disabled', cursor="hand2",
                                        relief=tk.RAISED, activebackground="#8e44ad")
        self.generate_viz_btn.pack(fill="x", pady=(10, 0))

    def create_csv_export_section(self, parent):
        """Create CSV export section"""
        self.csv_frame = tk.LabelFrame(parent, text="📊 CSV Data Export",
                                     font=('Arial', 12, 'bold'),
                                     bg="white", fg="#2c3e50", padx=15, pady=15)
        self.csv_frame.pack(fill="x", padx=10, pady=5)
        
        # Export options
        export_options_frame = tk.Frame(self.csv_frame, bg="white")
        export_options_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(export_options_frame, text="📋 Export Options:", bg="white",
               font=('Arial', 10, 'bold'), fg="#2c3e50").pack(anchor="w", pady=(0, 5))
        
        # Export type selection
        self.export_type = tk.StringVar(value="detailed")
        
        detailed_rb = tk.Radiobutton(export_options_frame, text="🏠 Detailed Building Data with Coordinates (Recommended)",
                                   variable=self.export_type, value="detailed",
                                   bg="white", font=('Arial', 10), fg="#2c3e50",
                                   selectcolor="#3498db", cursor="hand2")
        detailed_rb.pack(anchor="w", padx=10, pady=2)
        
        summary_rb = tk.Radiobutton(export_options_frame, text="📊 Summary Statistics Only",
                                  variable=self.export_type, value="summary",
                                  bg="white", font=('Arial', 10), fg="#2c3e50",
                                  selectcolor="#3498db", cursor="hand2")
        summary_rb.pack(anchor="w", padx=10, pady=2)
        
        complete_rb = tk.Radiobutton(export_options_frame, text="📁 Complete Analysis (Both)",
                                   variable=self.export_type, value="complete",
                                   bg="white", font=('Arial', 10), fg="#2c3e50",
                                   selectcolor="#3498db", cursor="hand2")
        complete_rb.pack(anchor="w", padx=10, pady=2)
        
        # Export button
        self.export_csv_btn = tk.Button(self.csv_frame, text="💾 Export Coordinate-Aware Analysis to CSV",
                                      command=self.export_to_csv, bg="#27ae60", fg="white",
                                      font=('Arial', 11, 'bold'), height=2, cursor="hand2",
                                      relief=tk.RAISED, activebackground="#229954",
                                      state='disabled')
        self.export_csv_btn.pack(fill="x", pady=(10, 0))
        
        # Info about CSV export
        self.create_info_box(self.csv_frame, "Coordinate-Aware CSV Export Contains:", [
            "• Individual building data with pixel and world coordinates",
            "• Population calculations for each building",
            "• Floor information with coordinate-based height calculation",
            "• Tile metadata and coordinate system information",
            "• Analysis parameters and methodology used",
            "• Summary statistics and totals",
            "• Timestamp and file information"
        ])

    # Continuing with the rest of the methods...
    def create_results_sections(self, parent):
        """Create results display sections"""
        # Quick summary cards
        self.create_summary_cards(parent)
        
        # Image preview section
        self.create_image_preview_section(parent)
        
        # Tabbed results
        self.create_tabbed_results(parent)

    def create_summary_cards(self, parent):
        """Create summary cards with modern design"""
        cards_frame = tk.LabelFrame(parent, text="📊 Quick Results", font=('Arial', 12, 'bold'),
                                  bg="white", fg="#2c3e50", padx=15, pady=15)
        cards_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Cards container with grid
        cards_container = tk.Frame(cards_frame, bg="white")
        cards_container.pack(fill="x", pady=(5, 0))
        
        # Configure grid
        for i in range(5):
            cards_container.grid_columnconfigure(i, weight=1, uniform="card")
        
        # Card data
        cards_data = [
            ("Total Population", "0", "#e74c3c", "👥"),
            ("Residential Buildings", "0", "#3498db", "🏠"),
            ("Standard Houses", "0", "#27ae60", "🏡"),
            ("Large Houses", "0", "#f39c12", "🏰"),
            ("Avg Floors", "0", "#9b59b6", "🏢")
        ]
        
        self.summary_cards = []
        for i, (title, value, color, icon) in enumerate(cards_data):
            card = self.create_summary_card(cards_container, title, value, color, icon, 0, i)
            self.summary_cards.append(card)

    def create_summary_card(self, parent, title, value, color, icon, row, col):
        """Create individual summary card with modern styling"""
        card_frame = tk.Frame(parent, bg=color, relief=tk.RAISED, bd=3)
        card_frame.grid(row=row, column=col, sticky="ew", padx=3, pady=5, ipady=8)
        
        # Icon
        icon_label = tk.Label(card_frame, text=icon, bg=color, fg="white",
                            font=('Arial', 18))
        icon_label.pack(pady=(3, 0))
        
        # Value
        value_label = tk.Label(card_frame, text=value, bg=color, fg="white",
                             font=('Arial', 14, 'bold'))
        value_label.pack()
        
        # Title
        title_label = tk.Label(card_frame, text=title, bg=color, fg="white",
                             font=('Arial', 8, 'bold'))
        title_label.pack(pady=(0, 3))
        
        return value_label

    def create_image_preview_section(self, parent):
        """Create image preview section with better layout"""
        images_frame = tk.LabelFrame(parent, text="🖼️ Image Analysis", font=('Arial', 12, 'bold'),
                                   bg="white", fg="#2c3e50", padx=15, pady=15)
        images_frame.pack(fill="x", padx=10, pady=5)
        
        # Images container
        images_container = tk.Frame(images_frame, bg="white")
        images_container.pack(fill="x", pady=(5, 0))
        images_container.grid_columnconfigure(0, weight=1)
        images_container.grid_columnconfigure(1, weight=1)
        
        # Original image
        original_frame = tk.Frame(images_container, bg="#ecf0f1", relief=tk.SUNKEN, bd=2)
        original_frame.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=5, ipady=10)
        
        tk.Label(original_frame, text="📷 Original Image", bg="#ecf0f1",
               font=('Arial', 11, 'bold'), fg="#2c3e50").pack(pady=(8, 5))
        
        self.original_label = tk.Label(original_frame, text="No image loaded",
                                     bg="#ecf0f1", fg="#7f8c8d", font=('Arial', 10))
        self.original_label.pack(expand=True, pady=10)
        
        # Processed image
        processed_frame = tk.Frame(images_container, bg="#ecf0f1", relief=tk.SUNKEN, bd=2)
        processed_frame.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=5, ipady=10)
        
        tk.Label(processed_frame, text="🔍 Analysis Results", bg="#ecf0f1",
               font=('Arial', 11, 'bold'), fg="#2c3e50").pack(pady=(8, 5))
        
        self.processed_label = tk.Label(processed_frame, text="Analysis not started",
                                      bg="#ecf0f1", fg="#7f8c8d", font=('Arial', 10))
        self.processed_label.pack(expand=True, pady=10)

    def create_tabbed_results(self, parent):
        """Create tabbed results section"""
        results_frame = tk.LabelFrame(parent, text="📈 Detailed Results & Visualizations",
                                    font=('Arial', 12, 'bold'),
                                    bg="white", fg="#2c3e50", padx=15, pady=15)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create notebook with custom styling
        style = ttk.Style()
        style.configure('Custom.TNotebook.Tab', padding=[20, 10])
        
        self.notebook = ttk.Notebook(results_frame, style='Custom.TNotebook')
        self.notebook.pack(fill="both", expand=True, pady=(10, 0))
        
        # Create tabs
        self.create_summary_tab()
        self.create_basic_charts_tab()
        self.create_advanced_viz_tab()
        self.create_statistical_tab()

    def create_summary_tab(self):
        """Create summary data tab"""
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="📋 Summary")
        
        # Create treeview with better styling
        columns = ("Metric", "Value", "Details")
        self.summary_tree = ttk.Treeview(summary_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        column_widths = {"Metric": 200, "Value": 150, "Details": 350}
        for col in columns:
            self.summary_tree.heading(col, text=col)
            self.summary_tree.column(col, width=column_widths.get(col, 150), anchor="w")
        
        # Add scrollbars
        tree_frame = tk.Frame(summary_frame)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.summary_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.summary_tree.xview)
        self.summary_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.summary_tree.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")

    def create_basic_charts_tab(self):
        """Create basic charts tab with scrolling"""
        self.basic_charts_frame = ScrollableFrame(self.notebook, bg="white")
        self.notebook.add(self.basic_charts_frame.main_frame, text="📊 Basic Charts")

    def create_advanced_viz_tab(self):
        """Create advanced visualizations tab with scrolling"""
        self.advanced_viz_frame = ScrollableFrame(self.notebook, bg="white")
        self.notebook.add(self.advanced_viz_frame.main_frame, text="🎯 Advanced Charts")

    def create_statistical_tab(self):
        """Create statistical analysis tab with scrolling"""
        self.statistical_frame = ScrollableFrame(self.notebook, bg="white")
        self.notebook.add(self.statistical_frame.main_frame, text="📈 Statistics")

    def create_status_bar(self, parent):
        """Create status bar with modern design"""
        status_frame = tk.Frame(parent, bg="#34495e", height=30)
        status_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        status_frame.grid_propagate(False)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please select an image file")
        
        # Status icon
        self.status_icon = tk.Label(status_frame, text="ℹ️", bg="#34495e", fg="white",
                                  font=('Arial', 10))
        self.status_icon.pack(side="left", padx=10, pady=5)
        
        # Status text
        self.status_label = tk.Label(status_frame, textvariable=self.status_var,
                                   bg="#34495e", fg="white", font=('Arial', 10))
        self.status_label.pack(side="left", pady=5)
        
        # Time display
        self.time_label = tk.Label(status_frame, text="", bg="#34495e", fg="#bdc3c7",
                                 font=('Arial', 9))
        self.time_label.pack(side="right", padx=10, pady=5)

    def create_progress_bar(self, parent):
        """Create progress bar for long operations"""
        self.progress_frame = tk.Frame(parent, bg="#2c3e50")
        self.progress_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        self.progress_frame.grid_remove()  # Hidden by default
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate',
                                          style='Custom.Horizontal.TProgressbar')
        self.progress_bar.pack(fill="x", padx=5, pady=2)

    def show_progress(self, show=True):
        """Show or hide progress bar"""
        if show:
            self.progress_frame.grid()
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()
            self.progress_frame.grid_remove()

    # ==================== Utility Methods ====================

    def update_status(self, message, icon="ℹ️"):
        """Update status bar with icon"""
        self.status_var.set(message)
        self.status_icon.config(text=icon)
        self.root.update_idletasks()

    def update_validation_status(self):
        """Update validation status based on current settings"""
        # Check if validation_label exists yet
        if not hasattr(self, 'validation_label') or self.validation_label is None:
            return True  # Return True during initialization

        # Check for individual tile uploads
        if hasattr(self, 'uploaded_tiles') and self.uploaded_tiles:
            self.validation_label.config(text="✅ Individual tiles uploaded - Select a tile and analyze", fg="#27ae60")
            return True

        if not self.file_path:
            self.validation_label.config(text="❌ Please select an image file", fg="#e74c3c")
            return False
        elif self.is_tif_file and not self.tiles_generated:
            self.validation_label.config(text="⚠️ TIF file - Generate tiles first OR upload individual tiles", fg="#f39c12")
            return False
        elif self.use_height_calculation.get() and not (self.dhm_path and self.dtm_path):
            self.validation_label.config(text="⚠️ Height calculation enabled - Select DHM/DTM files", fg="#f39c12")
            return False
        elif self.use_height_calculation.get() and not self.height_data_aligned:
            self.validation_label.config(text="⚠️ Height data not properly aligned", fg="#f39c12")
            return False
        else:
            if self.use_height_calculation.get():
                self.validation_label.config(text="✅ Ready for coordinate-aware analysis with floor calculation", fg="#27ae60")
            else:
                self.validation_label.config(text="✅ Ready for coordinate-aware analysis", fg="#27ae60")
            return True


    def set_tile_area(self, area):
        """Set tile area from preset buttons"""
        self.tile_area_sqm.set(area)
        self.update_tile_dimensions()
        side_length = int(math.sqrt(area))
        self.update_status(f"Tile area set to {area:,} m² ({side_length}×{side_length}m)", "✅")

    def update_tile_dimensions(self, event=None):
        """Update tile dimensions display when area changes"""
        try:
            area = float(self.tile_area_entry.get())
            if area > 0:
                side_length = math.sqrt(area)
                
                # Note: Actual pixel dimensions will vary based on image GSD
                self.tile_dimensions_label.config(
                    text=f"➡️ ~{side_length:.0f}×{side_length:.0f}m (pixels calculated from GSD)",
                    fg="#27ae60"
                )
            else:
                self.tile_dimensions_label.config(text="❌ Invalid area", fg="#e74c3c")
        except (ValueError, tk.TclError):
            self.tile_dimensions_label.config(text="⚠️ Enter valid number", fg="#f39c12")

    # ==================== File Handling Methods ====================

    def choose_image(self):
        """Enhanced file selection with better feedback"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("All Supported", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("TIFF files", "*.tiff *.tif"),
                ("Bitmap files", "*.bmp")
            ]
        )

        if file_path:
            self.file_path = file_path
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.file_info.config(text=f"📄 {filename} ({file_size:.1f} MB)")

            # Check if TIF file
            self.is_tif_file = file_path.lower().endswith(('.tif', '.tiff'))

            if self.is_tif_file:
                self.file_type_info.config(text="🗂️ TIF File - Generate tiles OR upload individual tiles", fg="#e67e22")
                self.tif_frame.pack(fill="x", padx=10, pady=5)
                self.tile_upload_frame.pack(fill="x", padx=10, pady=5)  # Show tile upload section
                self.height_frame.pack(fill="x", padx=10, pady=5)  # Show height data section
                self.update_status("TIF file selected - Generate tiles OR upload individual tiles for analysis", "🗂️")
                self.tiles_generated = False
                self.tiles_status.config(text="")

                # Reset height data alignment
                self.height_data_aligned = False
                self.check_height_data_alignment()
            else:
                self.file_type_info.config(text="🖼️ Standard image - Ready for direct analysis", fg="#27ae60")
                self.tif_frame.pack_forget()
                self.tile_upload_frame.pack_forget()  # Hide tile upload section
                self.height_frame.pack_forget()  # Hide height data section for non-TIF files
                self.update_status("Image file selected - Ready for analysis", "✅")

            # Load preview
            self.load_image_preview(file_path)
            self.update_validation_status()


    def load_image_preview(self, file_path):
        """Load image preview with better error handling"""
        try:
            if self.is_tif_file:
                self.load_tiff_preview(file_path)
            else:
                image = cv2.imread(file_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)
                    image_pil.thumbnail((300, 150), Image.Resampling.LANCZOS)
                    image_tk = ImageTk.PhotoImage(image_pil)
                    
                    self.original_label.config(image=image_tk, text="")
                    self.original_label.image = image_tk
                else:
                    raise Exception("Could not load image")
        
        except Exception as e:
            self.original_label.config(text=f"❌ Error loading preview:\n{str(e)}")
            print(f"Error loading preview: {e}")

    def load_tiff_preview(self, file_path):
        """Load TIFF preview with improved handling and realistic brightness"""
        try:
            with rasterio.open(file_path) as src:
                # Define preview window size
                height = min(150, src.height)
                width = min(300, src.width)
                window = rasterio.windows.Window(0, 0, width, height)

                # Read RGB bands or replicate grayscale to RGB
                if src.count >= 3:
                    data = src.read([1, 2, 3], window=window)
                else:
                    single_band = src.read(1, window=window)
                    data = np.stack([single_band] * 3, axis=0)

                # Convert CHW -> HWC for display
                image = np.transpose(data, (1, 2, 0))

                # Normalize only for display
                image = image.astype(np.float32)
                min_val = np.nanmin(image)
                max_val = np.nanmax(image)

                if max_val > min_val:
                    image = ((image - min_val) / (max_val - min_val)) * 255
                    image = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    image = np.full_like(image, 128, dtype=np.uint8)

                # Convert to PIL and resize if needed
                image_pil = Image.fromarray(image)
                image_pil.thumbnail((300, 150), Image.Resampling.LANCZOS)

                # Convert to Tkinter-compatible image
                image_tk = ImageTk.PhotoImage(image_pil)

                self.original_label.config(image=image_tk, text="")
                self.original_label.image = image_tk

        except Exception as e:
            self.original_label.config(text=f"❌ Error loading TIFF preview:\n{str(e)}")
            print(f"Error loading TIFF preview: {e}")

    def browse_save_location(self):
        """Browse for tile save location"""
        folder_path = filedialog.askdirectory(title="Select Directory to Save Coordinate-Aware Tiles")
        if folder_path:
            self.tiles_save_path = folder_path
            folder_name = os.path.basename(folder_path) or folder_path
            self.save_location_info.config(text=f"📁 {folder_name}", fg="#27ae60")
            tile_area = self.tile_area_sqm.get()
            side_length = int(math.sqrt(tile_area))
            self.update_status(f"Coordinate-aware tiles ({side_length}×{side_length}m) will be saved to: {folder_name}", "📁")
    def generate_tiles(self):
        """Generate area-based tiles with enhanced user experience"""
        if not self.file_path or not self.is_tif_file:
            messagebox.showerror("Error", "Please select a TIF file first")
            return
        
        if not self.tiles_save_path:
            messagebox.showerror("Error", "Please select a save location for tiles")
            return
        
        try:
            tile_area = float(self.tile_area_entry.get())
            if tile_area <= 0:
                raise ValueError("Tile area must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid tile area (positive number)")
            return
        
        # Show detailed confirmation
        side_length = int(math.sqrt(tile_area))
        
        confirm_msg = f"""Generate area-based tiles with the following configuration?

    📏 Tile Area: {tile_area:,} m²
    📐 Target Tile Dimensions: ~{side_length} × {side_length} meters  
    🗺️ Format: GeoTIFF with coordinate information preserved

    Note: Actual pixel dimensions will be calculated based on the image's 
    Ground Sample Distance (GSD) to ensure accurate area-based tiling.

    This will create multiple GeoTIFF image files in the selected directory.
    Coordinate metadata will be preserved for accurate height-based calculations.

    Existing tiles with similar names may be overwritten.

    Continue with area-based tile generation?"""
        
        if not messagebox.askyesno("Confirm Area-Based Tile Generation", confirm_msg):
            return
        
        # Start generation
        self.generate_tiles_btn.config(state='disabled', text="🔄 Generating Area-Based Tiles...")
        self.show_progress(True)
        self.update_status(f"Breaking TIF into area-based {side_length}×{side_length}m tiles...", "🔄")
        
        threading.Thread(target=self._generate_tiles_thread, daemon=True).start()


    def _generate_tiles_thread(self):
        """Background thread for area-based tile generation"""
        try:
            tile_area = self.tile_area_sqm.get()
            
            # Use the area-based tile creation method
            self.tile_metadata_list = self.create_tiles_from_tif_with_coordinates(
                self.file_path, self.tiles_save_path, tile_area
            )
            
            def update_ui():
                self.tiles_generated = True
                side_length = int(math.sqrt(tile_area))
                
                # Calculate average actual area from metadata
                avg_area = np.mean([t['area_sqm'] for t in self.tile_metadata_list])
                
                self.tiles_status.config(
                    text=f"✅ Generated {len(self.tile_metadata_list)} area-based tiles (avg: {avg_area:.0f} m² each, target: {tile_area:.0f} m²)"
                )
                self.generate_tiles_btn.config(state='normal', text="🔄 Generate Area-Based Tiles")
                self.show_progress(False)
                self.update_status(f"Area-based tiles generated successfully - {len(self.tile_metadata_list)} tiles", "✅")
                self.update_validation_status()
            
            self.root.after(0, update_ui)
        
        except Exception as e:
            error_message = str(e)
            def show_error():
                messagebox.showerror("Area-Based Tile Generation Error", f"Failed to generate area-based tiles:\n{error_message}")
                self.generate_tiles_btn.config(state='normal', text="🔄 Generate Area-Based Tiles")
                self.show_progress(False)
                self.update_status("Area-based tile generation failed", "❌")
            
            self.root.after(0, show_error)

    def visualize_file_alignment(self):
        """Create a visual representation of file alignment (call this after selecting files)"""
        if not (self.dhm_path and self.dtm_path and self.file_path and self.is_tif_file):
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            with rasterio.open(self.file_path) as src_img:
                with rasterio.open(self.dhm_path) as src_dhm:
                    with rasterio.open(self.dtm_path) as src_dtm:
                        
                        fig, ax = plt.subplots(figsize=(12, 10))
                        
                        # Get bounds
                        img_bounds = src_img.bounds
                        dhm_bounds = src_dhm.bounds
                        dtm_bounds = src_dtm.bounds
                        
                        # Create rectangles for each file
                        img_rect = patches.Rectangle(
                            (img_bounds.left, img_bounds.bottom),
                            img_bounds.right - img_bounds.left,
                            img_bounds.top - img_bounds.bottom,
                            linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.3,
                            label=f'Image ({os.path.basename(self.file_path)})'
                        )
                        
                        dhm_rect = patches.Rectangle(
                            (dhm_bounds.left, dhm_bounds.bottom),
                            dhm_bounds.right - dhm_bounds.left,
                            dhm_bounds.top - dhm_bounds.bottom,
                            linewidth=3, edgecolor='red', facecolor='red', alpha=0.3,
                            label=f'DSM ({os.path.basename(self.dhm_path)})'
                        )
                        
                        dtm_rect = patches.Rectangle(
                            (dtm_bounds.left, dtm_bounds.bottom),
                            dtm_bounds.right - dtm_bounds.left,
                            dtm_bounds.top - dtm_bounds.bottom,
                            linewidth=3, edgecolor='green', facecolor='green', alpha=0.3,
                            label=f'DTM ({os.path.basename(self.dtm_path)})'
                        )
                        
                        ax.add_patch(img_rect)
                        ax.add_patch(dhm_rect)
                        ax.add_patch(dtm_rect)
                        
                        # Set axis limits to show all rectangles
                        all_x = [img_bounds.left, img_bounds.right, dhm_bounds.left, dhm_bounds.right,
                                dtm_bounds.left, dtm_bounds.right]
                        all_y = [img_bounds.bottom, img_bounds.top, dhm_bounds.bottom, dhm_bounds.top,
                                dtm_bounds.bottom, dtm_bounds.top]
                        
                        margin_x = (max(all_x) - min(all_x)) * 0.1
                        margin_y = (max(all_y) - min(all_y)) * 0.1
                        
                        ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
                        ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
                        
                        ax.set_xlabel('X Coordinate (meters)', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Y Coordinate (meters)', fontsize=12, fontweight='bold')
                        ax.set_title('Geographic Alignment Visualization', fontsize=14, fontweight='bold')
                        ax.legend(loc='upper right', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.set_aspect('equal')
                        
                        # Add annotations
                        img_center_x = (img_bounds.left + img_bounds.right) / 2
                        img_center_y = (img_bounds.bottom + img_bounds.top) / 2
                        dhm_center_x = (dhm_bounds.left + dhm_bounds.right) / 2
                        dhm_center_y = (dhm_bounds.bottom + dhm_bounds.top) / 2
                        
                        distance = math.sqrt((img_center_x - dhm_center_x)**2 + (img_center_y - dhm_center_y)**2)
                        
                        plt.text(0.02, 0.98, f'Distance between files: {distance:,.0f}m ({distance/1000:.1f}km)',
                                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        
                        plt.tight_layout()
                        plt.show()
                        
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def calculate_tile_size_pixels(self, area_sqm):
        """Calculate tile size in pixels"""
        side_length_m = math.sqrt(area_sqm)
        return int(round(side_length_m / self.ground_sample_distance))

    # ==================== Analysis Methods ====================

    def start_analysis(self):
        """Start coordinate-aware analysis with comprehensive validation"""
        # Check if we have uploaded individual tiles
        if hasattr(self, 'uploaded_tiles') and self.uploaded_tiles:
            selected_indices = self.tile_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select a tile from the uploaded list for analysis")
                return
            
            # Analyze the selected individual tile
            selected_tile = self.uploaded_tiles[selected_indices[0]]
            self.analyze_individual_tile(selected_tile)
            return
        
        # Original validation logic
        if not self.file_path:
            messagebox.showerror("Error", "Please select an image file first")
            return

        if self.is_tif_file and not self.tiles_generated:
            messagebox.showerror("Error", "Please generate coordinate-aware tiles for TIF file first OR upload individual tiles")
            return

        # Rest of the existing validation code remains the same...
        # [Keep all the existing validation code here]
        
        # Check height data requirements
        if self.use_height_calculation.get():
            if not (self.dhm_path and self.dtm_path):
                messagebox.showerror("Error", "Please select both DHM and DTM files for height-based calculation")
                return

            if not self.height_data_aligned:
                response = messagebox.askyesno("Alignment Warning",
                                            "Height data files may not be properly aligned.\n"
                                            "This could result in incorrect floor calculations.\n\n"
                                            "Continue with analysis anyway?")
                if not response:
                    return

        # Validate parameters
        try:
            people_per_household = float(self.household_entry.get())
            standard_area = float(self.standard_area_entry.get())
            medium_threshold = float(self.medium_threshold_entry.get())
            large_threshold = float(self.large_threshold_entry.get())
            
            if self.use_height_calculation.get():
                floor_height = float(self.floor_height_entry.get())
                if floor_height <= 0:
                    raise ValueError("Floor height must be positive")
            
            if people_per_household <= 0 or standard_area <= 0:
                raise ValueError("People per household and standard area must be positive")
            if medium_threshold <= standard_area or large_threshold <= medium_threshold:
                raise ValueError("Thresholds must be: Standard < Medium < Large")
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Please enter valid parameters:\n{str(e)}")
            return

        # Start analysis
        self.analyze_btn.config(state='disabled', text="🔄 ANALYZING...")
        self.show_progress(True)
        
        if self.is_tif_file:
            tile_area = self.tile_area_sqm.get()
            side_length = int(math.sqrt(tile_area))
            if self.use_height_calculation.get():
                self.update_status(f"Analyzing coordinate-aware tiles with floor calculation ({side_length}×{side_length}m each)...", "🔍")
            else:
                self.update_status(f"Analyzing coordinate-aware tiles ({side_length}×{side_length}m each)...", "🔍")
        else:
            if self.use_height_calculation.get():
                self.update_status("Running coordinate-aware building count analysis with floor calculation...", "🔍")
            else:
                self.update_status("Running coordinate-aware building count analysis...", "🔍")

        threading.Thread(target=self._analyze_thread, daemon=True).start()

    def analyze_individual_tile(self, tile_path):
        """Analyze a single uploaded tile"""
        self.analyze_btn.config(state='disabled', text="🔄 ANALYZING TILE...")
        self.show_progress(True)
        
        filename = os.path.basename(tile_path)
        self.update_status(f"Analyzing individual tile: {filename}", "🔍")
        
        threading.Thread(target=self._analyze_individual_tile_thread, args=(tile_path,), daemon=True).start()

    def _analyze_individual_tile_thread(self, tile_path):
        """Background thread for individual tile analysis"""
        try:
            results = self.analyze_single_image(tile_path)
            results['processing_method'] = f'Individual Tile Analysis - {os.path.basename(tile_path)}'
            results['tile_analysis'] = True
            
            self.analysis_results = results
            self.root.after(0, lambda: self.display_results(results))
        except Exception as e:
            error_message = str(e)
            def show_error():
                messagebox.showerror("Analysis Error", f"Tile analysis failed:\n{error_message}")
                self.analyze_btn.config(state='normal', text="START ANALYSIS")
                self.show_progress(False)
                self.update_status("Tile analysis failed", "❌")
            self.root.after(0, show_error)

    def analyze_coordinate_aware_tiles(self):
        """Analyze all coordinate-aware tiles using building count methodology with optional floor calculation"""
        if not self.tile_metadata_list:
            raise Exception("No coordinate-aware tile metadata available")
        
        tile_area = self.tile_area_sqm.get()
        side_length = int(math.sqrt(tile_area))
        
        # Initialize aggregate results
        total_results = {
            'residential_count': 0,
            'non_residential_count': 0,
            'total_population': 0,
            'total_base_population': 0,
            'standard_houses': 0,
            'medium_houses': 0,
            'large_houses': 0,
            'building_details': [],
            'processing_method': f'Coordinate-Aware Building Count Analysis with Floor Calculation ({len(self.tile_metadata_list)} tiles of {side_length}×{side_length}m)' if self.use_height_calculation.get() else f'Coordinate-Aware Building Count Analysis ({len(self.tile_metadata_list)} tiles of {side_length}×{side_length}m)',
            'tile_count': len(self.tile_metadata_list),
            'tile_area': tile_area,
            'tile_dimensions': f'{side_length}×{side_length}m',
            'people_per_household': self.people_per_household.get(),
            'thresholds': {
                'standard': self.standard_house_area.get(),
                'medium': self.medium_house_threshold.get(),
                'large': self.large_house_threshold.get()
            },
            'floor_calculation_enabled': self.use_height_calculation.get(),
            'total_floors': 0,
            'average_floors': 1,
            'floor_height': self.floor_height.get() if self.use_height_calculation.get() else None,
            'floor_calculations': []
        }
        
        # Process each coordinate-aware tile with enhanced error handling
        successful_tiles = 0
        failed_tiles = 0
        
        for i, tile_metadata in enumerate(self.tile_metadata_list):
            try:
                print(f"Processing tile {i+1}/{len(self.tile_metadata_list)}: {os.path.basename(tile_metadata['path'])}")
                
                # Use enhanced tile analysis with 3-channel fixes
                tile_results = self.analyze_coordinate_aware_tile_enhanced(tile_metadata)
                
                if tile_results:
                    # Aggregate results only if analysis was successful
                    total_results['residential_count'] += tile_results['residential_count']
                    total_results['non_residential_count'] += tile_results['non_residential_count']
                    total_results['total_population'] += tile_results['total_population']
                    total_results['total_base_population'] += tile_results.get('total_base_population', 0)
                    total_results['standard_houses'] += tile_results['standard_houses']
                    total_results['medium_houses'] += tile_results['medium_houses']
                    total_results['large_houses'] += tile_results['large_houses']
                    total_results['building_details'].extend(tile_results['building_details'])
                    total_results['total_floors'] += tile_results.get('total_floors', 0)
                    total_results['floor_calculations'].extend(tile_results.get('floor_calculations', []))
                    successful_tiles += 1
                else:
                    failed_tiles += 1
                    print(f"Warning: Tile {i+1} analysis failed, skipping...")
                
            except Exception as e:
                failed_tiles += 1
                print(f"Error processing tile {i+1}: {str(e)}")
                # Continue processing other tiles
                continue
            
            # Update progress
            progress = (i + 1) / len(self.tile_metadata_list) * 100
            status_msg = f"Processed {i+1}/{len(self.tile_metadata_list)} coordinate-aware tiles ({progress:.1f}%) - Success: {successful_tiles}, Failed: {failed_tiles}"
            self.root.after(0, lambda msg=status_msg: self.update_status(msg, "🔍"))
        
        # Final processing summary
        print(f"\n=== TILE PROCESSING SUMMARY ===")
        print(f"Total tiles: {len(self.tile_metadata_list)}")
        print(f"Successfully processed: {successful_tiles}")
        print(f"Failed tiles: {failed_tiles}")
        print(f"Success rate: {(successful_tiles/len(self.tile_metadata_list)*100):.1f}%")
        
        # Calculate average floors
        if total_results['residential_count'] > 0:
            total_results['average_floors'] = total_results['total_floors'] / total_results['residential_count']
        
        # Final population count (ceiling)
        total_results['total_population'] = math.ceil(total_results['total_population'])
        total_results['total_base_population'] = math.ceil(total_results['total_base_population'])
        
        # Add processing statistics to results
        total_results['processing_stats'] = {
            'total_tiles': len(self.tile_metadata_list),
            'successful_tiles': successful_tiles,
            'failed_tiles': failed_tiles,
            'success_rate': (successful_tiles/len(self.tile_metadata_list)*100) if len(self.tile_metadata_list) > 0 else 0
        }
        
        self.building_data = total_results['building_details']
        
        return total_results

    def _analyze_thread(self):
        """Background analysis thread"""
        try:
            if self.is_tif_file:
                results = self.analyze_coordinate_aware_tiles()
            else:
                results = self.analyze_single_image(self.file_path)
            
            self.analysis_results = results
            self.root.after(0, lambda: self.display_results(results))
        
        except Exception as e:
            error_message = str(e)
            def show_error():
                messagebox.showerror("Analysis Error", f"Analysis failed:\n{error_message}")
                self.analyze_btn.config(state='normal', text=" START ANALYSIS")
                self.show_progress(False)
                self.update_status("Analysis failed", "❌")
            
            self.root.after(0, show_error)
    
    
    def analyze_coordinate_aware_tile_enhanced(self, tile_metadata):
        """Enhanced tile analysis with proper 3-channel image handling"""
        tile_path = tile_metadata['path']
        
        tile_results = {
            'residential_count': 0,
            'non_residential_count': 0,
            'total_population': 0,
            'total_base_population': 0,
            'standard_houses': 0,
            'medium_houses': 0,
            'large_houses': 0,
            'building_details': [],
            'total_floors': 0,
            'floor_calculations': []
        }
        
        try:
            print(f"  📍 Analyzing tile: {os.path.basename(tile_path)}")
            
            # Debug image information
            self.debug_image_info(tile_path)
            
            # Extract height data for this specific tile if enabled
            dhm_data, dtm_data, tile_transform = None, None, None
            if self.use_height_calculation.get():
                try:
                    dhm_data, dtm_data, tile_transform = self.extract_height_data_for_tile(tile_metadata)
                    print(f"  🗻 Height data extracted successfully")
                except Exception as e:
                    print(f"  ⚠️ Height data extraction failed: {e}")
                    dhm_data = dtm_data = tile_transform = None
            
            # Preprocess tile to ensure proper 3-channel RGB format
            try:
                processed_tile_path = self.preprocess_tile_for_yolo_enhanced(tile_path)
                print(f"  🔄 Tile preprocessed: {os.path.basename(processed_tile_path)}")
                
                # Validate the processed image has exactly 3 channels
                is_valid, validation_msg = self.validate_image_channels(processed_tile_path)
                if not is_valid:
                    print(f"  ❌ Channel validation failed: {validation_msg}")
                    return None
                else:
                    print(f"  ✅ Channel validation passed: {validation_msg}")
                    
            except Exception as e:
                print(f"  ❌ Tile preprocessing failed: {e}")
                return None
            
            # Run YOLO detection with enhanced error handling
            try:
                print(f"  🔍 Running YOLO detection...")
                
                # Double-check image format before YOLO
                test_img = cv2.imread(processed_tile_path)
                if test_img is None or len(test_img.shape) != 3 or test_img.shape[2] != 3:
                    raise Exception(f"Invalid image format for YOLO: {test_img.shape if test_img is not None else 'None'}")
                
                results = model.predict(source=processed_tile_path, save=False, imgsz=640, verbose=False)
                print(f"  ✅ YOLO detection completed")
                
            except Exception as e:
                print(f"  ❌ YOLO detection failed: {e}")
                # Clean up temporary file
                if processed_tile_path != tile_path and os.path.exists(processed_tile_path):
                    try:
                        os.remove(processed_tile_path)
                    except:
                        pass
                return None
            
            # Clean up temporary file after successful detection
            if processed_tile_path != tile_path and os.path.exists(processed_tile_path):
                try:
                    os.remove(processed_tile_path)
                except:
                    pass
            
            # Process detection results
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"  🏠 Found {len(result.boxes)} detections")
                    
                    for j, box in enumerate(result.boxes):
                        try:
                            # Extract detection information - FIXED tensor access
                            x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                            confidence = float(box.conf.cpu().numpy())
                            class_id = int(box.cls.cpu().numpy())
                            
                            # Calculate building area in square meters
                            building_area_px = width * height
                            building_area_sqm = building_area_px * (self.ground_sample_distance ** 2)
                            
                            # Calculate floors using coordinate-aware height data
                            floors = 1  # Default
                            if (self.use_height_calculation.get() and 
                                dhm_data is not None and dtm_data is not None):
                                try:
                                    floors = self.calculate_building_floors_from_tile_data(
                                        int(x_center), int(y_center), dhm_data, dtm_data, tile_transform
                                    )
                                except Exception as e:
                                    print(f"    ⚠️ Floor calculation failed for detection {j}: {e}")
                                    floors = 1
                            
                            # Convert pixel coordinates to world coordinates
                            try:
                                world_x, world_y = rasterio.transform.xy(tile_metadata['transform'], y_center, x_center)
                            except Exception as e:
                                print(f"    ⚠️ Coordinate transformation failed for detection {j}: {e}")
                                world_x = world_y = "N/A"
                            
                            # Only count residential buildings for population
                            if class_id == 0:  # Residential
                                tile_results['residential_count'] += 1
                                
                                # Determine house category and population multiplier
                                if building_area_sqm < self.medium_house_threshold.get():
                                    house_category = "Standard"
                                    population_multiplier = 1
                                    tile_results['standard_houses'] += 1
                                elif building_area_sqm < self.large_house_threshold.get():
                                    house_category = "Medium"
                                    population_multiplier = 2
                                    tile_results['medium_houses'] += 1
                                else:
                                    house_category = "Large"
                                    population_multiplier = 3
                                    tile_results['large_houses'] += 1
                                
                                # Calculate base population (without floors)
                                base_population = population_multiplier * self.people_per_household.get()
                                
                                # Calculate final population (with floors)
                                final_population = base_population * floors
                                
                                tile_results['total_base_population'] += base_population
                                tile_results['total_population'] += final_population
                                tile_results['total_floors'] += floors
                                
                                # Store floor calculation details
                                if self.use_height_calculation.get():
                                    tile_results['floor_calculations'].append({
                                        'world_x': world_x,
                                        'world_y': world_y,
                                        'floors': floors,
                                        'base_population': base_population,
                                        'final_population': final_population
                                    })
                            else:
                                tile_results['non_residential_count'] += 1
                                house_category = "Non-Residential"
                                population_multiplier = 0
                                base_population = 0
                                final_population = 0
                                floors = 1  # Default for non-residential
                            
                            # Store building details with proper field names
                            building_detail = {
                                'tile_path': tile_path,
                                'tile_row': tile_metadata['row'],
                                'tile_col': tile_metadata['col'],
                                'detection_id': j,
                                'pixel_x': float(x_center),
                                'pixel_y': float(y_center),
                                'world_x': float(world_x) if world_x != "N/A" else "N/A",
                                'world_y': float(world_y) if world_y != "N/A" else "N/A",
                                'width_px': float(width),  # Fixed field name
                                'height_px': float(height),  # Fixed field name
                                'area_sqm': building_area_sqm,
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': class_names.get(class_id, 'Unknown'),
                                'house_category': house_category,
                                'population_multiplier': population_multiplier,
                                'base_population': base_population,
                                'floors': floors,
                                'final_population': final_population,
                                'bbox': box.xyxy[0].cpu().numpy().tolist()
                            }
                            
                            tile_results['building_details'].append(building_detail)
                            
                        except Exception as e:
                            print(f"    ❌ Error processing detection {j}: {e}")
                            continue
                    
                    print(f"  📊 Tile results: {tile_results['residential_count']} residential, {tile_results['non_residential_count']} non-residential")
                else:
                    print(f"  📭 No detections found in tile")
            else:
                print(f"  📭 No YOLO results for tile")
            
            return tile_results
            
        except Exception as e:
            print(f"  ❌ Critical error analyzing tile {tile_path}: {e}")
            return None


    def preprocess_tile_for_yolo_enhanced(self, tile_path: str) -> str:
        """
        Always return a temporary 3-channel RGB PNG suitable for YOLO.
        The caller may delete the returned file when done.
        """
        try:
            import tempfile

            # TIFF / GeoTIFF
            if tile_path.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(tile_path) as src:
                    # Build 3 channels
                    if src.count >= 3:
                        data = src.read([1, 2, 3]).transpose(1, 2, 0)  # CHW -> HWC
                    else:
                        band = src.read(1)
                        data = np.stack([band, band, band], axis=2)

                # Normalize to 0–255 for PNG
                data = data.astype(np.float32)
                dmin, dmax = np.nanmin(data), np.nanmax(data)
                if dmax > dmin:
                    data = (data - dmin) / (dmax - dmin) * 255.0
                else:
                    data = np.full_like(data, 128, dtype=np.float32)
                data = np.clip(data, 0, 255).astype(np.uint8)

                tmp = tempfile.mktemp(suffix='.png')
                Image.fromarray(data, mode='RGB').save(tmp, 'PNG')
                return tmp

            # Regular image files
            image = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise Exception("Could not load image file")

            # Ensure 3 channels RGB
            if len(image.shape) == 2:  # grayscale
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 1:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise Exception(f"Unexpected channel count: {image.shape[2]}")
            else:
                raise Exception(f"Unexpected image shape: {image.shape}")

            tmp = tempfile.mktemp(suffix='.png')
            Image.fromarray(image_rgb, mode='RGB').save(tmp, 'PNG')
            return tmp

        except Exception as e:
            # Safe fallback so the pipeline never crashes
            import tempfile
            tmp = tempfile.mktemp(suffix='.png')
            Image.fromarray(np.full((640, 640, 3), 128, dtype=np.uint8), mode='RGB').save(tmp, 'PNG')
            print(f"    ❌ Preprocess failed ({e}). Using fallback RGB image.")
            return tmp

    

    def analyze_single_image(self, image_path):
        """Analyze a single image using coordinate-aware building count methodology"""
        results = {
            'residential_count': 0,
            'non_residential_count': 0,
            'total_population': 0,
            'total_base_population': 0,
            'standard_houses': 0,
            'medium_houses': 0,
            'large_houses': 0,
            'building_details': [],
            'processing_method': 'Coordinate-Aware Building Count Analysis (Single Image)' + (' with Floor Calculation' if self.use_height_calculation.get() else ''),
            'tile_count': 1,
            'tile_area': 'N/A - Single Image',
            'tile_dimensions': 'N/A - Single Image',
            'people_per_household': self.people_per_household.get(),
            'thresholds': {
                'standard': self.standard_house_area.get(),
                'medium': self.medium_house_threshold.get(),
                'large': self.large_house_threshold.get()
            },
            'floor_calculation_enabled': self.use_height_calculation.get(),
            'total_floors': 0,
            'average_floors': 1,
            'floor_height': self.floor_height.get() if self.use_height_calculation.get() else None,
            'floor_calculations': []
        }
        
        try:
            # Run YOLO detection
            # Preprocess the image for YOLO if needed
            processed_image_path = self.preprocess_tile_for_yolo(image_path)
            yolo_results = model.predict(source=processed_image_path, save=False, imgsz=640)

            # Clean up temporary file if created
            if processed_image_path != image_path and os.path.exists(processed_image_path):
                try:
                    os.remove(processed_image_path)
                except:
                    pass
            
            if yolo_results and len(yolo_results) > 0:
                result = yolo_results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        # Extract detection information
                        x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Calculate building area in square meters
                        building_area_px = width * height
                        building_area_sqm = building_area_px * (self.ground_sample_distance ** 2)
                        
                        # For single images, default floor calculation is 1 unless height data is available
                        floors = 1
                        
                        # Only count residential buildings for population
                        if class_id == 0:  # Residential
                            results['residential_count'] += 1
                            
                            # Determine house category and population multiplier
                            if building_area_sqm < self.medium_house_threshold.get():
                                house_category = "Standard"
                                population_multiplier = 1
                                results['standard_houses'] += 1
                            elif building_area_sqm < self.large_house_threshold.get():
                                house_category = "Medium"
                                population_multiplier = 2
                                results['medium_houses'] += 1
                            else:
                                house_category = "Large"
                                population_multiplier = 3
                                results['large_houses'] += 1
                            
                            # Calculate base and final population
                            base_population = population_multiplier * self.people_per_household.get()
                            final_population = base_population * floors
                            
                            results['total_base_population'] += base_population
                            results['total_population'] += final_population
                            results['total_floors'] += floors
                        else:
                            results['non_residential_count'] += 1
                            house_category = "Non-Residential"
                            population_multiplier = 0
                            base_population = 0
                            final_population = 0
                        
                        # Store building details
                        building_detail = {
                            'detection_id': i,
                            'pixel_x': float(x_center),
                            'pixel_y': float(y_center),
                            'world_x': 'N/A - Single Image',
                            'world_y': 'N/A - Single Image',
                            'width_px': float(width),
                            'height_px': float(height),
                            'area_sqm': building_area_sqm,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_names.get(class_id, 'Unknown'),
                            'house_category': house_category,
                            'population_multiplier': population_multiplier,
                            'base_population': base_population,
                            'floors': floors,
                            'final_population': final_population,
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        }
                        
                        results['building_details'].append(building_detail)
        
        except Exception as e:
            print(f"Error analyzing single image: {e}")
            raise
        
        # Calculate average floors
        if results['residential_count'] > 0:
            results['average_floors'] = results['total_floors'] / results['residential_count']
        
        # Final population count (ceiling)
        results['total_population'] = math.ceil(results['total_population'])
        results['total_base_population'] = math.ceil(results['total_base_population'])
        
        self.building_data = results['building_details']
        
        return results

    # ==================== Results Display Methods ====================

    def display_results(self, results):
        """Display analysis results with enhanced visualization"""
        self.analyze_btn.config(state='normal', text=" START ANALYSIS")
        self.show_progress(False)
        
        # Update summary cards
        self.update_summary_cards(results)
        
        # Update summary tree
        self.update_summary_tree(results)
        
        # Generate and display basic charts
        self.generate_basic_charts(results)
        
        # Enable export and visualization buttons
        self.export_csv_btn.config(state='normal')
        self.generate_viz_btn.config(state='normal')
        
        # Update processed image display
        self.update_processed_image_display(results)
        
        # Update status
        if results['floor_calculation_enabled']:
            self.update_status(f"Coordinate-aware analysis complete with floor calculation: {results['total_population']:,} people estimated", "✅")
        else:
            self.update_status(f"Coordinate-aware analysis complete: {results['total_population']:,} people estimated", "✅")
        
        # Success message
        messagebox.showinfo("Analysis Complete", 
                          f"Coordinate-aware analysis completed successfully!\n\n"
                          f"📊 Total Population: {results['total_population']:,} people\n"
                          f"🏠 Residential Buildings: {results['residential_count']:,}\n"
                          f"🏢 Non-Residential Buildings: {results['non_residential_count']:,}\n"
                          + (f"🏢 Average Floors: {results['average_floors']:.1f}\n" if results['floor_calculation_enabled'] else "") +
                          f"📈 View detailed results in the tabs below.")

    def update_summary_cards(self, results):
        """Update summary cards with results"""
        card_values = [
            f"{results['total_population']:,}",
            f"{results['residential_count']:,}",
            f"{results['standard_houses']:,}",
            f"{results['large_houses']:,}",
            f"{results['average_floors']:.1f}" if results['floor_calculation_enabled'] else "1.0"
        ]
        
        for card, value in zip(self.summary_cards, card_values):
            card.config(text=value)

    def update_summary_tree(self, results):
        """Update summary tree with detailed results"""
        # Clear existing items
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        # Population section
        pop_parent = self.summary_tree.insert("", "end", text="Population Analysis", 
                                            values=("📊 Population Analysis", "", ""))
        
        self.summary_tree.insert(pop_parent, "end", 
                               values=("Total Population", f"{results['total_population']:,} people", 
                                     f"Final estimate including coordinate-aware floor calculation" if results['floor_calculation_enabled'] else "Final estimate based on building count"))
        
        if results['floor_calculation_enabled']:
            self.summary_tree.insert(pop_parent, "end", 
                                   values=("Base Population (without floors)", f"{results['total_base_population']:,} people", 
                                         "Population before floor multiplication"))
        
        self.summary_tree.insert(pop_parent, "end", 
                               values=("People per Household", f"{results['people_per_household']:.1f}", 
                                     "Parameter used in calculations"))
        
        # Building section
        building_parent = self.summary_tree.insert("", "end", text="Building Analysis", 
                                                  values=("🏠 Building Analysis", "", ""))
        
        self.summary_tree.insert(building_parent, "end", 
                               values=("Total Buildings", f"{results['residential_count'] + results['non_residential_count']:,}", 
                                     "All detected buildings"))
        
        self.summary_tree.insert(building_parent, "end", 
                               values=("Residential Buildings", f"{results['residential_count']:,}", 
                                     "Buildings contributing to population"))
        
        self.summary_tree.insert(building_parent, "end", 
                               values=("Non-Residential Buildings", f"{results['non_residential_count']:,}", 
                                     "Commercial/Industrial buildings"))
        
        # House categories
        cat_parent = self.summary_tree.insert("", "end", text="House Categories", 
                                            values=("🏡 House Categories", "", ""))
        
        self.summary_tree.insert(cat_parent, "end", 
                               values=("Standard Houses", f"{results['standard_houses']:,}", 
                                     f"< {results['thresholds']['medium']:,} m² (1x multiplier)"))
        
        self.summary_tree.insert(cat_parent, "end", 
                               values=("Medium Houses", f"{results['medium_houses']:,}", 
                                     f"{results['thresholds']['medium']:,} - {results['thresholds']['large']:,} m² (2x multiplier)"))
        
        self.summary_tree.insert(cat_parent, "end", 
                               values=("Large Houses", f"{results['large_houses']:,}", 
                                     f"> {results['thresholds']['large']:,} m² (3x multiplier)"))
        
        # Floor information (if available)
        if results['floor_calculation_enabled']:
            floor_parent = self.summary_tree.insert("", "end", text="Floor Analysis", 
                                                   values=("🏢 Coordinate-Aware Floor Analysis", "", ""))
            
            self.summary_tree.insert(floor_parent, "end", 
                                   values=("Floor Calculation Enabled", "Yes", 
                                         "Using DHM/DTM height data"))
            
            self.summary_tree.insert(floor_parent, "end", 
                                   values=("Average Floors", f"{results['average_floors']:.1f}", 
                                         "Average floors per residential building"))
            
            self.summary_tree.insert(floor_parent, "end", 
                                   values=("Total Floor Count", f"{results['total_floors']:,}", 
                                         "Sum of all floors in residential buildings"))
            
            self.summary_tree.insert(floor_parent, "end", 
                                   values=("Floor Height Parameter", f"{results['floor_height']:.1f} m", 
                                         "Height per floor used in calculations"))
        
        # Processing information
        proc_parent = self.summary_tree.insert("", "end", text="Processing Details", 
                                             values=("⚙️ Processing Details", "", ""))
        
        self.summary_tree.insert(proc_parent, "end", 
                               values=("Processing Method", results['processing_method'], 
                                     "Coordinate-aware methodology used"))
        
        if results['tile_count'] > 1:
            self.summary_tree.insert(proc_parent, "end", 
                                   values=("Tile Count", f"{results['tile_count']:,}", 
                                         "Number of coordinate-aware tiles processed"))
            
            self.summary_tree.insert(proc_parent, "end", 
                                   values=("Tile Dimensions", results['tile_dimensions'], 
                                         "Size of each coordinate-aware tile"))
            
            self.summary_tree.insert(proc_parent, "end", 
                                   values=("Tile Area", f"{results['tile_area']:,} m²", 
                                         "Area covered by each tile"))
        
        # Expand all sections
        for item in self.summary_tree.get_children():
            self.summary_tree.item(item, open=True)

    def generate_basic_charts(self, results):
        """Generate basic charts with improved design"""
        # Clear previous charts
        for widget in self.basic_charts_frame.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Chart 1: Building Type Distribution
        self.create_building_type_chart(results)
        
        # Chart 2: House Category Distribution
        self.create_house_category_chart(results)
        
        # Chart 3: Population Breakdown
        self.create_population_breakdown_chart(results)
        
        # Chart 4: Floor Analysis (if available)
        if results['floor_calculation_enabled']:
            self.create_floor_analysis_chart(results)

    def create_building_type_chart(self, results):
        """Create building type distribution chart"""
        chart_frame = tk.LabelFrame(self.basic_charts_frame.scrollable_frame, 
                                  text="🏠 Building Type Distribution",
                                  font=('Arial', 11, 'bold'), bg="white", fg="#2c3e50",
                                  padx=10, pady=10)
        chart_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        fig = Figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_subplot(111)
        
        # Data
        labels = ['Residential', 'Non-Residential']
        sizes = [results['residential_count'], results['non_residential_count']]
        colors = ['#3498db', '#e74c3c']
        
        if sum(sizes) > 0:
            # Create pie chart
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 10})
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Building Type Distribution', fontsize=12, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, 'No buildings detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Building Type Distribution', fontsize=12, fontweight='bold')
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_house_category_chart(self, results):
        """Create house category distribution chart"""
        chart_frame = tk.LabelFrame(self.basic_charts_frame.scrollable_frame, 
                                  text="🏡 House Size Categories",
                                  font=('Arial', 11, 'bold'), bg="white", fg="#2c3e50",
                                  padx=10, pady=10)
        chart_frame.pack(fill="x", padx=10, pady=5)
        
        fig = Figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_subplot(111)
        
        # Data
        categories = ['Standard Houses\n(1x multiplier)', 'Medium Houses\n(2x multiplier)', 'Large Houses\n(3x multiplier)']
        counts = [results['standard_houses'], results['medium_houses'], results['large_houses']]
        colors = ['#27ae60', '#f39c12', '#9b59b6']
        
        if sum(counts) > 0:
            bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                           f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Residential Building Size Categories', fontsize=12, fontweight='bold', pad=20)
            ax.set_ylabel('Number of Buildings', fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No residential buildings detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Residential Building Size Categories', fontsize=12, fontweight='bold')
        
        # Format chart
        ax.tick_params(axis='x', rotation=0)
        fig.tight_layout()
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_population_breakdown_chart(self, results):
        """Create population breakdown chart"""
        chart_frame = tk.LabelFrame(self.basic_charts_frame.scrollable_frame, 
                                  text="👥 Population Breakdown by House Category",
                                  font=('Arial', 11, 'bold'), bg="white", fg="#2c3e50",
                                  padx=10, pady=10)
        chart_frame.pack(fill="x", padx=10, pady=5)
        
        fig = Figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_subplot(111)
        
        # Calculate population by category
        people_per_household = results['people_per_household']
        
        if results['floor_calculation_enabled']:
            # Calculate average floors per category from building details
            standard_floors = sum([b['floors'] for b in results.get('building_details', []) 
                                 if b['house_category'] == 'Standard']) or 0
            medium_floors = sum([b['floors'] for b in results.get('building_details', []) 
                               if b['house_category'] == 'Medium']) or 0
            large_floors = sum([b['floors'] for b in results.get('building_details', []) 
                              if b['house_category'] == 'Large']) or 0
            
            standard_pop = standard_floors * people_per_household
            medium_pop = medium_floors * people_per_household  
            large_pop = large_floors * people_per_household
        else:
            # Without floor calculation
            standard_pop = results['standard_houses'] * 1 * people_per_household
            medium_pop = results['medium_houses'] * 2 * people_per_household
            large_pop = results['large_houses'] * 3 * people_per_household
        
        # Data
        categories = ['Standard Houses', 'Medium Houses', 'Large Houses']
        populations = [standard_pop, medium_pop, large_pop]
        colors = ['#27ae60', '#f39c12', '#9b59b6']
        
        if sum(populations) > 0:
            bars = ax.bar(categories, populations, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar, pop in zip(bars, populations):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(populations)*0.01,
                           f'{pop:,.0f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Population Distribution by House Category' + 
                        (' (Including Floor Calculation)' if results['floor_calculation_enabled'] else ''), 
                        fontsize=12, fontweight='bold', pad=20)
            ax.set_ylabel('Population', fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No population calculated', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Population Distribution by House Category', fontsize=12, fontweight='bold')
        
        # Format chart
        fig.tight_layout()
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_floor_analysis_chart(self, results):
        """Create floor analysis chart when height data is available"""
        chart_frame = tk.LabelFrame(self.basic_charts_frame.scrollable_frame, 
                                  text="🏢 Coordinate-Aware Floor Analysis",
                                  font=('Arial', 11, 'bold'), bg="white", fg="#2c3e50",
                                  padx=10, pady=10)
        chart_frame.pack(fill="x", padx=10, pady=5)
        
        fig = Figure(figsize=(12, 8), facecolor='white')
        
        # Create subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # Get floor data from building details
        building_details = results.get('building_details', [])
        residential_buildings = [b for b in building_details if b['class_id'] == 0]
        
        if residential_buildings:
            floors_data = [b['floors'] for b in residential_buildings]
            
            # Chart 1: Floor distribution histogram
            ax1.hist(floors_data, bins=max(1, min(10, len(set(floors_data)))), 
                    color='#3498db', alpha=0.7, edgecolor='white', linewidth=2)
            ax1.set_title('Floor Distribution', fontweight='bold')
            ax1.set_xlabel('Number of Floors')
            ax1.set_ylabel('Number of Buildings')
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Chart 2: Average floors by category
            categories = ['Standard', 'Medium', 'Large']
            avg_floors = []
            for cat in categories:
                cat_buildings = [b for b in residential_buildings if b['house_category'] == cat]
                if cat_buildings:
                    avg_floors.append(sum(b['floors'] for b in cat_buildings) / len(cat_buildings))
                else:
                    avg_floors.append(0)
            
            colors = ['#27ae60', '#f39c12', '#9b59b6']
            bars = ax2.bar(categories, avg_floors, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels
            for bar, avg in zip(bars, avg_floors):
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_floors)*0.01,
                           f'{avg:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_title('Average Floors by Category', fontweight='bold')
            ax2.set_ylabel('Average Floors')
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Chart 3: Population impact comparison
            base_populations = [b['base_population'] for b in residential_buildings]
            final_populations = [b['final_population'] for b in residential_buildings]
            
            x = range(min(20, len(residential_buildings)))  # Show first 20 buildings
            width = 0.35
            
            ax3.bar([i - width/2 for i in x], base_populations[:20], width, 
                   label='Base Population', color='#95a5a6', alpha=0.7)
            ax3.bar([i + width/2 for i in x], final_populations[:20], width, 
                   label='Final Population (with floors)', color='#e74c3c', alpha=0.7)
            
            ax3.set_title('Population Impact of Floor Calculation (First 20 Buildings)', fontweight='bold')
            ax3.set_xlabel('Building Index')
            ax3.set_ylabel('Population')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            # No data available
            for ax in [ax1, ax2, ax3]:
                ax.text(0.5, 0.5, 'No residential buildings detected', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        
        fig.suptitle('Coordinate-Aware Floor Analysis Results', fontsize=14, fontweight='bold')
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_processed_image_display(self, results):
        """Update processed image display with detection results"""
        try:
            if self.is_tif_file and self.tile_metadata_list:
                # For TIF files, show a sample tile with detections
                sample_tile = self.tile_metadata_list[0]['path']
                self.display_processed_tile(sample_tile, results)
            else:
                # For single images
                self.display_processed_single_image(results)
        except Exception as e:
            self.processed_label.config(text=f"❌ Error displaying results:\n{str(e)}")
            print(f"Error displaying processed image: {e}")

    def display_processed_tile(self, tile_path, results):
        """Display a processed tile with detection overlays for TIF files."""
        try:
            # Load tile as RGB for drawing
            if tile_path.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(tile_path) as src:
                    if src.count >= 3:
                        data = src.read([1, 2, 3]).transpose(1, 2, 0)
                    else:
                        band = src.read(1)
                        data = np.stack([band, band, band], axis=2)
                    data = data.astype(np.float32)
                    dmin, dmax = np.nanmin(data), np.nanmax(data)
                    data = ((data - dmin) / (dmax - dmin) * 255.0) if dmax > dmin else np.full_like(data, 128)
                    image = np.clip(data, 0, 255).astype(np.uint8)
            else:
                image = cv2.cvtColor(cv2.imread(tile_path), cv2.COLOR_BGR2RGB)

            h, w = image.shape[:2]

            # Pick detections that belong to this tile
            tile_detections = [d for d in results.get('building_details', []) if d.get('tile_path') == tile_path]

            for d in tile_detections:
                x_center = d.get('pixel_x', 0.0)
                y_center = d.get('pixel_y', 0.0)
                width_px = d.get('width_px', d.get('width', 0.0))
                height_px = d.get('height_px', d.get('height', 0.0))

                x1 = int(x_center - width_px / 2)
                y1 = int(y_center - height_px / 2)
                x2 = int(x_center + width_px / 2)
                y2 = int(y_center + height_px / 2)

                # clamp to image
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                class_id = int(d.get('class_id', 0))
                confidence = float(d.get('confidence', 0.0))
                house_category = d.get('house_category', 'Unknown')
                floors = int(d.get('floors', 1))

                color = (255, 0, 255) if class_id == 0 else (0, 165, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                if results.get('floor_calculation_enabled', False) and floors > 1:
                    label = f"{house_category[:4]} {confidence:.2f} ({floors}F)"
                else:
                    label = f"{house_category[:4]} {confidence:.2f}"

                (lw, lh) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                y_text = max(0, y1 - lh - 2)
                cv2.rectangle(image, (x1, y_text), (x1 + lw + 4, y_text + lh + 2), color, -1)
                cv2.putText(image, label, (x1 + 2, y_text + lh), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            image_pil = Image.fromarray(image)
            image_pil.thumbnail((300, 150), Image.Resampling.LANCZOS)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.processed_label.config(image=image_tk, text="")
            self.processed_label.image = image_tk

        except Exception as e:
            self.processed_label.config(text=f"❌ Error displaying tile:\n{str(e)}")
            print(f"Error in display_processed_tile: {e}")


    def display_processed_single_image(self, _results):
        """Display processed single image with detection overlays (uses RGB preprocessed image)."""
        processed_path = None
        try:
            # Always convert to 3-channel RGB PNG before running YOLO
            processed_path = self.preprocess_tile_for_yolo_enhanced(self.file_path)

            image = cv2.imread(processed_path)
            if image is None:
                raise Exception("Could not load preprocessed image for display")

            # Run YOLO on the preprocessed RGB image (not on the original path)
            results_display = model.predict(source=processed_path, save=False, imgsz=640, verbose=False)

            if results_display and len(results_display) > 0:
                result = results_display[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())

                        color = colors.get(class_id, (255, 255, 255))
                        label = f"{class_names.get(class_id, 'Unknown')} {confidence:.2f}"

                        # Draw box
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        # Label background + text
                        (lw, lh) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        y_text = max(0, int(y1) - lh - 2)
                        cv2.rectangle(image, (int(x1), y_text), (int(x1) + lw + 4, y_text + lh + 2), color, -1)
                        cv2.putText(image, label, (int(x1) + 2, y_text + lh),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Convert to RGB/PIL and show
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_pil.thumbnail((300, 150), Image.Resampling.LANCZOS)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.processed_label.config(image=image_tk, text="")
            self.processed_label.image = image_tk

        except Exception as e:
            self.processed_label.config(text=f"❌ Error displaying image:\n{str(e)}")
        finally:
            # Clean up the temp file
            if processed_path and processed_path != self.file_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except:
                    pass

    # ==================== Export Methods ====================

    def export_to_csv(self):

        # Use building_data or analysis_results depending on what’s filled
        results = self.building_data if self.building_data else self.analysis_results

        if not results or len(results) == 0:
            messagebox.showwarning("No Data", "No analysis results available to export.")
            return

        try:
            # If results is a dict, wrap into list
            if isinstance(results, dict):
                results = [results]

            # Flatten nested dicts if necessary
            clean_results = []
            for item in results:
                if isinstance(item, dict):
                    flat = {}
                    for k, v in item.items():
                        if isinstance(v, (list, dict)):
                            flat[k] = str(v)  # convert lists/dicts to string
                        else:
                            flat[k] = v
                    clean_results.append(flat)
                else:
                    clean_results.append({"value": item})

            df = pd.DataFrame(clean_results)

            # Ask user for save path
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )

            if file_path:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{e}")



    def export_detailed_csv(self, filename, results):
        """Export detailed building data to CSV"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header information
            writer.writerow(['# Coordinate-Aware Building Analysis - Detailed Results'])
            writer.writerow(['# Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['# Processing Method:', results['processing_method']])
            writer.writerow(['# Floor Calculation:', 'Enabled' if results['floor_calculation_enabled'] else 'Disabled'])
            writer.writerow([''])
            
            # Summary
            writer.writerow(['## SUMMARY'])
            writer.writerow(['Total Population', results['total_population']])
            if results['floor_calculation_enabled']:
                writer.writerow(['Base Population (without floors)', results['total_base_population']])
            writer.writerow(['Residential Buildings', results['residential_count']])
            writer.writerow(['Non-Residential Buildings', results['non_residential_count']])
            writer.writerow(['Standard Houses', results['standard_houses']])
            writer.writerow(['Medium Houses', results['medium_houses']])
            writer.writerow(['Large Houses', results['large_houses']])
            if results['floor_calculation_enabled']:
                writer.writerow(['Average Floors', f"{results['average_floors']:.2f}"])
                writer.writerow(['Total Floors', results['total_floors']])
            writer.writerow([''])
            
            # Parameters
            writer.writerow(['## PARAMETERS'])
            writer.writerow(['People per Household', results['people_per_household']])
            writer.writerow(['Standard House Threshold', f"{results['thresholds']['standard']} m²"])
            writer.writerow(['Medium House Threshold', f"{results['thresholds']['medium']} m²"])
            writer.writerow(['Large House Threshold', f"{results['thresholds']['large']} m²"])
            if results['floor_calculation_enabled']:
                writer.writerow(['Floor Height', f"{results['floor_height']} m"])
            
            if results['tile_count'] > 1:
                writer.writerow(['Tile Count', results['tile_count']])
                writer.writerow(['Tile Area', f"{results['tile_area']} m²"])
                writer.writerow(['Tile Dimensions', results['tile_dimensions']])
            writer.writerow([''])
            
            # Building details header
            writer.writerow(['## BUILDING DETAILS'])
            
            # Column headers for building details
            headers = ['Building_ID', 'Tile_Path', 'Tile_Row', 'Tile_Col', 
                      'Pixel_X', 'Pixel_Y', 'World_X', 'World_Y',
                      'Width_Pixels', 'Height_Pixels', 'Area_SqM', 
                      'Confidence', 'Class_ID', 'Class_Name', 
                      'House_Category', 'Population_Multiplier', 
                      'Base_Population', 'Floors', 'Final_Population',
                      'BBox_X1', 'BBox_Y1', 'BBox_X2', 'BBox_Y2']
            
            writer.writerow(headers)
            
            # Building details data
            for i, building in enumerate(results['building_details']):
                bbox = building.get('bbox', [0, 0, 0, 0])
                row = [
                    i + 1,
                    building.get('tile_path', 'N/A'),
                    building.get('tile_row', 'N/A'),
                    building.get('tile_col', 'N/A'),
                    f"{building['pixel_x']:.2f}",
                    f"{building['pixel_y']:.2f}",
                    building.get('world_x', 'N/A'),
                    building.get('world_y', 'N/A'),
                    f"{building['width_px']:.2f}",
                    f"{building['height_px']:.2f}",
                    f"{building['area_sqm']:.2f}",
                    f"{building['confidence']:.3f}",
                    building['class_id'],
                    building['class_name'],
                    building['house_category'],
                    building['population_multiplier'],
                    f"{building['base_population']:.1f}",
                    building['floors'],
                    f"{building['final_population']:.1f}",
                    f"{bbox[0]:.2f}",
                    f"{bbox[1]:.2f}",
                    f"{bbox[2]:.2f}",
                    f"{bbox[3]:.2f}"
                ]
                writer.writerow(row)

    def export_summary_csv(self, filename, results):
        """Export summary statistics to CSV"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['# Coordinate-Aware Building Analysis - Summary'])
            writer.writerow(['# Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow([''])
            
            # Summary data
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Total Population', results['total_population'], 'people'])
            if results['floor_calculation_enabled']:
                writer.writerow(['Base Population (without floors)', results['total_base_population'], 'people'])
            writer.writerow(['Total Buildings', results['residential_count'] + results['non_residential_count'], 'buildings'])
            writer.writerow(['Residential Buildings', results['residential_count'], 'buildings'])
            writer.writerow(['Non-Residential Buildings', results['non_residential_count'], 'buildings'])
            writer.writerow(['Standard Houses', results['standard_houses'], 'buildings'])
            writer.writerow(['Medium Houses', results['medium_houses'], 'buildings'])
            writer.writerow(['Large Houses', results['large_houses'], 'buildings'])
            if results['floor_calculation_enabled']:
                writer.writerow(['Average Floors per Building', f"{results['average_floors']:.2f}", 'floors'])
                writer.writerow(['Total Floor Count', results['total_floors'], 'floors'])
            writer.writerow([''])
            
            # Parameters
            writer.writerow(['## Parameters Used'])
            writer.writerow(['Parameter', 'Value', 'Unit'])
            writer.writerow(['People per Household', results['people_per_household'], 'people'])
            writer.writerow(['Standard House Threshold', results['thresholds']['standard'], 'm²'])
            writer.writerow(['Medium House Threshold', results['thresholds']['medium'], 'm²'])
            writer.writerow(['Large House Threshold', results['thresholds']['large'], 'm²'])
            if results['floor_calculation_enabled']:
                writer.writerow(['Floor Height', results['floor_height'], 'm'])

    # ==================== Advanced Visualization Methods ====================

    def generate_all_visualizations(self):
        """Generate all selected advanced visualizations"""
        if not self.analysis_results:
            messagebox.showerror("Error", "No analysis results available for visualization")
            return
        
        # Clear previous charts
        for widget in self.advanced_viz_frame.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Clear statistical frame
        for widget in self.statistical_frame.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Check which charts are selected
        selected_charts = [key for key, var in self.chart_vars.items() if var.get()]
        
        if not selected_charts:
            messagebox.showwarning("No Charts Selected", "Please select at least one chart type to generate")
            return
        
        self.update_status("Generating advanced visualizations...", "🎨")
        self.show_progress(True)
        
        try:
            # Generate selected charts
            if 'building_count' in selected_charts:
                self.create_building_count_analysis()
            
            if 'population_dist' in selected_charts:
                self.create_population_distribution_analysis()
            
            if 'size_analysis' in selected_charts:
                self.create_size_analysis_charts()
            
            if 'floor_analysis' in selected_charts and self.analysis_results['floor_calculation_enabled']:
                self.create_advanced_floor_analysis()
            
            if 'statistical_comp' in selected_charts:
                self.create_statistical_analysis()
            
            self.show_progress(False)
            self.update_status("Advanced visualizations generated successfully", "✅")
            
            messagebox.showinfo("Visualization Complete", 
                              f"Generated {len(selected_charts)} advanced visualization(s)\n"
                              "Check the 'Advanced Charts' and 'Statistics' tabs")
        
        except Exception as e:
            self.show_progress(False)
            self.update_status("Error generating visualizations", "❌")
            messagebox.showerror("Visualization Error", f"Failed to generate visualizations:\n{str(e)}")

    def create_building_count_analysis(self):
        """Create detailed building count analysis charts"""
        chart_frame = tk.LabelFrame(self.advanced_viz_frame.scrollable_frame, 
                                  text="🏠 Advanced Building Count Analysis",
                                  font=('Arial', 12, 'bold'), bg="white", fg="#2c3e50",
                                  padx=15, pady=15)
        chart_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        if not HAS_ADVANCED_LIBS:
            tk.Label(chart_frame, text="Advanced visualization libraries not available\nInstall seaborn and pandas for enhanced charts",
                   bg="white", fg="#e74c3c", font=('Arial', 10)).pack(pady=20)
            return
        
        fig = Figure(figsize=(14, 10), facecolor='white')
        
        # Create subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # Prepare data
        building_details = self.analysis_results.get('building_details', [])
        if not building_details:
            ax1.text(0.5, 0.5, 'No building data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax2.text(0.5, 0.5, 'No building data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax3.text(0.5, 0.5, 'No building data available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
        else:
            df = pd.DataFrame(building_details)
            
            # Chart 1: Confidence distribution
            confidence_data = df['confidence']
            ax1.hist(confidence_data, bins=20, color='#3498db', alpha=0.7, edgecolor='white')
            ax1.set_title('Detection Confidence Distribution', fontweight='bold')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Number of Buildings')
            ax1.axvline(confidence_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {confidence_data.mean():.3f}')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Chart 2: Building area distribution
            residential_df = df[df['class_id'] == 0]
            if not residential_df.empty:
                area_data = residential_df['area_sqm']
                ax2.hist(area_data, bins=15, color='#27ae60', alpha=0.7, edgecolor='white')
                ax2.set_title('Residential Building Area Distribution', fontweight='bold')
                ax2.set_xlabel('Area (m²)')
                ax2.set_ylabel('Number of Buildings')
                ax2.axvline(area_data.mean(), color='red', linestyle='--', 
                           label=f'Mean: {area_data.mean():.1f} m²')
                ax2.legend()
                ax2.grid(alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No residential buildings', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
            
            # Chart 3: Detailed comparison
            if len(residential_df) > 0:
                categories = residential_df['house_category'].unique()
                x_pos = np.arange(len(categories))
                
                # Multiple metrics
                metrics = ['count', 'avg_area', 'total_population']
                metric_data = []
                
                for category in categories:
                    cat_data = residential_df[residential_df['house_category'] == category]
                    count = len(cat_data)
                    avg_area = cat_data['area_sqm'].mean() if len(cat_data) > 0 else 0
                    total_pop = cat_data['final_population'].sum() if len(cat_data) > 0 else 0
                    metric_data.append([count, avg_area/10, total_pop/10])  # Scale for visibility
                
                metric_data = np.array(metric_data).T
                
                width = 0.25
                colors = ['#3498db', '#e67e22', '#e74c3c']
                labels = ['Building Count', 'Avg Area (×10 m²)', 'Total Population (×10)']
                
                for i, (data, color, label) in enumerate(zip(metric_data, colors, labels)):
                    ax3.bar(x_pos + i*width, data, width, label=label, color=color, alpha=0.8)
                
                ax3.set_title('Comprehensive Category Comparison', fontweight='bold')
                ax3.set_xlabel('House Category')
                ax3.set_ylabel('Value (see legend for units)')
                ax3.set_xticks(x_pos + width)
                ax3.set_xticklabels(categories)
                ax3.legend()
                ax3.grid(axis='y', alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No residential buildings for comparison', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
        
        fig.suptitle('Advanced Building Count Analysis', fontsize=16, fontweight='bold')
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_population_distribution_analysis(self):
        """Create detailed population distribution analysis"""
        chart_frame = tk.LabelFrame(self.advanced_viz_frame.scrollable_frame, 
                                  text="👥 Population Distribution Analysis",
                                  font=('Arial', 12, 'bold'), bg="white", fg="#2c3e50",
                                  padx=15, pady=15)
        chart_frame.pack(fill="x", padx=10, pady=5)
        
        fig = Figure(figsize=(14, 8), facecolor='white')
        
        if self.analysis_results['floor_calculation_enabled']:
            # With floor data
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[1, :])
        else:
            # Without floor data
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax4 = fig.add_subplot(gs[1, :])
        
        building_details = self.analysis_results.get('building_details', [])
        
        if building_details:
            residential_buildings = [b for b in building_details if b['class_id'] == 0]
            
            if residential_buildings:
                # Chart 1: Population by building
                populations = [b['final_population'] for b in residential_buildings]
                ax1.hist(populations, bins=15, color='#e74c3c', alpha=0.7, edgecolor='white')
                ax1.set_title('Population per Building', fontweight='bold')
                ax1.set_xlabel('Population')
                ax1.set_ylabel('Number of Buildings')
                ax1.grid(alpha=0.3)
                
                # Chart 2: Cumulative population
                categories = ['Standard', 'Medium', 'Large']
                cat_populations = []
                for cat in categories:
                    cat_pop = sum(b['final_population'] for b in residential_buildings 
                                if b['house_category'] == cat)
                    cat_populations.append(cat_pop)
                
                colors = ['#27ae60', '#f39c12', '#9b59b6']
                wedges, texts, autotexts = ax2.pie(cat_populations, labels=categories, colors=colors, 
                                                 autopct='%1.1f%%', startangle=90)
                ax2.set_title('Population Distribution by Category', fontweight='bold')
                
                # Chart 3: Floor impact (if available)
                if self.analysis_results['floor_calculation_enabled']:
                    base_pops = [b['base_population'] for b in residential_buildings]
                    final_pops = [b['final_population'] for b in residential_buildings]
                    
                    ax3.scatter(base_pops, final_pops, alpha=0.6, color='#3498db')
                    ax3.plot([0, max(base_pops)], [0, max(base_pops)], 'r--', alpha=0.8, label='1:1 Line')
                    ax3.set_title('Floor Impact on Population', fontweight='bold')
                    ax3.set_xlabel('Base Population')
                    ax3.set_ylabel('Final Population (with floors)')
                    ax3.legend()
                    ax3.grid(alpha=0.3)
                
                # Chart 4: Building size vs population
                areas = [b['area_sqm'] for b in residential_buildings]
                ax4.scatter(areas, populations, c=[colors[categories.index(b['house_category'])] 
                           for b in residential_buildings], alpha=0.6, s=50)
                ax4.set_title('Building Area vs Population', fontweight='bold')
                ax4.set_xlabel('Building Area (m²)')
                ax4.set_ylabel('Population')
                ax4.grid(alpha=0.3)
                
                # Add category legend
                for i, (cat, color) in enumerate(zip(categories, colors)):
                    ax4.scatter([], [], c=color, label=cat, s=50)
                ax4.legend(title='House Category')
        
        title = 'Population Distribution Analysis'
        if self.analysis_results['floor_calculation_enabled']:
            title += ' (with Coordinate-Aware Floor Calculation)'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_size_analysis_charts(self):
        """Create building size analysis charts"""
        chart_frame = tk.LabelFrame(self.advanced_viz_frame.scrollable_frame, 
                                  text="📏 Building Size Analysis",
                                  font=('Arial', 12, 'bold'), bg="white", fg="#2c3e50",
                                  padx=15, pady=15)
        chart_frame.pack(fill="x", padx=10, pady=5)
        
        fig = Figure(figsize=(14, 10), facecolor='white')
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        building_details = self.analysis_results.get('building_details', [])
        residential_buildings = [b for b in building_details if b['class_id'] == 0]
        
        if residential_buildings and HAS_ADVANCED_LIBS:
            df = pd.DataFrame(residential_buildings)
            
            # Chart 1: Size distribution with categories
            ax1 = fig.add_subplot(gs[0, 0])
            areas = df['area_sqm']
            categories = df['house_category']
            
            for i, cat in enumerate(['Standard', 'Medium', 'Large']):
                cat_areas = areas[categories == cat]
                if len(cat_areas) > 0:
                    ax1.hist(cat_areas, bins=10, alpha=0.7, label=cat, 
                            color=['#27ae60', '#f39c12', '#9b59b6'][i])
            
            ax1.set_title('Area Distribution by Category', fontweight='bold')
            ax1.set_xlabel('Area (m²)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Chart 2: Size vs confidence
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.scatter(df['area_sqm'], df['confidence'], alpha=0.6, color='#3498db')
            ax2.set_title('Building Size vs Detection Confidence', fontweight='bold')
            ax2.set_xlabel('Area (m²)')
            ax2.set_ylabel('Confidence')
            ax2.grid(alpha=0.3)
            
            # Chart 3: Box plot of areas by category
            ax3 = fig.add_subplot(gs[1, 0])
            category_data = [areas[categories == cat].values for cat in ['Standard', 'Medium', 'Large']]
            category_labels = ['Standard', 'Medium', 'Large']
            
            bp = ax3.boxplot(category_data, labels=category_labels, patch_artist=True)
            colors = ['#27ae60', '#f39c12', '#9b59b6']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_title('Area Distribution by Category (Box Plot)', fontweight='bold')
            ax3.set_ylabel('Area (m²)')
            ax3.grid(alpha=0.3)
            
            # Chart 4: Size statistics
            ax4 = fig.add_subplot(gs[1, 1])
            stats_data = []
            stats_labels = []
            
            for cat in ['Standard', 'Medium', 'Large']:
                cat_areas = areas[categories == cat]
                if len(cat_areas) > 0:
                    stats_data.append([cat_areas.mean(), cat_areas.median(), cat_areas.std()])
                    stats_labels.append(cat)
            
            if stats_data:
                stats_data = np.array(stats_data).T
                x_pos = np.arange(len(stats_labels))
                width = 0.25
                
                ax4.bar(x_pos - width, stats_data[0], width, label='Mean', color='#3498db', alpha=0.8)
                ax4.bar(x_pos, stats_data[1], width, label='Median', color='#e67e22', alpha=0.8)
                ax4.bar(x_pos + width, stats_data[2], width, label='Std Dev', color='#e74c3c', alpha=0.8)
                
                ax4.set_title('Area Statistics by Category', fontweight='bold')
                ax4.set_ylabel('Area (m²)')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(stats_labels)
                ax4.legend()
                ax4.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Building Size Analysis', fontsize=16, fontweight='bold')
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_advanced_floor_analysis(self):
        """Create advanced floor analysis (only when height data is available)"""
        if not self.analysis_results['floor_calculation_enabled']:
            return
        
        chart_frame = tk.LabelFrame(self.advanced_viz_frame.scrollable_frame, 
                                  text="🏢 Advanced Coordinate-Aware Floor Analysis",
                                  font=('Arial', 12, 'bold'), bg="white", fg="#2c3e50",
                                  padx=15, pady=15)
        chart_frame.pack(fill="x", padx=10, pady=5)
        
        fig = Figure(figsize=(14, 12), facecolor='white')
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        building_details = self.analysis_results.get('building_details', [])
        residential_buildings = [b for b in building_details if b['class_id'] == 0]
        
        if residential_buildings:
            floors_data = [b['floors'] for b in residential_buildings]
            areas_data = [b['area_sqm'] for b in residential_buildings]
            base_pop_data = [b['base_population'] for b in residential_buildings]
            final_pop_data = [b['final_population'] for b in residential_buildings]
            
            # Chart 1: Floor frequency
            ax1 = fig.add_subplot(gs[0, 0])
            unique_floors = sorted(set(floors_data))
            floor_counts = [floors_data.count(f) for f in unique_floors]
            ax1.bar(unique_floors, floor_counts, color='#3498db', alpha=0.8)
            ax1.set_title('Floor Count Distribution', fontweight='bold')
            ax1.set_xlabel('Number of Floors')
            ax1.set_ylabel('Number of Buildings')
            ax1.grid(axis='y', alpha=0.3)
            
            # Chart 2: Floors vs building area
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.scatter(areas_data, floors_data, alpha=0.6, color='#27ae60')
            ax2.set_title('Building Area vs Floor Count', fontweight='bold')
            ax2.set_xlabel('Building Area (m²)')
            ax2.set_ylabel('Number of Floors')
            ax2.grid(alpha=0.3)
            
            # Chart 3: Population impact
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.scatter(base_pop_data, final_pop_data, alpha=0.6, color='#e74c3c')
            max_pop = max(max(base_pop_data), max(final_pop_data))
            ax3.plot([0, max_pop], [0, max_pop], 'k--', alpha=0.5, label='1:1 Line')
            ax3.set_title('Population Impact of Floor Calculation', fontweight='bold')
            ax3.set_xlabel('Base Population')
            ax3.set_ylabel('Final Population (with floors)')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # Chart 4: Floor efficiency
            ax4 = fig.add_subplot(gs[1, 1])
            efficiency = [final/base if base > 0 else 1 for final, base in zip(final_pop_data, base_pop_data)]
            ax4.hist(efficiency, bins=15, color='#9b59b6', alpha=0.7)
            ax4.set_title('Population Efficiency (Final/Base)', fontweight='bold')
            ax4.set_xlabel('Efficiency Ratio')
            ax4.set_ylabel('Number of Buildings')
            ax4.axvline(np.mean(efficiency), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(efficiency):.2f}')
            ax4.legend()
            ax4.grid(alpha=0.3)
            
            # Chart 5: Category comparison
            ax5 = fig.add_subplot(gs[2, :])
            categories = ['Standard', 'Medium', 'Large']
            cat_data = {cat: [] for cat in categories}
            
            for building in residential_buildings:
                cat = building['house_category']
                if cat in cat_data:
                    cat_data[cat].append(building['floors'])
            
            # Create box plot
            box_data = [cat_data[cat] for cat in categories if cat_data[cat]]
            box_labels = [cat for cat in categories if cat_data[cat]]
            
            if box_data:
                bp = ax5.boxplot(box_data, labels=box_labels, patch_artist=True)
                colors = ['#27ae60', '#f39c12', '#9b59b6']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax5.set_title('Floor Distribution by House Category', fontweight='bold')
                ax5.set_ylabel('Number of Floors')
                ax5.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Advanced Coordinate-Aware Floor Analysis', fontsize=16, fontweight='bold')
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_statistical_analysis(self):
        """Create statistical analysis charts and data"""
        # Clear previous content
        for widget in self.statistical_frame.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Statistical summary frame
        stats_frame = tk.LabelFrame(self.statistical_frame.scrollable_frame, 
                                  text="📈 Statistical Summary",
                                  font=('Arial', 12, 'bold'), bg="white", fg="#2c3e50",
                                  padx=15, pady=15)
        stats_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Create statistical text display
        stats_text = tk.Text(stats_frame, height=20, width=80, font=('Consolas', 10),
                           bg="#f8f9fa", relief=tk.SUNKEN, bd=2)
        stats_text.pack(fill="both", expand=True)
        
        # Calculate and display statistics
        building_details = self.analysis_results.get('building_details', [])
        residential_buildings = [b for b in building_details if b['class_id'] == 0]
        
        stats_content = self.generate_statistical_content(residential_buildings)
        stats_text.insert(tk.END, stats_content)
        stats_text.config(state='disabled')
        
        # Add scrollbar to text widget
        scrollbar = tk.Scrollbar(stats_frame, command=stats_text.yview)
        stats_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        
        # Correlation analysis (if enough data)
        if len(residential_buildings) > 5 and HAS_ADVANCED_LIBS:
            self.create_correlation_analysis(residential_buildings)

    def generate_statistical_content(self, residential_buildings):
        """Generate statistical analysis content"""
        content = []
        content.append("=" * 80)
        content.append("COORDINATE-AWARE BUILDING ANALYSIS - STATISTICAL REPORT")
        content.append("=" * 80)
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"Analysis Method: {self.analysis_results['processing_method']}")
        content.append("")
        
        if not residential_buildings:
            content.append("No residential buildings detected for statistical analysis.")
            return "\n".join(content)
        
        # Basic statistics
        content.append("BASIC STATISTICS")
        content.append("-" * 40)
        content.append(f"Total Residential Buildings: {len(residential_buildings):,}")
        
        # Area statistics
        areas = [b['area_sqm'] for b in residential_buildings]
        content.append(f"\nBUILDING AREA STATISTICS (m²)")
        content.append(f"  Mean:     {np.mean(areas):8.2f}")
        content.append(f"  Median:   {np.median(areas):8.2f}")
        content.append(f"  Std Dev:  {np.std(areas):8.2f}")
        content.append(f"  Min:      {np.min(areas):8.2f}")
        content.append(f"  Max:      {np.max(areas):8.2f}")
        content.append(f"  Range:    {np.max(areas) - np.min(areas):8.2f}")
        
        # Confidence statistics
        confidences = [b['confidence'] for b in residential_buildings]
        content.append(f"\nDETECTION CONFIDENCE STATISTICS")
        content.append(f"  Mean:     {np.mean(confidences):8.3f}")
        content.append(f"  Median:   {np.median(confidences):8.3f}")
        content.append(f"  Std Dev:  {np.std(confidences):8.3f}")
        content.append(f"  Min:      {np.min(confidences):8.3f}")
        content.append(f"  Max:      {np.max(confidences):8.3f}")
        
        # Floor statistics (if available)
        if self.analysis_results['floor_calculation_enabled']:
            floors = [b['floors'] for b in residential_buildings]
            content.append(f"\nFLOOR COUNT STATISTICS")
            content.append(f"  Mean:     {np.mean(floors):8.2f}")
            content.append(f"  Median:   {np.median(floors):8.2f}")
            content.append(f"  Std Dev:  {np.std(floors):8.2f}")
            content.append(f"  Min:      {np.min(floors):8.0f}")
            content.append(f"  Max:      {np.max(floors):8.0f}")
            content.append(f"  Mode:     {stats.mode(floors)[0]:8.0f}")
        
        # Population statistics
        populations = [b['final_population'] for b in residential_buildings]
        content.append(f"\nPOPULATION PER BUILDING STATISTICS")
        content.append(f"  Mean:     {np.mean(populations):8.2f}")
        content.append(f"  Median:   {np.median(populations):8.2f}")
        content.append(f"  Std Dev:  {np.std(populations):8.2f}")
        content.append(f"  Min:      {np.min(populations):8.2f}")
        content.append(f"  Max:      {np.max(populations):8.2f}")
        
        # Category breakdown
        content.append(f"\nCATEGORY BREAKDOWN")
        content.append("-" * 40)
        for category in ['Standard', 'Medium', 'Large']:
            cat_buildings = [b for b in residential_buildings if b['house_category'] == category]
            if cat_buildings:
                cat_areas = [b['area_sqm'] for b in cat_buildings]
                cat_pops = [b['final_population'] for b in cat_buildings]
                content.append(f"\n{category.upper()} HOUSES ({len(cat_buildings)} buildings)")
                content.append(f"  Area - Mean: {np.mean(cat_areas):6.1f} m², Range: {np.min(cat_areas):6.1f}-{np.max(cat_areas):6.1f} m²")
                content.append(f"  Population - Total: {sum(cat_pops):6.0f}, Mean: {np.mean(cat_pops):6.1f}")
                
                if self.analysis_results['floor_calculation_enabled']:
                    cat_floors = [b['floors'] for b in cat_buildings]
                    content.append(f"  Floors - Mean: {np.mean(cat_floors):6.1f}, Range: {np.min(cat_floors):1.0f}-{np.max(cat_floors):1.0f}")
        
        # Percentile analysis
        content.append(f"\nPERCENTILE ANALYSIS")
        content.append("-" * 40)
        for percentile in [10, 25, 50, 75, 90, 95, 99]:
            area_p = np.percentile(areas, percentile)
            pop_p = np.percentile(populations, percentile)
            content.append(f"  {percentile:2d}th percentile - Area: {area_p:6.1f} m², Population: {pop_p:6.1f}")
        
        # Processing summary
        content.append(f"\nPROCESSING SUMMARY")
        content.append("-" * 40)
        content.append(f"Coordinate-aware processing: {'Enabled' if self.is_tif_file else 'Standard image'}")
        if self.is_tif_file:
            content.append(f"Number of tiles processed: {self.analysis_results['tile_count']:,}")
            content.append(f"Tile dimensions: {self.analysis_results['tile_dimensions']}")
        content.append(f"Floor calculation: {'Enabled' if self.analysis_results['floor_calculation_enabled'] else 'Disabled'}")
        content.append(f"Total estimated population: {self.analysis_results['total_population']:,}")
        
        return "\n".join(content)

    def create_correlation_analysis(self, residential_buildings):
        """Create correlation analysis charts"""
        corr_frame = tk.LabelFrame(self.statistical_frame.scrollable_frame, 
                                 text="🔗 Correlation Analysis",
                                 font=('Arial', 12, 'bold'), bg="white", fg="#2c3e50",
                                 padx=15, pady=15)
        corr_frame.pack(fill="x", padx=10, pady=5)
        
        try:
            import seaborn as sns
            
            # Prepare data for correlation
            df_data = {
                'Area (m²)': [b['area_sqm'] for b in residential_buildings],
                'Confidence': [b['confidence'] for b in residential_buildings],
                'Population': [b['final_population'] for b in residential_buildings],
                'Multiplier': [b['population_multiplier'] for b in residential_buildings]
            }
            
            if self.analysis_results['floor_calculation_enabled']:
                df_data['Floors'] = [b['floors'] for b in residential_buildings]
            
            df = pd.DataFrame(df_data)
            
            # Create correlation matrix
            fig = Figure(figsize=(10, 8), facecolor='white')
            ax = fig.add_subplot(111)
            
            correlation_matrix = df.corr()
            
            # Create heatmap
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add labels
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.columns)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_matrix.columns)
            
            # Add correlation values
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                                 ha="center", va="center", color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
                                 fontweight='bold')
            
            ax.set_title('Variable Correlation Matrix', fontweight='bold', pad=20)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
            
            fig.tight_layout()
            
            # Embed chart
            canvas = FigureCanvasTkAgg(fig, master=corr_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            tk.Label(corr_frame, text=f"Error creating correlation analysis: {str(e)}",
                   bg="white", fg="#e74c3c", font=('Arial', 10)).pack(pady=20)

# ==================== Main Application ====================

# Replace your main() function with this enhanced version
def main():
    """Enhanced main function with error handling"""
    try:
        root = tk.Tk()
        app = BuildingCountPopulationEstimator(root)
        
        # Set up window closing event
        def on_closing():
            logging.info("Application closing")
            root.quit()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        logging.info("Starting main application loop")
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Critical application error: {e}", exc_info=True)
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Application Error", 
                f"A critical error occurred:\n{str(e)}\n\n"
                f"Please check the log file at:\n{APP_DATA_DIR}/logs/app.log")
        except:
            print(f"Critical error: {e}")
    finally:
        logging.info("Application terminated")

if __name__ == "__main__":
    main()
